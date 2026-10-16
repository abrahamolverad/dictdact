"""Microphone capture and whisper-friendly preprocessing."""
from __future__ import annotations

import logging
import math
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy import signal

log = logging.getLogger("dictdact.audio")

TARGET_SR = 16000


@dataclass
class DeviceInfo:
    index: int
    name: str
    channels: int
    default_samplerate: float
    is_default: bool
    fallback_reason: str | None = None


_open_streams = 0
_streams_lock = threading.Lock()
audio_lock = threading.RLock()   # serialises every PortAudio open/close/re-init in the process


def refresh_devices() -> bool:
    """PortAudio enumerates devices once; re-initialise so hot-plugged / unplugged USB mics show
    up. Only safe while no stream is open — NEVER call this from a hot path while recording."""
    with audio_lock:
        with _streams_lock:
            if _open_streams:
                return False
        try:
            sd._terminate()
            sd._initialize()
            return True
        except Exception as e:  # pragma: no cover
            log.debug("device refresh failed: %s", e)
            return False


def list_input_devices(refresh: bool = False) -> list[DeviceInfo]:
    if refresh:
        refresh_devices()
    devices = sd.query_devices()
    try:
        default_in = sd.default.device[0]
    except Exception:  # pragma: no cover - depends on host config
        default_in = -1
    out: list[DeviceInfo] = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            out.append(
                DeviceInfo(
                    index=i,
                    name=d["name"],
                    channels=d["max_input_channels"],
                    default_samplerate=float(d["default_samplerate"]),
                    is_default=(i == default_in),
                )
            )
    return out


def lid_closed() -> bool:
    import subprocess

    try:
        out = subprocess.run(["ioreg", "-r", "-k", "AppleClamshellState", "-d", "4"],
                             capture_output=True, text=True, timeout=3).stdout
        return '"AppleClamshellState" = Yes' in out
    except Exception:
        return False


def _best_fallback(devices: list[DeviceInfo]) -> DeviceInfo | None:
    """Pick a sensible mic when the preferred one is absent: an external/USB mic first,
    then the built-in one (unless the lid is closed), then whatever is left."""
    builtin = [d for d in devices if "macbook" in d.name.lower()]
    iphone = [d for d in devices if "iphone" in d.name.lower()]
    external = [d for d in devices if d not in builtin and d not in iphone]
    if external:
        return external[0]
    if builtin and not lid_closed():
        return builtin[0]
    return (iphone + builtin + devices or [None])[0]


def resolve_input_device(name: str | None, refresh_if_missing: bool = False) -> DeviceInfo | None:
    """Find the input device whose name matches ``name``. With ``refresh_if_missing`` the device
    list is re-enumerated once when the preferred mic is absent (a re-plugged USB mic). Falls back
    sensibly and logs why; ``fallback_reason`` on the returned DeviceInfo says so."""
    devices = list_input_devices()
    if name and refresh_if_missing and not any(name.strip().lower() in d.name.lower() for d in devices):
        if refresh_devices():
            devices = list_input_devices()
    if not devices:
        return None
    if not name:
        default = next((d for d in devices if d.is_default), None)
        if default and "macbook" in default.name.lower() and lid_closed():
            alt = _best_fallback(devices)
            if alt and alt is not default:
                log.warning("lid is closed: built-in mic is silent, using %s instead", alt.name)
                alt.fallback_reason = "lid closed: built-in mic is silent"
                return alt
        return default
    needle = name.strip().lower()
    for d in devices:
        if d.name.lower() == needle:
            return d
    for d in devices:
        if needle in d.name.lower():
            return d
    alt = _best_fallback(devices)
    log.warning("microphone %r not connected; using %s. Available: %s", name,
                alt.name if alt else "nothing", ", ".join(d.name for d in devices))
    if alt:
        alt.fallback_reason = f"{name} not connected"
    return alt


class Recorder:
    """Records mono float32 audio from one input device. Thread-safe start/stop."""

    def __init__(self, device: DeviceInfo | None, max_seconds: int = 600, preferred_name: str | None = None):
        self.device = device
        self.preferred_name = preferred_name if preferred_name is not None else (device.name if device else None)
        self.max_seconds = max_seconds
        # False = ask CoreAudio for 16 kHz (what the ASR wants). True = open the mic at its own
        # rate so the voice bank gets the take at recording quality; take_audio() still hands
        # the ASR 16 kHz. Read at every start(), so flipping it never touches a take in flight.
        self.native = False
        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._samplerate = TARGET_SR
        self._frames = 0
        self._peak = 0.0
        self._rms_history: deque[float] = deque(maxlen=64)   # ~2 s of 32 ms chunks, for the overlay
        self.overflowed = False
        self.hit_cap = False

    # -- stream lifecycle ---------------------------------------------------
    def _pick_samplerate(self, idx: int | None) -> int:
        if self.native:
            info = sd.query_devices(idx if idx is not None else sd.default.device[0])
            native = int(info["default_samplerate"])
            try:
                sd.check_input_settings(device=idx, channels=1, samplerate=native, dtype="float32")
                return native
            except Exception:
                pass
        try:
            sd.check_input_settings(device=idx, channels=1, samplerate=TARGET_SR, dtype="float32")
            return TARGET_SR
        except Exception:
            info = sd.query_devices(idx if idx is not None else sd.default.device[0])
            return int(info["default_samplerate"])

    def start(self) -> None:
        """Open the preferred mic (re-resolved by name every time, so a re-plugged USB mic is found
        and a missing one falls back). Retries once on a transient CoreAudio error."""
        global _open_streams
        with self._lock:
            if self._stream is not None:
                return
            self._chunks = []
            self._frames = 0
            self._peak = 0.0
            self._rms_history.clear()
            self.overflowed = False
            self.hit_cap = False
        with audio_lock:
            self.device = resolve_input_device(self.preferred_name, refresh_if_missing=True)
            stream, last_error, tried = None, None, []
            for attempt in (1, 2, 3):
                if self.device is None:
                    break
                try:
                    self._samplerate = self._pick_samplerate(self.device.index)
                    stream = sd.InputStream(device=self.device.index, channels=1, samplerate=self._samplerate,
                                            dtype="float32", blocksize=0, latency="low", callback=self._callback)
                    stream.start()
                    break
                except Exception as e:
                    stream, last_error = None, e
                    tried.append(self.device.name)
                    log.warning("opening %s failed (attempt %d): %s", self.device.name, attempt, e)
                    time.sleep(0.25)
                    if attempt == 1:
                        # The entry can be stale (device re-plugged): re-enumerate, then resolve again.
                        refresh_devices()
                        self.device = resolve_input_device(self.preferred_name)
                    else:
                        # The preferred device really will not open — fall back to any other mic.
                        others = [d for d in list_input_devices() if d.name not in tried]
                        alt = _best_fallback(others) if others else None
                        if alt is None:
                            break
                        alt.fallback_reason = f"{self.preferred_name or 'the selected mic'} could not be opened"
                        log.warning("falling back to %s", alt.name)
                        self.device = alt
            if stream is None:
                raise RuntimeError(f"could not open {' or '.join(tried) or 'any microphone'}: {last_error}")
            with _streams_lock:
                _open_streams += 1
        with self._lock:
            self._stream = stream
        log.debug("recording started on %s @ %d Hz", self.device.name if self.device else "default", self._samplerate)

    def _callback(self, indata, frames, time_info, status) -> None:  # PortAudio thread
        if status and status.input_overflow:
            self.overflowed = True
        mono = indata[:, 0].copy()
        with self._lock:
            if self._frames >= self.max_seconds * self._samplerate:
                self.hit_cap = True
                return
            self._chunks.append(mono)
            self._frames += len(mono)
            if len(mono):
                self._peak = float(np.max(np.abs(mono)))
                self._rms_history.append(float(np.sqrt(np.mean(mono * mono))))

    @property
    def recording(self) -> bool:
        return self._stream is not None

    @property
    def seconds(self) -> float:
        return self._frames / float(self._samplerate)

    def level_dbfs(self) -> float:
        return 20 * math.log10(self._peak + 1e-9)

    def latest_rms(self) -> float:
        with self._lock:
            return self._rms_history[-1] if self._rms_history else 0.0

    def max_rms(self) -> float:
        with self._lock:
            return max(self._rms_history) if self._rms_history else 0.0

    def take_raw(self):
        """Detach the stream and return (audio, samplerate, stream) exactly as the mic delivered
        it — no resampling. The audio is complete and safe even if closing the stream later hangs
        inside CoreAudio — close it with ``close_stream``."""
        with self._lock:
            stream, self._stream = self._stream, None
            chunks, self._chunks = self._chunks, []
            sr = self._samplerate
        if not chunks:
            return np.zeros(0, dtype=np.float32), sr, stream
        return np.concatenate(chunks).astype(np.float32), sr, stream

    def take_audio(self):
        """Detach the stream and return (audio at 16 kHz, stream)."""
        audio, sr, stream = self.take_raw()
        return to_target_sr(audio, sr), stream

    @staticmethod
    def close_stream(stream, timeout_s: float = 8.0) -> bool:
        """Stop+close a detached stream with a watchdog. Returns False if CoreAudio hung (the
        closing thread is abandoned; the caller decides how to recover)."""
        global _open_streams
        if stream is None:
            return True
        done = threading.Event()

        def _close():
            try:
                with audio_lock:
                    stream.stop()
                    stream.close()
            except Exception as e:  # pragma: no cover
                log.warning("error closing stream: %s", e)
            finally:
                done.set()

        threading.Thread(target=_close, name="dictdact-close-stream", daemon=True).start()
        ok = done.wait(timeout_s)
        with _streams_lock:
            _open_streams = max(0, _open_streams - 1)
        if not ok:
            log.error("CoreAudio did not return from stopping the input stream within %.0fs", timeout_s)
        return ok

    def stop(self) -> np.ndarray:
        """Stop and return the captured audio as float32 mono at 16 kHz."""
        audio, stream = self.take_audio()
        self.close_stream(stream)
        return audio

    def cancel(self) -> None:
        self.stop()


def to_target_sr(audio: np.ndarray, sr: int) -> np.ndarray:
    """Resample mono float32 audio to the 16 kHz the ASR models expect."""
    if sr == TARGET_SR or not len(audio):
        return audio
    g = math.gcd(TARGET_SR, sr)
    return signal.resample_poly(audio, TARGET_SR // g, sr // g).astype(np.float32)


class MicMonitor:
    """Opens an input device only to measure its level (for the "is it hearing me?" meter)."""

    def __init__(self, device: DeviceInfo | None):
        self.device = device
        self._stream: sd.InputStream | None = None
        self._rms = 0.0
        self._peak = 0.0
        self._lock = threading.Lock()

    def start(self) -> None:
        global _open_streams
        if self._stream is not None:
            return
        idx = self.device.index if self.device else None
        try:
            sd.check_input_settings(device=idx, channels=1, samplerate=TARGET_SR, dtype="float32")
            sr = TARGET_SR
        except Exception:
            sr = int(sd.query_devices(idx if idx is not None else sd.default.device[0])["default_samplerate"])
        with audio_lock:
            self._stream = sd.InputStream(device=idx, channels=1, samplerate=sr, dtype="float32", blocksize=0,
                                          latency="low", callback=self._callback)
            self._stream.start()
            with _streams_lock:
                _open_streams += 1

    def _callback(self, indata, frames, time_info, status) -> None:
        mono = indata[:, 0]
        if len(mono):
            with self._lock:
                self._rms = float(np.sqrt(np.mean(mono * mono)))
                self._peak = float(np.max(np.abs(mono)))

    def level(self) -> tuple[float, float]:
        with self._lock:
            return self._rms, self._peak

    @property
    def active(self) -> bool:
        return self._stream is not None

    def stop(self) -> bool:
        stream, self._stream = self._stream, None
        return Recorder.close_stream(stream)


# -- preprocessing -----------------------------------------------------------

@dataclass
class AudioStats:
    duration_s: float
    peak_dbfs: float
    speech_dbfs: float        # 95th percentile frame level = how loud the speech is
    noise_floor_dbfs: float   # 15th percentile frame level
    dynamic_range_db: float
    gain_db: float
    mode: str                 # "silence" | "whisper" | "quiet" | "normal"
    has_speech: bool

    def summary(self) -> str:
        return (f"{self.duration_s:.1f}s {self.mode} speech={self.speech_dbfs:.0f}dBFS "
                f"floor={self.noise_floor_dbfs:.0f}dBFS range={self.dynamic_range_db:.0f}dB gain={self.gain_db:+.0f}dB")


def analyze_and_normalize(x: np.ndarray, sr: int = TARGET_SR, max_gain_db: float = 40.0) -> tuple[np.ndarray, AudioStats]:
    """Trim silence, detect whispered/quiet speech, and normalize level so the ASR
    model sees a consistent signal whether you whisper or speak normally."""
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    duration = n / sr
    empty = AudioStats(duration, -120, -120, -120, 0, 0, "silence", False)
    if n < int(0.1 * sr):
        return x, empty

    x = x - float(x.mean())
    # 60 Hz high-pass: drops rumble / handling noise that dominates a whisper's SNR.
    sos = signal.butter(2, 60, "hp", fs=sr, output="sos")
    x = signal.sosfilt(sos, x).astype(np.float32)

    frame = int(0.02 * sr)
    nframes = n // frame
    if nframes < 5:
        return x, empty
    frames = x[: nframes * frame].reshape(nframes, frame)
    frame_db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9)
    noise_floor = float(np.percentile(frame_db, 15))
    speech_level = float(np.percentile(frame_db, 95))
    dyn = speech_level - noise_floor
    peak_db = 20 * math.log10(float(np.max(np.abs(x))) + 1e-9)

    has_speech = duration >= 0.35 and dyn >= 5.0 and speech_level > -65
    if not has_speech:
        mode = "silence"
    elif speech_level < -38:
        mode = "whisper"
    elif speech_level < -26:
        mode = "quiet"
    else:
        mode = "normal"

    # Trim to the speech region (+0.3 s padding) so long pauses don't become hallucinations.
    thr = max(noise_floor + 4.0, speech_level - 40.0)
    active = np.where(frame_db > thr)[0]
    if len(active):
        pad = int(0.3 * sr)
        start = max(0, int(active[0]) * frame - pad)
        end = min(n, (int(active[-1]) + 1) * frame + pad)
        x = x[start:end]

    # Normalize: bring the peak to -3 dBFS, boosting whispers up to max_gain_db.
    peak = float(np.max(np.abs(x))) + 1e-9
    gain_db = min(max_gain_db, -3.0 - 20 * math.log10(peak))
    if abs(gain_db) > 0.5:
        x = np.clip(x * (10 ** (gain_db / 20)), -1.0, 1.0).astype(np.float32)
    else:
        gain_db = 0.0

    stats = AudioStats(
        duration_s=len(x) / sr,
        peak_dbfs=peak_db,
        speech_dbfs=speech_level,
        noise_floor_dbfs=noise_floor,
        dynamic_range_db=dyn,
        gain_db=gain_db,
        mode=mode,
        has_speech=has_speech,
    )
    return x, stats


def save_wav(path: Path, x: np.ndarray, sr: int = TARGET_SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def load_wav(path: Path) -> np.ndarray:
    """Load any audio file to float32 mono 16 kHz (uses ffmpeg for non-WAV)."""
    import subprocess

    cmd = ["ffmpeg", "-v", "quiet", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(TARGET_SR), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()
