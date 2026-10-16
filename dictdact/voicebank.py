"""Voice bank: keep the speaker's natural dictation audio at the mic's own quality, for cloning.

People who dictate all day already produce hours of clean, natural, single-speaker speech — the
material a Professional Voice Clone (ElevenLabs) trains on, and that nobody finds time to sit down
and record. With ``voice_bank`` on, every take is also saved RAW (the mic's native rate, 24-bit
FLAC, untrimmed, un-normalized — never the 16 kHz copy the ASR sees), graded, and indexed with its
language. Collection stops by itself once a language holds ``voice_bank_target_min`` usable
minutes. ``export`` turns the best takes into upload-ready files; ``reference``/``say`` feed the
local Chatterbox clone.

ElevenLabs PVC guidance this follows (official docs, checked 2026-09-23): 30 min minimum, 2–3 h
recommended; one speaker, one consistent style; MP3 ≥ 192 kbps preferred over WAV; −23…−18 dB
RMS with peaks under −3 dB; trim long silences (pauses and fillers are reproduced).
"""
from __future__ import annotations

import json
import logging
import math
import os
import queue
import statistics
import subprocess
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import signal

from . import config as _config

log = logging.getLogger("dictdact.voicebank")

LANGS = ("es", "en")
MIN_NATIVE_SR = 32000          # below this the take is the ASR's 16 kHz copy — never bank it
MIN_TAKE_S = 2.0
MAX_STORED_S = 12 * 3600       # disk cap across languages, usable or not (~3.5 GB of FLAC)
DEFAULT_MIN_SNR = 20.0         # dB between the speech and the room between words
MAX_PAUSE_S = 0.6              # longer pauses are shortened to this in exports
EDGE_PAD_S = 0.2               # kept around speech so onsets and decays survive the trim
TARGET_SPEECH_DBFS = -19.0     # active-speech RMS after export (ElevenLabs: −23…−18 dB RMS)
PEAK_CEILING_DBFS = -4.0       # sample peak; leaves ~1 dB for MP3 overshoot under a −3 dB true peak
EXPORT_SR = 44100
GAP_S = 0.4                    # silence between takes in an export
FRAME_S = 0.02
MILESTONES_MIN = (30, 120, 180)  # ElevenLabs PVC minimum, the recommended 2 h, the default cap


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


# Local cloning (``say``) runs a TTS script the user installs separately (e.g. a Chatterbox
# setup). It is configured by environment only — there is no built-in default:
#   DICTDACT_CLONE_PYTHON  the Python interpreter of that TTS environment
#   DICTDACT_CLONE_SCRIPT  a script taking: --ref REF.wav --lang es|en --out OUT.wav "text"
CLONE_PYTHON: Path | None = _env_path("DICTDACT_CLONE_PYTHON")
CLONE_SCRIPT: Path | None = _env_path("DICTDACT_CLONE_SCRIPT")
CLONE_NOT_CONFIGURED = ("local voice cloning is not configured: set DICTDACT_CLONE_PYTHON to the Python of your "
                        "TTS environment and DICTDACT_CLONE_SCRIPT to a script that takes "
                        "--ref REF.wav --lang es|en --out OUT.wav \"text\"")


def bank_dir() -> Path:
    return _config.DATA_DIR / "voice-bank"


def _db(v: float) -> float:
    return 20 * math.log10(v + 1e-12)


# -- analysis ---------------------------------------------------------------------------------

def _frames_db(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """20 ms frame levels (dBFS) of an 80 Hz high-passed copy — rumble must not count as room
    tone. The saved audio itself is never filtered."""
    frame = max(1, int(FRAME_S * sr))
    n = len(x) // frame
    if n == 0:
        return np.zeros(0), frame
    hp = signal.sosfilt(signal.butter(2, 80, "hp", fs=sr, output="sos"), x.astype(np.float64))
    fr = hp[: n * frame].reshape(n, frame)
    return 20 * np.log10(np.sqrt(np.mean(fr * fr, axis=1)) + 1e-12), frame


def _active(frame_db: np.ndarray) -> tuple[np.ndarray, float]:
    """Frames that carry speech, and the room's level between words (10th percentile)."""
    if not len(frame_db):
        return np.zeros(0, dtype=bool), -120.0
    floor = float(np.percentile(frame_db, 10))
    loud = float(np.percentile(frame_db, 95))
    return frame_db > max(floor + 10.0, loud - 35.0), floor


def keep_mask(frame_db: np.ndarray, max_pause_s: float = MAX_PAUSE_S, pad_s: float = EDGE_PAD_S) -> np.ndarray:
    """Which frames an export keeps: speech ± pad, pauses up to ``max_pause_s`` (a longer pause
    keeps its first and last halves), nothing before the first word or after the last."""
    active, _ = _active(frame_db)
    n = len(active)
    keep = np.zeros(n, dtype=bool)
    idx = np.flatnonzero(active)
    if not len(idx):
        return keep
    pad = int(round(pad_s / FRAME_S))
    for i in idx:
        keep[max(0, i - pad): i + pad + 1] = True
    half = int(round(max_pause_s / FRAME_S / 2))
    i = int(np.flatnonzero(keep)[0])
    last = int(np.flatnonzero(keep)[-1])
    while i <= last:
        if keep[i]:
            i += 1
            continue
        j = i
        while j <= last and not keep[j]:
            j += 1
        if j - i <= 2 * half:
            keep[i:j] = True
        else:
            keep[i:i + half] = True
            keep[j - half:j] = True
        i = j
    return keep


def tighten(x: np.ndarray, sr: int, max_pause_s: float = MAX_PAUSE_S) -> np.ndarray:
    """The take without leading/trailing silence and with long pauses shortened. Cuts get a 5 ms
    fade so a join inside room tone never clicks. Samples themselves are untouched."""
    frame_db, frame = _frames_db(x, sr)
    keep = keep_mask(frame_db, max_pause_s)
    if not keep.any():
        return np.zeros(0, dtype=np.float32)
    fade = max(1, int(0.005 * sr))
    parts: list[np.ndarray] = []
    i, n = 0, len(keep)
    while i < n:
        if not keep[i]:
            i += 1
            continue
        j = i
        while j < n and keep[j]:
            j += 1
        seg = x[i * frame: j * frame].astype(np.float32).copy()
        if len(seg) > 2 * fade:
            if i > 0:
                seg[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            if j < n:
                seg[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        parts.append(seg)
        i = j
    return np.concatenate(parts)


@dataclass
class Metrics:
    duration_s: float
    voiced_s: float        # what an export keeps of this take (speech + short pauses)
    speech_s: float        # frames that carry speech
    speech_dbfs: float     # RMS over the speech frames
    floor_dbfs: float      # the room between words
    snr_db: float
    peak_dbfs: float
    clipped: int           # samples at full scale
    hum_share: float       # share of the room noise below 500 Hz (AC, fans, fridge, traffic)


def measure(x: np.ndarray, sr: int) -> Metrics:
    x = np.asarray(x, dtype=np.float32)
    frame_db, _ = _frames_db(x, sr)
    active, floor = _active(frame_db)
    if active.any():
        speech = float(10 * np.log10(np.mean(10 ** (frame_db[active] / 10)) + 1e-20))
    else:
        speech = floor
    voiced = float(keep_mask(frame_db).sum() * FRAME_S)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    return Metrics(
        duration_s=round(len(x) / sr, 2), voiced_s=round(voiced, 2),
        speech_s=round(float(active.sum() * FRAME_S), 2), speech_dbfs=round(speech, 1),
        floor_dbfs=round(floor, 1), snr_db=round(speech - floor, 1), peak_dbfs=round(_db(peak), 1),
        clipped=int(np.count_nonzero(np.abs(x) >= 0.999)), hum_share=round(_hum_share(x, sr, frame_db), 2),
    )


def _hum_share(x: np.ndarray, sr: int, frame_db: np.ndarray) -> float:
    """How much of the noise between words sits below 500 Hz. High = a hum the gain knob can't
    fix (air conditioning, fans, a fridge, traffic); low = hiss or broadband noise."""
    if len(frame_db) < 10:
        return 0.0
    frame = max(1, int(FRAME_S * sr))       # the same frames _frames_db measured
    quiet = np.flatnonzero(frame_db <= np.percentile(frame_db, 20))
    q = np.concatenate([x[i * frame:(i + 1) * frame] for i in quiet]).astype(np.float64)
    power = np.abs(np.fft.rfft(q * np.hanning(len(q)))) ** 2
    f = np.fft.rfftfreq(len(q), 1 / sr)
    band = f >= 40
    total = float(power[band].sum())
    return float(power[band & (f < 500)].sum()) / total if total > 0 else 0.0


def language_counts(text: str) -> tuple[int, int]:
    """(Spanish, English) function-word counts — one-language-only words, so a take that
    switches languages mid-way shows up."""
    from .format import _lang_counts

    return _lang_counts(text or "")


def infer_language(text: str) -> str | None:
    """es/en from the words themselves, for engines that report no language (Parakeet). Needs
    three one-language words and a 75 % majority; anything less stays unknown (not banked)."""
    es, en = language_counts(text)
    if es + en < 3:
        return None
    if es >= 0.75 * (es + en):
        return "es"
    if en >= 0.75 * (es + en):
        return "en"
    return None


def problems(e: dict, min_snr: float = DEFAULT_MIN_SNR) -> list[str]:
    """Why a banked take should not go into a clone ([] = usable). Computed on read, so the
    thresholds can change without touching the bank."""
    out = []
    if e.get("clipped", 0) >= 3:
        out.append("clipping")
    if e.get("snr_db", 0) < min_snr:
        out.append("noisy room")
    if e.get("speech_dbfs", -120) < -50 or e.get("mode") == "whisper":
        out.append("whispered / far from the mic")
    if e.get("voiced_s", 0) < MIN_TAKE_S:
        out.append("too short")
    es, en = e.get("es_words", 0), e.get("en_words", 0)
    own, other = (es, en) if e.get("lang") == "es" else (en, es)
    if own + other >= 4 and other / (own + other) > 0.25:
        out.append("mixed languages")
    return out


# -- the bank -----------------------------------------------------------------------------------

def load_index(root: Path | None = None, present_only: bool = False) -> list[dict]:
    """The bank's index. ``present_only`` drops entries whose audio file is gone — the speaker
    may delete takes (or the whole folder) by hand, and those must stop counting."""
    root = root or bank_dir()
    path = root / "index.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue            # a line the app is still writing
    if present_only:
        out = [e for e in out if (root / e.get("file", "")).is_file()]
    return out


def _best_device(by_dev: dict[str, dict[str, float]]) -> dict[str, float]:
    return {lang: max(devs.values(), default=0.0) for lang, devs in by_dev.items()}


class VoiceBank:
    """Saves takes on its own thread: the worker hands a take over and returns immediately."""

    def __init__(self, root: Path | None = None, notify=None, enabled=None):
        self._root = root
        # () -> bool, read again right before a write: switching the bank off must also drop
        # takes that were already queued, not just stop new ones.
        self.enabled = enabled
        self._generation = 0                # bumped by switched_off(): takes captured before it are dropped
        self._tally_sig = None              # index (size, mtime) the tally was computed from
        self._decoded: set = set()          # files already proven to decode (see _decodable_usable_s)
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=16)
        self._thread: threading.Thread | None = None
        self._tally: dict | None = None     # {"usable": {lang: voiced s}, "kept": {lang: recorded s}, "stored": s}
        self.notify = notify                # (title, message) → a macOS notification
        self.on_change = None               # () → called after every take the bank looked at
        self.last: dict | None = None       # what happened to the latest take (for the menu)

    @property
    def root(self) -> Path:
        return self._root or bank_dir()

    # -- tally (what the caps and the menu read) --
    @property
    def generation(self) -> int:
        """Stamp a take with this when its recording starts; pass it back to offer()."""
        return self._generation

    def _index_sig(self):
        try:
            st = (self.root / "index.jsonl").stat()
            return (st.st_size, st.st_mtime_ns)
        except OSError:
            return None

    def refresh(self) -> None:
        """Forget the cached tally (files may have been deleted by hand)."""
        with self._lock:
            self._tally = None

    def tally(self) -> dict:
        with self._lock:
            sig = self._index_sig()
            if self._tally is None or sig != self._tally_sig:
                by_dev: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
                kept: dict[str, float] = defaultdict(float)
                stored = 0.0
                self._tally_sig = sig
                for e in load_index(self.root, present_only=True):
                    stored += e.get("duration_s", 0)
                    kept[e["lang"]] += e.get("duration_s", 0)
                    if not problems(e):
                        by_dev[e["lang"]][e.get("device") or "?"] += e.get("voiced_s", 0)
                self._tally = {"usable_dev": {k: dict(v) for k, v in by_dev.items()}, "kept": dict(kept),
                               "stored": stored}
                self._tally["usable"] = _best_device(self._tally["usable_dev"])
            return {"usable": dict(self._tally["usable"]), "kept": dict(self._tally["kept"]),
                    "stored": self._tally["stored"]}

    def usable_minutes(self) -> dict[str, float]:
        """Per language, the usable minutes of ONE mic — the one export will use. A clone is
        trained on one mic; minutes from another mic must not make a language look ready."""
        return {k: v / 60 for k, v in self.tally()["usable"].items()}

    def kept_minutes(self) -> dict[str, float]:
        return {k: v / 60 for k, v in self.tally()["kept"].items()}

    # -- intake --
    def switched_off(self) -> None:
        """Call AFTER the setting reads off. Waits out a publication already in progress; once it
        returns nothing more is published, and takes queued before it are dropped even if the
        bank is switched back on before they are processed."""
        with self._lock:
            self._generation += 1

    def offer(self, audio: np.ndarray, sr: int, **meta) -> None:
        """Queue a take for banking. Never blocks the caller; a full queue drops the take."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="dictdact-voicebank", daemon=True)
                self._thread.start()
            meta = {**meta, "generation": meta.get("generation", self._generation)}
        try:
            self._queue.put_nowait((audio, sr, meta))
        except queue.Full:
            log.warning("voice bank: busy, take not banked")

    def _run(self) -> None:
        while True:
            audio, sr, meta = self._queue.get()
            try:
                self.bank(audio, sr, **meta)
            except Exception:
                log.exception("voice bank: could not save the take")
            finally:
                self._queue.task_done()
            if self.on_change is not None:
                try:
                    self.on_change()
                except Exception:
                    log.exception("voice bank: on_change failed")

    def wait(self, timeout: float = 10.0) -> None:
        """Tests/CLI: block until queued takes are written."""
        end = time.time() + timeout
        while self._queue.unfinished_tasks and time.time() < end:
            time.sleep(0.01)

    def skip_reason(self, sr: int, duration_s: float, *, language: str | None, device: str | None,
                    fallback: str | None, preferred: str | None, target_min: float) -> str | None:
        if sr < MIN_NATIVE_SR:
            return f"recorded at {sr} Hz (the bank was switched on mid-take)"
        if fallback or (preferred and (not device or preferred.strip().lower() not in device.lower())):
            return f"not the chosen mic ({device or '?'})"
        if language not in LANGS:
            return f"language {language!r}"
        if duration_s < MIN_TAKE_S:
            return "too short"
        # Both caps are re-checked against the disk before they refuse a take: files deleted by
        # hand must free their room, and a 'full' language must hold audio that really decodes.
        t = self.tally()
        if t["stored"] >= MAX_STORED_S:
            self.refresh()
            if self.tally()["stored"] >= MAX_STORED_S:
                return "the bank reached its disk cap"
        if t["usable"].get(language, 0) >= target_min * 60:
            self.refresh()
            if self._decodable_usable_s(language) >= target_min * 60:
                return f"{language} is full ({target_min:.0f} usable min)"
        return None

    def _decodable_usable_s(self, lang: str) -> float:
        """Usable seconds of ``lang`` whose audio actually decodes. Runs only when a language
        looks full (on the bank thread); a file that decoded once is not decoded again."""
        import soundfile as sf

        per_dev: dict[str, float] = defaultdict(float)
        for e in load_index(self.root, present_only=True):
            if e.get("lang") != lang or problems(e):
                continue
            path = self.root / e["file"]
            try:
                st = path.stat()
            except OSError:
                continue
            key = (e["file"], st.st_size, st.st_mtime_ns)    # a file changed since: decode it again
            if key not in self._decoded:
                try:
                    sf.read(str(self.root / e["file"]), dtype="float32")
                except Exception:
                    log.warning("voice bank: %s does not decode — not counted", e["file"])
                    continue
                self._decoded.add(key)
            per_dev[e.get("device") or "?"] += e.get("voiced_s", 0)
        return max(per_dev.values(), default=0.0)

    def bank(self, audio: np.ndarray, sr: int, *, block_id: str, ts: str, language: str | None, text: str,
             mode: str = "", device: str | None = None, fallback: str | None = None,
             preferred: str | None = None, target_min: float = 180.0, generation: int | None = None) -> dict | None:
        audio = np.asarray(audio, dtype=np.float32)
        if self._off(generation):
            return None
        if language is None:                 # an engine that reports none (Parakeet); a definite
            language = infer_language(text)  # other language from the ASR is still respected
        why = self.skip_reason(sr, len(audio) / sr if sr else 0, language=language, device=device,
                               fallback=fallback, preferred=preferred, target_min=target_min)
        if why:
            log.info("voice bank: skipped — %s", why)
            self.last = {"ts": ts, "lang": language, "seconds": round(len(audio) / sr, 1) if sr else 0.0,
                         "status": f"skipped — {why}"}
            return None
        import soundfile as sf

        m = measure(audio, sr)
        es, en = language_counts(text)
        day = (ts or datetime.now().isoformat())[:10]
        rel = Path(day) / f"{block_id}.flac"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        if self._off(generation):            # switched off while the take was being analysed
            return None
        sf.write(str(part), audio, sr, subtype="PCM_24", format="FLAC")
        entry = {"id": block_id, "ts": ts, "lang": language, "file": str(rel), "sr": sr, "device": device,
                 "mode": mode, **m.__dict__, "es_words": es, "en_words": en, "chars": len(text or "")}
        issues = problems(entry)
        with self._lock:
            # Last look under the lock: only a bank that is still on publishes the file and its
            # index line; otherwise the half-written take is removed.
            if self._off(generation):
                part.unlink(missing_ok=True)
                return None
            os.replace(part, path)
            fresh = self._tally is not None and self._index_sig() == self._tally_sig
            with (self.root / "index.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if fresh:
                self._tally_sig = self._index_sig()   # our own append: update in place below
            else:
                self._tally = None                     # changed behind our back: recount next time
            before = after = 0.0
            if self._tally is not None:
                self._tally["stored"] += m.duration_s
                self._tally["kept"][language] = self._tally["kept"].get(language, 0) + m.duration_s
                before = self._tally["usable"].get(language, 0)
                if not issues:
                    per = self._tally["usable_dev"].setdefault(language, {})
                    per[device or "?"] = per.get(device or "?", 0.0) + m.voiced_s
                    self._tally["usable"] = _best_device(self._tally["usable_dev"])
                after = self._tally["usable"].get(language, 0)
        log.info("voice bank: kept %.1fs %s (SNR %.0f dB, speech %.0f dBFS)%s", m.voiced_s, language,
                 m.snr_db, m.speech_dbfs, f" — not usable: {', '.join(issues)}" if issues else "")
        self.last = {"ts": ts, "lang": language, "seconds": m.voiced_s, "snr_db": m.snr_db,
                     "status": "usable ✓" if not issues else f"kept, not usable — {', '.join(issues)}"}
        self._announce(language, before, after, target_min)
        return entry

    def _off(self, generation: int | None = None) -> bool:
        if (self.enabled is not None and not self.enabled()) or (generation is not None and generation != self._generation):
            log.info("voice bank: skipped — switched off")
            return True
        return False

    def _announce(self, lang: str, before_s: float, after_s: float, target_min: float) -> None:
        """Tell the speaker when a language crosses a milestone, and (once a day) when most of what
        is being kept is too noisy to use — otherwise the bank fills with nothing usable, silently."""
        if self.notify is None:
            return
        name = {"es": "Spanish", "en": "English"}.get(lang, lang)
        for goal in [g for g in MILESTONES_MIN if g < target_min] + [target_min]:
            if before_s < goal * 60 <= after_s:
                if goal >= target_min:
                    msg = f"{name}: {goal:g} usable minutes — full, collection stopped. Export: dictdact voicebank export --lang {lang}"
                elif goal == 30:
                    msg = f"{name}: 30 usable minutes — enough to start an ElevenLabs Professional Voice Clone (2–3 h is best)"
                else:
                    msg = f"{name}: {goal} usable minutes collected"
                self._notify("Voice bank", msg)
        t = self.tally()
        kept = sum(t["kept"].values())       # recorded time: a take drowned in noise has no voiced time at all
        if kept < 5 * 60 or sum(t["usable"].values()) >= 0.25 * kept:
            return
        state_path = self.root / "state.json"
        try:
            state = json.loads(state_path.read_text())
        except (OSError, json.JSONDecodeError):
            state = {}
        today = datetime.now().strftime("%Y-%m-%d")
        if state.get("noise_warned") == today:
            return
        state["noise_warned"] = today
        state_path.write_text(json.dumps(state) + "\n")
        self._notify("Voice bank", "Most takes are too noisy to clone: the room is loud next to your voice. "
                                   "Speak closer to the mic or switch off the AC/fan while you dictate.")

    def _notify(self, title: str, message: str) -> None:
        log.info("voice bank: %s", message)
        try:
            self.notify(title, message)
        except Exception:
            log.exception("voice bank: notification failed")


# -- reading the bank ---------------------------------------------------------------------------

def summary(root: Path | None = None, min_snr: float = DEFAULT_MIN_SNR) -> dict:
    entries = load_index(root, present_only=True)
    out: dict = {"langs": {}, "stored_s": sum(e.get("duration_s", 0) for e in entries), "takes": len(entries)}
    for lang in LANGS:
        es = [e for e in entries if e.get("lang") == lang]
        if not es:
            continue
        ok_all = [e for e in es if not problems(e, min_snr)]
        # Progress counts ONE mic — the one with the most usable audio, which export will use.
        by_dev: dict[str, float] = defaultdict(float)
        for e in ok_all:
            by_dev[e.get("device") or "?"] += e["voiced_s"]
        device = max(by_dev, key=by_dev.get) if by_dev else None
        ok = [e for e in ok_all if (e.get("device") or "?") == device]
        reasons = Counter(r for e in es for r in problems(e, min_snr))
        if len(ok_all) > len(ok):
            reasons["another mic"] += len(ok_all) - len(ok)
        days = sorted({e["ts"][:10] for e in es})
        span = max(1, (datetime.fromisoformat(days[-1]) - datetime.fromisoformat(days[0])).days + 1)
        usable_s = sum(e["voiced_s"] for e in ok)
        out["langs"][lang] = {
            "takes": len(es), "usable_takes": len(ok),
            "all_min": sum(e["voiced_s"] for e in es) / 60, "usable_min": usable_s / 60,
            "recorded_min": sum(e.get("duration_s", 0) for e in es) / 60,
            "per_day_min": usable_s / 60 / span,
            "snr_db": statistics.median(e["snr_db"] for e in es),
            "speech_dbfs": statistics.median(e["speech_dbfs"] for e in es),
            "floor_dbfs": statistics.median(e["floor_dbfs"] for e in es),
            "clipped_takes": sum(1 for e in es if e.get("clipped", 0) >= 3),
            "hum_share": statistics.median(e.get("hum_share", 0.0) for e in es),
            "reasons": dict(reasons), "device": device,
            "devices": dict(Counter(e.get("device") for e in es)),
        }
    return out


LANG_NAMES = {"es": "Español", "en": "English"}
GOAL_MIN = 30                  # ElevenLabs PVC minimum


def _bar(frac: float, width: int = 10) -> str:
    n = max(0, min(width, int(round(frac * width))))
    return "▰" * n + "▱" * (width - n)


def menu_view(stats: dict, *, on: bool, last: dict | None, target_min: float = 180.0) -> dict:
    """Everything the menu-bar 'Voice bank' submenu shows, as plain strings (testable without
    AppKit): a title, one progress line per language (+ why takes were not usable), the latest
    take, the room verdict, and which languages can be exported."""
    langs = stats.get("langs", {})
    usable = {k: v["usable_min"] for k, v in langs.items()}
    if on:
        title = "Voice bank — ● recording" + (" · " + " · ".join(f"{k} {v:.1f}" for k, v in sorted(usable.items()))
                                               + " usable min" if usable else "")
    else:
        title = "Voice bank — off"
    lines = []
    for lang in LANGS:
        s = langs.get(lang)
        name = LANG_NAMES[lang]
        if not s:
            lines.append(f"{name}   {_bar(0)}   nothing yet")
            continue
        got = s["usable_min"]
        goal = GOAL_MIN if got < GOAL_MIN else target_min
        head = f"{got:.1f} / {goal:g} min usable" + (" ✓ ready for ElevenLabs" if got >= GOAL_MIN else "")
        lines.append(f"{name}   {_bar(got / goal)}   {head}  ·  {s.get('recorded_min', s['all_min']):.1f} min recorded, {s['takes']} takes")
        if s["reasons"]:
            # the language in the text keeps every line unique — rumps keys menu items by title
            lines.append(f"      ↳ {name} not usable: " + ", ".join(f"{k} ×{v}" for k, v in sorted(s["reasons"].items())))
    if last:
        lang = LANG_NAMES.get(last.get("lang") or "", last.get("lang") or "?")
        lines.append(f"Last take: {lang} · {last.get('seconds', 0):.0f} s · {last.get('status', '')}")
    snrs = [s["snr_db"] for s in langs.values()]
    if snrs:
        snr = min(snrs)
        if snr >= 28:
            lines.append(f"Room: your voice is {snr:.0f} dB above the noise — good")
        elif snr >= DEFAULT_MIN_SNR:
            lines.append(f"Room: your voice is {snr:.0f} dB above the noise — usable (28+ is ideal)")
        else:
            lines.append(f"Room: your voice is only {snr:.0f} dB above the noise — needs {DEFAULT_MIN_SNR:.0f}+")
            lines.append("Tip: speak 10–15 cm from the mic and switch the AC / fan off while dictating")
    exportable = {lang: usable.get(lang, 0) > 0 for lang in LANGS}
    return {"title": title, "lines": lines, "exportable": exportable, "usable": usable}


def window_view(*, on: bool, last: dict | None, target_min: float = 180.0, root: Path | None = None,
                texts: dict[str, str] | None = None, recent: int = 15,
                pace_min_per_day: dict[str, float] | None = None) -> dict:
    """What the window's Voice bank tab draws: per-language rings, the room gauge, and the latest
    takes with the words they carried — so the speaker can SEE each take land in the right
    language. ``texts`` maps block id → transcript (from history; the bank stores no text)."""
    root = root or bank_dir()
    stats = summary(root)
    entries = load_index(root, present_only=True)
    langs = {}
    for lang in LANGS:
        s = stats["langs"].get(lang)
        usable = s["usable_min"] if s else 0.0
        goal = GOAL_MIN if usable < GOAL_MIN else target_min
        # Pace = how much the user dictates in this language per day (history, last week) × the share of
        # their RECENT recording time that comes out usable — so moving closer to the mic shows up
        # in the estimate right away, and one good hour never promises "44 days".
        mine = sorted((e for e in entries if e.get("lang") == lang), key=lambda e: e.get("ts", ""))[-10:]
        rec_s = sum(e.get("duration_s", 0) for e in mine)
        share = sum(e.get("voiced_s", 0) for e in mine if not problems(e)) / rec_s if rec_s else 0.0
        rate = (pace_min_per_day or {}).get(lang, 0.0) * share
        langs[lang] = {
            "name": LANG_NAMES[lang], "usable_min": round(usable, 2), "goal_min": goal,
            "recorded_min": round(s.get("recorded_min", 0.0), 2) if s else 0.0,
            "takes": s["takes"] if s else 0, "usable_takes": s["usable_takes"] if s else 0,
            "ready": usable >= GOAL_MIN, "full": usable >= target_min,
            "eta_days": round((GOAL_MIN - usable) / rate, 1) if rate > 0 and usable < GOAL_MIN else None,
            "usable_min_per_day": round(rate, 1), "recent_usable_share": round(share, 2),
            "reasons": s["reasons"] if s else {},
        }
    rows = []
    if last and str(last.get("status", "")).startswith("skipped"):
        rows.append({"ts": last.get("ts"), "lang": last.get("lang"), "seconds": last.get("seconds", 0),
                     "snr_db": None, "usable": False, "status": last["status"], "text": ""})
    for e in sorted(entries, key=lambda e: e.get("ts", ""), reverse=True)[:recent]:
        issues = problems(e)
        rows.append({"ts": e.get("ts"), "lang": e.get("lang"), "seconds": e.get("voiced_s", 0),
                     "snr_db": e.get("snr_db"), "usable": not issues,
                     "status": "usable ✓" if not issues else ", ".join(issues),
                     "text": ((texts or {}).get(e["id"], "") or "")[:120]})
    last_snrs = [e["snr_db"] for e in sorted(entries, key=lambda e: e.get("ts", ""))[-10:]]
    room = None
    if last_snrs:
        now = statistics.median(last_snrs)
        room = {"median_snr": round(now, 1), "last_snr": round(last_snrs[-1], 1), "min_snr": DEFAULT_MIN_SNR,
                "good_snr": 28.0,
                "verdict": ("good" if now >= 28 else "usable" if now >= DEFAULT_MIN_SNR else "too noisy")}
    return {"on": on, "target_min": target_min, "goal_min": GOAL_MIN, "langs": langs, "takes": rows,
            "room": room, "stored_h": round(stats["stored_s"] / 3600, 2)}


def verdict(stats: dict) -> list[str]:
    """Plain-language read of the mic and room, from the median take."""
    lines = []
    for lang, s in stats["langs"].items():
        snr, lvl = s["snr_db"], s["speech_dbfs"]
        if snr >= 35:
            grade = "excellent — home-studio quality"
        elif snr >= 28:
            grade = "good — fine for a Professional Voice Clone"
        elif snr >= 20:
            grade = "usable, but the clone will learn the room noise too — quieter room or closer to the mic"
        else:
            grade = "too noisy for a clone as recorded"
        lines.append(f"{lang}: speech is {snr:.0f} dB above the room → {grade}.")
        if snr < 28 and s.get("hum_share", 0) >= 0.6:
            lines.append(f"{lang}: {s['hum_share'] * 100:.0f}% of the noise between your words is low hum (under 500 Hz: "
                         "air conditioning, fans, a fridge, traffic). The gain knob raises it together with your voice — "
                         "what helps is speaking 10–15 cm from the mic (each halving of the distance = +6 dB) or "
                         "switching the source off while you dictate.")
        if lvl < -32:
            lines.append(f"{lang}: speech averages {lvl:.0f} dBFS — quiet. Turn the mic gain up ~{min(15, -26 - lvl):.0f} dB "
                         "so it peaks around −12…−6 dBFS (exports normalize the level anyway).")
        elif s["clipped_takes"]:
            lines.append(f"{lang}: {s['clipped_takes']} take(s) clipped — turn the gain down a little.")
    return lines


def _candidates(lang: str, min_snr: float, device: str | None, root: Path) -> tuple[list[dict], str | None, Counter]:
    """Usable takes of ``lang`` on one mic, cleanest first. Callers take from the top until
    they have enough DECODED audio, so a missing or corrupt file is simply replaced by the next."""
    every = [e for e in load_index(root) if e.get("lang") == lang]
    entries = [e for e in every if (root / e.get("file", "")).is_file()]
    excluded = Counter(r for e in entries for r in problems(e, min_snr))
    if len(every) > len(entries):
        excluded["file missing or unreadable"] += len(every) - len(entries)
    ok = [e for e in entries if not problems(e, min_snr)]
    if device is None and ok:
        by_dev: dict[str, float] = defaultdict(float)
        for e in ok:
            by_dev[e.get("device") or "?"] += e["voiced_s"]
        device = max(by_dev, key=by_dev.get)
    if device is not None:
        other = [e for e in ok if (e.get("device") or "?") != device]
        if other:
            excluded["another mic"] += len(other)
        ok = [e for e in ok if (e.get("device") or "?") == device]
    return sorted(ok, key=lambda e: (-e["snr_db"], e["ts"])), device, excluded


def _load(e: dict, root: Path, sr_out: int = EXPORT_SR) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(str(root / e["file"]), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x[:, 0]
    y = tighten(x, sr)
    if sr != sr_out and len(y):
        g = math.gcd(sr, sr_out)
        y = signal.resample_poly(y, sr_out // g, sr // g).astype(np.float32)
    return y


def _level(y: np.ndarray, sr: int) -> np.ndarray:
    """Bring the take's speech to TARGET_SPEECH_DBFS without letting a sample past the ceiling.
    Gain only — no compression, no limiting, no EQ (ElevenLabs reproduces processing artifacts)."""
    if not len(y):
        return y
    frame_db, _ = _frames_db(y, sr)
    active, _ = _active(frame_db)
    speech = float(10 * np.log10(np.mean(10 ** (frame_db[active] / 10)) + 1e-20)) if active.any() else _db(float(np.sqrt(np.mean(y * y))))
    gain_db = min(TARGET_SPEECH_DBFS - speech, PEAK_CEILING_DBFS - _db(float(np.max(np.abs(y)))))
    return (y * 10 ** (gain_db / 20)).astype(np.float32)


def _write_chunk(y: np.ndarray, path: Path, fmt: str, sr: int = EXPORT_SR) -> Path:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "wav":
        out = path.with_suffix(".wav")
        sf.write(str(out), y, sr, subtype="PCM_16")
        return out
    tmp = path.with_suffix(".tmp.wav")
    sf.write(str(tmp), y, sr, subtype="PCM_24")
    out = path.with_suffix(".mp3")
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-codec:a", "libmp3lame", "-b:a", "192k", str(out)],
                       check=True, capture_output=True)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def export(lang: str, *, minutes: float = 180.0, out_dir: Path | None = None, fmt: str = "mp3",
           chunk_min: float = 10.0, min_snr: float = DEFAULT_MIN_SNR, device: str | None = None,
           root: Path | None = None) -> dict:
    """Write the cleanest ``minutes`` of ``lang`` as ElevenLabs-ready files (+ report.json)."""
    if lang not in LANGS:
        raise ValueError(f"language must be one of {LANGS}")
    if fmt not in ("mp3", "wav"):
        raise ValueError("format must be mp3 or wav")
    root = root or bank_dir()
    ranked, device, excluded = _candidates(lang, min_snr, device, root)
    if not ranked:
        raise RuntimeError(f"no usable {lang} takes in the bank yet")
    out_dir = out_dir or root / "exports" / f"{lang}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # Everything in the folder gets uploaded, so it must hold THIS export only: a leftover chunk
    # from an earlier run could carry a take this run excluded.
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"{out_dir} is not empty — export into a new folder")
    out_dir.mkdir(parents=True, exist_ok=True)
    gap = np.zeros(int(GAP_S * EXPORT_SR), dtype=np.float32)
    files, buf, buf_n, levels, missing, done_s, snrs = [], [], 0, [], 0, 0.0, []

    def flush():
        nonlocal buf, buf_n
        if not buf:
            return
        y = np.concatenate(buf)
        levels.append(_db(float(np.sqrt(np.mean(y * y)))))
        files.append(_write_chunk(y, out_dir / f"{lang}-{len(files) + 1:02d}", fmt))
        buf, buf_n = [], 0

    for e in ranked:                              # cleanest first; order does not matter for training
        if done_s >= minutes * 60:
            break
        try:
            y = _level(_load(e, root), EXPORT_SR)
        except (OSError, RuntimeError):           # corrupt audio behind a valid header: next take
            missing += 1
            excluded["file missing or unreadable"] += 1
            continue
        done_s += len(y) / EXPORT_SR
        snrs.append(e["snr_db"])
        buf += [y, gap]
        buf_n += len(y) + len(gap)
        if buf_n >= chunk_min * 60 * EXPORT_SR:
            flush()
    flush()
    if not files:
        raise RuntimeError(f"none of the {len(ranked)} usable {lang} takes could be read from {root}")
    report = {                                   # counts what was written, not what was selected
        "language": lang, "device": device, "format": fmt, "files": [f.name for f in files],
        "minutes": round(done_s / 60, 1), "takes": len(snrs), "unreadable": missing,
        "min_snr_db": min_snr, "median_snr_db": statistics.median(snrs),
        "file_rms_dbfs": [round(v, 1) for v in levels], "excluded": dict(excluded),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    short = "" if report["minutes"] >= GOAL_MIN else (
        f"\n⚠️  Only {report['minutes']} min here — a Professional Voice Clone needs at least {GOAL_MIN}.\n"
        f"   (An Instant Voice Clone works with 1–2 min.)\n")
    (out_dir / "UPLOAD-TO-ELEVENLABS.txt").write_text(
        f"{LANG_NAMES[lang]} — {report['minutes']} min of your own voice, {len(files)} file(s), {report['takes']} takes.\n"
        f"{short}\n"
        "1. elevenlabs.io → Voices → Add a new voice → Professional Voice Clone.\n"
        f"2. Language: {LANG_NAMES[lang]}. Upload EVERY audio file in this folder (not report.json).\n"
        "3. Do the voice verification: read the sentence it shows you out loud.\n"
        "4. Training takes a few hours; ElevenLabs emails you when it is ready.\n\n"
        "One voice per language: your Spanish and English accents are different, so export\n"
        "and upload each language as its own voice.\n", encoding="utf-8")
    report["dir"] = str(out_dir)
    return report


def reference(lang: str, *, seconds: float = 20.0, out: Path | None = None, min_snr: float = DEFAULT_MIN_SNR,
              root: Path | None = None) -> Path:
    """A ~``seconds`` reference clip (the cleanest takes, tightened and leveled) for zero-shot
    cloning such as the local Chatterbox voice."""
    import soundfile as sf

    root = root or bank_dir()
    ranked, _, _ = _candidates(lang, min_snr, None, root)
    if not ranked:
        raise RuntimeError(f"no usable {lang} takes in the bank yet")
    parts, n = [], 0
    gap = np.zeros(int(0.3 * EXPORT_SR), dtype=np.float32)
    for e in ranked:
        try:
            y = _level(_load(e, root), EXPORT_SR)[: int(seconds * EXPORT_SR) - n]
        except (OSError, RuntimeError):
            log.warning("voice bank: reference skips unreadable %s", e["file"])
            continue
        if len(y) < EXPORT_SR:          # a fragment under a second teaches the model nothing
            continue
        parts += [y, gap]
        n += len(y) + len(gap)
        if n >= seconds * EXPORT_SR:
            break
    if not parts:
        raise RuntimeError(f"no {lang} take long enough for a reference clip")
    out = out or root / f"reference-{lang}.wav"
    sf.write(str(out), np.concatenate(parts[:-1]), EXPORT_SR, subtype="PCM_16")
    return out


def say(text: str, lang: str, *, out: Path | None = None, ref: Path | None = None, root: Path | None = None,
        min_snr: float = DEFAULT_MIN_SNR) -> Path:
    """Speak ``text`` in the banked voice with the local Chatterbox clone (no cloud, no cost)."""
    if CLONE_PYTHON is None or CLONE_SCRIPT is None:
        raise RuntimeError(CLONE_NOT_CONFIGURED)
    if not CLONE_PYTHON.exists() or not CLONE_SCRIPT.exists():
        raise RuntimeError(f"local voice clone not found (DICTDACT_CLONE_PYTHON={CLONE_PYTHON}, "
                           f"DICTDACT_CLONE_SCRIPT={CLONE_SCRIPT})")
    ref = ref or reference(lang, root=root, min_snr=min_snr)
    out = (out or Path.cwd() / f"dictdact-voice-{lang}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.wav").resolve()
    subprocess.run([str(CLONE_PYTHON), str(CLONE_SCRIPT), "--ref", str(ref), "--lang", lang, "--out", str(out), text],
                   check=True)
    return out
