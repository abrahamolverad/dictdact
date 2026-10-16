"""Voice bank: raw-take collection, grading, and export for voice cloning.

Synthetic audio only — a 150 Hz fundamental + harmonics with a 4 Hz syllable envelope stands in
for "speech" (amplitude ~0.1), band-limited white noise stands in for "room" (amplitude ~1e-3).
Every test keeps its files under ``tmp_path`` (or an explicit ``root=``) — conftest.py already
isolates DICTDACT_CONFIG_DIR/DATA_DIR, but the bank itself also takes an explicit root, and the
instructions call for using it everywhere so nothing leaks into the shared per-session tmp dir.
"""
from __future__ import annotations

import shutil
import threading
import time
import types

import numpy as np
import pytest
import soundfile as sf
from scipy.ndimage import uniform_filter1d

from dictdact import voicebank as vb
from dictdact.voicebank import (VoiceBank, DEFAULT_MIN_SNR, EXPORT_SR, MAX_PAUSE_S, PEAK_CEILING_DBFS,
                                keep_mask, measure, problems, tighten)

SR = 44100
TS = "2026-09-23T10:00:00"

# Words that live in only ONE language (taken straight from format._ES_MARKERS/_EN_MARKERS so
# language_counts()/problems() can never be ambiguous about them).
ES_TEXT = "de la el los las un una que para pero porque esta estoy mas muy tambien"
EN_TEXT = "the of and that with this have has want make from they when very thanks"


def make_speech(seconds: float, sr: int = SR, amp: float = 0.1, seed: int = 0, syllable_hz: float = 4.0) -> np.ndarray:
    """150 Hz fundamental + 3 harmonics, gated at ``syllable_hz`` (duty ~55%) so there is real
    silence between "syllables", not just a sinusoidal dip."""
    t = np.arange(int(seconds * sr)) / sr
    sig = sum(np.sin(2 * np.pi * 150.0 * h * t) / h for h in (1, 2, 3, 4))
    phase = (t * syllable_hz) % 1.0
    env = uniform_filter1d((phase < 0.55).astype(np.float64), size=max(1, int(0.005 * sr)))
    sig = sig * env
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * amp
    return sig.astype(np.float32)


def make_room(seconds: float, sr: int = SR, amp: float = 1e-3, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 1, int(seconds * sr)).astype(np.float32) * amp)


def silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def good_audio(seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    """Clean speech-in-room: well above DEFAULT_MIN_SNR, not clipped, not whispered."""
    return (make_speech(seconds, seed=seed) + make_room(seconds, seed=seed + 100)).astype(np.float32)


def noisy_audio(seconds: float = 3.0, seed: int = 0) -> np.ndarray:
    """Same speech, room turned up until SNR falls below DEFAULT_MIN_SNR (~14 dB)."""
    return (make_speech(seconds, seed=seed) + make_room(seconds, amp=1e-2, seed=seed + 200)).astype(np.float32)


def bank_kwargs(**over):
    kw = dict(ts=TS, language="es", text=ES_TEXT, mode="normal", device="MacBook Pro Microphone",
              fallback=None, preferred=None, target_min=180.0)
    kw.update(over)
    return kw


# == 1. measure() ================================================================================

def test_measure_snr_near_the_constructed_ratio():
    x = good_audio(seconds=2.0)
    m = measure(x, SR)
    constructed_db = 20 * np.log10(0.1 / 1e-3)   # 40 dB — the amplitude ratio we built in
    assert abs(m.snr_db - constructed_db) < 12    # percentile-based estimator, not an exact match


def test_measure_speech_s_tracks_the_constructed_duty_cycle():
    x = good_audio(seconds=2.0)
    m = measure(x, SR)
    expected = 2.0 * 0.55   # the ~55% duty cycle baked into make_speech()
    assert abs(m.speech_s - expected) < 0.6


def test_measure_clipped_counts_full_scale_samples():
    x = silence(1.0)
    x[100:105] = 1.0        # 5 samples at full scale
    x[200] = -1.0            # 1 more, negative
    m = measure(x, SR)
    assert m.clipped == 6


def test_measure_pure_noise_does_not_report_high_snr():
    x = make_room(3.0, amp=1e-3)
    m = measure(x, SR)
    assert m.snr_db < 5     # no speech present — floor and "speech" level coincide


# == 2. keep_mask() / tighten() ==================================================================

def _long_pause_case():
    return np.concatenate([
        silence(2.0),
        (make_speech(1.0, seed=0) + make_room(1.0, seed=1)).astype(np.float32),
        make_room(3.0, amp=1e-4, seed=2),
        (make_speech(1.0, seed=3) + make_room(1.0, seed=4)).astype(np.float32),
        silence(2.0),
    ]).astype(np.float32)


def test_tighten_shortens_a_long_pause_and_drops_outer_silence():
    x = _long_pause_case()
    y = tighten(x, SR)
    # 1s speech + 0.6s (MAX_PAUSE_S) shortened pause + 1s speech, plus edge padding
    assert 2.6 <= len(y) / SR <= 3.4


def test_a_short_pause_is_kept_whole():
    x = np.concatenate([
        (make_speech(0.5, seed=10) + make_room(0.5, seed=11)).astype(np.float32),
        make_room(0.5, amp=1e-4, seed=12),   # well under MAX_PAUSE_S — nothing to shorten
        (make_speech(0.5, seed=13) + make_room(0.5, seed=14)).astype(np.float32),
    ]).astype(np.float32)
    y = tighten(x, SR)
    assert len(y) / SR == pytest.approx(len(x) / SR, abs=0.05)   # the pause is not cut at all


def test_tighten_never_invents_samples_except_the_fades():
    x = _long_pause_case()
    marker_idx = 2 * SR + 20000      # deep inside the first speech segment, clear of any fade edge
    marker = np.float32(0.123456)
    x[marker_idx] = marker
    y = tighten(x, SR)
    hits = np.flatnonzero(y == marker)
    assert len(hits) == 1            # present, unmodified, exactly once


def test_tighten_twice_is_nearly_idempotent():
    x = _long_pause_case()
    y1 = tighten(x, SR)
    y2 = tighten(y1, SR)
    assert abs(len(y2) - len(y1)) / SR < 0.15


def test_keep_mask_empty_for_pure_silence():
    frame_db, _ = vb._frames_db(silence(1.0), SR)
    assert not keep_mask(frame_db).any()


# == 3. VoiceBank.bank() =========================================================================

def _assert_nothing_written(root):
    assert not (root / "index.jsonl").exists()
    assert not list(root.rglob("*.flac"))


def test_skip_below_native_sample_rate(tmp_path):
    root = tmp_path / "case-sr"
    bank = VoiceBank(root=root)
    entry = bank.bank(good_audio(), 16000, block_id="x", **bank_kwargs())
    assert entry is None
    _assert_nothing_written(root)


def test_skip_fallback_mic(tmp_path):
    root = tmp_path / "case-fallback"
    bank = VoiceBank(root=root)
    entry = bank.bank(good_audio(), SR, block_id="x",
                      **bank_kwargs(fallback="lid closed: built-in mic is silent"))
    assert entry is None
    _assert_nothing_written(root)


def test_skip_non_preferred_mic(tmp_path):
    root = tmp_path / "case-nonpreferred"
    bank = VoiceBank(root=root)
    entry = bank.bank(good_audio(), SR, block_id="x",
                      **bank_kwargs(device="USB Mic", preferred="USB Studio Mic"))
    assert entry is None
    _assert_nothing_written(root)


def test_skip_unsupported_language(tmp_path):
    root = tmp_path / "case-lang"
    bank = VoiceBank(root=root)
    entry = bank.bank(good_audio(), SR, block_id="x", **bank_kwargs(language="fr"))
    assert entry is None
    _assert_nothing_written(root)


def test_skip_too_short(tmp_path):
    root = tmp_path / "case-short"
    bank = VoiceBank(root=root)
    entry = bank.bank(good_audio(seconds=1.0), SR, block_id="x", **bank_kwargs())
    assert entry is None
    _assert_nothing_written(root)


def test_skip_language_already_at_target(tmp_path):
    root = tmp_path / "case-full"
    bank = VoiceBank(root=root)
    assert bank.bank(good_audio(), SR, block_id="first", **bank_kwargs(target_min=0.01))   # 0.6 s cap
    entry = bank.bank(good_audio(seed=3), SR, block_id="x", **bank_kwargs(target_min=0.01))
    assert entry is None
    assert [e["id"] for e in vb.load_index(root)] == ["first"]
    assert not (root / TS[:10] / "x.flac").exists()


def test_skip_disk_cap(tmp_path, monkeypatch):
    root = tmp_path / "case-cap"
    monkeypatch.setattr(vb, "MAX_STORED_S", 0)
    bank = VoiceBank(root=root)
    entry = bank.bank(good_audio(), SR, block_id="x", **bank_kwargs())
    assert entry is None
    _assert_nothing_written(root)


def test_good_take_writes_flac_and_index(tmp_path):
    root = tmp_path / "case-good"
    bank = VoiceBank(root=root)
    audio = good_audio(seconds=3.0)
    entry = bank.bank(audio, SR, block_id="good1", **bank_kwargs())
    assert entry is not None
    for key in ("lang", "sr", "device", "snr_db", "voiced_s", "es_words", "en_words"):
        assert key in entry
    assert entry["lang"] == "es" and entry["sr"] == SR
    flac_path = root / entry["file"]
    assert flac_path.exists()
    x, sr = sf.read(str(flac_path), dtype="float32", always_2d=False)
    assert sr == SR
    assert len(x) == len(audio)
    assert np.max(np.abs(x - audio)) < 1e-5
    lines = (root / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert bank.usable_minutes().get("es", 0) > 0
    assert bank.tally()["stored"] > 0


def test_target_min_caps_one_language_but_not_the_other(tmp_path):
    root = tmp_path / "case-target"
    bank = VoiceBank(root=root)
    tiny = 0.01   # 0.6 s
    e1 = bank.bank(good_audio(seed=0), SR, block_id="es1", **bank_kwargs(target_min=tiny))
    assert e1 is not None
    e2 = bank.bank(good_audio(seed=1), SR, block_id="es2", **bank_kwargs(target_min=tiny))
    assert e2 is None                                # es is already "full" per the tiny target
    e3 = bank.bank(good_audio(seed=2), SR, block_id="en1",
                   **bank_kwargs(language="en", text=EN_TEXT, target_min=tiny))
    assert e3 is not None                            # en has not been touched yet


def test_bank_disabled_via_the_enabled_callback_writes_nothing(tmp_path):
    """cfg.voice_bank flipping off must also drop takes already queued, not just stop new
    offers — the bank re-reads ``enabled()`` right before every write (Pipeline wires this to
    ``cfg.voice_bank``)."""
    root = tmp_path / "case-disabled"
    bank = VoiceBank(root=root, enabled=lambda: False)
    entry = bank.bank(good_audio(), SR, block_id="x", **bank_kwargs())
    assert entry is None
    _assert_nothing_written(root)


def test_kept_minutes_counts_every_take_usable_or_not(tmp_path):
    root = tmp_path / "case-kept"
    bank = VoiceBank(root=root)
    bank.bank(good_audio(seconds=3.0), SR, block_id="ok", **bank_kwargs())
    bank.bank(noisy_audio(seconds=3.0, seed=9), SR, block_id="noisy", **bank_kwargs())
    kept = bank.kept_minutes()
    # both takes counted (kept = recorded time regardless of usability); usable only the clean one
    assert kept["es"] == pytest.approx(6.0 / 60, abs=1e-3)
    assert bank.usable_minutes()["es"] < kept["es"]


def test_announce_fires_on_crossing_the_30_minute_milestone(tmp_path):
    root = tmp_path / "case-announce"
    notes = []
    bank = VoiceBank(root=root, notify=lambda title, msg: notes.append((title, msg)))
    bank.tally()
    bank._tally["usable_dev"] = {"es": {"MacBook Pro Microphone": 29 * 60}}   # one take from 30 min
    bank._tally["usable"] = {"es": 29 * 60}
    bank._tally["kept"]["es"] = 29 * 60
    bank.bank(good_audio(seconds=90.0, seed=2), SR, block_id="z", **bank_kwargs())
    assert any("30 usable minutes" in msg for _, msg in notes)


# == 4. problems() ================================================================================

_CLEAN = dict(clipped=0, snr_db=30.0, speech_dbfs=-25.0, voiced_s=5.0, mode="normal",
             lang="es", es_words=10, en_words=0)


def test_problems_clean_entry_is_usable():
    assert problems(_CLEAN) == []


def test_problems_clipping():
    assert "clipping" in problems({**_CLEAN, "clipped": 3})


def test_problems_noisy_room():
    assert "noisy room" in problems({**_CLEAN, "snr_db": 10.0})


def test_problems_whispered_by_level():
    assert "whispered / far from the mic" in problems({**_CLEAN, "speech_dbfs": -55.0})


def test_problems_whispered_by_mode():
    assert "whispered / far from the mic" in problems({**_CLEAN, "mode": "whisper"})


def test_problems_too_short():
    assert "too short" in problems({**_CLEAN, "voiced_s": 1.0})


def test_problems_mixed_languages():
    assert "mixed languages" in problems({**_CLEAN, "es_words": 2, "en_words": 3})


# == 5. offer() + wait() =========================================================================

def test_offer_banks_on_a_background_thread(tmp_path):
    root = tmp_path / "case-thread"
    bank = VoiceBank(root=root)
    main_thread = threading.get_ident()
    seen = []
    real_bank = bank.bank

    def spy(*a, **kw):
        seen.append(threading.get_ident())
        return real_bank(*a, **kw)

    bank.bank = spy
    bank.offer(good_audio(), SR, block_id="t1", **bank_kwargs())
    bank.wait(timeout=5.0)
    assert seen and seen[0] != main_thread
    assert (root / "index.jsonl").exists()


def test_offer_never_raises_even_with_bad_metadata(tmp_path):
    root = tmp_path / "case-badmeta"
    bank = VoiceBank(root=root)
    # missing every required kwarg of bank(): would raise TypeError inside the worker, never here
    bank.offer(good_audio(), SR)
    bank.wait(timeout=2.0)
    _assert_nothing_written(root)   # the bad take never reached disk, but offer() itself is silent


# == 6. export() ==================================================================================

def _seed_export_bank(root):
    bank = VoiceBank(root=root)
    bank.bank(good_audio(seed=0), SR, block_id="a1", **bank_kwargs(device="MacBook Pro Microphone"))
    bank.bank(good_audio(seed=1), SR, block_id="a2", **bank_kwargs(device="MacBook Pro Microphone"))
    bank.bank(good_audio(seed=2), SR, block_id="b1", **bank_kwargs(device="USB Mic"))          # minority
    bank.bank(noisy_audio(seed=3), SR, block_id="a3", **bank_kwargs(device="MacBook Pro Microphone"))
    return bank


def test_export_majority_device_excludes_other_mic_and_noisy_take(tmp_path):
    root = tmp_path / "case-export"
    _seed_export_bank(root)
    report = vb.export("es", minutes=180.0, out_dir=root / "out", fmt="wav", chunk_min=10.0, root=root)
    assert report["device"] == "MacBook Pro Microphone"
    assert report["takes"] == 2
    assert "another mic" in report["excluded"]
    assert "noisy room" in report["excluded"]
    for f in report["files"]:
        assert (root / "out" / f).exists()
    assert (root / "out" / "report.json").exists()
    for field in ("language", "device", "format", "files", "minutes", "takes", "unreadable",
                  "min_snr_db", "median_snr_db", "file_rms_dbfs", "excluded", "created"):
        assert field in report


def test_export_wav_peak_is_under_the_ceiling(tmp_path):
    root = tmp_path / "case-peak"
    _seed_export_bank(root)
    report = vb.export("es", out_dir=root / "out", fmt="wav", root=root)
    for f in report["files"]:
        x, sr = sf.read(str(root / "out" / f))
        peak_dbfs = 20 * np.log10(np.max(np.abs(x)) + 1e-12)
        assert peak_dbfs <= PEAK_CEILING_DBFS + 0.1


def test_export_small_chunk_min_yields_multiple_files(tmp_path):
    root = tmp_path / "case-chunks"
    _seed_export_bank(root)
    report = vb.export("es", out_dir=root / "out", fmt="wav", chunk_min=0.05, root=root)
    assert len(report["files"]) > 1


def test_export_mp3_path():
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")

    def _run(root):
        _seed_export_bank(root)
        return vb.export("es", out_dir=root / "out", fmt="mp3", root=root)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        report = _run(root)
        assert report["files"]
        assert all(f.endswith(".mp3") for f in report["files"])
        assert all((root / "out" / f).exists() for f in report["files"])


def test_export_bad_language_raises_value_error(tmp_path):
    root = tmp_path / "case-badlang"
    _seed_export_bank(root)
    with pytest.raises(ValueError):
        vb.export("fr", root=root)


def test_export_bad_format_raises_value_error(tmp_path):
    root = tmp_path / "case-badfmt"
    _seed_export_bank(root)
    with pytest.raises(ValueError):
        vb.export("es", fmt="ogg", root=root)


def test_export_nothing_usable_raises_runtime_error(tmp_path):
    root = tmp_path / "case-empty"
    VoiceBank(root=root)   # never banks anything
    with pytest.raises(RuntimeError):
        vb.export("en", root=root)


# == 7. reference() ===============================================================================

def test_reference_writes_a_clip_of_about_the_requested_length(tmp_path):
    root = tmp_path / "case-ref"
    bank = VoiceBank(root=root)
    for i in range(3):
        bank.bank(good_audio(seconds=3.0, seed=i), SR, block_id=f"r{i}", **bank_kwargs())
    out = vb.reference("es", seconds=5.0, root=root, out=root / "ref.wav")
    x, sr = sf.read(str(out))
    assert sr == EXPORT_SR
    assert abs(len(x) / sr - 5.0) < 1.5


# == 8. say() ======================================================================================

def test_say_reports_clearly_when_local_cloning_is_not_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(vb, "CLONE_PYTHON", None)
    monkeypatch.setattr(vb, "CLONE_SCRIPT", None)
    with pytest.raises(RuntimeError, match="not configured.*DICTDACT_CLONE_PYTHON.*DICTDACT_CLONE_SCRIPT"):
        vb.say("hola", "es", root=tmp_path, ref=tmp_path / "ref.wav")


def test_clone_paths_come_only_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("DICTDACT_CLONE_PYTHON", raising=False)
    assert vb._env_path("DICTDACT_CLONE_PYTHON") is None
    monkeypatch.setenv("DICTDACT_CLONE_PYTHON", "  ")
    assert vb._env_path("DICTDACT_CLONE_PYTHON") is None
    monkeypatch.setenv("DICTDACT_CLONE_PYTHON", str(tmp_path / "py"))
    assert vb._env_path("DICTDACT_CLONE_PYTHON") == tmp_path / "py"


def test_say_raises_when_the_local_clone_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(vb, "CLONE_PYTHON", tmp_path / "no-such-python")
    monkeypatch.setattr(vb, "CLONE_SCRIPT", tmp_path / "no-such-script.py")
    with pytest.raises(RuntimeError):
        vb.say("hola", "es", root=tmp_path, ref=tmp_path / "ref.wav")


def test_say_builds_the_expected_command(tmp_path, monkeypatch):
    fake_python = tmp_path / "python"
    fake_python.touch()
    fake_script = tmp_path / "clone.py"
    fake_script.touch()
    monkeypatch.setattr(vb, "CLONE_PYTHON", fake_python)
    monkeypatch.setattr(vb, "CLONE_SCRIPT", fake_script)
    ref = tmp_path / "ref.wav"
    ref.touch()
    out = tmp_path / "out.wav"
    captured = {}

    def fake_run(cmd, check=True):
        captured["cmd"] = cmd

    monkeypatch.setattr(vb.subprocess, "run", fake_run)
    result = vb.say("hola mundo", "es", out=out, ref=ref, root=tmp_path)
    assert result == out.resolve()
    cmd = captured["cmd"]
    assert cmd[0] == str(fake_python)
    assert cmd[1] == str(fake_script)
    assert "--ref" in cmd and str(ref) in cmd
    assert "--lang" in cmd and "es" in cmd
    assert "--out" in cmd and str(out.resolve()) in cmd
    assert "hola mundo" in cmd


# == 9. Recorder (native vs target sample rate) ===================================================

def test_pick_samplerate_native_true_uses_device_default(monkeypatch):
    import dictdact.audio as au

    monkeypatch.setattr(au.sd, "query_devices", lambda idx=None: {"default_samplerate": 44100.0})
    monkeypatch.setattr(au.sd, "check_input_settings", lambda **kw: None)
    r = au.Recorder(None)
    r.native = True
    assert r._pick_samplerate(0) == 44100


def test_pick_samplerate_native_false_uses_target_sr(monkeypatch):
    import dictdact.audio as au

    monkeypatch.setattr(au.sd, "query_devices", lambda idx=None: {"default_samplerate": 44100.0})
    monkeypatch.setattr(au.sd, "check_input_settings", lambda **kw: None)
    r = au.Recorder(None)
    r.native = False
    assert r._pick_samplerate(0) == au.TARGET_SR == 16000


def test_take_raw_does_not_resample():
    import dictdact.audio as au

    r = au.Recorder(None)
    r._chunks = [np.ones(4410, dtype=np.float32) * 0.1]
    r._samplerate = 44100
    r._stream = "FAKE-STREAM"
    audio, sr, stream = r.take_raw()
    assert sr == 44100
    assert len(audio) == 4410
    assert stream == "FAKE-STREAM"


def test_take_audio_resamples_to_16k():
    import dictdact.audio as au

    r = au.Recorder(None)
    r._chunks = [np.ones(4410, dtype=np.float32) * 0.1]   # 0.1 s @ 44100
    r._samplerate = 44100
    r._stream = "FAKE-STREAM"
    audio16, stream = r.take_audio()
    assert stream == "FAKE-STREAM"
    assert abs(len(audio16) - 1600) <= 2   # 0.1 s @ 16000


# == 10. Pipeline._process ========================================================================

def _stub_pipeline(tmp_path, voice_bank: bool):
    from dictdact.config import Config
    from dictdact.history import History
    from dictdact.pipeline import Pipeline

    p = Pipeline.__new__(Pipeline)
    p.cfg = Config()
    p.cfg.paste = False
    p.cfg.copy_to_clipboard = False
    p.cfg.sounds = False
    p.cfg.keep_recordings = False
    p.cfg.voice_bank = voice_bank
    p.emit = lambda kind, payload=None: None
    p.history = History(path=tmp_path / "history.jsonl")
    p.context = types.SimpleNamespace(take=lambda: (None, []))
    p.recorder = None
    p._paste_lock = threading.Lock()
    p._last_retarget_app = None
    p._last_model_use = 0.0
    offers = []
    p.voicebank = types.SimpleNamespace(offer=lambda *a, **kw: offers.append((a, kw)))
    return p, offers


def _stub_engine_and_formatter(p):
    from dictdact.transcribe import Transcript

    p.engine = types.SimpleNamespace(
        transcribe=lambda x, extra_terms=None: Transcript(text="hola mundo", language="es", engine="fake"))
    p.formatter = types.SimpleNamespace(name="fake", format=lambda t: t)


def test_process_offers_native_audio_when_voice_bank_is_on(tmp_path):
    from dictdact.audio import DeviceInfo

    p, offers = _stub_pipeline(tmp_path, voice_bank=True)
    _stub_engine_and_formatter(p)
    audio = good_audio(seconds=2.0)
    device = DeviceInfo(index=0, name="Mic", channels=1, default_samplerate=44100.0, is_default=True)
    block = p._process(audio, "SomeApp", sr=44100, device=device)
    assert block is not None
    assert len(offers) == 1
    args, kwargs = offers[0]
    assert args[0] is audio           # the untouched native array, not the 16 kHz copy
    assert args[1] == 44100
    assert kwargs["language"] == "es"


def test_process_does_not_offer_when_voice_bank_is_off(tmp_path):
    from dictdact.audio import DeviceInfo

    p, offers = _stub_pipeline(tmp_path, voice_bank=False)
    _stub_engine_and_formatter(p)
    audio = good_audio(seconds=2.0, seed=5)
    device = DeviceInfo(index=0, name="Mic", channels=1, default_samplerate=44100.0, is_default=True)
    p._process(audio, "SomeApp", sr=44100, device=device)
    assert offers == []


def test_process_does_not_offer_when_already_at_target_sr(tmp_path):
    from dictdact.audio import DeviceInfo, TARGET_SR

    p, offers = _stub_pipeline(tmp_path, voice_bank=True)
    _stub_engine_and_formatter(p)
    audio16 = (make_speech(2.0, sr=TARGET_SR, seed=7) + make_room(2.0, sr=TARGET_SR, seed=8)).astype(np.float32)
    device = DeviceInfo(index=0, name="Mic", channels=1, default_samplerate=44100.0, is_default=True)
    p._process(audio16, "SomeApp", sr=TARGET_SR, device=device)
    assert offers == []


# == 11. CLI ========================================================================================

def test_cli_voicebank_status_on_empty_bank_does_not_crash(capsys):
    import dictdact.__main__ as main_module

    main_module.main(["voicebank", "status"])
    out = capsys.readouterr().out
    assert "voice bank" in out.lower()


def test_cli_voicebank_off_flips_the_saved_config_when_no_app_is_running(monkeypatch, capsys):
    import dictdact.__main__ as main_module
    from dictdact.config import Config

    monkeypatch.setattr(main_module, "_notify_running_app", lambda *a, **kw: False)
    cfg = Config.load()
    cfg.voice_bank = True
    cfg.save()

    main_module.main(["voicebank", "off"])

    out = capsys.readouterr().out
    assert "voice bank off" in out
    assert "not running" in out
    assert Config.load().voice_bank is False


# == 12. server.EDITABLE ============================================================================

def test_voice_bank_is_server_editable():
    from dictdact.server import EDITABLE

    assert "voice_bank" in EDITABLE
    assert EDITABLE["voice_bank"] == [True, False]


# == regressions: queued takes and unreadable files ============================================

def test_switching_off_drops_takes_already_queued(tmp_path):
    """offer() only checked the switch at hand-over; a take sitting in the queue
    was still written after the speaker switched the bank off."""
    on = {"v": True}
    gate = threading.Event()
    bank = VoiceBank(root=tmp_path, enabled=lambda: on["v"])
    real_bank = bank.bank

    def slow_bank(*a, **kw):
        gate.wait(5)                     # the take is queued, not yet written…
        return real_bank(*a, **kw)

    bank.bank = slow_bank
    bank.offer(good_audio(3.0), SR, block_id="queued", **bank_kwargs())
    on["v"] = False                      # …when the speaker switches the bank off
    gate.set()
    bank.wait()
    assert vb.load_index(tmp_path) == []
    assert not list(tmp_path.rglob("*.flac"))


def test_export_counts_only_what_it_wrote_and_fails_when_nothing_was(tmp_path):
    """export summed the SELECTED takes, so an unreadable file was reported as
    exported minutes and an export with zero files 'succeeded'."""
    bank = VoiceBank(root=tmp_path)
    for i in range(2):
        assert bank.bank(good_audio(20.0, seed=i), SR, block_id=f"t{i}", **bank_kwargs(ts=f"2026-09-23T10:00:0{i}"))
    entries = vb.load_index(tmp_path)
    (tmp_path / entries[0]["file"]).unlink()
    r = vb.export("es", root=tmp_path, out_dir=tmp_path / "out1", fmt="wav")
    assert r["takes"] == 1 and r["excluded"].get("file missing or unreadable") == 1
    written = sum(sf.info(str(tmp_path / "out1" / f)).duration for f in r["files"])
    assert abs(r["minutes"] * 60 - written) < 3.5      # minutes are rounded to 0.1 (6 s)
    assert r["minutes"] < 0.5                           # one 20 s take, not the two selected
    (tmp_path / entries[1]["file"]).unlink()
    with pytest.raises(RuntimeError):
        vb.export("es", root=tmp_path, out_dir=tmp_path / "out2", fmt="wav")


def test_noisy_room_warning_once_a_day(tmp_path):
    """Five recorded minutes that are mostly unusable → one notification per day, not one per take."""
    got = []
    bank = VoiceBank(root=tmp_path, notify=lambda title, msg: got.append(msg))
    loud = noisy_audio(62.0)
    for i in range(6):
        bank.bank(loud, SR, block_id=f"n{i}", **bank_kwargs(ts=f"2026-09-23T10:0{i}:00"))
    warnings = [m for m in got if "too noisy" in m]
    assert len(warnings) == 1
    assert (tmp_path / "state.json").exists()


def test_hum_share_tells_hum_from_hiss():
    t = np.arange(4 * SR) / SR
    speech = make_speech(4.0)
    hum = (0.01 * np.sin(2 * np.pi * 108 * t)).astype(np.float32)
    assert measure(speech + hum, SR).hum_share > 0.8
    assert measure(speech + make_room(4.0, amp=3e-3), SR).hum_share < 0.2


# == regressions: switch-off during analysis, no-language engines, stale export chunks =========

def test_switching_off_during_analysis_publishes_nothing(tmp_path, monkeypatch):
    """The switch was read once, before the analysis; turning the bank off while
    a take was being measured still wrote it."""
    on = {"v": True}
    bank = VoiceBank(root=tmp_path, enabled=lambda: on["v"])
    real_measure = vb.measure

    def measure_then_switch_off(x, sr):
        m = real_measure(x, sr)
        on["v"] = False
        return m

    monkeypatch.setattr(vb, "measure", measure_then_switch_off)
    assert bank.bank(good_audio(3.0), SR, block_id="mid", **bank_kwargs()) is None
    assert vb.load_index(tmp_path) == []
    assert not list(tmp_path.rglob("*.flac")) and not list(tmp_path.rglob("*.part"))


def test_engine_without_language_is_banked_from_the_words(tmp_path):
    """Parakeet reports no language, and the bank dropped every take."""
    bank = VoiceBank(root=tmp_path)
    e = bank.bank(good_audio(3.0), SR, block_id="pk", **bank_kwargs(language=None, text=ES_TEXT))
    assert e is not None and e["lang"] == "es"
    assert vb.infer_language(EN_TEXT) == "en"
    assert vb.infer_language("ok") is None
    assert vb.infer_language("de la el the of and") is None        # a 50/50 mix stays unknown


def test_export_refuses_a_folder_that_already_has_files(tmp_path):
    """A second export into the same folder left the first run's extra chunks —
    and the instructions say to upload everything in that folder."""
    bank = VoiceBank(root=tmp_path)
    assert bank.bank(good_audio(4.0), SR, block_id="a", **bank_kwargs())
    out = tmp_path / "out"
    vb.export("es", root=tmp_path, out_dir=out, fmt="wav")
    with pytest.raises(ValueError):
        vb.export("es", root=tmp_path, out_dir=out, fmt="wav")


# == regressions: config setter and the publication lock =======================================

def test_switched_off_waits_for_a_publication_in_progress(tmp_path):
    """The config setter never took the bank's publication lock, so 'off' could
    return while a take was still being published after it."""
    on = {"v": True}
    bank = VoiceBank(root=tmp_path, enabled=lambda: on["v"])
    started, release = threading.Event(), threading.Event()
    real_open = type(tmp_path).open

    def slow_open(self, *a, **kw):             # the index append happens under the lock
        if self.name == "index.jsonl":
            started.set()
            release.wait(5)
        return real_open(self, *a, **kw)

    import pathlib
    orig = pathlib.Path.open
    pathlib.Path.open = slow_open
    try:
        t = threading.Thread(target=lambda: bank.bank(good_audio(3.0), SR, block_id="inflight", **bank_kwargs()))
        t.start()
        assert started.wait(5)
        on["v"] = False
        done = threading.Event()
        threading.Thread(target=lambda: (bank.switched_off(), done.set())).start()
        assert not done.wait(0.2)              # off waits for the publication already under way…
        release.set()
        assert done.wait(5)
        t.join(5)
    finally:
        pathlib.Path.open = orig
    n = len(vb.load_index(tmp_path))
    assert n == 1                              # …which completes; nothing after the call returns
    bank.bank(good_audio(3.0, seed=2), SR, block_id="after", **bank_kwargs(ts="2026-09-23T10:00:05"))
    assert len(vb.load_index(tmp_path)) == n


def test_takes_queued_before_off_are_dropped_even_if_switched_back_on(tmp_path):
    on = {"v": True}
    gate = threading.Event()
    bank = VoiceBank(root=tmp_path, enabled=lambda: on["v"])
    real_bank = bank.bank
    bank.bank = lambda *a, **kw: (gate.wait(5), real_bank(*a, **kw))[1]
    bank.offer(good_audio(3.0), SR, block_id="queued", **bank_kwargs())
    on["v"] = False
    bank.switched_off()
    on["v"] = True                             # back on before the queued take is processed
    gate.set()
    bank.wait()
    assert vb.load_index(tmp_path) == []


def test_export_and_reference_backfill_a_missing_file(tmp_path):
    bank = VoiceBank(root=tmp_path)
    for i, seed in enumerate((0, 1)):
        assert bank.bank(good_audio(20.0, seed=seed), SR, block_id=f"t{i}", **bank_kwargs(ts=f"2026-09-23T10:00:0{i}"))
    best = max(vb.load_index(tmp_path), key=lambda e: e["snr_db"])
    (tmp_path / best["file"]).unlink()
    r = vb.export("es", root=tmp_path, out_dir=tmp_path / "out", fmt="wav", minutes=0.2)
    assert r["takes"] == 1 and r["excluded"].get("file missing or unreadable") == 1
    ref = vb.reference("es", root=tmp_path, seconds=5, out=tmp_path / "ref.wav")
    assert sf.info(str(ref)).duration > 4


# == regressions: generation stamping, deleted takes, corrupt FLAC =============================

def test_a_take_started_before_off_stays_out_even_if_switched_back_on_mid_transcription(tmp_path):
    """The generation was stamped when the take reached the bank, after
    transcription — off→on during transcription revived a take recorded before 'off'."""
    bank = VoiceBank(root=tmp_path, enabled=lambda: True)
    stamped = bank.generation                  # recording starts (the pipeline stamps it here)
    bank.switched_off()                        # off…
    bank.offer(good_audio(3.0), SR, block_id="old", generation=stamped, **bank_kwargs())   # …back on
    bank.wait()
    assert vb.load_index(tmp_path) == []
    bank.offer(good_audio(3.0), SR, block_id="new", generation=bank.generation, **bank_kwargs())
    bank.wait()
    assert [e["id"] for e in vb.load_index(tmp_path)] == ["new"]


def test_deleted_takes_stop_counting_toward_full(tmp_path):
    """A take deleted by hand still counted, so a language could stay 'full'
    forever with nothing left to export."""
    bank = VoiceBank(root=tmp_path)
    first = bank.bank(good_audio(), SR, block_id="first", **bank_kwargs(target_min=0.01))
    assert bank.bank(good_audio(seed=3), SR, block_id="x", **bank_kwargs(target_min=0.01)) is None
    (tmp_path / first["file"]).unlink()
    assert bank.bank(good_audio(seed=4), SR, block_id="y", **bank_kwargs(target_min=0.01)) is not None
    assert vb.summary(tmp_path)["takes"] == 1


def test_deleting_the_whole_bank_resets_the_running_tally(tmp_path):
    root = tmp_path / "bank"
    bank = VoiceBank(root=root)
    assert bank.bank(good_audio(), SR, block_id="a", **bank_kwargs())
    assert bank.usable_minutes().get("es", 0) > 0
    shutil.rmtree(root)
    assert bank.usable_minutes() == {} and bank.kept_minutes() == {}


def test_export_backfills_a_take_whose_audio_does_not_decode(tmp_path):
    """A FLAC with a valid header but corrupt audio passed the header check and
    was never replaced."""
    bank = VoiceBank(root=tmp_path)
    for i, seed in enumerate((0, 1)):
        assert bank.bank(good_audio(20.0, seed=seed), SR, block_id=f"t{i}", **bank_kwargs(ts=f"2026-09-23T10:00:0{i}"))
    best = max(vb.load_index(tmp_path), key=lambda e: e["snr_db"])
    path = tmp_path / best["file"]
    data = bytearray(path.read_bytes())
    for i in range(len(data) // 3, len(data) - 16):   # keep the header, wreck the frames
        data[i] = 0x55
    path.write_bytes(bytes(data))
    r = vb.export("es", root=tmp_path, out_dir=tmp_path / "out", fmt="wav", minutes=0.2)
    assert r["takes"] == 1 and r["excluded"].get("file missing or unreadable") == 1


# == regressions: opt-in proof and corrupt files in usable minutes =============================

def test_a_take_recorded_while_off_stays_out_even_on_a_mic_that_rejects_16k(tmp_path):
    """The sample rate was taken as proof of opt-in, but a mic that rejects
    16 kHz opens at its native rate with the bank off; switching the bank on after the take
    stopped then banked audio recorded while it was off."""
    import queue as _queue

    from dictdact.audio import DeviceInfo

    p, offers = _stub_pipeline(tmp_path, voice_bank=False)
    _stub_engine_and_formatter(p)
    p.voicebank.generation = 0
    p.before_record = None
    p._jobs = _queue.Queue()
    audio48 = (make_speech(2.0, sr=48000) + make_room(2.0, sr=48000)).astype(np.float32)

    class Rec:                                   # a mic that only does 48 kHz
        native = False
        recording = False
        device = DeviceInfo(index=0, name="Mic", channels=1, default_samplerate=48000.0, is_default=True)

        def start(self):
            self.recording = True

        def take_raw(self):
            self.recording = False
            return audio48, 48000, None

        @staticmethod
        def close_stream(stream):
            return True

    p.recorder = Rec()
    p._do_start()                                # bank OFF when the take starts
    p._do_stop("SomeApp")
    p.cfg.voice_bank = True                      # switched on before the take is processed
    p._jobs.get_nowait()()
    assert offers == []


def test_disk_cap_frees_room_when_files_are_deleted(tmp_path, monkeypatch):
    bank = VoiceBank(root=tmp_path)
    first = bank.bank(good_audio(), SR, block_id="a", **bank_kwargs())
    monkeypatch.setattr(vb, "MAX_STORED_S", first["duration_s"])       # the bank is now at its cap
    assert bank.bank(good_audio(seed=3), SR, block_id="b", **bank_kwargs()) is None
    (tmp_path / first["file"]).unlink()
    assert bank.bank(good_audio(seed=4), SR, block_id="c", **bank_kwargs()) is not None


def test_a_full_language_must_hold_audio_that_decodes(tmp_path):
    """Corrupt files counted as usable minutes and could stop collection with
    nothing exportable."""
    bank = VoiceBank(root=tmp_path)
    first = bank.bank(good_audio(), SR, block_id="a", **bank_kwargs(target_min=0.01))
    path = tmp_path / first["file"]
    data = bytearray(path.read_bytes())
    for i in range(len(data) // 3, len(data) - 16):
        data[i] = 0x55
    path.write_bytes(bytes(data))
    assert bank.bank(good_audio(seed=3), SR, block_id="b", **bank_kwargs(target_min=0.01)) is not None


# == regressions: --min-snr reach and decode-cache invalidation ================================

def test_reference_and_say_honour_min_snr(tmp_path, monkeypatch):
    """--min-snr reached status/export but not reference/say."""
    bank = VoiceBank(root=tmp_path)
    assert bank.bank(noisy_audio(12.0), SR, block_id="n", **bank_kwargs())   # ~14 dB: under the default bar
    with pytest.raises(RuntimeError):
        vb.reference("es", root=tmp_path, seconds=5, out=tmp_path / "r0.wav")
    assert vb.reference("es", root=tmp_path, seconds=5, out=tmp_path / "r1.wav", min_snr=5).exists()
    calls = []
    monkeypatch.setattr(vb, "CLONE_PYTHON", tmp_path / "r1.wav")     # any existing path
    monkeypatch.setattr(vb, "CLONE_SCRIPT", tmp_path / "r1.wav")
    monkeypatch.setattr(vb.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    vb.say("hola", "es", root=tmp_path, out=tmp_path / "o.wav", min_snr=5)
    assert calls and "--ref" in calls[0]


def test_decode_cache_notices_a_file_that_changed(tmp_path):
    """A file proven to decode was never decoded again, even after it changed."""
    bank = VoiceBank(root=tmp_path)
    first = bank.bank(good_audio(), SR, block_id="a", **bank_kwargs(target_min=0.01))
    assert bank.bank(good_audio(seed=3), SR, block_id="b", **bank_kwargs(target_min=0.01)) is None   # cached as good
    path = tmp_path / first["file"]
    data = bytearray(path.read_bytes())
    for i in range(len(data) // 3, len(data) - 16):
        data[i] = 0x55
    path.write_bytes(bytes(data))
    assert bank.bank(good_audio(seed=4), SR, block_id="c", **bank_kwargs(target_min=0.01)) is not None


# == Menu: progress per language ===============================================================

def test_menu_view_shows_progress_per_language_and_the_last_take(tmp_path):
    bank = VoiceBank(root=tmp_path)
    assert bank.bank(good_audio(6.0), SR, block_id="es1", **bank_kwargs())
    assert bank.bank(noisy_audio(6.0), SR, block_id="en1", **bank_kwargs(language="en", text=EN_TEXT,
                                                                           ts="2026-09-23T10:00:09"))
    v = vb.menu_view(vb.summary(tmp_path), on=True, last=bank.last)
    assert v["title"].startswith("Voice bank — ● recording")
    es = next(l for l in v["lines"] if l.startswith("Español"))
    en = next(l for l in v["lines"] if l.startswith("English"))
    assert "/ 30 min usable" in es and "1 takes" in es
    assert "0.0 / 30 min usable" in en
    assert any("English not usable: noisy room" in l for l in v["lines"])
    assert v["lines"][-3].startswith("Last take: English") or any(l.startswith("Last take: English") for l in v["lines"])
    assert v["exportable"] == {"es": True, "en": False}
    assert len(v["lines"]) == len(set(v["lines"]))          # rumps keys items by title
    off = vb.menu_view(vb.summary(tmp_path), on=False, last=None)
    assert off["title"] == "Voice bank — off"


def test_menu_view_marks_ready_after_30_minutes():
    stats = {"langs": {"es": {"usable_min": 31.0, "all_min": 40.0, "takes": 200, "reasons": {}, "snr_db": 30.0}}}
    v = vb.menu_view(stats, on=True, last=None)
    es = v["lines"][0]
    assert "31.0 / 180 min usable ✓ ready for ElevenLabs" in es
    assert any(l.startswith("English") and "nothing yet" in l for l in v["lines"])


def test_export_writes_upload_instructions(tmp_path):
    bank = VoiceBank(root=tmp_path)
    assert bank.bank(good_audio(6.0), SR, block_id="a", **bank_kwargs())
    r = vb.export("es", root=tmp_path, out_dir=tmp_path / "out", fmt="wav")
    txt = (tmp_path / "out" / "UPLOAD-TO-ELEVENLABS.txt").read_text()
    assert "Professional Voice Clone" in txt and "Español" in txt and "needs at least 30" in txt
    assert "UPLOAD-TO-ELEVENLABS.txt" not in r["files"]


def test_off_between_the_two_start_reads_still_keeps_the_take_out(tmp_path):
    """The setting was read before the generation, so an 'off' landing
    between the two reads — then 'on' again before processing — banked an off-started take."""
    import queue as _queue

    from dictdact.audio import DeviceInfo

    p, offers = _stub_pipeline(tmp_path, voice_bank=True)
    _stub_engine_and_formatter(p)
    bank = VoiceBank(root=tmp_path, enabled=lambda: p.cfg.voice_bank)
    p.voicebank = bank
    p.before_record = None
    p._jobs = _queue.Queue()
    audio = good_audio(3.0)

    class Rec:
        native = False
        recording = False
        device = DeviceInfo(index=0, name="Mic", channels=1, default_samplerate=44100.0, is_default=True)

        def start(self):
            self.recording = True

        def take_raw(self):
            self.recording = False
            return audio, 44100, None

        @staticmethod
        def close_stream(stream):
            return True

    p.recorder = Rec()
    real_gen = VoiceBank.generation

    class Gen:                                   # 'off' completes just as the generation is read
        def __get__(self, obj, owner):
            p.cfg.voice_bank = False
            obj.switched_off()
            return real_gen.__get__(obj, owner)  # …so the read sees the post-'off' value

    monkeypatch_gen = Gen()
    type(bank).generation = monkeypatch_gen
    try:
        p._do_start()
    finally:
        type(bank).generation = real_gen
    p._do_stop("SomeApp")
    p.cfg.voice_bank = True                      # back on before processing
    p._jobs.get_nowait()()
    bank.wait()
    assert vb.load_index(tmp_path) == []


def test_menu_shows_recorded_minutes_including_pauses(tmp_path):
    bank = VoiceBank(root=tmp_path)
    take = np.concatenate([good_audio(4.0), silence(4.0), good_audio(4.0, seed=2)])
    assert bank.bank(take, SR, block_id="p", **bank_kwargs())
    st = vb.summary(tmp_path)["langs"]["es"]
    assert st["recorded_min"] > st["all_min"]
    line = next(l for l in vb.menu_view(vb.summary(tmp_path), on=True, last=None)["lines"] if l.startswith("Español"))
    assert f"{st['recorded_min']:.1f} min recorded" in line


# == Window tab: voice-bank progress ===========================================================

def test_window_view_rings_room_and_takes_with_their_words(tmp_path):
    bank = VoiceBank(root=tmp_path)
    for i in range(3):
        assert bank.bank(good_audio(8.0, seed=i), SR, block_id=f"es{i}", **bank_kwargs(ts=f"2026-09-23T10:0{i}:00"))
    assert bank.bank(noisy_audio(8.0), SR, block_id="en0",
                     **bank_kwargs(language="en", text=EN_TEXT, ts="2026-09-23T10:05:00"))
    v = vb.window_view(on=True, last=None, root=tmp_path, texts={"es0": "hola que tal", "en0": "hello there"},
                       pace_min_per_day={"es": 20.0, "en": 10.0})
    es, en = v["langs"]["es"], v["langs"]["en"]
    assert es["usable_takes"] == 3 and es["goal_min"] == 30 and 0 < es["usable_min"] < 1
    assert es["recent_usable_share"] > 0.5 and es["eta_days"] is not None and es["eta_days"] < 10
    assert en["usable_takes"] == 0 and en["eta_days"] is None and en["reasons"] == {"noisy room": 1}
    assert v["takes"][0]["lang"] == "en" and v["takes"][0]["text"] == "hello there" and not v["takes"][0]["usable"]
    assert any(t["text"] == "hola que tal" and t["usable"] for t in v["takes"])
    assert v["room"]["median_snr"] > 0 and v["room"]["verdict"] in ("good", "usable", "too noisy")


def test_window_view_shows_a_skipped_take_on_top(tmp_path):
    v = vb.window_view(on=True, root=tmp_path, last={"ts": "2026-09-23T11:00:00", "lang": "es", "seconds": 3,
                                                     "status": "skipped — not the chosen mic (MacBook Pro Microphone)"})
    assert v["takes"][0]["status"].startswith("skipped") and v["room"] is None


def _backend(tmp_path, monkeypatch):
    from dictdact.config import Config
    from dictdact.history import Block, History
    from dictdact.server import Backend

    monkeypatch.setattr(vb, "bank_dir", lambda: tmp_path)
    hist = History(path=tmp_path / "history.jsonl")
    blk = Block.new(text=ES_TEXT, raw=ES_TEXT, language=None, duration_s=600.0)   # Parakeet: no language
    hist.append(blk)
    bank = VoiceBank(root=tmp_path)
    assert bank.bank(good_audio(8.0), SR, block_id=blk.id, **bank_kwargs(ts=blk.ts, text=ES_TEXT))
    cfg = Config()
    cfg.voice_bank = True
    pipeline = types.SimpleNamespace(history=hist, voicebank=bank, before_record=None)
    app = types.SimpleNamespace(pipeline=pipeline, cfg=cfg, events=None)
    return Backend(app)


def test_api_voicebank_view_through_the_backend(tmp_path, monkeypatch):
    """The API was only asserted by route strings. Parakeet history (no language) must
    still give a pace, and the takes carry their words."""
    b = _backend(tmp_path, monkeypatch)
    v = b.voicebank()
    es = v["langs"]["es"]
    assert v["on"] and es["usable_takes"] == 1 and es["usable_min_per_day"] > 0 and es["eta_days"] is not None
    assert v["takes"][0]["text"] == ES_TEXT[:120] and v["export"] == {}


def test_api_export_job_runs_and_each_run_has_its_own_id(tmp_path, monkeypatch):
    import subprocess

    from dictdact import output

    b = _backend(tmp_path, monkeypatch)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: None)
    monkeypatch.setattr(output, "notify", lambda *a, **kw: None)
    monkeypatch.setattr(vb, "export", lambda lang: (_ for _ in ()).throw(RuntimeError("boom")))
    ids = []
    for _ in range(2):
        job = b.voicebank_export("es")
        assert job["running"]
        for _ in range(200):
            if not b.voicebank()["export"].get("running"):
                break
            time.sleep(0.01)
        done = b.voicebank()["export"]
        assert done["error"] == "boom" and done["id"] == job["id"]
        ids.append(done["id"])
        time.sleep(0.002)
    assert ids[0] != ids[1]                      # the same failure twice is two jobs → two toasts
    with pytest.raises(ValueError):
        b.voicebank_export("fr")


def test_progress_counts_one_mic_like_export(tmp_path):
    """Progress summed every mic while export uses one, so a language could look
    PVC-ready with half the audio exportable."""
    bank = VoiceBank(root=tmp_path)
    for i in range(2):
        assert bank.bank(good_audio(8.0, seed=i), SR, block_id=f"a{i}",
                         **bank_kwargs(device="USB Studio Mic", ts=f"2026-09-23T10:0{i}:00"))
    assert bank.bank(good_audio(8.0, seed=9), SR, block_id="b0",
                     **bank_kwargs(device="MacBook Pro Microphone", ts="2026-09-23T10:05:00"))
    one = [e for e in vb.load_index(tmp_path) if e["device"] == "USB Studio Mic"]
    want = sum(e["voiced_s"] for e in one) / 60
    st = vb.summary(tmp_path)["langs"]["es"]
    assert abs(st["usable_min"] - want) < 1e-6 and st["device"] == "USB Studio Mic"
    assert st["reasons"].get("another mic") == 1
    assert abs(bank.usable_minutes()["es"] - want) < 1e-6
    assert abs(VoiceBank(root=tmp_path).usable_minutes()["es"] - want) < 1e-6
