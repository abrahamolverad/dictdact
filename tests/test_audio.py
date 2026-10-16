import numpy as np

from dictdact.audio import TARGET_SR, analyze_and_normalize


def speech_like(seconds=2.0, level_db=-12.0, seed=0):
    """Bursty band-limited noise with pauses: has the dynamics of speech."""
    rng = np.random.default_rng(seed)
    n = int(seconds * TARGET_SR)
    x = rng.normal(0, 1, n).astype(np.float32)
    env = (np.sin(np.linspace(0, seconds * 6 * np.pi, n)) > 0).astype(np.float32)
    x = x * env
    x /= np.abs(x).max() + 1e-9
    return x * 10 ** (level_db / 20)


def test_normal_speech_is_detected_and_kept_near_full_scale():
    y, st = analyze_and_normalize(speech_like())
    assert st.has_speech and st.mode == "normal"
    assert -4.5 < 20 * np.log10(np.abs(y).max()) < -1.5


def test_whisper_level_is_boosted_and_labelled():
    y, st = analyze_and_normalize(speech_like(level_db=-48), max_gain_db=40)
    assert st.has_speech and st.mode == "whisper"
    assert st.gain_db > 30
    assert 20 * np.log10(np.abs(y).max()) > -8


def test_gain_is_capped():
    _, st = analyze_and_normalize(speech_like(level_db=-70), max_gain_db=20)
    assert st.gain_db <= 20


def test_silence_and_flat_noise_are_rejected():
    rng = np.random.default_rng(1)
    _, st = analyze_and_normalize(rng.normal(0, 1e-3, 3 * TARGET_SR).astype(np.float32))
    assert not st.has_speech and st.mode == "silence"
    _, st = analyze_and_normalize(np.zeros(TARGET_SR, dtype=np.float32))
    assert not st.has_speech


def test_too_short_is_rejected():
    _, st = analyze_and_normalize(speech_like(seconds=0.2))
    assert not st.has_speech


def test_leading_and_trailing_silence_are_trimmed():
    x = np.concatenate([np.zeros(3 * TARGET_SR, np.float32), speech_like(1.0), np.zeros(3 * TARGET_SR, np.float32)])
    y, st = analyze_and_normalize(x)
    assert st.has_speech
    assert len(y) < 2.2 * TARGET_SR


def test_fallback_prefers_external_then_builtin(monkeypatch):
    import dictdact.audio as au
    D = au.DeviceInfo
    devs = [D(0, "iPhone Microphone", 1, 48000, False), D(1, "MacBook Pro Microphone", 1, 48000, True)]
    monkeypatch.setattr(au, "list_input_devices", lambda: devs)
    monkeypatch.setattr(au, "lid_closed", lambda: False)
    d = au.resolve_input_device("USB Studio Mic")
    assert d.name == "MacBook Pro Microphone" and "not connected" in d.fallback_reason
    monkeypatch.setattr(au, "lid_closed", lambda: True)
    assert au.resolve_input_device("USB Studio Mic").name == "iPhone Microphone"
    devs.append(D(5, "USB Studio Mic", 1, 48000, False))
    d = au.resolve_input_device("USB Studio Mic")
    assert d.name == "USB Studio Mic" and d.fallback_reason is None
