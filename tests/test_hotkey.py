import time

import pytest
from pynput.keyboard import Key, KeyCode

from dictdact import hotkey as hk


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def rig(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(hk.time, "monotonic", clock)
    events = []
    h = hk.Hotkeys(
        "ctrl", hold_threshold_ms=180, double_tap_lock=True,
        on_start=lambda: events.append("start"), on_stop=lambda: events.append("stop"),
        on_cancel=lambda: events.append("cancel"), on_lock=lambda: events.append("lock"),
        on_retrieve=lambda: events.append("retrieve"),
    )
    return h, events, clock


def press(h, key):
    h._on_press(key)


def release(h, key):
    h._on_release(key)


def test_hold_and_release_dictates(rig):
    h, events, clock = rig
    press(h, Key.ctrl)
    clock.t += 1.5
    release(h, Key.ctrl)
    assert events == ["start", "stop"]


def test_short_tap_is_cancelled(rig):
    h, events, clock = rig
    press(h, Key.ctrl)
    clock.t += 0.05
    release(h, Key.ctrl)
    assert events == ["start", "cancel"]


def test_ctrl_c_is_a_shortcut_not_dictation(rig):
    h, events, clock = rig
    press(h, Key.ctrl)
    clock.t += 0.3
    press(h, KeyCode.from_char("c"))
    release(h, KeyCode.from_char("c"))
    clock.t += 0.3
    release(h, Key.ctrl)
    assert events == ["start", "cancel"]


def test_double_tap_locks_then_tap_stops(rig):
    h, events, clock = rig
    press(h, Key.ctrl); clock.t += 0.05; release(h, Key.ctrl)       # tap 1
    clock.t += 0.2
    press(h, Key.ctrl); clock.t += 0.05; release(h, Key.ctrl)       # tap 2 → lock
    assert events == ["start", "cancel", "start", "lock"]
    assert h.locked and h.recording
    clock.t += 10
    press(h, KeyCode.from_char("x")); release(h, KeyCode.from_char("x"))  # typing while locked is fine
    assert h.recording
    press(h, Key.ctrl); clock.t += 0.05; release(h, Key.ctrl)       # tap → stop
    assert events[-1] == "stop" and not h.locked and not h.recording


def test_shortcut_while_locked_does_not_stop(rig):
    h, events, clock = rig
    press(h, Key.ctrl); clock.t += 0.05; release(h, Key.ctrl)
    clock.t += 0.2
    press(h, Key.ctrl); clock.t += 0.05; release(h, Key.ctrl)
    assert h.locked
    press(h, Key.ctrl); press(h, KeyCode.from_char("c")); release(h, KeyCode.from_char("c")); release(h, Key.ctrl)
    assert h.locked and h.recording and events[-1] == "lock"


def test_ctrl_alt_v_retrieves_and_cancels_recording(rig):
    h, events, clock = rig
    press(h, Key.ctrl)
    press(h, Key.alt)
    press(h, KeyCode(vk=9))  # macOS virtual key for V (char would be √ with ⌥ held)
    assert events == ["start", "cancel", "retrieve"]
    release(h, KeyCode(vk=9)); release(h, Key.alt); release(h, Key.ctrl)
    assert events == ["start", "cancel", "retrieve"]


def test_cmd_ctrl_alt_v_is_not_retrieve(rig):
    h, events, clock = rig
    press(h, Key.cmd); press(h, Key.ctrl); press(h, Key.alt); press(h, KeyCode(vk=9))
    assert "retrieve" not in events


def test_mouse_click_while_holding_cancels(rig):
    h, events, clock = rig
    press(h, Key.ctrl)
    clock.t += 0.3
    h._on_click(0, 0, None, True)
    clock.t += 0.3
    release(h, Key.ctrl)
    assert events == ["start", "cancel"]


def test_time_cap_forces_stop(rig):
    h, events, clock = rig
    press(h, Key.ctrl)
    h.force_stop_recording()
    assert events == ["start", "stop"]
    clock.t += 1
    release(h, Key.ctrl)
    assert events == ["start", "stop"]


def test_right_ctrl_only_config():
    h = hk.Hotkeys("ctrl_r", 180, True, *(lambda: None for _ in range(5)))
    assert Key.ctrl_r in h.keys and Key.ctrl not in h.keys


def test_parse_combo():
    assert hk.parse_combo("ctrl+alt+v") == ({"ctrl", "alt"}, "v")
    with pytest.raises(ValueError):
        hk.parse_combo("ctrl+alt")


def test_key_after_combo_window_does_not_cancel(rig):
    h, events, clock = rig
    press(h, Key.ctrl)
    clock.t += 3.0
    press(h, KeyCode.from_char("a")); release(h, KeyCode.from_char("a"))
    h._on_click(0, 0, None, True)
    assert events == ["start"] and h.recording
    clock.t += 2.0
    release(h, Key.ctrl)
    assert events == ["start", "stop"]


def test_extra_combo(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(hk.time, "monotonic", clock)
    hits = []
    h = hk.Hotkeys("ctrl", 180, True, *(lambda: None for _ in range(5)), extra_combos={"ctrl+alt+h": lambda: hits.append("h")})
    press(h, Key.ctrl); press(h, Key.alt); press(h, KeyCode(vk=4))  # H
    assert hits == ["h"]


def test_force_stop_used_before_hotkey_swap():
    # (documents the contract used by the app: force_stop_recording ends an active take)
    h = hk.Hotkeys("ctrl", 180, True, *(lambda: None for _ in range(5)))
    h.recording = True
    ended = []
    h.on_stop = lambda: ended.append(1)
    h.force_stop_recording()
    assert ended == [1] and h.recording is False
