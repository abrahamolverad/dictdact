"""Window picking for paste_target='mouse' — the pure logic, no Quartz needed."""
from dictdact.target import Window, pick_paste_target


def W(pid=1, app="A", layer=0, alpha=1.0, x=0, y=0, w=100, h=100):
    return Window(pid=pid, app=app, layer=layer, alpha=alpha, x=x, y=y, w=w, h=h)


def test_first_front_window_containing_the_point_wins():
    wins = [W(pid=1, app="Front"), W(pid=2, app="Back")]
    assert pick_paste_target((50, 50), wins, own_pid=999).app == "Front"


def test_point_outside_falls_through_to_the_window_below():
    wins = [W(pid=1, app="Front", x=200, y=200), W(pid=2, app="Back")]
    assert pick_paste_target((50, 50), wins, own_pid=999).app == "Back"


def test_menubar_dock_and_overlays_are_ignored():
    wins = [W(pid=1, app="MenuBar", layer=25), W(pid=2, app="App")]
    assert pick_paste_target((50, 50), wins, own_pid=999).app == "App"


def test_pointing_at_dictdacts_own_window_means_focus_behavior():
    # Falling through to the app BEHIND the dashboard would front something the user
    # is not pointing at — an own-pid hit must terminate the search instead.
    wins = [W(pid=7, app="DictDact"), W(pid=2, app="Behind")]
    assert pick_paste_target((50, 50), wins, own_pid=7) is None


def test_invisible_and_tiny_windows_are_ignored():
    wins = [W(pid=1, alpha=0.0), W(pid=2, w=4, h=4), W(pid=3, app="Real")]
    assert pick_paste_target((2, 2), wins, own_pid=9).app == "Real"


def test_desktop_or_no_hit_returns_none():
    assert pick_paste_target((50, 50), [W(x=200, y=200)], own_pid=9) is None


def test_edges_are_half_open():
    win = W(x=10, y=10, w=100, h=100)
    assert pick_paste_target((10, 10), [win], own_pid=9) is win        # top-left inclusive
    assert pick_paste_target((110, 110), [win], own_pid=9) is None     # bottom-right exclusive


def test_config_default_and_server_expose_the_knob():
    from dictdact.config import Config
    from dictdact.server import EDITABLE

    assert Config().paste_target == "focus"
    assert EDITABLE["paste_target"] == ["focus", "mouse"]


# -- the retarget decision itself (target module faked, no Quartz) --------------------
import queue
import threading

from dictdact.config import Config
from dictdact.pipeline import Pipeline


def _mouse_pipeline(monkeypatch, *, tgt, front_pid, activate=True, secure=False, ax_pid=None):
    import dictdact.pipeline as pl

    p = Pipeline.__new__(Pipeline)
    p.cfg = Config()
    p.cfg.paste_target = "mouse"
    p._paste_lock = threading.Lock()
    p._paste_spawn_lock = threading.Lock()
    p._repaste_inflight = False
    p._last_retarget_app = None
    p._jobs = queue.Queue()
    p.emit = lambda kind, payload=None: None
    calls = {"activated": []}
    monkeypatch.setattr(pl.target, "mouse_point", lambda: (10.0, 10.0))
    monkeypatch.setattr(pl.target, "onscreen_windows", lambda: [tgt] if tgt else [])
    monkeypatch.setattr(pl.target, "pick_paste_target", lambda pt, ws, own: tgt)
    monkeypatch.setattr(pl.target, "ax_pid_at_point", lambda x, y: ax_pid)
    monkeypatch.setattr(pl.target, "frontmost_pid", lambda: front_pid)
    monkeypatch.setattr(pl.target, "activate_pid", lambda pid, timeout_s=1.2: calls["activated"].append(pid) or activate)
    monkeypatch.setattr(pl.target, "focused_is_secure", lambda pid: secure)
    monkeypatch.setattr(pl.time, "sleep", lambda s: None)
    return p, calls


def test_secure_field_refused_even_when_already_frontmost(monkeypatch):
    tgt = W(pid=42, app="1Password")
    p, calls = _mouse_pipeline(monkeypatch, tgt=tgt, front_pid=42, secure=True)
    assert p._mouse_retarget_ok() is False       # the most obvious hole
    assert calls["activated"] == []              # no switch was needed, none attempted


def test_activation_failure_falls_back_to_focus_paste(monkeypatch):
    tgt = W(pid=42, app="TextEdit")
    p, calls = _mouse_pipeline(monkeypatch, tgt=tgt, front_pid=1, activate=False)
    assert p._mouse_retarget_ok() is True        # text is never lost
    assert calls["activated"] == [42]


def test_successful_switch_records_the_receiving_app(monkeypatch):
    tgt = W(pid=42, app="Notes")
    p, _ = _mouse_pipeline(monkeypatch, tgt=tgt, front_pid=1)
    assert p._mouse_retarget_ok() is True
    assert p._last_retarget_app == "Notes"


def test_ax_hit_test_overrides_a_transparent_overlay(monkeypatch):
    overlay = W(pid=42, app="HUD")
    real = W(pid=77, app="Chrome")
    import dictdact.pipeline as pl

    p, calls = _mouse_pipeline(monkeypatch, tgt=overlay, front_pid=1, ax_pid=77)
    monkeypatch.setattr(pl.target, "onscreen_windows", lambda: [overlay, real])
    assert p._mouse_retarget_ok() is True
    assert calls["activated"] == [77]            # the app the system would actually hit


def test_repaste_key_repeat_is_deduped(monkeypatch):
    import dictdact.pipeline as pl

    p = Pipeline.__new__(Pipeline)
    p._paste_spawn_lock = threading.Lock()
    p._repaste_inflight = False
    spawned = []
    monkeypatch.setattr(pl.threading, "Thread",
                        lambda target, name, daemon: spawned.append(name) or type("T", (), {"start": lambda s: None})())
    class B: text = "x"
    p.paste_block(B())
    p.paste_block(B())                            # key repeat: dropped, not stacked
    assert len(spawned) == 1
