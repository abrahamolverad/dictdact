"""Global hotkeys: hold-to-talk on a modifier key, double-tap to lock, a combo to re-paste."""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from pynput import keyboard, mouse
from pynput.keyboard import Key

log = logging.getLogger("dictdact.hotkey")

_CTRL = {Key.ctrl, Key.ctrl_l, Key.ctrl_r}
_ALT = {Key.alt, Key.alt_l, Key.alt_r, Key.alt_gr}
_CMD = {Key.cmd, Key.cmd_l, Key.cmd_r}
_SHIFT = {Key.shift, Key.shift_l, Key.shift_r}
_FAMILY = {**{k: "ctrl" for k in _CTRL}, **{k: "alt" for k in _ALT}, **{k: "cmd" for k in _CMD}, **{k: "shift" for k in _SHIFT}}

HOTKEYS = {
    "ctrl": _CTRL,
    "ctrl_l": {Key.ctrl, Key.ctrl_l},
    "ctrl_r": {Key.ctrl_r},
    "alt": _ALT,
    "alt_l": {Key.alt, Key.alt_l},
    "alt_r": {Key.alt_r},
    "cmd_r": {Key.cmd_r},
    "shift_r": {Key.shift_r},
}

# macOS ANSI virtual key codes, so the retrieve shortcut works even when ⌥ changes the char (v → √).
_VK = {c: i for i, c in enumerate("asdfhgzxcv")}
_VK.update({"b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "o": 31, "u": 32, "i": 34,
            "p": 35, "l": 37, "j": 38, "k": 40, "n": 45, "m": 46})


def parse_combo(spec: str) -> tuple[set[str], str]:
    parts = [p.strip().lower() for p in spec.replace("+", " ").split()]
    mods = {p for p in parts if p in ("ctrl", "alt", "cmd", "shift")}
    letters = [p for p in parts if p not in mods]
    if len(letters) != 1 or len(letters[0]) != 1:
        raise ValueError(f"bad shortcut {spec!r}; expected e.g. 'ctrl+alt+v'")
    if letters[0] not in _VK:
        raise ValueError(f"unsupported key {letters[0]!r} in {spec!r}; usable letters: "
                         f"{''.join(sorted(_VK))}")
    if not mods:
        raise ValueError(f"{spec!r} has no modifier; a bare letter would fire while you type")
    return mods, letters[0]


DEFAULT_RETRIEVE_COMBO = "ctrl+alt+v"

_PRETTY_ORDER = ("ctrl", "alt", "shift", "cmd")


def pretty_combo(spec: str) -> str:
    """'ctrl+alt+s' → 'Ctrl+Alt+S', in a stable order — one spelling for the menu, the
    guide and the log, so a shortcut the user changed is never advertised as the old one."""
    mods, letter = parse_combo(spec)
    return "+".join([m.capitalize() for m in _PRETTY_ORDER if m in mods] + [letter.upper()])


class Hotkeys:
    def __init__(
        self,
        hotkey: str,
        hold_threshold_ms: int,
        double_tap_lock: bool,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_cancel: Callable[[], None],
        on_lock: Callable[[], None],
        on_retrieve: Callable[[], None],
        retrieve_combo: str = DEFAULT_RETRIEVE_COMBO,
        double_tap_ms: int = 400,
        combo_window_s: float = 1.5,
        extra_combos: dict[str, Callable[[], None]] | None = None,
    ):
        if hotkey not in HOTKEYS:
            raise SystemExit(f"unknown hotkey {hotkey!r}; choose from {', '.join(HOTKEYS)}")
        self.keys = HOTKEYS[hotkey]
        self.hold_threshold = hold_threshold_ms / 1000.0
        self.double_tap = double_tap_ms / 1000.0
        self.double_tap_lock = double_tap_lock
        self.combo_window = combo_window_s
        self.on_start, self.on_stop, self.on_cancel, self.on_lock, self.on_retrieve = (
            on_start, on_stop, on_cancel, on_lock, on_retrieve)
        self.retrieve_mods, self.retrieve_letter = parse_combo(retrieve_combo)
        self.combos: list[tuple[set[str], str, Callable[[], None]]] = [
            (self.retrieve_mods, self.retrieve_letter, on_retrieve)]
        for spec, cb in (extra_combos or {}).items():
            mods, letter = parse_combo(spec)
            self.combos.append((mods, letter, cb))
        self._mods: set[str] = set()
        self._down = False
        self._t0 = 0.0
        self._combo = False
        self._last_tap = 0.0
        self.recording = False
        self.locked = False
        self._lock = threading.Lock()
        self._listener: keyboard.Listener | None = None
        self._mouse: mouse.Listener | None = None

    # -- helpers ------------------------------------------------------------
    def _is_letter(self, key, letter: str) -> bool:
        vk = getattr(key, "vk", None)
        if vk is not None and _VK.get(letter) == vk:
            return True
        ch = getattr(key, "char", None)
        return bool(ch) and ch.lower() == letter

    def _safe(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:
            log.exception("hotkey callback failed")

    # -- pynput callbacks ----------------------------------------------------
    def _on_press(self, key) -> None:
        with self._lock:
            fam = _FAMILY.get(key)
            if fam:
                self._mods.add(fam)
            if key in self.keys:
                if self._down:
                    return
                self._down, self._t0, self._combo = True, time.monotonic(), False
                if self.locked or self.recording:
                    return
                self.recording = True
                self._safe(self.on_start)
                return
            # Some other key while the hotkey is held → probably a shortcut, not dictation.
            if self._down:
                self._combo_hit(f"key {key}")
            for mods, letter, cb in self.combos:
                if mods and mods == self._mods and self._is_letter(key, letter):
                    self._safe(cb)
                    break

    def _combo_hit(self, what: str) -> None:
        """Another key/click while the hotkey is held. Early = a shortcut (discard);
        once a take is clearly under way, keep recording — losing a dictation is worse."""
        held = time.monotonic() - self._t0
        if held > self.combo_window:
            log.debug("%s after %.1fs of recording — ignoring (dictation continues)", what, held)
            return
        self._combo = True
        if self.recording and not self.locked:
            self.recording = False
            log.info("cancelled %.2fs recording: %s while holding the hotkey", held, what)
            self._safe(self.on_cancel)

    def _on_click(self, x, y, button, pressed) -> None:
        if not pressed:
            return
        with self._lock:
            if self._down:
                self._combo_hit("mouse click")

    def _on_release(self, key) -> None:
        with self._lock:
            fam = _FAMILY.get(key)
            if fam:
                self._mods.discard(fam)
            if key not in self.keys or not self._down:
                return
            self._down = False
            now = time.monotonic()
            held = now - self._t0
            if self._combo:
                return  # shortcut combo; recording was already cancelled (or locked one continues)
            if self.locked:
                # A clean tap while locked ends the hands-free recording.
                self.locked = False
                self.recording = False
                self._safe(self.on_stop)
                return
            if held < self.hold_threshold:
                if self.double_tap_lock and (now - self._last_tap) < self.double_tap and self.recording:
                    self.locked = True
                    self._last_tap = 0.0
                    self._safe(self.on_lock)
                else:
                    self._last_tap = now
                    if self.recording:
                        self.recording = False
                        log.debug("tap (%.0f ms) — not a dictation", held * 1000)
                        self._safe(self.on_cancel)
                return
            if self.recording:
                self.recording = False
                self._safe(self.on_stop)

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()
        self._mouse = mouse.Listener(on_click=self._on_click)
        self._mouse.daemon = True
        self._mouse.start()
        time.sleep(0.3)
        if not self._listener.is_alive():
            raise RuntimeError("keyboard listener died — Input Monitoring permission missing?")
        log.info("listening for hotkey (%s) and %s", ", ".join(sorted(str(k) for k in self.keys)),
                 ", ".join("+".join(sorted(m)) + "+" + l for m, l, _ in self.combos))

    def restart(self) -> None:
        """Re-create the listeners (needed after Input Monitoring is granted at runtime)."""
        self.stop()
        self._down = self._combo = False
        self.start()

    def force_stop_recording(self) -> None:
        """Used when the recording hits the time cap."""
        with self._lock:
            if self.recording:
                self.recording = False
                self.locked = False
                self._safe(self.on_stop)

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
        if self._mouse:
            self._mouse.stop()

    def join(self) -> None:
        if self._listener:
            self._listener.join()
