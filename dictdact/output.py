"""Deliver text to the user: clipboard, paste into the frontmost app, sounds."""
from __future__ import annotations

import logging
import subprocess
import time

log = logging.getLogger("dictdact.output")

_SOUNDS = {
    "start": "/System/Library/Sounds/Tink.aiff",
    "stop": "/System/Library/Sounds/Pop.aiff",
    "done": "/System/Library/Sounds/Blow.aiff",
    "error": "/System/Library/Sounds/Basso.aiff",
    "cancel": "/System/Library/Sounds/Morse.aiff",
}


def play(name: str, enabled: bool = True) -> None:
    if not enabled:
        return
    path = _SOUNDS.get(name)
    if path:
        try:
            subprocess.Popen(["afplay", "-v", "0.5", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass


def frontmost_app_name() -> str | None:
    try:
        from AppKit import NSWorkspace

        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        return str(app.localizedName()) if app else None
    except Exception:
        return None


def get_clipboard_text() -> str | None:
    from AppKit import NSPasteboard, NSPasteboardTypeString

    pb = NSPasteboard.generalPasteboard()
    s = pb.stringForType_(NSPasteboardTypeString)
    return str(s) if s is not None else None


def set_clipboard_text(text: str) -> None:
    from AppKit import NSPasteboard, NSPasteboardTypeString

    pb = NSPasteboard.generalPasteboard()
    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)


_VK_V, _VK_CMD = 9, 55


def _send_cmd_v() -> None:
    """Post ⌘V as raw Quartz events by virtual key code.

    pynput's Controller maps 'v' through the keyboard layout (TSM/HIToolbox), and macOS 26
    asserts that this happens on the main thread — it crashed DictDact from the worker thread
    (EXC_BREAKPOINT in TSMGetInputSourceProperty). CGEventPost is thread-safe and needs no lookup."""
    from Quartz import (CGEventCreateKeyboardEvent, CGEventPost, CGEventSetFlags, kCGEventFlagMaskCommand,
                        kCGHIDEventTap)

    def post(vk: int, down: bool, flags: int) -> None:
        ev = CGEventCreateKeyboardEvent(None, vk, down)
        CGEventSetFlags(ev, flags)
        CGEventPost(kCGHIDEventTap, ev)

    post(_VK_CMD, True, kCGEventFlagMaskCommand)
    post(_VK_V, True, kCGEventFlagMaskCommand)
    post(_VK_V, False, kCGEventFlagMaskCommand)
    post(_VK_CMD, False, 0)


MODIFIER_WAIT_S = 2.5


def wait_modifiers_released(timeout_s: float = MODIFIER_WAIT_S) -> bool:
    """Block until no modifier key is physically held (Ctrl/Alt/Shift/Cmd), or ``timeout_s``.

    Ctrl+Alt+V fires on the V press while Ctrl and Alt are still down; a ⌘V posted at that
    instant reaches the app merged with the live modifiers (Ctrl+Alt+⌘+V), which is not a
    paste in most apps. False = still held at the deadline: the caller must NOT send ⌘V
    (a merged keystroke is worse than no paste) — the text stays in the clipboard."""
    try:
        from Quartz import (CGEventSourceFlagsState, kCGEventFlagMaskAlternate, kCGEventFlagMaskCommand,
                            kCGEventFlagMaskControl, kCGEventFlagMaskShift, kCGEventSourceStateCombinedSessionState)
    except Exception:
        return True
    mask = kCGEventFlagMaskControl | kCGEventFlagMaskAlternate | kCGEventFlagMaskCommand | kCGEventFlagMaskShift
    deadline = time.monotonic() + timeout_s
    flags = 0
    while time.monotonic() < deadline:
        flags = CGEventSourceFlagsState(kCGEventSourceStateCombinedSessionState) & mask
        if not flags:
            return True
        time.sleep(0.01)
    log.warning("modifier keys still held after %.1fs (flags 0x%x) — text left in the clipboard, no ⌘V sent",
                timeout_s, flags)
    return False


def paste_text(text: str, restore_clipboard: bool = True) -> bool:
    """Paste ``text`` at the cursor of the frontmost app via the clipboard + ⌘V.
    Returns True if the keystroke was sent (macOS Accessibility permission required)."""
    from . import perms

    if perms.accessibility() is False:
        # ⌘V would be silently dropped by macOS; leave the text in the clipboard instead.
        log.error("Accessibility not granted — text copied to clipboard, not pasted")
        set_clipboard_text(text)
        return False
    previous = get_clipboard_text() if restore_clipboard else None
    set_clipboard_text(text)
    time.sleep(0.08)
    if not wait_modifiers_released():
        return False                      # the text is in the clipboard; ⌘V by hand once the keys are up
    try:
        _send_cmd_v()
    except Exception as e:
        log.error("could not send ⌘V: %s", e)
        return False
    if restore_clipboard and previous is not None:
        time.sleep(0.6)  # give the target app time to read the pasteboard
        set_clipboard_text(previous)
    return True


def notify(title: str, message: str) -> None:
    safe = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')
    try:
        subprocess.Popen(["osascript", "-e", f'display notification "{safe(message)}" with title "{safe(title)}"'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass
