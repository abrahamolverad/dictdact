"""Where should a paste land? Resolving the app under the mouse pointer.

`paste_target = "mouse"` pastes into the app the user is *pointing at*, not whichever app
happens to hold keyboard focus — dictating while looking at one window and pasting into
another is the failure this fixes. Strictly app-level: the target app is activated (never
clicked) and macOS restores its focused window/field. Raising a specific window of a
multi-window app is deliberately out of scope.

Coordinate note: the mouse position comes from CGEventGetLocation and window frames from
CGWindowList — both in the same Quartz space (origin top-left of the primary display,
y grows downward), so points compare directly. NSEvent.mouseLocation would need a flip.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

log = logging.getLogger("dictdact.target")


@dataclass(frozen=True)
class Window:
    pid: int
    app: str
    layer: int
    alpha: float
    x: float
    y: float
    w: float
    h: float

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h


def mouse_point() -> tuple[float, float]:
    from Quartz import CGEventCreate, CGEventGetLocation

    loc = CGEventGetLocation(CGEventCreate(None))
    return float(loc.x), float(loc.y)


def onscreen_windows() -> list[Window]:
    """Front-to-back windows, primitives only. Owner pid/name, bounds and layer are
    readable without the Screen Recording permission (only window *titles* are gated)."""
    from Quartz import (CGWindowListCopyWindowInfo, kCGNullWindowID,
                        kCGWindowListExcludeDesktopElements, kCGWindowListOptionOnScreenOnly)

    out: list[Window] = []
    for w in CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
                                        kCGNullWindowID) or []:
        b = w.get("kCGWindowBounds") or {}
        out.append(Window(
            pid=int(w.get("kCGWindowOwnerPID", 0)),
            app=str(w.get("kCGWindowOwnerName", "") or ""),
            layer=int(w.get("kCGWindowLayer", 0)),
            alpha=float(w.get("kCGWindowAlpha", 1.0)),
            x=float(b.get("X", 0)), y=float(b.get("Y", 0)),
            w=float(b.get("Width", 0)), h=float(b.get("Height", 0)),
        ))
    return out


def pick_paste_target(point: tuple[float, float], windows: list[Window], own_pid: int) -> Window | None:
    """The frontmost NORMAL window under ``point`` that isn't ours. None ⇒ the pointer is
    over the desktop, the menu bar, DictDact's own window, an overlay… — caller keeps the
    focus behavior. Pointing at OUR window must terminate the search (falling through
    would front whatever app happens to sit behind the dashboard)."""
    px, py = point
    for w in windows:                     # layer filter first; within layer 0 the list is front-to-back
        if w.layer != 0:                  # menu bar, Dock, overlays, status items
            continue
        if w.alpha <= 0.05 or w.w < 8 or w.h < 8:
            continue
        if w.contains(px, py):
            return None if w.pid == own_pid else w
    return None


def frontmost_pid() -> int | None:
    """menuBarOwningApplication, not frontmostApplication: with a notification banner on
    screen the latter reports transient helpers (UserNotificationCenter) that own no
    normal window at all — measured 8/8 wrong in testing."""
    try:
        from AppKit import NSWorkspace

        ws = NSWorkspace.sharedWorkspace()
        app = ws.menuBarOwningApplication() or ws.frontmostApplication()
        return int(app.processIdentifier()) if app else None
    except Exception:
        return None


def _ax_element(create, *args, timeout_s: float = 0.5):
    """Create an AX element and bound its messaging timeout: a wedged target app must
    cost us half a second, not the multi-second system default (the paste lock is held)."""
    from ApplicationServices import AXUIElementSetMessagingTimeout

    elem = create(*args)
    try:
        AXUIElementSetMessagingTimeout(elem, timeout_s)
    except Exception:
        pass
    return elem


def ax_pid_at_point(px: float, py: float) -> int | None:
    """The pid the system's own AX hit-test reaches at ``point``. CGWindowList cannot see
    hit-testability, so a click-through transparent overlay (screen-recording HUDs…) or a
    save-panel XPC sheet wins the geometric pick; the AX road resolves to the app a user
    interaction would actually reach. None on any doubt."""
    try:
        from ApplicationServices import (AXUIElementCopyElementAtPosition, AXUIElementCreateSystemWide,
                                         AXUIElementGetPid)

        sys_elem = _ax_element(AXUIElementCreateSystemWide)
        err, elem = AXUIElementCopyElementAtPosition(sys_elem, px, py, None)
        if err or elem is None:
            return None
        err, pid = AXUIElementGetPid(elem, None)
        return int(pid) if not err else None
    except Exception:
        return None


def _activate_now(app) -> None:
    from AppKit import NSApplicationActivateIgnoringOtherApps

    app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)


def _ax_front(pid: int) -> bool:
    """Front via Accessibility (AXFrontmost = true) — thread-agnostic and uses the trust
    the app already holds; the road window managers take."""
    try:
        from ApplicationServices import AXUIElementCreateApplication, AXUIElementSetAttributeValue

        return AXUIElementSetAttributeValue(_ax_element(AXUIElementCreateApplication, pid), "AXFrontmost", True) == 0
    except Exception:
        return False


def _front_is(pid: int, wait_s: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if frontmost_pid() == pid:
            return True
        time.sleep(0.05)
    return False


def activate_pid(pid: int, timeout_s: float = 1.2) -> bool:
    """Bring ``pid`` frontmost and wait until macOS agrees. Activation restores the app's
    own focused window and first responder — no synthetic clicks, ever.

    Inside an AppKit app the NSRunningApplication call only takes effect on the MAIN
    thread — from the paste lane it is silently ignored (verified on macOS 26). Dispatch
    there when this is the menu-bar app; the CLI paste path already runs on its main
    thread. The AX road is the belt for anything that still refuses."""
    try:
        from AppKit import NSRunningApplication

        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if app is None:
            return False
        if app.isActive():
            return True
        try:
            from Foundation import NSOperationQueue, NSThread

            if NSThread.isMainThread():
                _activate_now(app)
            else:
                NSOperationQueue.mainQueue().addOperationWithBlock_(lambda: _activate_now(app))
        except Exception:
            _activate_now(app)
        if _front_is(pid, timeout_s):
            return True
        if _ax_front(pid) and _front_is(pid, 0.6):
            log.debug("fronted pid %d via AX (activation call was refused)", pid)
            return True
        return False
    except Exception:
        log.exception("could not activate pid %d", pid)
        return False


def focused_is_secure(pid: int) -> bool:
    """True when ``pid``'s focused element is a secure (password) field. A dictation must
    not be ⌘V'd into a password box the user never focused themselves; any doubt ⇒ False
    (never block a normal paste on AX weirdness)."""
    try:
        from ApplicationServices import (AXUIElementCopyAttributeValue, AXUIElementCreateApplication)

        err, focused = AXUIElementCopyAttributeValue(_ax_element(AXUIElementCreateApplication, pid),
                                                     "AXFocusedUIElement", None)
        if err or focused is None:
            return False
        err, subrole = AXUIElementCopyAttributeValue(focused, "AXSubrole", None)
        return (not err) and str(subrole) == "AXSecureTextField"
    except Exception:
        return False
