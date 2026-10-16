"""The DictDact window: a native NSWindow hosting the local web dashboard in a WKWebView."""
from __future__ import annotations

import logging
import os

import objc
from AppKit import (
    NSAppearance, NSApp, NSBackingStoreBuffered, NSEvent, NSMakeRect, NSMakeSize, NSScreen, NSWindow,
    NSWindowStyleMaskClosable, NSWindowStyleMaskMiniaturizable, NSWindowStyleMaskResizable,
    NSWindowStyleMaskTitled, NSWorkspace, NSApplicationActivateIgnoringOtherApps,
    NSWindowCollectionBehaviorFullScreenPrimary, NSWindowCollectionBehaviorMoveToActiveSpace,
)
from Foundation import NSObject, NSURL, NSURLRequest
from WebKit import WKWebView, WKWebViewConfiguration

log = logging.getLogger("dictdact.window")


class MainWindow(NSObject):
    def initWithURL_(self, url: str):
        self = objc.super(MainWindow, self).init()
        if self is None:
            return None
        self.url = url
        self.prev_app = None
        style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable
                 | NSWindowStyleMaskMiniaturizable)
        w = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 980, 680), style, NSBackingStoreBuffered, False)
        w.setTitle_("DictDact")
        w.setMinSize_(NSMakeSize(720, 480))
        w.setReleasedWhenClosed_(False)
        w.setFrameAutosaveName_("DictDactMain")
        w.setAppearance_(NSAppearance.appearanceNamed_("NSAppearanceNameDarkAqua"))
        w.setTitlebarAppearsTransparent_(True)
        # Without this the window stays on the Space it was created on — which, after an external
        # display is unplugged, is a Space you can no longer reach: the window "opens" invisibly.
        w.setCollectionBehavior_(NSWindowCollectionBehaviorMoveToActiveSpace | NSWindowCollectionBehaviorFullScreenPrimary)
        w.setDelegate_(self)
        w.center()
        cfg = WKWebViewConfiguration.alloc().init()
        self.webview = WKWebView.alloc().initWithFrame_configuration_(w.contentView().bounds(), cfg)
        self.webview.setAutoresizingMask_(18)  # width | height sizable
        try:
            self.webview.setValue_forKey_(False, "drawsBackground")
        except Exception:
            pass
        w.contentView().addSubview_(self.webview)
        self.window = w
        self._loaded = False
        return self

    # -- NSWindowDelegate: close = hide ------------------------------------------------------
    def windowShouldClose_(self, sender):
        self.window.orderOut_(None)
        return False

    @objc.python_method
    def _screen_under_mouse(self):
        p = NSEvent.mouseLocation()
        for s in NSScreen.screens():
            f = s.frame()
            if f.origin.x <= p.x <= f.origin.x + f.size.width and f.origin.y <= p.y <= f.origin.y + f.size.height:
                return s
        return NSScreen.mainScreen() or (NSScreen.screens()[0] if NSScreen.screens() else None)

    @objc.python_method
    def ensure_on_screen(self) -> None:
        """A remembered position can point at a display that is gone, or leave the window hanging
        off the edge. Keep the size, but always place the whole window inside a live screen."""
        f = self.window.frame()
        area = max(1.0, f.size.width * f.size.height)
        best, best_screen = 0.0, None
        for s in NSScreen.screens():
            v = s.visibleFrame()
            w = max(0.0, min(f.origin.x + f.size.width, v.origin.x + v.size.width) - max(f.origin.x, v.origin.x))
            h = max(0.0, min(f.origin.y + f.size.height, v.origin.y + v.size.height) - max(f.origin.y, v.origin.y))
            if w * h > best:
                best, best_screen = w * h, s
        if best / area > 0.999 and f.size.width >= 900 and f.size.height >= 620:
            return                                   # already fully visible at a usable size
        screen = best_screen if best > 0 else self._screen_under_mouse()
        if screen is None:
            return
        v = screen.visibleFrame()
        # A restored frame can be tiny (or shrunk to the minimum by an earlier clamp) — give the
        # dashboard a comfortable size whenever it comes back smaller than it is usable at.
        width = min(max(f.size.width, 1000.0), v.size.width - 40)
        height = min(max(f.size.height, 700.0), v.size.height - 40)
        if best / area < 0.5:                        # mostly off-screen: start fresh in the middle
            x = v.origin.x + (v.size.width - width) / 2
            y = v.origin.y + (v.size.height - height) / 2
        else:                                        # nudge it back inside, keeping where it was
            x = min(max(f.origin.x, v.origin.x + 20), v.origin.x + v.size.width - width - 20)
            y = min(max(f.origin.y, v.origin.y + 20), v.origin.y + v.size.height - height - 20)
        log.info("window was not fully on screen — moving it into view")
        self.window.setFrame_display_(NSMakeRect(x, y, width, height), True)

    @objc.python_method
    def show(self, path: str = "") -> None:
        front = NSWorkspace.sharedWorkspace().frontmostApplication()
        if front is not None and int(front.processIdentifier()) != os.getpid():
            self.prev_app = front
        if not self._loaded:
            self.webview.loadRequest_(NSURLRequest.requestWithURL_(NSURL.URLWithString_(self.url + path)))
            self._loaded = True
        elif path:
            self.webview.evaluateJavaScript_completionHandler_(f"window.dictdactNavigate && window.dictdactNavigate({path!r})", None)
        self.ensure_on_screen()
        self.window.makeKeyAndOrderFront_(None)
        self.window.setIsVisible_(True)
        NSApp.activateIgnoringOtherApps_(True)

    @objc.python_method
    def hide(self) -> None:
        self.window.orderOut_(None)

    @objc.python_method
    def is_visible(self) -> bool:
        return bool(self.window.isVisible())

    @objc.python_method
    def return_to_previous_app(self) -> bool:
        """Hide and re-front the app that was active before the window opened. False when
        there is none to return to (the window was opened while DictDact was frontmost) —
        the caller then keeps its own targeting."""
        self.hide()
        if self.prev_app is not None:
            self.prev_app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
            return True
        return False
