"""First-run window: download the speech models with a progress bar and walk the user
through the two macOS permissions. Main thread only."""
from __future__ import annotations

import logging
import subprocess

import objc
from AppKit import (
    NSApp, NSBackingStoreBuffered, NSButton, NSColor, NSFont, NSMakeRect, NSProgressIndicator,
    NSProgressIndicatorStyleBar, NSTextField, NSWindow, NSWindowStyleMaskClosable, NSWindowStyleMaskTitled,
)
from Foundation import NSObject

from . import perms
from .models import Download

log = logging.getLogger("dictdact.setup")

ACCESSIBILITY_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
MIC_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"


def _label(text, x, y, w, h=20, size=13, bold=False, color=None):
    f = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
    f.setStringValue_(text)
    f.setBezeled_(False)
    f.setDrawsBackground_(False)
    f.setEditable_(False)
    f.setSelectable_(False)
    f.setFont_(NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size))
    if color is not None:
        f.setTextColor_(color)
    return f


class SetupWindow(NSObject):
    def initWithRepos_onReady_(self, repos, on_ready):
        self = objc.super(SetupWindow, self).init()
        if self is None:
            return None
        self.on_ready = on_ready
        self.download = Download(list(repos)) if repos else None
        self.models_ok = not repos
        self._ready_fired = False

        W, H = 520, 330
        w = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, W, H), NSWindowStyleMaskTitled | NSWindowStyleMaskClosable, NSBackingStoreBuffered, False)
        w.setTitle_("Welcome to DictDact")
        w.setReleasedWhenClosed_(False)
        w.center()
        c = w.contentView()
        c.addSubview_(_label("DictDact — local dictation", 24, H - 44, W - 48, 24, size=18, bold=True))
        c.addSubview_(_label("Hold Ctrl, talk (or whisper) in Spanish or English, release — clean text appears "
                             "where your cursor is. Everything runs on this Mac.", 24, H - 92, W - 48, 40, size=12,
                             color=NSColor.secondaryLabelColor()))

        y = H - 130
        self.model_status = _label("", 24, y, W - 48, 20)
        c.addSubview_(_label("1. Speech models", 24, y + 20, 200, 20, bold=True))
        c.addSubview_(self.model_status)
        self.bar = NSProgressIndicator.alloc().initWithFrame_(NSMakeRect(24, y - 18, W - 48, 16))
        self.bar.setStyle_(NSProgressIndicatorStyleBar)
        self.bar.setIndeterminate_(False)
        self.bar.setMinValue_(0.0)
        self.bar.setMaxValue_(1.0)
        c.addSubview_(self.bar)

        y -= 66
        c.addSubview_(_label("2. Microphone", 24, y + 20, 200, 20, bold=True))
        self.mic_status = _label("", 24, y, W - 200, 20)
        c.addSubview_(self.mic_status)
        self.mic_button = NSButton.alloc().initWithFrame_(NSMakeRect(W - 184, y - 2, 160, 28))
        self.mic_button.setTitle_("Open Microphone settings")
        self.mic_button.setTarget_(self)
        self.mic_button.setAction_("openMic:")
        c.addSubview_(self.mic_button)

        y -= 58
        c.addSubview_(_label("3. Accessibility (to see the Ctrl key and paste)", 24, y + 20, 360, 20, bold=True))
        self.ax_status = _label("", 24, y, W - 200, 20)
        c.addSubview_(self.ax_status)
        self.ax_button = NSButton.alloc().initWithFrame_(NSMakeRect(W - 184, y - 2, 160, 28))
        self.ax_button.setTitle_("Open Accessibility settings")
        self.ax_button.setTarget_(self)
        self.ax_button.setAction_("openAccessibility:")
        c.addSubview_(self.ax_button)

        self.footer = _label("", 24, 16, W - 48, 20, size=12, color=NSColor.secondaryLabelColor())
        c.addSubview_(self.footer)
        self.window = w
        return self

    # -- actions ----------------------------------------------------------------------
    def openMic_(self, sender):
        perms.request_microphone()
        subprocess.Popen(["open", MIC_URL])

    def openAccessibility_(self, sender):
        perms.accessibility(prompt=True)
        subprocess.Popen(["open", ACCESSIBILITY_URL])

    # -- lifecycle --------------------------------------------------------------------
    @objc.python_method
    def show(self):
        if self.download is not None:
            self.download.start()
        perms.request_microphone()
        self.refresh()
        self.window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)

    @objc.python_method
    def refresh(self) -> bool:
        """Update the rows; returns True once everything is ready."""
        d = self.download
        if d is None or d.finished:
            self.models_ok = True
            self.bar.setDoubleValue_(1.0)
            self.model_status.setStringValue_("✅ Whisper + formatter ready")
        elif d.error:
            self.model_status.setStringValue_(f"❌ Download failed: {d.error[:70]} — check your internet and reopen DictDact")
            self.model_status.setTextColor_(NSColor.systemRedColor())
        else:
            gb = d.total_bytes / 1e9 if d.total_bytes else 0
            self.bar.setDoubleValue_(d.fraction)
            self.model_status.setStringValue_(
                f"⬇️ Downloading {d.current or '…'}  {d.done_bytes / 1e9:.1f} / {gb:.1f} GB (one-time)")

        mic = perms.microphone()
        mic_ok = mic in ("granted", "unknown")
        self.mic_status.setStringValue_("✅ Allowed" if mic_ok else ("❌ Denied — enable DictDact" if mic == "denied"
                                                                 else "⏳ Allow when macOS asks"))
        self.mic_button.setHidden_(mic_ok)

        ax = perms.accessibility()
        self.ax_status.setStringValue_("✅ Granted" if ax else "⏳ Turn on DictDact in the list, then come back")
        self.ax_button.setHidden_(bool(ax))

        ready = self.models_ok and bool(ax) and mic_ok
        self.footer.setStringValue_("All set — hold Ctrl and talk. Closing this window…" if ready
                                    else "DictDact will start listening as soon as the steps above are done.")
        if ready and not self._ready_fired:
            self._ready_fired = True
            self.on_ready()
        return ready

    @objc.python_method
    def close(self):
        self.window.orderOut_(None)
