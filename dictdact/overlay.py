"""Floating pill at the bottom of the screen showing a live waveform of your voice
(Wispr-Flow style), then the transcribing / structuring / pasted states."""
from __future__ import annotations

import math
from collections import deque

import objc
from AppKit import (
    NSBackingStoreBuffered, NSBezierPath, NSColor, NSEvent, NSFont, NSFontAttributeName, NSFontWeightMedium,
    NSForegroundColorAttributeName, NSMakePoint, NSMakeRect, NSPanel, NSScreen, NSScreenSaverWindowLevel, NSString,
    NSView, NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorIgnoresCycle, NSWindowCollectionBehaviorStationary, NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)

NBARS = 34
BAR_W, BAR_GAP = 3.0, 2.5
HEIGHT = 44.0
WAVE_WIDTH = 36 + NBARS * (BAR_W + BAR_GAP) + 14 + 96   # dot + bars + label area
MAX_WIDTH = 560.0

_COLORS = {
    "recording": (0.96, 0.26, 0.21),   # red
    "locked": (0.98, 0.66, 0.15),      # amber
    "processing": (0.35, 0.62, 1.0),   # blue
    "done": (0.30, 0.85, 0.45),        # green
    "error": (1.0, 0.45, 0.25),        # orange
    "nosignal": (1.0, 0.45, 0.25),
}


def level_to_norm(rms: float) -> float:
    """Map an RMS level to a 0..1 bar height: -62 dBFS → 0, -18 dBFS → 1 (whispers stay visible)."""
    db = 20 * math.log10(rms + 1e-9)
    x = min(1.0, max(0.0, (db + 62.0) / 44.0))
    return x ** 0.7


class WaveView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(WaveView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.bars = deque([0.0] * NBARS, maxlen=NBARS)
        self.state = "recording"
        self.label = "Listening"
        self.phase = 0.0
        return self

    def isOpaque(self):
        return False

    def _text_attrs(self, size, rgb, alpha=1.0):
        return {
            NSFontAttributeName: NSFont.systemFontOfSize_weight_(size, NSFontWeightMedium),
            NSForegroundColorAttributeName: NSColor.colorWithCalibratedRed_green_blue_alpha_(*rgb, alpha),
        }

    def drawRect_(self, rect):
        b = self.bounds()
        w, h = b.size.width, b.size.height
        pill = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(0, 0, w, h), h / 2, h / 2)
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.08, 0.08, 0.10, 0.94).setFill()
        pill.fill()
        NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.10).setStroke()
        pill.setLineWidth_(1.0)
        pill.stroke()

        r, g, bl = _COLORS.get(self.state, _COLORS["recording"])
        pulse = 0.65 + 0.35 * math.sin(self.phase) if self.state in ("recording", "locked", "processing") else 1.0
        NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, bl, pulse).setFill()
        NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(16, h / 2 - 5, 10, 10)).fill()

        x = 36.0
        if self.state in ("recording", "locked", "nosignal"):
            max_h = h - 16
            n = len(self.bars)
            for i, v in enumerate(self.bars):
                bh = max(2.0, v * max_h)
                alpha = 0.35 + 0.65 * (i + 1) / n
                if self.state == "nosignal":
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.45, 0.25, 0.5).setFill()
                else:
                    NSColor.colorWithCalibratedWhite_alpha_(1.0, alpha).setFill()
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(x, h / 2 - bh / 2, BAR_W, bh), 1.5, 1.5).fill()
                x += BAR_W + BAR_GAP
            x += 10
        elif self.state == "processing":
            # a soft sweep across dimmed bars
            max_h = h - 16
            n = NBARS
            center = (math.sin(self.phase * 0.6) + 1) / 2 * (n - 1)
            for i in range(n):
                d = abs(i - center)
                v = max(0.08, math.exp(-(d * d) / 18.0))
                bh = max(2.0, v * max_h * 0.6)
                NSColor.colorWithCalibratedRed_green_blue_alpha_(r, g, bl, 0.25 + 0.6 * v).setFill()
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(x, h / 2 - bh / 2, BAR_W, bh), 1.5, 1.5).fill()
                x += BAR_W + BAR_GAP
            x += 10

        if self.label:
            color = (1.0, 1.0, 1.0) if self.state in ("recording", "locked", "processing") else (r, g, bl)
            attrs = self._text_attrs(12.5, color, 0.92)
            s = NSString.stringWithString_(self.label)
            size = s.sizeWithAttributes_(attrs)
            s.drawAtPoint_withAttributes_(NSMakePoint(x, h / 2 - size.height / 2), attrs)


class Overlay:
    """Owns the floating panel. All methods must be called on the main thread."""

    def __init__(self):
        self.width = WAVE_WIDTH
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, self.width, HEIGHT),
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel,
            NSBackingStoreBuffered, False)
        panel.setLevel_(NSScreenSaverWindowLevel)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setIgnoresMouseEvents_(True)
        panel.setHidesOnDeactivate_(False)
        panel.setReleasedWhenClosed_(False)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorStationary | NSWindowCollectionBehaviorIgnoresCycle)
        self.view = WaveView.alloc().initWithFrame_(NSMakeRect(0, 0, self.width, HEIGHT))
        panel.setContentView_(self.view)
        self.panel = panel
        self.visible = False

    # -- geometry ---------------------------------------------------------------
    def _screen(self):
        p = NSEvent.mouseLocation()
        for s in NSScreen.screens():
            f = s.frame()
            if f.origin.x <= p.x <= f.origin.x + f.size.width and f.origin.y <= p.y <= f.origin.y + f.size.height:
                return s
        return NSScreen.mainScreen() or NSScreen.screens()[0]

    def _place(self):
        vf = self._screen().visibleFrame()
        x = vf.origin.x + (vf.size.width - self.width) / 2
        y = vf.origin.y + 22
        self.panel.setFrame_display_(NSMakeRect(x, y, self.width, HEIGHT), True)
        self.view.setFrame_(NSMakeRect(0, 0, self.width, HEIGHT))

    def _resize_for_label(self, label: str, with_bars: bool):
        s = NSString.stringWithString_(label or "")
        tw = s.sizeWithAttributes_(self.view._text_attrs(12.5, (1, 1, 1))).width if label else 0
        base = 36 + (NBARS * (BAR_W + BAR_GAP) + 10 if with_bars else 0)
        self.width = min(MAX_WIDTH, max(120.0, base + tw + 18))

    # -- API -------------------------------------------------------------------
    def show(self, state: str, label: str = "") -> None:
        self.view.state = state
        self.view.label = label
        self._resize_for_label(label, with_bars=state in ("recording", "locked", "nosignal", "processing"))
        self._place()
        self.view.setNeedsDisplay_(True)
        if not self.visible:
            self.panel.orderFrontRegardless()
            self.visible = True

    def set_state(self, state: str, label: str | None = None) -> None:
        if label is None:
            label = self.view.label
        self.show(state, label)

    def hide(self) -> None:
        if self.visible:
            self.panel.orderOut_(None)
            self.visible = False
        self.view.bars.extend([0.0] * NBARS)

    def push_level(self, rms: float) -> None:
        self.view.bars.append(level_to_norm(rms))

    def tick(self) -> None:
        self.view.phase += 0.28
        if self.visible:
            self.view.setNeedsDisplay_(True)

    def snapshot_png(self, path: str) -> None:
        """Render the pill to a PNG (used to check the drawing without a screen)."""
        from AppKit import NSBitmapImageRep, NSPNGFileType

        rep = self.view.bitmapImageRepForCachingDisplayInRect_(self.view.bounds())
        self.view.cacheDisplayInRect_toBitmapImageRep_(self.view.bounds(), rep)
        rep.representationUsingType_properties_(NSPNGFileType, None).writeToFile_atomically_(path, True)
