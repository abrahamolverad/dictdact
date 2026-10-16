"""Render the 🎙 emoji into an .icns (usage: make-icon.py out.icns)."""
import pathlib
import subprocess
import sys
import tempfile

from AppKit import (NSBezierPath, NSBitmapImageRep, NSColor, NSFont, NSImage, NSMakeRect, NSPNGFileType, NSString)
from Foundation import NSMakeSize

out = pathlib.Path(sys.argv[1])
with tempfile.TemporaryDirectory() as td:
    iconset = pathlib.Path(td) / "DictDact.iconset"
    iconset.mkdir()
    for size in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            px = size * scale
            img = NSImage.alloc().initWithSize_(NSMakeSize(px, px))
            img.lockFocus()
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.11, 0.11, 0.13, 1).setFill()
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(0, 0, px, px), px * 0.22, px * 0.22).fill()
            s = NSString.stringWithString_("🎙")
            attrs = {"NSFont": NSFont.systemFontOfSize_(px * 0.62)}
            w = s.sizeWithAttributes_(attrs)
            s.drawAtPoint_withAttributes_(((px - w.width) / 2, (px - w.height) / 2), attrs)
            img.unlockFocus()
            rep = NSBitmapImageRep.imageRepWithData_(img.TIFFRepresentation())
            png = rep.representationUsingType_properties_(NSPNGFileType, None)
            name = f"icon_{size}x{size}" + ("@2x" if scale == 2 else "") + ".png"
            png.writeToFile_atomically_(str(iconset / name), True)
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)], check=True)
