#!/bin/bash
# Build a RELOCATABLE DictDact.app (bundled CPython + all wheels; no Homebrew, no venv) into dist/stage,
# then sign it and package a DMG. Works on any Apple-silicon Mac with `uv` (and Xcode CLT for clang).
#
#   scripts/build-release.sh                         # ad-hoc signed (Gatekeeper will warn on other Macs)
#   SIGN_IDENTITY="Developer ID Application: Name (TEAMID)" NOTARY_PROFILE=dictdact-notary scripts/build-release.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VERSION="$(sed -nE 's/^version = "(.*)"/\1/p' pyproject.toml | head -1)"
PYVER=3.12
DIST="$ROOT/dist"; STAGE="$DIST/stage"; APP="$STAGE/DictDact.app"
ENT="$ROOT/scripts/entitlements.plist"
command -v uv >/dev/null || { echo "uv is required: brew install uv"; exit 1; }
xcrun -f clang >/dev/null 2>&1 || { echo "Xcode command line tools are required: xcode-select --install"; exit 1; }

echo "==> DictDact $VERSION: staging relocatable app"
rm -rf "$STAGE"; mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" 

# 1. Relocatable CPython (python-build-standalone via uv)
uv python install "$PYVER" >/dev/null 2>&1 || true
PYEXE="$(UV_PYTHON_PREFERENCE=only-managed uv python find --no-project "$PYVER")"
PYROOT="$(cd "$(dirname "$PYEXE")/.." && pwd -P)"
cp -R "$PYROOT/." "$APP/Contents/Resources/python/"
BPY="$APP/Contents/Resources/python/bin/python$PYVER"
rm -f "$APP/Contents/Resources/python/lib/python$PYVER/EXTERNALLY-MANAGED"   # it's ours now
rm -rf "$APP/Contents/Resources/python/lib/python$PYVER/test" "$APP/Contents/Resources/python/lib/python$PYVER/idlelib" \
       "$APP/Contents/Resources/python/lib/python$PYVER/tkinter" "$APP/Contents/Resources/python/share"

# 2. Dependencies. mlx-whisper declares torch (515 MB) but only its model-conversion module imports it.
echo "==> installing dependencies into the bundle"
uv pip install --python "$BPY" --no-deps "mlx-whisper==$(sed -nE 's/^mlx-whisper==(.*)/\1/p' scripts/requirements-release.txt)" >/dev/null
grep -v "^mlx-whisper==" scripts/requirements-release.txt > "$DIST/.req.txt"
uv pip install --python "$BPY" -r "$DIST/.req.txt" >/dev/null
uv pip install --python "$BPY" --no-deps "$ROOT" >/dev/null
SP="$APP/Contents/Resources/python/lib/python$PYVER/site-packages"
rm -rf "$SP"/PyObjCTest "$SP"/pip "$SP"/setuptools "$SP"/_distutils_hack "$SP"/pkg_resources 2>/dev/null || true
find "$APP/Contents/Resources/python" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$APP/Contents/Resources/python" -name "*.dSYM" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$APP/Contents/Resources/python" -name "*.pyc" -delete 2>/dev/null || true

# 3. The interpreter must live in Contents/MacOS so macOS treats DictDact.app as its main bundle
#    (bundle identifier, NSMicrophoneUsageDescription, TCC identity, SMAppService all depend on it).
cp "$BPY" "$APP/Contents/MacOS/python"
ln -sfn "Resources/python/lib" "$APP/Contents/lib"       # @rpath/../lib + Python's prefix landmark
BPY="$APP/Contents/MacOS/python"

# 4. Launcher (Mach-O) + metadata + icon
echo "==> launcher, Info.plist, icon"
xcrun clang -O2 -arch arm64 -mmacosx-version-min=14.0 -o "$APP/Contents/MacOS/DictDact" scripts/launcher.c
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>DictDact</string>
  <key>CFBundleDisplayName</key><string>DictDact</string>
  <key>CFBundleIdentifier</key><string>com.abrahamolvera.dictdact</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleExecutable</key><string>DictDact</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>DictDact</string>
  <key>CFBundleSupportedPlatforms</key><array><string>MacOSX</string></array>
  <key>LSArchitecturePriority</key><array><string>arm64</string></array>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSUIElement</key><true/>
  <key>NSAppSleepDisabled</key><true/>
  <key>NSSupportsAutomaticTermination</key><false/>
  <key>NSSupportsSuddenTermination</key><false/>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSHumanReadableCopyright</key><string>© Abraham Olvera. Runs entirely on your Mac.</string>
  <key>NSMicrophoneUsageDescription</key><string>DictDact listens to your microphone while you hold Ctrl, to transcribe what you say on this Mac. Audio never leaves the computer.</string>
</dict></plist>
PLIST
"$BPY" "$ROOT/scripts/make-icon.py" "$APP/Contents/Resources/DictDact.icns"
# Bundled LaunchAgent for "start at login + restart on crash" (registered via SMAppService from the menu)
mkdir -p "$APP/Contents/Library/LaunchAgents"
cat > "$APP/Contents/Library/LaunchAgents/com.abrahamolvera.dictdact.agent.plist" <<AGENT
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.abrahamolvera.dictdact.agent</string>
  <key>BundleProgram</key><string>Contents/MacOS/DictDact</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>5</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>AssociatedBundleIdentifiers</key><string>com.abrahamolvera.dictdact</string>
</dict></plist>
AGENT

# 4. Sanity: nothing may reference Homebrew or this user's home
echo "==> relocatability check"
"$BPY" "$ROOT/scripts/check-relocatable.py" "$APP"
env -i PATH=/usr/bin:/bin HOME=/tmp "$BPY" -c "
import sys, dictdact, mlx_whisper, mlx_lm, sounddevice, rumps, pynput, parakeet_mlx, AVFoundation
from Foundation import NSBundle
assert NSBundle.mainBundle().bundleIdentifier() == 'com.abrahamolvera.dictdact', NSBundle.mainBundle().bundlePath()
assert 'DictDact.app/Contents' in sys.prefix, sys.prefix
print('   bundled python imports OK, main bundle =', NSBundle.mainBundle().bundleIdentifier())"

# 5. Sign
echo "==> signing"
sign_file() { # $1 file, $2 with-entitlements(1/0)
  if [ -n "${SIGN_IDENTITY:-}" ]; then
    if [ "$2" = "1" ]; then codesign --force --options runtime --timestamp --entitlements "$ENT" --sign "$SIGN_IDENTITY" "$1"
    else codesign --force --options runtime --timestamp --sign "$SIGN_IDENTITY" "$1"; fi
  fi
}
if [ -n "${SIGN_IDENTITY:-}" ]; then
  find "$APP/Contents/Resources/python" -type f -print0 | xargs -0 file | grep "Mach-O" | cut -d: -f1 | while read -r f; do
    case "$f" in */bin/python*) sign_file "$f" 1 ;; *) sign_file "$f" 0 ;; esac
  done
  sign_file "$APP/Contents/MacOS/python" 1
  sign_file "$APP/Contents/MacOS/DictDact" 1
  codesign --force --options runtime --timestamp --entitlements "$ENT" --sign "$SIGN_IDENTITY" "$APP"
else
  codesign --force --deep --sign - "$APP"
  echo "   (ad-hoc signature: other Macs will need System Settings → Privacy & Security → Open Anyway)"
fi
codesign --verify --deep --strict "$APP" && echo "   signature OK"

# 6. DMG
echo "==> packaging"
DMG="$DIST/DictDact-$VERSION.dmg"
rm -f "$DMG"; ln -sfn /Applications "$STAGE/Applications"
hdiutil create -quiet -volname "DictDact $VERSION" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
rm -f "$STAGE/Applications"
if [ -n "${SIGN_IDENTITY:-}" ]; then codesign --force --timestamp --sign "$SIGN_IDENTITY" "$DMG"; fi
if [ -n "${NOTARY_PROFILE:-}" ]; then
  echo "==> notarizing (this takes a few minutes)"
  xcrun notarytool submit "$DMG" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$DMG"
fi
du -sh "$APP" "$DMG" | sed 's/^/   /'
echo "==> done: $DMG"
