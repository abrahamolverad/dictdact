#!/bin/bash
# Build ~/Applications/DictDact.app: a real app bundle around the project venv so macOS
# attributes Microphone / Input Monitoring / Accessibility permissions to "DictDact".
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="${1:-$HOME/Applications/DictDact.app}"
VENV="$ROOT/.venv"
[ -x "$VENV/bin/python" ] || { echo "venv missing — run scripts/install.sh first"; exit 1; }

PYREAL="$("$VENV/bin/python" -c 'import os,sys; print(os.path.realpath(sys.executable))')"
FW_DIR="$(dirname "$(dirname "$PYREAL")")"
GUI_BIN="$FW_DIR/Resources/Python.app/Contents/MacOS/Python"
PYBIN="$PYREAL"; [ -x "$GUI_BIN" ] && PYBIN="$GUI_BIN"
VERSION="$("$VENV/bin/python" -c 'import dictdact; print(dictdact.__version__)')"

if [ -d "$APP" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "$APP already exists. Code changes do NOT need a rebuild (just relaunch the app)."
  echo "Rebuilding re-signs the bundle and macOS will FORGET its Microphone / Input Monitoring /"
  echo "Accessibility grants — you would have to grant them again. Run with FORCE=1 to rebuild anyway."
  exit 0
fi
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$PYBIN" "$APP/Contents/MacOS/python"
ln -s "$VENV/lib" "$APP/Contents/lib"
cp "$VENV/pyvenv.cfg" "$APP/Contents/pyvenv.cfg"

cat > "$APP/Contents/MacOS/DictDact" <<'LAUNCH'
#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
exec "$DIR/python" -m dictdact run
LAUNCH
chmod +x "$APP/Contents/MacOS/DictDact"

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
  <key>LSUIElement</key><true/>
  <key>NSAppSleepDisabled</key><true/>
  <key>NSSupportsAutomaticTermination</key><false/>
  <key>NSSupportsSuddenTermination</key><false/>
  <key>LSMinimumSystemVersion</key><string>13.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>DictDact listens to your microphone while you hold Ctrl, to transcribe what you say on this Mac.</string>
</dict></plist>
PLIST

"$VENV/bin/python" "$ROOT/scripts/make-icon.py" "$APP/Contents/Resources/DictDact.icns"

codesign --force --deep --sign - "$APP" 2>/dev/null || echo "(ad-hoc codesign skipped)"
echo "built $APP"
