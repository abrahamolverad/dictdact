#!/bin/bash
# One-shot setup: venv + deps + models + DictDact.app + `dictdact` on PATH.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
command -v uv >/dev/null || brew install uv
command -v ffmpeg >/dev/null || brew install ffmpeg
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install -e ".[parakeet]" --python .venv/bin/python
.venv/bin/python -m dictdact setup
./scripts/make-app.sh
BIN=/opt/homebrew/bin; [ -w "$BIN" ] || BIN="$HOME/.local/bin"; mkdir -p "$BIN"
ln -sf "$ROOT/.venv/bin/dictdact" "$BIN/dictdact"
echo
echo "Installed. Next:"
echo "  open ~/Applications/DictDact.app      # then grant Input Monitoring + Accessibility when asked"
echo "  dictdact doctor                        # check everything"
