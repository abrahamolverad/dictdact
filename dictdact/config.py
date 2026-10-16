from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("DICTDACT_CONFIG_DIR", Path.home() / ".config" / "dictdact"))
CONFIG_PATH = CONFIG_DIR / "config.json"
DATA_DIR = Path(os.environ.get("DICTDACT_DATA_DIR", Path.home() / ".local" / "share" / "dictdact"))
HISTORY_PATH = DATA_DIR / "history.jsonl"
LOG_PATH = DATA_DIR / "dictdact.log"
RECORDINGS_DIR = DATA_DIR / "recordings"


@dataclass
class Config:
    # --- input ---
    microphone: str | None = None          # device name (or substring); None = system default input
    hotkey: str = "ctrl"                    # "ctrl" (either side) | "ctrl_l" | "ctrl_r" | "alt_r" | "cmd_r"
    hold_threshold_ms: int = 180            # shorter than this = a tap, not dictation
    double_tap_lock: bool = True            # tap-tap the hotkey to record hands-free; tap again to stop
    retrieve_combo: str = "ctrl+alt+v"      # re-paste the last dictation at the keyboard cursor.
                                            # Its own key on purpose: ⌘V pastes whatever the
                                            # clipboard happens to hold, this always pastes your
                                            # last take. Needs at least one modifier
    max_seconds: int = 600                  # safety cap on a single recording

    # --- speech to text ---
    engine: str = "whisper"                 # "whisper" (mlx-whisper) | "parakeet" (parakeet-mlx)
    whisper_model: str = "mlx-community/whisper-large-v3-turbo"
    parakeet_model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    language: str | None = None             # None = auto-detect (mixed es/en) | "es" | "en"
    max_gain_db: float = 40.0               # how much a whispered recording may be boosted
    context_terms: bool = True              # prime Whisper with the names visible in the active window

    # --- text structuring ---
    formatter: str = "mlx"                  # "mlx" | "ollama" | "rules" | "off"
    mlx_formatter_model: str = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    ollama_model: str = "gemma4:12b"
    ollama_url: str = "http://127.0.0.1:11434"
    formatter_timeout_s: float = 12.0   # wall-clock ceiling on one take's LLM pass; past it the
                                        # deterministic cleanup delivers the words instead of the
                                        # user waiting out a GPU another app is saturating
    formatter_always: bool = False          # False = short, already-clean takes skip the LLM (instant)
    formatter_style: str = "clean"          # "clean" = verbatim (enforced) | "improve" | "translate_en"
                                            # | "translate_es" | "summary" — explicit transformations

    # --- staying fast under memory pressure ---
    keep_warm_s: float = 45.0               # touch the models every N s while idle so macOS keeps
                                            # their weights resident (≤0 = off)
    wire_memory: bool = True                # wire MLX buffers (models can never be paged out)

    # --- output ---
    paste: bool = True                      # paste into the frontmost app at the cursor
    paste_target: str = "focus"             # "focus" = app with keyboard focus | "mouse" = app under the pointer
    restore_clipboard: bool = False         # put the previous clipboard text back after pasting.
                                            # Off by default: nothing can confirm a ⌘V actually
                                            # landed, and restoring took the dictation away from
                                            # the one retry that always works — pressing ⌘V.
    copy_to_clipboard: bool = True          # also leave the text in the clipboard when paste is off
    sounds: bool = True
    dock_icon: bool = True                  # show DictDact in the Dock and ⌘-Tab (not just the menu bar)
    show_window_at_launch: bool = False
    keep_recordings: bool = False           # save WAVs of every dictation under ~/.local/share/dictdact/recordings
    voice_bank: bool = False                # also keep every take RAW (mic's own rate, 24-bit FLAC) under
                                            # ~/.local/share/dictdact/voice-bank, graded, for voice cloning
    voice_bank_target_min: float = 180.0    # per language: collection stops at this many usable minutes

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        cfg = cls()
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError as e:
                raise SystemExit(f"config {path} is not valid JSON: {e}")
            known = {f.name for f in fields(cls)}
            for k, v in data.items():
                if k in known:
                    setattr(cfg, k, v)
        return cfg

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n")

    def ensure_dirs(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if self.keep_recordings:
            RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
