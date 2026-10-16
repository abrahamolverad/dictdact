"""When to re-touch the models so macOS keeps their weights resident.

Under heavy memory pressure (another app taking tens of GB) the compressor/pager
evicts DictDact's idle model weights; the next dictation then repays the page-in
*after* the key is released. Two counter-measures share this logic:

- a heartbeat touch every ``keep_warm_s`` while idle, and
- a pre-warm touch on hotkey-down, so any page-in overlaps with the user speaking.
"""
from __future__ import annotations

# A pre-warm on hotkey-down is pointless if the models ran this recently — they are
# certainly still resident. Kept small so the pre-warm still fires when the heartbeat
# is disabled or was missed (sleep, clock jump).
HOTKEY_PREWARM_IDLE_S = 20.0


def warm_due(last_use_ts: float, now: float, interval_s: float | None) -> bool:
    """True when the models have not been used for ``interval_s`` and a touch is due.

    ``interval_s`` may come straight from a hand-edited config.json — a mistyped value must
    disable the heartbeat quietly, never turn its 5 s poll into a traceback pump."""
    try:
        interval = float(interval_s) if interval_s is not None else 0.0
    except (TypeError, ValueError):
        return False
    if interval <= 0:
        return False
    return (now - last_use_ts) >= interval
