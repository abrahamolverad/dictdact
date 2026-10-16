"""Per-day totals over the whole history — the numbers behind "words per day".

The sidebar and the chart in Dictations both read this. history.jsonl is already megabytes and
``/api/state`` polls once a second, so the file is parsed only when it actually changed.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import date, timedelta

from .history import Block, History


@dataclass
class Day:
    date: str                 # YYYY-MM-DD, local time — the day the user was speaking
    takes: int = 0
    words: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {"date": self.date, "takes": self.takes, "words": self.words,
                "seconds": round(self.seconds, 1)}


@dataclass
class Snapshot:
    days: list[Day] = field(default_factory=list)   # every calendar day, first take → today
    today: Day = field(default_factory=lambda: Day(date.today().isoformat()))
    words: int = 0
    takes: int = 0
    seconds: float = 0.0
    first_day: str | None = None
    days_since_first: int = 0      # calendar days from the first take to today, inclusive
    days_used: int = 0             # days the user actually said something
    words_per_day: float = 0.0     # words ÷ days_since_first — the headline number
    best: Day | None = None        # the loudest day, for the chart caption

    def as_dict(self) -> dict:
        return {
            "days": [d.as_dict() for d in self.days],
            "today": self.today.as_dict(),
            "totals": {"words": self.words, "takes": self.takes, "seconds": round(self.seconds, 1),
                       "first_day": self.first_day, "days_since_first": self.days_since_first,
                       "days_used": self.days_used, "words_per_day": round(self.words_per_day, 1),
                       "best": self.best.as_dict() if self.best else None},
        }


def _span(first: date, last: date):
    d = first
    while d <= last:
        yield d
        d += timedelta(days=1)


def build(blocks: list[Block], today: str | None = None) -> Snapshot:
    """Aggregate every take into its own day.

    Days the user said nothing are kept as zeros: the chart has to show the gaps, not close them up.
    """
    iso_today = today or date.today().isoformat()
    per: dict[str, Day] = {}
    for b in blocks:
        key = (b.ts or "")[:10]
        try:
            date.fromisoformat(key)
        except ValueError:
            continue                      # a hand-edited line without a real ts is not a day
        day = per.get(key)
        if day is None:
            day = per[key] = Day(key)
        day.takes += 1
        day.words += len(b.text.split())
        day.seconds += b.duration_s or 0.0

    snap = Snapshot(today=Day(iso_today))
    if not per:
        return snap

    first = date.fromisoformat(min(per))
    # a take stamped in the future (clock skew) still gets its bar, and never a negative span
    last = max(date.fromisoformat(max(per)), date.fromisoformat(iso_today))
    snap.days = [per.get(d.isoformat()) or Day(d.isoformat()) for d in _span(first, last)]
    snap.today = per.get(iso_today) or Day(iso_today)
    snap.words = sum(d.words for d in per.values())
    snap.takes = sum(d.takes for d in per.values())
    snap.seconds = sum(d.seconds for d in per.values())
    snap.first_day = first.isoformat()
    snap.days_since_first = (last - first).days + 1
    snap.days_used = len(per)
    snap.words_per_day = snap.words / snap.days_since_first
    snap.best = max(per.values(), key=lambda d: (d.words, d.date))
    return snap


class Stats:
    """The parsed history plus its per-day totals, re-read only when the file changes."""

    def __init__(self, history: History):
        self.history = history
        self._lock = threading.Lock()
        self._key: tuple[int, int] | None = None
        self._blocks: list[Block] = []
        self._snap: Snapshot | None = None

    def _file_key(self) -> tuple[int, int] | None:
        """(mtime, size) of history.jsonl. None — no file to key on (a fake history in tests,
        or nothing dictated yet) — means "never cache", so a fake is always read live."""
        path = getattr(self.history, "path", None)
        if path is None:
            return None
        try:
            st = path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def load(self, today: str | None = None) -> tuple[list[Block], Snapshot]:
        """The history and its totals from one read. Called every second by /api/state."""
        iso = today or date.today().isoformat()
        with self._lock:
            # Sampled BEFORE the read: an append that lands mid-read is then keyed to the old
            # file and gets picked up next call. Sampling after would cache a stale count under
            # the new key and hold it until the take after that one.
            key = self._file_key()
            if key is not None and key == self._key and self._snap is not None and self._snap.today.date == iso:
                return list(self._blocks), self._snap      # midnight rolls the cache too
            blocks = self.history.all()
            snap = build(blocks, iso)
            self._key, self._blocks, self._snap = key, blocks, snap
            return list(blocks), snap
