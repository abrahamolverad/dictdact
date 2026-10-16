"""Context for one take: the names visible in the app the user is dictating into.

If the window in front shows "Simone", "Hyrox" and "ManyChat", those are the words most likely
to be said next — and the ones Whisper is most likely to misspell. They go into Whisper's prompt
tail for THAT take only. Read through Accessibility with hard bounds (time, node count, text
size) on its own thread; a take never waits for it — whatever was ready when the audio stops
is used, nothing otherwise.
"""
from __future__ import annotations

import collections
import logging
import threading
import time

from .learn import capitalized_names

log = logging.getLogger("dictdact.context")

MAX_TERMS = 15
TIME_BUDGET_S = 0.3
MAX_NODES = 300
MAX_CHARS = 20000


def terms_from_texts(texts: list[str], exclude: set[str] | None = None, limit: int = MAX_TERMS) -> list[str]:
    """Pure part: the capitalized names in some text, most frequent first."""
    counts: collections.Counter = collections.Counter()
    for t in texts:
        for w in capitalized_names(t, include_initial=True):   # screen text: titles and list items
            counts[w] += 1
    ex = {e.lower() for e in (exclude or ())}
    return [w for w, _ in counts.most_common() if w.lower() not in ex][:limit]


def visible_texts(time_budget_s: float = TIME_BUDGET_S, max_nodes: int = MAX_NODES) -> tuple[str | None, list[str]]:
    """(app name, texts) of the frontmost window through Accessibility. Bounded; any failure
    → (name, []). Needs the Accessibility grant the app already has for pasting."""
    try:
        from AppKit import NSWorkspace
        from ApplicationServices import (AXUIElementCopyAttributeValue, AXUIElementCreateApplication,
                                         AXUIElementSetMessagingTimeout, kAXChildrenAttribute,
                                         kAXFocusedWindowAttribute, kAXTitleAttribute, kAXValueAttribute)
    except Exception:
        return None, []
    app = NSWorkspace.sharedWorkspace().menuBarOwningApplication()
    name = str(app.localizedName()) if app else None
    if app is None:
        return None, []
    t0 = time.monotonic()
    texts: list[str] = []
    total = 0
    seen = 0
    try:
        ax = AXUIElementCreateApplication(app.processIdentifier())
        AXUIElementSetMessagingTimeout(ax, 0.1)
        err, win = AXUIElementCopyAttributeValue(ax, kAXFocusedWindowAttribute, None)
        if err or win is None:
            return name, []
        stack = [(win, 0)]
        while stack:
            if time.monotonic() - t0 > time_budget_s or seen >= max_nodes or total >= MAX_CHARS:
                break
            el, depth = stack.pop()
            seen += 1
            for attr in (kAXValueAttribute, kAXTitleAttribute):
                _, val = AXUIElementCopyAttributeValue(el, attr, None)
                if isinstance(val, str) and len(val) > 2:
                    texts.append(val[: MAX_CHARS - total])
                    total += len(val)
            if depth < 10:
                _, kids = AXUIElementCopyAttributeValue(el, kAXChildrenAttribute, None)
                for k in reversed(list(kids or [])[:80]):
                    stack.append((k, depth + 1))
    except Exception as e:
        log.debug("context read failed: %s", e)
    return name, texts


class ContextReader:
    """Kick off a read at hotkey-down; `take()` returns the terms if the read finished."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._gen = 0                                  # one generation per take (a
        self._result: tuple[int, str | None, list[str]] | None = None   # slow read must never land on a later take)

    def start(self, exclude: set[str] | None = None) -> None:
        with self._lock:
            self._gen += 1
            gen = self._gen
            self._result = None                        # whatever an earlier take left behind is gone
        if not self.enabled:
            return

        def run():
            name, texts = visible_texts()
            terms = terms_from_texts(texts, exclude)
            with self._lock:
                if gen == self._gen:                   # only the take that asked for it
                    self._result = (gen, name, terms)

        threading.Thread(target=run, name="dictdact-context", daemon=True).start()

    def take(self) -> tuple[str | None, list[str]]:
        with self._lock:
            r = self._result
            self._result = None
            if r is None or r[0] != self._gen:
                return None, []
            return r[1], r[2]
