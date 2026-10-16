"""Every dictation is kept as its own block so nothing is ever lost."""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .config import HISTORY_PATH

log = logging.getLogger("dictdact.history")


@dataclass
class Block:
    id: str
    ts: str
    text: str
    raw: str
    language: str | None = None
    engine: str = ""
    formatter: str = ""
    mode: str = ""
    duration_s: float = 0.0
    app: str | None = None
    pasted: bool = False
    audio_path: str | None = None
    timings: dict = field(default_factory=dict)

    @classmethod
    def new(cls, **kw) -> "Block":
        now = datetime.now()
        return cls(id=now.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
                   ts=now.isoformat(timespec="seconds"), **kw)

    def preview(self, width: int = 60) -> str:
        one = " ".join(self.text.split())
        return one if len(one) <= width else one[: width - 1] + "…"


class History:
    def __init__(self, path: Path = HISTORY_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()      # the worker appends while the window may rewrite a fix

    def append(self, block: Block) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(block), ensure_ascii=False) + "\n")

    def update(self, block: Block) -> None:
        """Rewrite the line for ``block`` (used to flip ``pasted`` and to save a fix). Atomic:
        a crash mid-rewrite must never truncate the whole history."""
        with self._lock:
            blocks = self.all()
            tmp = self.path.with_name(self.path.name + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for b in blocks:
                    if b.id == block.id:
                        b = block
                    f.write(json.dumps(asdict(b), ensure_ascii=False) + "\n")
            tmp.replace(self.path)

    def all(self) -> list[Block]:
        if not self.path.exists():
            return []
        out: list[Block] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Block(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as e:
                log.warning("skipping bad history line: %s", e)
        return out

    def last(self, n: int = 10) -> list[Block]:
        return self.all()[-n:][::-1]

    def latest(self) -> Block | None:
        blocks = self.last(1)
        return blocks[0] if blocks else None
