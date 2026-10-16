"""Which models DictDact needs, whether they are on disk, and downloading them with progress."""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("dictdact.models")

# Approximate sizes, used for the progress bar before the exact file list is known.
_APPROX_BYTES = {
    "mlx-community/whisper-large-v3-turbo": 1_620_000_000,
    "mlx-community/Qwen3-4B-Instruct-2507-4bit": 2_280_000_000,
    "mlx-community/parakeet-tdt-0.6b-v3": 2_400_000_000,
}


def required_repos(cfg) -> list[str]:
    repos = [cfg.whisper_model if cfg.engine == "whisper" else cfg.parakeet_model]
    if cfg.formatter == "mlx":
        repos.append(cfg.mlx_formatter_model)
    return repos


def is_cached(repo: str) -> bool:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snapshot_download(repo, local_files_only=True)
        return True
    except (LocalEntryNotFoundError, FileNotFoundError, OSError, ValueError):
        return False
    except Exception:  # pragma: no cover - hub internals
        return False


def missing_repos(cfg) -> list[str]:
    return [r for r in required_repos(cfg) if not is_cached(r)]


def _repo_cache_dir(repo: str) -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))


def _dir_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


@dataclass
class Download:
    repos: list[str]
    total_bytes: int = 0
    done_bytes: int = 0
    current: str = ""
    finished: bool = False
    error: str | None = None
    _sizes: dict[str, int] = field(default_factory=dict)

    @property
    def fraction(self) -> float:
        return min(1.0, self.done_bytes / self.total_bytes) if self.total_bytes else 0.0

    def start(self) -> None:
        threading.Thread(target=self._run, name="dictdact-download", daemon=True).start()

    def _expected(self, repo: str) -> int:
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(repo, files_metadata=True)
            size = sum((s.size or 0) for s in info.siblings)
            if size > 0:
                return size
        except Exception as e:
            log.warning("could not size %s: %s", repo, e)
        return _APPROX_BYTES.get(repo, 2_000_000_000)

    def _run(self) -> None:
        from huggingface_hub import snapshot_download

        try:
            self._sizes = {r: self._expected(r) for r in self.repos}
            self.total_bytes = sum(self._sizes.values())
            finished_bytes = 0
            for repo in self.repos:
                self.current = repo.split("/")[-1]
                stop = threading.Event()

                def watch():
                    while not stop.is_set():
                        self.done_bytes = finished_bytes + min(self._sizes[repo], _dir_bytes(_repo_cache_dir(repo)))
                        stop.wait(0.5)

                t = threading.Thread(target=watch, daemon=True)
                t.start()
                try:
                    snapshot_download(repo)
                finally:
                    stop.set()
                    t.join(timeout=1)
                finished_bytes += self._sizes[repo]
                self.done_bytes = finished_bytes
            self.finished = True
        except Exception as e:
            log.exception("model download failed")
            self.error = f"{type(e).__name__}: {e}"
