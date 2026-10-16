from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from .config import DATA_DIR, LOG_PATH


def setup_logging(verbose: bool = False) -> logging.Logger:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("dictdact")
    log.setLevel(logging.DEBUG)
    if log.handlers:
        return log
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = RotatingFileHandler(LOG_PATH, maxBytes=2_000_000, backupCount=2)
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.addHandler(sh)
    return log
