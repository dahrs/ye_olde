"""Logging setup — one rotating file under `logs/` (gitignored, created on
first use) that every part of the pipeline writes to, so an error is always
somewhere to grep for after the fact rather than only ever printed to a
terminal that's since scrolled away.

Usage: `log = get_logger(__name__)` at module scope, same as the stdlib
pattern this wraps. Internal/library code should mostly *not* call
`log.error`/`log.exception` — see CLAUDE.md "Error handling": let the
exception propagate, and let the boundary (a CLI `main()`, a FastAPI
handler) log it once, with full context, when it catches it. A narrow,
already-handled fallback (a specific, expected exception with documented
recovery behavior) may still log at `debug`/`info` so the fallback is
visible in the log file instead of disappearing silently.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "ye_olde.log"
_ROOT_LOGGER_NAME = "ye_olde"
_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Attaches the rotating file handler to the `ye_olde` logger tree, once
    per process. Safe to call more than once (idempotent) and safe to call
    before `logs/` exists (created here).
    """
    global _configured
    if _configured:
        return
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(_LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(level)
    root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Returns a logger under the `ye_olde` tree, ensuring the rotating file
    handler is attached first. `name` is conventionally the caller's
    `__name__`.
    """
    configure_logging()
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
