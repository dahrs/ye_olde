"""Logging setup for this service — its own rotating file under `logs/`
(gitignored, created on first use), separate from the main `ye_olde`
package's `logs/ye_olde.log` (see config.py's docstring for why this service
doesn't depend on that package at all). Mirrors
`ye_olde.common.logging`'s pattern deliberately, not by importing it.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "search_api.log"
_ROOT_LOGGER_NAME = "search_api"
_configured = False


def configure_logging(level: int = logging.INFO) -> None:
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
    configure_logging()
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
