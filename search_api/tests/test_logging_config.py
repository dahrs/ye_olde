"""Unit tests for app/logging_config.py's rotating-file logging setup.

Mirrors `ye_olde.common.logging`'s own test file's isolation approach:
`configure_logging`/`get_logger` mutate real process-wide state (the
stdlib `logging` module's own logger registry, plus this module's own
`_configured` flag and `logs/`-relative file paths), so every test points
`_LOG_DIR`/`_LOG_FILE` at a `tmp_path` and resets `_configured` first, then
removes any handler it added from the real `"search_api"` logger
afterward.
"""

from __future__ import annotations

import logging
import logging.handlers

import pytest

from app import logging_config


@pytest.fixture(autouse=True)
def _isolated_logging(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(logging_config, "_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(logging_config, "_LOG_FILE", tmp_path / "logs" / "search_api.log")
    monkeypatch.setattr(logging_config, "_configured", False)
    root = logging.getLogger("search_api")
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    for handler in list(root.handlers):
        if handler not in original_handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(original_level)


def _our_handlers() -> list[logging.handlers.RotatingFileHandler]:
    """Rotating handlers on the real `"search_api"` logger that point at
    *this test's* isolated `_LOG_FILE` -- see the analogous helper in
    `tests/unit/test_logging.py` for why a plain handler-count assertion
    would be contaminated by another test module's import-time
    `get_logger(__name__)` call.
    """
    root = logging.getLogger("search_api")
    target = str(logging_config._LOG_FILE.resolve())
    return [
        h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == target
    ]


def test_configure_logging_creates_log_dir_and_attaches_rotating_handler() -> None:
    logging_config.configure_logging()

    assert logging_config._LOG_DIR.is_dir()
    assert len(_our_handlers()) == 1


def test_configure_logging_sets_the_requested_level() -> None:
    logging_config.configure_logging(level=logging.DEBUG)

    assert logging.getLogger("search_api").level == logging.DEBUG


def test_configure_logging_defaults_to_info_level() -> None:
    logging_config.configure_logging()

    assert logging.getLogger("search_api").level == logging.INFO


def test_configure_logging_is_idempotent_within_one_process() -> None:
    logging_config.configure_logging()
    logging_config.configure_logging()
    logging_config.configure_logging(level=logging.DEBUG)  # a later call's level is also ignored

    assert len(_our_handlers()) == 1
    assert logging.getLogger("search_api").level == logging.INFO


def test_get_logger_ensures_the_handler_is_attached_without_an_explicit_configure_call() -> None:
    assert _our_handlers() == []

    logging_config.get_logger("some.module")

    assert len(_our_handlers()) == 1


def test_get_logger_prefixes_the_given_name_under_the_search_api_namespace() -> None:
    logger = logging_config.get_logger("app.loader")

    assert logger.name == "search_api.app.loader"


def test_get_logger_output_reaches_the_rotating_log_file() -> None:
    logger = logging_config.get_logger("some.module")
    logger.info("a specific, greppable message")
    for handler in _our_handlers():
        handler.flush()

    assert logging_config._LOG_FILE.read_text(encoding="utf-8").find("a specific, greppable message") != -1
