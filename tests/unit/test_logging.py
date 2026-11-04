"""Unit tests for common/logging.py's rotating-file logging setup.

`configure_logging`/`get_logger` mutate real process-wide state (the
stdlib `logging` module's own logger registry, plus this module's own
`_configured` flag and `logs/`-relative file paths) -- every test points
`_LOG_DIR`/`_LOG_FILE` at a `tmp_path` and resets `_configured` first, then
removes any handler it added from the real `"ye_olde"` logger afterward, so
this file never writes to the real `logs/ye_olde.log` or leaks a handler
into another test.
"""

from __future__ import annotations

import logging
import logging.handlers

import pytest

from ye_olde.common import logging as ye_olde_logging


@pytest.fixture(autouse=True)
def _isolated_logging(tmp_path, monkeypatch):
    monkeypatch.setattr(ye_olde_logging, "_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(ye_olde_logging, "_LOG_FILE", tmp_path / "logs" / "ye_olde.log")
    monkeypatch.setattr(ye_olde_logging, "_configured", False)
    root = logging.getLogger("ye_olde")
    original_handlers = list(root.handlers)
    original_level = root.level
    yield
    for handler in list(root.handlers):
        if handler not in original_handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(original_level)


def _our_handlers() -> list[logging.handlers.RotatingFileHandler]:
    """Rotating handlers on the real, process-wide `"ye_olde"` logger that
    point at *this test's* isolated `_LOG_FILE` -- not a raw count of every
    handler on that logger. Other test modules' import-time
    `get_logger(__name__)` calls (a module-level `_log = get_logger(__name__)`
    is this project's own convention) attach a real handler pointing at the
    real `logs/ye_olde.log` the first time any of them is imported in this
    process, and that handler is never removed for the rest of the pytest
    session -- a plain handler-count assertion would be contaminated by it
    whenever this file doesn't happen to run first.
    """
    root = logging.getLogger("ye_olde")
    target = str(ye_olde_logging._LOG_FILE.resolve())
    return [
        h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == target
    ]


def test_configure_logging_creates_log_dir_and_attaches_rotating_handler():
    ye_olde_logging.configure_logging()

    assert ye_olde_logging._LOG_DIR.is_dir()
    assert len(_our_handlers()) == 1


def test_configure_logging_sets_the_requested_level():
    ye_olde_logging.configure_logging(level=logging.DEBUG)

    assert logging.getLogger("ye_olde").level == logging.DEBUG


def test_configure_logging_defaults_to_info_level():
    ye_olde_logging.configure_logging()

    assert logging.getLogger("ye_olde").level == logging.INFO


def test_configure_logging_is_idempotent_within_one_process():
    ye_olde_logging.configure_logging()
    ye_olde_logging.configure_logging()
    ye_olde_logging.configure_logging(level=logging.DEBUG)  # a later call's level is also ignored

    assert len(_our_handlers()) == 1
    assert logging.getLogger("ye_olde").level == logging.INFO


def test_get_logger_ensures_the_handler_is_attached_without_an_explicit_configure_call():
    assert _our_handlers() == []

    ye_olde_logging.get_logger("some.module")

    assert len(_our_handlers()) == 1


def test_get_logger_prefixes_the_given_name_under_the_ye_olde_namespace():
    logger = ye_olde_logging.get_logger("ingest.survey")

    assert logger.name == "ye_olde.ingest.survey"


def test_get_logger_called_with_a_real_modules_dunder_name_double_prefixes():
    # Documents real, observed behavior rather than assumed-correct
    # behavior: every call site in this codebase does `get_logger(__name__)`,
    # and `__name__` for a module inside this package is already fully
    # dotted (e.g. "ye_olde.ingest.survey") -- so the resulting logger name
    # is "ye_olde.ye_olde.ingest.survey", not "ye_olde.ingest.survey". This
    # doesn't break log capture (the name is still a dotted descendant of
    # the "ye_olde" logger the rotating handler is attached to, so records
    # still propagate and get written), but it is a real double-prefix in
    # every log line's logger name -- flagged here rather than silently
    # assumed intentional.
    logger = ye_olde_logging.get_logger("ye_olde.ingest.survey")

    assert logger.name == "ye_olde.ye_olde.ingest.survey"
    assert logger.name.startswith("ye_olde.")  # still a descendant of the handler-bearing root


def test_get_logger_output_reaches_the_rotating_log_file():
    logger = ye_olde_logging.get_logger("some.module")
    logger.info("a specific, greppable message")
    for handler in _our_handlers():
        handler.flush()

    assert ye_olde_logging._LOG_FILE.read_text(encoding="utf-8").find("a specific, greppable message") != -1
