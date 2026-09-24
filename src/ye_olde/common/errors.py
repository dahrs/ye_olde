"""Project-wide exception hierarchy.

Only exceptions a boundary layer (a CLI `main()`, a future FastAPI route in
the translation pipeline itself) needs to distinguish from an arbitrary bug
live here. Everything else — "this argument is malformed", "this file
doesn't match the expected shape" — should keep raising a plain builtin
(`ValueError`, `TypeError`, `KeyError`, ...) rather than a project-specific
type invented just to wrap it; see CLAUDE.md "Error handling" for the full
rule. Domain code lets these propagate uncaught; only the boundary catches
`YeOldeError` (or a subclass), logs it via `common.logging.get_logger`, and
converts it to a response.
"""

from __future__ import annotations


class YeOldeError(Exception):
    """Base class for every exception this project defines on purpose."""


class ConfigurationError(YeOldeError):
    """A required setting is missing or invalid (see config.py)."""


class LLMError(YeOldeError, RuntimeError):
    """Base for failures talking to the configured generation LLM.

    Also a `RuntimeError` so existing `except ValueError` call sites around
    `call_llm_json` (which do mean to catch JSON-repair failures, a
    `ValueError` subclass) do not accidentally start swallowing these too.
    """


class LLMNotConfiguredError(LLMError):
    """`LITELLM_MODEL` isn't set."""


class LLMEmptyResponseError(LLMError):
    """The model returned no usable content, even after the configured
    empty-reply retry (see ingest/llm_client.py).
    """
