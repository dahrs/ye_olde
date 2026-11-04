"""Unit tests for common/errors.py's exception hierarchy.

This module defines no functions of its own -- what's actually load-bearing
is the *catchability* contract each subclass promises (see each class's own
docstring and CLAUDE.md "Error handling"): which `except` clauses a
boundary layer can use to catch one of these without also silently
swallowing an unrelated bug.
"""

from __future__ import annotations

import pytest

from ye_olde.common.errors import (
    ConfigurationError,
    LLMEmptyResponseError,
    LLMError,
    LLMNotConfiguredError,
    YeOldeError,
)


def test_yeolde_error_is_a_plain_exception():
    assert issubclass(YeOldeError, Exception)
    with pytest.raises(YeOldeError):
        raise YeOldeError("boom")


def test_configuration_error_is_a_yeolde_error():
    assert issubclass(ConfigurationError, YeOldeError)
    with pytest.raises(YeOldeError):
        raise ConfigurationError("missing SEARCH_API_URL")


def test_llm_error_is_both_yeolde_error_and_runtime_error():
    # Deliberate multiple inheritance (see the class's own docstring): an
    # existing `except ValueError` call site around `call_llm_json` (which
    # means to catch its own JSON-repair ValueError) must NOT also catch
    # this.
    assert issubclass(LLMError, YeOldeError)
    assert issubclass(LLMError, RuntimeError)
    assert not issubclass(LLMError, ValueError)

    with pytest.raises(RuntimeError):
        raise LLMError("no model configured")
    with pytest.raises(YeOldeError):
        raise LLMError("no model configured")


def test_llm_error_subclass_is_not_swallowed_by_a_value_error_handler():
    try:
        raise LLMNotConfiguredError("LITELLM_MODEL isn't set")
    except ValueError:
        pytest.fail("LLMNotConfiguredError must not be catchable as a ValueError")
    except LLMError:
        pass


def test_llm_not_configured_error_is_an_llm_error():
    assert issubclass(LLMNotConfiguredError, LLMError)


def test_llm_empty_response_error_is_an_llm_error():
    assert issubclass(LLMEmptyResponseError, LLMError)


def test_llm_not_configured_and_empty_response_are_distinct_sibling_types():
    # A caller branching on *which* LLM failure occurred must be able to
    # tell these two apart, not just catch LLMError generically.
    assert not issubclass(LLMNotConfiguredError, LLMEmptyResponseError)
    assert not issubclass(LLMEmptyResponseError, LLMNotConfiguredError)


def test_exception_message_is_preserved_through_str():
    assert str(YeOldeError("something specific went wrong")) == "something specific went wrong"
