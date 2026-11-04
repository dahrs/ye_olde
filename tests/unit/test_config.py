"""Unit tests for config.py's `Settings`/`get_settings`.

Every test that checks a *default* value passes `_env_file=None` and
clears the specific environment variables it cares about first -- this
repo has a real, populated `.env` at the repo root (gitignored, real
credentials), so a test that let `Settings()` read it by accident would
see whatever that file happens to contain instead of this module's own
documented defaults, and would pass or fail depending on local machine
state rather than the code under test.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ye_olde.config import Settings, get_settings

_ENV_VARS = (
    "EMBEDDING_MODEL",
    "LITELLM_MODEL",
    "LITELLM_API_KEY",
    "LITELLM_API_BASE",
    "AWS_BEARER_TOKEN_BEDROCK",
    "GEMINI_API_KEY",
    "LITELLM_EXTRA_BODY",
    "LITELLM_NO_THINKING_EXTRA_BODY",
    "LITELLM_TIMEOUT_SECONDS",
    "SEARCH_API_URL",
    "SEARCH_API_TIMEOUT_SECONDS",
    "HF_DATASET_REPO_ID",
    "HF_TOKEN",
    "GALLICA_COOKIE",
)


def _clean_settings(monkeypatch, **env_overrides: str) -> Settings:
    for key in _ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env_overrides.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_settings_defaults_when_nothing_is_configured(monkeypatch):
    settings = _clean_settings(monkeypatch)

    assert settings.embedding_model == "intfloat/multilingual-e5-large"
    assert settings.litellm_model == ""
    assert settings.litellm_timeout_seconds == 6000.0
    assert settings.search_api_url == ""
    assert settings.search_api_timeout_seconds == 120.0
    assert settings.hf_dataset_repo_id == ""
    assert settings.gallica_cookie == ""


def test_settings_reads_values_from_environment_variables(monkeypatch):
    settings = _clean_settings(
        monkeypatch,
        LITELLM_MODEL="anthropic/claude-sonnet-5",
        LITELLM_API_KEY="sk-ant-test",
        SEARCH_API_URL="https://example.invalid",
    )

    assert settings.litellm_model == "anthropic/claude-sonnet-5"
    assert settings.litellm_api_key == "sk-ant-test"
    assert settings.search_api_url == "https://example.invalid"
    # Untouched fields keep their own defaults -- one field's override
    # doesn't leak into another's.
    assert settings.embedding_model == "intfloat/multilingual-e5-large"


def test_settings_constructor_kwargs_override_environment_variables(monkeypatch):
    monkeypatch.setenv("LITELLM_MODEL", "gemini/gemini-3.8-flash")

    settings = Settings(_env_file=None, litellm_model="anthropic/claude-sonnet-5")

    assert settings.litellm_model == "anthropic/claude-sonnet-5"


def test_settings_ignores_an_unrelated_environment_variable(monkeypatch):
    monkeypatch.setenv("SOME_UNRELATED_TOOLS_OWN_VARIABLE", "whatever")

    # model_config's extra="ignore" -- must not raise on an unrecognized var.
    settings = Settings(_env_file=None)

    assert isinstance(settings, Settings)


def test_settings_rejects_a_non_numeric_timeout(monkeypatch):
    monkeypatch.setenv("LITELLM_TIMEOUT_SECONDS", "not-a-number")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_rejects_a_non_numeric_search_api_timeout(monkeypatch):
    monkeypatch.setenv("SEARCH_API_TIMEOUT_SECONDS", "also-not-a-number")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_get_settings_returns_a_settings_instance():
    assert isinstance(get_settings(), Settings)
