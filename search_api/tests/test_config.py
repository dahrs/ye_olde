"""Unit tests for app/config.py's `Settings`/`get_settings`.

Every default-value test passes `_env_file=None` and clears the specific
environment variables it cares about first -- this directory has a real,
populated `.env` (gitignored), so a test that let `Settings()` read it by
accident would see whatever that file happens to contain instead of this
module's own documented defaults, and would pass or fail depending on
local machine state rather than the code under test.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings, get_settings

_ENV_VARS = ("HF_DATASET_REPO_ID", "HF_TOKEN", "HF_CACHE_DIR", "EMBEDDING_MODEL")


def _clean_settings(monkeypatch: pytest.MonkeyPatch, **env_overrides: str) -> Settings:
    for key in _ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env_overrides.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_settings_defaults_when_nothing_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _clean_settings(monkeypatch)

    assert settings.hf_dataset_repo_id == ""
    assert settings.hf_token == ""
    assert settings.hf_cache_dir == "/tmp/hf-cache"
    assert settings.embedding_model == "intfloat/multilingual-e5-large"


def test_settings_embedding_model_default_matches_ye_oldes_own_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Deliberately kept identical, not imported (config.py's own docstring):
    # a mismatch wouldn't error, it would silently produce meaningless
    # similarity scores against shards built with the other default. This
    # test is the one place that's allowed to cross the two services' own
    # "never depend on each other" boundary, purely to check the two
    # hardcoded literals haven't drifted apart.
    from ye_olde.config import Settings as YeOldeSettings

    settings = _clean_settings(monkeypatch)

    assert settings.embedding_model == YeOldeSettings(_env_file=None).embedding_model


def test_settings_reads_values_from_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _clean_settings(
        monkeypatch,
        HF_DATASET_REPO_ID="dahrs/ye_olde_data-index",
        HF_TOKEN="hf_test_token",
        HF_CACHE_DIR="/var/cache/hf",
    )

    assert settings.hf_dataset_repo_id == "dahrs/ye_olde_data-index"
    assert settings.hf_token == "hf_test_token"
    assert settings.hf_cache_dir == "/var/cache/hf"
    # Untouched field keeps its own default.
    assert settings.embedding_model == "intfloat/multilingual-e5-large"


def test_settings_constructor_kwargs_override_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_DATASET_REPO_ID", "from/env")

    settings = Settings(_env_file=None, hf_dataset_repo_id="from/kwarg")

    assert settings.hf_dataset_repo_id == "from/kwarg"


def test_settings_ignores_an_unrelated_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_UNRELATED_TOOLS_OWN_VARIABLE", "whatever")

    settings = Settings(_env_file=None)  # extra="ignore" -- must not raise

    assert isinstance(settings, Settings)


def test_settings_rejects_a_non_string_assigned_to_a_string_field(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, hf_dataset_repo_id={"not": "a string"})  # type: ignore[arg-type]


def test_get_settings_returns_a_settings_instance() -> None:
    assert isinstance(get_settings(), Settings)
