"""Unit tests for prompt/loader.py's YAML-backed prompt lookup.

`_load_namespace` is `functools.cache`d by `namespace` name, so every test
below uses a namespace name unique to itself and clears the cache before
and after running -- reusing a namespace string across tests would return
a stale cached result from whichever test ran first, not read the fresh
file *this* test wrote.
"""

from __future__ import annotations

import pytest
import yaml

from ye_olde.prompt import loader


@pytest.fixture(autouse=True)
def _clear_namespace_cache():
    loader._load_namespace.cache_clear()
    yield
    loader._load_namespace.cache_clear()


def _write_namespace(tmp_path, monkeypatch, name: str, contents: dict[str, str]) -> None:
    monkeypatch.setattr(loader, "_PROMPT_DIR", tmp_path)
    (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(contents), encoding="utf-8")


def test_load_prompt_returns_the_stored_text(tmp_path, monkeypatch):
    _write_namespace(tmp_path, monkeypatch, "test_ns_one", {"greet_system": "hello there"})

    assert loader.load_prompt("test_ns_one", "greet_system") == "hello there"


def test_load_prompt_raises_for_a_missing_namespace_file(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "_PROMPT_DIR", tmp_path)

    with pytest.raises(loader.PromptNotFoundError):
        loader.load_prompt("nonexistent_namespace", "whatever_key")


def test_load_prompt_raises_for_a_missing_key_in_an_existing_namespace(tmp_path, monkeypatch):
    _write_namespace(tmp_path, monkeypatch, "test_ns_two", {"real_key": "text"})

    with pytest.raises(loader.PromptNotFoundError):
        loader.load_prompt("test_ns_two", "missing_key")


def test_load_prompt_raises_for_an_empty_namespace_file(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "_PROMPT_DIR", tmp_path)
    (tmp_path / "test_ns_empty.yaml").write_text("", encoding="utf-8")

    with pytest.raises(loader.PromptNotFoundError):
        loader.load_prompt("test_ns_empty", "any_key")


def test_load_prompt_caches_namespace_reads_across_calls(tmp_path, monkeypatch):
    _write_namespace(tmp_path, monkeypatch, "test_ns_cache", {"k": "v1"})
    assert loader.load_prompt("test_ns_cache", "k") == "v1"

    # Rewriting the file on disk must NOT be picked up without a cache
    # clear -- `_load_namespace` is `@cache`d by namespace name, by design
    # (a YAML read on every single `load_prompt` call would be wasted work
    # for a value that's effectively static for the process's lifetime).
    (tmp_path / "test_ns_cache.yaml").write_text(yaml.safe_dump({"k": "v2"}), encoding="utf-8")
    assert loader.load_prompt("test_ns_cache", "k") == "v1"


def test_prompt_not_found_error_is_a_key_error():
    # Raised instead of returning an empty prompt an LLM call would
    # otherwise silently run with (module docstring) -- it's a KeyError
    # subclass so an existing `except KeyError` call site still catches it.
    assert issubclass(loader.PromptNotFoundError, KeyError)
