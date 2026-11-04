"""Unit tests for ingest/embed.py's sentence-embedding wrapper.

`_get_model` does a deferred `from sentence_transformers import
SentenceTransformer` inside its own body (so importing this module doesn't
pull in the real, heavy library) and is `@lru_cache`d by model name -- every
test injects a fake `sentence_transformers` module into `sys.modules`
(same technique `test_llm_client.py` uses for `litellm`) rather than
loading a real multi-GB model, and clears the cache before/after so one
test's fake model instance never leaks into another's assertions.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from ye_olde.config import Settings
from ye_olde.ingest import embed


@pytest.fixture(autouse=True)
def _clear_model_cache():
    embed._get_model.cache_clear()
    yield
    embed._get_model.cache_clear()


@dataclass
class _FakeSentenceTransformer:
    model_name: str
    encode_calls: list[dict[str, Any]] = field(default_factory=list)

    def encode(self, sentences: list[str], **kwargs: Any) -> list[list[float]]:
        self.encode_calls.append({"sentences": sentences, **kwargs})
        return [[float(len(s)), 0.0] for s in sentences]


def _install_fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Injects a fake `sentence_transformers` module whose `SentenceTransformer`
    records every model name it was constructed with (into the returned
    list) and returns a fresh `_FakeSentenceTransformer` each time. Built via
    `SimpleNamespace` rather than a class with a `SentenceTransformer` *method*
    -- the module attribute needs that exact name regardless of the
    underlying callable's own identifier, so a lowercase factory function
    sidesteps pep8-naming's N802 rather than relying on a per-file-ignore.
    """
    constructed_with: list[str] = []

    def _fake_constructor(model_name: str) -> _FakeSentenceTransformer:
        constructed_with.append(model_name)
        return _FakeSentenceTransformer(model_name=model_name)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=_fake_constructor)
    )
    return constructed_with


def test_get_model_constructs_with_the_given_model_name(monkeypatch):
    constructed_with = _install_fake_sentence_transformers(monkeypatch)

    model = embed._get_model("intfloat/multilingual-e5-large")

    assert constructed_with == ["intfloat/multilingual-e5-large"]
    assert model.model_name == "intfloat/multilingual-e5-large"


def test_get_model_caches_the_same_model_name(monkeypatch):
    constructed_with = _install_fake_sentence_transformers(monkeypatch)

    first = embed._get_model("same/model")
    second = embed._get_model("same/model")

    assert constructed_with == ["same/model"]  # constructed only once
    assert first is second


def test_get_model_constructs_separately_for_different_model_names(monkeypatch):
    constructed_with = _install_fake_sentence_transformers(monkeypatch)

    first = embed._get_model("model/a")
    second = embed._get_model("model/b")

    assert constructed_with == ["model/a", "model/b"]
    assert first is not second


def test_get_model_evicts_the_least_recently_used_entry_past_maxsize_two(monkeypatch):
    constructed_with = _install_fake_sentence_transformers(monkeypatch)

    embed._get_model("a")  # cache: [a]
    embed._get_model("b")  # cache: [a, b]
    embed._get_model("c")  # maxsize=2 -- evicts "a" (the least recently used); cache: [b, c]

    assert constructed_with == ["a", "b", "c"]

    embed._get_model("b")  # still cached -- no re-construction
    assert constructed_with == ["a", "b", "c"]

    embed._get_model("a")  # was evicted -- must be rebuilt
    assert constructed_with == ["a", "b", "c", "a"]


def test_embed_units_resolves_explicit_model_name_over_settings(monkeypatch):
    seen_names: list[str] = []

    def fake_get_model(model_name: str) -> _FakeSentenceTransformer:
        seen_names.append(model_name)
        return _FakeSentenceTransformer(model_name=model_name)

    monkeypatch.setattr(embed, "_get_model", fake_get_model)
    monkeypatch.setattr(embed, "get_settings", lambda: Settings(_env_file=None, embedding_model="settings/model"))

    embed.embed_units(["hello"], model_name="explicit/model")

    assert seen_names == ["explicit/model"]


def test_embed_units_falls_back_to_the_configured_embedding_model(monkeypatch):
    seen_names: list[str] = []

    def fake_get_model(model_name: str) -> _FakeSentenceTransformer:
        seen_names.append(model_name)
        return _FakeSentenceTransformer(model_name=model_name)

    monkeypatch.setattr(embed, "_get_model", fake_get_model)
    monkeypatch.setattr(embed, "get_settings", lambda: Settings(_env_file=None, embedding_model="settings/model"))

    embed.embed_units(["hello"])

    assert seen_names == ["settings/model"]


def test_embed_units_prefixes_every_unit_with_the_e5_query_prefix(monkeypatch):
    fake_model = _FakeSentenceTransformer(model_name="m")
    monkeypatch.setattr(embed, "_get_model", lambda model_name: fake_model)

    embed.embed_units(["gladly would I see him", "for sothe"], model_name="m")

    assert fake_model.encode_calls[0]["sentences"] == ["query: gladly would I see him", "query: for sothe"]


def test_embed_units_requests_normalized_embeddings_without_a_progress_bar(monkeypatch):
    fake_model = _FakeSentenceTransformer(model_name="m")
    monkeypatch.setattr(embed, "_get_model", lambda model_name: fake_model)

    embed.embed_units(["one unit"], model_name="m")

    call = fake_model.encode_calls[0]
    assert call["normalize_embeddings"] is True
    assert call["show_progress_bar"] is False


def test_embed_units_returns_a_numpy_array(monkeypatch):
    fake_model = _FakeSentenceTransformer(model_name="m")
    monkeypatch.setattr(embed, "_get_model", lambda model_name: fake_model)

    result = embed.embed_units(["a", "bb"], model_name="m")

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)  # two units, the fake's own 2-dim vectors


def test_embed_units_with_no_units_returns_an_empty_array(monkeypatch):
    fake_model = _FakeSentenceTransformer(model_name="m")
    monkeypatch.setattr(embed, "_get_model", lambda model_name: fake_model)

    result = embed.embed_units([], model_name="m")

    assert isinstance(result, np.ndarray)
    assert len(result) == 0
    assert fake_model.encode_calls[0]["sentences"] == []
