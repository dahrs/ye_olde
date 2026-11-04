"""Unit tests for app/embedding.py's /search query embedder.

`_get_model` does a deferred `from sentence_transformers import
SentenceTransformer` inside its own body and is `@lru_cache`d by
`(model_name, hf_token)` -- every test injects a fake `sentence_transformers`
module into `sys.modules` rather than loading a real multi-GB model, and
clears the cache before/after so one test's fake model instance never
leaks into another's assertions.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from app import embedding


@pytest.fixture(autouse=True)
def _clear_model_cache():
    embedding._get_model.cache_clear()
    yield
    embedding._get_model.cache_clear()


@dataclass
class _FakeSentenceTransformer:
    model_name: str
    token: str | None
    encode_calls: list[dict[str, Any]] = field(default_factory=list)

    def encode(self, sentences: list[str], **kwargs: Any) -> list[list[float]]:
        self.encode_calls.append({"sentences": sentences, **kwargs})
        return [[float(len(s)), 0.0] for s in sentences]


def _install_fake_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str | None]]:
    """Injects a fake `sentence_transformers` module into `sys.modules` so
    `_get_model`'s own `from sentence_transformers import SentenceTransformer`
    resolves to a fake constructor instead of loading a real model. Built via
    `SimpleNamespace` rather than a class with a `SentenceTransformer` *method*
    -- the module attribute this needs to expose is named `SentenceTransformer`
    regardless of what the underlying callable's own identifier is, so a
    lowercase factory function sidesteps pep8-naming's N802 instead of
    relying on this directory's own per-file-ignore for it.
    """
    constructed_with: list[tuple[str, str | None]] = []

    def _fake_constructor(model_name: str, token: str | None = None) -> _FakeSentenceTransformer:
        constructed_with.append((model_name, token))
        return _FakeSentenceTransformer(model_name=model_name, token=token)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=_fake_constructor)
    )
    return constructed_with


def test_get_model_constructs_with_the_given_model_name_and_token(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed_with = _install_fake_sentence_transformers(monkeypatch)

    model = embedding._get_model("intfloat/multilingual-e5-large", "hf_real_token")

    assert constructed_with == [("intfloat/multilingual-e5-large", "hf_real_token")]
    assert model.model_name == "intfloat/multilingual-e5-large"


def test_get_model_passes_none_instead_of_an_empty_token(monkeypatch: pytest.MonkeyPatch) -> None:
    # `SentenceTransformer(model_name, token=hf_token or None)` -- an empty
    # string must become None, not be passed through literally (an empty
    # bearer token is not "no token", it's a malformed one).
    constructed_with = _install_fake_sentence_transformers(monkeypatch)

    embedding._get_model("some/model", "")

    assert constructed_with == [("some/model", None)]


def test_get_model_caches_by_the_full_model_name_and_token_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed_with = _install_fake_sentence_transformers(monkeypatch)

    first = embedding._get_model("same/model", "tok-a")
    second = embedding._get_model("same/model", "tok-a")

    assert constructed_with == [("same/model", "tok-a")]  # constructed only once
    assert first is second


def test_get_model_treats_a_different_token_as_a_different_cache_key(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed_with = _install_fake_sentence_transformers(monkeypatch)

    first = embedding._get_model("same/model", "tok-a")
    second = embedding._get_model("same/model", "tok-b")

    assert constructed_with == [("same/model", "tok-a"), ("same/model", "tok-b")]
    assert first is not second


def test_embed_query_defaults_hf_token_to_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str]] = []

    def fake_get_model(model_name: str, hf_token: str) -> _FakeSentenceTransformer:
        seen.append((model_name, hf_token))
        return _FakeSentenceTransformer(model_name=model_name, token=hf_token)

    monkeypatch.setattr(embedding, "_get_model", fake_get_model)

    embedding.embed_query("a query", model_name="m")

    assert seen == [("m", "")]


def test_embed_query_passes_through_an_explicit_hf_token(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str]] = []

    def fake_get_model(model_name: str, hf_token: str) -> _FakeSentenceTransformer:
        seen.append((model_name, hf_token))
        return _FakeSentenceTransformer(model_name=model_name, token=hf_token)

    monkeypatch.setattr(embedding, "_get_model", fake_get_model)

    embedding.embed_query("a query", model_name="m", hf_token="hf_real_token")

    assert seen == [("m", "hf_real_token")]


def test_embed_query_prefixes_the_text_with_the_e5_query_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = _FakeSentenceTransformer(model_name="m", token=None)
    monkeypatch.setattr(embedding, "_get_model", lambda model_name, hf_token: fake_model)

    embedding.embed_query("greetings, sir", model_name="m")

    assert fake_model.encode_calls[0]["sentences"] == ["query: greetings, sir"]


def test_embed_query_requests_normalized_embeddings_without_a_progress_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = _FakeSentenceTransformer(model_name="m", token=None)
    monkeypatch.setattr(embedding, "_get_model", lambda model_name, hf_token: fake_model)

    embedding.embed_query("text", model_name="m")

    call = fake_model.encode_calls[0]
    assert call["normalize_embeddings"] is True
    assert call["show_progress_bar"] is False


def test_embed_query_returns_a_float32_numpy_array(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = _FakeSentenceTransformer(model_name="m", token=None)
    monkeypatch.setattr(embedding, "_get_model", lambda model_name, hf_token: fake_model)

    result = embedding.embed_query("text", model_name="m")

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (1, 2)  # one query, the fake's own 2-dim vector
