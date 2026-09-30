"""Unit tests for the Search API client + temporal reranking. `httpx.get` is
monkeypatched throughout — none of this should need a live search_api.
"""

from __future__ import annotations

import pytest

from ye_olde import retrieval
from ye_olde.config import Settings


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def _hit(pair_id: str, year: int, score: float, **overrides: object) -> dict:
    base = {
        "pair_id": pair_id,
        "lang": "enm",
        "year": year,
        "work": "Sir Gawayne",
        "text": f"text-{pair_id}",
        "score": score,
        "other_lang": "eng",
        "other_year": year + 500,
        "other_work": "Sir Gawayne",
        "other_text": f"other-{pair_id}",
        "citation": "",
    }
    base.update(overrides)
    return base


def test_search_raises_when_search_api_url_not_set(monkeypatch):
    monkeypatch.setattr(retrieval, "get_settings", lambda: Settings(search_api_url=""))
    with pytest.raises(ValueError, match="SEARCH_API_URL"):
        retrieval.search("gladly", "enm", 1400)


def test_search_requests_year_but_never_window(monkeypatch):
    # `year` alone enables /search's own server-side semantic-vs-lexical
    # weighting (spec §10); `window` is never sent -- year+window together
    # would trigger /search's hard include/exclude filter, which would bias
    # the local-density count this module's own reranking depends on.
    monkeypatch.setattr(retrieval, "get_settings", lambda: Settings(search_api_url="http://localhost:8000"))
    captured = {}

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse({"results": []})

    monkeypatch.setattr(retrieval.httpx, "get", fake_get)
    retrieval.search("gladly", "enm", 1400)

    assert captured["url"] == "http://localhost:8000/search"
    assert captured["params"] == {"text": "gladly", "lang": "enm", "year": 1400, "top_k": retrieval._FETCH_K}
    assert "window" not in captured["params"]


def test_search_reranks_close_year_above_higher_similarity_far_year(monkeypatch):
    # Far-year candidate has higher raw similarity, but with enough local
    # density the temporal penalty should still put the close-year one first.
    hits = [_hit("far", year=1000, score=0.95)] + [_hit(f"near{i}", year=1400, score=0.80) for i in range(400)]
    monkeypatch.setattr(retrieval, "get_settings", lambda: Settings(search_api_url="http://localhost:8000"))
    monkeypatch.setattr(retrieval.httpx, "get", lambda *a, **k: _FakeResponse({"results": hits}))

    results = retrieval.search("gladly", "enm", 1400, top_k=3)
    assert results[0].pair_id.startswith("near")
    assert results[0].year == 1400


def test_search_uses_gentle_lambda_when_nothing_is_local(monkeypatch):
    # Only one candidate exists at all, far from the query -- local_count=0
    # should mean lambda=0, i.e. the far candidate's score is untouched.
    hits = [_hit("only", year=1000, score=0.7)]
    monkeypatch.setattr(retrieval, "get_settings", lambda: Settings(search_api_url="http://localhost:8000"))
    monkeypatch.setattr(retrieval.httpx, "get", lambda *a, **k: _FakeResponse({"results": hits}))

    results = retrieval.search("gladly", "enm", 1400)
    assert len(results) == 1
    assert results[0].score == pytest.approx(0.7)
    assert results[0].similarity == pytest.approx(0.7)


def test_search_truncates_to_top_k_after_reranking(monkeypatch):
    hits = [_hit(f"p{i}", year=1400, score=0.5 + i * 0.01) for i in range(10)]
    monkeypatch.setattr(retrieval, "get_settings", lambda: Settings(search_api_url="http://localhost:8000"))
    monkeypatch.setattr(retrieval.httpx, "get", lambda *a, **k: _FakeResponse({"results": hits}))

    results = retrieval.search("gladly", "enm", 1400, top_k=2)
    assert len(results) == 2
    assert results[0].pair_id == "p9"  # highest raw similarity, all same year


def _lookup_result(pair_id: str, confidence: float, span: str | None = "gladly", **overrides: object) -> dict:
    base = {
        "pair_id": pair_id,
        "target_sentence": "Gladly I would see him",
        "highlighted_span": {"token_idx": [0], "surface": span} if span else None,
        "source_sentence": "gladly would I see him",
        "citation": "",
        "confidence": confidence,
    }
    base.update(overrides)
    return base


def test_lookup_raises_when_search_api_url_not_set(monkeypatch):
    monkeypatch.setattr(retrieval, "get_settings", lambda: Settings(search_api_url=""))
    with pytest.raises(ValueError, match="SEARCH_API_URL"):
        retrieval.lookup("gladly", "enm", 1400, "eng", 1999)


def test_lookup_sends_all_five_query_params(monkeypatch):
    monkeypatch.setattr(retrieval, "get_settings", lambda: Settings(search_api_url="http://localhost:8000"))
    captured = {}

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse({"results": []})

    monkeypatch.setattr(retrieval.httpx, "get", fake_get)
    retrieval.lookup("gladly", "enm", 1400, "eng", 1999)

    assert captured["url"] == "http://localhost:8000/lookup"
    assert captured["params"] == {
        "text": "gladly",
        "lang": "enm",
        "year": 1400,
        "target_lang": "eng",
        "target_year": 1999,
    }


def test_lookup_extracts_highlighted_span_surface(monkeypatch):
    monkeypatch.setattr(retrieval, "get_settings", lambda: Settings(search_api_url="http://localhost:8000"))
    monkeypatch.setattr(retrieval.httpx, "get", lambda *a, **k: _FakeResponse({"results": [_lookup_result("p1", 0.9)]}))

    results = retrieval.lookup("gladly", "enm", 1400, "eng", 1999)
    assert len(results) == 1
    assert results[0].highlighted_span == "gladly"


def test_lookup_preserves_none_when_no_highlighted_span(monkeypatch):
    monkeypatch.setattr(retrieval, "get_settings", lambda: Settings(search_api_url="http://localhost:8000"))
    monkeypatch.setattr(
        retrieval.httpx, "get", lambda *a, **k: _FakeResponse({"results": [_lookup_result("p1", 0.9, span=None)]})
    )

    results = retrieval.lookup("gladly", "enm", 1400, "eng", 1999)
    assert results[0].highlighted_span is None


def test_lookup_sorts_by_confidence_descending(monkeypatch):
    monkeypatch.setattr(retrieval, "get_settings", lambda: Settings(search_api_url="http://localhost:8000"))
    payload = {"results": [_lookup_result("low", 0.3), _lookup_result("high", 0.95), _lookup_result("mid", 0.6)]}
    monkeypatch.setattr(retrieval.httpx, "get", lambda *a, **k: _FakeResponse(payload))

    results = retrieval.lookup("gladly", "enm", 1400, "eng", 1999)
    assert [r.pair_id for r in results] == ["high", "mid", "low"]
