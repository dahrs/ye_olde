"""Unit tests for fallback/: `retrieval.lookup` is monkeypatched throughout
— none of this should need a live Search API, matching test_retrieval.py's
own mocking of httpx one layer down.
"""

from __future__ import annotations

import pytest

from ye_olde import fallback
from ye_olde.retrieval import LookupHit


def _hit(span: str | None, confidence: float, **overrides: object) -> LookupHit:
    base: dict[str, object] = {
        "pair_id": "p1",
        "target_sentence": "sentence containing the word",
        "highlighted_span": span,
        "source_sentence": "source sentence",
        "citation": "Sir Gawayne",
        "confidence": confidence,
    }
    base.update(overrides)
    return LookupHit(**base)


def test_resolve_returns_the_highlighted_word_not_the_whole_sentence(monkeypatch):
    # The bug this guards against: an earlier version used a whole
    # /search hit's sentence text as if it were a single word's attested
    # form. /lookup's highlighted_span is the actual word-level form.
    monkeypatch.setattr(fallback.retrieval, "lookup", lambda *a, **k: [_hit("gaeth", 0.9)])
    result = fallback.resolve("goes", "eng", 2000, "enm", 1400)
    assert result.status == "attested"
    assert result.form == "gaeth"


def test_resolve_skips_a_hit_with_no_highlighted_span(monkeypatch):
    # Word-alignment hasn't covered this pair (spec §6) -- no word-level
    # form to extract, so it must never fall back to the whole sentence.
    hits = [_hit(None, 0.95), _hit("gaeth", 0.8)]
    monkeypatch.setattr(fallback.retrieval, "lookup", lambda *a, **k: hits)
    result = fallback.resolve("goes", "eng", 2000, "enm", 1400)
    assert result.status == "attested"
    assert result.form == "gaeth"


def test_resolve_skips_low_confidence_hits(monkeypatch):
    hits = [_hit("weak-match", 0.2), _hit("gaeth", 0.9)]
    monkeypatch.setattr(fallback.retrieval, "lookup", lambda *a, **k: hits)
    result = fallback.resolve("goes", "eng", 2000, "enm", 1400)
    assert result.form == "gaeth"


def test_resolve_falls_back_to_source_term_when_only_unusable_hits_exist(monkeypatch):
    # A hit with no span and a hit below threshold -- neither usable.
    hits = [_hit(None, 0.99), _hit("weak", 0.1)]
    monkeypatch.setattr(fallback.retrieval, "lookup", lambda *a, **k: hits)
    result = fallback.resolve("television", "eng", 2000, "enm", 1400)
    assert result.status == "anachronism-passthrough"
    assert result.form == "television"


def test_resolve_falls_back_to_source_term_when_no_hits_at_all(monkeypatch):
    monkeypatch.setattr(fallback.retrieval, "lookup", lambda *a, **k: [])
    result = fallback.resolve("television", "eng", 2000, "enm", 1400)
    assert result.status == "anachronism-passthrough"
    assert result.form == "television"


def test_resolve_passes_through_both_language_year_pairs(monkeypatch):
    captured = {}

    def fake_lookup(text, lang, year, target_lang, target_year):
        captured.update(text=text, lang=lang, year=year, target_lang=target_lang, target_year=target_year)
        return []

    monkeypatch.setattr(fallback.retrieval, "lookup", fake_lookup)
    fallback.resolve("light", "enm", 1400, "eng", 1999)
    assert captured == {"text": "light", "lang": "enm", "year": 1400, "target_lang": "eng", "target_year": 1999}


def test_resolve_loan_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="first_borrowing_year"):
        fallback.resolve_loan("term", "eng", 2000, "enm", 1400)


def test_resolve_constructed_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="constructed-compound"):
        fallback.resolve_constructed("term", "eng", 2000, "enm", 1400)
