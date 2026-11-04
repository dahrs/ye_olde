"""Unit tests for app/schemas.py's response/query models.

These are plain `pydantic.BaseModel`s with no custom validators or
methods -- what's actually worth verifying per model is its validation
*contract*: which fields are required vs. optional-with-a-default, that
nested models compose correctly, and that a malformed value for a typed
field is rejected rather than silently coerced.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import (
    AttestQuery,
    AttestResponse,
    AttestResult,
    HealthResponse,
    HighlightedSpan,
    LookupQuery,
    LookupResponse,
    LookupResult,
    LookupTarget,
    SearchQuery,
    SearchResponse,
    SearchResult,
)


def test_attest_result_requires_its_core_fields_and_defaults_the_optional_ones() -> None:
    result = AttestResult(
        iso_code="enm",
        form="gladly",
        lemma="gladly",
        sense_id="en:gladly.adv.01",
        attestation_year=1400,
        source="Sir Gawayne",
        doc_type="attestation",
    )

    assert result.donor_language is None
    assert result.first_borrowing_year is None
    assert result.register is None
    assert result.dialect is None
    assert result.ner is None


def test_attest_result_rejects_a_missing_required_field() -> None:
    with pytest.raises(ValidationError):
        AttestResult.model_validate({"iso_code": "enm", "form": "gladly"})  # missing lemma/sense_id/.../doc_type


def test_attest_query_defaults_ner_to_none() -> None:
    query = AttestQuery(lemma="gladly", lang="enm", year=1400)

    assert query.ner is None


def test_attest_response_nests_a_list_of_results() -> None:
    response = AttestResponse(
        query=AttestQuery(lemma="gladly", lang="enm", year=1400),
        results=[
            AttestResult(
                iso_code="enm",
                form="gladly",
                lemma="gladly",
                sense_id="en:gladly.adv.01",
                attestation_year=1400,
                source="Sir Gawayne",
                doc_type="attestation",
            )
        ],
    )

    assert response.query.lemma == "gladly"
    assert len(response.results) == 1
    assert response.results[0].form == "gladly"


def test_highlighted_span_requires_both_fields() -> None:
    span = HighlightedSpan(token_idx=[3], surface="sothe")

    assert span.token_idx == [3]
    assert span.surface == "sothe"
    with pytest.raises(ValidationError):
        HighlightedSpan.model_validate({"surface": "sothe"})  # missing token_idx


def test_lookup_result_defaults_optional_fields_to_none() -> None:
    result = LookupResult(
        pair_id="p1",
        target_sentence="Gladly I would see him",
        source_sentence="gladly would I see him",
        citation="Sir Gawayne 1:1",
        confidence=0.9,
    )

    assert result.highlighted_span is None
    assert result.matched_upos is None
    assert result.matched_ner is None
    assert result.matched_lemma is None


def test_lookup_result_nests_a_highlighted_span_when_given() -> None:
    result = LookupResult(
        pair_id="p1",
        target_sentence="Gladly I would see him",
        source_sentence="gladly would I see him",
        citation="Sir Gawayne 1:1",
        confidence=0.9,
        highlighted_span=HighlightedSpan(token_idx=[0], surface="gladly"),
    )

    assert result.highlighted_span is not None
    assert result.highlighted_span.surface == "gladly"


def test_lookup_result_rejects_a_non_numeric_confidence() -> None:
    with pytest.raises(ValidationError):
        LookupResult.model_validate(
            {
                "pair_id": "p1",
                "target_sentence": "x",
                "source_sentence": "y",
                "citation": "z",
                "confidence": "not-a-number",
            }
        )


def test_lookup_query_defaults_tag_fields_to_none() -> None:
    query = LookupQuery(text="gladly", lang="enm", year=1400)

    assert query.upos is None
    assert query.ner is None


def test_lookup_response_nests_query_target_and_results() -> None:
    response = LookupResponse(
        query=LookupQuery(text="gladly", lang="enm", year=1400),
        target=LookupTarget(lang="eng", year=1999),
        results=[],
    )

    assert response.query.text == "gladly"
    assert response.target.lang == "eng"
    assert response.results == []


def test_health_response_requires_both_fields() -> None:
    health = HealthResponse(status="ok", hf_dataset_repo_id="dahrs/ye_olde_data-index")

    assert health.status == "ok"
    with pytest.raises(ValidationError):
        HealthResponse.model_validate({"status": "ok"})  # missing hf_dataset_repo_id


def test_search_query_year_and_tags_are_optional() -> None:
    query = SearchQuery(text="greetings, sir", lang="enm")

    assert query.year is None
    assert query.upos is None
    assert query.ner is None


def test_search_result_defaults_optional_fields_to_none() -> None:
    result = SearchResult(
        pair_id="p1",
        lang="enm",
        year=1400,
        text="gladly would I see him",
        score=0.8,
        other_lang="eng",
        other_year=1999,
        other_text="Gladly I would see him",
        citation="Sir Gawayne 1:1",
    )

    assert result.work is None
    assert result.other_work is None
    assert result.matched_upos is None


def test_search_response_nests_query_and_results() -> None:
    response = SearchResponse(
        query=SearchQuery(text="greetings, sir", lang="enm"),
        results=[
            SearchResult(
                pair_id="p1",
                lang="enm",
                year=1400,
                text="gladly would I see him",
                score=0.8,
                other_lang="eng",
                other_year=1999,
                other_text="Gladly I would see him",
                citation="Sir Gawayne 1:1",
            )
        ],
    )

    assert response.query.text == "greetings, sir"
    assert response.results[0].score == 0.8
