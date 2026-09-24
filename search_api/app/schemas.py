"""Response shapes served by the API — mirrors spec §3a (attestation) and §3c
(aligned example pairs), trimmed to what a client actually needs back rather
than the full indexed record.
"""

from __future__ import annotations

from pydantic import BaseModel


class AttestResult(BaseModel):
    iso_code: str
    form: str
    lemma: str
    sense_id: str
    attestation_year: int
    source: str
    doc_type: str
    donor_language: str | None = None
    first_borrowing_year: int | None = None
    register: str | None = None
    dialect: str | None = None


class AttestQuery(BaseModel):
    lemma: str
    lang: str
    year: int


class AttestResponse(BaseModel):
    query: AttestQuery
    results: list[AttestResult]


class HighlightedSpan(BaseModel):
    token_idx: list[int]
    surface: str


class LookupResult(BaseModel):
    pair_id: str
    target_sentence: str
    # None when the pair's word-alignment step hasn't run yet (spec §6) — the
    # sentence pair still matched on the queried source token, just without a
    # precise span to highlight in the target.
    highlighted_span: HighlightedSpan | None = None
    source_sentence: str
    citation: str
    confidence: float


class LookupQuery(BaseModel):
    text: str
    lang: str
    year: int


class LookupTarget(BaseModel):
    lang: str
    year: int


class LookupResponse(BaseModel):
    query: LookupQuery
    target: LookupTarget
    results: list[LookupResult]


class HealthResponse(BaseModel):
    status: str
    hf_dataset_repo_id: str
