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
    # NER tag from the closed CoNLL-2003 vocabulary (PER/LOC/ORG/MISC/O) —
    # spec §3d/§4: a period name-form lookup is just this endpoint filtered
    # to ner="PER", not a separate registry subsystem. None (not "O") means
    # the ingesting contribution never set it, distinct from an explicit
    # "confirmed not a name" — most rows today, since no §3b contribution
    # carries this field yet.
    ner: str | None = None


class AttestQuery(BaseModel):
    lemma: str
    lang: str
    year: int
    ner: str | None = None


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
    # Straight from the pairs shard's `sentence_confidence` (see
    # ye_olde.ingest.align._finalize_records) — its `sentence_method` sibling
    # field says which of three incompatible scales produced it: embedding
    # cosine similarity (roughly [-1, 1]), trigram Dice coefficient ([0, 1],
    # noisier on short sentences), or LLM-self-reported confidence. Not
    # normalized across those — a client sorting/filtering by `confidence`
    # alone across a corpus indexed by more than one alignment mode gets a
    # meaningless cross-provenance ordering. Flagged, not fixed: reconciling
    # the scales is a real design decision (normalize at index time? expose
    # method alongside confidence and let clients weight it?), not a bug fix.
    confidence: float
    # The matched query-side token's own UD-style tag (spec §3d/§9), straight
    # off the alignment_link whose head word this match landed on — None
    # whenever the match isn't a link's head word, which is most tokens:
    # `ingest.align._build_link` only tags a link's own span, not every
    # token in the sentence (spec §9's "no point extending the index for a
    # signal nothing populates or queries yet" is finally closed here, but
    # only for linked spans). A caller that tagged its own query with
    # `ye_olde.classify.classify` compares that tag against these to tell a
    # genuine same-tag match (e.g. a name matching a name) from a match that
    # only shares spelling — see `/lookup`'s `upos`/`ner` params.
    matched_upos: str | None = None
    matched_ner: str | None = None
    matched_lemma: str | None = None


class LookupQuery(BaseModel):
    text: str
    lang: str
    year: int
    # Echo of the caller's own classify()-produced tag for `text` (spec
    # §3d/§9) — not used to filter results (a query-time tag can be wrong,
    # and most indexed tokens have no tag to compare against at all, see
    # `LookupResult.matched_upos`), only to rank same-tagged matches ahead of
    # untagged ones and untagged ones ahead of actively-disagreeing ones.
    upos: str | None = None
    ner: str | None = None


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


class SearchQuery(BaseModel):
    text: str
    lang: str
    year: int | None = None
    # Echo of the caller's own classify()-produced tag for `text` (spec
    # §3d/§9), same as `LookupQuery.upos`/`.ner` — see `SearchResult.tag_matched`
    # for how it affects ranking.
    upos: str | None = None
    ner: str | None = None


class SearchResult(BaseModel):
    pair_id: str
    lang: str
    year: int
    work: str | None = None
    text: str
    score: float
    # The aligned counterpart passage (spec §3c) — every indexed passage
    # today comes from a pair, so a semantic hit always has one; there's no
    # unpaired, single-language semantic index yet.
    other_lang: str
    other_year: int
    other_work: str | None = None
    other_text: str
    citation: str
    # True when the query's `upos`/`ner` (spec §3d/§9) matched some
    # alignment_link's tag on this passage's query-side — a small ranking
    # boost (`loader._TAG_MATCH_BOOST`), not a filter: /search is
    # passage-level, and most tokens in a passage have no tag to check at
    # all (only a link's head word does, see `LookupResult.matched_upos`),
    # so excluding untagged passages would throw away real matches for no
    # reason. Always False when neither param was given.
    tag_matched: bool = False


class SearchResponse(BaseModel):
    query: SearchQuery
    results: list[SearchResult]
