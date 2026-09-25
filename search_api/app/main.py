"""FastAPI app — spec §10 endpoints.

Field-mapping note: the /lookup handler below assumes flattened Parquet
columns (source_tokens, target_tokens, alignment_links, ...) for the §3c
aligned-pair record. The exact column layout the ingestion job (§6) actually
writes hasn't been finalized yet — this is provisional and will need
reconciling once real pair shards exist.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import loader
from .config import get_settings
from .logging_config import get_logger
from .schemas import (
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
)

app = FastAPI(title="ye_olde Search API")
_log = get_logger(__name__)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # The one place this service catches a broad exception: logged in full
    # here (see logging_config.py), then turned into an opaque 500 rather
    # than leaking a stack trace to the client. Anything more specific
    # (a malformed request, a missing shard) should be handled — and, if
    # worth surfacing, logged — at the point it happens instead of relying
    # on this catch-all.
    _log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", hf_dataset_repo_id=settings.hf_dataset_repo_id)


@app.get("/attest", response_model=AttestResponse)
def attest(lemma: str, lang: str, year: int, window: int = 50) -> AttestResponse:
    table = loader.load_relational(lang, year - window, year + window)
    results: list[AttestResult] = []
    if table.num_rows:
        for row in table.to_pylist():
            if row.get("lemma") == lemma:
                results.append(AttestResult(**row))
    return AttestResponse(query=AttestQuery(lemma=lemma, lang=lang, year=year), results=results)


@app.get("/lookup", response_model=LookupResponse)
def lookup(text: str, lang: str, year: int, target_lang: str, target_year: int) -> LookupResponse:
    table = loader.load_pairs(lang, target_lang)
    results: list[LookupResult] = []
    if table.num_rows:
        for row in table.to_pylist():
            # A pair shard's "source"/"target" columns are fixed at
            # ingestion time (whichever direction the alignment job ran),
            # not by which direction *this* query is asking — e.g. a
            # enm->eng shard still has to answer an eng->enm query. Each
            # row carries its own source.lang_code/target.lang_code
            # (spec §3c), so pick which column is the query side per row
            # rather than assuming the shard's stored order matches.
            row_source_lang = (row.get("source") or {}).get("lang_code")
            row_target_lang = (row.get("target") or {}).get("lang_code")
            if row_source_lang == lang:
                query_tokens, query_text_key = row.get("source_tokens") or [], "source_text"
                other_tokens, other_text_key = row.get("target_tokens") or [], "target_text"
                link_query_key, link_other_key = "source_idx", "target_idx"
            elif row_target_lang == lang:
                query_tokens, query_text_key = row.get("target_tokens") or [], "target_text"
                other_tokens, other_text_key = row.get("source_tokens") or [], "source_text"
                link_query_key, link_other_key = "target_idx", "source_idx"
            else:
                continue  # row doesn't actually involve the queried language

            match_idx = {i for i, t in enumerate(query_tokens) if t.lower() == text.lower()}
            if not match_idx:
                continue
            # Prefer a precise word-level highlight when alignment_links
            # exist; fall back to the matched sentence pair with no
            # highlight when they don't (§6's word-alignment step hasn't
            # run on this pair yet) rather than dropping a real match.
            span = None
            for link in row.get("alignment_links") or []:
                if match_idx & set(link[link_query_key]):
                    span_idx = link[link_other_key]
                    span = HighlightedSpan(
                        token_idx=span_idx,
                        surface=" ".join(other_tokens[i] for i in span_idx),
                    )
                    break
            results.append(
                LookupResult(
                    pair_id=row["pair_id"],
                    target_sentence=row[other_text_key],
                    highlighted_span=span,
                    source_sentence=row[query_text_key],
                    citation=row.get("citation") or "",
                    confidence=row.get("sentence_confidence") or 0.0,
                )
            )
    return LookupResponse(
        query=LookupQuery(text=text, lang=lang, year=year),
        target=LookupTarget(lang=target_lang, year=target_year),
        results=results,
    )


@app.get("/search", response_model=SearchResponse)
def search(text: str, lang: str, year: int | None = None, window: int = 50, top_k: int = 5) -> SearchResponse:
    """Semantic passage search (spec §10): ranks indexed passages by
    embedding similarity to `text` rather than requiring an exact token
    match — the RAG-style "find a sentence, not just an exact word" lookup
    /lookup can't do. See loader.search_passages for the retrieval logic.
    """
    results = loader.search_passages(lang, text, year=year, window=window, top_k=top_k)
    return SearchResponse(query=SearchQuery(text=text, lang=lang, year=year), results=results)
