"""FastAPI app — spec §10 endpoints. See loader.py for the actual matching/
ranking logic (`lookup_passages`, `search_passages`, spec §10's "Hybrid
retrieval scoring") — the handlers below just shape requests/responses.
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
    LookupQuery,
    LookupResponse,
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
def attest(lemma: str, lang: str, year: int, window: int = 50, ner: str | None = None) -> AttestResponse:
    """Exact/relational lookup (spec §3a). `ner`, when given, additionally
    filters to rows tagged that way — this is also how a period name-form
    lookup works (spec §3d/§4: `ner=PER` finds attested spellings of a
    given name the same way an untagged query finds a common word's
    attested spelling; there's no separate name-registry endpoint or code
    path).
    """
    table = loader.load_relational(lang, year - window, year + window)
    results: list[AttestResult] = []
    if table.num_rows:
        for row in table.to_pylist():
            if row.get("lemma") != lemma:
                continue
            # `ner` truthy, not just "is not None": a client sending
            # `?ner=` (empty string) means "no filter", the same as
            # omitting the param entirely -- treating "" as a literal
            # value to match against would silently return zero results,
            # since no row's ner is ever the empty string (None, "O", or a
            # real tag, never "").
            if ner and row.get("ner") != ner:
                continue
            results.append(AttestResult(**row))
    return AttestResponse(query=AttestQuery(lemma=lemma, lang=lang, year=year, ner=ner), results=results)


@app.get("/lookup", response_model=LookupResponse)
def lookup(
    text: str,
    lang: str,
    year: int,
    target_lang: str,
    target_year: int,
    upos: str | None = None,
    ner: str | None = None,
) -> LookupResponse:
    """`upos`/`ner`, when given (spec §3d/§9 — typically the caller's own
    `ye_olde.classify.classify` tag for `text`), rank a result whose
    matched token carries the same tag ahead of an untagged or
    disagreeing one; see `loader.lookup_passages`/`loader._tag_preference_rank`.
    They never filter — most indexed tokens have no tag to compare at all.
    """
    results = loader.lookup_passages(text, lang, target_lang, upos=upos, ner=ner)
    return LookupResponse(
        query=LookupQuery(text=text, lang=lang, year=year, upos=upos, ner=ner),
        target=LookupTarget(lang=target_lang, year=target_year),
        results=results,
    )


@app.get("/search", response_model=SearchResponse)
def search(
    text: str,
    lang: str,
    year: int | None = None,
    window: int | None = None,
    top_k: int = 5,
    upos: str | None = None,
    ner: str | None = None,
) -> SearchResponse:
    """Semantic + lexical hybrid passage search (spec §10's "Hybrid
    retrieval scoring") — the RAG-style "find a sentence, not just an exact
    word" lookup /lookup can't do. `year` (without `window`) weights the
    blend by how far that era is from the present; `year`+`window` together
    also hard-filter results outside that range, same as before. `upos`/
    `ner` (spec §3d/§9) add a small ranking boost to a passage with a
    matching tagged token, never a filter — see `loader.search_passages`
    for the full scoring logic.
    """
    results = loader.search_passages(lang, text, year=year, window=window, top_k=top_k, upos=upos, ner=ner)
    return SearchResponse(query=SearchQuery(text=text, lang=lang, year=year, upos=upos, ner=ner), results=results)
