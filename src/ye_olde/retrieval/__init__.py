"""Search API client + temporal reranking — spec §2, §4, §10.

An `httpx` client against `ye_olde.config.Settings.search_api_url` — never
embeds a vector store, never holds the corpus locally, and never embeds the
query itself either: `search_api/app/embedding.py` does that server-side
(spec §10), so this module sends plain text over HTTP and nothing more.

Two independent layers of temporal weighting are involved here, and it's
worth being precise about which is which:

  - **Server-side** (`search_api`'s own "Hybrid retrieval scoring", spec
    §10): `year` alone (no `window`) tells `/search` how far the query's
    *era* is from the present, so it can weight semantic (FAISS) vs.
    lexical (n-gram/BM25) evidence appropriately — `multilingual-e5-large`
    trusts recent English much more than `enm`/`ang`. This module always
    sends `year` for exactly that reason.
  - **Client-side, here** (`score = similarity * exp(-λ * year_gap)`, spec
    §4/§9): reranks the *results* `/search` already returned by how close
    each individual result's own year is to the query's — with an
    adaptively-chosen λ (`ye_olde.resolver.compute_lambda`), not a fixed
    constant.

`window` is deliberately never sent: `/search` treats `year`+`window`
*together* as a hard include/exclude filter, which would both bias the
local-density count `resolver.compute_lambda` needs and discard exactly the
"nothing closer exists, so widen the net" case λ=0 exists to handle. `year`
alone only feeds the server-side weighting above, no filtering.
"""

from __future__ import annotations

import math
from typing import Any

import httpx
from pydantic import BaseModel

from .. import resolver
from ..config import get_settings

# How many raw candidates to ask /search for before reranking — generous on
# purpose: local-density counting and temporal reranking both need a wide
# semantic-similarity sample to work with, not just the final top_k the
# caller actually wants back. Open decision, same spirit as
# resolver.LOCAL_WINDOW_YEARS.
_FETCH_K = 100


class RankedPassage(BaseModel):
    pair_id: str
    lang: str
    year: int
    work: str | None
    text: str
    other_lang: str
    other_year: int
    other_work: str | None
    other_text: str
    citation: str
    similarity: float  # the Search API's own score, before reranking
    score: float  # similarity * exp(-λ * year_gap) — what results are sorted by


class LookupHit(BaseModel):
    """One `/lookup` result (spec §3c) — a Linguee-style aligned pair, with
    the specific attested word/phrase highlighted where word-alignment
    covered it. `highlighted_span` is the actual word-level attested form;
    `target_sentence`/`source_sentence` are the surrounding sentence, for
    context only — `fallback.resolve` (the reason this exists) must never
    use a whole sentence as if it were a single word's attested form (an
    earlier version of `fallback.resolve` did exactly that against
    `/search`'s sentence-level results — this is why `/lookup`, not
    `/search`, backs the native rung).
    """

    pair_id: str
    target_sentence: str
    highlighted_span: str | None  # None when word-alignment hasn't covered this pair yet (spec §6)
    source_sentence: str
    citation: str
    confidence: float


def lookup(text: str, lang: str, year: int, target_lang: str, target_year: int) -> list[LookupHit]:
    """Linguee-style aligned-example lookup against the Search API (spec
    §10's `/lookup`) — given `text` at `(lang, year)`, returns attested
    pairs whose target side is `(target_lang, target_year)`, each carrying
    the specific attested word/phrase highlighted (when word-alignment
    covered it) rather than only the surrounding sentence. Results are
    sorted by `confidence`, highest first — `/lookup` itself doesn't
    guarantee an order. `/lookup` doesn't yet filter by year server-side
    (only one shard exists today, spec §13e) — `year`/`target_year` are
    still sent (the response echoes them) but aren't a real filter yet;
    this is a known, existing limitation of `/lookup` itself, not
    something this client works around.
    """
    settings = get_settings()
    if not settings.search_api_url:
        raise ValueError("SEARCH_API_URL is not set — see .env.example")

    response = httpx.get(
        f"{settings.search_api_url}/lookup",
        params={"text": text, "lang": lang, "year": year, "target_lang": target_lang, "target_year": target_year},
        timeout=settings.search_api_timeout_seconds,
    )
    response.raise_for_status()
    hits: list[dict[str, Any]] = response.json()["results"]

    parsed = [
        LookupHit(
            pair_id=hit["pair_id"],
            target_sentence=hit["target_sentence"],
            highlighted_span=(hit.get("highlighted_span") or {}).get("surface"),
            source_sentence=hit["source_sentence"],
            citation=hit["citation"],
            confidence=hit["confidence"],
        )
        for hit in hits
    ]
    parsed.sort(key=lambda h: h.confidence, reverse=True)
    return parsed


def search(text: str, lang: str, year: int, *, top_k: int = 5) -> list[RankedPassage]:
    """Semantic search against the Search API (spec §10's `/search`),
    reranked by temporal proximity to `year` with an adaptively-chosen decay
    rate (see `ye_olde.resolver.compute_lambda` for why it's adaptive rather
    than fixed).
    """
    settings = get_settings()
    if not settings.search_api_url:
        raise ValueError("SEARCH_API_URL is not set — see .env.example")

    response = httpx.get(
        f"{settings.search_api_url}/search",
        # `year` (no `window`) enables /search's own server-side semantic-
        # vs-lexical weighting without triggering its hard filter — see this
        # module's docstring for why the two are kept separate.
        params={"text": text, "lang": lang, "year": year, "top_k": _FETCH_K},
        timeout=settings.search_api_timeout_seconds,
    )
    response.raise_for_status()
    hits: list[dict[str, Any]] = response.json()["results"]

    local_count = sum(1 for hit in hits if abs(hit["year"] - year) <= resolver.LOCAL_WINDOW_YEARS)
    lam = resolver.compute_lambda(local_count)

    ranked = []
    for hit in hits:
        similarity = hit["score"]
        year_gap = abs(hit["year"] - year)
        ranked.append(
            RankedPassage(
                pair_id=hit["pair_id"],
                lang=hit["lang"],
                year=hit["year"],
                work=hit.get("work"),
                text=hit["text"],
                other_lang=hit["other_lang"],
                other_year=hit["other_year"],
                other_work=hit.get("other_work"),
                other_text=hit["other_text"],
                citation=hit["citation"],
                similarity=similarity,
                score=similarity * math.exp(-lam * year_gap),
            )
        )
    ranked.sort(key=lambda p: p.score, reverse=True)
    return ranked[:top_k]
