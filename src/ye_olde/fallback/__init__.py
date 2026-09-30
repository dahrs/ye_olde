"""Native -> loan -> constructed -> temporal-loan fallback chain — spec §2, §3a.

**Implemented today: only the native rung, plus a safe default.** Loan
attestation (gating on `first_borrowing_year`) and the constructed-compound
rung both need §3a relational attestation data (`/attest`) that doesn't
exist yet — no §3b contribution has been ingested (spec §13e: `/attest`
always returns `results: []` today). Building loan/constructed logic
against nothing to query would mean guessing rather than retrieving —
exactly what this project's retrieval-grounded design exists to avoid — so
`resolve_loan`/`resolve_constructed` raise `NotImplementedError` rather than
fabricating a result.

**The native rung is backed by `retrieval.lookup` (`/lookup`), not
`retrieval.search` (`/search`) — a correctness fix, not a style choice.**
An earlier version called `/search` and used a hit's whole `text` field as
the "attested form." `/search` is sentence/passage-level (spec §10: the
RAG-style "find a sentence, not just an exact word" endpoint,
`search_api/app/main.py`'s own docstring) — its results have no per-word
span at all, so that earlier version was feeding whole attested *sentences*
into `generation.generate`'s prompt as if they were single-word
substitutions for one content-word token. `/lookup` (spec §3c) is the
endpoint actually built for this: given a term at one point in a language's
history, it returns aligned pairs with the specific attested word/phrase
highlighted (`highlighted_span.surface`) wherever word-alignment covered
it. A hit with no `highlighted_span` (word-alignment didn't reach that
pair, spec §6) is skipped rather than falling back to its whole sentence —
the same principle, applied consistently: never let a sentence stand in
for a word.

**Trade-off accepted by this fix, and why**: `/lookup` doesn't return a
per-hit year the way `/search` does (spec §10 — it doesn't filter or rank
by year server-side yet either, only one shard exists today), so the
symmetric year-distance-decay preference an earlier version of this
docstring described (closest attestation to `year_b` in either direction)
has no year data to operate on anymore and isn't implemented here. Ranking
falls back to `/lookup`'s own `confidence` (`retrieval.lookup` sorts by
it). Restoring year-aware ranking needs `/lookup` to carry/filter by year
server-side first (§9, a genuine follow-up, not solved here) — a real,
temporarily-lost capability, not silently dropped: flagged here and in the
spec doc rather than left for someone to notice was missing.

`resolve`'s only two statuses now: `"attested"` (a real word-level match
cleared the confidence threshold) or `"anachronism-passthrough"` (nothing
usable found at all — the untranslated source term, spec §2's SENSE
branch). This pipeline still can't distinguish "genuinely anachronistic"
from "not yet in our corpus" — an honest limitation, not a bug.

The constructed-compound rung stays entirely unimplemented per spec §9's
own flag: no concrete method was ever settled on (see
`docs/diachronic-translation-pipeline-plan.md`'s discussion of the
options) — `resolve_constructed` raises rather than guessing at one now.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .. import retrieval

# Below this confidence, a /lookup hit isn't trusted as a genuine attested
# match for the queried term — open, unvalidated constant, same honesty as
# ye_olde.resolver.LAMBDA_MAX (spec §9): needs empirical tuning once real
# query examples exist across more than one corpus. Compared against
# LookupHit.confidence, which itself mixes incompatible scales depending on
# how the underlying pair was aligned (search_api/app/schemas.py's
# LookupResult.confidence comment) — an existing, documented limitation
# this module inherits rather than re-solves.
_NATIVE_MATCH_THRESHOLD = 0.75


class FallbackResult(BaseModel):
    """A fallback-chain resolution — a public function's return value, so a
    `BaseModel` per CLAUDE.md's Pydantic guidance. `status` intentionally
    excludes `"loan"`/`"constructed"` — see module docstring for why those
    two rungs raise instead of ever producing a result.
    """

    status: Literal["attested", "anachronism-passthrough"]
    form: str
    note: str


def resolve(term: str, lang_a: str, year_a: int, lang_b: str, year_b: int, *, top_k: int = 5) -> FallbackResult:
    """Resolves `term` (a span in `lang_a`/`year_a`, already known to
    express a single sense) to its attested word-level form in
    `lang_b`/`year_b` — the native rung of spec §2's fallback chain, the
    only rung with real data behind it today. See module docstring for why
    this calls `retrieval.lookup`, not `retrieval.search`, and for the
    year-aware-ranking capability this currently lacks.
    """
    hits = retrieval.lookup(term, lang_a, year_a, lang_b, year_b)
    for hit in hits[:top_k]:
        if hit.highlighted_span is None:
            continue  # matched the sentence but no word-level span -- never usable as a "form"
        if hit.confidence >= _NATIVE_MATCH_THRESHOLD:
            return FallbackResult(
                status="attested",
                form=hit.highlighted_span,
                note=f"attested via {hit.citation or 'corpus'} lookup (confidence {hit.confidence:.2f})",
            )

    return FallbackResult(
        status="anachronism-passthrough",
        form=term,
        note=(
            f"no attested {lang_b} form found for {lang_a!r} {year_a} term {term!r} in the indexed "
            "corpus — retained the source term. This may mean the concept is genuinely anachronistic, "
            "or simply that the corpus doesn't cover it yet; loan/constructed-compound gating needs "
            "§3a attestation data that doesn't exist yet (see module docstring)."
        ),
    )


def resolve_loan(term: str, lang_a: str, year_a: int, lang_b: str, year_b: int) -> FallbackResult:
    raise NotImplementedError(
        "the loan rung needs first_borrowing_year data from /attest (spec §3a), which has no "
        "real data yet — see search_api §13e"
    )


def resolve_constructed(term: str, lang_a: str, year_a: int, lang_b: str, year_b: int) -> FallbackResult:
    raise NotImplementedError(
        "the constructed-compound rung has no settled method yet — spec §9 flags this as an "
        "open decision, not something to guess at silently"
    )


__all__ = ["FallbackResult", "resolve", "resolve_loan", "resolve_constructed"]
