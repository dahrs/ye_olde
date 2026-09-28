"""Lexical + hybrid-scoring primitives for `/lookup` and `/search` — spec
§10's "Hybrid retrieval scoring". Pure functions, no I/O, so every one of
them is directly unit-testable without mocking anything.

Why this exists at all: `multilingual-e5-large`'s training data is
overwhelmingly modern web text — it has essentially no real exposure to
`enm`/`ang` orthography. A purely embedding-based `/search`, and a purely
exact-token `/lookup`, both get systematically worse exactly where the
historical corpus needs them to work best. Character-level lexical matching
doesn't need training data at all, so it doesn't carry that same bias.
Neither signal is "better" in general — they fail in different,
complementary ways — so both endpoints combine them rather than picking one.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

_TRIGRAM_SIZE = 3


def char_trigrams(token: str) -> set[str]:
    """Character trigrams of `token`, lowercased and padded with 2 spaces on
    each side (the standard `pg_trgm`-style convention) so short tokens and
    word-boundary positions still produce a useful number of trigrams.

    Must stay byte-identical to `ye_olde.ingest.scorer.char_trigrams` — this
    is the query-time half of the same definition. The two services don't
    share a dependency tree by design (see `config.py`'s docstring), but
    postings built by one and queried by the other only match if the
    trigram definition itself is identical between them.
    """
    padded = f"  {token.lower()}  "
    return {padded[i : i + _TRIGRAM_SIZE] for i in range(len(padded) - _TRIGRAM_SIZE + 1)}


def token_char_length(ngram_count: int) -> int:
    """Inverse of `char_trigrams`' counting: a token of character length L
    has exactly `L + (_TRIGRAM_SIZE - 1)` trigrams (padding adds
    `_TRIGRAM_SIZE - 1` characters on each side, and the sliding window
    covers `padded_length - _TRIGRAM_SIZE + 1` positions — the algebra
    collapses to that). Lets `lookup_ngram_threshold`'s length-adaptive
    threshold be computed from a stored `token_ngram_count` posting column
    without needing the token's actual surface text on hand.
    """
    return max(0, ngram_count - (_TRIGRAM_SIZE - 1))


def dice_coefficient(a: set[str], b: set[str]) -> float:
    """Dice similarity between two trigram sets: `2*|A∩B| / (|A|+|B|)`.
    0.0 if both are empty (nothing to compare), not a division by zero.
    """
    if not a and not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def levenshtein(a: str, b: str) -> int:
    """Classic edit-distance dynamic program, case-insensitive.
    `O(len(a)*len(b))` — spec §10 is explicit that this only ever runs
    against an already n-gram-narrowed shortlist, never the whole corpus:
    there's no inverted-index equivalent for arbitrary edit distance the way
    there is for n-gram overlap, which is exactly why n-gram matching is the
    candidate-generation step and this is only the refinement step.
    """
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def bm25_score(
    term_freqs: dict[str, int],
    doc_freqs: dict[str, int],
    *,
    doc_len: int,
    avg_doc_len: float,
    corpus_size: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Okapi BM25 score for one document given, for the query's terms: how
    often each appears in this document (`term_freqs`, 0/absent for a term
    that doesn't) and how many documents in the corpus contain it at all
    (`doc_freqs`, same convention). `k1`/`b` use the standard Okapi BM25
    literature defaults (Robertson/Sparck-Jones) — not an arbitrary guess,
    though still untuned against this corpus specifically (spec §9).

    IDF uses the "+1 inside the log" variant (Lucene/Elasticsearch's
    default), not the bare textbook formula — the textbook version goes
    negative for a term in more than half the corpus, which would *penalize*
    a document for containing a common word rather than merely not
    rewarding it; the `+1` keeps IDF positive for any term, including one in
    literally every document. `max(0.0, ...)` is a defensive backstop on
    top of that, not the thing actually doing the work.
    """
    score = 0.0
    for term, tf in term_freqs.items():
        if tf <= 0:
            continue
        df = doc_freqs.get(term, 0)
        idf = max(0.0, math.log((corpus_size - df + 0.5) / (df + 0.5) + 1))
        length_norm = k1 * (1 - b + b * doc_len / avg_doc_len) if avg_doc_len > 0 else k1
        score += idf * (tf * (k1 + 1)) / (tf + length_norm)
    return score


# How much a query's temporal distance from the present discounts trust in
# the semantic (embedding) signal — open, unvalidated (spec §9).
SEMANTIC_SCALE_YEARS = 200.0


def semantic_weight(year: int, *, scale_years: float = SEMANTIC_SCALE_YEARS, now: int | None = None) -> float:
    """How much to trust the semantic (FAISS/embedding) signal vs. the
    lexical (n-gram/BM25) signal for a query at `year` — spec §10's "Hybrid
    retrieval scoring". 1.0 at the present, decaying smoothly toward 0
    further back, on the same saturating-curve family as
    `ye_olde.resolver.compute_lambda` (deliberately — this is the same kind
    of "how much to trust which evidence, given the era" decision, just
    implemented here rather than imported, for the same reason
    `embedding.py` duplicates `ye_olde.ingest.embed` instead of importing
    it). Continuous by year rather than discrete by language code, since
    the effect is real *within* `eng` too (1550s text has no fixed
    orthography either) — not just a proxy for which language code this is.

    `now` is injectable for tests; defaults to the real current year so
    "distance from the present" tracks actual present time rather than a
    hardcoded epoch that would silently go stale.
    """
    current_year = now if now is not None else datetime.now(UTC).year
    distance = max(0, current_year - year)
    return math.exp(-distance / scale_years)


# /lookup's n-gram Dice threshold for candidate refinement (spec §10), as a
# function of combined query+candidate character length rather than a
# single flat number. Originally motivated by þ->th substitutions
# (þe/the, þat/that, ...) scoring *below* a flat 0.5 threshold at every
# length tested, because replacing one character with two shifts every
# trigram after it, disrupting a short word's few trigrams far more than
# an in-place substitution would.
#
# þe/the itself (pair_length=5) turned out to be a dead end, though: it is
# numerically *identical* on every character-level metric (Dice, edit
# distance, both at every normalization tried) to the/we, the/be, and
# the/to — genuinely different, unrelated words. No string-similarity
# formula can accept one without the other, because from a pure character-
# editing standpoint þ->th and w/b/t->th really are the same size of
# change; telling them apart needs knowing *which* character was
# substituted, which no generic distance metric encodes. A per-language
# substitution table (mapping þ->th, etc.) would work, but was explicitly
# rejected — unmaintainable across every language/pair this project will
# eventually support, which defeats the entire point of choosing
# language-agnostic character methods in the first place. So þe/the is an
# accepted loss: this stays permissive enough for þat/that, þis/this,
# þou/thou (pair_length=7, no identical-twin problem — see
# `lookup_max_edit_distance` below for how those specifically stay
# distinguishable from short common-word collisions), without chasing the
# one case that turned out to be provably unreachable this way.
#
# threshold_max is deliberately the same value the original flat threshold
# used (0.5) — long pairs converge back to unchanged behavior, this only
# adds permissiveness at the short end.
LOOKUP_THRESHOLD_MAX = 0.5
LOOKUP_THRESHOLD_K = 10.0


def lookup_ngram_threshold(
    pair_length: int, *, threshold_max: float = LOOKUP_THRESHOLD_MAX, k: float = LOOKUP_THRESHOLD_K
) -> float:
    """Minimum n-gram Dice score for a candidate token to survive
    `/lookup`'s refinement step, given `pair_length` (`len(query) +
    len(candidate_token)`) — see this module's comment above for why this
    is a function of length rather than a constant. Same saturating-curve
    shape as `semantic_weight`/`ye_olde.resolver.compute_lambda` (rising
    toward a ceiling here, rather than decaying — same family, different
    role), for the same "how much to trust this evidence" reasoning.
    """
    if pair_length <= 0:
        return 0.0
    return threshold_max * (1.0 - math.exp(-pair_length / k))


# Companion to lookup_ngram_threshold: the n-gram Dice threshold alone
# wasn't enough. Making it length-adaptive fixed þ->th at longer lengths,
# but at pair_length~5 a flat max_edit_distance=2 turned out to be the real
# problem — 2 edits is up to two-thirds of a 3-letter word, so "the"
# fuzzy-matched most short, common, *unrelated* words (he, we, be, to,
# she, ...) once the Dice gate stopped blocking them, live-verified as
# ~97% of a real corpus matching a single query for "the". Edit distance
# needs the same length-adaptive treatment Dice got, for the same root
# reason: an absolute edit count means something different at every
# length. edit_max=2 is the same ceiling as the original flat cap — long
# pairs converge back to unchanged behavior, same principle as
# threshold_max above.
EDIT_DISTANCE_MAX = 2.0
EDIT_DISTANCE_K = 4.5


def lookup_max_edit_distance(
    pair_length: int, *, edit_max: float = EDIT_DISTANCE_MAX, k: float = EDIT_DISTANCE_K
) -> int:
    """Maximum edit distance still accepted as a match for `/lookup`'s
    refinement step, given `pair_length` — see this module's comment above.
    Rounded to an integer since edit distance itself only ever takes
    integer values; `k=4.5` puts the rounding boundary between
    pair_length=5 (rounds to 1 — rejects þe/the and its identical twins)
    and pair_length=7 (rounds to 2 — keeps þat/that, þis/this, þou/thou).
    """
    if pair_length <= 0:
        return 0
    return round(edit_max * (1.0 - math.exp(-pair_length / k)))


def normalize(scores: list[float]) -> list[float]:
    """Min-max normalizes `scores` to `[0, 1]` so differently-scaled signals
    (cosine similarity, Dice coefficient, BM25) can be linearly blended
    meaningfully (spec §10 — chosen over Reciprocal Rank Fusion specifically
    because it preserves a smoothly-interpretable weighted mix). All-equal
    input (including a single score, or all-zero) normalizes to 1.0 for
    every entry rather than dividing by a zero range — a flat signal
    shouldn't silently zero itself out of a blend just because there's
    nothing to spread it across.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]
