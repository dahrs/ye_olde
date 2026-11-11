"""Algorithmic sentence aligners — the non-LLM matching techniques align.py
builds on. Two independent scoring methods share the same banded,
mutual-nearest-neighbor, monotonic matching skeleton, factored into
`_band_and_match` (itself built on `_band`, `scorer.find_top1_top2`, and
`_longest_increasing_j`) so the two aligners below differ only in how a
candidate pair's score is computed, not in how that score becomes a match:

  - `mutual_nearest_neighbor_align` (embedding-based): the cheap first pass
    of the hybrid pipeline. Conceptually the same idea as Vecalign/
    Bleualign (spec §4): embed both sides, restrict the search to a band
    around the position a monotonic translation would predict (so cost is
    O(n * band), not O(n * m), on a whole book), accept only mutual
    nearest-neighbor matches with a healthy margin over their runner-up (a
    simplified stand-in for LASER-style margin scoring — full margin
    scoring normalizes against each side's k-nearest-neighbor mean; this
    uses the gap to the single runner-up, which is cheaper and,
    empirically, a reasonable proxy), and finally keep only the longest
    run of matches that's monotonic in both indices, dropping the rare
    crossing match these texts essentially never justify.
  - `lexical_align` (character-trigram Dice similarity): the same
    algorithm, scored by surface-character overlap instead of semantic
    embeddings — a classic, pre-neural sentence-alignment technique (no
    model download, no LLM, no network call at all). Meaningful for this
    project specifically because enm/eng are historically related, so a
    genuine translation pair often still shares real cognate spelling
    overlap (þe/the, soþe/soothe) even after centuries of drift — the same
    reasoning `search_api/app/lexical.py`'s hybrid retrieval scoring is
    built on. Used by align.py's `mode="algorithmic"` as a from-scratch,
    no-model-at-all baseline: how much of the hybrid aligner's quality
    actually comes from semantic embeddings/LLM judgment, versus what pure
    lexical overlap alone would already get you.
  - `llm_judgment_align` (LLM-scored): same banded/mutual-nearest-neighbor/
    monotonic skeleton again, scored by asking the configured LLM to judge
    each in-band candidate directly, instead of either metric above.
    `eval/alignment_benchmark_eval.py` found (2026-11-10) that neither
    embedding model tested (`multilingual-e5-large`, the classics-specific
    `bowphs/SPhilBerta`) reliably separates genuine Ancient-Greek<->English
    translations from adversarial near-misses (shared vocabulary, wrong
    meaning), while a plain LLM judgment call did -- at real LLM-call cost,
    unlike the two metrics above, which is why this one is checkpoint-aware
    (see its own docstring) and used by align.py's `mode="llm_judge"`
    deliberately, not as every mode's default.

The two scoring metrics (embedding cosine similarity, trigram Dice
similarity) and the shared margin-acceptance check live in `scorer.py`, not
here — this module owns the matching *algorithm* (banding the search space,
enforcing monotonicity), `scorer.py` owns *how a candidate pair gets a
number and how that number becomes an accept/reject decision*. See
`scorer.py`'s module docstring for why that split exists.

The margin/threshold defaults below are not empirically tuned (spec §9
flags exactly this kind of constant as an open decision) — they're a
starting point to revisit once real aligned output exists to inspect.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..common.llm_client import call_llm_json
from ..common.logging import get_logger
from ..common.ud_tags import parse_confidence
from ..prompt import load_prompt
from .checkpoint import Checkpoint
from .embed import embed_units
from .scorer import char_trigrams, cosine_similarities, dice_coefficient, find_top1_top2

_log = get_logger(__name__)

_LLM_JUDGMENT_SYSTEM_PROMPT = (
    load_prompt("ingest", "llm_judgment_align_system")
    + "\n\n"
    + load_prompt("common", "translation_correspondence_calibration")
)


@dataclass(frozen=True)
class SentenceMatch:
    i: int  # index into units_a
    j: int  # index into units_b
    score: float  # raw score of the matched pair, for reporting: cosine similarity from
    # mutual_nearest_neighbor_align, trigram Dice coefficient from lexical_align — the two
    # scales aren't comparable pair-for-pair, see each function's own docstring


def _band(center_input_idx: int, n_from: int, n_to: int, band_size: int) -> tuple[int, int]:
    center = round(center_input_idx / n_from * n_to) if n_from else 0
    return max(0, center - band_size), min(n_to, center + band_size + 1)


def mutual_nearest_neighbor_align(
    units_a: list[str],
    units_b: list[str],
    *,
    model_name: str | None = None,
    band_ratio: float = 0.1,
    min_band: int = 20,
    margin_threshold: float = 0.05,
) -> list[SentenceMatch]:
    """Returns monotonic, mutually-best-matching (i, j) sentence pairs: `i`
    picks `j` as its best partner within its band, `j` independently picks
    `i` back, and the winning margin over each side's runner-up is at least
    `margin_threshold`. `score` on the returned matches is the raw cosine
    similarity (more interpretable downstream than a margin value), even
    though the margin — not the raw score — is what gated acceptance.
    """
    n_a, n_b = len(units_a), len(units_b)
    if n_a == 0 or n_b == 0:
        return []

    emb_a = embed_units(units_a, model_name=model_name)
    emb_b = embed_units(units_b, model_name=model_name)
    return _band_and_match(
        n_a,
        n_b,
        band_ratio=band_ratio,
        min_band=min_band,
        margin_threshold=margin_threshold,
        score_a_to_b=lambda i, lo, hi: cosine_similarities(emb_a[i], emb_b[lo:hi]),
        score_b_to_a=lambda j, lo, hi: cosine_similarities(emb_b[j], emb_a[lo:hi]),
    )


def _band_and_match(
    n_a: int,
    n_b: int,
    *,
    band_ratio: float,
    min_band: int,
    margin_threshold: float,
    score_a_to_b: Callable[[int, int, int], np.ndarray],
    score_b_to_a: Callable[[int, int, int], np.ndarray],
) -> list[SentenceMatch]:
    """The banded/mutual-nearest-neighbor/margin-gated/monotonic matching
    skeleton shared by both aligners above — the only thing that differs
    between `mutual_nearest_neighbor_align` and `lexical_align` is *how* a
    candidate pair's score is computed (`score_a_to_b`/`score_b_to_a`, each
    called as `(index, lo, hi) -> score array over units_b[lo:hi]` or its
    mirror); everything about turning those scores into accepted,
    monotonic matches — banding, the top1/top2 margin gate, the mutual-pick
    intersection, sorting, and the LIS cleanup — is identical, so it's
    written once here instead of once per aligner. Callers are expected to
    have already handled the `n_a == 0 or n_b == 0` case themselves, before
    doing whatever per-unit precompute (embedding, trigram extraction) this
    function's callbacks close over — this function re-checks it too, as a
    cheap safety net, but does none of that precompute itself.
    """
    if n_a == 0 or n_b == 0:
        return []
    band_size = max(min_band, round(band_ratio * max(n_a, n_b)))

    best_j_for_i: dict[int, tuple[int, float]] = {}
    for i in range(n_a):
        lo, hi = _band(i, n_a, n_b, band_size)
        sims = score_a_to_b(i, lo, hi)
        local_j, top1, top2 = find_top1_top2(sims)
        if top1 - top2 >= margin_threshold:
            best_j_for_i[i] = (lo + local_j, top1)

    best_i_for_j: dict[int, int] = {}
    for j in range(n_b):
        lo, hi = _band(j, n_b, n_a, band_size)
        sims = score_b_to_a(j, lo, hi)
        local_i, top1, top2 = find_top1_top2(sims)
        if top1 - top2 >= margin_threshold:
            best_i_for_j[j] = lo + local_i

    mutual = [
        SentenceMatch(i=i, j=j, score=score) for i, (j, score) in best_j_for_i.items() if best_i_for_j.get(j) == i
    ]
    mutual.sort(key=lambda m: m.i)
    return _longest_increasing_j(mutual)


def _longest_increasing_j(matches: list[SentenceMatch]) -> list[SentenceMatch]:
    """Patience-sorting LIS (by `.j`, input already sorted by `.i`) with
    reconstruction — keeps the longest subsequence of matches whose target
    indices are also strictly increasing, discarding any match that would
    otherwise imply these two editions reorder content relative to one
    another.
    """
    if not matches:
        return []
    tails: list[int] = []  # indices into `matches`; tails[k] = end of the best length-(k+1) run found so far
    predecessor = [-1] * len(matches)
    for idx, m in enumerate(matches):
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if matches[tails[mid]].j < m.j:
                lo = mid + 1
            else:
                hi = mid
        if lo > 0:
            predecessor[idx] = tails[lo - 1]
        if lo == len(tails):
            tails.append(idx)
        else:
            tails[lo] = idx

    chain = []
    k = tails[-1] if tails else -1
    while k != -1:
        chain.append(matches[k])
        k = predecessor[k]
    chain.reverse()
    return chain


def lexical_align(
    units_a: list[str],
    units_b: list[str],
    *,
    band_ratio: float = 0.1,
    min_band: int = 20,
    margin_threshold: float = 0.05,
) -> list[SentenceMatch]:
    """Same banded, mutual-nearest-neighbor, monotonic algorithm as
    `mutual_nearest_neighbor_align`, scored by character-trigram Dice
    similarity (`scorer.char_trigrams`/`scorer.dice_coefficient`) instead of
    embedding cosine similarity — see this module's docstring for why that's
    a meaningful baseline for this project specifically. No model download,
    no LLM, no network call.

    `score` on the returned matches is the raw Dice coefficient — not on
    the same numeric scale as `mutual_nearest_neighbor_align`'s cosine
    similarity, so the two aren't directly comparable pair-for-pair.

    Scoring here is an unvectorized Python-level loop (one `dice_coefficient`
    call per in-band candidate), unlike the embedding aligner's single
    matrix multiply over its whole band — a real, measurable cost at
    whole-book scale, and a known, deliberately-deferred gap: a vectorized
    version needs either an approximate hashed-feature representation
    (trades exact Dice values for speed) or an exact shared-vocabulary
    sparse matrix (more code, a real engineering lift), and this is the
    from-scratch *baseline* mode (`align_corpus_pair`'s default is
    `mode="hybrid"`), not the pipeline's default path — not worth the
    accuracy/complexity tradeoff until someone actually needs `mode=
    "algorithmic"` to run fast at scale.
    """
    n_a, n_b = len(units_a), len(units_b)
    if n_a == 0 or n_b == 0:
        return []

    grams_a = [char_trigrams(u) for u in units_a]
    grams_b = [char_trigrams(u) for u in units_b]
    return _band_and_match(
        n_a,
        n_b,
        band_ratio=band_ratio,
        min_band=min_band,
        margin_threshold=margin_threshold,
        score_a_to_b=lambda i, lo, hi: np.array([dice_coefficient(grams_a[i], grams_b[k]) for k in range(lo, hi)]),
        score_b_to_a=lambda j, lo, hi: np.array([dice_coefficient(grams_b[j], grams_a[k]) for k in range(lo, hi)]),
    )


def _normalize_judgment_reply(reply: object, *, batch_size: int) -> list[dict[str, Any]] | None:
    if batch_size == 1 and isinstance(reply, dict):
        reply = [reply]
    if isinstance(reply, list) and len(reply) == batch_size and all(isinstance(x, dict) for x in reply):
        return reply
    return None


def _judgment_batch_with_repair(prompt: str, *, batch_size: int, model: str | None) -> list[dict[str, Any]]:
    """One LLM call scoring a whole batch of candidates against one source
    unit, with one repair retry on a count/shape mismatch -- the same
    "exactly one repair attempt, then degrade the batch rather than lose
    it" shape as `ye_olde.ingest.survey.classify_links`'s own batch
    repair, applied here to score objects instead of link decisions.
    """
    reply = call_llm_json(prompt, system=_LLM_JUDGMENT_SYSTEM_PROMPT, model=model)
    normalized = _normalize_judgment_reply(reply, batch_size=batch_size)
    if normalized is not None:
        return normalized

    repair_prompt = (
        f"{prompt}\n\n---\n\nYour previous reply was: {reply!r}\n\nThat has the wrong shape. I gave you exactly "
        f"{batch_size} numbered candidates above and need exactly {batch_size} score objects back, same order. "
        "Reply again with ONLY the corrected JSON array, no prose, no markdown code fence."
    )
    reply = call_llm_json(repair_prompt, system=_LLM_JUDGMENT_SYSTEM_PROMPT, model=model)
    normalized = _normalize_judgment_reply(reply, batch_size=batch_size)
    if normalized is not None:
        return normalized

    _log.warning(
        "batch of %d candidates still malformed after repair retry, degrading to score 0.0: %r", batch_size, reply
    )
    return [{"score": 0.0} for _ in range(batch_size)]


def _llm_judgment_scores(
    source: str, candidates: list[str], *, lang_a: str, lang_b: str, model: str | None
) -> list[float]:
    """One LLM call scoring `source` (language `lang_a`) against every one
    of `candidates` (language `lang_b`) -- the batched building block
    `llm_judgment_align`'s banded matching calls once per unit, covering
    that unit's whole band in one call rather than one call per candidate
    pair (the only thing that keeps this tractable at all: a single-pair-
    per-call design would mean `O(n * band)` calls instead of `O(n)`).
    """
    if not candidates:
        return []
    prompt = f"SOURCE ({lang_a}): {source}\n\nCANDIDATES ({lang_b}):\n" + "\n".join(
        f"{i}. {c}" for i, c in enumerate(candidates)
    )
    reply = _judgment_batch_with_repair(prompt, batch_size=len(candidates), model=model)
    return [parse_confidence(item.get("score")) or 0.0 for item in reply]


def llm_judgment_align(
    units_a: list[str],
    units_b: list[str],
    *,
    lang_a: str,
    lang_b: str,
    model: str | None = None,
    band_ratio: float = 0.1,
    min_band: int = 20,
    margin_threshold: float = 0.05,
    checkpoint: Checkpoint | None = None,
) -> list[SentenceMatch]:
    """Same banded, mutual-nearest-neighbor, monotonic algorithm as
    `mutual_nearest_neighbor_align`/`lexical_align`, scored by asking the
    configured LLM to directly judge whether each in-band candidate is a
    genuine translation of the source unit — see this module's docstring
    for why (`eval/alignment_benchmark_eval.py`'s 2026-11-10 finding: this
    scores cross-script pairs far more reliably than either embedding
    model tested), at the cost of a real LLM call per unit (batched over
    that unit's whole band in one call — see `_llm_judgment_scores`).

    `checkpoint`, if given, caches each unit's own batch-scored band under
    a key encoding direction+index (`"a2b:<i>"`/`"b2a:<j>"`) — a real
    requirement here, not a nice-to-have: at real LLM-call cost (unlike the
    two free aligners above), an interrupted run (e.g. hitting an API
    billing limit mid-document) has to resume from its last completed
    unit, never re-paying for ones already scored. A candidate whose batch
    reply stayed malformed after one repair retry gets a score of `0.0`
    (never a crash, never a silently-dropped unit) — see
    `_judgment_batch_with_repair`.

    `lang_a`/`lang_b` are passed through only as a labeling hint in the
    prompt text; nothing in this function's own control flow branches on
    them.
    """
    n_a, n_b = len(units_a), len(units_b)
    if n_a == 0 or n_b == 0:
        return []

    def score_a_to_b(i: int, lo: int, hi: int) -> np.ndarray:
        def compute() -> list[float]:
            return _llm_judgment_scores(units_a[i], units_b[lo:hi], lang_a=lang_a, lang_b=lang_b, model=model)

        scores = checkpoint.get_or_compute(f"a2b:{i}", compute) if checkpoint is not None else compute()
        return np.asarray(scores, dtype=float)

    def score_b_to_a(j: int, lo: int, hi: int) -> np.ndarray:
        def compute() -> list[float]:
            return _llm_judgment_scores(units_b[j], units_a[lo:hi], lang_a=lang_b, lang_b=lang_a, model=model)

        scores = checkpoint.get_or_compute(f"b2a:{j}", compute) if checkpoint is not None else compute()
        return np.asarray(scores, dtype=float)

    return _band_and_match(
        n_a,
        n_b,
        band_ratio=band_ratio,
        min_band=min_band,
        margin_threshold=margin_threshold,
        score_a_to_b=score_a_to_b,
        score_b_to_a=score_b_to_a,
    )
