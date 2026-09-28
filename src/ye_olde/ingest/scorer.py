"""Similarity/scoring metrics shared by the cleaning-and-alignment stage of
the ingest pipeline (spec §6): embedding cosine similarity, character-trigram
Dice similarity, and the margin-acceptance check both of `sentence_align.py`'s
aligners are built on.

**Why this module exists.** `sentence_align.py`'s two aligners
(`mutual_nearest_neighbor_align`, scored by embedding cosine similarity, and
`lexical_align`, scored by character-trigram Dice overlap) already shared one
piece of logic outright: the top1-vs-top2 margin check that turns either raw
score into an accept/reject decision. It lived as a private helper in that
module before this one existed. Separately, `index.py`'s own `char_trigrams`
(used to build the `/lookup` fuzzy-match postings at indexing time, spec
§10) was a byte-identical copy of a private trigram-extraction helper
`sentence_align.py` also had — kept separate only because `sentence_align.py`
deliberately avoids `index.py`'s much heavier `pyarrow`/`faiss` imports at
this earlier, lighter pipeline stage, not because the trigram logic itself
differed. That's exactly the "same logic needed by two modules in the same
deployable" case CLAUDE.md's "Avoid duplicate functions" section calls out —
so the metric functions (not the heavier table-building code around them)
move here, into a module with no dependency beyond `numpy`, and both
`sentence_align.py` and `index.py` import them from here instead of each
keeping (or, in `sentence_align.py`'s case, re-deriving) its own copy.

**What deliberately stays out of this module:**
  - `sentence_align._band` / `sentence_align._longest_increasing_j` are the
    matching *algorithm* (search-space banding, monotonicity enforcement)
    built on top of these scores, not a metric themselves — they stay in
    `sentence_align.py`.
  - `index.build_ngram_table` / `index.build_bm25_table` build on-disk
    postings *from* `char_trigrams` for `search_api` to score at query
    time — they're indexing-artifact builders, not metrics computed during
    cleaning/alignment, so they stay in `index.py`.
  - `search_api/app/lexical.py`'s own `char_trigrams`/`dice_coefficient`/
    `bm25_score` stay independently duplicated, on purpose: `search_api` is
    a separately deployed service with its own dependency tree by design
    (see `search_api/app/config.py`'s docstring), so it never imports
    `ye_olde`. That's the one deliberate exception CLAUDE.md's dedup rule
    already carves out — not an oversight this module is meant to fix. The
    trigram *definition* must still stay byte-identical between the two
    copies, or postings built at index time won't match trigrams computed
    at query time (see `char_trigrams` below).
  - The LLM-reported `confidence` field `align.align_block` reads off a
    model's own reply isn't a computed metric — it's a value the model
    reports about its own answer — so it has no home here either.
"""

from __future__ import annotations

import numpy as np

_TRIGRAM_SIZE = 3


def char_trigrams(text: str) -> set[str]:
    """Character trigrams of `text`, lowercased and padded with 2 spaces on
    each side (the standard `pg_trgm`-style convention) so short tokens/units
    and word-boundary positions still produce a useful number of trigrams.

    Used two ways in this pipeline: scoring whole-sentence similarity
    (`dice_coefficient`, via `sentence_align.lexical_align`) and, per-token,
    building the fuzzy-match postings `index.build_ngram_table` writes for
    `search_api`'s `/lookup` to read (spec §10's "Hybrid retrieval
    scoring"). See this module's docstring for why `search_api/app/
    lexical.py` keeps an independent, must-stay-identical copy of this exact
    definition rather than importing it from here.
    """
    padded = f"  {text.lower()}  "
    return {padded[i : i + _TRIGRAM_SIZE] for i in range(len(padded) - _TRIGRAM_SIZE + 1)}


def dice_coefficient(a: set[str], b: set[str]) -> float:
    """Dice similarity between two trigram sets: `2*|A∩B| / (|A|+|B|)`. 0.0
    if both are empty, not a division by zero.
    """
    if not a and not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def cosine_similarities(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Cosine similarity of `query` (one embedding vector) against every row
    of `candidates`, as a plain dot product. Valid only because
    `ye_olde.ingest.embed.embed_units` guarantees L2-normalized output —
    this function does not re-normalize, deliberately, to stay cheap on the
    per-unit-in-a-loop hot path `sentence_align.mutual_nearest_neighbor_align`
    calls it from.
    """
    result: np.ndarray = query @ candidates.T
    return result


def top1_top2(sims: np.ndarray) -> tuple[int, float, float]:
    """Returns `(local_argmax, top1_score, top2_score)` from a 1-D score
    array — `top2` is `-inf` if there was only one candidate to begin with
    (nothing to compare a margin against).

    Shared margin-acceptance primitive for both `sentence_align.py`
    aligners: a candidate is only accepted as a confident match if
    `top1 - top2` clears the caller's `margin_threshold`, regardless of
    which metric above produced the scores — this function knows nothing
    about cosine similarity or Dice coefficients specifically, only about
    turning a score array into a margin-gated decision.
    """
    if sims.size == 1:
        return 0, float(sims[0]), float("-inf")
    top2_idx = np.argpartition(sims, -2)[-2:]
    ordered = top2_idx[np.argsort(sims[top2_idx])[::-1]]
    return int(ordered[0]), float(sims[ordered[0]]), float(sims[ordered[1]])
