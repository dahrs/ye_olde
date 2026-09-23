"""Algorithmic (embedding-based) sentence aligner — the cheap first pass of
the hybrid pipeline in align.py. Conceptually the same idea as Vecalign/
Bleualign (spec §4): embed both sides, restrict the search to a band around
the position a monotonic translation would predict (so cost is
O(n * band), not O(n * m), on a whole book), accept only mutual
nearest-neighbor matches with a healthy margin over their runner-up (a
simplified stand-in for LASER-style margin scoring — full margin scoring
normalizes against each side's k-nearest-neighbor mean; this uses the
gap to the single runner-up, which is cheaper and, empirically, a
reasonable proxy), and finally keep only the longest run of matches that's
monotonic in both indices, dropping the rare crossing match these texts
essentially never justify.

The margin/threshold defaults below are not empirically tuned (spec §9
flags exactly this kind of constant as an open decision) — they're a
starting point to revisit once real aligned output exists to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embed import embed_units


@dataclass(frozen=True)
class SentenceMatch:
    i: int  # index into units_a
    j: int  # index into units_b
    score: float  # raw cosine similarity of the matched pair, for reporting


def _band(center_input_idx: int, n_from: int, n_to: int, band_size: int) -> tuple[int, int]:
    center = round(center_input_idx / n_from * n_to) if n_from else 0
    return max(0, center - band_size), min(n_to, center + band_size + 1)


def _top1_top2(sims: np.ndarray) -> tuple[int, float, float]:
    """Returns (local_argmax, top1_score, top2_score); top2 is -inf if the
    band only had one candidate (nothing to compare a margin against).
    """
    if sims.size == 1:
        return 0, float(sims[0]), float("-inf")
    top2_idx = np.argpartition(sims, -2)[-2:]
    ordered = top2_idx[np.argsort(sims[top2_idx])[::-1]]
    return int(ordered[0]), float(sims[ordered[0]]), float(sims[ordered[1]])


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
    band_size = max(min_band, round(band_ratio * max(n_a, n_b)))

    best_j_for_i: dict[int, tuple[int, float]] = {}
    for i in range(n_a):
        lo, hi = _band(i, n_a, n_b, band_size)
        sims = emb_a[i] @ emb_b[lo:hi].T
        local_j, top1, top2 = _top1_top2(sims)
        if top1 - top2 >= margin_threshold:
            best_j_for_i[i] = (lo + local_j, top1)

    best_i_for_j: dict[int, int] = {}
    for j in range(n_b):
        lo, hi = _band(j, n_b, n_a, band_size)
        sims = emb_b[j] @ emb_a[lo:hi].T
        local_i, top1, top2 = _top1_top2(sims)
        if top1 - top2 >= margin_threshold:
            best_i_for_j[j] = lo + local_i

    mutual = [
        SentenceMatch(i=i, j=j, score=score)
        for i, (j, score) in best_j_for_i.items()
        if best_i_for_j.get(j) == i
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
