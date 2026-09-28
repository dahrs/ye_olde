"""Unit tests for the algorithmic sentence aligners. `embed_units` is
monkeypatched to return hand-built vectors for the embedding-based aligner's
tests, so those run with no model download and no network access;
`lexical_align`'s tests need no mocking at all (pure character comparison).
"""

from __future__ import annotations

import numpy as np
import pytest

from ye_olde.ingest import sentence_align
from ye_olde.ingest.sentence_align import lexical_align


def test_mutual_nearest_neighbor_align_clean_case(monkeypatch):
    # a_i and b_i are identical one-hot vectors -> perfect, unambiguous
    # matches at every aligned index; b4 has no counterpart in a.
    vectors = {
        "a0": [1, 0, 0, 0, 0],
        "a1": [0, 1, 0, 0, 0],
        "a2": [0, 0, 1, 0, 0],
        "a3": [0, 0, 0, 1, 0],
        "b0": [1, 0, 0, 0, 0],
        "b1": [0, 1, 0, 0, 0],
        "b2": [0, 0, 1, 0, 0],
        "b3": [0, 0, 0, 1, 0],
        "b4": [0, 0, 0, 0, 1],
    }
    monkeypatch.setattr(
        sentence_align,
        "embed_units",
        lambda units, **k: np.array([vectors[u] for u in units], dtype=float),
    )

    matches = sentence_align.mutual_nearest_neighbor_align(["a0", "a1", "a2", "a3"], ["b0", "b1", "b2", "b3", "b4"])

    assert [(m.i, m.j) for m in matches] == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert all(m.score == pytest.approx(1.0) for m in matches)


def test_mutual_nearest_neighbor_align_rejects_low_margin(monkeypatch):
    # a0 is equidistant from b0 and b1 (margin 0) -> should not be accepted
    # as a confident match at the default margin threshold.
    vectors = {
        "a0": [0.7, 0.7, 0, 0],
        "b0": [1, 0, 0, 0],
        "b1": [0, 1, 0, 0],
    }
    monkeypatch.setattr(
        sentence_align,
        "embed_units",
        lambda units, **k: np.array([vectors[u] for u in units], dtype=float),
    )

    matches = sentence_align.mutual_nearest_neighbor_align(["a0"], ["b0", "b1"], margin_threshold=0.05)
    assert matches == []


def test_mutual_nearest_neighbor_align_drops_crossing_match(monkeypatch):
    # a0 best-matches b1 and a1 best-matches b0 (a genuine crossing) -- the
    # longest-increasing-subsequence cleanup must keep only one of the two
    # rather than emit a non-monotonic pair of matches.
    vectors = {
        "a0": [0, 1, 0],
        "a1": [1, 0, 0],
        "b0": [1, 0, 0],
        "b1": [0, 1, 0],
    }
    monkeypatch.setattr(
        sentence_align,
        "embed_units",
        lambda units, **k: np.array([vectors[u] for u in units], dtype=float),
    )

    matches = sentence_align.mutual_nearest_neighbor_align(["a0", "a1"], ["b0", "b1"])
    js = [m.j for m in matches]
    assert js == sorted(js)
    assert len(matches) == 1


def test_empty_inputs_return_no_matches(monkeypatch):
    monkeypatch.setattr(sentence_align, "embed_units", lambda units, **k: np.zeros((len(units), 3)))
    assert sentence_align.mutual_nearest_neighbor_align([], ["b0"]) == []
    assert sentence_align.mutual_nearest_neighbor_align(["a0"], []) == []


def test_lexical_align_matches_near_identical_strings():
    # a0/b0 and a1/b1 share almost all their characters (cognate-like
    # overlap); a2 has no real counterpart on the b side at all.
    units_a = ["the sothe light of day", "many wordes were spoken", "no match here at all"]
    units_b = ["the soothe light of day", "many words were spoken", "something totally unrelated"]

    matches = lexical_align(units_a, units_b)

    assert [(m.i, m.j) for m in matches] == [(0, 0), (1, 1)]
    assert all(0.0 < m.score <= 1.0 for m in matches)


def test_lexical_align_rejects_low_margin():
    # a0 is equally (dis)similar to both b0 and b1 -> no confident winner.
    matches = lexical_align(["abc"], ["xyz", "zyx"], margin_threshold=0.05)
    assert matches == []


def test_lexical_align_drops_crossing_match():
    # a0 best-matches b1 and a1 best-matches b0 -- a genuine crossing that
    # the longest-increasing-subsequence cleanup must resolve to one match.
    units_a = ["alpha beta", "gamma delta"]
    units_b = ["gamma delta", "alpha beta"]

    matches = lexical_align(units_a, units_b)
    js = [m.j for m in matches]
    assert js == sorted(js)
    assert len(matches) <= 1


def test_lexical_align_empty_inputs_return_no_matches():
    assert lexical_align([], ["b0"]) == []
    assert lexical_align(["a0"], []) == []
