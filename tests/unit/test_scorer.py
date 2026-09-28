"""Unit tests for the shared cleaning/alignment metrics (spec §6):
character-trigram extraction, Dice similarity, cosine similarity, and the
top1/top2 margin check."""

from __future__ import annotations

import numpy as np
import pytest

from ye_olde.ingest import scorer


def test_char_trigrams_pads_and_lowercases():
    assert scorer.char_trigrams("Sir") == {"  s", " si", "sir", "ir ", "r  "}
    assert scorer.char_trigrams("SIR") == scorer.char_trigrams("sir")


def test_dice_coefficient_identical_sets_is_one():
    grams = scorer.char_trigrams("sothe")
    assert scorer.dice_coefficient(grams, grams) == pytest.approx(1.0)


def test_dice_coefficient_disjoint_sets_is_zero():
    assert scorer.dice_coefficient({"abc"}, {"xyz"}) == 0.0


def test_dice_coefficient_both_empty_is_zero_not_division_error():
    assert scorer.dice_coefficient(set(), set()) == 0.0


def test_cosine_similarities_identical_vector_is_one():
    query = np.array([1.0, 0.0, 0.0])
    candidates = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    sims = scorer.cosine_similarities(query, candidates)
    assert sims[0] == pytest.approx(1.0)
    assert sims[1] == pytest.approx(0.0)


def test_top1_top2_single_candidate_has_no_runner_up():
    idx, top1, top2 = scorer.top1_top2(np.array([0.5]))
    assert idx == 0
    assert top1 == pytest.approx(0.5)
    assert top2 == float("-inf")


def test_top1_top2_orders_by_score_descending():
    idx, top1, top2 = scorer.top1_top2(np.array([0.2, 0.9, 0.6]))
    assert idx == 1
    assert top1 == pytest.approx(0.9)
    assert top2 == pytest.approx(0.6)
