"""Unit tests for the algorithmic sentence aligners. `embed_units` is
monkeypatched to return hand-built vectors for the embedding-based aligner's
tests, so those run with no model download and no network access;
`lexical_align`'s tests need no mocking at all (pure character comparison);
`llm_judgment_align`'s tests monkeypatch `call_llm_json` the same way
`test_classify.py`/`test_survey.py` do for their own LLM-backed functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from ye_olde.ingest import sentence_align
from ye_olde.ingest.checkpoint import Checkpoint
from ye_olde.ingest.sentence_align import lexical_align, llm_judgment_align


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


# ---------------------------------------------------------------------------
# llm_judgment_align
# ---------------------------------------------------------------------------

# Scores keyed by (source_text, candidate_text) -- the fake LLM below looks
# each candidate up here instead of doing any real judgment.
_GRC_ENG_SCORES: dict[tuple[str, str], float] = {
    ("a0", "b0"): 0.95,
    ("a0", "b1"): 0.1,
    ("a1", "b0"): 0.1,
    ("a1", "b1"): 0.9,
    ("b0", "a0"): 0.95,
    ("b0", "a1"): 0.1,
    ("b1", "a0"): 0.1,
    ("b1", "a1"): 0.9,
}


def _fake_judgment_call(source_lookup: dict[tuple[str, str], float]):
    def fake_call_llm_json(prompt: str, *, system: str, model: str | None = None) -> list[dict[str, float]]:
        # The prompt always starts "SOURCE (<lang>): <text>\n\nCANDIDATES (<lang>):\n0. <c0>\n1. <c1>...".
        source_line, _, candidates_block = prompt.partition("\n\nCANDIDATES")
        source_text = source_line.split(": ", 1)[1]
        candidate_lines = candidates_block.splitlines()[1:]
        candidates = [line.split(". ", 1)[1] for line in candidate_lines if line.strip()]
        return [{"score": source_lookup.get((source_text, c), 0.0)} for c in candidates]

    return fake_call_llm_json


def test_llm_judgment_align_clean_case(monkeypatch):
    monkeypatch.setattr(sentence_align, "call_llm_json", _fake_judgment_call(_GRC_ENG_SCORES))

    matches = llm_judgment_align(["a0", "a1"], ["b0", "b1"], lang_a="grc", lang_b="eng")

    assert [(m.i, m.j) for m in matches] == [(0, 0), (1, 1)]


def test_llm_judgment_align_rejects_low_margin(monkeypatch):
    # a0 scores identically against both candidates -> no confident winner.
    monkeypatch.setattr(sentence_align, "call_llm_json", _fake_judgment_call({("a0", "b0"): 0.5, ("a0", "b1"): 0.5}))

    matches = llm_judgment_align(["a0"], ["b0", "b1"], lang_a="grc", lang_b="eng", margin_threshold=0.05)

    assert matches == []


def test_llm_judgment_align_empty_inputs_return_no_matches(monkeypatch):
    monkeypatch.setattr(sentence_align, "call_llm_json", lambda *a, **k: pytest.fail("must not call the LLM"))

    assert llm_judgment_align([], ["b0"], lang_a="grc", lang_b="eng") == []
    assert llm_judgment_align(["a0"], [], lang_a="grc", lang_b="eng") == []


def test_llm_judgment_align_resumes_from_a_checkpoint_without_repaying(tmp_path, monkeypatch):
    call_count = 0

    def counting_call(prompt: str, *, system: str, model: str | None = None) -> list[dict[str, float]]:
        nonlocal call_count
        call_count += 1
        return _fake_judgment_call(_GRC_ENG_SCORES)(prompt, system=system, model=model)

    monkeypatch.setattr(sentence_align, "call_llm_json", counting_call)
    checkpoint = Checkpoint(tmp_path / "align_progress.json", meta={})

    first = llm_judgment_align(["a0", "a1"], ["b0", "b1"], lang_a="grc", lang_b="eng", checkpoint=checkpoint)
    calls_after_first_run = call_count

    # Re-run with the *same* checkpoint instance (as a real interrupted-then-
    # resumed align_corpus.py invocation would, loading the same on-disk
    # progress file) -- every unit was already scored, so this must not
    # make a single additional LLM call.
    second = llm_judgment_align(["a0", "a1"], ["b0", "b1"], lang_a="grc", lang_b="eng", checkpoint=checkpoint)

    assert calls_after_first_run > 0
    assert call_count == calls_after_first_run  # no new calls on the resumed run
    assert [(m.i, m.j) for m in second] == [(m.i, m.j) for m in first]


def test_llm_judgment_scores_degrades_to_zero_after_a_failed_repair(monkeypatch):
    # Both the original call and the repair retry come back the wrong shape
    # -- must degrade to score 0.0 for every candidate, never raise and
    # never silently drop the unit.
    monkeypatch.setattr(sentence_align, "call_llm_json", lambda *a, **k: {"not": "a list"})

    scores = sentence_align._llm_judgment_scores("a0", ["b0", "b1"], lang_a="grc", lang_b="eng", model=None)

    assert scores == [0.0, 0.0]


def test_llm_judgment_scores_recovers_via_the_repair_retry(monkeypatch):
    attempts = []

    def flaky_call(prompt: str, *, system: str, model: str | None = None) -> object:
        attempts.append(prompt)
        if len(attempts) == 1:
            return [{"score": 0.9}]  # wrong length: 1 score for 2 candidates
        return [{"score": 0.9}, {"score": 0.1}]

    monkeypatch.setattr(sentence_align, "call_llm_json", flaky_call)

    scores = sentence_align._llm_judgment_scores("a0", ["b0", "b1"], lang_a="grc", lang_b="eng", model=None)

    assert scores == [0.9, 0.1]
    assert len(attempts) == 2  # exactly one repair retry, not a retry loop
