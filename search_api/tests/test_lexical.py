import math

import pytest

from app import lexical


def test_char_trigrams_pads_and_lowercases():
    assert lexical.char_trigrams("Sir") == {"  s", " si", "sir", "ir ", "r  "}
    assert lexical.char_trigrams("SIR") == lexical.char_trigrams("sir")


def test_char_trigrams_matches_ingest_side_definition():
    # Byte-identical to ye_olde.ingest.index.char_trigrams is load-bearing --
    # postings built by one and queried by the other only match if the
    # definition is the same. Hardcoded expected output here rather than
    # importing ye_olde (search_api never depends on it, spec §10) so this
    # test still catches an accidental drift in either copy.
    assert lexical.char_trigrams("sooth") == {"  s", " so", "soo", "oot", "oth", "th ", "h  "}


def test_token_char_length_is_the_inverse_of_char_trigrams_counting():
    for word in ["i", "to", "sir", "gladly", "consolation"]:
        assert lexical.token_char_length(len(lexical.char_trigrams(word))) == len(word)


def test_token_char_length_never_negative_for_a_degenerate_count():
    assert lexical.token_char_length(0) == 0


def test_lookup_ngram_threshold_rises_with_pair_length_toward_threshold_max():
    # 100 rather than something much larger: past a certain length,
    # 1 - exp(-length/k) rounds to exactly 1.0 in float64 (same class of
    # precision limit as resolver.compute_lambda's own test), which would
    # make this a test of floating-point rounding rather than monotonicity.
    values = [lexical.lookup_ngram_threshold(n) for n in (0, 2, 5, 10, 50, 100)]
    assert values == sorted(values)
    assert values[0] == 0.0
    assert values[-1] < lexical.LOOKUP_THRESHOLD_MAX
    assert values[-1] == pytest.approx(lexical.LOOKUP_THRESHOLD_MAX, rel=1e-2)


def test_lookup_ngram_threshold_matches_worked_example_for_þe_the():
    # þe/the: dice=0.222 at pair_length=5 -- the case that motivated making
    # this length-adaptive at all. Should clear the threshold with real
    # margin (not a boundary tie), per the worked numbers in lexical.py's
    # own comment.
    threshold = lexical.lookup_ngram_threshold(5)
    assert threshold < 0.222
    assert threshold == pytest.approx(0.197, abs=1e-3)


def test_lookup_ngram_threshold_zero_or_negative_length_is_zero():
    assert lexical.lookup_ngram_threshold(0) == 0.0
    assert lexical.lookup_ngram_threshold(-1) == 0.0


def test_lookup_ngram_threshold_long_pairs_converge_to_the_original_flat_value():
    # threshold_max is deliberately the old flat 0.5 -- long pairs should
    # see essentially unchanged behavior from before this was made adaptive.
    assert lexical.lookup_ngram_threshold(200) == pytest.approx(0.5, abs=1e-4)


def test_dice_coefficient_identical_sets_is_one():
    a = lexical.char_trigrams("sooth")
    assert lexical.dice_coefficient(a, a) == pytest.approx(1.0)


def test_dice_coefficient_disjoint_sets_is_zero():
    assert lexical.dice_coefficient({"abc"}, {"xyz"}) == 0.0


def test_dice_coefficient_both_empty_is_zero_not_a_division_error():
    assert lexical.dice_coefficient(set(), set()) == 0.0


def test_levenshtein_identical_strings_is_zero():
    assert lexical.levenshtein("sooth", "sooth") == 0
    assert lexical.levenshtein("Sooth", "sooth") == 0  # case-insensitive


def test_levenshtein_counts_single_edits():
    assert lexical.levenshtein("sooth", "sothe") == 2  # transposition = 2 single-char edits
    assert lexical.levenshtein("cat", "cats") == 1  # one insertion
    assert lexical.levenshtein("cat", "bat") == 1  # one substitution


def test_bm25_score_zero_for_no_matching_terms():
    assert lexical.bm25_score({}, {}, doc_len=10, avg_doc_len=10, corpus_size=100) == 0.0


def test_bm25_score_rewards_higher_term_frequency():
    low_tf = lexical.bm25_score({"gladly": 1}, {"gladly": 5}, doc_len=10, avg_doc_len=10, corpus_size=100)
    high_tf = lexical.bm25_score({"gladly": 5}, {"gladly": 5}, doc_len=10, avg_doc_len=10, corpus_size=100)
    assert high_tf > low_tf


def test_bm25_score_rewards_rarer_terms():
    common = lexical.bm25_score({"the": 1}, {"the": 90}, doc_len=10, avg_doc_len=10, corpus_size=100)
    rare = lexical.bm25_score({"gladly": 1}, {"gladly": 2}, doc_len=10, avg_doc_len=10, corpus_size=100)
    assert rare > common


def test_bm25_score_near_zero_for_term_in_every_document():
    # The "+1 inside the log" IDF variant (Lucene/Elasticsearch's default)
    # never goes negative even for a term in literally every document, so
    # this stays a small positive contribution rather than a penalty --
    # still far smaller than a genuinely rare term's contribution (the
    # rewards-rarer-terms test above).
    score = lexical.bm25_score({"the": 1}, {"the": 100}, doc_len=10, avg_doc_len=10, corpus_size=100)
    assert 0.0 < score < 0.01


def test_semantic_weight_is_one_at_the_present():
    assert lexical.semantic_weight(2026, now=2026) == pytest.approx(1.0)


def test_semantic_weight_decays_toward_zero_further_back():
    recent = lexical.semantic_weight(2020, now=2026)
    medieval = lexical.semantic_weight(1400, now=2026)
    ancient = lexical.semantic_weight(600, now=2026)
    assert recent > medieval > ancient
    assert ancient == pytest.approx(0.0, abs=1e-3)


def test_semantic_weight_clamps_future_years_to_zero_distance():
    assert lexical.semantic_weight(2100, now=2026) == pytest.approx(1.0)


def test_semantic_weight_matches_exponential_formula():
    weight = lexical.semantic_weight(1826, now=2026, scale_years=200.0)
    assert weight == pytest.approx(math.exp(-1.0))


def test_normalize_empty_list():
    assert lexical.normalize([]) == []


def test_normalize_min_max_to_unit_range():
    assert lexical.normalize([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


def test_normalize_all_equal_scores_returns_all_ones():
    assert lexical.normalize([0.7, 0.7, 0.7]) == [1.0, 1.0, 1.0]
    assert lexical.normalize([0.0]) == [1.0]
