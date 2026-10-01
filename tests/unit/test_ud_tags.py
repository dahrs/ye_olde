"""Unit tests for common/ud_tags.py's closed-vocabulary normalizers, plus
the cross-source consistency check: the same word, tagged differently by
Stanza's own output shape vs. a typical LLM-proposed shape, must normalize
to the *same* final vocabulary — proving `align.py` (LLM path),
`classify.py` (LLM path), and `annotators.py` (Stanza path) can never
produce two different notations for the same tag space, only possibly
different *values* within one shared standard (spec §3d — the `eng` vs.
`lat` analogy: different values are fine, different standards are not).
"""

from __future__ import annotations

from ye_olde.common.ud_tags import (
    TOOL_CONFIDENCE,
    VALID_DEPREL_BASE,
    VALID_NER,
    VALID_UPOS,
    normalize_deprel,
    normalize_ner,
    normalize_upos,
    parse_confidence,
)


def test_normalize_upos_passes_through_a_valid_tag():
    assert normalize_upos("NOUN") == "NOUN"


def test_normalize_upos_uppercases_a_differently_cased_tag():
    assert normalize_upos("propn") == "PROPN"
    assert normalize_upos("Verb") == "VERB"


def test_normalize_upos_rejects_an_invalid_tag_as_x():
    assert normalize_upos("PROPNOUN") == "X"
    assert normalize_upos("") == "X"


def test_normalize_deprel_passes_through_a_valid_core_relation():
    assert normalize_deprel("nsubj") == "nsubj"


def test_normalize_deprel_keeps_a_legitimate_ud_subtype():
    # Both observed live from Stanza's own Latin/Greek PROIEL models.
    assert normalize_deprel("nsubj:pass") == "nsubj:pass"
    assert normalize_deprel("aux:pass") == "aux:pass"
    assert normalize_deprel("obl:arg") == "obl:arg"  # observed live from Stanza's Latin ITTB model


def test_normalize_deprel_lowercases_a_differently_cased_tag():
    assert normalize_deprel("NSUBJ") == "nsubj"
    assert normalize_deprel("Nsubj:Pass") == "nsubj:pass"


def test_normalize_deprel_rejects_an_invalid_base_relation_as_dep():
    assert normalize_deprel("hallucinated_relation") == "dep"
    assert normalize_deprel("hallucinated_relation:subtype") == "dep"


def test_normalize_ner_passes_through_a_valid_tag():
    assert normalize_ner("PER") == "PER"


def test_normalize_ner_rejects_an_invalid_tag_as_o():
    assert normalize_ner("PERSON") == "O"
    assert normalize_ner("") == "O"


def test_valid_sets_are_all_uppercase_or_lowercase_as_normalizers_expect():
    # normalize_upos/normalize_ner uppercase before comparing; normalize_deprel lowercases.
    assert all(tag == tag.upper() for tag in VALID_UPOS)
    assert all(tag == tag.upper() for tag in VALID_NER)
    assert all(tag == tag.lower() for tag in VALID_DEPREL_BASE)


def test_the_same_latin_word_tagged_two_different_ways_normalizes_identically():
    """The exact check requested: one Latin word ("Gallia", tagged as a
    proper noun acting as a passive subject), expressed the way Stanza's
    own PROIEL-model output looks (upper-case UD tags, verbatim) and the
    way an LLM's JSON reply typically looks (free-form casing, no
    guaranteed vocabulary) — both must land on the identical final
    (upos, deprel) pair after normalization. This is the actual guarantee
    `ingest.align._annotate_side` depends on: whichever of the two
    produced a link's tag, the caller downstream sees one standard.
    """
    stanza_shaped = {"upos": "PROPN", "deprel": "nsubj:pass"}  # verbatim from a real live Stanza (la_proiel) call
    llm_shaped = {"upos": "propn", "deprel": "NSUBJ:PASS"}  # a plausible, differently-cased LLM JSON reply

    stanza_result = (normalize_upos(stanza_shaped["upos"]), normalize_deprel(stanza_shaped["deprel"]))
    llm_result = (normalize_upos(llm_shaped["upos"]), normalize_deprel(llm_shaped["deprel"]))

    assert stanza_result == llm_result == ("PROPN", "nsubj:pass")


def test_parse_confidence_passes_through_an_in_range_value():
    assert parse_confidence(0.85) == 0.85
    assert parse_confidence(0) == 0.0
    assert parse_confidence(1) == 1.0


def test_parse_confidence_accepts_a_numeric_string():
    assert parse_confidence("0.7") == 0.7


def test_parse_confidence_clamps_out_of_range_values():
    assert parse_confidence(1.5) == 1.0
    assert parse_confidence(-0.3) == 0.0


def test_parse_confidence_returns_none_for_missing_or_unusable_values():
    assert parse_confidence(None) is None
    assert parse_confidence("") is None
    assert parse_confidence("not a number") is None
    assert parse_confidence(float("nan")) is None
    assert (
        parse_confidence(True) is None
    )  # bool is technically an int subclass -- explicitly rejected, not a real score


def test_tool_confidence_is_the_maximum_valid_value():
    assert TOOL_CONFIDENCE == 1.0
