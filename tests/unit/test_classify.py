"""Unit tests for classify/: LLM calls are mocked throughout — none of this
should need a live model to be correct, matching test_align.py's pattern.
"""

from __future__ import annotations

from ye_olde import classify
from ye_olde.common import function_words


def test_classify_resolves_tags_in_order(monkeypatch, tmp_path):
    monkeypatch.setattr(function_words, "DEFAULT_ROOT", tmp_path)
    fake_response = [
        {"text": "Robin", "lemma": "Robin", "upos": "PROPN", "feats": "", "deprel": "nsubj", "ner": "PER"},
        {"text": "walks", "lemma": "walk", "upos": "VERB", "feats": "Tense=Pres", "deprel": "root", "ner": "O"},
    ]
    monkeypatch.setattr(classify, "call_llm_json", lambda *a, **k: fake_response)

    tags = classify.classify("Robin walks", "eng", 2000)
    assert [t.text for t in tags] == ["Robin", "walks"]
    assert tags[0].upos == "PROPN"
    assert tags[0].ner == "PER"
    assert tags[0].is_name is True
    assert tags[1].is_name is False
    assert tags[1].lemma == "walk"


def test_classify_empty_sentence_returns_no_tags(monkeypatch, tmp_path):
    monkeypatch.setattr(function_words, "DEFAULT_ROOT", tmp_path)
    monkeypatch.setattr(classify, "call_llm_json", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no call")))
    assert classify.classify("", "eng", 2000) == []


def test_classify_falls_back_when_response_misaligned(monkeypatch, tmp_path):
    monkeypatch.setattr(function_words, "DEFAULT_ROOT", tmp_path)
    # Only one tag for two tokens -- position 1 has nothing to verify against.
    fake_response = [{"text": "Robin", "lemma": "Robin", "upos": "PROPN", "feats": "", "deprel": "nsubj", "ner": "PER"}]
    monkeypatch.setattr(classify, "call_llm_json", lambda *a, **k: fake_response)

    tags = classify.classify("Robin walks", "eng", 2000)
    assert len(tags) == 2
    assert tags[0].ner == "PER"
    # unverifiable position -> safe minimal fallback, not a guessed tag
    assert tags[1] == classify.TokenTag(text="walks", lemma=None, upos="X", feats=None, deprel=None, ner="O")


def test_classify_falls_back_when_claimed_text_does_not_match_real_token(monkeypatch, tmp_path):
    monkeypatch.setattr(function_words, "DEFAULT_ROOT", tmp_path)
    # The model claims to be tagging a different word than the real token
    # at this position -- must not be trusted even though it's well-formed.
    fake_response = [{"text": "Bob", "lemma": "Bob", "upos": "PROPN", "feats": "", "deprel": "nsubj", "ner": "PER"}]
    monkeypatch.setattr(classify, "call_llm_json", lambda *a, **k: fake_response)

    tags = classify.classify("Robin", "eng", 2000)
    assert tags == [classify.TokenTag(text="Robin", lemma=None, upos="X", feats=None, deprel=None, ner="O")]


def test_classify_parses_self_reported_confidence_per_field(monkeypatch, tmp_path):
    monkeypatch.setattr(function_words, "DEFAULT_ROOT", tmp_path)
    fake_response = [
        {
            "text": "walks",
            "lemma": "walk",
            "lemma_confidence": 0.95,
            "upos": "VERB",
            "upos_confidence": "0.7",  # numeric string, same as a real LLM JSON reply can send
            "feats": "Tense=Pres",
            "feats_confidence": -0.2,  # out of range -> clamped
            # deprel intentionally omitted -> deprel stays None, confidence must too.
            "deprel_confidence": 0.5,
            "ner": "O",
        }
    ]
    monkeypatch.setattr(classify, "call_llm_json", lambda *a, **k: fake_response)

    tags = classify.classify("walks", "eng", 2000)
    assert tags[0].lemma_confidence == 0.95
    assert tags[0].upos_confidence == 0.7
    assert tags[0].feats_confidence == 0.0
    assert tags[0].deprel is None
    assert tags[0].deprel_confidence is None


def test_classify_confidence_is_none_when_not_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(function_words, "DEFAULT_ROOT", tmp_path)
    fake_response = [{"text": "Robin", "lemma": "Robin", "upos": "PROPN", "feats": "", "deprel": "nsubj", "ner": "PER"}]
    monkeypatch.setattr(classify, "call_llm_json", lambda *a, **k: fake_response)

    tags = classify.classify("Robin", "eng", 2000)
    assert tags[0].lemma_confidence is None
    assert tags[0].upos_confidence is None


def test_classify_fallback_tag_has_no_confidence(monkeypatch, tmp_path):
    monkeypatch.setattr(function_words, "DEFAULT_ROOT", tmp_path)
    fake_response = [{"text": "Bob", "lemma": "Bob", "upos": "PROPN", "feats": "", "deprel": "nsubj", "ner": "PER"}]
    monkeypatch.setattr(classify, "call_llm_json", lambda *a, **k: fake_response)

    tags = classify.classify("Robin", "eng", 2000)
    assert tags[0].lemma_confidence is None
    assert tags[0].upos_confidence is None


def test_classify_rejects_invalid_ner_tag(monkeypatch, tmp_path):
    monkeypatch.setattr(function_words, "DEFAULT_ROOT", tmp_path)
    fake_response = [
        {"text": "Robin", "lemma": "Robin", "upos": "PROPN", "feats": "", "deprel": "nsubj", "ner": "NAME"}
    ]
    monkeypatch.setattr(classify, "call_llm_json", lambda *a, **k: fake_response)

    tags = classify.classify("Robin", "eng", 2000)
    assert tags[0].ner == "O"  # not a recognized tag from the closed vocabulary -> safe default


def test_classify_non_list_response_falls_back_for_every_token(monkeypatch, tmp_path):
    monkeypatch.setattr(function_words, "DEFAULT_ROOT", tmp_path)
    monkeypatch.setattr(classify, "call_llm_json", lambda *a, **k: {"unexpected": "shape"})
    tags = classify.classify("Robin walks", "eng", 2000)
    assert all(t.upos == "X" and t.ner == "O" for t in tags)


def test_classify_records_closed_class_tokens_into_the_function_word_registry(monkeypatch, tmp_path):
    monkeypatch.setattr(function_words, "DEFAULT_ROOT", tmp_path)
    fake_response = [{"text": "the", "lemma": "the", "upos": "DET", "feats": "", "deprel": "det", "ner": "O"}]
    monkeypatch.setattr(classify, "call_llm_json", lambda *a, **k: fake_response)

    classify.classify("the", "eng", 2000)
    assert function_words.load("eng", 2000).get("the") == "DET"


def test_classify_never_overrides_a_verified_disagreeing_tag_with_the_cache(monkeypatch, tmp_path):
    # The bug this guards against: an earlier version let the registry
    # clobber any fresh tag that disagreed with it, then re-recorded the
    # clobbered (wrong) value -- permanently mis-tagging every future
    # occurrence of a genuinely ambiguous closed-class word. "þe" was
    # recorded as DET from an earlier sentence; this call's own verified
    # response tags it PRON (a real, different grammatical role) -- that
    # must be trusted, not overridden.
    monkeypatch.setattr(function_words, "DEFAULT_ROOT", tmp_path)
    function_words.record("enm", 1400, "þe", "DET")

    fake_response = [{"text": "þe", "lemma": "the", "upos": "PRON", "feats": "", "deprel": "det", "ner": "O"}]
    monkeypatch.setattr(classify, "call_llm_json", lambda *a, **k: fake_response)

    tags = classify.classify("þe", "enm", 1400)
    assert tags[0].upos == "PRON"  # this call's own verified tag wins, not the stale cache


def test_classify_fills_in_from_cache_only_when_this_calls_tag_is_unverifiable(monkeypatch, tmp_path):
    monkeypatch.setattr(function_words, "DEFAULT_ROOT", tmp_path)
    function_words.record("enm", 1400, "þe", "DET")

    # response text doesn't match the real token -> _resolve_tags falls
    # back to the "X" placeholder, which IS eligible for the cache fill-in.
    fake_response = [{"text": "wrong-word", "lemma": "the", "upos": "PRON", "feats": "", "deprel": "det", "ner": "O"}]
    monkeypatch.setattr(classify, "call_llm_json", lambda *a, **k: fake_response)

    tags = classify.classify("þe", "enm", 1400)
    assert tags[0].upos == "DET"  # unverifiable ("X") -> cache fills it in
