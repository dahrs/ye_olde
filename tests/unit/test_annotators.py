"""Unit tests for annotators.py: real Stanza is never touched here —
`_pipeline`/`_annotate_text` are monkeypatched to small fakes, the same
"no network/heavy-model access needed" pattern this project's other tests
use for litellm/nltk/sentence-transformers.
"""

from __future__ import annotations

import pytest

from ye_olde.ingest import annotators
from ye_olde.ingest.annotators import ToolTokenTag


@pytest.fixture(autouse=True)
def _clear_caches():
    annotators._annotate_text.cache_clear()
    annotators._pipeline.cache_clear()
    yield
    annotators._annotate_text.cache_clear()
    annotators._pipeline.cache_clear()


def test_has_tool_true_for_registered_language():
    assert annotators.has_tool("lat") is True
    assert annotators.has_tool("grc") is True


def test_has_tool_false_for_unregistered_language():
    # enm (Middle English) and xno (Anglo-Norman) have no Stanza pipeline
    # as of the registry's source data -- ang (Old English) does, unlike
    # in an earlier, narrower version of this registry, so it's
    # deliberately not used as a negative example here anymore.
    assert annotators.has_tool("enm") is False
    assert annotators.has_tool("xno") is False


def test_annotate_span_returns_none_for_unregistered_language():
    assert annotators.annotate_span("enm", "þe knyght rode", "þe") is None


def test_annotate_span_matches_a_unique_token(monkeypatch):
    fake_tags = (
        ToolTokenTag(text="Gallia", lemma="Gallia", upos="PROPN", feats="Case=Nom", deprel="nsubj:pass"),
        ToolTokenTag(text="est", lemma="sum", upos="AUX", feats="Mood=Ind", deprel="aux:pass"),
    )
    monkeypatch.setattr(annotators, "_annotate_text", lambda lang, text: fake_tags)

    tag = annotators.annotate_span("lat", "Gallia est", "Gallia")
    assert tag is not None
    assert tag.lemma == "Gallia"
    assert tag.upos == "PROPN"
    assert tag.deprel == "nsubj:pass"


def test_annotate_span_is_case_insensitive(monkeypatch):
    fake_tags = (ToolTokenTag(text="Gallia", lemma="Gallia", upos="PROPN", feats=None, deprel="nsubj"),)
    monkeypatch.setattr(annotators, "_annotate_text", lambda lang, text: fake_tags)

    tag = annotators.annotate_span("lat", "Gallia est", "gallia")  # query lowercased, tool's own tag stays "Gallia"
    assert tag is not None
    assert tag.text == "Gallia"


def test_annotate_span_returns_none_when_span_not_found(monkeypatch):
    monkeypatch.setattr(
        annotators,
        "_annotate_text",
        lambda lang, text: (ToolTokenTag(text="est", lemma="sum", upos="AUX", feats=None, deprel="aux"),),
    )
    assert annotators.annotate_span("lat", "Gallia est", "Gallia") is None


def test_annotate_span_returns_none_when_span_is_ambiguous(monkeypatch):
    # The same surface form appears twice -- which occurrence is meant is
    # ambiguous without positional matching, so this deliberately doesn't
    # guess (see module docstring).
    fake_tags = (
        ToolTokenTag(text="lux", lemma="lux", upos="NOUN", feats=None, deprel="nsubj"),
        ToolTokenTag(text="lux", lemma="lux", upos="NOUN", feats=None, deprel="obj"),
    )
    monkeypatch.setattr(annotators, "_annotate_text", lambda lang, text: fake_tags)
    assert annotators.annotate_span("lat", "lux et lux", "lux") is None


def test_annotate_span_returns_none_for_empty_span():
    assert annotators.annotate_span("lat", "Gallia est", "") is None


def test_annotate_span_never_matches_a_multi_word_span(monkeypatch):
    fake_tags = (
        ToolTokenTag(text="in", lemma="in", upos="ADP", feats=None, deprel="case"),
        ToolTokenTag(text="partes", lemma="pars", upos="NOUN", feats=None, deprel="obl"),
    )
    monkeypatch.setattr(annotators, "_annotate_text", lambda lang, text: fake_tags)
    # No single tool token's surface form equals "in partes" -- correctly
    # unmatched rather than guessed at from the two adjacent tokens.
    assert annotators.annotate_span("lat", "divisa in partes", "in partes") is None


def test_pipeline_retries_without_mwt_when_language_has_no_mwt_model(monkeypatch):
    import stanza
    from stanza.pipeline.core import UnsupportedProcessorError

    calls = []

    def fake_pipeline(stanza_lang, processors, verbose):
        calls.append(processors)
        if "mwt" in processors:
            raise UnsupportedProcessorError("mwt", stanza_lang)
        return "pipeline-built-without-mwt"

    monkeypatch.setattr(stanza, "Pipeline", fake_pipeline)
    assert annotators._pipeline("ang") == "pipeline-built-without-mwt"
    assert calls == ["tokenize,mwt,pos,lemma,depparse", "tokenize,pos,lemma,depparse"]


def test_pipeline_reraises_unsupported_processor_error_for_other_processors(monkeypatch):
    import stanza
    from stanza.pipeline.core import UnsupportedProcessorError

    def fake_pipeline(stanza_lang, processors, verbose):
        raise UnsupportedProcessorError("depparse", stanza_lang)  # not the mwt case -- not this fallback's job

    monkeypatch.setattr(stanza, "Pipeline", fake_pipeline)
    with pytest.raises(UnsupportedProcessorError):
        annotators._pipeline("lat")
