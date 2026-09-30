"""Unit tests for api.translate() — the fixed classify -> fallback ->
generate -> self_check call sequence. Every one of those is mocked; none of
this should need a live model or Search API.
"""

from __future__ import annotations

from ye_olde import api
from ye_olde.classify import TokenTag
from ye_olde.fallback import FallbackResult
from ye_olde.generation import SelfCheckResult


def test_translate_wires_content_words_through_fallback(monkeypatch):
    tags = [
        TokenTag(text="Robin", lemma="Robin", upos="PROPN", feats=None, deprel="nsubj", ner="PER"),
        TokenTag(text="walks", lemma="walk", upos="VERB", feats="Tense=Pres", deprel="root", ner="O"),
        TokenTag(text=".", lemma=None, upos="PUNCT", feats=None, deprel="punct", ner="O"),
    ]
    monkeypatch.setattr(api.classify, "classify", lambda *a, **k: tags)

    resolve_calls = []

    def fake_resolve(term, lang_a, year_a, lang_b, year_b, **kwargs):
        resolve_calls.append((term, lang_a, year_a, lang_b, year_b))
        return FallbackResult(status="attested", form="wandreþ", note="attested in corpus")

    monkeypatch.setattr(api.fallback, "resolve", fake_resolve)
    monkeypatch.setattr(api.generation, "generate", lambda *a, **k: "Robin wandreþ.")
    monkeypatch.setattr(
        api.generation,
        "self_check",
        lambda *a, **k: SelfCheckResult(approved=True, sentence="Robin wandreþ.", note=""),
    )

    result = api.translate("Robin walks.", "eng", 2000, "enm", 1400)

    # only the one content (non-name, non-punct) word went through fallback
    assert resolve_calls == [("walk", "eng", 2000, "enm", 1400)]
    assert result.sentence == "Robin wandreþ."
    statuses = {a.span: a.status for a in result.annotations}
    assert statuses["Robin"] == "name-passthrough"
    assert statuses["walks"] == "attested"
    assert "." not in statuses  # punctuation never becomes a grounded form


def test_translate_carries_self_check_note_when_not_approved(monkeypatch):
    monkeypatch.setattr(api.classify, "classify", lambda *a, **k: [])
    monkeypatch.setattr(api.generation, "generate", lambda *a, **k: "draft")
    monkeypatch.setattr(
        api.generation,
        "self_check",
        lambda *a, **k: SelfCheckResult(approved=False, sentence="corrected", note="fixed grammar"),
    )

    result = api.translate("hi", "eng", 2000, "eng", 2000)
    assert result.sentence == "corrected"


def test_translate_keeps_per_token_note_and_appends_self_check_note_when_flagged(monkeypatch):
    # An earlier version discarded fallback's own note entirely and
    # stamped the same single self_check note onto every annotation
    # regardless of relevance -- this checks both are now preserved.
    tags = [TokenTag(text="walks", lemma="walk", upos="VERB", feats=None, deprel="root", ner="O")]
    monkeypatch.setattr(api.classify, "classify", lambda *a, **k: tags)
    monkeypatch.setattr(
        api.fallback,
        "resolve",
        lambda *a, **k: FallbackResult(status="attested", form="gaeth", note="attested in Sir Gawayne (1400)"),
    )
    monkeypatch.setattr(api.generation, "generate", lambda *a, **k: "draft")
    monkeypatch.setattr(
        api.generation,
        "self_check",
        lambda *a, **k: SelfCheckResult(approved=False, sentence="corrected", note="fixed grammar"),
    )

    result = api.translate("walks", "eng", 2000, "enm", 1400)
    note = result.annotations[0].note
    assert "attested in Sir Gawayne (1400)" in note  # per-token note kept
    assert "fixed grammar" in note  # self-check note appended, not substituted


def test_translate_omits_self_check_note_when_approved(monkeypatch):
    tags = [TokenTag(text="walks", lemma="walk", upos="VERB", feats=None, deprel="root", ner="O")]
    monkeypatch.setattr(api.classify, "classify", lambda *a, **k: tags)
    monkeypatch.setattr(
        api.fallback,
        "resolve",
        lambda *a, **k: FallbackResult(status="anachronism-passthrough", form="walk", note="no attested form"),
    )
    monkeypatch.setattr(api.generation, "generate", lambda *a, **k: "draft")
    monkeypatch.setattr(
        api.generation, "self_check", lambda *a, **k: SelfCheckResult(approved=True, sentence="draft", note="")
    )

    result = api.translate("walks", "eng", 2000, "enm", 1400)
    # the anachronism-passthrough annotation's own explanation survives
    # even though the sentence overall passed self-check.
    assert result.annotations[0].note == "no attested form"


def test_translate_identity_case_with_no_content_words(monkeypatch):
    monkeypatch.setattr(api.classify, "classify", lambda *a, **k: [])
    monkeypatch.setattr(api.generation, "generate", lambda *a, **k: "hi")
    monkeypatch.setattr(
        api.generation, "self_check", lambda *a, **k: SelfCheckResult(approved=True, sentence="hi", note="")
    )

    result = api.translate("hi", "eng", 2000, "eng", 2000)
    assert result.sentence == "hi"
    assert result.annotations == []
