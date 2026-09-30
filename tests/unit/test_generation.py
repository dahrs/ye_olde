"""Unit tests for generation/: LLM calls are mocked throughout — none of
this should need a live model to be correct.
"""

from __future__ import annotations

import pytest

from ye_olde import generation
from ye_olde.generation import GroundedForm


def test_generate_returns_sentence_from_response(monkeypatch):
    monkeypatch.setattr(generation, "call_llm_json", lambda *a, **k: {"sentence": "Ic eom hal."})
    forms = [GroundedForm(span="I am well", form="Ic eom hal", status="attested")]
    result = generation.generate("I am well.", "ang", 900, forms)
    assert result == "Ic eom hal."


def test_generate_raises_on_missing_sentence_field(monkeypatch):
    monkeypatch.setattr(generation, "call_llm_json", lambda *a, **k: {"unexpected": "shape"})
    with pytest.raises(ValueError, match="sentence"):
        generation.generate("hi", "eng", 2000, [])


def test_generate_raises_on_empty_sentence(monkeypatch):
    monkeypatch.setattr(generation, "call_llm_json", lambda *a, **k: {"sentence": "   "})
    with pytest.raises(ValueError, match="sentence"):
        generation.generate("hi", "eng", 2000, [])


def test_self_check_approves_when_response_says_so(monkeypatch):
    monkeypatch.setattr(
        generation, "call_llm_json", lambda *a, **k: {"approved": True, "sentence": "Ic eom hal.", "note": "ok"}
    )
    forms = [GroundedForm(span="I am well", form="Ic eom hal", status="attested")]
    result = generation.self_check("Ic eom hal.", "ang", 900, forms)
    assert result.approved is True
    assert result.sentence == "Ic eom hal."


def test_self_check_returns_corrected_sentence_when_not_approved(monkeypatch):
    monkeypatch.setattr(
        generation,
        "call_llm_json",
        lambda *a, **k: {"approved": False, "sentence": "Ic eom hal, la.", "note": "fixed a dangling particle"},
    )
    forms = [GroundedForm(span="I am well", form="Ic eom hal", status="attested")]
    result = generation.self_check("Ic eom hal la.", "ang", 900, forms)
    assert result.approved is False
    assert result.sentence == "Ic eom hal, la."
    assert "particle" in result.note


def test_self_check_falls_back_to_draft_when_sentence_field_missing(monkeypatch):
    monkeypatch.setattr(generation, "call_llm_json", lambda *a, **k: {"approved": True})
    result = generation.self_check("draft text", "eng", 2000, [])
    assert result.sentence == "draft text"


def test_self_check_raises_on_non_dict_response(monkeypatch):
    monkeypatch.setattr(generation, "call_llm_json", lambda *a, **k: ["not", "a", "dict"])
    with pytest.raises(ValueError):
        generation.self_check("draft", "eng", 2000, [])
