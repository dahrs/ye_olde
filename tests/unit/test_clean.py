"""Unit tests for clean.py's local-model self-review pass."""

from __future__ import annotations

from pathlib import Path

from ye_olde.ingest import clean
from ye_olde.ingest.corpus_files import CorpusFile


def test_review_units_with_llm_leaves_correct_units_unchanged(monkeypatch):
    units = ["A correct unit.", "Another correct unit."]
    monkeypatch.setattr(clean, "call_llm_json", lambda *a, **k: units)

    reviewed = clean.review_units_with_llm(units, lang_code="enm", work="Work")
    assert reviewed == units


def test_review_units_with_llm_applies_a_correction(monkeypatch):
    units = ["A unit with [Sidenote: leftover] noise.", "A clean unit."]
    monkeypatch.setattr(
        clean, "call_llm_json", lambda *a, **k: ["A unit with noise.", "A clean unit."]
    )

    reviewed = clean.review_units_with_llm(units, lang_code="enm", work="Work")
    assert reviewed == ["A unit with noise.", "A clean unit."]


def test_review_units_with_llm_fails_safe_on_wrong_length_reply(monkeypatch):
    units = ["Unit one.", "Unit two."]
    monkeypatch.setattr(clean, "call_llm_json", lambda *a, **k: ["only one item"])

    reviewed = clean.review_units_with_llm(units, lang_code="enm", work="Work")
    assert reviewed == units  # malformed reply -> originals kept, nothing lost


def test_clean_corpus_file_runs_review_only_for_local_models(monkeypatch, tmp_path):
    cf = CorpusFile(Path("a.txt"), "enm", 1400, "Work", "Author", "Src")
    monkeypatch.setattr(clean, "clean_chunk_with_llm", lambda chunk, **k: ["one unit"])

    review_calls = []
    monkeypatch.setattr(
        clean, "review_units_with_llm", lambda units, **k: (review_calls.append(1) or units)
    )

    monkeypatch.setattr(clean, "is_local_model", lambda *a, **k: False)
    clean.clean_corpus_file(cf, "raw text", cache_path=tmp_path / "a.cleaned.json", use_cache=False)
    assert review_calls == []

    monkeypatch.setattr(clean, "is_local_model", lambda *a, **k: True)
    clean.clean_corpus_file(cf, "raw text", cache_path=tmp_path / "b.cleaned.json", use_cache=False)
    assert len(review_calls) == 1
