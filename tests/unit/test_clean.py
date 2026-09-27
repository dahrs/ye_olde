"""Unit tests for clean.py: paragraph chunking, chunk-provenance tracking
in the cache format, and the local-model self-review pass.
"""

from __future__ import annotations

import json
from pathlib import Path

from ye_olde.ingest import clean
from ye_olde.ingest.corpus_files import CorpusFile


def test_chunk_text_respects_max_chars():
    paragraphs = [f"Paragraph {i} " + "word " * 20 for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = clean.chunk_text(text, max_chars=300)
    assert all(len(c) <= 300 or "\n\n" not in c for c in chunks)
    assert "".join(chunks).count("Paragraph") == 10


def test_review_units_with_llm_leaves_correct_units_unchanged(monkeypatch):
    units = ["A correct unit.", "Another correct unit."]
    monkeypatch.setattr(clean, "call_llm_json", lambda *a, **k: units)

    reviewed = clean.review_units_with_llm(units, lang_code="enm", work="Work")
    assert reviewed == units


def test_review_units_with_llm_applies_a_correction(monkeypatch):
    units = ["A unit with [Sidenote: leftover] noise.", "A clean unit."]
    monkeypatch.setattr(clean, "call_llm_json", lambda *a, **k: ["A unit with noise.", "A clean unit."])

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
    monkeypatch.setattr(clean, "review_units_with_llm", lambda units, **k: review_calls.append(1) or units)

    monkeypatch.setattr(clean, "is_local_model", lambda *a, **k: False)
    clean.clean_corpus_file(cf, "raw text", cache_path=tmp_path / "a.cleaned.json", use_cache=False)
    assert review_calls == []

    monkeypatch.setattr(clean, "is_local_model", lambda *a, **k: True)
    clean.clean_corpus_file(cf, "raw text", cache_path=tmp_path / "b.cleaned.json", use_cache=False)
    assert len(review_calls) == 1


def test_clean_corpus_file_saves_chunk_unit_counts(monkeypatch, tmp_path):
    """The cache file must record how many units came from each chunk, so
    two independently-cleaned runs of the same file can be compared
    chunk-for-chunk later (see CleanedDocument).
    """
    cf = CorpusFile(Path("a.txt"), "enm", 1400, "Work", "Author", "Src")
    # chunk_text() itself is tested elsewhere (test_chunk_text_respects_max_chars)
    # -- stub it here so this test controls the chunk count directly rather
    # than depending on the real paragraph-size-based splitting threshold.
    monkeypatch.setattr(clean, "chunk_text", lambda raw_text: ["chunk one", "chunk two", "chunk three"])

    chunk_outputs = iter([["u1", "u2"], ["u3"], ["u4", "u5", "u6"]])
    monkeypatch.setattr(clean, "clean_chunk_with_llm", lambda chunk, **k: next(chunk_outputs))
    monkeypatch.setattr(clean, "is_local_model", lambda *a, **k: False)

    cache_path = tmp_path / "a.cleaned.json"
    units = clean.clean_corpus_file(cf, "irrelevant raw text", cache_path=cache_path, use_cache=False)

    assert units == ["u1", "u2", "u3", "u4", "u5", "u6"]
    saved = json.loads(cache_path.read_text(encoding="utf-8"))
    assert saved == {"units": units, "chunk_unit_counts": [2, 1, 3]}


def test_clean_corpus_file_saves_chunk_unit_counts_regardless_of_model(monkeypatch, tmp_path):
    """Provenance tracking is plain bookkeeping about the source chunking,
    not a model-specific behavior — it must be saved identically whether
    the configured model is local or a hosted API.
    """
    cf = CorpusFile(Path("a.txt"), "enm", 1400, "Work", "Author", "Src")
    monkeypatch.setattr(clean, "chunk_text", lambda raw_text: ["chunk one", "chunk two"])
    monkeypatch.setattr(clean, "clean_chunk_with_llm", lambda chunk, **k: ["u"])
    # a no-op review pass so it doesn't change unit count/content for this check
    monkeypatch.setattr(clean, "review_units_with_llm", lambda units, **k: units)

    for is_local, name in [(False, "hosted.cleaned.json"), (True, "local.cleaned.json")]:
        monkeypatch.setattr(clean, "is_local_model", lambda *a, is_local=is_local, **k: is_local)
        cache_path = tmp_path / name
        clean.clean_corpus_file(cf, "irrelevant raw text", cache_path=cache_path, use_cache=False)
        saved = json.loads(cache_path.read_text(encoding="utf-8"))
        assert saved["chunk_unit_counts"] == [1, 1], f"is_local_model={is_local}"


def test_clean_corpus_file_reads_back_cached_chunk_unit_counts(monkeypatch, tmp_path):
    cf = CorpusFile(Path("a.txt"), "enm", 1400, "Work", "Author", "Src")
    cache_path = tmp_path / "a.cleaned.json"
    cache_path.write_text(json.dumps({"units": ["x", "y"], "chunk_unit_counts": [1, 1]}), encoding="utf-8")

    def fail_if_called(*a, **k):
        raise AssertionError("should not re-clean when a valid cache exists")

    monkeypatch.setattr(clean, "clean_chunk_with_llm", fail_if_called)

    units = clean.clean_corpus_file(cf, "irrelevant raw text", cache_path=cache_path, use_cache=True)
    assert units == ["x", "y"]  # only the flat unit list crosses clean_corpus_file's own return contract


def test_clean_corpus_file_resumes_partial_with_chunk_unit_counts(monkeypatch, tmp_path):
    cf = CorpusFile(Path("a.txt"), "enm", 1400, "Work", "Author", "Src")
    monkeypatch.setattr(clean, "chunk_text", lambda raw_text: ["chunk one", "chunk two"])
    cache_path = tmp_path / "a.cleaned.json"
    partial_path = tmp_path / "a.cleaned.json.partial"
    partial_path.write_text(
        json.dumps({"total_chunks": 2, "next_chunk": 1, "units": ["u1"], "chunk_unit_counts": [1]}),
        encoding="utf-8",
    )

    calls = []

    def fake_clean_chunk(chunk, **k):
        calls.append(chunk)
        return ["u2", "u3"]

    monkeypatch.setattr(clean, "clean_chunk_with_llm", fake_clean_chunk)
    monkeypatch.setattr(clean, "is_local_model", lambda *a, **k: False)

    units = clean.clean_corpus_file(cf, "irrelevant raw text", cache_path=cache_path, use_cache=True)

    assert len(calls) == 1  # only the un-done second chunk was actually cleaned
    assert units == ["u1", "u2", "u3"]
    saved = json.loads(cache_path.read_text(encoding="utf-8"))
    assert saved["chunk_unit_counts"] == [1, 2]
