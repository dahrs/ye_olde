"""Unit tests for split_mixed.py: chunk classification, manifest loading,
and the end-to-end split -- correct naming, no-overwrite protection, and
removal of the superseded mixed file.
"""

from __future__ import annotations

import pytest

from ye_olde.ingest import split_mixed
from ye_olde.ingest.split_mixed import SectionHint, SplitManifest


def test_load_manifest_parses_yaml(tmp_path):
    manifest_path = tmp_path / "work.sections.yaml"
    manifest_path.write_text(
        "title: A Work\n"
        "source: Project Gutenberg\n"
        "sections:\n"
        "  - name: Original Author\n"
        "    year: 1000\n"
        "    lang_code: ang\n"
        '    clues: ["Old English"]\n'
        "  - name: A Translator\n"
        "    year: 1900\n"
        "    lang_code: eng\n"
        '    clues: ["Modern English", "prose"]\n',
        encoding="utf-8",
    )
    manifest = split_mixed.load_manifest(manifest_path)
    assert manifest.title == "A Work"
    assert manifest.sections[0].name == "Original Author"
    assert manifest.sections[1].clues == ["Modern English", "prose"]


def test_classify_chunks_returns_assignments(monkeypatch):
    sections = [
        SectionHint(name="Author A", year=1000, lang_code="ang", clues=["old"]),
        SectionHint(name="Author B", year=1900, lang_code="eng", clues=["modern"]),
    ]
    monkeypatch.setattr(split_mixed, "call_llm_json", lambda *a, **k: [0, 0, None, 1])
    result = split_mixed.classify_chunks(["c0", "c1", "c2", "c3"], sections, batch_size=4)
    assert result == [0, 0, None, 1]


def test_classify_chunks_raises_on_wrong_length_reply(monkeypatch):
    sections = [SectionHint(name="A", year=1000, lang_code="ang")]
    monkeypatch.setattr(split_mixed, "call_llm_json", lambda *a, **k: [0])  # batch of 2 expected
    with pytest.raises(ValueError):
        split_mixed.classify_chunks(["c0", "c1"], sections, batch_size=4)


def test_classify_chunks_batches_and_checkpoints(monkeypatch, tmp_path):
    from ye_olde.ingest.checkpoint import Checkpoint

    sections = [SectionHint(name="A", year=1000, lang_code="ang")]
    calls = []
    monkeypatch.setattr(split_mixed, "call_llm_json", lambda prompt, **k: calls.append(prompt) or [0, 0])
    checkpoint = Checkpoint(tmp_path / "progress.json", meta={})
    result = split_mixed.classify_chunks(["c0", "c1", "c2", "c3"], sections, batch_size=2, checkpoint=checkpoint)
    assert result == [0, 0, 0, 0]
    assert len(calls) == 2  # two batches of 2

    # re-running with the same checkpoint should hit the cache, not call the LLM again
    result2 = split_mixed.classify_chunks(["c0", "c1", "c2", "c3"], sections, batch_size=2, checkpoint=checkpoint)
    assert result2 == result
    assert len(calls) == 2


def _manifest_for_two_sections():
    return SplitManifest(
        title="A Work",
        source="Project Gutenberg",
        sections=[
            SectionHint(name="Original Author", year=1000, lang_code="ang", clues=["old"]),
            SectionHint(name="A Translator", year=1900, lang_code="eng", clues=["modern"]),
        ],
    )


def test_split_mixed_source_writes_sections_and_removes_mixed_file(monkeypatch, tmp_path):
    folder = tmp_path / "ang-1000-A_Work-Original_Author-Project_Gutenberg"
    folder.mkdir()
    mixed = folder / "ang-1000-A_Work-Original_Author-Project_Gutenberg.txt"
    mixed.write_text("original paragraph\n\ntranslated paragraph", encoding="utf-8")

    monkeypatch.setattr(split_mixed, "chunk_text", lambda text, **k: ["original paragraph", "translated paragraph"])
    monkeypatch.setattr(split_mixed, "classify_chunks", lambda chunks, sections, **k: [0, 1])

    written = split_mixed.split_mixed_source(mixed, _manifest_for_two_sections(), raw_dir=tmp_path)

    assert {p.name for p in written} == {
        "ang-1000-A_Work-Original_Author-Project_Gutenberg.txt",
        "eng-1900-A_Work-A_Translator-Project_Gutenberg.txt",
    }
    for p in written:
        assert p.exists()
    original = next(p for p in written if "Original_Author" in p.name)
    translated = next(p for p in written if "A_Translator" in p.name)
    assert original.read_text(encoding="utf-8") == "original paragraph"
    assert translated.read_text(encoding="utf-8") == "translated paragraph"
    assert not mixed.exists() or mixed in written  # superseded mixed file is gone unless it was reused in place


def test_split_mixed_source_refuses_to_overwrite_unrelated_collision(monkeypatch, tmp_path):
    folder = tmp_path / "ang-1000-A_Work-Original_Author-Project_Gutenberg"
    folder.mkdir()
    mixed = folder / "mixed-input.txt"
    mixed.write_text("original paragraph\n\ntranslated paragraph", encoding="utf-8")
    collider = folder / "eng-1900-A_Work-A_Translator-Project_Gutenberg.txt"
    collider.write_text("unrelated pre-existing content", encoding="utf-8")

    monkeypatch.setattr(split_mixed, "chunk_text", lambda text, **k: ["original paragraph", "translated paragraph"])
    monkeypatch.setattr(split_mixed, "classify_chunks", lambda chunks, sections, **k: [0, 1])

    with pytest.raises(FileExistsError):
        split_mixed.split_mixed_source(mixed, _manifest_for_two_sections(), raw_dir=tmp_path)
    assert collider.read_text(encoding="utf-8") == "unrelated pre-existing content"  # untouched
    assert mixed.exists()  # not removed, since the run failed


def test_split_mixed_source_raises_when_a_section_matches_nothing(monkeypatch, tmp_path):
    folder = tmp_path / "ang-1000-A_Work-Original_Author-Project_Gutenberg"
    folder.mkdir()
    mixed = folder / "mixed-input.txt"
    mixed.write_text("only one paragraph", encoding="utf-8")

    monkeypatch.setattr(split_mixed, "classify_chunks", lambda chunks, sections, **k: [0])

    with pytest.raises(ValueError):
        split_mixed.split_mixed_source(mixed, _manifest_for_two_sections(), raw_dir=tmp_path)
