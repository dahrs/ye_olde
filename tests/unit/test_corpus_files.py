"""Unit tests for corpus_files.py: parsing and discovering `data/raw/`
filenames.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ye_olde.ingest.corpus_files import (
    build_corpus_filename,
    discover_corpus_files,
    oldest_parseable_file,
    parse_corpus_filename,
    sanitize_name_field,
)


def test_parse_corpus_filename():
    cf = parse_corpus_filename(Path("enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg.txt"))
    assert cf.lang_code == "enm"
    assert cf.year == 1400
    assert cf.title == "Sir_Gawayne_and_the_Green_Knight"
    assert cf.author == "Richard Morris"
    assert cf.source == "Project Gutenberg"
    assert cf.work_label == "Sir Gawayne and the Green Knight"


def test_parse_corpus_filename_rejects_too_few_fields():
    with pytest.raises(ValueError):
        parse_corpus_filename(Path("eng-1999-onlythree.txt"))


def test_discover_corpus_files_requires_at_least_two(tmp_path):
    (tmp_path / "eng-1999-Work-Author-Source.txt").write_text("hello")
    with pytest.raises(ValueError):
        discover_corpus_files(tmp_path)


def test_discover_corpus_files_sorts_by_year(tmp_path):
    (tmp_path / "eng-1999-Work-Author-Source.txt").write_text("modern")
    (tmp_path / "enm-1400-Work-Author-Source.txt").write_text("older")
    files = discover_corpus_files(tmp_path)
    assert [cf.year for cf in files] == [1400, 1999]


def test_sanitize_name_field_collapses_whitespace_and_punctuation():
    assert sanitize_name_field("Chaucer's Translation") == "Chaucer_s_Translation"
    assert sanitize_name_field("  leading and trailing  ") == "leading_and_trailing"
    assert sanitize_name_field("a---b") == "a_b"


def test_sanitize_name_field_rejects_empty_result():
    with pytest.raises(ValueError):
        sanitize_name_field("   ---   ")


def test_build_corpus_filename_round_trips():
    filename = build_corpus_filename(
        lang_code="enm", year=1400, title="Sir Gawayne", author="Richard Morris", source="Project Gutenberg", ext=".txt"
    )
    assert filename == "enm-1400-Sir_Gawayne-Richard_Morris-Project_Gutenberg.txt"
    cf = parse_corpus_filename(Path(filename))
    assert cf.year == 1400
    assert cf.work_label == "Sir Gawayne"


def test_build_corpus_filename_rejects_non_numeric_year():
    with pytest.raises(ValueError):
        build_corpus_filename(
            lang_code="eng", year="nineteen", title="Work", author="Author", source="Source", ext=".txt"
        )


def test_oldest_parseable_file_skips_unparseable_and_picks_minimum_year(tmp_path):
    (tmp_path / "eng-1999-Work-Author-Source.txt").write_text("modern")
    (tmp_path / "enm-1400-Work-Author-Source.txt").write_text("older")
    (tmp_path / "not_a_conforming_name.txt").write_text("legacy, non-conformant")
    oldest = oldest_parseable_file(tmp_path)
    assert oldest is not None
    assert oldest.year == 1400


def test_oldest_parseable_file_returns_none_for_missing_or_empty_folder(tmp_path):
    assert oldest_parseable_file(tmp_path / "does_not_exist") is None
    assert oldest_parseable_file(tmp_path) is None
