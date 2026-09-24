"""Unit tests for corpus_files.py: parsing and discovering `data/raw/`
filenames.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ye_olde.ingest.corpus_files import discover_corpus_files, parse_corpus_filename


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
