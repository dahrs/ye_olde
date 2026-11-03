"""Unit tests for corpus_files.py: parsing and discovering `data/raw/`
filenames.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ye_olde.ingest.corpus_files import (
    authors_plausibly_match,
    build_corpus_filename,
    discover_corpus_files,
    find_matching_work_folder,
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


def test_build_corpus_filename_renders_negative_int_year_as_bc_token():
    filename = build_corpus_filename(
        lang_code="lat", year=-19, title="Aeneid", author="Virgil", source="Perseus", ext=".txt"
    )
    assert filename == "lat-BC19-Aeneid-Virgil-Perseus.txt"
    cf = parse_corpus_filename(Path(filename))
    assert cf.year == -19


def test_build_corpus_filename_accepts_bc_token_string_directly():
    filename = build_corpus_filename(
        lang_code="grc", year="BC800", title="Iliad", author="Homer", source="Perseus", ext=".txt"
    )
    cf = parse_corpus_filename(Path(filename))
    assert cf.year == -800


def test_parse_corpus_filename_bc_token_sorts_older_than_any_ad_year():
    bc = parse_corpus_filename(Path("lat-BC19-Aeneid-Virgil-Perseus.txt"))
    ad = parse_corpus_filename(Path("lat-523-De_Consolatione-Boethius-Archive_org.txt"))
    assert bc.year < ad.year


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


def test_authors_plausibly_match_substring_and_unknown_cases():
    assert authors_plausibly_match("Vergil", "P. Vergilius Maro")
    assert authors_plausibly_match("unknown", "unknown")
    assert not authors_plausibly_match("unknown", "Tibullus")  # placeholder never matches a real name
    assert not authors_plausibly_match("Tibullus", "Sextus Propertius")  # unrelated authors
    assert not authors_plausibly_match("", "Tibullus")


def test_find_matching_work_folder_requires_title_and_author_both(tmp_path):
    """Caught live: matching on title alone silently merged Tibullus's
    "Elegiae" with Propertius's "Elegiae" -- two different, unrelated
    authors' two different works, not the same text from a new source.
    """
    folder = tmp_path / "lat-BC26-Elegiae-Tibullus-Perseus_Digital_Library"
    folder.mkdir()
    (folder / "lat-BC26-Elegiae-Tibullus-Perseus_Digital_Library.txt").write_text("tibullus text")

    assert find_matching_work_folder(tmp_path, "Elegiae", "Sextus Propertius") is None
    assert find_matching_work_folder(tmp_path, "Elegiae", "Tibullus") == folder


def test_find_matching_work_folder_handles_non_latin_titles(tmp_path):
    """Caught live: ASCII-only title normalization collapsed any
    non-Latin-script title (e.g. Greek) to an empty string, making every
    such title look identical to every other one.
    """
    folder = tmp_path / "grc-BC270-work"
    folder.mkdir()
    (folder / "grc-BC270-Ἰδύλλια-Theocritus-Perseus_Digital_Library.txt").write_text("idylls")

    assert find_matching_work_folder(tmp_path, "Ἰδύλλια", "Theocritus") == folder
    assert find_matching_work_folder(tmp_path, "Ἐπιγράμματα", "Theocritus") is None


def test_find_matching_work_folder_matches_against_the_oldest_file_not_any_sibling(tmp_path):
    """A translation's own `author` field is its translator, not the
    work's original author -- matching must use the folder's oldest file
    (the original-language edition) as the identity anchor, not whichever
    sibling file happens to be iterated.
    """
    folder = tmp_path / "lat-523-De_Consolatione_Philosophiae-Boethius-Source"
    folder.mkdir()
    (folder / "lat-523-De_Consolatione_Philosophiae-Boethius-Source.txt").write_text("original")
    (folder / "eng-1897-De_Consolatione_Philosophiae-H_R_James-Source.txt").write_text("translation")

    # a NEW translation by a third, different translator should still find this folder, by
    # matching against the original author (Boethius), not the existing translator (H. R. James)
    assert find_matching_work_folder(tmp_path, "De Consolatione Philosophiae", "Boethius") == folder
