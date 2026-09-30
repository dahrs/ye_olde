from __future__ import annotations

from ye_olde.common import function_words


def test_load_returns_empty_for_unseen_lang_year(tmp_path):
    assert function_words.load("enm", 1400, root=tmp_path) == {}


def test_record_and_load_round_trip(tmp_path):
    function_words.record("enm", 1400, "þe", "DET", root=tmp_path)
    assert function_words.load("enm", 1400, root=tmp_path) == {"þe": "DET"}


def test_record_ignores_open_class_upos(tmp_path):
    function_words.record("enm", 1400, "liyt", "NOUN", root=tmp_path)
    assert function_words.load("enm", 1400, root=tmp_path) == {}


def test_record_ignores_empty_surface_form(tmp_path):
    function_words.record("enm", 1400, "   ", "DET", root=tmp_path)
    assert function_words.load("enm", 1400, root=tmp_path) == {}


def test_record_is_case_insensitive_on_the_key(tmp_path):
    function_words.record("enm", 1400, "The", "DET", root=tmp_path)
    assert function_words.load("enm", 1400, root=tmp_path) == {"the": "DET"}


def test_known_upos_returns_none_when_not_recorded(tmp_path):
    assert function_words.known_upos("enm", 1400, "þe", root=tmp_path) is None


def test_known_upos_returns_recorded_tag(tmp_path):
    function_words.record("enm", 1400, "þe", "DET", root=tmp_path)
    assert function_words.known_upos("enm", 1400, "þe", root=tmp_path) == "DET"
    assert function_words.known_upos("enm", 1400, "ÞE", root=tmp_path) == "DET"  # case-insensitive lookup too


def test_registries_are_scoped_per_lang_and_year(tmp_path):
    function_words.record("enm", 1400, "the", "DET", root=tmp_path)
    assert function_words.known_upos("eng", 1400, "the", root=tmp_path) is None
    assert function_words.known_upos("enm", 1999, "the", root=tmp_path) is None


def test_record_is_idempotent(tmp_path):
    function_words.record("enm", 1400, "the", "DET", root=tmp_path)
    path = tmp_path / "enm" / "1400.json"
    mtime_before = path.stat().st_mtime_ns
    function_words.record("enm", 1400, "the", "DET", root=tmp_path)  # same form/upos again
    assert path.stat().st_mtime_ns == mtime_before  # file not rewritten
