"""Unit tests for ingest/contributions.py — spec §5/§3a's "validated
contribution -> indexed attestation data" leg. `classify.classify` and
`resolve_sense_id` are monkeypatched throughout (both are already covered
by their own test files, `test_classify.py`/`test_sense.py`) — this file
tests this module's own orchestration, not their internals again.
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from jsonschema import ValidationError

from ye_olde import classify
from ye_olde.classify import TokenTag
from ye_olde.ingest import contributions


def _write(path: Path, entry: dict) -> None:
    path.write_text(json.dumps(entry), encoding="utf-8")


def _dictionary_entry(**overrides: object) -> dict:
    base = {
        "doc_type": "dictionary_entry",
        "origin": {
            "lang_code": "ang",
            "tokens": [
                {"surface": "lēoht", "alt": [], "class": None},
                {"surface": "bēam", "alt": ["beam"], "class": "common"},
            ],
        },
        "target": {"lang_code": "eng", "text": "ray of light"},
        "metadata": {
            "date": {"year": 1000, "precision": "circa"},
            "location": "Wessex, England",
            "source": "Bosworth-Toller",
            "author": None,
            "contributor": "handle",
            "license": "CC-BY-SA-4.0",
            "validation": {"status": "validated"},
        },
    }
    base.update(overrides)
    return base


def _tag(text: str, lemma: str | None, upos: str, ner: str = "O") -> TokenTag:
    return TokenTag(text=text, lemma=lemma, upos=upos, feats=None, deprel=None, ner=ner)


# ---------------------------------------------------------------------------
# load_contributions
# ---------------------------------------------------------------------------


def test_load_contributions_returns_only_validated_entries(tmp_path):
    _write(tmp_path / "a.json", _dictionary_entry())
    pending = _dictionary_entry()
    pending["metadata"]["validation"]["status"] = "pending"
    _write(tmp_path / "b.json", pending)

    loaded = contributions.load_contributions(tmp_path)

    assert len(loaded) == 1
    assert loaded[0]["metadata"]["validation"]["status"] == "validated"


def test_load_contributions_searches_recursively(tmp_path):
    nested = tmp_path / "ang" / "nested.json"
    nested.parent.mkdir(parents=True)
    _write(nested, _dictionary_entry())

    assert len(contributions.load_contributions(tmp_path)) == 1


def test_load_contributions_raises_for_a_schema_invalid_entry(tmp_path):
    invalid = _dictionary_entry()
    del invalid["metadata"]["source"]  # required by the schema
    _write(tmp_path / "bad.json", invalid)

    with pytest.raises(ValidationError):
        contributions.load_contributions(tmp_path)


def test_load_contributions_returns_empty_list_for_an_empty_directory(tmp_path):
    assert contributions.load_contributions(tmp_path) == []


# ---------------------------------------------------------------------------
# build_relational_row
# ---------------------------------------------------------------------------


def test_build_relational_row_skips_grammar_note():
    entry = {
        "doc_type": "grammar_note",
        "origin": {"lang_code": "ang", "text": "some prose"},
        "target": None,
        "metadata": {
            "date": {"year": 1000},
            "source": "x",
            "contributor": "x",
            "license": "x",
            "validation": {"status": "validated"},
        },
    }
    assert contributions.build_relational_row(entry) is None


def test_build_relational_row_skips_a_null_target():
    entry = _dictionary_entry(target=None)
    assert contributions.build_relational_row(entry) is None


def test_build_relational_row_skips_when_origin_has_no_tokens():
    entry = _dictionary_entry(origin={"lang_code": "ang", "text": "free-text origin, no tokens"})
    assert contributions.build_relational_row(entry) is None


def test_build_relational_row_skips_when_every_token_is_a_gap():
    entry = _dictionary_entry(origin={"lang_code": "ang", "tokens": [{"surface": None, "gap": True}]})
    assert contributions.build_relational_row(entry) is None


def test_build_relational_row_builds_a_name_entry_without_classify_or_sense_resolution(monkeypatch):
    monkeypatch.setattr(
        classify, "classify", lambda *a, **k: (_ for _ in ()).throw(AssertionError("names bypass classify"))
    )
    monkeypatch.setattr(
        contributions, "resolve_sense_id", lambda *a, **k: (_ for _ in ()).throw(AssertionError("names bypass sense"))
    )
    entry = _dictionary_entry(
        origin={"lang_code": "ang", "tokens": [{"surface": "Beowulf", "class": "name"}]},
        target={"lang_code": "eng", "text": "Beowulf"},
    )

    row = contributions.build_relational_row(entry)

    assert row is not None
    assert row["form"] == "Beowulf"
    assert row["lemma"] == "Beowulf"
    assert row["ner"] == "PER"
    assert row["sense_id"] == "name:Beowulf"


def test_build_relational_row_builds_a_common_word_entry_via_classify_and_sense(monkeypatch):
    monkeypatch.setattr(
        classify, "classify", lambda text, lang, year, *, model=None: [_tag("lēoht", "lēoht", "NOUN", "O")]
    )
    seen_gloss = {}

    def fake_resolve_sense_id(lemma, upos, gloss, *, model_name=None):
        seen_gloss.update(lemma=lemma, upos=upos, gloss=gloss)
        return "en:light.n.01", "illumination"

    monkeypatch.setattr(contributions, "resolve_sense_id", fake_resolve_sense_id)

    row = contributions.build_relational_row(_dictionary_entry())

    assert row is not None
    assert row["form"] == "lēoht bēam"
    assert row["lemma"] == "lēoht"
    assert row["sense_id"] == "en:light.n.01"
    assert row["iso_code"] == "ang"
    assert row["attestation_year"] == 1000
    assert row["source"] == "Bosworth-Toller"
    assert row["doc_type"] == "dictionary_entry"
    assert row["donor_language"] is None
    assert seen_gloss == {"lemma": "lēoht", "upos": "NOUN", "gloss": "ray of light"}


def test_build_relational_row_picks_the_first_content_tag_skipping_punct_and_unverifiable(monkeypatch):
    monkeypatch.setattr(
        classify,
        "classify",
        lambda text, lang, year, *, model=None: [
            _tag(",", None, "PUNCT"),
            _tag("x", None, "X"),
            _tag("lēoht", "lēoht", "NOUN"),
        ],
    )
    monkeypatch.setattr(contributions, "resolve_sense_id", lambda *a, **k: ("en:light.n.01", "d"))

    row = contributions.build_relational_row(_dictionary_entry())

    assert row is not None
    assert row["lemma"] == "lēoht"  # the PUNCT/X tags were skipped, not mistaken for the real content token


def test_build_relational_row_falls_back_to_form_when_classify_returns_no_tags(monkeypatch):
    monkeypatch.setattr(classify, "classify", lambda text, lang, year, *, model=None: [])
    monkeypatch.setattr(contributions, "resolve_sense_id", lambda *a, **k: ("ye_olde:x.n.01", "d"))

    row = contributions.build_relational_row(_dictionary_entry())

    assert row is not None
    assert row["lemma"] == "lēoht bēam"  # falls back to the full surface form
    assert row["ner"] == "O"


# ---------------------------------------------------------------------------
# build_relational_shards
# ---------------------------------------------------------------------------


def test_build_relational_shards_groups_by_iso_code(monkeypatch, tmp_path):
    monkeypatch.setattr(classify, "classify", lambda text, lang, year, *, model=None: [_tag(text, text, "NOUN")])
    monkeypatch.setattr(contributions, "resolve_sense_id", lambda *a, **k: ("en:x.n.01", "d"))
    ang_entry = _dictionary_entry()
    enm_entry = _dictionary_entry(
        origin={"lang_code": "enm", "tokens": [{"surface": "gladly"}]},
        target={"lang_code": "eng", "text": "gladly"},
    )

    written = contributions.build_relational_shards([ang_entry, enm_entry], tmp_path)

    assert {p.parent.name for p in written} == {"ang", "enm"}
    for path in written:
        assert path.parent.parent.name == "relational"


def test_build_relational_shards_names_the_shard_by_year_range(monkeypatch, tmp_path):
    monkeypatch.setattr(classify, "classify", lambda text, lang, year, *, model=None: [_tag(text, text, "NOUN")])
    monkeypatch.setattr(contributions, "resolve_sense_id", lambda *a, **k: ("en:x.n.01", "d"))
    early = _dictionary_entry()
    early["metadata"]["date"]["year"] = 900
    late = _dictionary_entry(origin={"lang_code": "ang", "tokens": [{"surface": "other"}]})
    late["metadata"]["date"]["year"] = 1050

    written = contributions.build_relational_shards([early, late], tmp_path)

    assert len(written) == 1
    assert written[0].name == "900-1050.parquet"


def test_build_relational_shards_writes_rows_readable_as_parquet(monkeypatch, tmp_path):
    monkeypatch.setattr(classify, "classify", lambda text, lang, year, *, model=None: [_tag(text, text, "NOUN")])
    monkeypatch.setattr(contributions, "resolve_sense_id", lambda *a, **k: ("en:light.n.01", "d"))

    written = contributions.build_relational_shards([_dictionary_entry()], tmp_path)

    table = pq.read_table(written[0])
    assert table.num_rows == 1
    assert table.column("sense_id").to_pylist() == ["en:light.n.01"]


def test_build_relational_shards_skips_unusable_entries_without_writing_an_empty_file(monkeypatch, tmp_path):
    monkeypatch.setattr(classify, "classify", lambda text, lang, year, *, model=None: [_tag(text, text, "NOUN")])
    monkeypatch.setattr(contributions, "resolve_sense_id", lambda *a, **k: ("en:x.n.01", "d"))
    grammar_note = {
        "doc_type": "grammar_note",
        "origin": {"lang_code": "ang", "text": "prose"},
        "target": None,
        "metadata": {
            "date": {"year": 1000},
            "source": "x",
            "contributor": "x",
            "license": "x",
            "validation": {"status": "validated"},
        },
    }

    written = contributions.build_relational_shards([grammar_note, _dictionary_entry()], tmp_path)

    assert len(written) == 1  # only the usable dictionary_entry produced a shard


def test_build_relational_shards_returns_empty_list_for_no_entries(tmp_path):
    assert contributions.build_relational_shards([], tmp_path) == []


# ---------------------------------------------------------------------------
# build_relational_index_for_contributions_root
# ---------------------------------------------------------------------------


def test_build_relational_index_returns_empty_list_when_nothing_is_validated(tmp_path):
    pending = _dictionary_entry()
    pending["metadata"]["validation"]["status"] = "pending"
    _write(tmp_path / "a.json", pending)

    assert contributions.build_relational_index_for_contributions_root(tmp_path, tmp_path / "out") == []


def test_build_relational_index_builds_and_writes_real_shards(monkeypatch, tmp_path):
    monkeypatch.setattr(classify, "classify", lambda text, lang, year, *, model=None: [_tag(text, text, "NOUN")])
    monkeypatch.setattr(contributions, "resolve_sense_id", lambda *a, **k: ("en:light.n.01", "d"))
    _write(tmp_path / "entry.json", _dictionary_entry())

    written = contributions.build_relational_index_for_contributions_root(tmp_path, tmp_path / "out")

    assert len(written) == 1
    assert written[0].is_file()
