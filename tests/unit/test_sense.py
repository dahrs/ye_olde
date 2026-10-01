"""Unit tests for sense.py's embedding-based sense_id resolution. Real
sentence-transformers/nltk are never touched here — `_embed_gloss` and
`_load_wordnet_index` are monkeypatched to small fakes, the same
"no network/heavy-model access needed" pattern `common.llm_client`'s tests
use for litellm.
"""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from ye_olde.ingest import sense


@pytest.fixture(autouse=True)
def _reset_mint_counters():
    sense._mint_counters.clear()
    yield
    sense._mint_counters.clear()


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


# Three orthonormal-ish directions so "close to A" vs "close to B" is
# unambiguous regardless of exact values.
_LIGHT_DIR = _unit([1.0, 0.0, 0.0])
_CIGARETTE_LIGHTER_DIR = _unit([0.0, 1.0, 0.0])
_UNRELATED_DIR = _unit([0.0, 0.0, 1.0])


def _wordnet_index() -> tuple[np.ndarray, list[str], list[str], list[str]]:
    embeddings = np.stack([_LIGHT_DIR, _CIGARETTE_LIGHTER_DIR])
    names = ["light.n.01", "light.n.02"]
    definitions = ["the visual effect of illumination on objects", "a device for lighting a cigarette"]
    pos_tags = ["n", "n"]
    return embeddings, names, definitions, pos_tags


def test_resolve_sense_id_matches_the_closest_wordnet_synset(monkeypatch, tmp_path):
    monkeypatch.setattr(sense, "_embed_gloss", lambda gloss, *, model_name: _LIGHT_DIR)
    monkeypatch.setattr(sense, "_load_wordnet_index", lambda model_name, root: _wordnet_index())

    sense_id, definition = sense.resolve_sense_id("light", "NOUN", "illumination that lets one see", root=tmp_path)
    assert sense_id == "en:light.n.01"
    assert "illumination" in definition


def test_resolve_sense_id_mints_when_below_similarity_threshold(monkeypatch, tmp_path):
    monkeypatch.setattr(sense, "_embed_gloss", lambda gloss, *, model_name: _UNRELATED_DIR)
    monkeypatch.setattr(sense, "_load_wordnet_index", lambda model_name, root: _wordnet_index())

    sense_id, definition = sense.resolve_sense_id("wergild", "NOUN", "a payment owed for a killing", root=tmp_path)
    assert sense_id == "ye_olde:wergild.n.01"
    assert definition == "a payment owed for a killing"


def test_resolve_sense_id_mints_when_wordnet_index_not_built_yet(monkeypatch, tmp_path):
    monkeypatch.setattr(sense, "_embed_gloss", lambda gloss, *, model_name: _LIGHT_DIR)
    monkeypatch.setattr(sense, "_load_wordnet_index", lambda model_name, root: None)  # not built yet

    sense_id, definition = sense.resolve_sense_id("light", "NOUN", "illumination", root=tmp_path)
    assert sense_id == "ye_olde:light.n.01"
    assert definition == "illumination"


def test_resolve_sense_id_persists_a_minted_sense_and_reuses_it_next_time(monkeypatch, tmp_path):
    monkeypatch.setattr(sense, "_embed_gloss", lambda gloss, *, model_name: _UNRELATED_DIR)
    monkeypatch.setattr(sense, "_load_wordnet_index", lambda model_name, root: None)

    first_id, _ = sense.resolve_sense_id("wergild", "NOUN", "a payment owed for a killing", root=tmp_path)
    assert first_id == "ye_olde:wergild.n.01"

    # A second, unrelated call in between must not shift the counter used below.
    sense.resolve_sense_id("bloodmoney", "NOUN", "compensation paid to a victim's kin", root=tmp_path)

    # A near-identical gloss for the *same* concept, in a totally separate
    # call (as if from a different sentence pair elsewhere in the corpus) --
    # must reuse the existing minted id, not mint a third one.
    second_id, second_def = sense.resolve_sense_id("were_gyld", "NOUN", "a payment owed after a killing", root=tmp_path)
    assert second_id == "ye_olde:wergild.n.01"
    assert second_def == "a payment owed for a killing"  # the *original* minted gloss, not the new call's


def test_resolve_sense_id_closed_class_upos_mints_without_touching_embeddings_or_index(monkeypatch, tmp_path):
    def _boom_embed(gloss, *, model_name):
        raise AssertionError("should not embed a closed-class gloss")

    def _boom_index(model_name, root):
        raise AssertionError("should not load the WordNet index for a closed-class UPOS")

    monkeypatch.setattr(sense, "_embed_gloss", _boom_embed)
    monkeypatch.setattr(sense, "_load_wordnet_index", _boom_index)

    sense_id, definition = sense.resolve_sense_id("the", "DET", "definite article", root=tmp_path)
    assert sense_id == "ye_olde:the.x.01"
    assert definition == "definite article"
    # nothing persisted for a closed-class mint either
    assert sense._load_minted_index("any-model", tmp_path) == []


def test_resolve_sense_id_repeated_mints_of_distinct_concepts_increment_counter(monkeypatch, tmp_path):
    monkeypatch.setattr(sense, "_embed_gloss", lambda gloss, *, model_name: _UNRELATED_DIR)
    monkeypatch.setattr(sense, "_load_wordnet_index", lambda model_name, root: None)

    # two glosses far enough apart in embedding space that neither matches
    # the other's minted entry -- distinct concepts, same lemma slug.
    first, _ = sense.resolve_sense_id("thane", "NOUN", "a minor noble", root=tmp_path)
    monkeypatch.setattr(sense, "_embed_gloss", lambda gloss, *, model_name: -_UNRELATED_DIR)
    second, _ = sense.resolve_sense_id("thane", "NOUN", "an unrelated second sense", root=tmp_path)
    assert first == "ye_olde:thane.n.01"
    assert second == "ye_olde:thane.n.02"


def test_index_is_complete_false_when_files_missing(tmp_path):
    assert sense._index_is_complete("some-model", tmp_path) is False


def test_index_is_complete_false_while_progress_file_present(tmp_path):
    npy_path, json_path, progress_path = sense._index_paths("some-model", tmp_path)
    npy_path.write_bytes(b"")
    json_path.write_text("{}")
    progress_path.write_text('{"total": 10, "dim": 4, "completed": 5}')
    assert sense._index_is_complete("some-model", tmp_path) is False


def test_index_is_complete_true_once_progress_file_removed(tmp_path):
    npy_path, json_path, _progress_path = sense._index_paths("some-model", tmp_path)
    npy_path.write_bytes(b"")
    json_path.write_text("{}")
    assert sense._index_is_complete("some-model", tmp_path) is True


class _FakeSynset:
    def __init__(self, name: str, definition: str):
        self._name = name
        self._definition = definition

    def name(self) -> str:
        return self._name

    def definition(self) -> str:
        return self._definition


def _install_fake_nltk(monkeypatch, synsets_by_pos: dict):
    fake_wordnet = types.SimpleNamespace(all_synsets=lambda pos: synsets_by_pos.get(pos, []))
    fake_corpus = types.SimpleNamespace(wordnet=fake_wordnet)
    fake_data = types.SimpleNamespace(find=lambda path: None)
    fake_nltk = types.SimpleNamespace(data=fake_data, download=lambda *a, **k: None, corpus=fake_corpus)
    monkeypatch.setitem(sys.modules, "nltk", fake_nltk)
    monkeypatch.setitem(sys.modules, "nltk.corpus", fake_corpus)


def test_build_wordnet_index_writes_complete_index(monkeypatch, tmp_path):
    synsets = {
        "n": [
            _FakeSynset("cat.n.01", "a small domesticated carnivore"),
            _FakeSynset("dog.n.01", "a domesticated canid"),
        ],
        "v": [],
        "a": [],
        "r": [],
    }
    _install_fake_nltk(monkeypatch, synsets)
    monkeypatch.setattr(sense, "embed_units", lambda units, *, model_name: np.ones((len(units), 4), dtype=np.float32))

    sense.build_wordnet_index(model_name="fake-model", root=tmp_path, batch_size=64)

    npy_path, json_path, progress_path = sense._index_paths("fake-model", tmp_path)
    assert npy_path.is_file()
    assert json_path.is_file()
    assert not progress_path.is_file()  # deleted -- build is complete
    assert sense._index_is_complete("fake-model", tmp_path) is True

    meta = json.loads(json_path.read_text())
    assert meta["names"] == ["cat.n.01", "dog.n.01"]


class _FakeDiskUsage:
    def __init__(self, free: int):
        self.free = free


def test_build_wordnet_index_resumes_without_re_embedding_done_rows(monkeypatch, tmp_path):
    synsets = {"n": [_FakeSynset(f"word{i}.n.01", f"definition {i}") for i in range(6)], "v": [], "a": [], "r": []}
    _install_fake_nltk(monkeypatch, synsets)

    embed_calls: list[list[str]] = []

    def fake_embed_units(units, *, model_name):
        embed_calls.append(list(units))
        return np.ones((len(units), 4), dtype=np.float32)

    monkeypatch.setattr(sense, "embed_units", fake_embed_units)

    # First call processes only the first batch (batch_size=2), then we
    # simulate an interruption by not letting it finish -- min_free_gb
    # forces it to stop cleanly after one batch.
    monkeypatch.setattr(sense.shutil, "disk_usage", lambda path: _FakeDiskUsage(free=0))
    sense.build_wordnet_index(model_name="fake-model", root=tmp_path, batch_size=2, min_free_gb=1.0)

    _npy_path, _json_path, progress_path = sense._index_paths("fake-model", tmp_path)
    assert progress_path.is_file()  # stopped early -- still incomplete
    progress = json.loads(progress_path.read_text())
    assert progress["completed"] == 2
    # embed_units is called once up front just to probe the embedding
    # dimension (a single-item call), then once per batch actually embedded.
    assert embed_calls == [["definition 0"], ["definition 0", "definition 1"]]

    # Now let it run to completion with plenty of free space -- it must
    # resume from row 2, not re-embed rows 0-1 (past the dim-probe call).
    embed_calls.clear()
    monkeypatch.setattr(sense.shutil, "disk_usage", lambda path: _FakeDiskUsage(free=999 * 1024**3))
    sense.build_wordnet_index(model_name="fake-model", root=tmp_path, batch_size=2, min_free_gb=1.0)

    assert not progress_path.is_file()  # now complete
    assert embed_calls == [
        ["definition 0"],
        ["definition 2", "definition 3"],
        ["definition 4", "definition 5"],
    ]


def test_ensure_wordnet_index_ready_skips_build_when_already_complete(monkeypatch, tmp_path):
    npy_path, json_path, _progress_path = sense._index_paths("fake-model", tmp_path)
    npy_path.write_bytes(b"")
    json_path.write_text("{}")

    def fail_if_called(**k):
        raise AssertionError("build_wordnet_index should not run when the index is already complete")

    monkeypatch.setattr(sense, "build_wordnet_index", fail_if_called)

    sense.ensure_wordnet_index_ready(model_name="fake-model", root=tmp_path)


def test_ensure_wordnet_index_ready_builds_when_missing(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sense, "build_wordnet_index", lambda **k: calls.append(k))

    sense.ensure_wordnet_index_ready(model_name="fake-model", root=tmp_path)

    assert len(calls) == 1
    assert calls[0]["model_name"] == "fake-model"
    assert calls[0]["root"] == tmp_path


def test_ensure_wordnet_index_ready_builds_when_left_incomplete(monkeypatch, tmp_path):
    npy_path, json_path, progress_path = sense._index_paths("fake-model", tmp_path)
    npy_path.write_bytes(b"")
    json_path.write_text("{}")
    progress_path.write_text('{"total": 10, "dim": 4, "completed": 3}')

    calls = []
    monkeypatch.setattr(sense, "build_wordnet_index", lambda **k: calls.append(k))

    sense.ensure_wordnet_index_ready(model_name="fake-model", root=tmp_path)

    assert len(calls) == 1
