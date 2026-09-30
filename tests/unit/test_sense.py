"""Unit tests for sense.py's embedding-based sense_id resolution. Real
sentence-transformers/nltk are never touched here — `_embed_gloss` and
`_load_wordnet_index` are monkeypatched to small fakes, the same
"no network/heavy-model access needed" pattern `common.llm_client`'s tests
use for litellm.
"""

from __future__ import annotations

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
