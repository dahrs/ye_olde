from fastapi.testclient import TestClient

from app import loader
from app.main import app

client = TestClient(app)


def test_search_shapes_loader_results_into_response(monkeypatch) -> None:
    fake_hits = [
        {
            "pair_id": "p2",
            "lang": "enm",
            "year": 1400,
            "work": "Sir Gawayne",
            "text": "Gladly sir for sothe",
            "score": 0.87,
            "other_lang": "eng",
            "other_year": 1999,
            "other_work": "Sir Gawayne",
            "other_text": "Gladly, sir, for sooth",
            "citation": "",
        }
    ]
    captured: dict[str, object] = {}

    def fake_search_passages(lang, text, *, year=None, window=None, top_k=5, upos=None, ner=None):
        captured.update(lang=lang, text=text, year=year, window=window, top_k=top_k, upos=upos, ner=ner)
        return fake_hits

    monkeypatch.setattr(loader, "search_passages", fake_search_passages)

    response = client.get("/search", params={"text": "gladly indeed", "lang": "enm", "year": 1400})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == {"text": "gladly indeed", "lang": "enm", "year": 1400, "upos": None, "ner": None}
    assert body["results"] == [
        {
            "pair_id": "p2",
            "lang": "enm",
            "year": 1400,
            "work": "Sir Gawayne",
            "text": "Gladly sir for sothe",
            "score": 0.87,
            "other_lang": "eng",
            "other_year": 1999,
            "other_work": "Sir Gawayne",
            "other_text": "Gladly, sir, for sooth",
            "citation": "",
            "matched_upos": None,
            "matched_ner": None,
            "matched_lemma": None,
        }
    ]
    assert captured == {
        "lang": "enm",
        "text": "gladly indeed",
        "year": 1400,
        "window": None,
        "top_k": 5,
        "upos": None,
        "ner": None,
    }


def test_search_returns_empty_results_when_nothing_indexed(monkeypatch) -> None:
    monkeypatch.setattr(loader, "search_passages", lambda *a, **k: [])

    response = client.get("/search", params={"text": "anything", "lang": "ang"})
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_search_passages_ranks_across_shards_and_pair_dirs(monkeypatch) -> None:
    """Exercises loader.search_passages itself (not just the endpoint):
    two shards, one of them for a lang pair discovered only via
    _lang_pair_dirs_containing, results merged and sorted by score. No
    `year` given -> pure FAISS similarity, the pre-hybrid-scoring behavior
    (see the hybrid-specific test below for the year-given path).
    """
    import faiss as faiss_lib
    import numpy as np
    import pyarrow as pa

    row_a = {
        "pair_id": "a1",
        "source": {"lang_code": "enm", "year": 1400, "work": "Work A"},
        "target": {"lang_code": "eng", "year": 1999, "work": "Work A"},
        "source_text": "close match",
        "target_text": "close match (modern)",
        "citation": None,
    }
    row_b = {
        "pair_id": "b1",
        "source": {"lang_code": "enm", "year": 1400, "work": "Work B"},
        "target": {"lang_code": "eng", "year": 1999, "work": "Work B"},
        "source_text": "far match",
        "target_text": "far match (modern)",
        "citation": None,
    }
    table_a = pa.Table.from_pylist([row_a])
    table_b = pa.Table.from_pylist([row_b])

    def make_index(vectors):
        idx = faiss_lib.IndexIDMap(faiss_lib.IndexFlatIP(vectors.shape[1]))
        idx.add_with_ids(vectors.astype("float32"), np.arange(len(vectors), dtype="int64"))
        return idx

    # row_a's source vector is identical to the query; row_b's is orthogonal.
    index_a = make_index(np.array([[1.0, 0.0], [1.0, 0.0]]))
    index_b = make_index(np.array([[0.0, 1.0], [0.0, 1.0]]))

    monkeypatch.setattr(loader, "_lang_pair_dirs_containing", lambda lang: [("enm", "eng")])
    monkeypatch.setattr(
        loader,
        "load_pair_shards",
        lambda a, b: [(table_a, index_a, "pairs/enm_eng/a.parquet"), (table_b, index_b, "pairs/enm_eng/b.parquet")],
    )
    monkeypatch.setattr(
        loader, "embed_query", lambda text, *, model_name, hf_token: np.array([[1.0, 0.0]], dtype="float32")
    )

    results = loader.search_passages("enm", "close match", top_k=5)
    assert [r.pair_id for r in results] == ["a1", "b1"]
    assert results[0].score > results[1].score
    assert results[0].other_text == "close match (modern)"


def test_search_passages_hybrid_blend_favors_lexical_for_an_ancient_query(monkeypatch) -> None:
    """With `year` given, a candidate that's a poor semantic match but a
    strong lexical (n-gram/BM25) match should win once the query's era is
    old enough that semantic_weight pushes most of the trust onto the
    lexical signals — the whole point of the hybrid design (spec §10).
    """
    import faiss as faiss_lib
    import numpy as np
    import pyarrow as pa

    row_semantic = {
        "pair_id": "semantic-match",
        "source": {"lang_code": "enm", "year": 1400, "work": "W"},
        "target": {"lang_code": "eng", "year": 1999, "work": "W"},
        "source_text": "completely different words",
        "target_text": "x",
        "citation": None,
        "sentence_confidence": 0.9,
        "alignment_links": [],
    }
    row_lexical = {
        "pair_id": "lexical-match",
        "source": {"lang_code": "enm", "year": 1400, "work": "W"},
        "target": {"lang_code": "eng", "year": 1999, "work": "W"},
        "source_text": "gladly sir for sothe",
        "target_text": "x",
        "citation": None,
        "sentence_confidence": 0.9,
        "alignment_links": [],
    }
    table = pa.Table.from_pylist([row_semantic, row_lexical])

    idx = faiss_lib.IndexIDMap(faiss_lib.IndexFlatIP(2))
    # row_semantic (id 0) is the closer FAISS match; row_lexical (id 2) is
    # semantically orthogonal but textually identical to the query.
    idx.add_with_ids(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype="float32"), np.arange(4))

    monkeypatch.setattr(loader, "_lang_pair_dirs_containing", lambda lang: [("enm", "eng")])
    monkeypatch.setattr(loader, "load_pair_shards", lambda a, b: [(table, idx, "pairs/enm_eng/1400-1999.parquet")])
    monkeypatch.setattr(
        loader, "embed_query", lambda text, *, model_name, hf_token: np.array([[1.0, 0.0]], dtype="float32")
    )
    monkeypatch.setattr(loader, "load_bm25_postings", lambda shard_path, terms: pa.table({}))

    # A genuinely ancient query year -> semantic_weight is near 0, so the
    # lexical signal should dominate despite the weaker FAISS score.
    results = loader.search_passages("enm", "gladly sir for sothe", year=600, top_k=2)
    assert results[0].pair_id == "lexical-match"


def test_search_passages_upos_ner_boost_a_tagged_match_past_a_slightly_closer_untagged_one(monkeypatch) -> None:
    """Spec §3d/§9's UD-tag wiring: a passage whose alignment_link carries
    the caller's queried `ner` gets `loader._TAG_MATCH_BOOST` (0.1) added
    to its score — enough to overtake a competitor with no tag at all
    whose plain FAISS similarity is only slightly higher (gap 0.05 < the
    0.1 boost), but not a filter: both still come back, and omitting
    upos/ner leaves the untagged row on top exactly as before this
    parameter existed.
    """
    import faiss as faiss_lib
    import numpy as np
    import pyarrow as pa

    row_untagged = {
        "pair_id": "untagged",
        "source": {"lang_code": "enm", "year": 1400, "work": "W"},
        "target": {"lang_code": "eng", "year": 1999, "work": "W"},
        "source_text": "a nameless passage",
        "target_text": "x",
        "citation": None,
        "alignment_links": [],
    }
    row_tagged = {
        "pair_id": "tagged",
        "source": {"lang_code": "enm", "year": 1400, "work": "W"},
        "target": {"lang_code": "eng", "year": 1999, "work": "W"},
        "source_text": "robin shot the arrow",
        "target_text": "x",
        "citation": None,
        "alignment_links": [{"source_idx": [0], "target_idx": [0], "sense_id": None, "source_ner": "PER"}],
    }
    table = pa.Table.from_pylist([row_untagged, row_tagged])

    idx = faiss_lib.IndexIDMap(faiss_lib.IndexFlatIP(2))
    # row_untagged (id 0) is slightly closer to the query than row_tagged
    # (id 2) -- 0.95 vs 0.90, a 0.05 gap smaller than the 0.1 tag boost.
    idx.add_with_ids(np.array([[0.95, 0.0], [0.0, 1.0], [0.90, 0.0], [0.0, 1.0]], dtype="float32"), np.arange(4))

    monkeypatch.setattr(loader, "_lang_pair_dirs_containing", lambda lang: [("enm", "eng")])
    monkeypatch.setattr(loader, "load_pair_shards", lambda a, b: [(table, idx, "pairs/enm_eng/1400-1999.parquet")])
    monkeypatch.setattr(
        loader, "embed_query", lambda text, *, model_name, hf_token: np.array([[1.0, 0.0]], dtype="float32")
    )

    unboosted = loader.search_passages("enm", "robin shot the arrow", top_k=2)
    assert unboosted[0].pair_id == "untagged"
    assert unboosted[0].matched_ner is None

    boosted = loader.search_passages("enm", "robin shot the arrow", top_k=2, ner="PER")
    assert boosted[0].pair_id == "tagged"
    assert boosted[0].matched_ner == "PER"
    assert len(boosted) == 2  # a non-matching candidate is still returned, just ranked lower -- a boost, not a filter

    # An empty-string upos alongside a real ner must not cancel the ner
    # boost out -- same main.attest-established convention as the test
    # above: "" means "not queried," not "match a link whose tag is the
    # literal empty string" (which no real tag ever is).
    boosted_with_empty_upos = loader.search_passages("enm", "robin shot the arrow", top_k=2, upos="", ner="PER")
    assert boosted_with_empty_upos[0].pair_id == "tagged"
    assert boosted_with_empty_upos[0].matched_ner == "PER"
