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

    def fake_search_passages(lang, text, *, year=None, window=50, top_k=5):
        captured.update(lang=lang, text=text, year=year, window=window, top_k=top_k)
        return fake_hits

    monkeypatch.setattr(loader, "search_passages", fake_search_passages)

    response = client.get("/search", params={"text": "gladly indeed", "lang": "enm", "year": 1400})
    assert response.status_code == 200
    body = response.json()
    assert body["query"] == {"text": "gladly indeed", "lang": "enm", "year": 1400}
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
        }
    ]
    assert captured == {"lang": "enm", "text": "gladly indeed", "year": 1400, "window": 50, "top_k": 5}


def test_search_returns_empty_results_when_nothing_indexed(monkeypatch) -> None:
    monkeypatch.setattr(loader, "search_passages", lambda *a, **k: [])

    response = client.get("/search", params={"text": "anything", "lang": "ang"})
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_search_passages_ranks_across_shards_and_pair_dirs(monkeypatch) -> None:
    """Exercises loader.search_passages itself (not just the endpoint):
    two shards, one of them for a lang pair discovered only via
    _lang_pair_dirs_containing, results merged and sorted by score.
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
    monkeypatch.setattr(loader, "load_pair_shards_with_vectors", lambda a, b: [(table_a, index_a), (table_b, index_b)])
    monkeypatch.setattr(
        loader, "embed_query", lambda text, *, model_name, hf_token: np.array([[1.0, 0.0]], dtype="float32")
    )

    results = loader.search_passages("enm", "close match", top_k=5)
    assert [r.pair_id for r in results] == ["a1", "b1"]
    assert results[0].score > results[1].score
    assert results[0].other_text == "close match (modern)"
