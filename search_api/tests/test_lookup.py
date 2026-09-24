import pyarrow as pa

from app import loader
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

_ROWS = [
    {
        "pair_id": "p1",
        "source_text": "gladly would I see him",
        "target_text": "Gladly I would see that hero",
        "source_tokens": ["gladly", "would", "I", "see", "him"],
        "target_tokens": ["Gladly", "I", "would", "see", "that", "hero"],
        "citation": None,
        "sentence_confidence": 0.9,
        "alignment_links": [],
    },
    {
        "pair_id": "p2",
        "source_text": "Gladly sir for sothe",
        "target_text": "Gladly, sir, for sooth",
        "source_tokens": ["Gladly", "sir", "for", "sothe"],
        "target_tokens": ["Gladly", ",", "sir", ",", "for", "sooth"],
        "citation": None,
        "sentence_confidence": 0.95,
        "alignment_links": [{"source_idx": [0], "target_idx": [0], "sense_id": None}],
    },
]


def test_lookup_falls_back_to_unhighlighted_match_without_alignment_links(monkeypatch) -> None:
    monkeypatch.setattr(loader, "load_pairs", lambda *a, **k: pa.Table.from_pylist(_ROWS))

    response = client.get(
        "/lookup",
        params={"text": "gladly", "lang": "enm", "year": 1400, "target_lang": "eng", "target_year": 1999},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2

    # p1 has no alignment_links — still returned, but with no highlight.
    p1 = next(r for r in results if r["pair_id"] == "p1")
    assert p1["highlighted_span"] is None
    assert p1["citation"] == ""  # null citation, not the literal None Python would give via .get(k, default)

    # p2 has a real link — the precise span is highlighted.
    p2 = next(r for r in results if r["pair_id"] == "p2")
    assert p2["highlighted_span"] == {"token_idx": [0], "surface": "Gladly"}
