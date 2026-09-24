import pyarrow as pa
from fastapi.testclient import TestClient

from app import loader
from app.main import app

client = TestClient(app)

_ROWS = [
    {
        "pair_id": "p1",
        "source": {"lang_code": "enm", "year": 1400},
        "target": {"lang_code": "eng", "year": 1999},
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
        "source": {"lang_code": "enm", "year": 1400},
        "target": {"lang_code": "eng", "year": 1999},
        "source_text": "Gladly sir for sothe",
        "target_text": "Gladly, sir, for sooth",
        "source_tokens": ["Gladly", "sir", "for", "sothe"],
        "target_tokens": ["Gladly", ",", "sir", ",", "for", "sooth"],
        "citation": None,
        "sentence_confidence": 0.95,
        "alignment_links": [
            {"source_idx": [0], "target_idx": [0], "sense_id": None},
            {"source_idx": [3], "target_idx": [5], "sense_id": None},  # sothe <-> sooth
        ],
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


def test_lookup_handles_reverse_direction(monkeypatch) -> None:
    # These pairs were stored source=enm/target=eng (as the alignment job ran
    # them), but a query asking eng -> enm (the reverse) still needs to match
    # against the modern-English "sooth" and highlight the enm "sothe" — it
    # must not silently search the enm side for an English spelling.
    monkeypatch.setattr(loader, "load_pairs", lambda *a, **k: pa.Table.from_pylist(_ROWS))

    response = client.get(
        "/lookup",
        params={"text": "sooth", "lang": "eng", "year": 1999, "target_lang": "enm", "target_year": 1400},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1

    result = results[0]
    assert result["pair_id"] == "p2"
    # source_sentence is now the query side (eng), target_sentence the enm side.
    assert result["source_sentence"] == "Gladly, sir, for sooth"
    assert result["target_sentence"] == "Gladly sir for sothe"
    assert result["highlighted_span"] == {"token_idx": [3], "surface": "sothe"}
