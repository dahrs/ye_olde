import pyarrow as pa
from fastapi.testclient import TestClient

from app import loader
from app.main import app

client = TestClient(app)

_EMPTY_TABLE = pa.table({})


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_attest_empty_without_data(monkeypatch) -> None:
    # Explicitly isolated (not just relying on an unconfigured HF_DATASET_REPO_ID
    # in .env — that assumption broke the moment a real deployment populated it)
    # so this test's meaning ("no data yet") holds regardless of the real .env's
    # current state.
    monkeypatch.setattr(loader, "load_relational", lambda *a, **k: _EMPTY_TABLE)
    response = client.get("/attest", params={"lemma": "light", "lang": "eng", "year": 1850})
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_lookup_empty_without_data(monkeypatch) -> None:
    monkeypatch.setattr(loader, "load_pairs", lambda *a, **k: _EMPTY_TABLE)
    response = client.get(
        "/lookup",
        params={"text": "light", "lang": "eng", "year": 2026, "target_lang": "enm", "target_year": 1382},
    )
    assert response.status_code == 200
    assert response.json()["results"] == []
