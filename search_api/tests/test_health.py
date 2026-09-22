from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_attest_empty_without_data() -> None:
    response = client.get("/attest", params={"lemma": "light", "lang": "eng", "year": 1850})
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_lookup_empty_without_data() -> None:
    response = client.get(
        "/lookup",
        params={"text": "light", "lang": "eng", "year": 2026, "target_lang": "enm", "target_year": 1382},
    )
    assert response.status_code == 200
    assert response.json()["results"] == []
