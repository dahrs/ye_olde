import pyarrow as pa
from fastapi.testclient import TestClient

from app import loader
from app.main import app

client = TestClient(app)

_ROWS = [
    {
        "iso_code": "enm",
        "form": "Robin",
        "lemma": "Robin",
        "sense_id": "ye_olde:robin.n.01",
        "attestation_year": 1400,
        "source": "Sir Gawayne",
        "doc_type": "attestation",
        "donor_language": None,
        "first_borrowing_year": None,
        "register": None,
        "dialect": None,
        "ner": "PER",
    },
    {
        "iso_code": "enm",
        "form": "light",
        "lemma": "light",
        "sense_id": "en:light.n.01",
        "attestation_year": 1400,
        "source": "Sir Gawayne",
        "doc_type": "attestation",
        "donor_language": None,
        "first_borrowing_year": None,
        "register": None,
        "dialect": None,
        "ner": None,
    },
]


def test_attest_filters_by_lemma(monkeypatch) -> None:
    monkeypatch.setattr(loader, "load_relational", lambda *a, **k: pa.Table.from_pylist(_ROWS))
    response = client.get("/attest", params={"lemma": "light", "lang": "enm", "year": 1400})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["form"] == "light"


def test_attest_ner_filter_finds_a_name_the_same_way_as_a_common_word(monkeypatch) -> None:
    # spec §3d/§4: a period name-form lookup is just this endpoint filtered
    # to ner=PER, not a separate registry subsystem.
    monkeypatch.setattr(loader, "load_relational", lambda *a, **k: pa.Table.from_pylist(_ROWS))
    response = client.get("/attest", params={"lemma": "Robin", "lang": "enm", "year": 1400, "ner": "PER"})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["ner"] == "PER"


def test_attest_ner_filter_excludes_a_matching_lemma_with_a_different_or_missing_ner(monkeypatch) -> None:
    monkeypatch.setattr(loader, "load_relational", lambda *a, **k: pa.Table.from_pylist(_ROWS))
    response = client.get("/attest", params={"lemma": "light", "lang": "enm", "year": 1400, "ner": "PER"})
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_attest_without_ner_param_returns_rows_regardless_of_ner(monkeypatch) -> None:
    monkeypatch.setattr(loader, "load_relational", lambda *a, **k: pa.Table.from_pylist(_ROWS))
    response = client.get("/attest", params={"lemma": "Robin", "lang": "enm", "year": 1400})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 1
