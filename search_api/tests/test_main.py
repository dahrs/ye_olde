"""Unit tests for app/main.py that aren't already covered by
test_health.py/test_attest.py/test_lookup.py/test_search.py: the
catch-all `unhandled_exception_handler` (never exercised elsewhere), the
full `/health` response shape, and the `/lookup`/`/search` query-echo
fields (those files assert on `results`, not on the echoed `query`/
`target` objects).
"""

from __future__ import annotations

import logging

import pyarrow as pa
import pytest
from fastapi.testclient import TestClient

from app import loader
from app.config import Settings
from app.main import app

client = TestClient(app)
# A route that raises must come back as this service's own opaque 500,
# not propagate into the test process -- TestClient's default
# `raise_server_exceptions=True` would otherwise bypass the registered
# handler entirely and re-raise in-process.
_non_raising_client = TestClient(app, raise_server_exceptions=False)


def test_health_echoes_the_configured_hf_dataset_repo_id(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.main.get_settings", lambda: Settings(_env_file=None, hf_dataset_repo_id="dahrs/ye_olde_data-index")
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "hf_dataset_repo_id": "dahrs/ye_olde_data-index"}


def test_unhandled_exception_returns_an_opaque_500(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("boom -- some unexpected internal failure")

    monkeypatch.setattr(loader, "load_relational", boom)

    response = _non_raising_client.get("/attest", params={"lemma": "x", "lang": "enm", "year": 1400})

    assert response.status_code == 500
    assert response.json() == {"detail": "internal server error"}


def test_unhandled_exception_is_logged_with_the_request_path(monkeypatch, caplog) -> None:
    # `_log` in main.py is bound once at import time to the real "search_api"
    # logger -- `caplog` taps the standard logging propagation chain rather
    # than trying to redirect that already-bound logger's own handler.
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(loader, "load_relational", boom)

    with caplog.at_level(logging.ERROR, logger="search_api"):
        _non_raising_client.get("/attest", params={"lemma": "x", "lang": "enm", "year": 1400})

    assert "/attest" in caplog.text
    assert "RuntimeError" in caplog.text


def test_unhandled_exception_from_a_different_route_still_returns_500(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(loader, "lookup_passages", boom)

    response = _non_raising_client.get(
        "/lookup",
        params={"text": "light", "lang": "eng", "year": 2026, "target_lang": "enm", "target_year": 1382},
    )

    assert response.status_code == 500


def test_lookup_echoes_query_and_target(monkeypatch) -> None:
    monkeypatch.setattr(loader, "lookup_passages", lambda *a, **k: [])

    response = client.get(
        "/lookup",
        params={
            "text": "light",
            "lang": "eng",
            "year": 2026,
            "target_lang": "enm",
            "target_year": 1382,
            "upos": "NOUN",
        },
    )

    body = response.json()
    assert body["query"] == {"text": "light", "lang": "eng", "year": 2026, "upos": "NOUN", "ner": None}
    assert body["target"] == {"lang": "enm", "year": 1382}


def test_search_echoes_query_and_passes_parameters_through(monkeypatch) -> None:
    seen_kwargs: dict[str, object] = {}

    def fake_search_passages(lang, text, **kwargs):
        seen_kwargs["lang"] = lang
        seen_kwargs["text"] = text
        seen_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(loader, "search_passages", fake_search_passages)

    response = client.get("/search", params={"text": "greetings, sir", "lang": "enm", "year": 1400, "top_k": 3})

    assert response.json()["query"] == {
        "text": "greetings, sir",
        "lang": "enm",
        "year": 1400,
        "upos": None,
        "ner": None,
    }
    assert seen_kwargs["year"] == 1400
    assert seen_kwargs["top_k"] == 3


@pytest.mark.parametrize("ner_param", [None, "PER"])
def test_attest_builds_a_result_for_every_matching_row(monkeypatch, ner_param) -> None:
    rows = [
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
        }
    ]
    monkeypatch.setattr(loader, "load_relational", lambda *a, **k: pa.Table.from_pylist(rows))

    params = {"lemma": "Robin", "lang": "enm", "year": 1400}
    if ner_param is not None:
        params["ner"] = ner_param
    response = client.get("/attest", params=params)

    assert len(response.json()["results"]) == 1
