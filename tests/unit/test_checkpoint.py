"""Unit tests for the incremental JSON checkpoint used to make ingestion's
billable LLM-calling stages resumable after a crash (e.g. an API billing
error) partway through a run.
"""

from __future__ import annotations

import pytest

from ye_olde.ingest.checkpoint import Checkpoint


def test_get_or_compute_caches_and_persists(tmp_path):
    path = tmp_path / "progress.json"
    calls = []

    def compute():
        calls.append(1)
        return {"value": 42}

    cp = Checkpoint(path, meta={"n": 1})
    assert cp.get_or_compute("k1", compute) == {"value": 42}
    assert cp.get_or_compute("k1", compute) == {"value": 42}
    assert len(calls) == 1  # second call is served from the in-memory cache

    # a fresh Checkpoint instance over the same file picks up what was saved
    cp2 = Checkpoint(path, meta={"n": 1})
    assert cp2.get_or_compute("k1", compute) == {"value": 42}
    assert len(calls) == 1  # still not recomputed


def test_failed_compute_is_not_cached(tmp_path):
    path = tmp_path / "progress.json"
    cp = Checkpoint(path, meta={"n": 1})

    def failing():
        raise ValueError("simulated API failure")

    with pytest.raises(ValueError):
        cp.get_or_compute("k1", failing)

    # retrying the same key actually retries, rather than replaying a failure
    assert cp.get_or_compute("k1", lambda: "recovered") == "recovered"


def test_mismatched_meta_discards_old_entries(tmp_path):
    path = tmp_path / "progress.json"
    Checkpoint(path, meta={"n": 1}).get_or_compute("k1", lambda: "old")

    # different meta (e.g. a re-run with a different block_size) -> stale,
    # discarded rather than silently reused against different inputs
    cp = Checkpoint(path, meta={"n": 2})
    calls = []

    def compute():
        calls.append(1)
        return "new"

    assert cp.get_or_compute("k1", compute) == "new"
    assert len(calls) == 1


def test_clear_removes_entries_and_file(tmp_path):
    path = tmp_path / "progress.json"
    cp = Checkpoint(path, meta={"n": 1})
    cp.get_or_compute("k1", lambda: "value")
    assert path.exists()

    cp.clear()
    assert not path.exists()
    assert cp.entries == {}
