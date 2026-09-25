"""Unit tests for the indexing job (spec §10): Parquet pairs shard + FAISS
vector shard from §3c bitext records. `embed_units` is monkeypatched to
return hand-built vectors, so these run with no model download.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from ye_olde.ingest import index

_RECORDS = [
    {
        "pair_id": "p1",
        "source": {"lang_code": "enm", "year": 1400, "work": "Sir Gawayne", "author": "Richard Morris"},
        "target": {"lang_code": "eng", "year": 1999, "work": "Sir Gawayne", "author": "W A Neilson"},
        "source_text": "gladly would I see him",
        "target_text": "Gladly I would see that hero",
        "source_tokens": ["gladly", "would", "I", "see", "him"],
        "target_tokens": ["Gladly", "I", "would", "see", "that", "hero"],
        "citation": None,
        "sentence_confidence": 0.9,
        "alignment_links": [],
        "sentence_method": "embedding:test",
        "metadata": {"license": None, "ingested_at": "2026-09-23T00:00:00+00:00"},
    },
    {
        "pair_id": "p2",
        "source": {"lang_code": "enm", "year": 1400, "work": "Sir Gawayne", "author": "Richard Morris"},
        "target": {"lang_code": "eng", "year": 1999, "work": "Sir Gawayne", "author": "W A Neilson"},
        "source_text": "Gladly sir for sothe",
        "target_text": "Gladly, sir, for sooth",
        "source_tokens": ["Gladly", "sir", "for", "sothe"],
        "target_tokens": ["Gladly", ",", "sir", ",", "for", "sooth"],
        "citation": None,
        "sentence_confidence": 0.95,
        "alignment_links": [{"source_idx": [3], "target_idx": [5], "sense_id": None}],
        "sentence_method": "embedding:test",
        "metadata": {"license": None, "ingested_at": "2026-09-23T00:00:00+00:00"},
    },
]

# One 3-dim one-hot vector per text (4 texts: p1 source/target, p2 source/target)
# so cosine similarity is unambiguous in tests below.
_VECTORS = {
    "gladly would I see him": [1, 0, 0],
    "Gladly I would see that hero": [1, 0, 0],
    "Gladly sir for sothe": [0, 1, 0],
    "Gladly, sir, for sooth": [0, 1, 0],
}


def _fake_embed_units(units: list[str], **kwargs: object) -> np.ndarray:
    return np.array([_VECTORS[u] for u in units], dtype="float32")


def test_group_by_lang_pair_groups_across_records():
    groups = index.group_by_lang_pair(_RECORDS)
    assert set(groups) == {("enm", "eng")}
    assert len(groups[("enm", "eng")]) == 2


def test_build_pairs_table_round_trips_columns():
    table = index.build_pairs_table(_RECORDS)
    rows = table.to_pylist()
    assert rows[0]["pair_id"] == "p1"
    assert rows[1]["source_text"] == "Gladly sir for sothe"


def test_build_vector_index_ids_map_to_source_then_target(monkeypatch):
    monkeypatch.setattr(index, "embed_units", _fake_embed_units)
    faiss_index = index.build_vector_index(_RECORDS)
    assert faiss_index.ntotal == 4  # 2 records x 2 sides

    # id 0 = p1 source ("gladly would I see him", vector [1,0,0]); querying
    # with that exact vector should return id 0 as the top hit.
    query = np.array([[1, 0, 0]], dtype="float32")
    distances, ids = faiss_index.search(query, 1)
    assert ids[0][0] == 0
    assert distances[0][0] == pytest.approx(1.0)

    # id 3 = p2 target ("Gladly, sir, for sooth", vector [0,1,0]).
    query2 = np.array([[0, 1, 0]], dtype="float32")
    distances2, ids2 = faiss_index.search(query2, 1)
    assert ids2[0][0] in (2, 3)  # either side of p2, both share the same vector here


def test_write_pair_shard_writes_matching_parquet_and_faiss(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(index, "embed_units", _fake_embed_units)
    parquet_path, faiss_path = index.write_pair_shard(_RECORDS, tmp_path)

    assert parquet_path == tmp_path / "pairs" / "enm_eng" / "1400-1999.parquet"
    assert faiss_path == tmp_path / "vectors" / "pairs" / "enm_eng" / "1400-1999.faiss"
    assert parquet_path.is_file()
    assert faiss_path.is_file()

    table = pq.read_table(parquet_path)
    assert table.num_rows == 2

    import faiss as faiss_lib

    loaded = faiss_lib.read_index(str(faiss_path))
    assert loaded.ntotal == 4


def test_write_pair_shard_raises_on_empty_records(tmp_path: Path):
    with pytest.raises(ValueError, match="no records"):
        index.write_pair_shard([], tmp_path)


def test_build_index_for_processed_root_discovers_and_groups(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(index, "embed_units", _fake_embed_units)
    work_dir = tmp_path / "processed" / "sir_gawayne"
    work_dir.mkdir(parents=True)
    bitext_path = work_dir / "enm-eng.bitext.jsonl"
    with bitext_path.open("w", encoding="utf-8") as f:
        for rec in _RECORDS:
            f.write(json.dumps(rec))
            f.write("\n")

    out_dir = tmp_path / "index"
    written = index.build_index_for_processed_root(tmp_path / "processed", out_dir)

    assert len(written) == 1
    parquet_path, faiss_path = written[0]
    assert parquet_path.is_file()
    assert faiss_path.is_file()


def test_build_index_for_processed_root_raises_when_nothing_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        index.build_index_for_processed_root(tmp_path / "processed", tmp_path / "index")
