"""Builds the semantic passage index + Parquet pairs shard from §3c bitext
output — spec §10's "indexing job", applied to the one data track that
actually has data today (aligned example pairs; contribution-derived §3a
rows have no data yet — `contributions/` is empty, see this package's
`__init__.py`).

Two artifacts come out of the same pass over the data, because a FAISS hit
is meaningless without the row it points back to:

  - `pairs/<lang_a>_<lang_b>/<year_from>-<year_to>.parquet` — the exact
    columns `search_api/app/main.py`'s `/lookup` and `/attest` already read
    (see `align.py`'s `PairRecord` docstring: it was written expecting this).
  - `vectors/pairs/<lang_a>_<lang_b>/<year_from>-<year_to>.faiss` — one
    L2-normalized embedding per side (`source_text`, `target_text`) of
    every row in that *same* shard: `id = row_index * 2 (+1 for target)`.
    This id scheme is only meaningful paired with that exact shard's row
    order — a FAISS shard and its Parquet counterpart must always be
    loaded together, never mixed with a different shard's table
    (`search_api/app/loader.py`'s vector-search path keeps them paired for
    exactly this reason instead of concatenating tables the way the plain
    exact-match path does).

Both sides of every pair are embedded (not just one) because a query can
come from either language — `/lookup` already handles either direction, and
semantic search needs to too.

Volume-based quantile splitting (spec §10, for when a language pair's data
outgrows one shard) isn't implemented here — not yet needed at this corpus's
scale (a few hundred pairs from one work). Revisit once it is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .embed import embed_units


def _read_bitext_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_bitext_records(paths: list[Path]) -> list[dict[str, Any]]:
    """Reads and concatenates every `*.bitext.jsonl` file in `paths`."""
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(_read_bitext_jsonl(path))
    return records


def group_by_lang_pair(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Groups §3c records by `(source.lang_code, target.lang_code)` — the
    same grouping key `pairs/<lang_a>_<lang_b>/` shards are named after.
    Records from more than one work land in the same group here if they
    share a language pair, which is the merge point for a future work
    contributing to an already-indexed pair.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for rec in records:
        key = (rec["source"]["lang_code"], rec["target"]["lang_code"])
        groups.setdefault(key, []).append(rec)
    return groups


def _year_range(records: list[dict[str, Any]]) -> tuple[int, int]:
    years = [r["source"]["year"] for r in records] + [r["target"]["year"] for r in records]
    return min(years), max(years)


def build_pairs_table(records: list[dict[str, Any]]) -> pa.Table:
    """Flattens §3c records into a Parquet table — row order here *is* the
    row order `build_vector_index`'s ids are relative to, so the same list,
    in the same order, must be passed to both.
    """
    return pa.Table.from_pylist(records)


def build_vector_index(records: list[dict[str, Any]], *, model_name: str | None = None) -> faiss.Index:
    """One vector per side (`source_text`, `target_text`) of every record,
    in the multilingual embedder's shared space. `id = row_index * 2` for
    the source side, `row_index * 2 + 1` for the target side — see this
    module's docstring for why that pairing with the Parquet table matters.
    """
    texts: list[str] = []
    for rec in records:
        texts.append(rec["source_text"])
        texts.append(rec["target_text"])
    vectors = embed_units(texts, model_name=model_name).astype("float32")
    dim = vectors.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    ids = np.arange(len(texts), dtype="int64")
    index.add_with_ids(vectors, ids)
    return index


def write_pair_shard(
    records: list[dict[str, Any]], out_dir: Path, *, model_name: str | None = None
) -> tuple[Path, Path]:
    """Writes both the Parquet pairs shard and its FAISS vector shard for
    one `(lang_a, lang_b)` group under `out_dir` (the spec §10 layout —
    `data/index/` locally, or the HF Dataset repo root once pushed).
    Returns `(parquet_path, faiss_path)`.
    """
    if not records:
        raise ValueError("no records to index")
    lang_a = records[0]["source"]["lang_code"]
    lang_b = records[0]["target"]["lang_code"]
    year_from, year_to = _year_range(records)

    table = build_pairs_table(records)
    parquet_dir = out_dir / "pairs" / f"{lang_a}_{lang_b}"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / f"{year_from}-{year_to}.parquet"
    pq.write_table(table, parquet_path)

    index = build_vector_index(records, model_name=model_name)
    faiss_dir = out_dir / "vectors" / "pairs" / f"{lang_a}_{lang_b}"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = faiss_dir / f"{year_from}-{year_to}.faiss"
    faiss.write_index(index, str(faiss_path))

    return parquet_path, faiss_path


def build_index_for_processed_root(
    processed_root: Path, out_dir: Path, *, model_name: str | None = None
) -> list[tuple[Path, Path]]:
    """Reads every `*.bitext.jsonl` under every `<processed_root>/<work>/`
    folder, groups records across *all* works by language pair, and writes
    one pairs+vectors shard per pair. Returns the list of
    `(parquet_path, faiss_path)` written, one per language pair found.
    """
    paths = sorted(Path(processed_root).glob("*/*.bitext.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no *.bitext.jsonl files under {processed_root}")
    records = load_bitext_records(paths)
    groups = group_by_lang_pair(records)
    out_dir = Path(out_dir)
    return [write_pair_shard(recs, out_dir, model_name=model_name) for recs in groups.values()]
