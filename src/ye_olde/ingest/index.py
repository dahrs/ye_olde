"""Builds the passage index (vector + lexical) + Parquet pairs shard from
§3c bitext output — spec §10's "indexing job", applied to the one data track
that actually has data today (aligned example pairs; contribution-derived
§3a rows have no data yet — `contributions/` is empty, see this package's
`__init__.py`).

Four artifacts come out of the same pass over the data, all keyed by the
same `row_index` (position within the `records` list passed to each build
function — the same list, same order, must be used for all four):

  - `pairs/<lang_a>_<lang_b>/<year_from>-<year_to>.parquet` — the exact
    columns `search_api/app/main.py`'s `/lookup` and `/attest` already read
    (see `align.py`'s `PairRecord` docstring: it was written expecting this).
  - `vectors/pairs/<lang_a>_<lang_b>/<year_from>-<year_to>.faiss` — one
    L2-normalized embedding per side (`source_text`, `target_text`) of
    every row: `id = row_index * 2 (+1 for target)`. Only meaningful paired
    with that exact shard's row order — never mixed with a different
    shard's table (`search_api/app/loader.py`'s vector-search path keeps
    them paired for exactly this reason instead of concatenating tables
    the way the plain exact-match path does).
  - `ngrams/pairs/<lang_a>_<lang_b>/<year_from>-<year_to>.parquet` —
    character-trigram postings for every token on both sides, spec §10's
    "Hybrid retrieval scoring": the inverted index `/lookup`'s fuzzy
    candidate generation reads (same technique as PostgreSQL's `pg_trgm`),
    and `/search`'s lexical scoring reads too.
  - `bm25/pairs/<lang_a>_<lang_b>/<year_from>-<year_to>.parquet` — term
    postings (same spec section) for `/search`'s BM25 signal. Document
    frequency and corpus size aren't stored here — `search_api` derives
    them at query time (see that spec section for why: they're cheap to
    derive from data already being loaded regardless).

Both sides of every pair get vectors/n-grams/terms (not just one) because a
query can come from either language — `/lookup` already handles either
direction, and semantic/lexical search need to too.

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


_TRIGRAM_SIZE = 3

# Which side each record contributes tokens from, and what search_api's
# `side` column value means for it (0=source, 1=target) — shared by the
# n-gram and BM25 table builders since both index the same token lists.
_SIDES: tuple[tuple[int, str], ...] = ((0, "source_tokens"), (1, "target_tokens"))


def char_trigrams(token: str) -> set[str]:
    """Character trigrams of `token`, lowercased and padded with 2 spaces on
    each side (the standard `pg_trgm`-style convention) so short tokens and
    word-boundary positions still produce a useful number of trigrams —
    spec §10's "Hybrid retrieval scoring". A duplicate of this exact
    function lives in `search_api/app/lexical.py` for computing a *query's*
    trigrams at request time — the two services don't share a dependency
    tree by design (see `search_api/app/config.py`'s docstring), but the
    trigram *definition* must stay identical between them or postings built
    here wouldn't match queries computed there.
    """
    padded = f"  {token.lower()}  "
    return {padded[i : i + _TRIGRAM_SIZE] for i in range(len(padded) - _TRIGRAM_SIZE + 1)}


def build_ngram_table(records: list[dict[str, Any]]) -> pa.Table:
    """Character-trigram postings for every token on both sides of every
    record — the inverted index `/lookup`'s fuzzy candidate generation reads
    (spec §10). `row_index` matches `build_pairs_table`'s row order; both
    must be built from the same `records` list, in the same order.
    """
    rows: list[dict[str, Any]] = []
    for row_index, rec in enumerate(records):
        for side, tokens_key in _SIDES:
            for token_index, token in enumerate(rec[tokens_key]):
                grams = char_trigrams(token)
                rows.extend(
                    {
                        "ngram": gram,
                        "row_index": row_index,
                        "side": side,
                        "token_index": token_index,
                        "token_ngram_count": len(grams),
                    }
                    for gram in grams
                )
    return pa.Table.from_pylist(rows)


def build_bm25_table(records: list[dict[str, Any]]) -> pa.Table:
    """Term-frequency postings for every side of every record — spec §10's
    BM25 index. Document frequency and corpus size aren't stored here:
    `search_api` derives document frequency from how many rows match a term
    in this same table, and corpus size / average document length from the
    already-loaded pairs table's row count and token-list lengths (see
    `search_api/app/lexical.py`) — cheaper than maintaining a second,
    separately-updated statistics artifact.
    """
    rows: list[dict[str, Any]] = []
    for row_index, rec in enumerate(records):
        for side, tokens_key in _SIDES:
            counts: dict[str, int] = {}
            for token in rec[tokens_key]:
                term = token.lower()
                counts[term] = counts.get(term, 0) + 1
            rows.extend(
                {"term": term, "row_index": row_index, "side": side, "term_frequency": tf}
                for term, tf in counts.items()
            )
    return pa.Table.from_pylist(rows)


def write_pair_shard(
    records: list[dict[str, Any]], out_dir: Path, *, model_name: str | None = None
) -> tuple[Path, Path, Path, Path]:
    """Writes the Parquet pairs shard, its FAISS vector shard, its n-gram
    postings shard, and its BM25 postings shard, all for one
    `(lang_a, lang_b)` group under `out_dir` (the spec §10 layout —
    `data/index/` locally, or the HF Dataset repo root once pushed).
    Returns `(parquet_path, faiss_path, ngram_path, bm25_path)`.
    """
    if not records:
        raise ValueError("no records to index")
    lang_a = records[0]["source"]["lang_code"]
    lang_b = records[0]["target"]["lang_code"]
    year_from, year_to = _year_range(records)
    shard_name = f"{lang_a}_{lang_b}/{year_from}-{year_to}"

    def _write(subdir: str, table: pa.Table) -> Path:
        path = out_dir / subdir / f"{shard_name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path)
        return path

    parquet_path = _write("pairs", build_pairs_table(records))
    ngram_path = _write("ngrams/pairs", build_ngram_table(records))
    bm25_path = _write("bm25/pairs", build_bm25_table(records))

    index = build_vector_index(records, model_name=model_name)
    faiss_dir = out_dir / "vectors" / "pairs" / f"{lang_a}_{lang_b}"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = faiss_dir / f"{year_from}-{year_to}.faiss"
    faiss.write_index(index, str(faiss_path))

    return parquet_path, faiss_path, ngram_path, bm25_path


def build_index_for_processed_root(
    processed_root: Path, out_dir: Path, *, model_name: str | None = None
) -> list[tuple[Path, Path, Path, Path]]:
    """Reads every `*.bitext.jsonl` under every `<processed_root>/<work>/`
    folder, groups records across *all* works by language pair, and writes
    one pairs+vectors+ngrams+bm25 shard set per pair. Returns the list of
    `(parquet_path, faiss_path, ngram_path, bm25_path)` written, one per
    language pair found.
    """
    paths = sorted(Path(processed_root).glob("*/*.bitext.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no *.bitext.jsonl files under {processed_root}")
    records = load_bitext_records(paths)
    groups = group_by_lang_pair(records)
    out_dir = Path(out_dir)
    return [write_pair_shard(recs, out_dir, model_name=model_name) for recs in groups.values()]
