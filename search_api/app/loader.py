"""Downloads and reads Parquet shards from the HF Dataset repo (spec §10).

Shard layout (spec §10 / §3c):
  relational/<iso_code>/<year_from>-<year_to>.parquet
  pairs/<lang_a>_<lang_b>/<year_from>-<year_to>.parquet
  vectors/pairs/<lang_a>_<lang_b>/<year_from>-<year_to>.faiss

The FAISS shards are written by ye_olde.ingest.index — one embedding per
side (source_text, target_text) of every row in the *same-named* pairs
Parquet shard, id = row_index * 2 (+1 for target). That id scheme only
means anything paired with that exact shard's own row order, which is why
`load_pair_shards_with_vectors` below keeps each shard's table and index
together instead of concatenating tables the way `load_pairs` does for
plain exact-match lookup.

Every lookup here is soft-fail: no dataset repo configured yet, no repo
found, or no shards for a given iso_code/lang-pair yet, all return an empty
table (or []) rather than raising — this service is meant to come up and
serve correctly before any corpus has been gathered (spec §11).
"""

from __future__ import annotations

import re
from functools import lru_cache

import faiss
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from .config import get_settings
from .embedding import embed_query
from .schemas import SearchResult

_SHARD_RE = re.compile(r"^(?P<prefix>.+)/(?P<start>\d+)-(?P<end>\d+)\.parquet$")
_PAIRS_DIR_RE = re.compile(r"^pairs/(?P<lang_a>[a-z]+)_(?P<lang_b>[a-z]+)/")


@lru_cache(maxsize=1)
def _list_repo_files() -> tuple[str, ...]:
    settings = get_settings()
    if not settings.hf_dataset_repo_id:
        return ()
    try:
        api = HfApi(token=settings.hf_token or None)
        return tuple(api.list_repo_files(settings.hf_dataset_repo_id, repo_type="dataset"))
    except RepositoryNotFoundError:
        return ()


def refresh() -> None:
    """Drop the cached file listing — call after new data shards are pushed."""
    _list_repo_files.cache_clear()


def _matching_shards(prefix: str, year_from: int | None, year_to: int | None) -> list[str]:
    out = []
    for path in _list_repo_files():
        match = _SHARD_RE.match(path)
        if not match or not path.startswith(prefix + "/"):
            continue
        shard_start, shard_end = int(match.group("start")), int(match.group("end"))
        if year_from is not None and shard_end < year_from:
            continue
        if year_to is not None and shard_start > year_to:
            continue
        out.append(path)
    return out


def _download(path: str) -> str | None:
    settings = get_settings()
    try:
        return hf_hub_download(
            repo_id=settings.hf_dataset_repo_id,
            repo_type="dataset",
            filename=path,
            token=settings.hf_token or None,
            cache_dir=settings.hf_cache_dir,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None


def _load_shards(shards: list[str], year_from: int | None, year_to: int | None) -> pa.Table:
    filters = None
    if year_from is not None or year_to is not None:
        filters = []
        if year_from is not None:
            filters.append(("attestation_year", ">=", year_from))
        if year_to is not None:
            filters.append(("attestation_year", "<=", year_to))
    tables = []
    for shard in shards:
        local_path = _download(shard)
        if local_path is None:
            continue
        tables.append(pq.read_table(local_path, filters=filters))
    if not tables:
        return pa.table({})
    return pa.concat_tables(tables)


def load_relational(iso_code: str, year_from: int | None = None, year_to: int | None = None) -> pa.Table:
    shards = _matching_shards(f"relational/{iso_code}", year_from, year_to)
    return _load_shards(shards, year_from, year_to)


def load_pairs(lang_a: str, lang_b: str, year_from: int | None = None, year_to: int | None = None) -> pa.Table:
    shards = _matching_shards(f"pairs/{lang_a}_{lang_b}", year_from, year_to)
    if not shards:
        shards = _matching_shards(f"pairs/{lang_b}_{lang_a}", year_from, year_to)
    # Pairs shards aren't filtered on attestation_year — the alignment record
    # doesn't have that column (spec §3c uses source.year/target.year instead).
    return _load_shards(shards, None, None)


def _vector_shard_path(parquet_shard_path: str) -> str:
    """Maps e.g. pairs/enm_eng/1400-1999.parquet to
    vectors/pairs/enm_eng/1400-1999.faiss — the naming convention
    ye_olde.ingest.index writes both halves of a shard under (see that
    module's docstring).
    """
    stem = parquet_shard_path[: -len(".parquet")]
    return f"vectors/{stem}.faiss"


def load_pair_shards_with_vectors(lang_a: str, lang_b: str) -> list[tuple[pa.Table, faiss.Index | None]]:
    """Like load_pairs, but keeps each shard's table paired with its own
    FAISS index rather than concatenating tables across shards — a FAISS id
    is only meaningful against the exact shard it was built from, so shards
    must never be merged before resolving a vector-search hit back to a row.
    The index is None for a shard that has a pairs Parquet file but no
    matching vector file yet (soft-fail, same philosophy as the rest of this
    module) — that shard's rows are simply skipped by semantic search
    without affecting the others.
    """
    shard_paths = _matching_shards(f"pairs/{lang_a}_{lang_b}", None, None)
    if not shard_paths:
        shard_paths = _matching_shards(f"pairs/{lang_b}_{lang_a}", None, None)
    shards: list[tuple[pa.Table, faiss.Index | None]] = []
    for shard_path in shard_paths:
        local_parquet = _download(shard_path)
        if local_parquet is None:
            continue
        table = pq.read_table(local_parquet)
        local_faiss = _download(_vector_shard_path(shard_path))
        vector_index = faiss.read_index(local_faiss) if local_faiss is not None else None
        shards.append((table, vector_index))
    return shards


def _lang_pair_dirs_containing(lang: str) -> list[tuple[str, str]]:
    """Every `(lang_a, lang_b)` pair directory in the repo that has `lang`
    on either side — lets /search look across every indexed pair without
    the caller needing to already know the counterpart language.
    """
    found: set[tuple[str, str]] = set()
    for path in _list_repo_files():
        match = _PAIRS_DIR_RE.match(path)
        if not match:
            continue
        lang_a, lang_b = match.group("lang_a"), match.group("lang_b")
        if lang in (lang_a, lang_b):
            found.add((lang_a, lang_b))
    return sorted(found)


def search_passages(
    lang: str,
    text: str,
    *,
    year: int | None = None,
    window: int = 50,
    top_k: int = 5,
) -> list[SearchResult]:
    """Semantic passage search — the "follow-up once §6's alignment pipeline
    has produced real embeddings" this module used to only forecast.
    Embeds `text` with the same model the indexed shards were built with
    (settings.embedding_model, spec §10), and returns the top_k closest
    indexed passages tagged with language `lang` (within `window` years of
    `year`, if given), across every pair directory that has `lang` on
    either side. Each result carries its aligned counterpart passage too,
    since that's the pairing this data actually has (spec §3c) — there's no
    unpaired, single-language semantic index yet.

    Returns SearchResult directly (unlike load_relational/load_pairs, which
    return a bare pa.Table for main.py to shape) — the row here is already a
    fixed, known shape, so building the pydantic model here rather than
    passing an untyped dict up keeps this typed end to end. Soft-fails to []
    wherever a shard or vector file doesn't exist yet, same as the rest of
    this module.
    """
    settings = get_settings()
    query_vector = embed_query(text, model_name=settings.embedding_model, hf_token=settings.hf_token)

    candidates: list[SearchResult] = []
    for lang_a, lang_b in _lang_pair_dirs_containing(lang):
        for table, vector_index in load_pair_shards_with_vectors(lang_a, lang_b):
            if vector_index is None or table.num_rows == 0:
                continue
            k = min(top_k, vector_index.ntotal)
            distances, ids = vector_index.search(query_vector, k)
            rows = table.to_pylist()
            for score, vec_id in zip(distances[0], ids[0], strict=True):
                if vec_id < 0:
                    continue
                row_idx, side = divmod(int(vec_id), 2)
                if row_idx >= len(rows):
                    continue
                row = rows[row_idx]
                query_side = row["source"] if side == 0 else row["target"]
                other_side = row["target"] if side == 0 else row["source"]
                if query_side["lang_code"] != lang:
                    continue  # defensive: shard's own tag disagrees with the folder name
                if year is not None and abs(query_side["year"] - year) > window:
                    continue
                query_text_key = "source_text" if side == 0 else "target_text"
                other_text_key = "target_text" if side == 0 else "source_text"
                candidates.append(
                    SearchResult(
                        pair_id=row["pair_id"],
                        lang=query_side["lang_code"],
                        year=query_side["year"],
                        work=query_side.get("work"),
                        text=row[query_text_key],
                        score=float(score),
                        other_lang=other_side["lang_code"],
                        other_year=other_side["year"],
                        other_work=other_side.get("work"),
                        other_text=row[other_text_key],
                        citation=row.get("citation") or "",
                    )
                )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]
