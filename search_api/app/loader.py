"""Downloads and reads Parquet shards from the HF Dataset repo (spec §10).

Shard layout (spec §10 / §3c):
  relational/<iso_code>/<year_from>-<year_to>.parquet
  pairs/<lang_a>_<lang_b>/<year_from>-<year_to>.parquet

Vector (FAISS) shards aren't loaded here yet — semantic /lookup matching is a
follow-up once §6's alignment pipeline has produced real embeddings; today
/lookup matches on exact source tokens within the aligned pairs table.

Every lookup here is soft-fail: no dataset repo configured yet, no repo
found, or no shards for a given iso_code/lang-pair yet, all return an empty
table rather than raising — this service is meant to come up and serve
correctly before any corpus has been gathered (spec §11).
"""

from __future__ import annotations

import re
from functools import lru_cache

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from .config import get_settings

_SHARD_RE = re.compile(r"^(?P<prefix>.+)/(?P<start>\d+)-(?P<end>\d+)\.parquet$")


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
