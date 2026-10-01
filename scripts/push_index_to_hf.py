#!/usr/bin/env python
"""Pushes locally-built Search API index shards (`scripts/build_index.py`'s
output, `data/index/` by default) to the Hugging Face Dataset repo
`search_api` actually serves from — spec §13d, previously a manual,
copy-pasted-inline-snippet step with no script of its own.

Uploads the whole local index directory in one call
(`huggingface_hub.HfApi.upload_folder`), preserving whatever shard layout
is actually present under `--index-dir` (`pairs/<a>_<b>/...`,
`vectors/pairs/<a>_<b>/...`, `ngrams/pairs/<a>_<b>/...`,
`bm25/pairs/<a>_<b>/...`, and `relational/<iso>/...` once that exists) —
nothing about the shard-naming convention is hardcoded here, it's just
whatever `scripts/build_index.py` already wrote.

**Deliberately a separate, explicit step — not wired into
`build_index.py`/`align_corpus.py` automatically.** This pushes to a live,
public dataset repo that `search_api` serves directly; auto-pushing every
local build would mean a bad ingestion run (bad alignment, a corrupted
shard) reaches production with no human check in between. Run this
yourself once you're satisfied with a local build's output.

Does **not** trigger a `search_api` redeploy — its file listing is cached
per-process with no live-refresh endpoint (spec §13d); after a push, either
wait for the service's next natural cold start (~15 min idle) or manually
redeploy (spec §13c) to see the new data immediately.

Usage:
    python scripts/push_index_to_hf.py --dry-run     # see what would be pushed first
    python scripts/push_index_to_hf.py
    python scripts/push_index_to_hf.py --index-dir data/index --repo-id dahrs/ye_olde_data-index
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ye_olde.config import get_settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--index-dir", default="data/index", help="local index root to upload (default: data/index)")
    parser.add_argument("--repo-id", default=None, help="overrides HF_DATASET_REPO_ID from .env")
    parser.add_argument("--dry-run", action="store_true", help="list what would be uploaded, without uploading")
    args = parser.parse_args(argv)

    settings = get_settings()
    repo_id = args.repo_id or settings.hf_dataset_repo_id
    if not repo_id:
        print("[push_index_to_hf] no --repo-id given and HF_DATASET_REPO_ID isn't set in .env", file=sys.stderr)
        raise SystemExit(1)
    if not settings.hf_token and not args.dry_run:
        print("[push_index_to_hf] HF_TOKEN isn't set in .env -- needed to push (must be write-scoped)", file=sys.stderr)
        raise SystemExit(1)

    index_dir = Path(args.index_dir)
    if not index_dir.is_dir():
        print(f"[push_index_to_hf] {index_dir} doesn't exist -- run scripts/build_index.py first", file=sys.stderr)
        raise SystemExit(1)

    files = sorted(p for p in index_dir.rglob("*") if p.is_file())
    if not files:
        print(f"[push_index_to_hf] {index_dir} has no files to push", file=sys.stderr)
        raise SystemExit(1)

    verb = "would push" if args.dry_run else "pushing"
    print(f"[push_index_to_hf] {verb} {len(files)} files to dataset repo {repo_id!r}:", file=sys.stderr)
    for f in files:
        print(f"  {f.relative_to(index_dir)}", file=sys.stderr)
    if args.dry_run:
        return

    from huggingface_hub import HfApi

    HfApi(token=settings.hf_token).upload_folder(folder_path=str(index_dir), repo_id=repo_id, repo_type="dataset")
    print(
        "[push_index_to_hf] done. search_api won't see this until its next cold start or a manual "
        "redeploy (spec §13c/§13d) -- its file listing is cached per-process with no live-refresh.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
