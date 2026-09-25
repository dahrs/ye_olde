#!/usr/bin/env python
"""CLI entrypoint for `ye_olde.ingest.index.build_index_for_processed_root`
(spec §10's indexing job).

Reads every `data/processed/<work>/*.bitext.jsonl` written by
`scripts/align_corpus.py`, groups them by language pair, and writes a
Parquet pairs shard + FAISS vector shard per pair — the artifacts
`search_api/` reads (locally under `--output-dir` for now; pushing them to
the Hugging Face Dataset repo is a separate, later step, spec §10).

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --processed-dir data/processed --output-dir data/index
"""

from __future__ import annotations

import argparse
import sys

from ye_olde.ingest.index import build_index_for_processed_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="root containing one folder per work, each with *.bitext.jsonl files (default: data/processed)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/index",
        help="root to write pairs/ and vectors/pairs/ shards under (default: data/index)",
    )
    args = parser.parse_args(argv)

    written = build_index_for_processed_root(args.processed_dir, args.output_dir)
    for parquet_path, faiss_path in written:
        print(f"[index] wrote {parquet_path} + {faiss_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
