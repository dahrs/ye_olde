#!/usr/bin/env python
"""CLI entrypoint for `ye_olde.ingest.contributions.build_relational_index_for_contributions_root`
(spec §5/§3a's "validated community contribution -> indexed attestation
data" leg).

Reads every validated `*.json` under `--contributions-dir` (default
`contributions/`), tags each one's headword via `ye_olde.classify.classify`
and resolves a `sense_id` via `ye_olde.ingest.sense.resolve_sense_id`
(reusing the contribution's own `target.text` as the English gloss), and
writes one Parquet shard per `iso_code` under `--output-dir` — the
`relational/<iso_code>/<year_from>-<year_to>.parquet` shards
`search_api`'s `/attest` reads.

Usage:
    python scripts/build_relational_index.py
    python scripts/build_relational_index.py --contributions-dir contributions --output-dir data/index
"""

from __future__ import annotations

import argparse
import sys

from ye_olde.common.llm_client import get_usage_summary
from ye_olde.common.logging import get_logger
from ye_olde.ingest.contributions import build_relational_index_for_contributions_root

_log = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--contributions-dir",
        default="contributions",
        help="root containing validated contribution *.json files (default: contributions)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/index",
        help="root to write relational/ shards under (default: data/index)",
    )
    parser.add_argument("--model", default=None, help="litellm model string, overriding LITELLM_MODEL from .env")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="sentence-transformers model for sense_id resolution, overriding EMBEDDING_MODEL from .env",
    )
    args = parser.parse_args(argv)

    try:
        written = build_relational_index_for_contributions_root(
            args.contributions_dir, args.output_dir, model=args.model, embedding_model=args.embedding_model
        )
    except Exception:
        # Boundary catch (CLAUDE.md "Error handling") — a schema-invalid
        # contribution, an LLM/embedding-model failure, or a file I/O error
        # all need to be caught and logged somewhere, or they print a raw
        # traceback and never reach logs/ye_olde.log.
        _log.exception("build_relational_index_for_contributions_root failed for %r", args.contributions_dir)
        raise SystemExit(1) from None
    finally:
        usage = get_usage_summary()
        print(
            f"[cost] {usage['calls']} LLM calls, "
            f"{usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion tokens, "
            f"~${usage['cost_usd']:.4f}",
            file=sys.stderr,
        )

    for path in written:
        print(path)
    if not written:
        print("no validated contributions found -- nothing written", file=sys.stderr)


if __name__ == "__main__":
    main()
