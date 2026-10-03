#!/usr/bin/env python
"""CLI entrypoint for `ye_olde.ingest.split_mixed` (spec §6 step 0b):
splits one already-acquired file that bundles several distinct editions of
the same work into separate, correctly-named `data/raw/` files.

Usage:
    python scripts/split_mixed_source.py data/raw/<work>/<mixed-file>.txt --manifest <manifest.yaml>

See `ye_olde.ingest.split_mixed`'s module docstring for the manifest format.
`--model` defaults to `None`, same as `align_corpus.py`/`acquire_corpus.py`
-- the configured `LITELLM_MODEL` from `.env` is used unless overridden.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ye_olde.common.llm_client import get_usage_summary
from ye_olde.common.logging import get_logger
from ye_olde.ingest.checkpoint import Checkpoint
from ye_olde.ingest.split_mixed import load_manifest, split_mixed_source

_log = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", help="path to the already-acquired mixed file, e.g. data/raw/<work>/<file>.txt")
    parser.add_argument(
        "--manifest",
        required=True,
        help="path to a *.sections.yaml manifest describing the editions bundled in this file",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="root data/raw/ directory, used to resolve a folder rename if needed (default: data/raw)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="litellm model string for the distinguisher LLM call, overriding LITELLM_MODEL from .env",
    )
    parser.add_argument("--chunk-size", type=int, default=2000, help="max characters per classified chunk")
    parser.add_argument("--batch-size", type=int, default=8, help="chunks per LLM call")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore/discard cached batch results from a previous interrupted run and reclassify everything",
    )
    args = parser.parse_args(argv)

    path = Path(args.path)
    manifest = load_manifest(Path(args.manifest))
    checkpoint = Checkpoint(
        path.with_suffix(path.suffix + ".split_progress.json"),
        meta={"manifest": args.manifest, "model": args.model},
    )
    if args.no_cache:
        checkpoint.clear()

    try:
        written = split_mixed_source(
            path,
            manifest,
            raw_dir=Path(args.raw_dir),
            model=args.model,
            chunk_max_chars=args.chunk_size,
            batch_size=args.batch_size,
            checkpoint=checkpoint,
        )
    except Exception:
        # Boundary catch (CLAUDE.md "Error handling") -- unlike acquire_corpus.py's per-URL loop,
        # one mixed file is one unit of work here, so there's nothing to skip past: log and exit.
        _log.exception("split_mixed_source failed for %r", args.path)
        raise SystemExit(1) from None
    finally:
        usage = get_usage_summary()
        print(
            f"[cost] {usage['calls']} LLM calls, "
            f"{usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion tokens, "
            f"~${usage['cost_usd']:.4f}",
            file=sys.stderr,
        )

    for written_path in written:
        print(written_path)


if __name__ == "__main__":
    main()
