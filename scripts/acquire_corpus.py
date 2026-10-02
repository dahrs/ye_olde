#!/usr/bin/env python
"""CLI entrypoint for `ye_olde.ingest.acquire` (spec §6 step 0): fetches
every URL listed in a sources file (default `data/sources.yaml`), infers
its `data/raw/` filename via the configured LLM, and saves it.

Usage:
    python scripts/acquire_corpus.py
    python scripts/acquire_corpus.py --sources data/sources.yaml --raw-dir data/raw

Each entry is processed independently: one entry's failure (a dead link, a
site error, a filename collision) is logged and reported, and the rest of
the sources file still runs. Resumable by default -- re-running after a
partial failure only (re-)fetches entries that aren't already saved;
`--no-cache` forces every entry to be re-fetched and re-named.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ye_olde.common.llm_client import get_usage_summary
from ye_olde.common.logging import get_logger
from ye_olde.ingest.acquire import acquire_source, load_sources
from ye_olde.ingest.checkpoint import Checkpoint

_log = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sources",
        default="data/sources.yaml",
        help="path to the sources worklist (default: data/sources.yaml)",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="root directory new files are saved under (default: data/raw)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="litellm model string, overriding LITELLM_MODEL from .env",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore/discard the resume checkpoint and re-fetch/re-name every entry, even ones "
        "already saved in a previous run",
    )
    args = parser.parse_args(argv)

    sources_path = Path(args.sources)
    entries = load_sources(sources_path)
    if not entries:
        print(f"no entries in {sources_path}", file=sys.stderr)
        raise SystemExit(1)

    checkpoint = Checkpoint(
        Path(args.raw_dir) / ".acquire_progress.json",
        meta={"sources_file": str(sources_path), "model": args.model},
    )
    if args.no_cache:
        checkpoint.clear()

    succeeded: list[Path] = []
    failed: list[tuple[str, str]] = []
    for entry in entries:
        try:
            path = acquire_source(entry, raw_dir=Path(args.raw_dir), model=args.model, checkpoint=checkpoint)
        except Exception as exc:
            # Boundary catch (CLAUDE.md "Error handling"), applied per entry rather than once for
            # the whole run -- these are independent units of work, and one bad URL (a dead link,
            # a site throwing a 403, a filename collision) shouldn't stop the rest of the batch.
            _log.exception("failed to acquire %r", entry.url)
            print(f"[FAILED] {entry.url}: {exc}", file=sys.stderr)
            failed.append((entry.url, str(exc)))
        else:
            print(str(path))
            succeeded.append(path)

    usage = get_usage_summary()
    print(
        f"[cost] {usage['calls']} LLM calls, "
        f"{usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion tokens, "
        f"~${usage['cost_usd']:.4f}",
        file=sys.stderr,
    )
    print(f"\n{len(succeeded)} succeeded, {len(failed)} failed", file=sys.stderr)
    for url, reason in failed:
        print(f"  - {url}: {reason}", file=sys.stderr)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
