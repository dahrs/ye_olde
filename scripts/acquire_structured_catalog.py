#!/usr/bin/env python
"""CLI entrypoint for `ye_olde.ingest.structured_sources`'s LLM-free path:
discovers every work in one or more structured-source repos and acquires
them straight into `data/raw/`, without hand-maintaining a `sources.yaml`
entry per work.

Usage:
    python scripts/acquire_structured_catalog.py --repo PerseusDL/canonical-latinLit
    python scripts/acquire_structured_catalog.py --repo PerseusDL/canonical-latinLit \
        --repo PerseusDL/canonical-greekLit --limit 5 --model claude_code_cli/claude-sonnet-5

`--repo` may be given more than once (each is discovered and acquired
independently). `--limit` caps how many works are acquired per repo --
useful for a small validation run before committing to a full, much
larger batch. Each work is processed independently: one work's failure
(a fetch error, no confident composition year from either Wikidata or the
LLM fallback, a filename collision) is logged and reported, and the rest
of the batch still runs. Resumable by default, same as acquire_corpus.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ye_olde.common.llm_client import get_usage_summary
from ye_olde.common.logging import get_logger
from ye_olde.ingest.acquire import SourceEntry, acquire_source
from ye_olde.ingest.checkpoint import Checkpoint
from ye_olde.ingest.structured_sources import discover_perseus_cts_urls

_log = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--repo",
        action="append",
        required=True,
        dest="repos",
        help='a "PerseusDL/canonical-<lang>Lit"-shaped GitHub repo to discover and acquire; repeatable',
    )
    parser.add_argument("--branch", default="master", help="git branch to read each repo from (default: master)")
    parser.add_argument(
        "--raw-dir", default="data/raw", help="root directory new files are saved under (default: data/raw)"
    )
    parser.add_argument("--model", default=None, help="litellm model string, overriding LITELLM_MODEL from .env")
    parser.add_argument(
        "--limit", type=int, default=None, help="acquire at most this many works per repo (default: no limit)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore/discard the resume checkpoint and re-acquire every work, even ones already saved",
    )
    args = parser.parse_args(argv)

    checkpoint = Checkpoint(
        Path(args.raw_dir) / ".structured_catalog_progress.json",
        meta={"repos": args.repos, "branch": args.branch, "model": args.model},
    )
    if args.no_cache:
        checkpoint.clear()

    succeeded: list[Path] = []
    failed: list[tuple[str, str]] = []
    for repo in args.repos:
        try:
            urls = discover_perseus_cts_urls(repo, branch=args.branch)
        except Exception:
            _log.exception("failed to discover works in %r", repo)
            print(f"[FAILED] discovering {repo}: see logs/ye_olde.log", file=sys.stderr)
            continue
        if args.limit is not None:
            urls = urls[: args.limit]
        print(f"[{repo}] {len(urls)} work(s) to acquire", file=sys.stderr)

        for url in urls:
            entry = SourceEntry(url=url)
            try:
                paths = acquire_source(entry, raw_dir=Path(args.raw_dir), model=args.model, checkpoint=checkpoint)
            except Exception as exc:
                # Boundary catch (CLAUDE.md "Error handling"), per work -- independent units of
                # work, so one bad/undated/unreachable work shouldn't stop the rest of the batch.
                _log.exception("failed to acquire %r", url)
                print(f"[FAILED] {url}: {exc}", file=sys.stderr)
                failed.append((url, str(exc)))
            else:
                for path in paths:
                    print(str(path))
                succeeded.extend(paths)

    usage = get_usage_summary()
    print(
        f"[cost] {usage['calls']} LLM calls, "
        f"{usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion tokens, "
        f"~${usage['cost_usd']:.4f}",
        file=sys.stderr,
    )
    print(f"\n{len(succeeded)} file(s) written, {len(failed)} work(s) failed", file=sys.stderr)
    for url, reason in failed:
        print(f"  - {url}: {reason}", file=sys.stderr)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
