#!/usr/bin/env python
"""CLI entrypoint for `ye_olde.ingest.align.align_corpus_folder` (spec §6).

Writes one `.jsonl` bitext file per language pair — the only output format;
see align.py's `align_corpus_folder` docstring for why.

Usage:
    python scripts/align_corpus.py data/raw/enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg
    python scripts/align_corpus.py data/raw/<work> --mode llm

Also importable directly:
    from ye_olde.ingest.align import align_corpus_folder
    align_corpus_folder("data/raw/<work>")
"""

from __future__ import annotations

import argparse
import sys

from ye_olde.ingest.align import align_corpus_folder
from ye_olde.ingest.llm_client import get_usage_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", help="path to a data/raw/<work>/ folder of 2+ parallel-translation files")
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="root output directory; bitext files land in <output-dir>/<work>/ (default: data/processed)",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "llm"],
        default="hybrid",
        help="'hybrid' (default): cheap embedding-based sentence matching, LLM only for the gaps it "
        "misses plus word-link extraction. 'llm': the LLM proposes sentence matches from scratch "
        "(no sentence-transformers dependency, more LLM calls on a long text).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="litellm model string, overriding LITELLM_MODEL from .env",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="sentence-transformers model for --mode hybrid, overriding EMBEDDING_MODEL from .env",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=40,
        help="content units per LLM alignment window (default: 40)",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.05,
        help="hybrid mode only: minimum winning margin (top match's cosine similarity minus the "
        "runner-up's) for the embedding aligner to accept a sentence match outright (default: 0.05)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore/discard cached progress and start over: re-clean every file even if a cached "
        "<file>.cleaned.json exists, and discard any <lang_a>-<lang_b>.align_progress.json from an "
        "interrupted alignment run instead of resuming it. Default: reuse both, so re-running after "
        "a crash (e.g. an API billing error) only pays for the calls it hadn't already made.",
    )
    args = parser.parse_args(argv)

    try:
        written = align_corpus_folder(
            args.folder,
            output_dir=args.output_dir,
            mode=args.mode,
            model=args.model,
            embedding_model=args.embedding_model,
            block_size=args.block_size,
            margin_threshold=args.margin_threshold,
            use_cache=not args.no_cache,
        )
    finally:
        # printed even on failure (e.g. a billing error mid-run) — that's
        # exactly when knowing what was already spent matters most, and
        # nothing here is lost: cleaning/alignment progress checkpoints
        # what succeeded so a re-run doesn't re-pay for it.
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
        print("no bitext files written", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
