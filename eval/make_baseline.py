#!/usr/bin/env python
"""Snapshots a work's current `data/processed/<work>/` cleaning+alignment
output into `eval/baselines/<work>/`, tagged with the model that produced
it and a capture timestamp — the fixed reference point
`run_local_model_eval.py` compares a local model's output against.

Deliberately separate from the eval runner: a baseline is a stable
reference you create once (or refresh deliberately, e.g. after a pipeline
fix — as happened for Sir Gawayne's mojibake fixes), not something that
gets silently recreated by every eval run. Re-running an eval against
several different local models should always compare against the same
baseline unless you explicitly ask for a fresh one.

Usage:
    python eval/make_baseline.py data/raw/<work> --model-label claude-sonnet-5-api
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import slugify  # noqa: E402


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def make_baseline(
    work_folder: str | Path,
    *,
    model_label: str,
    processed_root: str | Path = "data/processed",
    baseline_root: str | Path = "eval/baselines",
    timestamp: str | None = None,
) -> list[Path]:
    work_name = Path(work_folder).name
    processed_dir = Path(processed_root) / work_name
    if not processed_dir.is_dir():
        raise FileNotFoundError(
            f"no processed output for {work_name!r} at {processed_dir} — run scripts/align_corpus.py on it first"
        )

    ts = timestamp or _timestamp()
    tag = slugify(model_label)
    out_dir = Path(baseline_root) / work_name
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for pattern in ("*.cleaned.json", "*.bitext.jsonl"):
        for path in sorted(processed_dir.glob(pattern)):
            suffix = "".join(path.suffixes)  # e.g. ".cleaned.json" or ".bitext.jsonl"
            stem = path.name[: -len(suffix)]
            dest = out_dir / f"{stem}.baseline_{tag}_{ts}{suffix}"
            shutil.copy2(path, dest)
            written.append(dest)

    if not written:
        raise FileNotFoundError(f"{processed_dir} has no *.cleaned.json or *.bitext.jsonl files to snapshot")
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", help="path to the data/raw/<work>/ folder this baseline is for")
    parser.add_argument(
        "--model-label",
        default="claude-sonnet-5-api",
        help="label identifying what produced this baseline (default: claude-sonnet-5-api)",
    )
    args = parser.parse_args(argv)

    written = make_baseline(args.folder, model_label=args.model_label)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
