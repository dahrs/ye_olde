#!/usr/bin/env python
"""Evaluation harness — NOT a unit/integration test; run this deliberately
(see eval/README.md), not automatically on every change to the pipeline.

Compares a local LLM's cleaning and alignment output against a fixed
Claude-API-produced baseline for the same work, as two independent
comparisons so a difference in one stage isn't misattributed to the other:

  1. Cleaning: the local model cleans the *same raw source files* the
     baseline was cleaned from. Compared against the baseline's cleaned
     units.
  2. Alignment: the local model aligns the *baseline's* cleaned units
     (not its own step-1 cleaning output) for both sides of the pair, so
     both alignment runs share identical input and only the aligning
     model differs. Compared against the baseline's aligned bitext.

If no baseline exists yet for this work, one is created automatically
from `data/processed/<work>/` (see make_baseline.py) before continuing.

Usage:
    python eval/run_local_model_eval.py data/raw/<work>
"""

from __future__ import annotations

# Edit these to hardcode which local model/server to evaluate; leave both
# None to fall back to .env's LITELLM_MODEL/LITELLM_API_BASE instead (the
# same setting the main pipeline uses). See eval/README.md.
LOCAL_MODEL: str | None = None
LOCAL_API_BASE: str | None = None

import argparse
import io
import itertools
import json
import os
import re
import sys
import time
from contextlib import contextmanager, redirect_stderr
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_baseline import make_baseline  # noqa: E402
from metrics import compare_alignment, compare_cleaning, format_report, slugify  # noqa: E402
from ye_olde.config import get_settings  # noqa: E402
from ye_olde.ingest.align import align_corpus_pair, write_jsonl  # noqa: E402
from ye_olde.ingest.clean import clean_corpus_file  # noqa: E402
from ye_olde.ingest.corpus_files import CorpusFile, discover_corpus_files  # noqa: E402
from ye_olde.ingest.extract import extract_text, strip_boilerplate  # noqa: E402
from ye_olde.ingest.llm_client import get_usage_summary, is_local_model, reset_usage  # noqa: E402

_DROPPED_RE = re.compile(r"verification: dropped (\d+)/(\d+) pairs")


class _Tee(io.TextIOBase):
    """Writes to two streams at once — lets align.py's live stderr progress
    still print to the console while also being captured for parsing (the
    verification-drop count has no other return path; see align.py).
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self):
        for stream in self._streams:
            stream.flush()


@contextmanager
def _temporary_api_base(api_base: str | None):
    """clean_corpus_file/align_corpus_pair take `model=` directly, but
    litellm's `api_base` is only ever read from `.env` inside llm_client —
    there's no parameter for it. This overrides it for the duration of the
    eval without touching the actual `.env` file.
    """
    if not api_base:
        yield
        return
    key = "LITELLM_API_BASE"
    previous = os.environ.get(key)
    os.environ[key] = api_base
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _resolve_local_model() -> tuple[str, str | None]:
    if LOCAL_MODEL:
        return LOCAL_MODEL, LOCAL_API_BASE
    settings = get_settings()
    if not settings.litellm_model:
        raise SystemExit(
            "No local model configured: set LOCAL_MODEL at the top of this script, or LITELLM_MODEL in .env."
        )
    return settings.litellm_model, settings.litellm_api_base or None


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _find_or_make_baseline(work_folder: Path) -> dict[str, Path]:
    work_name = work_folder.name
    baseline_dir = Path("eval/baselines") / work_name
    cleaned = sorted(baseline_dir.glob("*.baseline_*.cleaned.json")) if baseline_dir.is_dir() else []
    bitext = sorted(baseline_dir.glob("*.baseline_*.bitext.jsonl")) if baseline_dir.is_dir() else []
    if not cleaned or not bitext:
        print(f"[eval] no baseline found for {work_name!r} — creating one from data/processed/{work_name}/")
        make_baseline(work_folder, model_label="claude-sonnet-5-api")
        cleaned = sorted(baseline_dir.glob("*.baseline_*.cleaned.json"))
        bitext = sorted(baseline_dir.glob("*.baseline_*.bitext.jsonl"))

    by_key: dict[str, Path] = {path.name.split(".baseline_")[0]: path for path in cleaned}
    if len(bitext) != 1:
        raise SystemExit(f"expected exactly 1 baseline bitext file for {work_name!r}, found {len(bitext)}: {bitext}")
    by_key["__bitext__"] = bitext[0]
    return by_key


def run_cleaning_eval(
    files: list[CorpusFile], baseline_by_key: dict[str, Path], model: str, tag: str, ts: str
) -> dict[str, dict]:
    out_dir = Path("eval/outputs") / files[0].path.parent.name / "cleaning"
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    for cf in files:
        baseline_path = baseline_by_key.get(cf.path.stem)
        if baseline_path is None:
            print(f"[eval] skip cleaning-compare for {cf.path.name}: no baseline cleaned file for it")
            continue
        baseline_units = json.loads(baseline_path.read_text(encoding="utf-8"))

        raw_text = strip_boilerplate(extract_text(cf.path))
        reset_usage()
        start = time.monotonic()
        local_units = clean_corpus_file(cf, raw_text, cache_path=None, model=model, use_cache=False)
        duration = time.monotonic() - start
        usage = get_usage_summary()

        out_path = out_dir / f"{cf.path.stem}.{tag}.{ts}.json"
        out_path.write_text(json.dumps(local_units, ensure_ascii=False, indent=2), encoding="utf-8")

        metrics = compare_cleaning(baseline_units, local_units, raw_text_len=len(raw_text))
        metrics.update(
            local_cost_usd=usage["cost_usd"],
            local_duration_seconds=duration,
            local_output=str(out_path),
            baseline_source=str(baseline_path),
        )
        reports[cf.path.name] = metrics
    return reports


def run_alignment_eval(
    files: list[CorpusFile], baseline_by_key: dict[str, Path], model: str, tag: str, ts: str
) -> dict[str, dict]:
    if len(files) < 2:
        raise SystemExit("need at least 2 source files to evaluate alignment")

    baseline_bitext_path = baseline_by_key["__bitext__"]
    baseline_records = [json.loads(line) for line in baseline_bitext_path.read_text(encoding="utf-8").splitlines()]

    out_dir = Path("eval/outputs") / files[0].path.parent.name / "alignment"
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    for cf_a, cf_b in itertools.combinations(sorted(files, key=lambda cf: cf.year), 2):
        base_a, base_b = baseline_by_key.get(cf_a.path.stem), baseline_by_key.get(cf_b.path.stem)
        if base_a is None or base_b is None:
            print(
                f"[eval] skip alignment-compare for {cf_a.lang_code}-{cf_b.lang_code}: missing baseline cleaned file(s)"
            )
            continue
        units_a = json.loads(base_a.read_text(encoding="utf-8"))
        units_b = json.loads(base_b.read_text(encoding="utf-8"))
        pair_records = [
            r
            for r in baseline_records
            if r["source"]["lang_code"] == cf_a.lang_code and r["target"]["lang_code"] == cf_b.lang_code
        ]

        reset_usage()
        stderr_buffer = io.StringIO()
        start = time.monotonic()
        with redirect_stderr(_Tee(sys.stderr, stderr_buffer)):
            local_records = align_corpus_pair(
                cf_a, cf_b, units_a, units_b, mode="hybrid", model=model, embedding_model=None
            )
        duration = time.monotonic() - start
        usage = get_usage_summary()

        local_dropped = local_candidates = None
        match = _DROPPED_RE.search(stderr_buffer.getvalue())
        if match:
            local_dropped, local_candidates = int(match.group(1)), int(match.group(2))

        out_path = out_dir / f"{cf_a.lang_code}-{cf_b.lang_code}.{tag}.{ts}.bitext.jsonl"
        write_jsonl(local_records, out_path)

        metrics = compare_alignment(
            pair_records,
            local_records,
            source_units=units_a,
            target_units=units_b,
            local_dropped=local_dropped,
            local_candidates=local_candidates,
        )
        metrics.update(
            local_cost_usd=usage["cost_usd"],
            local_duration_seconds=duration,
            local_output=str(out_path),
            baseline_source=str(baseline_bitext_path),
        )
        reports[f"{cf_a.lang_code}-{cf_b.lang_code}"] = metrics
    return reports


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", help="path to the data/raw/<work>/ folder to evaluate")
    args = parser.parse_args(argv)

    work_folder = Path(args.folder)
    model, api_base = _resolve_local_model()
    print(f"[eval] evaluating model={model!r} api_base={api_base!r} against {work_folder.name}")
    if not is_local_model(model):
        print(
            f"[eval] warning: {model!r} doesn't resolve to a known local-inference provider — "
            "this will make real, billed API calls if run.",
            file=sys.stderr,
        )

    with _temporary_api_base(api_base):
        baseline_by_key = _find_or_make_baseline(work_folder)
        files = discover_corpus_files(work_folder)

        tag, ts = slugify(model), _timestamp()
        cleaning_reports = run_cleaning_eval(files, baseline_by_key, model, tag, ts)
        alignment_reports = run_alignment_eval(files, baseline_by_key, model, tag, ts)

    sections = {f"Cleaning: {name}": m for name, m in cleaning_reports.items()}
    sections.update({f"Alignment: {name}": m for name, m in alignment_reports.items()})

    report_text = format_report(f"Local model eval: {model} vs. baseline ({work_folder.name})", sections)
    report_dir = Path("eval/reports") / work_folder.name
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{tag}.{ts}.md"
    report_path.write_text(report_text, encoding="utf-8")

    print()
    print(report_text)
    print(f"[eval] report written to {report_path}")


if __name__ == "__main__":
    main()
