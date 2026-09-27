#!/usr/bin/env python
"""Evaluation harness — NOT a unit/integration test; run this deliberately
(see eval/README.md), not automatically on every change to the pipeline.

Compares two independently-produced cleaning+alignment runs of the *same*
work — e.g. two Claude API runs of the same source files, to check how
deterministic an LLM's cleaning/alignment judgment calls are across
repeated runs. This is a same-model (or at least externally-supplied,
unverified-model) determinism check, not a model comparison — for that,
see run_local_model_eval.py, which compares a local model against a fixed
baseline instead.

A "run" is a directory holding `<corpus-file-stem><suffix>.cleaned.json`
per source file and `<lang_a>-<lang_b><suffix>.bitext.jsonl` per language
pair — this covers both `data/processed/<work>/` (suffix="") and an
`eval/baselines/<work>/` snapshot (suffix=".baseline_<tag>_<timestamp>").

Usage:
    python eval/compare_runs.py data/raw/<work> \\
        --run-a-dir eval/baselines/<work> \\
        --run-a-suffix ".baseline_claude-sonnet-5-api-run1-imperfect_20260924T000348Z" \\
        --run-a-label "run1 (imperfect)" --run-a-model "anthropic/claude-sonnet-5" \\
        --run-b-dir data/processed/<work> \\
        --run-b-label "run2" --run-b-model "anthropic/claude-sonnet-5"
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import compare_alignment, compare_cleaning, format_report, slugify  # noqa: E402
from ye_olde.ingest.clean import CleanedDocument  # noqa: E402
from ye_olde.ingest.corpus_files import discover_corpus_files  # noqa: E402
from ye_olde.ingest.extract import extract_text, strip_boilerplate  # noqa: E402


def load_units(path: Path) -> list[str]:
    """A `.cleaned.json` predating the `chunk_unit_counts` field (see
    `ye_olde.ingest.clean.CleanedDocument`) is a bare JSON array of units
    rather than a `CleanedDocument` object — handle both so this script
    keeps working when one side of the comparison is an older run.
    """
    text = path.read_text(encoding="utf-8")
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return parsed
    return CleanedDocument.model_validate_json(text).units


def _cleaned_path(run_dir: Path, stem: str, suffix: str) -> Path:
    return run_dir / f"{stem}{suffix}.cleaned.json"


def _bitext_path(run_dir: Path, lang_pair: str, suffix: str) -> Path:
    return run_dir / f"{lang_pair}{suffix}.bitext.jsonl"


def compare_runs(
    work_folder: str | Path,
    *,
    run_a_dir: str | Path,
    run_a_suffix: str,
    run_a_label: str,
    run_a_model: str,
    run_b_dir: str | Path,
    run_b_suffix: str,
    run_b_label: str,
    run_b_model: str,
    run_a_dropped: int | None = None,
    run_a_candidates: int | None = None,
    run_b_dropped: int | None = None,
    run_b_candidates: int | None = None,
) -> str:
    work_folder = Path(work_folder)
    run_a_dir, run_b_dir = Path(run_a_dir), Path(run_b_dir)
    files = discover_corpus_files(work_folder)

    files_compared: list[str] = []
    cleaning_sections: dict[str, dict] = {}
    units_by_stem: dict[str, tuple[list[str], list[str]]] = {}

    for cf in files:
        stem = cf.path.stem
        path_a, path_b = _cleaned_path(run_a_dir, stem, run_a_suffix), _cleaned_path(run_b_dir, stem, run_b_suffix)
        units_a, units_b = load_units(path_a), load_units(path_b)
        units_by_stem[stem] = (units_a, units_b)

        raw_len = len(strip_boilerplate(extract_text(cf.path)))
        cleaning_sections[f"{stem} ({cf.lang_code})"] = compare_cleaning(units_a, units_b, raw_text_len=raw_len)
        files_compared += [f"source: {cf.path}", f"{run_a_label} cleaned: {path_a}", f"{run_b_label} cleaned: {path_b}"]

    alignment_sections: dict[str, dict] = {}
    for cf_a, cf_b in itertools.combinations(sorted(files, key=lambda cf: cf.year), 2):
        lang_pair = f"{cf_a.lang_code}-{cf_b.lang_code}"
        path_a, path_b = (
            _bitext_path(run_a_dir, lang_pair, run_a_suffix),
            _bitext_path(run_b_dir, lang_pair, run_b_suffix),
        )
        if not path_a.exists() or not path_b.exists():
            continue
        records_a = [json.loads(line) for line in path_a.read_text(encoding="utf-8").splitlines()]
        records_b = [json.loads(line) for line in path_b.read_text(encoding="utf-8").splitlines()]
        # Coverage ratios need one shared unit list for the denominator (see
        # compare_alignment) -- run_b's cleaned units are used for both
        # sides' coverage here, so those numbers are relative to run_b, not
        # an average of the two runs' (differing) cleaning output.
        _, units_b_a = units_by_stem[cf_a.path.stem]
        _, units_b_b = units_by_stem[cf_b.path.stem]
        metrics = compare_alignment(
            records_a,
            records_b,
            source_units=units_b_a,
            target_units=units_b_b,
            local_dropped=run_b_dropped,
            local_candidates=run_b_candidates,
        )
        if run_a_dropped is not None and run_a_candidates:
            metrics["run_a_verification_drop_rate"] = run_a_dropped / run_a_candidates
        metrics["run_a_cleaned_unit_counts"] = {
            cf_a.lang_code: len(units_by_stem[cf_a.path.stem][0]),
            cf_b.lang_code: len(units_by_stem[cf_b.path.stem][0]),
        }
        metrics["run_b_cleaned_unit_counts"] = {
            cf_a.lang_code: len(units_by_stem[cf_a.path.stem][1]),
            cf_b.lang_code: len(units_by_stem[cf_b.path.stem][1]),
        }
        alignment_sections[f"{lang_pair} (coverage relative to {run_b_label}'s cleaned units)"] = metrics
        files_compared += [f"{run_a_label} bitext: {path_a}", f"{run_b_label} bitext: {path_b}"]

    header = {
        "compared_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "work": work_folder.name,
        f"{run_a_label} model": run_a_model,
        f"{run_b_label} model": run_b_model,
        "files_compared": files_compared,
        "caveat": (
            "Cleaning/alignment overlap below is a strict exact-string match after "
            "whitespace/case normalization (see eval/metrics.py compare_cleaning). A "
            "legitimate but differently-split-or-merged unit (e.g. one run keeping two "
            "lines as a single unit where the other splits them -- a real risk on "
            "irregularly-punctuated source text) counts as a full mismatch here even "
            "when no content was lost or changed, only re-segmented. Low overlap alongside "
            "similar *_coverage_ratio values (fraction of raw text kept) points at "
            "segmentation differences rather than content differences; low overlap "
            "alongside diverging coverage ratios points at real content differences."
        ),
    }

    sections = {"Run metadata": header}
    sections.update({f"Cleaning: {k}": v for k, v in cleaning_sections.items()})
    sections.update({f"Alignment: {k}": v for k, v in alignment_sections.items()})
    return format_report(f"Run comparison: {run_a_label} vs. {run_b_label} ({work_folder.name})", sections)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", help="path to the data/raw/<work>/ folder both runs were produced from")
    parser.add_argument("--run-a-dir", required=True, help="directory holding run A's *.cleaned.json/*.bitext.jsonl")
    parser.add_argument(
        "--run-a-suffix",
        default="",
        help='inserted before ".cleaned.json"/".bitext.jsonl", e.g. ".baseline_<tag>_<ts>"',
    )
    parser.add_argument("--run-a-label", required=True, help='short label for run A, e.g. "run1 (imperfect)"')
    parser.add_argument("--run-a-model", required=True, help="litellm model string that produced run A")
    parser.add_argument(
        "--run-a-dropped", type=int, default=None, help="run A's align.py verification-drop numerator, if known"
    )
    parser.add_argument(
        "--run-a-candidates", type=int, default=None, help="run A's align.py verification-drop denominator, if known"
    )
    parser.add_argument("--run-b-dir", required=True, help="directory holding run B's *.cleaned.json/*.bitext.jsonl")
    parser.add_argument(
        "--run-b-suffix",
        default="",
        help='inserted before ".cleaned.json"/".bitext.jsonl", e.g. ".baseline_<tag>_<ts>"',
    )
    parser.add_argument("--run-b-label", required=True, help='short label for run B, e.g. "run2"')
    parser.add_argument("--run-b-model", required=True, help="litellm model string that produced run B")
    parser.add_argument(
        "--run-b-dropped", type=int, default=None, help="run B's align.py verification-drop numerator, if known"
    )
    parser.add_argument(
        "--run-b-candidates", type=int, default=None, help="run B's align.py verification-drop denominator, if known"
    )
    parser.add_argument(
        "--out", default=None, help="report output path (default: eval/reports/<work>/<a-label>-vs-<b-label>.md)"
    )
    args = parser.parse_args(argv)

    work_folder = Path(args.folder)
    report = compare_runs(
        work_folder,
        run_a_dir=args.run_a_dir,
        run_a_suffix=args.run_a_suffix,
        run_a_label=args.run_a_label,
        run_a_model=args.run_a_model,
        run_a_dropped=args.run_a_dropped,
        run_a_candidates=args.run_a_candidates,
        run_b_dir=args.run_b_dir,
        run_b_suffix=args.run_b_suffix,
        run_b_label=args.run_b_label,
        run_b_model=args.run_b_model,
        run_b_dropped=args.run_b_dropped,
        run_b_candidates=args.run_b_candidates,
    )

    out_path = (
        Path(args.out)
        if args.out
        else Path("eval/reports") / work_folder.name / f"{slugify(args.run_a_label)}-vs-{slugify(args.run_b_label)}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[compare] report written to {out_path}")


if __name__ == "__main__":
    main()
