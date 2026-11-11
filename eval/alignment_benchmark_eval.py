#!/usr/bin/env python
"""Evaluation harness — NOT a unit/integration test; run this deliberately
(see eval/README.md), not automatically on every change to the pipeline.

Benchmarks candidate sentence-alignment *scoring* methods (an embedding
model's cosine similarity, or a plain LLM call) against a small,
hand-verified set of positive (genuinely-aligned) and negative
(certainly-not-aligned) sentence pairs per language pair. The question
this answers is "which method actually tells positives apart from
negatives for this language pair" — a much cheaper, more targeted question
than "which method produces the best full-document alignment"
(`run_local_model_eval.py` already covers that one, for the English-family
case).

Extend this by adding to, not editing around:
  - A new embedding model to compare: add its name to EMBEDDING_MODELS below.
  - A new language pair to test: add a new `eval/alignment_benchmarks/<a>_<b>.yaml`
    file (copy an existing one's shape) — every file under that directory is
    picked up automatically.
  - A new scoring method entirely (not an embedding model, not the plain LLM
    call): add a function with the same `(source, target, lang_a, lang_b, ...)
    -> float | None` shape to `alignment_methods.py`, then call it in
    `run_benchmark` below alongside the existing ones.

Usage:
    python eval/alignment_benchmark_eval.py
    python eval/alignment_benchmark_eval.py --benchmark grc_eng
    python eval/alignment_benchmark_eval.py --model claude_code_cli/claude-sonnet-5
"""

from __future__ import annotations

# Edit this to change which embedding models get compared — every entry is
# tried against every benchmark file found under eval/alignment_benchmarks/.
EMBEDDING_MODELS: list[str] = [
    "intfloat/multilingual-e5-large",
    "bowphs/SPhilBerta",
]
# Whether to also benchmark the "plain LLM call" method (alignment_methods.
# llm_score) alongside the embedding models above — real LLM calls cost
# money, so this is a real decision, not a hardcoded truth.
INCLUDE_LLM_METHOD = True

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alignment_methods import embedding_score, llm_score  # noqa: E402
from metrics import format_report  # noqa: E402
from ye_olde.common.llm_client import get_usage_summary  # noqa: E402

_BENCHMARKS_DIR = Path(__file__).resolve().parent / "alignment_benchmarks"
_REPORTS_DIR = Path(__file__).resolve().parent / "reports" / "alignment_benchmarks"

ScoreFn = Callable[[str, str, str, str], "float | None"]


def load_benchmark(path: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded


def _score_group(pairs: list[dict[str, Any]], score_fn: ScoreFn, lang_a: str, lang_b: str) -> list[float | None]:
    return [score_fn(p["source"], p["target"], lang_a, lang_b) for p in pairs]


def _separation_metrics(positive_scores: list[float | None], negative_scores: list[float | None]) -> dict[str, Any]:
    """Summarizes how well a method's scores for the positives separate
    from its scores for the negatives: the smallest a positive scored,
    the largest a negative scored, whether the two groups are cleanly
    separated at all (`min(positives) > max(negatives)`), and a basic
    `score >= 0.5` accuracy check. `None` scores (a method that genuinely
    couldn't produce one) are excluded from the numbers but counted, not
    silently dropped.
    """
    pos = [s for s in positive_scores if s is not None]
    neg = [s for s in negative_scores if s is not None]
    missing = (len(positive_scores) - len(pos)) + (len(negative_scores) - len(neg))
    result: dict[str, Any] = {
        "positive_scores": [round(s, 4) for s in pos],
        "negative_scores": [round(s, 4) for s in neg],
        "missing_scores": missing,
    }
    if pos:
        result["positive_min"] = round(min(pos), 4)
    if neg:
        result["negative_max"] = round(max(neg), 4)
    if pos and neg:
        result["cleanly_separated"] = min(pos) > max(neg)
        correct = sum(1 for s in pos if s >= 0.5) + sum(1 for s in neg if s < 0.5)
        result["accuracy_at_0.5"] = round(correct / (len(pos) + len(neg)), 4)
    return result


def run_benchmark(benchmark: dict[str, Any], *, model: str | None) -> dict[str, dict[str, Any]]:
    """Runs every configured method against one loaded benchmark file's
    positives/negatives, returning `{method_label: separation_metrics}`.
    """
    lang_a, lang_b = benchmark["source_lang"], benchmark["target_lang"]
    positives, negatives = benchmark["positives"], benchmark["negatives"]

    sections: dict[str, dict[str, Any]] = {}
    for model_name in EMBEDDING_MODELS:

        def embed_fn(source: str, target: str, lang_a: str, lang_b: str, _model_name: str = model_name) -> float | None:
            return embedding_score(source, target, lang_a, lang_b, model_name=_model_name)

        pos_scores = _score_group(positives, embed_fn, lang_a, lang_b)
        neg_scores = _score_group(negatives, embed_fn, lang_a, lang_b)
        sections[f"embedding: {model_name}"] = _separation_metrics(pos_scores, neg_scores)

    if INCLUDE_LLM_METHOD:

        def llm_fn(source: str, target: str, lang_a: str, lang_b: str) -> float | None:
            return llm_score(source, target, lang_a, lang_b, model=model)

        pos_scores = _score_group(positives, llm_fn, lang_a, lang_b)
        neg_scores = _score_group(negatives, llm_fn, lang_a, lang_b)
        sections["llm call"] = _separation_metrics(pos_scores, neg_scores)

    return sections


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--benchmark", default=None, help="run only <name> from alignment_benchmarks/<name>.yaml (default: all)"
    )
    parser.add_argument("--model", default=None, help="litellm model string for the LLM method, overriding .env")
    parser.add_argument("--out-dir", default=str(_REPORTS_DIR), help="where to write the per-benchmark reports")
    args = parser.parse_args(argv)

    if args.benchmark:
        name = args.benchmark
        benchmark_paths = [Path(name) if name.endswith(".yaml") else _BENCHMARKS_DIR / f"{name}.yaml"]
    else:
        benchmark_paths = sorted(_BENCHMARKS_DIR.glob("*.yaml"))
    if not benchmark_paths:
        print(f"no benchmark files found under {_BENCHMARKS_DIR}", file=sys.stderr)
        raise SystemExit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for path in benchmark_paths:
            benchmark = load_benchmark(path)
            sections = run_benchmark(benchmark, model=args.model)
            report = format_report(f"Alignment method benchmark: {path.stem}", sections)
            out_path = out_dir / f"{path.stem}.md"
            out_path.write_text(report, encoding="utf-8")
            print(report)
            print(f"[alignment_benchmark_eval] wrote {out_path}", file=sys.stderr)
    finally:
        usage = get_usage_summary()
        print(
            f"[cost] {usage['calls']} LLM calls, "
            f"{usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion tokens, "
            f"~${usage['cost_usd']:.4f}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
