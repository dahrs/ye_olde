"""Comparison metrics shared by eval scripts (see eval/README.md). Kept
deliberately separate from `ye_olde.ingest` — these are evaluation-only
concerns (how do two runs compare?), not pipeline concerns (how do you
produce a run?), and have no reason to ship with the installed package.

Every metric function returns a plain JSON-serializable dict, so a report
is always just "dump these dicts" — see `format_report`.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter


def slugify(text: str) -> str:
    """Filesystem-safe component for a filename, e.g. for a model string
    ("ollama/llama3.2:3b") or work title.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return slug or "unknown"


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).lower()


def compare_cleaning(baseline_units: list[str], local_units: list[str], *, raw_text_len: int | None = None) -> dict:
    """Baseline vs. local-model cleaning output. `overlap` is a strict,
    exact-after-normalization match on whole units — a legitimate but
    differently-split/merged unit (one model keeping two sentences as one
    unit where the other keeps them as two) won't count as agreement here.
    That's a known conservative bias in this first version of the metric,
    not a bug; a fuzzier unit-alignment comparison is a reasonable future
    addition to this file.
    """
    base_norm = [_normalize(u) for u in baseline_units]
    local_norm = [_normalize(u) for u in local_units]
    base_set, local_set = set(base_norm), set(local_norm)
    overlap = base_set & local_set

    result = {
        "baseline_unit_count": len(baseline_units),
        "local_unit_count": len(local_units),
        "exact_overlap_units": len(overlap),
        "overlap_precision": len(overlap) / len(local_set) if local_set else 0.0,
        "overlap_recall": len(overlap) / len(base_set) if base_set else 0.0,
    }
    if raw_text_len:
        result["baseline_coverage_ratio"] = sum(len(u) for u in baseline_units) / raw_text_len
        result["local_coverage_ratio"] = sum(len(u) for u in local_units) / raw_text_len
    return result


def _pair_key(record: dict) -> tuple[str, str]:
    return _normalize(record["source_text"]), _normalize(record["target_text"])


def _confidence_link_stats(records: list[dict]) -> dict:
    if not records:
        return {"avg_confidence": 0.0, "avg_links_per_pair": 0.0, "method_split": {}}
    confidences = [r.get("sentence_confidence", 0.0) or 0.0 for r in records]
    link_counts = [len(r.get("alignment_links") or []) for r in records]
    methods = Counter(r.get("sentence_method", "unknown").split(":")[0] for r in records)
    return {
        "avg_confidence": sum(confidences) / len(records),
        "avg_links_per_pair": sum(link_counts) / len(records),
        "method_split": dict(methods),
    }


def _coverage_ratio(records: list[dict], units: list[str], side: str) -> float:
    """Rough approximation: total character length of every pair's `side`
    text (source_text/target_text) as a fraction of the cleaned units'
    total character length. Not exact — pairs can overlap the same
    underlying material — but a reasonable proxy for "how much of the
    available cleaned content ended up used in some pair."
    """
    total_units_len = sum(len(u) for u in units)
    if not total_units_len:
        return 0.0
    total_pairs_len = sum(len(r[f"{side}_text"]) for r in records)
    return total_pairs_len / total_units_len


def compare_alignment(
    baseline_records: list[dict],
    local_records: list[dict],
    *,
    source_units: list[str],
    target_units: list[str],
    baseline_dropped: int | None = None,
    baseline_candidates: int | None = None,
    local_dropped: int | None = None,
    local_candidates: int | None = None,
) -> dict:
    """Baseline vs. local-model alignment output, where both were run
    against the *same* cleaned source/target units (see run_local_model_eval.py
    — only the aligning model differs between the two runs being compared).
    """
    base_keys = {_pair_key(r) for r in baseline_records}
    local_keys = {_pair_key(r) for r in local_records}
    full_overlap = base_keys & local_keys

    base_sources = {k[0] for k in base_keys}
    local_sources = {k[0] for k in local_keys}
    source_overlap = base_sources & local_sources

    result = {
        "baseline_pair_count": len(baseline_records),
        "local_pair_count": len(local_records),
        "full_pair_overlap": len(full_overlap),
        "full_pair_precision": len(full_overlap) / len(local_keys) if local_keys else 0.0,
        "full_pair_recall": len(full_overlap) / len(base_keys) if base_keys else 0.0,
        "source_span_overlap": len(source_overlap),
        "source_span_precision": len(source_overlap) / len(local_sources) if local_sources else 0.0,
        "source_span_recall": len(source_overlap) / len(base_sources) if base_sources else 0.0,
        "baseline_source_coverage": _coverage_ratio(baseline_records, source_units, "source"),
        "local_source_coverage": _coverage_ratio(local_records, source_units, "source"),
        "baseline_target_coverage": _coverage_ratio(baseline_records, target_units, "target"),
        "local_target_coverage": _coverage_ratio(local_records, target_units, "target"),
        "baseline": _confidence_link_stats(baseline_records),
        "local": _confidence_link_stats(local_records),
    }
    if baseline_candidates:
        result["baseline_verification_drop_rate"] = (baseline_dropped or 0) / baseline_candidates
    if local_candidates:
        result["local_verification_drop_rate"] = (local_dropped or 0) / local_candidates
    return result


def format_report(title: str, sections: dict[str, dict]) -> str:
    """Renders metric dicts as a plain Markdown report."""
    lines = [f"# {title}", ""]
    for heading, metrics in sections.items():
        lines.append(f"## {heading}")
        lines.append("")
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"- **{key}**: {value:.4f}")
            elif isinstance(value, dict):
                lines.append(f"- **{key}**: {value}")
            else:
                lines.append(f"- **{key}**: {value}")
        lines.append("")
    return "\n".join(lines)
