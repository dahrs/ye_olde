#!/usr/bin/env python
"""CLI entrypoint for `ye_olde.ingest.survey` (spec §6 step -1: pre-
acquisition triage): starting from any index-of-works page on any
digitization site, follows promising links hop by hop until it reaches
actual document URLs worth investigating -- *before* anything is
downloaded.

How it works, stage by stage:
1. Fetch the page it's given.
2. Extract links -- pull out every real link on the page, throwing away
   site navigation like "Donate" or "Help".
3. Classify each link -- ask the LLM whether it looks like an old-language
   text worth chasing, a dictionary/grammar, irrelevant, or unclear --
   without auto-following an author's full bibliography page or any other
   unusually wide listing page (both too costly to expand automatically).
4. Resolve candidates -- for anything worth chasing, follow it one more
   hop to find the actual downloadable file, using the site's own
   "download formats" section when there is one.
5. Save results -- merge everything found into three running files: real
   candidates, dictionary/grammar-type references, and anything unclear
   that needs a human look.

Usage:
    python scripts/survey_catalog.py https://www.gutenberg.org/browse/languages/ang
    python scripts/survey_catalog.py <url> <url> ...

Works identically for any starting URL -- a single language's page, the
page listing all of them, an author index, or an equivalent page on another
site entirely; nothing here is Gutenberg-specific except two small optional
shortcuts in `ye_olde.ingest.survey` that just don't fire elsewhere.

Merges results into three accumulating YAML files under `--catalog-dir`
(default `data/catalog/`), keyed by document URL so re-running (more pages
later, or a retry) merges instead of duplicating:
- `bilingual_references.yaml` -- dictionaries/grammars/glossaries found.
- `acquisition_candidates.yaml` -- old-language primary-text candidates,
  ready for a human to check against `data/raw/` and queue into
  `data/sources.yaml`.
- `needs_manual_review.yaml` -- genuinely unclear from the page alone;
  meant to be resolved by hand (including live web search), not re-run.

`irrelevant` results are not written anywhere -- they only live in the
resume checkpoint, so a link already told "no" is never reclassified.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

from ye_olde.common.llm_client import get_usage_summary
from ye_olde.common.logging import get_logger
from ye_olde.ingest.checkpoint import Checkpoint
from ye_olde.ingest.survey import SurveyResult, crawl

_log = get_logger(__name__)

_OUTPUT_FILES = {
    "old_text_candidate": "acquisition_candidates.yaml",
    "bilingual_reference": "bilingual_references.yaml",
    "uncertain": "needs_manual_review.yaml",
}


def _result_key(result: SurveyResult) -> str:
    return result.document_url or result.found_via_chain[-1]


def _merge_into(path: Path, new_results: list[SurveyResult]) -> int:
    """Merges `new_results` into the YAML file at `path`, keyed by
    `_result_key` so a result already recorded there (from an earlier run,
    or found again via a different starting page) is updated in place
    rather than duplicated. Returns how many entries the file holds after
    merging.
    """
    existing: dict[str, dict[str, object]] = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
        existing = {entry["key"]: entry for entry in raw}

    for result in new_results:
        existing[_result_key(result)] = {"key": _result_key(result), **result.model_dump()}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(list(existing.values()), sort_keys=False, allow_unicode=True), encoding="utf-8")
    return len(existing)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("urls", nargs="+", help="one or more index-page URLs to crawl")
    parser.add_argument(
        "--catalog-dir",
        default="data/catalog",
        help="directory the three output YAML files are merged into (default: data/catalog)",
    )
    parser.add_argument("--max-depth", type=int, default=3, help="max hops to follow from each starting URL")
    parser.add_argument("--max-pages", type=int, default=300, help="max pages fetched per starting URL")
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=3.0,
        help="stop crawling a starting URL (gracefully, keeping what's already found) once its own "
        "LLM spend hits this -- confirmed live that page/depth limits alone don't bound $ spent "
        "when 'follow' cascades through author-bibliography pages (default: 3.0)",
    )
    parser.add_argument("--model", default=None, help="litellm model string, overriding LITELLM_MODEL from .env")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore/discard the resume checkpoint and re-fetch/re-classify every page",
    )
    args = parser.parse_args(argv)

    catalog_dir = Path(args.catalog_dir)
    failed: list[tuple[str, str]] = []
    all_results_by_category: dict[str, list[SurveyResult]] = {
        "old_text_candidate": [],
        "bilingual_reference": [],
        "uncertain": [],
    }

    for start_url in args.urls:
        # Python's builtin hash() is randomized per-process (PYTHONHASHSEED) -- useless for a
        # checkpoint filename that needs to resolve to the same path across separate runs.
        url_digest = hashlib.sha256(start_url.encode("utf-8")).hexdigest()[:16]
        checkpoint = Checkpoint(
            catalog_dir / f".survey_progress_{url_digest}.json",
            meta={"start_url": start_url, "model": args.model, "max_depth": args.max_depth},
        )
        if args.no_cache:
            checkpoint.clear()

        try:
            results = crawl(
                start_url,
                max_depth=args.max_depth,
                max_pages=args.max_pages,
                max_cost_usd=args.max_cost_usd,
                model=args.model,
                checkpoint=checkpoint,
            )
        except Exception as exc:
            # Boundary catch (CLAUDE.md "Error handling"), per starting URL -- these are
            # independent crawls, one site's failure shouldn't lose the others.
            _log.exception("survey crawl failed for %r", start_url)
            print(f"[FAILED] {start_url}: {exc}", file=sys.stderr)
            failed.append((start_url, str(exc)))
            continue

        for result in results:
            all_results_by_category[result.category].append(result)
        print(f"{start_url}: {len(results)} actionable result(s)", file=sys.stderr)

    for category, filename in _OUTPUT_FILES.items():
        count = _merge_into(catalog_dir / filename, all_results_by_category[category])
        print(f"{catalog_dir / filename}: {count} total entries", file=sys.stderr)

    usage = get_usage_summary()
    print(
        f"[cost] {usage['calls']} LLM calls, "
        f"{usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion tokens, "
        f"~${usage['cost_usd']:.4f}",
        file=sys.stderr,
    )
    if failed:
        print(f"\n{len(failed)} starting URL(s) failed:", file=sys.stderr)
        for url, reason in failed:
            print(f"  - {url}: {reason}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
