# eval/

Evaluation tooling for the ingestion pipeline (`ye_olde.ingest`) — **not**
unit or acceptance tests. Nothing here runs automatically on a change or a
commit; these scripts are run deliberately, when there's a specific
question to answer (e.g. "is this local model good enough to replace the
Claude API for this pipeline?"). This is a home for that kind of tooling
generally — more scripts/baselines are expected to be added over time as
new questions come up, not just the local-model comparison below.

## Local model vs. Claude API comparison

`run_local_model_eval.py` compares a local LLM's cleaning and alignment
output against a fixed Claude-API-produced baseline for the same work, as
two **independent** comparisons so a difference in one pipeline stage
isn't misattributed to the other:

1. **Cleaning**: the local model cleans the *same raw source files* the
   baseline was cleaned from. Compared against the baseline's cleaned
   units.
2. **Alignment**: the local model aligns the *baseline's* cleaned units
   (not its own step-1 cleaning output) for both sides of the pair, so
   both alignment runs share identical input and only the aligning model
   differs. Compared against the baseline's aligned bitext.

### Usage

```bash
python eval/run_local_model_eval.py data/raw/<work>
```

Which local model/server to test is set at the top of
`run_local_model_eval.py` (`LOCAL_MODEL`/`LOCAL_API_BASE`) — edit those
directly, or leave both `None` to fall back to `.env`'s
`LITELLM_MODEL`/`LITELLM_API_BASE` (the same setting the main pipeline
uses).

If no baseline exists yet for the given work, one is created automatically
from `data/processed/<work>/` (must already have been produced by
`scripts/align_corpus.py`) — see `make_baseline.py` to create/refresh one
explicitly instead. A baseline is a **stable reference snapshot**: it's
not silently recreated on every eval run, so comparing several different
local models against the same work always compares against the same fixed
point.

### Layout

```
eval/
├── metrics.py               # comparison metric functions (shared, reusable)
├── make_baseline.py          # snapshot data/processed/<work>/ -> eval/baselines/<work>/
├── run_local_model_eval.py   # local model vs. Claude baseline comparison
├── compare_runs.py           # two arbitrary runs of the same work vs. each other (e.g. determinism checks)
├── baselines/<work>/         # frozen reference snapshots, tagged <model>_<timestamp>
├── outputs/<work>/           # local-model run outputs, tagged <model>_<timestamp>
└── reports/<work>/           # generated comparison reports (Markdown)
```

### Metrics reported

- **Pair/unit count & coverage**: how much each run produced, and (for
  alignment) the deterministic-verification drop rate.
- **Content overlap**: how much the local model's output agrees with the
  baseline's, at both a strict (exact match) and looser (same source span)
  level — a rough precision/recall proxy against the API as reference.
- **Confidence & link density**: average `sentence_confidence`, average
  `alignment_links` per pair, and the embedding-vs-llm method split
  (hybrid mode only).
- **Cost & wall-clock time**: local-model cost (should be ~$0) and
  duration. The baseline's original cost/time isn't reconstructable after
  the fact unless it was recorded at baseline-creation time — treat its
  absence as "not recorded," not "free."

## Same-work determinism check (e.g. Claude run vs. Claude run)

`compare_runs.py` compares two independently-produced cleaning+alignment
runs of the *same* work against each other — unlike `run_local_model_eval.py`,
neither run has to be a fixed "baseline" and both can be the same model
(this is how Claude's own run-to-run determinism was checked for Sir
Gawayne: re-running `scripts/align_corpus.py --no-cache` on unchanged
source files and comparing the fresh output against an archived earlier
run). It reuses the same `compare_cleaning`/`compare_alignment` metrics,
and its report always records exactly which files, models, and timestamp
were compared (see `eval/reports/<work>/*.md`) so a report is self-contained
evidence, not something you have to reconstruct from memory later.

### Usage

```bash
python eval/compare_runs.py data/raw/<work> \
    --run-a-dir eval/baselines/<work> --run-a-suffix ".baseline_<tag>_<timestamp>" \
    --run-a-label "run1-imperfect" --run-a-model "anthropic/claude-sonnet-5" \
    --run-b-dir data/processed/<work> \
    --run-b-label "run2" --run-b-model "anthropic/claude-sonnet-5" \
    --run-b-dropped 8 --run-b-candidates 760
```

`--run-a-dropped`/`--run-a-candidates`/`--run-b-dropped`/`--run-b-candidates`
are align.py's verification-drop numerator/denominator for each run, if
known (align.py only prints this to stderr, so it's lost for a run whose
console output wasn't captured — leave the flag off rather than guess; the
report only shows a drop rate when both numbers for that run are given).

### Known limitations (first version — improve as needed, not exhaustive)

- Content-overlap metrics compare units/pairs after whitespace/case
  normalization only. A legitimate but differently-split or -merged
  unit/pair (one model keeping two sentences together where the other
  splits them) won't count as agreement — a real but conservative bias in
  the metric, not a bug.
- No local server exists in some environments (this was true of the
  sandbox this was first built in) — the script was validated with a
  smoke test using a faked local model, not a real one. Point it at an
  actual running local server to get a real comparison.
- `compare_cleaning` compares baseline and local units as flat,
  whole-file lists — it doesn't yet use each unit's source-chunk
  provenance (`CleanedDocument.chunk_unit_counts`, see
  `ye_olde.ingest.clean`) to compare chunk-for-chunk instead. That
  provenance is saved by every cleaning run regardless of model, so a
  chunk-scoped comparison is a reasonable future addition here.
- A `.cleaned.json` written before `chunk_unit_counts` existed is a bare
  JSON array, not a `CleanedDocument` object — `clean_corpus_file` and
  `run_local_model_eval.py` both now expect the new shape and will raise
  a pydantic validation error on an old-format file. Re-clean (or
  re-run `make_baseline.py` against a freshly re-cleaned
  `data/processed/<work>/`) to pick up the new format; there's no
  migration path back onto an old cache file.
