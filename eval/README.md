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
├── run_local_model_eval.py   # the comparison itself
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
