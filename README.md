# ye_olde
A language translator using highly modern tools where the focus is on old and ancient languages.

Translates `(sentence, lang_code, year) → (sentence, lang_code, year)` — between any two points in a language's own history — using retrieval over a corpus partitioned by `(lang_code, year)`, not fine-tuning. Full design: [docs/diachronic-translation-pipeline-plan.md](docs/diachronic-translation-pipeline-plan.md).

## Status

Translation pipeline (`src/ye_olde/`): scaffolding only, no logic implemented yet.
Search API (`search_api/`): stood up and deployable, no real data yet. See the spec's §11 for
the first scoped build task (gather + align a corpus, then it has something to serve).

## Project layout

```
src/ye_olde/
├── api.py          # translate() — the public entrypoint (spec §1)
├── config.py       # settings: embedding model, generation LLM, Search API URL
├── resolver/       # (lang_code, year) -> weighted corpus partitions (§2, §3a)
├── classify/       # name vs. common-word vs. anachronism token classification (§2)
├── retrieval/      # thin HTTP client for the Search API (§10) + temporal rerank
├── fallback/       # native -> loan -> constructed -> temporal-loan chain (§2, §3a)
├── generation/      # grounded LLM generation + self-check (§2)
├── ingest/         # corpus ingestion into the queryable index (§3a, §5, §6)
└── schema/         # JSON Schema for community contributions (§3b)

search_api/         # standalone service (spec §10) — deployed to a Hugging Face Space,
                     # separately from the pipeline above; see search_api/README.md
contributions/      # community-submitted JSON entries (§3b), validated by CI on PR
data/               # gitignored corpora/index artifacts (raw/processed/index)
scripts/            # CLI entrypoints (ingestion runner, translate demo, search_api deploy)
tests/              # unit/ and integration/
```

## Setup

```
uv sync
cp .env.example .env   # fill in EMBEDDING_MODEL / LITELLM_MODEL / SEARCH_API_URL
```

## Search API (spec §10)

```
cd search_api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in HF_DATASET_REPO_ID once a dataset repo exists
uvicorn app.main:app --reload --port 8000
```

Deploys automatically to a Hugging Face Space on push to `main` — see
`.github/workflows/deploy-search-api.yml` and `scripts/deploy_search_api.py`. One-time setup
(HF account resources + GitHub secrets) isn't automatable from here; see the steps given
alongside this scaffold.
