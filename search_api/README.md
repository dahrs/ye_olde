# ye_olde Search API

Serves the diachronic index described in `docs/diachronic-translation-pipeline-plan.md` §10 of
the main repo (github.com/dahrs/ye_olde). Reads Parquet shards from a Hugging Face Dataset repo
(`HF_DATASET_REPO_ID`) and exposes them over HTTP so the translation pipeline never needs the
corpus or index locally — this is the only thing that talks to the data.

Deployed to Google Cloud Run by `.github/workflows/deploy-search-api.yml` on every push to
`main` that touches this directory (data storage stays on the free Hugging Face Dataset repo;
only compute moved — Hugging Face's free CPU tier for Docker/Gradio Spaces was discontinued
mid-2026, see spec §10).

## Endpoints

- `GET /health`
- `GET /attest?lemma=&lang=&year=&window=` — exact/relational lookup (spec §3a) for the
  loanword/anachronism fallback chain (spec §2).
- `GET /lookup?text=&lang=&year=&target_lang=&target_year=` — Linguee-style aligned example
  lookup (spec §3c): attested sentences at the target period with the corresponding span
  highlighted. Currently token-exact matching only — semantic (embedding) matching is a
  follow-up once §6's alignment pipeline has produced real embeddings.

Both return an empty `results: []` rather than erroring when no data exists yet for the
requested `iso_code`/year range — this service is meant to run correctly before any corpus has
been gathered (spec §11).

## Local development

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in HF_DATASET_REPO_ID
uvicorn app.main:app --reload --port 8000
```
