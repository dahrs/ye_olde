---
title: Ye Olde Search API
emoji: 📜
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ye_olde Search API

Serves the diachronic index described in `docs/diachronic-translation-pipeline-plan.md` §10 of
the main repo (github.com/dahrs/ye_olde). Reads Parquet shards from a Hugging Face Dataset repo
(HF_DATASET_REPO_ID) and exposes them over HTTP so the translation pipeline never needs the
corpus or index locally — this is the only thing that talks to the data.

Deployed by `.github/workflows/deploy-search-api.yml` on every push to `main` that touches this
directory. Not a UI — this Space exists only to run the Docker container behind the endpoints
below.

## Endpoints

- `GET /health`
- `GET /attest?lemma=&lang=&year=&window=` — exact/relational lookup (spec §3a) for the
  loanword/anachronism fallback chain (spec §2).
- `GET /lookup?text=&lang=&year=&target_lang=&target_year=` — Linguee-style aligned example
  lookup (spec §3c): attested sentences at the target period with the corresponding span
  highlighted. Currently token-exact matching only — semantic (embedding) matching is a
  follow-up once §6's alignment pipeline has produced real embeddings.

Both return an empty `results: []` rather than erroring when no data exists yet for the
requested `iso_code`/year range — this Space is meant to run correctly before any corpus has
been gathered (spec §11).
