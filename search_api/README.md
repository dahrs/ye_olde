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
  highlighted. Token-exact matching only.
- `GET /search?text=&lang=&year=&window=&top_k=` — semantic passage search: ranks indexed
  passages by embedding similarity to `text` (not exact match), across every language pair that
  has `lang` on either side, optionally restricted to within `window` years of `year`. Each
  result carries its aligned counterpart passage (spec §3c) alongside the match, since every
  indexed passage today comes from a pair. This is what a RAG-style "find a sentence, not just
  an exact word" query needs, and what `generation/`'s eventual context-bundle step (spec §2)
  will build on. Embeds the query with `sentence-transformers` at request time
  (`EMBEDDING_MODEL`, must match whatever built the indexed shards — see
  `ye_olde.ingest.index`) — this is why the deployed service runs at `--memory=4Gi --cpu=2`
  rather than Cloud Run's 512Mi/1vCPU default: the model doesn't fit in the default. It's also
  why the *first* call after the service scales to zero is slow (~60-80s measured — the ~2.2GB
  model re-downloads from the HF Hub every cold start, since Cloud Run's ephemeral storage
  doesn't persist between instances); warm calls are ~1-2s. `embedding.py` authenticates that
  download with the HF token to avoid anonymous-read rate-limiting, same as the data shards below.

`/attest` returns an empty `results: []` — no relational/attestation data has been ingested yet
(spec §3a; only §3c aligned pairs exist so far). `/lookup` and `/search` both have real data:
704 aligned `(enm, eng)` sentence pairs from *Sir Gawayne and the Green Knight* (1400 ↔ 1999).
All three are still soft-fail by design for whatever *isn't* covered yet — a language pair or
year range with no shard returns `[]` rather than erroring (spec §11); `/search` specifically
returns `[]` for a shard that has a `pairs/` Parquet file but no matching `vectors/pairs/` FAISS
file, without affecting other shards.

## Local development

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in HF_TOKEN (HF_DATASET_REPO_ID already defaults to the real repo)
uvicorn app.main:app --reload --port 8000
```

## Building the index (before there's anything to serve)

`search_api` only serves what's already been pushed to the HF Dataset repo — it never builds an
index itself. That happens on the ingestion side:

```
python scripts/align_corpus.py data/raw/<work>       # -> data/processed/<work>/*.bitext.jsonl
python scripts/build_index.py                        # -> data/index/{pairs,vectors/pairs}/...
```

`build_index.py` writes locally (`data/index/` by default); pushing those shards to the
configured `HF_DATASET_REPO_ID` is a separate step (spec §10) not automated yet.
