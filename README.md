# ye_olde
A language translator using highly modern tools where the focus is on old and ancient languages.

Translates `(sentence, lang_code, year) → (sentence, lang_code, year)` — between any two points in a language's own history — using retrieval over a corpus partitioned by `(lang_code, year)`, not fine-tuning. Full design: [docs/diachronic-translation-pipeline-plan.md](docs/diachronic-translation-pipeline-plan.md).

## Status

Scaffolding only — no pipeline logic implemented yet. See the spec's §9 for the first scoped build task.

## Project layout

```
src/ye_olde/
├── api.py          # translate() — the public entrypoint (spec §1)
├── config.py       # settings: embedding model, generation LLM, vector store backend
├── resolver/       # (lang_code, year) -> weighted corpus partitions (§2, §3a)
├── classify/       # name vs. common-word vs. anachronism token classification (§2)
├── retrieval/      # embedding + vector store + temporal rerank (§2, §4)
├── fallback/       # native -> loan -> constructed -> temporal-loan chain (§2, §3a)
├── generation/      # grounded LLM generation + self-check (§2)
├── ingest/         # corpus ingestion into the queryable index (§3a, §5, §6)
└── schema/         # JSON Schema for community contributions (§3b)

contributions/      # community-submitted JSON entries (§3b), validated by CI on PR
data/               # gitignored corpora/index artifacts (raw/processed/index)
scripts/            # CLI entrypoints (ingestion runner, translate demo)
tests/              # unit/ and integration/
```

## Setup

```
uv sync
cp .env.example .env   # fill in EMBEDDING_MODEL / LITELLM_MODEL / VECTOR_STORE
```
