"""Corpus ingestion into the queryable index — spec §3a, §5, §6.

`align.py` (+ `clean.py`, `extract.py`, `corpus_files.py`) implements §6's
data acquisition & alignment pipeline: given a `data/raw/<work>/` folder of
2+ parallel-translation files, it produces §3c aligned example-pair bitext
files, one per language pair. See `scripts/align_corpus.py` for the CLI.

`index.py` implements the indexing job (spec §10) for that same §3c bitext
output: Parquet pairs shards + a FAISS semantic-search index per language
pair, in the layout `search_api/` reads. See `scripts/build_index.py`.

Turning validated community contributions (spec §3b format) into the §3a
indexed schema (embedded + stored in the vector store, plus structured
fields for the fallback chain's exact-match gating) is not yet implemented
— there's no contribution data yet either (`contributions/` is empty).
"""
