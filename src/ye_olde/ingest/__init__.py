"""Corpus ingestion into the queryable index — spec §3a, §5, §6.

`align.py` (+ `clean.py`, `extract.py`, `corpus_files.py`) implements §6's
data acquisition & alignment pipeline: given a `data/raw/<work>/` folder of
2+ parallel-translation files, it produces §3c aligned example-pair bitext
files, one per language pair. See `scripts/align_corpus.py` for the CLI.

Turning validated community contributions (spec §3b format) into the §3a
indexed schema (embedded + stored in the vector store, plus structured
fields for the fallback chain's exact-match gating) is not yet implemented.
"""
