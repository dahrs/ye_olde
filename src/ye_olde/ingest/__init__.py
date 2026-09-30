"""Corpus ingestion into the queryable index — spec §3a, §5, §6.

`align.py` (+ `clean.py`, `extract.py`, `corpus_files.py`) implements §6's
data acquisition & alignment pipeline: given a `data/raw/<work>/` folder of
2+ parallel-translation files, it produces §3c aligned example-pair bitext
files, one per language pair. See `scripts/align_corpus.py` for the CLI.

`scorer.py` holds the similarity/scoring metrics the cleaning-and-alignment
stage measures matches by (embedding cosine similarity, character-trigram
Dice similarity, and the shared margin-acceptance check) — pulled out of
`sentence_align.py`/`index.py` into one dependency-light module both import,
instead of each keeping its own copy. See `scorer.py`'s module docstring for
the full "why", including what deliberately stayed out of it.

`index.py` implements the indexing job (spec §10) for that same §3c bitext
output: Parquet pairs shards + a FAISS semantic-search index per language
pair, in the layout `search_api/` reads. See `scripts/build_index.py`.

`ye_olde.common.llm_client` (moved out of this package — `classify/` and
`generation/` need the same LLM-calling boundary `clean.py`/`align.py` use,
not just ingestion) is the shared entrypoint (hosted API or local inference
server, via litellm). `ye_olde.common.claude_cli_client` is a third,
independent option alongside it — reuses a Claude Code subscription login
instead of an API key — importable on its own by any script, not just
through `llm_client.py`'s dispatch. See that module's docstring for what it
is and why.

Turning validated community contributions (spec §3b format) into the §3a
indexed schema (embedded + stored in the vector store, plus structured
fields for the fallback chain's exact-match gating) is not yet implemented
— there's no contribution data yet either (`contributions/` is empty).
"""
