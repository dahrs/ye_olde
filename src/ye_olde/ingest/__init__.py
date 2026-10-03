"""Corpus ingestion into the queryable index — spec §3a, §5, §6.

`acquire.py` implements §6's data *acquisition* step, previously entirely
manual: given a URL (a Project Gutenberg `.txt` file, a PDF, or an HTML
"read online" page) from the worklist in `data/sources.yaml`, it fetches
the content, infers the five `data/raw/` naming fields from an LLM reading
it, and saves it with a filename `corpus_files.build_corpus_filename`
guarantees the rest of this pipeline can read back. See
`scripts/acquire_corpus.py` for the CLI.

`split_mixed.py` handles the case `acquire.py` doesn't: a single acquired
file that bundles several distinct editions of one work together (e.g. a
Gutenberg compilation containing the original-language text plus one or
more later translations in one `.txt`). Given a small human-written
manifest describing each bundled edition, it asks an LLM to assign each
paragraph-chunk to the edition it belongs to and writes out one real
`data/raw/` file per edition. See `scripts/split_mixed_source.py`.

`align.py` (+ `clean.py`, `extract.py`, `corpus_files.py`) implements §6's
alignment pipeline: given a `data/raw/<work>/` folder of 2+ parallel-
translation files (built up by `acquire.py` above, one file at a time), it
produces §3c aligned example-pair bitext files, one per language pair. See
`scripts/align_corpus.py` for the CLI.

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
