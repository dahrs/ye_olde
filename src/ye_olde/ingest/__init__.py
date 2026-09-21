"""Corpus ingestion into the queryable index — spec §3a, §5, §6.

Turns raw corpus text (bootstrapping corpora) or validated community
contributions (spec §3b format) into the indexed schema in spec §3a: embedded
+ stored in the vector store, plus structured fields in a relational table
for the fallback chain's exact-match gating. Not yet implemented.
"""
