"""Embedding, vector store, and temporal reranking — spec §2, §4.

Retrieval is keyed on sense_id, not literal lemma. Temporal reranking uses
score = similarity * exp(-lambda * year_gap) (lambda tuned empirically, spec §8).
Backend (embedding model, vector store) is read from ye_olde.config.Settings —
not yet implemented.
"""
