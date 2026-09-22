"""Search API client + temporal reranking — spec §2, §4, §10.

This is an HTTP client (httpx) against ye_olde.config.Settings.search_api_url
— it never embeds a vector store or holds the corpus locally; the actual
index lives behind search_api/ (spec §10). Retrieval is keyed on sense_id,
not literal lemma. Temporal reranking uses
score = similarity * exp(-lambda * year_gap) (lambda tuned empirically, spec §8).
Not yet implemented — the search_api service itself has no real data yet
(spec §11), so there's nothing to call against.
"""
