"""Embeds an incoming /search query so it can be compared against the FAISS
shards ye_olde.ingest.index built (spec §10).

Deliberately a standalone copy, not an import of ye_olde.ingest.embed: this
service has its own dependency tree by design (see config.py's docstring),
so the handful of lines that matter (load the model, apply the e5 "query: "
prefix, L2-normalize) are duplicated here rather than adding the whole
ye_olde package as a search_api dependency for one function.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Same convention as ye_olde.ingest.embed: intfloat's e5 model family wants
# "query: "/"passage: " prefixes, and "query: " on both sides is what the
# model card recommends for symmetric similarity search like this. Harmless
# for non-e5 encoders, so applied unconditionally.
_E5_PREFIX = "query: "


@lru_cache(maxsize=2)
def _get_model(model_name: str, hf_token: str) -> SentenceTransformer:
    from sentence_transformers import SentenceTransformer

    # Every cold start (scale-to-zero service) re-downloads this model from
    # the Hugging Face Hub — an anonymous request, same as the Parquet/FAISS
    # shard downloads in loader.py, and subject to the same rate-limiting
    # (spec §10, discovered live deploying this). Passing the token here
    # closes that gap for the model weights too, reusing config.py's
    # existing hf_token setting rather than adding a second credential.
    return SentenceTransformer(model_name, token=hf_token or None)


def embed_query(text: str, *, model_name: str, hf_token: str = "") -> np.ndarray:
    """Returns a single L2-normalized embedding vector for `text`, as
    float32 (FAISS's expected dtype)."""
    model = _get_model(model_name, hf_token)
    vector = model.encode([_E5_PREFIX + text], normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vector, dtype="float32")
