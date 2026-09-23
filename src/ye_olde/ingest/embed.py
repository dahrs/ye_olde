"""Sentence-embedding wrapper for the algorithmic half of the hybrid
aligner (sentence_align.py) — spec §4's "pretrained, used as-is" embedding
component, used here for bitext mining rather than retrieval serving.

This runs at corpus-preparation time on whatever machine builds the
bitext (not the Pi 5 deployment target `config.py` is tuned for), so
pulling in `sentence-transformers` here doesn't affect the runtime
footprint of the deployed pipeline.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..config import get_settings

# intfloat's e5 model family is trained with a "query: "/"passage: " input
# convention; for symmetric similarity tasks (STS, bitext mining — this is
# not asymmetric retrieval) the model card recommends "query: " on both
# sides. Harmless for non-e5 encoders, so it's applied unconditionally
# rather than gated on the model name.
_E5_PREFIX = "query: "


@lru_cache(maxsize=2)
def _get_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_units(units: list[str], *, model_name: str | None = None) -> np.ndarray:
    """Encodes `units` into L2-normalized embeddings — normalized so a dot
    product between two rows is directly a cosine similarity — using the
    project's configured embedding model (`EMBEDDING_MODEL` in `.env`,
    same setting `ye_olde.retrieval` will eventually use to query the
    Search API's vector index).
    """
    resolved = model_name or get_settings().embedding_model
    model = _get_model(resolved)
    prefixed = [_E5_PREFIX + u for u in units]
    return np.asarray(model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False))
