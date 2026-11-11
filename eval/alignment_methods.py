"""Reusable scoring methods for alignment_benchmark_eval.py — kept separate
from that script so the scoring functions themselves (not the CLI/reporting
machinery around them) can be imported directly, including by something
other than this one eval script later.

Every function here takes one (source, target) sentence pair plus their
ISO 639 language codes and returns a score in `[0.0, 1.0]`, or `None` when
a method genuinely couldn't produce one — the one common interface
`alignment_benchmark_eval.py` iterates over regardless of which method
produced it, so adding a new method later means adding one more function
with this same shape, not touching the harness around it.
"""

from __future__ import annotations

from ye_olde.common.llm_client import call_llm_json
from ye_olde.common.ud_tags import parse_confidence
from ye_olde.ingest.embed import embed_units
from ye_olde.ingest.scorer import cosine_similarities
from ye_olde.prompt import load_prompt

_PAIR_SCORE_SYSTEM_PROMPT = (
    load_prompt("eval", "pair_alignment_score_system")
    + "\n\n"
    + load_prompt("common", "translation_correspondence_calibration")
)


def embedding_score(source: str, target: str, lang_a: str, lang_b: str, *, model_name: str) -> float | None:
    """Cosine similarity between `source` and `target`'s embeddings under
    `model_name` — the same mechanism `ye_olde.ingest.sentence_align`
    scores real candidate pairs with (`scorer.cosine_similarities`),
    applied here to one already-chosen pair instead of a whole banded
    search. `lang_a`/`lang_b` aren't used by this method (an embedding
    model gets no language hint at all) — accepted anyway so every method
    in this module shares one call signature, regardless of which inputs
    it actually needs.
    """
    vectors = embed_units([source, target], model_name=model_name)
    return float(cosine_similarities(vectors[0], vectors[1:2])[0])


def llm_score(source: str, target: str, lang_a: str, lang_b: str, *, model: str | None = None) -> float | None:
    """Asks the configured LLM directly whether `target` is a genuine
    translation of `source` — the "plain LLM call" baseline, with no
    embedding model and no alignment algorithm involved at all. `None`
    (not `0.0`) when the reply didn't include a usable score — see
    `parse_confidence`'s own docstring for why that distinction matters:
    a model that didn't answer is not the same claim as one voting zero.
    """
    prompt = f"SOURCE ({lang_a}): {source}\nTARGET ({lang_b}): {target}"
    reply = call_llm_json(prompt, system=_PAIR_SCORE_SYSTEM_PROMPT, model=model)
    if not isinstance(reply, dict):
        return None
    return parse_confidence(reply.get("score"))
