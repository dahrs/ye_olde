"""Closed tag vocabularies shared between corpus-time annotation
(`ingest.align`) and query-time annotation (`classify`) — spec §3d.

One shared place specifically because both paths must enforce the *same*
closed set: `classify`'s whole purpose (per its own docstring) is that a
query-time tag matches an identically-tagged indexed token, which only
holds if both sides actually reject the same out-of-vocabulary values
rather than one validating and the other trusting the LLM verbatim.
"""

from __future__ import annotations

# CoNLL-2003 NER tag set (spec §3d/§4) — reused verbatim rather than
# invented, same "use an existing standard" reasoning as the UPOS/deprel
# vocabularies in prompt/ingest.yaml and prompt/classify.yaml.
VALID_NER = frozenset({"PER", "LOC", "ORG", "MISC", "O"})


def normalize_ner(raw: str) -> str:
    """An LLM-proposed NER tag, normalized to `VALID_NER` — anything not
    in the closed set (a variant spelling, a hallucinated tag, empty)
    becomes `"O"` rather than being persisted/matched verbatim. `"O"` is
    the correct default, not just a fallback: it means "not a named
    entity," which is also the right answer for "the model's tag couldn't
    be trusted."
    """
    tag = raw.strip().upper()
    return tag if tag in VALID_NER else "O"
