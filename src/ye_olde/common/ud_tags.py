"""Closed tag vocabularies shared between corpus-time annotation
(`ingest.align`), query-time annotation (`classify`), and tool-based
annotation (`ingest.annotators`, Stanza) — spec §3d.

One shared place specifically because **every** path must enforce the
*same* closed set, regardless of which of the three produced the raw tag.
`classify`'s whole purpose (per its own docstring) is that a query-time tag
matches an identically-tagged indexed token; that only holds if the LLM
path, the corpus-alignment LLM path, and any tool-based path all reject
the same out-of-vocabulary values through the same functions, rather than
each trusting its own source verbatim (or, before this module covered
UPOS/DEPREL too, not validating them at all — only NER was closed-set
enforced anywhere in code; a plausible-looking but nonstandard UPOS like
`"Noun"` or a hallucinated one could reach a corpus shard unchecked). It is
intentionally *not* enough that Stanza's own output happens to already be
UD-compliant — running it through the identical normalizer here closes the
loop completely rather than trusting that by convention.
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


# Universal Dependencies' own 17-tag UPOS inventory (spec §3d) — the exact
# list already spelled out in prompt/ingest.yaml and prompt/classify.yaml.
VALID_UPOS = frozenset(
    {
        "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
        "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
    }
)  # fmt: skip


def normalize_upos(raw: str) -> str:
    """An LLM- or tool-proposed UPOS tag, normalized to `VALID_UPOS` —
    anything outside the closed set becomes `"X"` (UD's own "other"
    catch-all — already the meaning `classify._resolve_tags` uses for an
    unverifiable tag, so this reuses rather than invents a second
    "couldn't classify this" value).
    """
    tag = raw.strip().upper()
    return tag if tag in VALID_UPOS else "X"


# Universal Dependencies' core relation set (spec §3d) — the exact list
# already spelled out in prompt/ingest.yaml and prompt/classify.yaml.
VALID_DEPREL_BASE = frozenset(
    {
        "nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "obl", "vocative",
        "expl", "dislocated", "advcl", "advmod", "discourse", "aux", "cop",
        "mark", "nmod", "appos", "nummod", "acl", "amod", "det", "clf",
        "case", "conj", "cc", "fixed", "flat", "compound", "list",
        "parataxis", "orphan", "goeswith", "reparandum", "punct", "root",
        "dep",
    }
)  # fmt: skip


def normalize_deprel(raw: str) -> str:
    """A DEPREL tag, normalized against `VALID_DEPREL_BASE`. UD allows a
    language-specific `:subtype` suffix on any core relation (e.g.
    `nsubj:pass`, `aux:pass` — both observed live from Stanza's own
    PROIEL-trained Latin model, and both already used in this codebase's
    own test fixtures) — only the base relation *before* the colon is
    checked against the closed set, so that legitimate variation is kept
    rather than rejected, while a genuinely invalid/hallucinated base
    relation still falls back to `"dep"` (UD's own generic "some
    dependency" catch-all).
    """
    tag = raw.strip().lower()
    base = tag.split(":", 1)[0]
    return tag if base in VALID_DEPREL_BASE else "dep"


# Fixed confidence assigned to a tag from a registered tool (Stanza,
# `ingest.annotators`) rather than an LLM's self-reported number — a real
# treebank-trained model's deterministic prediction isn't a "guess" the way
# an LLM's free-text claim is, and its tags are already unconditionally
# preferred over the LLM's own proposal (`ingest.align._annotate_side`).
# This is a fixed sentinel representing that established trust, not a
# per-tag probability Stanza's own simple Word API actually exposes —
# documented here so it's never mistaken for a genuine model-internal
# confidence score.
TOOL_CONFIDENCE = 1.0


def parse_confidence(raw: object) -> float | None:
    """Clamps an LLM-proposed confidence value to `[0.0, 1.0]`, or `None`
    if `raw` isn't a usable number (missing, non-numeric, NaN). `None`
    (not `0.0`) for a missing/bad value deliberately means "unknown," not
    "the model reported zero confidence" — those are different claims, and
    conflating them would misrepresent a field the model never filled in
    as an explicit vote of no confidence.
    """
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN is the only float that doesn't equal itself
        return None
    return max(0.0, min(1.0, value))
