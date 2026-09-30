"""Public entrypoint — spec §1 (docs/diachronic-translation-pipeline-plan.md).

translate(sentence, lang_a, year_a, lang_b, year_b) -> Translation

Wires `classify` -> `fallback` -> `generation` into the fixed call sequence
spec §2 describes. Domain code, not a boundary layer (CLAUDE.md's "Error
handling": a boundary is a FastAPI route or a CLI script's `main()`) — this
function lets `classify`/`fallback`/`generation`'s exceptions
(`ValueError`, `common.errors.LLMError` and subclasses, `httpx` errors from
`retrieval`) propagate uncaught rather than catching them itself; whatever
eventually calls this (a future API route, a CLI demo script) is the actual
boundary and is responsible for catching/logging.

**What's real today vs. what's an honest limitation**, so a caller isn't
surprised: only `fallback`'s native rung has real data behind it (see that
module's docstring) — a content word with no strong attested match gets
`"anachronism-passthrough"`, not a genuine loan/constructed-compound
resolution. Only content-word tokens (`classify.TokenTag.upos` in
`NOUN`/`PROPN`/`VERB`/`ADJ`/`ADV`) go through `fallback` at all — function
words are left for `generation` to render using its own grammatical
competence for the target period, the same "prefer content words over
function words" scoping this codebase's alignment prompts already use
(`prompt/ingest.yaml`), not a per-token substitution cipher. A name
(`classify.TokenTag.is_name`) bypasses `fallback` entirely and passes
through unchanged (spec §2/§4: no period name-form registry exists to look
one up in yet).

No pivot representation beyond `list[generation.GroundedForm]` exists yet —
deliberately: the spec's original "sense units, names, source structure"
pivot node was intentionally left undesigned until real classify/fallback
output existed to design it against (see project history). `GroundedForm`
is that first real look at what the pivot needs to carry; a richer
representation (phrase-level spans, not one token at a time; explicit
source structure) is a likely next iteration once this shape is proven
against real translate() calls, not a placeholder pretending to be final.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from . import classify, fallback, generation

_CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


class Annotation(BaseModel):
    span: str
    status: Literal["attested", "loan", "constructed", "anachronism-passthrough", "name-passthrough"]
    note: str


class Translation(BaseModel):
    sentence: str
    annotations: list[Annotation]


def _resolve_forms(
    tags: list[classify.TokenTag], lang_a: str, year_a: int, lang_b: str, year_b: int, *, model: str | None
) -> list[generation.GroundedForm]:
    forms: list[generation.GroundedForm] = []
    for tag in tags:
        if tag.is_name:
            forms.append(
                generation.GroundedForm(
                    span=tag.text,
                    form=tag.text,
                    status="name-passthrough",
                    note="proper name, no period name-form registry available — passed through unchanged",
                )
            )
            continue
        if tag.upos not in _CONTENT_UPOS:
            continue  # function word: generation renders it directly, not looked up per-token
        term = tag.lemma or tag.text
        result = fallback.resolve(term, lang_a, year_a, lang_b, year_b)
        forms.append(generation.GroundedForm(span=tag.text, form=result.form, status=result.status, note=result.note))
    return forms


def _annotation_note(form: generation.GroundedForm, checked: generation.SelfCheckResult) -> str:
    """`form.note` (per-token provenance — why *this* span got its
    status/form, from `fallback`/the name-passthrough branch) is always
    kept, never discarded or blanked just because the sentence overall
    passed self-check. `checked.note` is sentence-level, so it's appended
    (not substituted) only when self-check actually flagged something —
    an earlier version threw away `form.note` entirely and stamped the
    same single `checked.note` onto every annotation regardless of
    relevance.
    """
    if checked.approved or not checked.note:
        return form.note
    if not form.note:
        return f"self-check: {checked.note}"
    return f"{form.note} — self-check: {checked.note}"


def translate(
    sentence: str,
    lang_a: str,
    year_a: int,
    lang_b: str,
    year_b: int,
    *,
    model: str | None = None,
) -> Translation:
    """`(sentence, lang_a, year_a) -> (sentence, lang_b, year_b)` — spec §1.
    `year_a` feeds `classify`'s per-`(lang, year)` function-word registry
    (`common.function_words`) — a growing cache of closed-class surface
    forms this exact language/period has confidently seen before, consulted
    for consistency and grown with every call — and, together with
    `lang_a`, is passed to `fallback.resolve` so it can query `/lookup`
    (spec §3c: source term + period -> target period's attested form).
    """
    tags = classify.classify(sentence, lang_a, year_a, model=model)
    forms = _resolve_forms(tags, lang_a, year_a, lang_b, year_b, model=model)

    draft = generation.generate(sentence, lang_b, year_b, forms, model=model)
    checked = generation.self_check(draft, lang_b, year_b, forms, model=model)

    annotations = [Annotation(span=f.span, status=f.status, note=_annotation_note(f, checked)) for f in forms]
    return Translation(sentence=checked.sentence, annotations=annotations)
