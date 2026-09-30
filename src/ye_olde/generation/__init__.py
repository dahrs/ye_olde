"""Grounded LLM generation + self-check — spec §2.

Generates the output sentence constrained to retrieved attested forms
(`GroundedForm`, assembled by `translate()` from `classify`/`fallback`'s
output), then a second, independent LLM call self-checks the result against
those same forms before `translate()` returns it.

**Self-check is always a second `call_llm_json` call — never a tool the
generation call might invoke mid-turn, and this is a deliberate decision,
not the simplest option left unmade.** The driving requirement: this has to
work identically whether `LITELLM_MODEL` resolves to a hosted API or a
local inference server (both are first-class, equally-supported backends
for this project — see the root README's "Local LLM" section) *without*
either `generate` or `self_check` needing to know or care which one is in
use. `call_llm_json` (`common.llm_client`) already provides exactly that —
one uniform call shape regardless of backend. A tool-calling-based design
would break that uniformity: not every backend supports tool calling the
same way (or reliably at all — locally-served, template-dependent models in
particular), so using it here would mean either restricting self-check to
backends that support it well, or adding per-backend branching this module
would otherwise never need. Beyond that portability argument, there's a
reliability one too: "mandatory" is a much stronger guarantee as plain
Python control flow than as a model's compliance with a forced-tool-use
instruction — the same reasoning that already keeps `retrieval/`
deterministic (spec §2), just consistently extended to the other
grounding-critical step. See `docs/diachronic-translation-pipeline-plan.md`
§2/§9 for the full writeup.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ..common.llm_client import call_llm_json
from ..prompt import load_prompt

_GENERATE_SYSTEM_PROMPT = load_prompt("generation", "generate_system")
_SELF_CHECK_SYSTEM_PROMPT = load_prompt("generation", "self_check_system")


class GroundedForm(BaseModel):
    """One resolved span `generate`/`self_check` must ground their output
    in — the context bundle spec §2 describes, trimmed to what these two
    calls actually need. `status` is deliberately a plain `str`, not
    `fallback.FallbackResult`'s narrower `Literal` — `translate()` also
    produces a `"name-passthrough"` status for classify-routed names that
    never go through `fallback` at all.
    """

    span: str
    form: str
    status: Literal["attested", "anachronism-passthrough", "name-passthrough"]
    # Per-token provenance (e.g. fallback.FallbackResult.note) -- carried
    # through to the final Translation.annotations rather than discarded,
    # so a caller can see *why* a span got this status/form. Not sent to
    # the LLM (_format_forms below doesn't include it) -- purely for the
    # human-facing Annotation api.py builds afterward.
    note: str = ""


def _format_forms(forms: list[GroundedForm]) -> str:
    return "\n".join(f'- {{"span": {f.span!r}, "form": {f.form!r}, "status": {f.status!r}}}' for f in forms)


def generate(sentence: str, lang_b: str, year_b: int, forms: list[GroundedForm], *, model: str | None = None) -> str:
    """One LLM call: composes a sentence in `lang_b`/`year_b`, constrained
    to `forms`. Raises `ValueError` if the reply isn't the expected shape
    — same "don't guess past a malformed reply" stance as
    `ingest.align.align_block`.
    """
    prompt = (
        f"SOURCE SENTENCE (meaning/structure only, do not copy verbatim): {sentence}\n"
        f"TARGET: {lang_b}, year {year_b}\n"
        f"GROUNDED FORMS:\n{_format_forms(forms)}"
    )
    raw = call_llm_json(prompt, system=_GENERATE_SYSTEM_PROMPT, model=model)
    sentence_out = raw.get("sentence") if isinstance(raw, dict) else None
    if not isinstance(sentence_out, str) or not sentence_out.strip():
        raise ValueError(f"expected a JSON object with a non-empty 'sentence' string, got {raw!r}")
    return sentence_out


class SelfCheckResult(BaseModel):
    """`self_check`'s return value — a public function's return value, so a
    `BaseModel` per CLAUDE.md's Pydantic guidance.
    """

    approved: bool
    sentence: str
    note: str


def self_check(
    draft_sentence: str, lang_b: str, year_b: int, forms: list[GroundedForm], *, model: str | None = None
) -> SelfCheckResult:
    """Second, independent LLM call verifying `draft_sentence` actually uses
    every `forms[i].form` — see module docstring for why this is always a
    separate call, unconditionally made, never a tool `generate` might
    invoke (or skip) on its own.
    """
    prompt = (
        f"DRAFT SENTENCE: {draft_sentence}\nTARGET: {lang_b}, year {year_b}\nGROUNDED FORMS:\n{_format_forms(forms)}"
    )
    raw = call_llm_json(prompt, system=_SELF_CHECK_SYSTEM_PROMPT, model=model)
    if not isinstance(raw, dict):
        raise ValueError(f"expected a JSON object, got {raw!r}")
    sentence = str(raw.get("sentence") or "").strip() or draft_sentence
    return SelfCheckResult(
        approved=_coerce_approved(raw.get("approved")), sentence=sentence, note=str(raw.get("note") or "")
    )


def _coerce_approved(value: object) -> bool:
    """`bool(value)` is wrong here: a JSON-mode reply that quotes its
    boolean (`"approved": "false"`) is a real, observed quirk on some
    backends, and `bool("false")` is `True` (any non-empty string is
    truthy) — silently treating a rejected check as approved, exactly the
    failure self-check exists to catch. Only a real `True`, or the string
    `"true"` (case-insensitive), count as approved; everything else
    (`False`, `"false"`, `None`, missing) does not.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


__all__ = ["GroundedForm", "SelfCheckResult", "generate", "self_check"]
