---
name: add-llm-prompt
description: Use before writing or editing any text sent to an LLM as a system prompt or prompt template in ye_olde.* (ingest, classify, generation, fallback, ...) — adding a new LLM call, changing existing prompt wording, or adding a scored/structured field to one. Not for search_api, which has no LLM calls and no prompt/ package. Pair with llm-scoring-rules when the prompt asks the model to self-report a numeric score.
---

# Adding or editing an LLM prompt

**Prompts are data, not code.** The literal wording always lives in
`src/ye_olde/prompt/<namespace>.yaml` — one file per top-level package that
calls an LLM (`ingest.yaml`, `classify.yaml`, `generation.yaml` today) —
never as a Python string literal in the calling module. This keeps wording
reviewable/diffable without touching control flow, and keeps every prompt
in the project discoverable in one place.

## How to wire it up

1. Add the prompt text under a new key in the right namespace file (or a
   new namespace file if the package doesn't have one yet).
2. Load it with `ye_olde.prompt.load_prompt(namespace, key) -> str`.
3. Bind the loaded result to a module-level constant for use within that
   module (e.g. `_SYSTEM_PROMPT` in `ingest/clean.py`) — only the *text*
   moves to YAML, not the reference to it. Don't call `load_prompt` inline
   at every call site.
4. `load_prompt` raises `PromptNotFoundError` (not a silent empty string)
   if the namespace file or key doesn't exist — never swallow this; an LLM
   call should never run with a missing prompt it didn't notice was missing.

## Reusing wording across more than one prompt

A real YAML alias only resolves within a single block scalar — not across
separate block scalars or across files — so it can't be used to share text
between two system prompts. Instead:

- **Used in exactly one system prompt** — write it inline in that prompt's
  own YAML entry.
- **Used in two or more system prompts** — put it in
  `src/ye_olde/prompt/common.yaml` as its own key, and have each module
  that needs it call `load_prompt("common", key)` and concatenate the
  result onto its own system prompt in Python (see `align.py`'s or
  `classify/__init__.py`'s `_*_SYSTEM_PROMPT` construction for the pattern).
  The wording still lives entirely in YAML — only the decision of which
  fragments a given prompt assembles is code.

## Scope

`search_api` has no LLM calls and no `prompt/` package of its own — this
doesn't apply there.

If the prompt asks the model to self-report any kind of numeric score
(confidence, quality, relevance, ...), also load the `llm-scoring-rules`
skill before writing that part of the wording.
