---
name: llm-scoring-rules
description: Use whenever writing, reviewing, or extending a prompt that asks an LLM to self-report a numeric score of any kind — confidence, quality, relevance, similarity, severity, etc. Applies before adding a new scored field to an existing prompt, and when auditing an existing prompt that already asks for one. Not for scores computed deterministically in code (embedding similarity, string overlap) — only for a number the model itself estimates and reports back.
---

# LLM self-reported scoring: definition, calibration, no few-shot

A prompt that asks an LLM for a bare number ("confidence, 0.0-1.0") without
saying what that number measures or what separates a low one from a high
one gets an uncalibrated, likely overconfident value clustered near 0.9 —
worse than not asking at all, since it looks like signal but isn't one.

## Required, every time a prompt asks for a self-reported score

1. **Definition** — state precisely what the number is a probability/degree
   *of*. Not "confidence" alone — confidence in what, judged against what
   standard? ("your own probability that a trained linguist annotating this
   by hand would write the same tag" is a definition; "your confidence in
   this tag" is not.)
2. **Calibration anchors** — give the model bands (e.g. 3-4) describing what
   qualifies for a high vs. a low score, tied to concrete situations it can
   recognize in the task at hand, not just a number line. A model has no
   innate sense of where 0.6 sits without being told what kind of case lands
   there.
3. **An explicit anti-default warning**, when there's room for the model to
   lazily pick one number for everything (e.g. "always 0.9") — name that
   failure mode directly so the model doesn't default into it.

## Deliberately omit: few-shot examples

For a *scoring* instruction specifically (as opposed to a task instruction
in general, where few-shot can help), skip worked examples by default.
Reasoning: the space of things being scored here is usually far wider than
2-3 examples can represent, and a model tends to anchor its answers toward
the specific examples shown rather than learning the general calibration
rule — the opposite of the goal. Add examples only if asked, or if plain
calibration-band wording is demonstrated (via eval or spot-check) to still
produce clustering.

## Where the wording goes

Don't just fix one prompt in isolation — check whether the same scoring
instruction is needed in more than one system prompt (a corpus-time and a
query-time prompt tagging the same schema, e.g.) before deciding where the
text lives:

- **Used in exactly one system prompt** — write it inline, in that
  prompt's own YAML entry, next to the field it scores.
- **Used in two or more system prompts within the same project** — don't
  retype it per prompt. In a project with a prompts-as-YAML-data convention
  (see this repo's `src/ye_olde/prompt/`), that means a shared location:
  either a repeated block within one namespace file if that project has
  already established that pattern for other repeated instructions (check
  first — e.g. `ingest.yaml`'s header comment on why it duplicates its own
  annotation-tags block rather than using a YAML alias, since aliases don't
  resolve inside a block scalar), or a dedicated shared-fragments file
  (`prompt/common.yaml` in this repo) loaded by each module that needs it
  and concatenated onto its own system prompt in Python — the wording still
  lives entirely in the YAML data, only the *assembly* of which fragments a
  given prompt uses is code. Never hardcode the calibration text as a
  Python string constant; that violates "prompts are data, not code."

## Checklist before shipping a new/edited scored field

- [ ] The prompt states what the number measures, not just its range.
- [ ] The prompt gives calibration bands tied to recognizable cases.
- [ ] No few-shot examples added for the scoring instruction itself (unless
      explicitly requested).
- [ ] Checked for reuse across other prompts in the project before deciding
      where the wording lives.
- [ ] Code that consumes the score still validates/clamps it independently
      (a prompt's wording is guidance, never a guarantee — the model can
      still return an out-of-range value, a string, or nothing at all; see
      this repo's `common/ud_tags.parse_confidence` for the pattern: clamp
      to range, `None` for anything unusable, never silently coerce a
      missing value to a fake "0").
