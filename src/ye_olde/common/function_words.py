"""Persisted, self-populating registry of closed-class function words per
`(lang, year)` — one JSON file per pair, growing automatically as
POS-tagged tokens are observed during query-time classification
(`classify`) — not yet from ingestion (`ingest.align`'s `_build_link` never
calls this module; it only tags a link's own head word, not every token,
see that function's docstring) — rather than requiring a hand-maintained
stopword list per language/period this project would otherwise need one of
for every language it ever adds (spec §3d's whole reason for avoiding
per-language hardcoded resources — DMNES's given-names-only coverage was
exactly this kind of trap).

**Not the same thing as the stopword filtering `ingest.sense` does when
scoring WordNet-gloss overlap** — that filters common *English* words out
of a comparison (gloss text is always English by design, spec §3d,
regardless of source language), a fixed, small, deliberately-not-learned
list for a fixed-language purpose. This registry is the opposite: it
tracks which *surface forms of the source/target corpus language itself*
are closed-class, growing per language and period as real data is seen —
a genuinely different concept that happens to have come up in the same
conversation.

Keyed by surface form (not lemma) deliberately: the point is to let a
caller skip *re-tagging* a token it has already confidently seen before,
and that decision has to happen before lemmatization exists for this
occurrence — a lemma is exactly what tagging would produce, so it can't
also be the cache key. A period's spelling is reasonably stable within one
corpus, so this is still a useful cache even though two spellings of the
same underlying word (`þe`/`the`) are tracked as separate entries.
"""

from __future__ import annotations

import json
from pathlib import Path

# Standard closed-class UPOS tags (spec §3d's UD subset) — a token tagged as
# any of these is a function word by definition; open-class tags (NOUN,
# PROPN, VERB, ADJ, ADV) are never recorded here, since those are exactly
# the words sense-lookup/retrieval cares about and must keep re-examining.
_CLOSED_CLASS_UPOS = {"DET", "ADP", "AUX", "CCONJ", "SCONJ", "PART", "PRON", "INTJ"}

DEFAULT_ROOT = Path("data/function_words")


def _resolve_root(root: Path | None) -> Path:
    # `root` defaults to `None` and is resolved to `DEFAULT_ROOT` here,
    # inside the function body, rather than as `root: Path = DEFAULT_ROOT`
    # in each signature below -- a default *argument value* is bound once,
    # at function-definition (import) time, so a caller (or a test)
    # monkeypatching `function_words.DEFAULT_ROOT` afterward would silently
    # have no effect on a call that omits `root=`. Reading it here, at call
    # time, is what actually makes that monkeypatchable.
    return root if root is not None else DEFAULT_ROOT


def _path(lang: str, year: int, *, root: Path | None) -> Path:
    return _resolve_root(root) / lang / f"{year}.json"


def load(lang: str, year: int, *, root: Path | None = None) -> dict[str, str]:
    """Returns `{surface_form_lowercased: upos}` recorded so far for
    `(lang, year)` — `{}` if this pair has never been seen before.
    """
    path = _path(lang, year, root=root)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    forms = data.get("forms")
    return {str(k): str(v) for k, v in forms.items()} if isinstance(forms, dict) else {}


def record(lang: str, year: int, surface_form: str, upos: str, *, root: Path | None = None) -> None:
    """Adds `surface_form -> upos` to `(lang, year)`'s registry if `upos`
    is closed-class — a no-op otherwise (this registry only ever grows
    with function words; an open-class word is never cached out of the
    "needs re-tagging" path this exists to shortcut). Idempotent: already
    knowing the same form/upos is a no-op, not a rewritten file.
    """
    upos = upos.upper()
    key = surface_form.lower().strip()
    if not key or upos not in _CLOSED_CLASS_UPOS:
        return
    existing = load(lang, year, root=root)
    if existing.get(key) == upos:
        return
    existing[key] = upos
    path = _path(lang, year, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"forms": existing}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def known_upos(lang: str, year: int, surface_form: str, *, root: Path | None = None) -> str | None:
    """The previously-recorded UPOS for `surface_form` in `(lang, year)`'s
    registry, or `None` if it isn't (yet) known there — callers use this to
    skip trusting a fresh LLM guess for a token this exact corpus has
    already confidently classified before, and instead use the recorded
    tag directly.
    """
    return load(lang, year, root=root).get(surface_form.lower().strip())


def record_many(lang: str, year: int, updates: dict[str, str], *, root: Path | None = None) -> None:
    """Batched `record()`: merges every `surface_form: upos` pair in
    `updates` into the `(lang, year)` registry with a single `load()` and,
    only if something actually changed, a single write — for a caller
    (`classify()`) that would otherwise call `record()` once per token,
    each doing its own full read-modify-write of the same file. Entries
    whose `upos` isn't closed-class are silently skipped, same filter as
    `record`.
    """
    existing = load(lang, year, root=root)
    changed = False
    for surface_form, upos in updates.items():
        upos = upos.upper()
        key = surface_form.lower().strip()
        if not key or upos not in _CLOSED_CLASS_UPOS or existing.get(key) == upos:
            continue
        existing[key] = upos
        changed = True
    if not changed:
        return
    path = _path(lang, year, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"forms": existing}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


__all__ = ["known_upos", "load", "record", "record_many"]
