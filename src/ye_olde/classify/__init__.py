"""Classifies every token of the live input sentence — spec §2, §3b, §3d.

One LLM call, applying the exact same UD-style tag schema
`ye_olde.ingest.align` already uses to annotate corpus text at indexing time
(spec §3d) — the same technique, applied fresh to text that can't be
precomputed, not a separate system. `translate()` uses this to decide, per
token, "name" (bypass sense-retrieval, §4's "Period name-form lookup" via
`/attest?ner=PER`) vs. "common word" (go to sense retrieval, keyed on the
`upos`/`ner` tags so a query for a tagged name preferentially matches
identically-tagged indexed tokens rather than falling back to a plain
lexical match).

Never trusts the LLM's per-token alignment to the input blindly: a returned
tag whose own `text` doesn't match the corresponding deterministically
tokenized surface form (same tokenizer as `ingest.align`, via
`common.tokenize`) is a misalignment, not a tagging disagreement, and gets a
safe, minimal fallback tag instead of a tag that actually describes a
different word — the same "verify what the model claims about known text"
principle as `ingest.align._verify_pairs_against_source`.

Also consults and grows `common.function_words`'s self-populating
per-`(lang, year)` registry of closed-class surface forms — loaded once per
`classify()` call (not once per token: an earlier version called
`function_words.known_upos`/`record` inside the per-token loop, each doing
its own full read-modify-write of the same small file) and used only to
fill in a token this call's own LLM response left *unverifiable* (the "X"
fallback from `_resolve_tags`, below) — never to override a real,
verified tag that actively disagrees with the cache. That distinction
matters: a genuinely ambiguous closed-class word ("that" as SCONJ vs. DET,
"for" as ADP vs. SCONJ) can play a different grammatical role sentence to
sentence, so disagreement is real signal, not drift to be silently
overridden and then reinforced — an earlier version treated any cached
value as authoritative over a fresh tag, which self-reinforced whichever
tag happened to be recorded first regardless of context. `year` is
required, not optional: every real call in this project's domain has one
(even an estimated/circa one) — an optional param implying a legitimate
"unknown year" case would misrepresent that. This does not yet skip the LLM
call for known tokens (still one call per `classify()` invocation, tagging
every token) — only *trusts the cache to fill in a gap* once one exists;
excluding already-known tokens from the prompt entirely (a further, real
token-cost saving) is a natural next increment, not built yet.
"""

from __future__ import annotations

from pydantic import BaseModel

from ..common import function_words
from ..common.llm_client import call_llm_json
from ..common.tokenize import tokenize
from ..common.ud_tags import normalize_ner
from ..prompt import load_prompt

_CLASSIFY_SYSTEM_PROMPT = load_prompt("classify", "classify_system")


class TokenTag(BaseModel):
    """One token's UD-style annotation (spec §3d) — a public function's
    return value, so a `BaseModel` per CLAUDE.md's Pydantic guidance, not a
    bare dict.
    """

    text: str
    lemma: str | None
    upos: str
    feats: str | None
    deprel: str | None
    ner: str

    @property
    def is_name(self) -> bool:
        """True for a token classify has tagged as a named entity of any
        kind — spec §2's CLS branch: a name bypasses sense-retrieval
        entirely (§4's "Period name-form lookup", now just `/attest`
        filtered to `ner=PER`/etc., not a separate registry lookup).
        """
        return self.ner != "O"


def classify(sentence: str, lang: str, year: int, *, model: str | None = None) -> list[TokenTag]:
    """Tags every token of `sentence` (already known to be in language
    `lang`, at `year`) with the spec §3d schema. Returns one `TokenTag` per
    token, in `common.tokenize.tokenize`'s own token order — always the
    same length as `tokenize(sentence)`, even where a proposed tag was
    rejected as unverifiable (see module docstring).
    """
    tokens = tokenize(sentence)
    if not tokens:
        return []
    prompt = f"LANGUAGE: {lang}\nTOKENS (in order, 0-indexed):\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(tokens))
    raw = call_llm_json(prompt, system=_CLASSIFY_SYSTEM_PROMPT, model=model)
    known = function_words.load(lang, year)  # one read for the whole call, not one per token
    tags = [_apply_known_function_word(tag, known) for tag in _resolve_tags(tokens, raw)]
    function_words.record_many(lang, year, {tag.text: tag.upos for tag in tags})  # one write, only if changed
    return tags


def _apply_known_function_word(tag: TokenTag, known: dict[str, str]) -> TokenTag:
    """Fills in `tag.upos` from the registry only when this call's own tag
    is `"X"` — unverifiable (see `_resolve_tags`), not a real linguistic
    judgment to weigh against. Never overrides a verified, disagreeing tag
    — see module docstring for why a disagreement is signal, not noise.
    """
    if tag.upos != "X":
        return tag
    known_upos = known.get(tag.text.lower().strip())
    if known_upos is None:
        return tag
    return tag.model_copy(update={"upos": known_upos})


def _resolve_tags(tokens: list[str], raw: object) -> list[TokenTag]:
    raw_list = raw if isinstance(raw, list) else []
    tags: list[TokenTag] = []
    for i, token in enumerate(tokens):
        entry = raw_list[i] if i < len(raw_list) and isinstance(raw_list[i], dict) else {}
        claimed_text = str(entry.get("text") or "").strip()
        if claimed_text.lower() != token.lower():
            # Misaligned reply (wrong position, dropped/merged token, ...):
            # a safe minimal tag for the *real* token, never a tag that
            # actually describes something else.
            tags.append(TokenTag(text=token, lemma=None, upos="X", feats=None, deprel=None, ner="O"))
            continue
        ner = normalize_ner(str(entry.get("ner") or ""))
        tags.append(
            TokenTag(
                text=token,
                lemma=str(entry.get("lemma") or "").strip() or None,
                upos=str(entry.get("upos") or "").strip() or "X",
                feats=str(entry.get("feats") or "").strip() or None,
                deprel=str(entry.get("deprel") or "").strip() or None,
                ner=ner,
            )
        )
    return tags


__all__ = ["TokenTag", "classify"]
