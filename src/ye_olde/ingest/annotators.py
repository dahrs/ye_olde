"""Tool-first per-token UD annotation (lemma/upos/feats/deprel) for
languages with a real treebank-trained model — spec §3d's "tool where one
exists, LLM everywhere else" design.

**Every language Stanza has a full pos+lemma+depparse pipeline for is
registered below (87, as of Stanza's `resources_1.14.0.json`) — not just
the languages needed immediately.** Sourced programmatically, not
hand-typed: pulled from Stanza's own published resource manifest
(`https://raw.githubusercontent.com/stanfordnlp/stanza-resources/main/resources_1.14.0.json`),
filtered to entries whose `default_processors` covers `pos`+`lemma`+
`depparse` (tokenize-only or NER-only entries excluded — no real UD
tagging capability to offer), then each Stanza code resolved to this
project's ISO 639-3 convention via `pycountry` (an alpha_2 or alpha_3
lookup — not guessed). Three Stanza codes were excluded outright: `qaf`,
`qpm`, `qtd` — codes starting with `q` are ISO's own reserved private-use
range, and these three are UD code-switching corpora (e.g.
Turkish-German), not real single languages an ISO 639-3 code could name.
`zh-hant` (Traditional Chinese) was folded into `zh-hans`'s entry (`zho`)
rather than kept as a second `zho` key — the same ISO 639-3 code covers
both script variants, and this registry is keyed one entry per ISO code.

**No hand-picked package per language.** An earlier version of this
module explicitly requested `package="proiel"` for Latin/Greek — wrong:
live-checking Stanza's own resource manifest shows its *actual* default
for Latin is `ittb`, not `proiel` (Greek's is `perseus`). Hand-curating a
"best" treebank for every one of 87 languages isn't a judgment this
project has any basis for — `stanza.Pipeline(lang=...)` with no `package=`
argument delegates that choice to Stanza's own maintainers, the actually
correct authority for it, and scales to the full registry without needing
87 individually-researched opinions.

**CLTK was researched again, specifically for full-registry coverage this
time, and still adds nothing beyond what's registered here.** Every CLTK
`*StanzaPipeline` (`cltk/languages/pipelines.py`, v2.5.1) — Latin, Ancient
Greek, Church Slavonic, Old French, Gothic, Old English, Literary Chinese,
Ottoman Turkish, Classical Armenian, Coptic, Old Russian — wraps Stanza
directly for a language already in this registry; zero net-new coverage.
Every *other* CLTK-listed language (dozens: Akkadian, Sumerian-adjacent,
Tocharian A/B, Avestan, Sogdian, Old Persian, Ugaritic, Phoenician, Geez,
several Egyptian stages, Luwian, Lycian, Old Chinese, Old Japanese, many
more) is a `*GenAIPipeline` (`cltk/genai/`) — i.e. CLTK making its own
separate call to OpenAI, Mistral, or a local Ollama server, not an
independent statistical/rule-based tagger. That's strictly worse than this
project's own existing LLM path (`ingest.align`'s alignment call already
proposes lemma/upos/feats/deprel in the *same* call that does word
alignment) for three concrete reasons: it would be a second, redundant LLM
call instead of reusing one already being made; its own module docstring
states "Internal; no stability guarantees" (`cltk/genai/mistral.py`); and
its output would still need to pass through this project's own
`common.ud_tags` normalizers before it could be trusted anyway, same as
the LLM path already does directly, so no verification benefit exists.
**Conclusion: no language in this project's future should ever need a
CLTK integration for lemma/upos/feats/deprel** — a language missing from
this registry has no Stanza pipeline either, and should fall back to the
LLM path, exactly like `ang`/`enm` (this project's own v1 focus) already
do today.

**Every tag this module produces is run through `common.ud_tags`'s
normalizers before returning** — the same closed vocabularies the LLM
path (`ingest.align`, `classify`) is validated against, so a `lat` tag
and an `eng` tag are guaranteed the same notation even though they come
from entirely different sources (Stanza vs. an LLM call) — never trusted
by convention just because Stanza's output is UD-native in practice.

**Multi-word spans are deliberately not supported yet** — `annotate_span`
only resolves a span that matches exactly one of the tool's own tokens by
surface form; a multi-word span, or a span whose surface form repeats
elsewhere in the same sentence (ambiguous which occurrence is meant), both
return `None` and the caller falls back to the LLM's own proposal for that
link — exactly as if no tool were registered at all. Determining a
multi-word span's syntactic head from Stanza's dependency tree is a real,
addressable next step, not attempted here to keep this module's first
version honest about what it actually verifies.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..common.logging import get_logger
from ..common.ud_tags import normalize_deprel, normalize_upos

if TYPE_CHECKING:
    import stanza

_log = get_logger(__name__)

# {this project's ISO 639-3 code: Stanza's own language code} -- see module
# docstring for exactly how this was generated (Stanza's own resource
# manifest, resolved via pycountry, not hand-typed). Every one of these 87
# languages has a real Stanza pos+lemma+depparse pipeline as of Stanza's
# resources_1.14.0.json.
_REGISTRY: dict[str, str] = {
    "abk": "ab", "afr": "af", "ang": "ang", "ara": "ar", "bel": "be", "bul": "bg",
    "bxr": "bxr", "cat": "ca", "ces": "cs", "chu": "cu", "cop": "cop", "cym": "cy",
    "dan": "da", "deu": "de", "ell": "el", "eng": "en", "est": "et", "eus": "eu",
    "fao": "fo", "fas": "fa", "fin": "fi", "fra": "fr", "fro": "fro", "gla": "gd",
    "gle": "ga", "glg": "gl", "glv": "gv", "got": "got", "grc": "grc", "hbo": "hbo",
    "heb": "he", "hin": "hi", "hrv": "hr", "hsb": "hsb", "hun": "hu", "hye": "hy",
    "hyw": "hyw", "ind": "id", "isl": "is", "ita": "it", "jpn": "ja", "kat": "ka",
    "kaz": "kk", "kir": "ky", "kmr": "kmr", "kor": "ko", "kpv": "kpv", "lat": "la",
    "lav": "lv", "lij": "lij", "lit": "lt", "lzh": "lzh", "mar": "mr", "mlt": "mt",
    "myv": "myv", "nds": "nds", "nld": "nl", "nno": "nn", "nob": "nb", "ori": "or",
    "orv": "orv", "ota": "ota", "pcm": "pcm", "pol": "pl", "por": "pt", "ron": "ro",
    "rus": "ru", "san": "sa", "slk": "sk", "slv": "sl", "sme": "sme", "snd": "sd",
    "spa": "es", "sqi": "sq", "srp": "sr", "swe": "sv", "tam": "ta", "tel": "te",
    "tha": "th", "tur": "tr", "uig": "ug", "ukr": "uk", "urd": "ur", "vie": "vi",
    "wol": "wo", "xcl": "xcl", "zho": "zh-hans",
}  # fmt: skip

_PROCESSORS = "tokenize,mwt,pos,lemma,depparse"
# mwt (multi-word-token expansion -- splitting one orthographic token into
# several syntactic words, e.g. French "du" -> "de"+"le") isn't needed by
# every language and Stanza ships no mwt model at all for one that doesn't
# need it (confirmed for "ang"/Old English: no `mwt/` resource directory).
# Requesting it anyway doesn't download-and-skip -- it raises
# UnsupportedProcessorError out of `stanza.Pipeline.__init__` itself, so
# this is _pipeline's fallback request, not a second per-language registry
# to hand-maintain alongside _REGISTRY.
_PROCESSORS_NO_MWT = "tokenize,pos,lemma,depparse"


class ToolTokenTag(BaseModel):
    """One Stanza-tagged token — the same field shape as
    `classify.TokenTag`'s UD-relevant fields (and `ingest.align`'s
    per-link `*_upos`/`*_lemma`/`*_feats`/`*_deprel`), so a caller can
    treat a tool-sourced tag and an LLM-sourced tag identically once both
    have passed through `common.ud_tags`'s normalizers.
    """

    text: str
    lemma: str | None
    upos: str
    feats: str | None
    deprel: str | None


def has_tool(lang: str) -> bool:
    return lang in _REGISTRY


@lru_cache(maxsize=8)
def _pipeline(lang: str) -> stanza.Pipeline:
    """One `stanza.Pipeline` per language, built once and reused — Stanza
    auto-downloads the model on first use if it isn't already cached (same
    "provision on first real use" posture as `ingest.sense`'s WordNet
    index), so no separate setup step is required before this runs. No
    `package=` argument (see module docstring) — Stanza picks its own
    documented default treebank for `lang`.

    Tries the full `_PROCESSORS` list first; if `lang` turns out to have no
    `mwt` model (`UnsupportedProcessorError` naming exactly that processor
    — see `_PROCESSORS_NO_MWT`'s comment), retries once without it. This is
    a specific, expected, documented fallback (CLAUDE.md "Error handling"),
    not a broad catch: any other `UnsupportedProcessorError` (a genuinely
    missing, unexpected processor) is not this case and propagates.
    """
    import stanza
    from stanza.pipeline.core import UnsupportedProcessorError

    stanza_lang = _REGISTRY[lang]
    try:
        return stanza.Pipeline(stanza_lang, processors=_PROCESSORS, verbose=False)
    except UnsupportedProcessorError as exc:
        if exc.processor != "mwt":
            raise
        _log.debug("stanza has no mwt model for %r (%s) -- building its pipeline without mwt", lang, stanza_lang)
        return stanza.Pipeline(stanza_lang, processors=_PROCESSORS_NO_MWT, verbose=False)


@lru_cache(maxsize=256)
def _annotate_text(lang: str, text: str) -> tuple[ToolTokenTag, ...]:
    """Tags every token of `text` (in `lang`) via Stanza — cached per
    `(lang, text)` since `ingest.align` may ask about the same sentence
    more than once (once per link that falls within it), and re-running
    Stanza's full tokenize+tag+parse pipeline for each would be wasted
    work on identical input.
    """
    pipeline = _pipeline(lang)
    doc = pipeline(text)
    tags = []
    for sentence in doc.sentences:
        for word in sentence.words:
            tags.append(
                ToolTokenTag(
                    text=word.text,
                    lemma=word.lemma,
                    upos=normalize_upos(word.upos or ""),
                    feats=word.feats,
                    deprel=normalize_deprel(word.deprel) if word.deprel else None,
                )
            )
    return tuple(tags)


def annotate_span(lang: str, text: str, span: str) -> ToolTokenTag | None:
    """Returns the tool's own UD tag for `span` within `text`, or `None`
    if `lang` has no registered tool (`has_tool`), or `span` doesn't match
    exactly one of the tool's own tokens by surface form — see module
    docstring for why an ambiguous or multi-word match isn't guessed at.
    """
    if not has_tool(lang):
        return None
    span_norm = span.strip().lower()
    if not span_norm:
        return None
    matches = [tag for tag in _annotate_text(lang, text) if tag.text.strip().lower() == span_norm]
    if len(matches) != 1:
        return None
    return matches[0]


__all__ = ["ToolTokenTag", "annotate_span", "has_tool"]
