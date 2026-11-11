"""LLM-free ingestion for a source that publishes its own structured,
machine-readable catalog metadata -- an alternative to `acquire.py`'s
LLM-based `propose_metadata()`, used instead of it whenever a URL matches
something registered here. `acquire.acquire_source` tries
`resolve_structured_documents` first; `None` (no match) means "this source
isn't known to skip the LLM for" and it falls through to the ordinary LLM
path unchanged -- "not specified" always means LLM, exactly as asked.

First and so far only real case: Perseus Digital Library's CTS format
(github.com/PerseusDL/canonical-<lang>Lit). Each work directory carries a
`__cts__.xml` file that explicitly, textually tags every file belonging to
that work as an edition (the original-language text), a translation (its
own `xml:lang`), or a commentary (about the work, not the work itself --
always skipped, same reasoning as skipping a biography page in
`survey.py`'s prompt). Each individual document's own TEI header is just
as structured: `<author>` (the original author), `<editor role="translator">`
(a translation's translator) or plain `<editor>` (an edition's editor), and
`<sourceDesc>`'s `<imprint><date>` (that specific print edition/translation's
own publication year) -- no LLM call needed to tell any of this apart, or
to read off a name/year, for any of it.

One exception, scoped deliberately narrowly: a classical edition's own
`<date>` is the modern SCHOLARLY EDITION's publication year (e.g. 1881 for
Greenough's Aeneid edition), not Virgil's actual ~19 BC composition date --
and this corpus's `year` field means the latter (see the existing
`ang-995-...`/`lat-523-...` folders: both already date the original work
itself, not a later editor's printing of it). Perseus doesn't carry that
historical date anywhere in its own metadata, so `_composition_year` asks
the configured LLM for it. Wikidata's own structured "inception" (P571)
claim was tried first in an earlier version of this -- dropped after it
was caught live giving a confidently wrong answer for a well-documented
case (100 BC for the Aeneid, vs. Virgil's actual, extremely well-attested
~19 BC; the LLM got this one right): Wikidata's P571 is crowd-sourced, not
curated by a classicist, so a wrong claim there can't be told apart from a
correct one without an independent check anyway. This lookup is
deliberately only ever used for the edition entry in the work's OWN
original language (never a translation, whose own publication year Perseus
already states directly and reliably).

A translation whose own label/description identifies an AI model (so far:
Perseus's own `anon.beowulf.brunetti-deepseek2026-eng1`, labeled "(AI
draft)") keeps that provenance as its `author` field (e.g. "DeepSeek"),
never silently presented as a period-appropriate human translator's work.
"""

from __future__ import annotations

import html.entities
import re
from dataclasses import dataclass
from typing import Literal
from xml.etree import ElementTree

import httpx
from pydantic import BaseModel

from ..common.llm_client import call_llm_json
from ..common.logging import get_logger
from ..prompt import load_prompt

_log = get_logger(__name__)

_YEAR_SYSTEM_PROMPT = load_prompt("ingest", "lookup_composition_year_system")

_USER_AGENT = "ye_olde-corpus-acquirer/0.1 (+https://github.com/dahrs/ye_olde; structured-source ingestion)"
_HTTP_TIMEOUT_SECONDS = 60.0

Role = Literal["edition", "translation"]


class ResolvedDocument(BaseModel):
    """One document this module has fully resolved, ready for
    `acquire.py` to write under `data/raw/` -- the structured-source
    equivalent of what `acquire.propose_metadata` infers via an LLM for
    the generic path. `role` is kept (not just folded into the filename)
    because it drives `acquire.py`'s merge-with-an-existing-folder
    decision: an edition has at most one legitimate slot per language in
    a work's folder, a translation may legitimately have several (the
    same work translated again in a later era).

    `original_author` is kept separate from `author` -- `author` is
    role-specific (the original author for an edition, but a translator's
    own name for a translation), while `original_author` is always the
    work's true original author (from each document's own TEI header,
    which names it consistently across every file in the cluster, edition
    or translation alike). `acquire.py` matches an existing `data/raw/`
    folder by title + `original_author`, never by `author` -- matching by
    a translator's name would never find the folder a different
    translator's version of the very same work already lives in.
    """

    lang_code: str
    year: int  # may be negative (BC) -- see corpus_files.build_corpus_filename
    title: str
    author: str
    original_author: str
    source: str
    role: Role
    content: str
    origin_url: str


def resolve_structured_documents(url: str, *, model: str | None = None) -> list[ResolvedDocument] | None:
    """`None` for any URL that doesn't match a source registered here --
    the default, meaning "not known to skip the LLM," so `acquire.py`
    falls through to its own LLM-based path unchanged. For a match,
    returns every document belonging to that URL's work (edition +
    translations, commentary dropped) in one call -- a single Perseus
    `__cts__.xml` resolves a whole work's cluster of files at once, rather
    than needing one `sources.yaml` entry per individual file.
    """
    if _is_perseus_cts_url(url):
        return _resolve_perseus_work(url, model=model)
    return None


# ---------------------------------------------------------------------------
# Perseus Digital Library (CTS)
# ---------------------------------------------------------------------------

_CTS_NS = "{http://chs.harvard.edu/xmlns/cts}"
_XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

_PERSEUS_CTS_URL_RE = re.compile(
    r"^https://raw\.githubusercontent\.com/PerseusDL/canonical-\w+Lit/[^/]+/data/.+/__cts__\.xml$"
)

# Block-level TEI content elements worth extracting text from -- a <div>/<lg>/<body> is a
# structural wrapper only, its own text comes from these descendants instead (and would
# otherwise be double-counted if also extracted at the wrapper level).
_BLOCK_CONTENT_TAGS = {"l", "p", "head", "ab", "item", "speaker"}
# Descendant subtrees to drop entirely while extracting running text -- editorial/critical
# apparatus (footnotes, textual-criticism deletions, image captions), not the work's own content.
_SKIP_DESCENDANT_TAGS = {"note", "milestone", "pb", "cb", "gap", "del", "figure"}

_AI_DRAFT_RE = re.compile(r"\bAI draft\b", re.IGNORECASE)
_AI_NAME_YEAR_RE = re.compile(r"\(([A-Za-z][\w.\- ]*?),\s*(\d{3,4})\)")
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")

# Fallback free-text parsing of a __cts__.xml <ti:description>, used only when a document's own
# TEI header lacks the field entirely -- normally teiHeader's structured tags are used instead.
_EDITED_OR_TRANSLATED_BY_RE = re.compile(r"(?:edited|translated)\s+by\s+(.+)", re.IGNORECASE)
_NAME_UNTIL_PLACE_RE = re.compile(r"^(.*?),\s*[A-Z][\w.]*\s*:")
_ROLE_SUFFIX_RE = re.compile(r"\b(?:editor|translator)s?\b", re.IGNORECASE)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])\.\s+")


def _is_perseus_cts_url(url: str) -> bool:
    return bool(_PERSEUS_CTS_URL_RE.match(url))


_UNDEFINED_ENTITY_RE = re.compile(rb"&(\w+;)")
# XML's own predefined entities -- never substitute these, even though they also appear in the
# stdlib's HTML5 table (substituting &amp; through this path would double-escape it).
_XML_PREDEFINED_ENTITIES = {b"lt;", b"gt;", b"amp;", b"apos;", b"quot;"}


def _parse_xml(xml_bytes: bytes) -> ElementTree.Element:
    """`ElementTree.fromstring`, tolerant of a real issue seen live in
    several older `canonical-latinLit` source files: a bare HTML named
    entity (`&dagger;`, `&mdash;`, `&lsquo;`, ...) that plain XML doesn't
    define, which otherwise makes `ElementTree.fromstring` raise
    `ParseError` on an otherwise well-formed document. Replaces
    any such entity -- by name, from the stdlib's full HTML5 entity table
    -- with its literal Unicode character before parsing.
    """

    def _replace(m: re.Match[bytes]) -> bytes:
        name = m.group(1)
        if name in _XML_PREDEFINED_ENTITIES:
            return m.group(0)
        char = html.entities.html5.get(name.decode("ascii", errors="replace"))
        return char.encode("utf-8") if char is not None else m.group(0)

    return ElementTree.fromstring(_UNDEFINED_ENTITY_RE.sub(_replace, xml_bytes))


@dataclass(frozen=True)
class _CtsEntry:
    role: str  # "edition" | "translation" | "commentary"
    urn: str
    lang: str
    label: str
    description: str


def _local_tag(elem: ElementTree.Element) -> str:
    return elem.tag.rsplit("}", 1)[-1]


def _tei_ns(root: ElementTree.Element) -> str:
    """Most Perseus TEI documents declare the modern TEI P5 namespace on
    `<TEI>`; a few older ones (seen live: `canonical-latinLit`'s Claudian
    translation, root `<TEI.2>`) are the legacy, unnamespaced TEI P3/P4
    DTD instead. Returns the namespace prefix to search this particular
    document with -- the real TEI P5 namespace for a modern document,
    `""` for a legacy one -- so every `_tei_*`/`_extract_tei_text` lookup
    below works against either without needing to know which one it got.
    """
    return root.tag.rsplit("}", 1)[0] + "}" if root.tag.startswith("{") else ""


def _cts_child_text(el: ElementTree.Element, tag: str) -> str:
    child = el.find(f"{_CTS_NS}{tag}")
    return child.text.strip() if child is not None and child.text else ""


# This project's lang_code is always ISO 639-3's "terminology" (T) code, never the older ISO
# 639-2 "bibliographic" (B) one where the two differ -- see corpus_files.py's module docstring
# for the standing rule. Perseus's own __cts__.xml isn't always consistent about which one it
# tags xml:lang with -- confirmed live: the same Thucydides work's German translations are
# tagged "ger" (B) for two entries and "deu" (T) for two others, despite all four being the same
# (Modern Standard) language. Normalized to the T code on the way in so this corpus never
# silently splits one language across two lang_codes.
_ISO_639_2B_TO_639_3 = {
    "alb": "sqi",
    "arm": "hye",
    "baq": "eus",
    "bur": "mya",
    "chi": "zho",
    "cze": "ces",
    "dut": "nld",
    "fre": "fra",
    "geo": "kat",
    "ger": "deu",
    "gre": "ell",
    "ice": "isl",
    "mac": "mkd",
    "mao": "mri",
    "may": "msa",
    "per": "fas",
    "rum": "ron",
    "slo": "slk",
    "tib": "bod",
    "wel": "cym",
}


def _normalize_lang_code(code: str) -> str:
    return _ISO_639_2B_TO_639_3.get(code, code)


def _parse_cts_work(xml_bytes: bytes) -> tuple[str, str, list[_CtsEntry]]:
    """Returns `(work_lang, title, entries)` from a work-level `__cts__.xml`."""
    root = _parse_xml(xml_bytes)
    work_lang = _normalize_lang_code(root.get(_XML_LANG, ""))
    titles = root.findall(f"{_CTS_NS}title")
    title = next((t.text.strip() for t in titles if t.get(_XML_LANG) == "eng" and t.text), "")
    if not title:
        title = next((t.text.strip() for t in titles if t.text), "")
    if not title:
        raise ValueError("no <ti:title> found in __cts__.xml")

    entries: list[_CtsEntry] = []
    for role in ("edition", "translation", "commentary"):
        for el in root.findall(f"{_CTS_NS}{role}"):
            urn = el.get("urn", "")
            if not urn:
                continue
            entries.append(
                _CtsEntry(
                    role=role,
                    urn=urn,
                    lang=_normalize_lang_code(el.get(_XML_LANG) or work_lang),
                    label=_cts_child_text(el, "label"),
                    description=_cts_child_text(el, "description"),
                )
            )
    return work_lang, title, entries


def _extract_tei_text(xml_bytes: bytes) -> str:
    root = _parse_xml(xml_bytes)
    ns = _tei_ns(root)
    body = root.find(f".//{ns}body")
    if body is None:
        raise ValueError("no <body> found in TEI document")
    lines: list[str] = []
    for elem in body.iter():
        if _local_tag(elem) in _BLOCK_CONTENT_TAGS:
            text = " ".join(_text_excluding_skipped(elem).split())
            if text:
                lines.append(text)
    return "\n".join(lines)


def _text_excluding_skipped(elem: ElementTree.Element) -> str:
    parts = [elem.text or ""]
    for child in elem:
        if _local_tag(child) not in _SKIP_DESCENDANT_TAGS:
            parts.append(_text_excluding_skipped(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _tei_header_text(root: ElementTree.Element, path: str) -> str | None:
    el = root.find(path)
    return el.text.strip() if el is not None and el.text else None


def _tei_translator(root: ElementTree.Element) -> str | None:
    ns = _tei_ns(root)
    return _tei_header_text(root, f".//{ns}titleStmt/{ns}editor[@role='translator']")


def _tei_plain_editor(root: ElementTree.Element) -> str | None:
    """An editor who isn't a translator -- `role` absent (most documents) or
    explicitly `"editor"` (seen live: Perseus's own Beowulf glossary sets it
    redundantly), as opposed to `role="translator"` (`_tei_translator`'s job).
    """
    ns = _tei_ns(root)
    for el in root.findall(f".//{ns}titleStmt/{ns}editor"):
        if el.get("role") in (None, "editor") and el.text:
            return el.text.strip()
    return None


def _tei_author(root: ElementTree.Element) -> str | None:
    ns = _tei_ns(root)
    return _tei_header_text(root, f".//{ns}titleStmt/{ns}author")


def _tei_publication_year(root: ElementTree.Element) -> int | None:
    ns = _tei_ns(root)
    date_text = _tei_header_text(root, f".//{ns}sourceDesc/{ns}biblStruct/{ns}monogr/{ns}imprint/{ns}date")
    return _extract_year(date_text) if date_text else None


def _extract_year(text: str) -> int | None:
    matches = _YEAR_RE.findall(text)
    return int(matches[-1]) if matches else None


def _extract_role_name(description: str) -> str | None:
    """Fallback-only free-text parsing of a `__cts__.xml <ti:description>`
    for the translator/editor name, for the rare document whose own TEI
    header doesn't state one. Handles both description styles actually
    seen in Perseus's data: "edited/translated by <Name>, <City>: ..." and
    "<Name>, editor/translator. <City>: ...".
    """
    # Trailing whitespace/comma/semicolon only -- not "." -- since a trailing period is often
    # part of a legitimate abbreviation (e.g. "Greenough, J.B."), not a separator to discard.
    m = _EDITED_OR_TRANSLATED_BY_RE.search(description)
    if m:
        place_match = _NAME_UNTIL_PLACE_RE.search(m.group(1))
        name = place_match.group(1) if place_match else m.group(1)
        return name.strip(" ,;") or None

    for sentence in _SENTENCE_BOUNDARY_RE.split(description):
        role_match = _ROLE_SUFFIX_RE.search(sentence)
        if role_match:
            name = sentence[: role_match.start()]
            return name.strip(" ,;") or None
    return None


def _ai_draft_provenance(label: str, description: str) -> tuple[str, int | None] | None:
    """If this entry is explicitly labeled as AI-generated (so far: a
    Perseus `(AI draft)` marker), returns `(model_name, year)` so the AI's
    own identity is recorded as the translator -- never silently presented
    as a period-appropriate human translation.
    """
    combined = f"{label} {description}"
    if not _AI_DRAFT_RE.search(combined):
        return None
    name_year_match = _AI_NAME_YEAR_RE.search(label) or _AI_NAME_YEAR_RE.search(description)
    if name_year_match:
        return name_year_match.group(1).strip(), int(name_year_match.group(2))
    return "AI", _extract_year(combined)


_COMPOSITION_YEAR_ATTEMPTS = 3


def _composition_year(title: str, author: str, *, model: str | None) -> tuple[int, bool] | None:
    """The work's own historical composition year, `(year, is_bc)`, from
    the configured LLM's own classical-chronology knowledge. `None` means
    every attempt declined to give one -- treated as a genuinely disputed
    date (e.g. an oral-tradition epic with no scholarly-agreed century),
    and the caller skips this document rather than writing a fabricated
    one.

    Retries up to `_COMPOSITION_YEAR_ATTEMPTS` times, accepting the first
    confident answer -- caught live, this call is genuinely inconsistent
    on a borderline-but-actually-knowable case (Claudian's minor poems,
    a real author with a well-documented floruit but less individually-
    fixed dating than his major works, flipped between a specific year and
    "no confident answer" across identical back-to-back calls), while a
    truly disputed case (Beowulf) declined consistently across every
    retry. A wrong-but-confident answer isn't something retrying fixes or
    worsens either way (a well-attested case like the Aeneid answered
    identically every time tried); this only rescues the "got unlucky on
    this one call" case.

    An earlier version of this tried Wikidata's structured "inception"
    (P571) claim first, on the reasoning that a structured claim needs no
    LLM guessing. Dropped after it was caught live giving a confidently
    wrong answer for a well-documented case (100 BC for the Aeneid, vs.
    Virgil's actual, extremely well-attested ~19 BC -- the LLM got this one
    right): Wikidata's P571 is crowd-sourced, not curated by a classicist,
    so a wrong claim there can't be told apart from a correct one without
    an independent check anyway, which defeats the point of skipping the
    LLM for this field in the first place.
    """
    prompt = f"Title: {title}\nOriginal author: {author}"
    for attempt in range(1, _COMPOSITION_YEAR_ATTEMPTS + 1):
        result = call_llm_json(prompt, system=_YEAR_SYSTEM_PROMPT, model=model)
        if not isinstance(result, dict):
            raise ValueError(f"expected a JSON object from the composition-year LLM, got {type(result).__name__}")
        year = result.get("year")
        if year is not None:
            return int(year), bool(result.get("bc", False))
        _log.debug("composition-year attempt %d/%d for %r gave no year", attempt, _COMPOSITION_YEAR_ATTEMPTS, title)
    return None


def _resolve_perseus_work(cts_url: str, *, model: str | None) -> list[ResolvedDocument]:
    response = httpx.get(cts_url, timeout=_HTTP_TIMEOUT_SECONDS, headers={"User-Agent": _USER_AGENT})
    response.raise_for_status()
    work_lang, title, entries = _parse_cts_work(response.content)
    base_url = cts_url.rsplit("/", 1)[0] + "/"

    documents: list[ResolvedDocument] = []
    composition_year: tuple[int, bool] | None | Literal["unset"] = "unset"
    for entry in entries:
        if entry.role == "commentary":
            continue

        doc_filename = entry.urn.split(":")[-1] + ".xml"
        doc_url = base_url + doc_filename
        try:
            doc_response = httpx.get(doc_url, timeout=_HTTP_TIMEOUT_SECONDS, headers={"User-Agent": _USER_AGENT})
            doc_response.raise_for_status()
        except httpx.HTTPError as exc:
            _log.warning("skipping %s (%s %s): could not fetch %s: %s", title, entry.role, entry.urn, doc_url, exc)
            continue

        tei_root = _parse_xml(doc_response.content)
        content = _extract_tei_text(doc_response.content)
        header_author = _tei_author(tei_root)

        ai = _ai_draft_provenance(entry.label, entry.description)
        if ai is not None:
            author, year_val = ai
            is_bc = False
        elif entry.role == "edition" and entry.lang == work_lang:
            # The one case needing the real historical year -- an edition entry whose OWN
            # language matches the work's declared original language. An "edition" entry in a
            # DIFFERENT language (seen live: Beowulf's own glossary is tagged <ti:edition
            # xml:lang="eng">, even though the work's original language is "ang") is a modern
            # reference work with its own real print date already in its metadata, not an
            # ancient text needing historical dating -- falls through to the branch below instead.
            if composition_year == "unset":
                composition_year = _composition_year(title=title, author=header_author or "unknown", model=model)
            if composition_year is None:
                _log.warning("skipping %s edition (%s): the LLM gave no confident composition year", title, entry.urn)
                continue
            year_val, is_bc = composition_year
            author = header_author or "unknown"
        else:  # a translation, or an edition not in the work's own original language (e.g. a glossary)
            author = (
                _tei_translator(tei_root)
                or _tei_plain_editor(tei_root)
                or header_author
                or _extract_role_name(entry.description)
            ) or "unknown"
            year_val = _tei_publication_year(tei_root) or _extract_year(entry.description)
            is_bc = False

        if year_val is None:
            _log.warning("skipping %s (%s %s): no publication year could be determined", title, entry.role, entry.urn)
            continue

        documents.append(
            ResolvedDocument(
                lang_code=entry.lang,
                year=-year_val if is_bc else year_val,
                title=title,
                author=author,
                original_author=header_author or "unknown",
                source="Perseus Digital Library",
                role=entry.role,  # type: ignore[arg-type]
                content=content,
                origin_url=doc_url,
            )
        )
    return documents


def discover_perseus_cts_urls(repo: str, *, branch: str = "master") -> list[str]:
    """Every work-level `__cts__.xml` URL in a PerseusDL canonical-*Lit repo
    (e.g. `"PerseusDL/canonical-latinLit"`) -- one GitHub API call (its git
    tree, recursive) instead of hand-listing hundreds of `sources.yaml`
    entries one by one. A work-level file sits two directories below
    `data/` (`data/<author>/<work>/__cts__.xml`); the author-level one
    directly under it (`data/<author>/__cts__.xml`, author group metadata,
    not a work) is excluded by requiring that extra path segment.
    """
    response = httpx.get(
        f"https://api.github.com/repos/{repo}/git/trees/{branch}",
        params={"recursive": "1"},
        timeout=_HTTP_TIMEOUT_SECONDS,
        headers={"User-Agent": _USER_AGENT},
    )
    response.raise_for_status()
    tree = response.json()
    if tree.get("truncated"):
        raise ValueError(f"{repo}'s git tree was truncated by the GitHub API -- can't enumerate every work from it")
    urls = []
    for item in tree.get("tree", []):
        path = item["path"]
        if (
            item["type"] == "blob"
            and path.startswith("data/")
            and path.endswith("/__cts__.xml")
            and path.count("/") == 3
        ):
            urls.append(f"https://raw.githubusercontent.com/{repo}/{branch}/{path}")
    return urls
