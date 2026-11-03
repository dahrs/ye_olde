"""Corpus acquisition (spec §6 step 0, previously entirely manual): given a
URL, resolves it into one or more `data/raw/<work>/` files with filenames
`corpus_files.build_corpus_filename` guarantees are valid for the rest of
the pipeline. Two ways a URL gets resolved:

- `structured_sources.resolve_structured_documents` first -- for a source
  registered there (so far: Perseus Digital Library's CTS catalogs), it
  reads that source's own structured metadata and returns every document
  the naming fields are needed for, with no LLM call at all. `None` (the
  URL isn't one of those) falls through to the next path -- "not a known
  LLM-free source" always means the LLM path, never a silent skip.
- Otherwise, the original path: fetch it (a Project Gutenberg `.txt` file,
  a PDF on some other site, an HTML "read online" page), infer the five
  `data/raw/` naming fields (`lang_code`/`year`/`title`/`author`/`source`)
  from the actual fetched content via the configured LLM.

A structured-source result that turns out to already be covered by an
earlier, less reliable acquisition (matched by title via
`corpus_files.find_matching_work_folder`) is merged into that same folder
rather than duplicated -- see `_write_structured_documents`.

The list of URLs to fetch lives in `data/sources.yaml` (`load_sources`), not
hardcoded here -- see that file's own comments for its format.

Every failure here (a bad HTTP status, a decode error, a malformed LLM
reply, a filename collision) raises rather than being caught -- this is
domain code, not a boundary; `scripts/acquire_corpus.py`'s `main()` is the
boundary that catches, logs, and reports per entry (CLAUDE.md "Error
handling").
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx
import yaml
from bs4 import BeautifulSoup
from pydantic import BaseModel

from ..common.llm_client import call_llm_json
from ..common.logging import get_logger
from ..prompt import load_prompt
from . import authenticated_sites, structured_sources
from .checkpoint import Checkpoint
from .corpus_files import (
    SUPPORTED_EXTENSIONS,
    authors_plausibly_match,
    build_corpus_filename,
    find_matching_work_folder,
    parse_corpus_filename,
    rename_folder_to_match_oldest,
)
from .structured_sources import ResolvedDocument

_log = get_logger(__name__)

_SYSTEM_PROMPT = load_prompt("ingest", "propose_filename_system")

# A descriptive, contactable UA -- some of this project's actual sources
# (archive.org in particular) are known to reject or deprioritize a generic/
# missing User-Agent.
_USER_AGENT = "ye_olde-corpus-acquirer/0.1 (+https://github.com/; personal research corpus tool)"

_FETCH_TIMEOUT_SECONDS = 60.0
_MAX_ATTEMPTS = 3
_RETRYABLE_STATUSES = {429, 502, 503, 504}

# How much of the fetched content to show the naming LLM -- enough to
# reliably include a title page/front matter without spending the whole
# context window on a potentially very long work.
_PREVIEW_CHARS = 4000
_PDF_PREVIEW_PAGES = 3

ContentKind = Literal["txt", "pdf", "html"]


class SourceEntry(BaseModel):
    """One `data/sources.yaml` entry -- user-edited external input, so
    validated on the way in (CLAUDE.md "Pydantic") rather than trusted as a
    bare dict.
    """

    url: str
    folder: str | None = None
    hints: dict[str, str] = {}


def load_sources(path: Path) -> list[SourceEntry]:
    """Reads `data/sources.yaml`-style worklist into validated entries."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    return [SourceEntry.model_validate(entry) for entry in raw]


@dataclass(frozen=True)
class FetchedContent:
    """Raw result of one HTTP fetch -- internal/algorithmic, not a public
    return value or validated input, so a plain dataclass (same reasoning
    as `corpus_files.CorpusFile`), not a `BaseModel`.
    """

    raw: bytes
    content_type: str
    final_url: str


def fetch_source(url: str, *, extra_headers: dict[str, str] | None = None) -> FetchedContent:
    """GETs `url`, following redirects, with up to `_MAX_ATTEMPTS` tries.

    Only retries conditions that are plausibly transient (a connection-level
    error, or one of `_RETRYABLE_STATUSES`) -- a permanent client error
    (404, 403, ...) raises immediately via `raise_for_status()` rather than
    wasting two more attempts on something that won't change.

    `extra_headers` -- from `authenticated_sites.headers_for(url)` at every
    real call site -- merges over (and can override) the default headers
    below; `{}`/`None` for every URL outside that module's registry, which
    is every URL this function has ever handled before this parameter
    existed. Zero behavior change for the plain, no-identification-needed
    path that stays the default.
    """
    headers = {"User-Agent": _USER_AGENT, **(extra_headers or {})}
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = httpx.get(
                url,
                follow_redirects=True,
                timeout=_FETCH_TIMEOUT_SECONDS,
                headers=headers,
            )
        except httpx.TransportError as exc:
            last_exc = exc
        else:
            if response.status_code in _RETRYABLE_STATUSES:
                last_exc = httpx.HTTPStatusError(
                    f"{response.status_code} from {url}", request=response.request, response=response
                )
            else:
                response.raise_for_status()
                return FetchedContent(
                    raw=response.content,
                    content_type=response.headers.get("content-type", ""),
                    final_url=str(response.url),
                )
        if attempt < _MAX_ATTEMPTS:
            _log.debug("fetch attempt %d/%d for %r failed: %s -- retrying", attempt, _MAX_ATTEMPTS, url, last_exc)
            time.sleep(2 ** (attempt - 1))
    assert last_exc is not None  # the loop only exits without returning if an exception was recorded
    raise last_exc


def classify_content(content_type: str, url: str) -> ContentKind:
    """Decides whether a fetched response is a PDF, plain text, or an HTML
    page to scrape, from the `Content-Type` header first (the authoritative
    signal) and the URL's own suffix as a fallback for a server that sends a
    generic type like `application/octet-stream`.
    """
    mime = content_type.split(";")[0].strip().lower()
    if mime == "application/pdf":
        return "pdf"
    if mime == "text/html":
        return "html"
    if mime == "text/plain":
        return "txt"
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in (".html", ".htm"):
        return "html"
    return "txt"


def extract_html_text(html: str) -> str:
    """Generic visible-text extraction from an HTML "read online" page
    (e.g. a Poetry Foundation poem page) -- drops markup noise that isn't
    part of the work's own content, leaves everything else as plain text
    for the LLM naming/cleaning stages to judge.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _decode_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        _log.debug("content was not valid UTF-8 -- falling back to latin-1 with replacement")
        return raw.decode("latin-1", errors="replace")


def _pdf_preview_text(raw: bytes) -> str:
    """Text from a PDF's first few pages, for the naming LLM's eyes only --
    never saved, never fed downstream. Deliberately independent of
    `extract.py`'s own PDF extraction (boilerplate-stripping, mojibake-
    fixing the file actually saved to disk): that happens later, by
    `align_corpus.py`, against the raw `.pdf` this module saves unmodified.
    """
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    pages = reader.pages[:_PDF_PREVIEW_PAGES]
    return "\n\n".join(page.extract_text() or "" for page in pages)


def propose_metadata(url: str, text_preview: str, hints: dict[str, str], *, model: str | None = None) -> dict[str, str]:
    """Infers the five `data/raw/` naming fields from the actual fetched
    content via the configured LLM (`propose_filename_system`, see
    `src/ye_olde/prompt/ingest.yaml`), then applies `hints` as overrides --
    a hinted field is never second-guessed by the model's answer.
    """
    domain = urlparse(url).netloc
    prompt = f"URL: {url}\nDomain: {domain}\n\nFetched content excerpt:\n{text_preview}"
    result = call_llm_json(prompt, system=_SYSTEM_PROMPT, model=model)
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object from the naming LLM, got {type(result).__name__}")
    metadata = {key: str(result[key]) for key in ("lang_code", "year", "title", "author", "source")}
    metadata.update(hints)
    return metadata


def acquire_source(
    entry: SourceEntry,
    *,
    raw_dir: Path,
    model: str | None = None,
    checkpoint: Checkpoint | None = None,
) -> list[Path]:
    """Fetches, names, and saves one `data/sources.yaml` entry under
    `raw_dir` -- one file for the ordinary LLM path, or one per
    edition/translation when a structured source (e.g. Perseus) resolves
    this URL to a whole work's cluster at once. Resumable via `checkpoint`
    (keyed by URL) -- a URL already recorded as done is returned from cache
    without re-fetching or re-calling the LLM. Never overwrites an existing
    file outside the structured-source merge case (see
    `_write_structured_documents`): a computed filename that already
    exists on disk otherwise raises `FileExistsError` instead.
    """

    def compute() -> list[str]:
        return [str(p) for p in _acquire_source_uncached(entry, raw_dir=raw_dir, model=model)]

    result = checkpoint.get_or_compute(entry.url, compute) if checkpoint is not None else compute()
    return [Path(p) for p in result]


def _acquire_source_uncached(entry: SourceEntry, *, raw_dir: Path, model: str | None) -> list[Path]:
    structured = structured_sources.resolve_structured_documents(entry.url, model=model)
    if structured is not None:
        return _write_structured_documents(structured, raw_dir=Path(raw_dir))

    fetched = fetch_source(entry.url, extra_headers=authenticated_sites.headers_for(entry.url))
    kind = classify_content(fetched.content_type, fetched.final_url)

    if kind == "pdf":
        ext = ".pdf"
        preview = _pdf_preview_text(fetched.raw)[:_PREVIEW_CHARS]
    else:
        text = _decode_text(fetched.raw)
        if kind == "html":
            text = extract_html_text(text)
        ext = ".txt"
        preview = text[:_PREVIEW_CHARS]

    metadata = propose_metadata(entry.url, preview, entry.hints, model=model)
    filename = build_corpus_filename(ext=ext, **metadata)
    new_file = parse_corpus_filename(Path(filename))

    folder_name = entry.folder or Path(filename).stem
    folder_path = Path(raw_dir) / folder_name
    folder_path = rename_folder_to_match_oldest(folder_path, new_file, raw_dir=Path(raw_dir))
    target = folder_path / filename
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing file {target}")

    folder_path.mkdir(parents=True, exist_ok=True)
    if kind == "pdf":
        target.write_bytes(fetched.raw)
    else:
        target.write_text(text, encoding="utf-8")
    return [target]


def _write_structured_documents(documents: list[ResolvedDocument], *, raw_dir: Path) -> list[Path]:
    """Writes every document a structured source resolved for one work,
    merging into an existing `data/raw/<work>/` folder when
    `corpus_files.find_matching_work_folder` finds one already covering
    this title and original author (e.g. from an earlier, less reliable
    LLM-crawl acquisition) rather than creating a duplicate. Within the
    matched (or newly created) folder, a document *overwrites* whatever
    already occupies its slot -- same `lang_code` for an edition (a work
    has at most one original-language text), same `lang_code` AND a
    plausibly-matching translator for a translation (a work can have
    several genuine translations into the same language, from different
    eras OR even the same year by different translators -- caught live:
    two different 1894 English translations of Catullus, by Burton and by
    Smithers, are not the same slot just because their years coincide) --
    on the reasoning that a structured source resolving this far is more
    reliable than the LLM guess that produced whatever's already there.
    Anything that doesn't match an existing slot is added as a new file
    instead.
    """
    if not documents:
        return []
    raw_dir = Path(raw_dir)
    folder_path = find_matching_work_folder(raw_dir, documents[0].title, documents[0].original_author)

    written: list[Path] = []
    for doc in documents:
        filename = build_corpus_filename(
            lang_code=doc.lang_code, year=doc.year, title=doc.title, author=doc.author, source=doc.source, ext=".txt"
        )
        new_file = parse_corpus_filename(Path(filename))
        if folder_path is None:
            folder_path = raw_dir / Path(filename).stem  # anchors a brand-new work on its first document
        folder_path = rename_folder_to_match_oldest(folder_path, new_file, raw_dir=raw_dir)

        superseded = _find_overwrite_target(folder_path, doc)
        target = superseded if superseded is not None else folder_path / filename
        if target.exists() and superseded is None:
            raise FileExistsError(f"refusing to overwrite existing file {target}")

        folder_path.mkdir(parents=True, exist_ok=True)
        if superseded is not None and superseded != folder_path / filename:
            _log.info("replacing %s with more reliably-acquired %s", superseded, filename)
            superseded.unlink()
        final_target = folder_path / filename
        final_target.write_text(doc.content, encoding="utf-8")
        written.append(final_target)
    return written


def _find_overwrite_target(folder: Path, doc: ResolvedDocument) -> Path | None:
    """The existing file in `folder` that `doc` should replace, if any --
    see `_write_structured_documents` for the matching rule.
    """
    if not folder.is_dir():
        return None
    for existing in folder.iterdir():
        if not (existing.is_file() and existing.suffix.lower() in SUPPORTED_EXTENSIONS):
            continue
        try:
            parsed = parse_corpus_filename(existing)
        except ValueError:
            continue
        if parsed.lang_code != doc.lang_code:
            continue
        if doc.role == "edition":
            return existing
        if parsed.year == doc.year and authors_plausibly_match(parsed.author, doc.author):
            return existing
    return None
