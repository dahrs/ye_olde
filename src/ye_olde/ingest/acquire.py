"""Corpus acquisition (spec §6 step 0, previously entirely manual): given a
URL to a Project Gutenberg `.txt` file, a PDF on some other site, or an HTML
"read online" page, fetches it, infers the five `data/raw/` naming fields
(`lang_code`/`year`/`title`/`author`/`source`) from the actual fetched
content via the configured LLM, and saves it under `data/raw/<work>/` with a
filename `corpus_files.build_corpus_filename` guarantees is valid for the
rest of the pipeline.

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
from .checkpoint import Checkpoint
from .corpus_files import build_corpus_filename, parse_corpus_filename, rename_folder_to_match_oldest

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


def fetch_source(url: str) -> FetchedContent:
    """GETs `url`, following redirects, with up to `_MAX_ATTEMPTS` tries.

    Only retries conditions that are plausibly transient (a connection-level
    error, or one of `_RETRYABLE_STATUSES`) -- a permanent client error
    (404, 403, ...) raises immediately via `raise_for_status()` rather than
    wasting two more attempts on something that won't change.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            response = httpx.get(
                url,
                follow_redirects=True,
                timeout=_FETCH_TIMEOUT_SECONDS,
                headers={"User-Agent": _USER_AGENT},
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
) -> Path:
    """Fetches, names, and saves one `data/sources.yaml` entry under
    `raw_dir`. Resumable via `checkpoint` (keyed by URL) -- a URL already
    recorded as done is returned from cache without re-fetching or re-
    calling the LLM. Never overwrites an existing file: a computed filename
    that already exists on disk raises `FileExistsError` instead.
    """
    if checkpoint is not None and entry.url in checkpoint.entries:
        return Path(str(checkpoint.entries[entry.url]))

    def compute() -> str:
        return str(_acquire_source_uncached(entry, raw_dir=raw_dir, model=model))

    result = checkpoint.get_or_compute(entry.url, compute) if checkpoint is not None else compute()
    return Path(result)


def _acquire_source_uncached(entry: SourceEntry, *, raw_dir: Path, model: str | None) -> Path:
    fetched = fetch_source(entry.url)
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
    return target
