"""Unit tests for acquire.py: content classification, HTML text extraction,
fetch retry behavior, collision/no-overwrite protection, resumability, and
the folder-renamed-to-match-its-oldest-file behavior.
"""

from __future__ import annotations

import httpx
import pytest

from ye_olde.ingest import acquire
from ye_olde.ingest.acquire import SourceEntry
from ye_olde.ingest.checkpoint import Checkpoint
from ye_olde.ingest.structured_sources import ResolvedDocument


def test_classify_content_prefers_content_type_header():
    assert acquire.classify_content("application/pdf", "https://example.com/book") == "pdf"
    assert acquire.classify_content("text/html; charset=utf-8", "https://example.com/book") == "html"
    assert acquire.classify_content("text/plain", "https://example.com/book") == "txt"


def test_classify_content_falls_back_to_url_suffix():
    assert acquire.classify_content("application/octet-stream", "https://example.com/book.pdf") == "pdf"
    assert acquire.classify_content("application/octet-stream", "https://example.com/page.html") == "html"
    assert acquire.classify_content("application/octet-stream", "https://example.com/book.txt") == "txt"


def test_extract_html_text_drops_script_and_nav_noise():
    html = """
    <html><body>
    <nav>Site navigation</nav>
    <script>var x = 1;</script>
    <article>The actual poem content.</article>
    </body></html>
    """
    text = acquire.extract_html_text(html)
    assert "The actual poem content." in text
    assert "navigation" not in text
    assert "var x" not in text


class _FakeResponse:
    def __init__(self, status_code, content=b"", content_type="text/plain", url="https://example.com"):
        self.status_code = status_code
        self.content = content
        self.headers = {"content-type": content_type}
        self.url = url
        self.request = None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"{self.status_code}", request=None, response=self)  # type: ignore[arg-type]


def test_fetch_source_retries_transient_failure_then_succeeds(monkeypatch):
    monkeypatch.setattr(acquire.time, "sleep", lambda *a, **k: None)
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if len(calls) < 2:
            raise httpx.ConnectTimeout("timed out")
        return _FakeResponse(200, content=b"hello", url=url)

    monkeypatch.setattr(acquire.httpx, "get", fake_get)
    result = acquire.fetch_source("https://example.com/book.txt")
    assert result.raw == b"hello"
    assert len(calls) == 2


def test_fetch_source_raises_immediately_on_permanent_error(monkeypatch):
    monkeypatch.setattr(acquire.time, "sleep", lambda *a, **k: None)
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return _FakeResponse(404, url=url)

    monkeypatch.setattr(acquire.httpx, "get", fake_get)
    with pytest.raises(httpx.HTTPStatusError):
        acquire.fetch_source("https://example.com/missing.txt")
    assert len(calls) == 1  # no retries wasted on a permanent error


def test_fetch_source_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(acquire.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(acquire.httpx, "get", lambda url, **k: _FakeResponse(503, url=url))
    with pytest.raises(httpx.HTTPStatusError):
        acquire.fetch_source("https://example.com/flaky.txt")


def _stub_pipeline(monkeypatch, metadata):
    monkeypatch.setattr(
        acquire,
        "fetch_source",
        lambda url, **k: acquire.FetchedContent(raw=b"Some Book content", content_type="text/plain", final_url=url),
    )
    monkeypatch.setattr(acquire, "call_llm_json", lambda *a, **k: metadata)


def test_acquire_source_writes_file_and_refuses_to_overwrite(monkeypatch, tmp_path):
    metadata = {"lang_code": "eng", "year": "1999", "title": "A Book", "author": "Author", "source": "Test Source"}
    _stub_pipeline(monkeypatch, metadata)

    entry = SourceEntry(url="https://example.com/book.txt")
    (path,) = acquire.acquire_source(entry, raw_dir=tmp_path)
    assert path.read_text(encoding="utf-8") == "Some Book content"
    assert path.name == "eng-1999-A_Book-Author-Test_Source.txt"

    # a second, different entry whose metadata collides with the same filename must not overwrite it
    other_entry = SourceEntry(url="https://example.com/book-mirror.txt")
    with pytest.raises(FileExistsError):
        acquire.acquire_source(other_entry, raw_dir=tmp_path)
    assert path.read_text(encoding="utf-8") == "Some Book content"  # untouched


def test_acquire_source_is_resumable_via_checkpoint(monkeypatch, tmp_path):
    metadata = {"lang_code": "eng", "year": "1999", "title": "A Book", "author": "Author", "source": "Test Source"}
    calls = []
    monkeypatch.setattr(
        acquire,
        "fetch_source",
        lambda url, **k: (
            calls.append(url) or acquire.FetchedContent(raw=b"content", content_type="text/plain", final_url=url)
        ),
    )
    monkeypatch.setattr(acquire, "call_llm_json", lambda *a, **k: metadata)

    checkpoint = Checkpoint(tmp_path / "progress.json", meta={})
    entry = SourceEntry(url="https://example.com/book.txt")
    first = acquire.acquire_source(entry, raw_dir=tmp_path, checkpoint=checkpoint)
    second = acquire.acquire_source(entry, raw_dir=tmp_path, checkpoint=checkpoint)
    assert first == second
    assert len(calls) == 1  # second call served from the checkpoint, no re-fetch


def test_acquire_source_renames_folder_when_new_file_is_older(monkeypatch, tmp_path):
    existing_folder = tmp_path / "eng-1999-A_Book-Author-Test_Source"
    existing_folder.mkdir()
    (existing_folder / "eng-1999-A_Book-Author-Test_Source.txt").write_text("modern edition", encoding="utf-8")

    older_metadata = {
        "lang_code": "enm",
        "year": "1400",
        "title": "A Book",
        "author": "Older Author",
        "source": "Test Source",
    }
    _stub_pipeline(monkeypatch, older_metadata)

    entry = SourceEntry(url="https://example.com/older.txt", folder="eng-1999-A_Book-Author-Test_Source")
    (path,) = acquire.acquire_source(entry, raw_dir=tmp_path)

    new_folder = tmp_path / "enm-1400-A_Book-Older_Author-Test_Source"
    assert path.parent == new_folder
    assert not existing_folder.exists()
    assert (new_folder / "eng-1999-A_Book-Author-Test_Source.txt").exists()  # the older edition's sibling moved with it


def _doc(**overrides: object) -> ResolvedDocument:
    defaults: dict[str, object] = {
        "lang_code": "lat",
        "year": -26,
        "title": "Elegiae",
        "author": "Tibullus",
        "original_author": "Tibullus",
        "source": "Perseus Digital Library",
        "role": "edition",
        "content": "text",
        "origin_url": "https://example.com/cts.xml",
    }
    defaults.update(overrides)
    return ResolvedDocument.model_validate(defaults)


def test_write_structured_documents_does_not_merge_different_authors_same_title(tmp_path):
    """Caught live: two different, unrelated authors' different works
    sharing a conventional title ("Elegiae") must not be merged just
    because the title string matches.
    """
    existing = tmp_path / "lat-BC16-Elegiae-Sextus_Propertius-Perseus_Digital_Library"
    existing.mkdir()
    (existing / "lat-BC16-Elegiae-Sextus_Propertius-Perseus_Digital_Library.txt").write_text("propertius")

    doc = _doc(year=-26, author="Tibullus", original_author="Tibullus")
    (written,) = acquire._write_structured_documents([doc], raw_dir=tmp_path)

    assert written.parent != existing  # a new, separate folder -- not merged into Propertius's
    assert (existing / "lat-BC16-Elegiae-Sextus_Propertius-Perseus_Digital_Library.txt").read_text() == "propertius"


def test_write_structured_documents_keeps_two_translations_by_different_people_same_year(tmp_path):
    """Caught live: two different translators' translations of the same
    work, coincidentally published the same year, were being treated as
    the same slot and one silently overwrote the other.
    """
    edition = _doc(lang_code="lat", year=-55, title="Carmina", author="Catullus", original_author="Catullus")
    first_translation = _doc(
        lang_code="eng",
        year=1894,
        title="Carmina",
        author="Sir Richard Francis Burton",
        original_author="Catullus",
        role="translation",
        content="burton's translation",
    )
    acquire._write_structured_documents([edition, first_translation], raw_dir=tmp_path)

    second_translation = _doc(
        lang_code="eng",
        year=1894,
        title="Carmina",
        author="Leonard C. Smithers",
        original_author="Catullus",
        role="translation",
        content="smithers's translation",
    )
    acquire._write_structured_documents([second_translation], raw_dir=tmp_path)

    folder = tmp_path / "lat-BC55-Carmina-Catullus-Perseus_Digital_Library"
    burton = folder / "eng-1894-Carmina-Sir_Richard_Francis_Burton-Perseus_Digital_Library.txt"
    smithers = folder / "eng-1894-Carmina-Leonard_C_Smithers-Perseus_Digital_Library.txt"
    assert burton.read_text() == "burton's translation"  # not overwritten
    assert smithers.read_text() == "smithers's translation"  # added alongside, not instead


def test_write_structured_documents_overwrites_same_language_edition(tmp_path):
    existing = tmp_path / "lat-523-De_Consolatione_Philosophiae-Boethius-Archive_org"
    existing.mkdir()
    (existing / "lat-523-De_Consolatione_Philosophiae-Boethius-Archive_org.txt").write_text("old scan")

    doc = _doc(
        lang_code="lat",
        year=524,
        title="De Consolatione Philosophiae",
        author="Boethius",
        original_author="Boethius",
        content="perseus edition",
    )
    (written,) = acquire._write_structured_documents([doc], raw_dir=tmp_path)

    assert written.read_text() == "perseus edition"
    assert not (existing / "lat-523-De_Consolatione_Philosophiae-Boethius-Archive_org.txt").exists()


def test_load_sources_parses_yaml(tmp_path):
    sources_path = tmp_path / "sources.yaml"
    sources_path.write_text(
        "- url: https://example.com/a.txt\n"
        "  folder: some-work\n"
        "  hints:\n"
        "    year: '1850'\n"
        "- url: https://example.com/b.txt\n",
        encoding="utf-8",
    )
    entries = acquire.load_sources(sources_path)
    assert entries[0].url == "https://example.com/a.txt"
    assert entries[0].folder == "some-work"
    assert entries[0].hints == {"year": "1850"}
    assert entries[1].folder is None
    assert entries[1].hints == {}
