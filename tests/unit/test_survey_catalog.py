"""Unit tests for scripts/survey_catalog.py.

`_result_key`/`_merge_into` carry real merge/dedup logic and get full
arrange/act/assert coverage. `main()` itself is argparse-plus-a-single-
`crawl()`-call plumbing (CLAUDE.md "Error handling": a boundary layer) --
covered with a couple of thin integration-style smoke tests instead,
mocking `crawl` rather than re-testing `ye_olde.ingest.survey` (already
covered by `test_survey.py`).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import survey_catalog  # noqa: E402
from ye_olde.ingest.survey import SurveyResult  # noqa: E402


def _result(
    document_url: str | None, chain: list[str], category: str = "old_text_candidate", **extra: object
) -> SurveyResult:
    return SurveyResult(
        document_url=document_url,
        found_via_chain=chain,
        title=extra.get("title", "a title"),
        category=category,  # type: ignore[arg-type]
        reasoning=extra.get("reasoning", ""),
        related_to=extra.get("related_to", ""),
    )


def test_result_key_prefers_document_url_when_present():
    result = _result("https://example.invalid/doc.txt", ["https://example.invalid/index"])

    assert survey_catalog._result_key(result) == "https://example.invalid/doc.txt"


def test_result_key_falls_back_to_the_last_chain_entry_without_a_document_url():
    result = _result(None, ["https://example.invalid/index", "https://example.invalid/author"])

    assert survey_catalog._result_key(result) == "https://example.invalid/author"


def test_merge_into_creates_a_new_file_when_none_exists(tmp_path):
    path = tmp_path / "catalog" / "acquisition_candidates.yaml"
    result = _result("https://example.invalid/a.txt", ["https://example.invalid/a.txt"])

    count = survey_catalog._merge_into(path, [result])

    assert count == 1
    assert path.is_file()
    written = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert written[0]["key"] == "https://example.invalid/a.txt"
    assert written[0]["document_url"] == "https://example.invalid/a.txt"


def test_merge_into_appends_a_new_key_to_an_existing_file(tmp_path):
    path = tmp_path / "acquisition_candidates.yaml"
    survey_catalog._merge_into(path, [_result("https://example.invalid/a.txt", ["https://example.invalid/a.txt"])])

    count = survey_catalog._merge_into(
        path, [_result("https://example.invalid/b.txt", ["https://example.invalid/b.txt"])]
    )

    assert count == 2
    keys = {entry["key"] for entry in yaml.safe_load(path.read_text(encoding="utf-8"))}
    assert keys == {"https://example.invalid/a.txt", "https://example.invalid/b.txt"}


def test_merge_into_updates_an_existing_entry_with_the_same_key_in_place(tmp_path):
    path = tmp_path / "acquisition_candidates.yaml"
    survey_catalog._merge_into(
        path, [_result("https://example.invalid/a.txt", ["https://example.invalid/a.txt"], reasoning="first pass")]
    )

    count = survey_catalog._merge_into(
        path,
        [_result("https://example.invalid/a.txt", ["https://example.invalid/a.txt"], reasoning="re-crawled, updated")],
    )

    assert count == 1  # same key -- updated in place, not duplicated
    entries = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert len(entries) == 1
    assert entries[0]["reasoning"] == "re-crawled, updated"


def test_merge_into_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "does" / "not" / "exist" / "yet" / "acquisition_candidates.yaml"

    survey_catalog._merge_into(path, [_result("https://example.invalid/a.txt", ["https://example.invalid/a.txt"])])

    assert path.is_file()


def _fake_usage() -> dict[str, int | float]:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}


def test_main_sorts_crawl_results_into_their_own_category_files(tmp_path, monkeypatch):
    monkeypatch.setattr(survey_catalog, "get_usage_summary", _fake_usage)
    monkeypatch.setattr(
        survey_catalog,
        "crawl",
        lambda start_url, **kwargs: [
            _result("https://example.invalid/old.txt", [start_url], category="old_text_candidate"),
            _result("https://example.invalid/dict.txt", [start_url], category="bilingual_reference"),
            _result(None, [start_url], category="uncertain"),
        ],
    )

    survey_catalog.main(["https://example.invalid/index", "--catalog-dir", str(tmp_path)])

    candidates = yaml.safe_load((tmp_path / "acquisition_candidates.yaml").read_text(encoding="utf-8"))
    references = yaml.safe_load((tmp_path / "bilingual_references.yaml").read_text(encoding="utf-8"))
    review = yaml.safe_load((tmp_path / "needs_manual_review.yaml").read_text(encoding="utf-8"))
    assert len(candidates) == 1 and candidates[0]["document_url"] == "https://example.invalid/old.txt"
    assert len(references) == 1 and references[0]["document_url"] == "https://example.invalid/dict.txt"
    assert len(review) == 1 and review[0]["document_url"] is None


def test_main_exits_with_1_when_every_starting_url_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(survey_catalog, "get_usage_summary", _fake_usage)

    def _raise(start_url, **kwargs):
        raise RuntimeError("site unreachable")

    monkeypatch.setattr(survey_catalog, "crawl", _raise)

    with pytest.raises(SystemExit) as exc_info:
        survey_catalog.main(["https://example.invalid/index", "--catalog-dir", str(tmp_path)])

    assert exc_info.value.code == 1


def test_main_continues_to_the_next_url_after_one_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(survey_catalog, "get_usage_summary", _fake_usage)

    def fake_crawl(start_url, **kwargs):
        if start_url == "https://example.invalid/bad":
            raise RuntimeError("site unreachable")
        return [_result("https://example.invalid/good.txt", [start_url])]

    monkeypatch.setattr(survey_catalog, "crawl", fake_crawl)

    with pytest.raises(SystemExit):  # the earlier failure still exits non-zero overall
        survey_catalog.main(
            ["https://example.invalid/bad", "https://example.invalid/index", "--catalog-dir", str(tmp_path)]
        )

    # ...but the second, successful URL's result was still written.
    candidates = yaml.safe_load((tmp_path / "acquisition_candidates.yaml").read_text(encoding="utf-8"))
    assert len(candidates) == 1
    assert candidates[0]["document_url"] == "https://example.invalid/good.txt"
