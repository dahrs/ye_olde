"""Thin integration-style smoke tests for scripts/acquire_corpus.py.

`main()` is argparse-plus-a-per-entry-loop plumbing over
`ye_olde.ingest.acquire` (already covered by `test_acquire.py`) -- these
tests mock `load_sources`/`acquire_source` and check the wiring around
them (per-entry failure isolation, exit code, `--no-cache`), not
`acquire.py`'s own fetch/naming logic again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import acquire_corpus  # noqa: E402
from ye_olde.ingest.acquire import SourceEntry  # noqa: E402


def _fake_usage() -> dict[str, int | float]:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}


def test_main_exits_when_sources_file_has_no_entries(tmp_path, monkeypatch, capsys):
    sources = tmp_path / "sources.yaml"
    sources.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(acquire_corpus, "get_usage_summary", _fake_usage)

    with pytest.raises(SystemExit) as exc_info:
        acquire_corpus.main(["--sources", str(sources), "--raw-dir", str(tmp_path / "raw")])

    assert exc_info.value.code == 1
    assert "no entries" in capsys.readouterr().err


def test_main_prints_each_succeeded_path_and_returns_normally(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(acquire_corpus, "load_sources", lambda path: [SourceEntry(url="https://example.invalid/a.txt")])
    monkeypatch.setattr(
        acquire_corpus,
        "acquire_source",
        lambda entry, *, raw_dir, model, checkpoint: [raw_dir / "a.txt"],
    )
    monkeypatch.setattr(acquire_corpus, "get_usage_summary", _fake_usage)

    acquire_corpus.main(["--raw-dir", str(tmp_path / "raw")])  # must not raise

    assert str(tmp_path / "raw" / "a.txt") in capsys.readouterr().out


def test_main_continues_past_a_failing_entry_and_exits_with_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        acquire_corpus,
        "load_sources",
        lambda path: [
            SourceEntry(url="https://example.invalid/bad.txt"),
            SourceEntry(url="https://example.invalid/good.txt"),
        ],
    )

    def fake_acquire_source(entry, *, raw_dir, model, checkpoint):
        if "bad" in entry.url:
            raise RuntimeError("404")
        return [raw_dir / "good.txt"]

    monkeypatch.setattr(acquire_corpus, "acquire_source", fake_acquire_source)
    monkeypatch.setattr(acquire_corpus, "get_usage_summary", _fake_usage)

    with pytest.raises(SystemExit) as exc_info:
        acquire_corpus.main(["--raw-dir", str(tmp_path / "raw")])

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "bad.txt" in err and "1 succeeded, 1 failed" in err


def test_main_no_cache_clears_an_existing_checkpoint(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    checkpoint_path = raw_dir / ".acquire_progress.json"
    checkpoint_path.write_text('{"meta": {}, "entries": {"stale": "entry"}}', encoding="utf-8")
    monkeypatch.setattr(acquire_corpus, "load_sources", lambda path: [SourceEntry(url="https://example.invalid/a.txt")])
    monkeypatch.setattr(acquire_corpus, "acquire_source", lambda entry, *, raw_dir, model, checkpoint: [])
    monkeypatch.setattr(acquire_corpus, "get_usage_summary", _fake_usage)

    acquire_corpus.main(["--raw-dir", str(raw_dir), "--no-cache"])

    assert not checkpoint_path.exists()
