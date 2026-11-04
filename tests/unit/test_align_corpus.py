"""Thin integration-style smoke tests for scripts/align_corpus.py.

`main()` is argparse-plus-a-single-`align_corpus_folder()`-call plumbing
(already covered by `test_align.py`) -- mocked here to check the wiring
(argument passthrough, the cost summary always printing even on failure,
the empty-result exit code), not `align.py`'s own logic again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import align_corpus  # noqa: E402


def _fake_usage() -> dict[str, int | float]:
    return {"calls": 3, "prompt_tokens": 100, "completion_tokens": 50, "cost_usd": 0.05}


def test_main_passes_cli_arguments_through_to_align_corpus_folder(monkeypatch):
    seen_kwargs: dict[str, object] = {}

    def fake_align_corpus_folder(folder, **kwargs):
        seen_kwargs["folder"] = folder
        seen_kwargs.update(kwargs)
        return ["data/processed/work/enm-1400_eng-1999.bitext.jsonl"]

    monkeypatch.setattr(align_corpus, "align_corpus_folder", fake_align_corpus_folder)
    monkeypatch.setattr(align_corpus, "get_usage_summary", _fake_usage)

    align_corpus.main(["data/raw/some-work", "--mode", "algorithmic", "--block-size", "10"])

    assert seen_kwargs["folder"] == "data/raw/some-work"
    assert seen_kwargs["mode"] == "algorithmic"
    assert seen_kwargs["block_size"] == 10


def test_main_prints_every_written_path(monkeypatch, capsys):
    monkeypatch.setattr(
        align_corpus, "align_corpus_folder", lambda folder, **kwargs: ["a.bitext.jsonl", "b.bitext.jsonl"]
    )
    monkeypatch.setattr(align_corpus, "get_usage_summary", _fake_usage)

    align_corpus.main(["data/raw/some-work"])

    out = capsys.readouterr().out
    assert "a.bitext.jsonl" in out
    assert "b.bitext.jsonl" in out


def test_main_exits_with_1_when_nothing_was_written(monkeypatch, capsys):
    monkeypatch.setattr(align_corpus, "align_corpus_folder", lambda folder, **kwargs: [])
    monkeypatch.setattr(align_corpus, "get_usage_summary", _fake_usage)

    with pytest.raises(SystemExit) as exc_info:
        align_corpus.main(["data/raw/some-work"])

    assert exc_info.value.code == 1
    assert "no bitext files written" in capsys.readouterr().err


def test_main_still_prints_the_cost_summary_when_alignment_raises(monkeypatch, capsys):
    def fake_align_corpus_folder(folder, **kwargs):
        raise RuntimeError("API billing error")

    monkeypatch.setattr(align_corpus, "align_corpus_folder", fake_align_corpus_folder)
    monkeypatch.setattr(align_corpus, "get_usage_summary", _fake_usage)

    with pytest.raises(SystemExit) as exc_info:
        align_corpus.main(["data/raw/some-work"])

    assert exc_info.value.code == 1
    assert "[cost]" in capsys.readouterr().err
