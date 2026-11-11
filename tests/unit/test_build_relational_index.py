"""Thin integration-style smoke tests for scripts/build_relational_index.py.

`main()` is argparse-plus-a-single-call plumbing over
`ye_olde.ingest.contributions` (already covered by `test_contributions.py`)
-- mocked here to check the wiring, not that module's own logic again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_relational_index  # noqa: E402


def _fake_usage() -> dict[str, int | float]:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}


def test_main_passes_arguments_through(monkeypatch):
    seen: dict[str, object] = {}

    def fake_build(contributions_root, out_dir, *, model, embedding_model):
        seen.update(
            contributions_root=contributions_root, out_dir=out_dir, model=model, embedding_model=embedding_model
        )
        return []

    monkeypatch.setattr(build_relational_index, "build_relational_index_for_contributions_root", fake_build)
    monkeypatch.setattr(build_relational_index, "get_usage_summary", _fake_usage)

    build_relational_index.main(
        ["--contributions-dir", "contributions", "--output-dir", "data/index", "--model", "gemini/x"]
    )

    assert seen == {
        "contributions_root": "contributions",
        "out_dir": "data/index",
        "model": "gemini/x",
        "embedding_model": None,
    }


def test_main_prints_each_written_path(monkeypatch, capsys):
    monkeypatch.setattr(
        build_relational_index,
        "build_relational_index_for_contributions_root",
        lambda *a, **k: [Path("data/index/relational/ang/900-1050.parquet")],
    )
    monkeypatch.setattr(build_relational_index, "get_usage_summary", _fake_usage)

    build_relational_index.main([])

    assert "900-1050.parquet" in capsys.readouterr().out


def test_main_notes_but_does_not_exit_nonzero_when_nothing_was_written(monkeypatch, capsys):
    monkeypatch.setattr(build_relational_index, "build_relational_index_for_contributions_root", lambda *a, **k: [])
    monkeypatch.setattr(build_relational_index, "get_usage_summary", _fake_usage)

    build_relational_index.main([])  # must not raise -- an empty contributions/ is normal today

    assert "nothing written" in capsys.readouterr().err


def test_main_exits_with_1_and_still_prints_cost_when_building_raises(monkeypatch, capsys):
    def boom(*a, **k):
        raise ValueError("schema-invalid contribution")

    monkeypatch.setattr(build_relational_index, "build_relational_index_for_contributions_root", boom)
    monkeypatch.setattr(build_relational_index, "get_usage_summary", _fake_usage)

    with pytest.raises(SystemExit) as exc_info:
        build_relational_index.main([])

    assert exc_info.value.code == 1
    assert "[cost]" in capsys.readouterr().err
