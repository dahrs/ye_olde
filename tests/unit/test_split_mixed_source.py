"""Thin integration-style smoke tests for scripts/split_mixed_source.py.

`main()` is argparse-plus-a-single-`split_mixed_source()`-call plumbing
(already covered by `test_split_mixed.py`) -- `load_manifest` is also
mocked here since this is a unit test, not a round-trip of the real YAML
manifest format.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import split_mixed_source as split_mixed_source_script  # noqa: E402


def _fake_usage() -> dict[str, int | float]:
    return {"calls": 1, "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.01}


def test_main_passes_cli_arguments_through_to_split_mixed_source(tmp_path, monkeypatch):
    seen_kwargs: dict[str, object] = {}
    monkeypatch.setattr(split_mixed_source_script, "load_manifest", lambda path: "fake-manifest")
    monkeypatch.setattr(split_mixed_source_script, "get_usage_summary", _fake_usage)

    def fake_split_mixed_source(path, manifest, **kwargs):
        seen_kwargs["path"] = path
        seen_kwargs["manifest"] = manifest
        seen_kwargs.update(kwargs)
        return []

    monkeypatch.setattr(split_mixed_source_script, "split_mixed_source", fake_split_mixed_source)

    split_mixed_source_script.main(
        [
            str(tmp_path / "mixed.txt"),
            "--manifest",
            str(tmp_path / "manifest.yaml"),
            "--chunk-size",
            "500",
        ]
    )

    assert seen_kwargs["path"] == Path(tmp_path / "mixed.txt")
    assert seen_kwargs["manifest"] == "fake-manifest"
    assert seen_kwargs["chunk_max_chars"] == 500


def test_main_prints_every_written_path(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(split_mixed_source_script, "load_manifest", lambda path: "fake-manifest")
    monkeypatch.setattr(split_mixed_source_script, "get_usage_summary", _fake_usage)
    monkeypatch.setattr(
        split_mixed_source_script, "split_mixed_source", lambda path, manifest, **kwargs: ["a.txt", "b.txt"]
    )

    split_mixed_source_script.main([str(tmp_path / "mixed.txt"), "--manifest", str(tmp_path / "manifest.yaml")])

    out = capsys.readouterr().out
    assert "a.txt" in out and "b.txt" in out


def test_main_exits_with_1_and_still_prints_cost_when_splitting_raises(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(split_mixed_source_script, "load_manifest", lambda path: "fake-manifest")
    monkeypatch.setattr(split_mixed_source_script, "get_usage_summary", _fake_usage)

    def fake_split_mixed_source(path, manifest, **kwargs):
        raise ValueError("a section matched no content")

    monkeypatch.setattr(split_mixed_source_script, "split_mixed_source", fake_split_mixed_source)

    with pytest.raises(SystemExit) as exc_info:
        split_mixed_source_script.main([str(tmp_path / "mixed.txt"), "--manifest", str(tmp_path / "manifest.yaml")])

    assert exc_info.value.code == 1
    assert "[cost]" in capsys.readouterr().err
