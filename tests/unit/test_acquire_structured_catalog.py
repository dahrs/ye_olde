"""Thin integration-style smoke tests for scripts/acquire_structured_catalog.py.

`main()` is argparse-plus-a-per-repo-then-per-work-loop over
`discover_perseus_cts_urls`/`acquire_source` -- mocked here to check the
wiring (per-repo discovery failure, `--limit`, per-work failure isolation,
exit code), not `ye_olde.ingest.structured_sources`'s own logic again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import acquire_structured_catalog  # noqa: E402


def _fake_usage() -> dict[str, int | float]:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}


def test_main_acquires_every_discovered_url(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        acquire_structured_catalog,
        "discover_perseus_cts_urls",
        lambda repo, branch: ["https://example.invalid/a.txt", "https://example.invalid/b.txt"],
    )
    monkeypatch.setattr(
        acquire_structured_catalog,
        "acquire_source",
        lambda entry, *, raw_dir, model, checkpoint: [raw_dir / Path(entry.url).name],
    )
    monkeypatch.setattr(acquire_structured_catalog, "get_usage_summary", _fake_usage)

    acquire_structured_catalog.main(
        ["--repo", "PerseusDL/canonical-latinLit", "--raw-dir", str(tmp_path / "raw")]
    )  # must not raise

    out = capsys.readouterr().out
    assert str(tmp_path / "raw" / "a.txt") in out
    assert str(tmp_path / "raw" / "b.txt") in out


def test_main_respects_the_limit_flag(tmp_path, monkeypatch):
    acquired: list[str] = []
    monkeypatch.setattr(
        acquire_structured_catalog,
        "discover_perseus_cts_urls",
        lambda repo, branch: [f"https://example.invalid/{i}.txt" for i in range(5)],
    )

    def fake_acquire_source(entry, *, raw_dir, model, checkpoint):
        acquired.append(entry.url)
        return [raw_dir / "x.txt"]

    monkeypatch.setattr(acquire_structured_catalog, "acquire_source", fake_acquire_source)
    monkeypatch.setattr(acquire_structured_catalog, "get_usage_summary", _fake_usage)

    acquire_structured_catalog.main(
        ["--repo", "PerseusDL/canonical-latinLit", "--raw-dir", str(tmp_path / "raw"), "--limit", "2"]
    )

    assert len(acquired) == 2


def test_main_continues_to_the_next_repo_after_discovery_fails(tmp_path, monkeypatch, capsys):
    def fake_discover(repo, branch):
        if repo == "PerseusDL/canonical-badLit":
            raise RuntimeError("repo not found")
        return ["https://example.invalid/good.txt"]

    monkeypatch.setattr(acquire_structured_catalog, "discover_perseus_cts_urls", fake_discover)
    monkeypatch.setattr(
        acquire_structured_catalog,
        "acquire_source",
        lambda entry, *, raw_dir, model, checkpoint: [raw_dir / "good.txt"],
    )
    monkeypatch.setattr(acquire_structured_catalog, "get_usage_summary", _fake_usage)

    acquire_structured_catalog.main(
        [
            "--repo",
            "PerseusDL/canonical-badLit",
            "--repo",
            "PerseusDL/canonical-latinLit",
            "--raw-dir",
            str(tmp_path / "raw"),
        ]
    )  # must not raise -- a failed repo is reported, not fatal

    out = capsys.readouterr().out
    assert str(tmp_path / "raw" / "good.txt") in out


def test_main_exits_with_1_when_a_work_fails_to_acquire(tmp_path, monkeypatch):
    monkeypatch.setattr(
        acquire_structured_catalog,
        "discover_perseus_cts_urls",
        lambda repo, branch: ["https://example.invalid/a.txt"],
    )

    def fake_acquire_source(entry, *, raw_dir, model, checkpoint):
        raise RuntimeError("no confident composition year")

    monkeypatch.setattr(acquire_structured_catalog, "acquire_source", fake_acquire_source)
    monkeypatch.setattr(acquire_structured_catalog, "get_usage_summary", _fake_usage)

    with pytest.raises(SystemExit) as exc_info:
        acquire_structured_catalog.main(["--repo", "PerseusDL/canonical-latinLit", "--raw-dir", str(tmp_path / "raw")])

    assert exc_info.value.code == 1
