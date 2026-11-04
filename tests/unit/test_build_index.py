"""Thin integration-style smoke test for scripts/build_index.py.

`main()` is argparse-plus-a-single-`build_index_for_processed_root()`-call
-- the real indexing logic is `ye_olde.ingest.index`, already covered by
`test_index.py`. Just checks the CLI wiring: default/overridden
directories reach the call, and each written shard tuple is printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_index  # noqa: E402


def test_main_passes_directory_arguments_through(monkeypatch):
    seen: dict[str, str] = {}

    def fake_build_index_for_processed_root(processed_dir, output_dir):
        seen["processed_dir"] = processed_dir
        seen["output_dir"] = output_dir
        return []

    monkeypatch.setattr(build_index, "build_index_for_processed_root", fake_build_index_for_processed_root)

    build_index.main(["--processed-dir", "data/processed", "--output-dir", "data/index"])

    assert seen == {"processed_dir": "data/processed", "output_dir": "data/index"}


def test_main_uses_default_directories_when_not_given(monkeypatch):
    seen: dict[str, str] = {}

    def fake_build_index_for_processed_root(processed_dir, output_dir):
        seen["processed_dir"] = processed_dir
        seen["output_dir"] = output_dir
        return []

    monkeypatch.setattr(build_index, "build_index_for_processed_root", fake_build_index_for_processed_root)

    build_index.main([])

    assert seen == {"processed_dir": "data/processed", "output_dir": "data/index"}


def test_main_prints_each_written_shard_group(monkeypatch, capsys):
    monkeypatch.setattr(
        build_index,
        "build_index_for_processed_root",
        lambda processed_dir, output_dir: [("p.parquet", "v.faiss", "n.parquet", "b.parquet")],
    )

    build_index.main([])

    err = capsys.readouterr().err
    assert "p.parquet" in err and "v.faiss" in err and "n.parquet" in err and "b.parquet" in err
