"""Unit tests for scripts/push_index_to_hf.py.

Unlike most other `scripts/*.py` entrypoints, this one has no underlying
`ye_olde.ingest.*` module behind it (its own docstring: "previously a
manual, copy-pasted-inline-snippet step with no script of its own") -- the
repo-id/token/index-dir/empty-dir validation and the dry-run branch all
live directly in `main()`, so unlike a thin argparse wrapper, this gets
full arrange/act/assert coverage of every branch.

The real upload call (`huggingface_hub.HfApi(...).upload_folder(...)`) is
a deferred import inside `main()` itself -- every non-dry-run test injects
a fake `huggingface_hub` module into `sys.modules` rather than hitting the
real Hugging Face Hub.

**`_fake_settings` always passes an explicit `hf_token`/`hf_dataset_repo_id`
default, never relies on both being merely absent.** `_env_file=None` only
disables reading the `.env` *file* -- it does nothing to stop
`pydantic-settings` from still reading a same-named real process
environment variable, and this machine's `HF_TOKEN`/`HF_DATASET_REPO_ID`
turned out to be real, live, write-scoped credentials once `.env` itself
was filled in for real use. An earlier version of this helper omitted
`hf_token` unless a test explicitly overrode it, which worked fine until
that real env var started being present -- at which point
`test_main_exits_when_token_missing_and_not_a_dry_run` silently stopped
testing the missing-token path at all and instead made a real,
authenticated push to the live `dahrs/ye_olde_data-index` dataset repo
with this file's own throwaway `_make_index_dir` fixture content
(confirmed live, 2026-11-10 -- `pairs/shard.parquet` had to be cleaned up
off the real repo afterward). Never assume an env var this project's own
`.env` might plausibly set is simply unset; clear it explicitly instead.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import push_index_to_hf  # noqa: E402
from ye_olde.config import Settings  # noqa: E402

_SENSITIVE_DEFAULTS: dict[str, str] = {
    "hf_dataset_repo_id": "",
    "hf_token": "",
    "hf_cache_dir": "/tmp/hf-cache",
}


def _fake_settings(**overrides: str) -> Settings:
    values = {**_SENSITIVE_DEFAULTS, **overrides}
    return Settings(_env_file=None, **values)


def _make_index_dir(tmp_path: Path, *, with_files: bool = True) -> Path:
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    if with_files:
        (index_dir / "pairs").mkdir()
        (index_dir / "pairs" / "shard.parquet").write_bytes(b"fake parquet bytes")
    return index_dir


def _install_fake_hf_api(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    upload_calls: list[dict[str, object]] = []

    class _FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            self.token = token

        def upload_folder(self, **kwargs: object) -> None:
            upload_calls.append({"token": self.token, **kwargs})

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=_FakeHfApi))
    return upload_calls


def test_main_exits_when_no_repo_id_given_and_none_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(push_index_to_hf, "get_settings", lambda: _fake_settings())
    index_dir = _make_index_dir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        push_index_to_hf.main(["--index-dir", str(index_dir)])

    assert exc_info.value.code == 1


def test_main_uses_the_cli_repo_id_over_the_configured_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(push_index_to_hf, "get_settings", lambda: _fake_settings(hf_dataset_repo_id="from/settings"))
    index_dir = _make_index_dir(tmp_path)

    push_index_to_hf.main(["--index-dir", str(index_dir), "--repo-id", "from/cli", "--dry-run"])

    assert "from/cli" in capsys.readouterr().err


def test_main_falls_back_to_the_configured_repo_id(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(push_index_to_hf, "get_settings", lambda: _fake_settings(hf_dataset_repo_id="from/settings"))
    index_dir = _make_index_dir(tmp_path)

    push_index_to_hf.main(["--index-dir", str(index_dir), "--dry-run"])

    assert "from/settings" in capsys.readouterr().err


def test_main_exits_when_token_missing_and_not_a_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        push_index_to_hf, "get_settings", lambda: _fake_settings(hf_dataset_repo_id="dahrs/ye_olde_data-index")
    )
    index_dir = _make_index_dir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        push_index_to_hf.main(["--index-dir", str(index_dir)])  # no --dry-run, no token configured

    assert exc_info.value.code == 1


def test_main_dry_run_does_not_require_a_token(tmp_path, monkeypatch):
    monkeypatch.setattr(
        push_index_to_hf, "get_settings", lambda: _fake_settings(hf_dataset_repo_id="dahrs/ye_olde_data-index")
    )
    index_dir = _make_index_dir(tmp_path)

    push_index_to_hf.main(["--index-dir", str(index_dir), "--dry-run"])  # must not raise


def test_main_exits_when_index_dir_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(
        push_index_to_hf,
        "get_settings",
        lambda: _fake_settings(hf_dataset_repo_id="dahrs/ye_olde_data-index", hf_token="tok"),
    )

    with pytest.raises(SystemExit) as exc_info:
        push_index_to_hf.main(["--index-dir", str(tmp_path / "does_not_exist")])

    assert exc_info.value.code == 1


def test_main_exits_when_index_dir_has_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(
        push_index_to_hf,
        "get_settings",
        lambda: _fake_settings(hf_dataset_repo_id="dahrs/ye_olde_data-index", hf_token="tok"),
    )
    index_dir = _make_index_dir(tmp_path, with_files=False)

    with pytest.raises(SystemExit) as exc_info:
        push_index_to_hf.main(["--index-dir", str(index_dir)])

    assert exc_info.value.code == 1


def test_main_dry_run_lists_files_without_uploading(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        push_index_to_hf,
        "get_settings",
        lambda: _fake_settings(hf_dataset_repo_id="dahrs/ye_olde_data-index", hf_token="tok"),
    )
    index_dir = _make_index_dir(tmp_path)

    # huggingface_hub is deliberately NOT mocked here: a dry run must return
    # before the `from huggingface_hub import HfApi` line is ever reached,
    # so importing the real (possibly absent) package would otherwise fail.
    push_index_to_hf.main(["--index-dir", str(index_dir), "--dry-run"])

    output = capsys.readouterr().err
    assert "would push" in output
    assert "pairs/shard.parquet" in output


def test_main_uploads_the_index_directory_when_not_a_dry_run(tmp_path, monkeypatch):
    monkeypatch.setattr(
        push_index_to_hf,
        "get_settings",
        lambda: _fake_settings(hf_dataset_repo_id="dahrs/ye_olde_data-index", hf_token="hf_real_token"),
    )
    index_dir = _make_index_dir(tmp_path)
    upload_calls = _install_fake_hf_api(monkeypatch)

    push_index_to_hf.main(["--index-dir", str(index_dir)])

    assert len(upload_calls) == 1
    call = upload_calls[0]
    assert call["token"] == "hf_real_token"
    assert call["folder_path"] == str(index_dir)
    assert call["repo_id"] == "dahrs/ye_olde_data-index"
    assert call["repo_type"] == "dataset"
