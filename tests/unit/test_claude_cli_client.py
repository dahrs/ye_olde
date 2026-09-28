"""Unit tests for the standalone Claude Code CLI backend. `subprocess.run`
is monkeypatched so these run with no `claude` binary required and no real
subscription/API call.
"""

from __future__ import annotations

import json
import types

import pytest

from ye_olde.ingest import claude_cli_client
from ye_olde.ingest.claude_cli_client import ClaudeCliCompletion, complete_via_claude_cli


def _fake_result(stdout: str, returncode: int = 0, stderr: str = ""):
    return types.SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_complete_via_claude_cli_parses_result_and_usage(monkeypatch):
    payload = {
        "is_error": False,
        "result": '["a", "b"]',
        "stop_reason": "end_turn",
        "total_cost_usd": 0.05,
        "usage": {
            "input_tokens": 2,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 10,
            "output_tokens": 8,
        },
    }
    captured_cmd = {}

    def fake_run(cmd, **kwargs):
        captured_cmd["cmd"] = cmd
        return _fake_result(json.dumps(payload))

    monkeypatch.setattr(claude_cli_client.subprocess, "run", fake_run)

    result = complete_via_claude_cli("clean this", system="you are a cleaner", model="sonnet")

    assert result == ClaudeCliCompletion(
        content='["a", "b"]', finish_reason=None, prompt_tokens=112, completion_tokens=8, cost_usd=0.05
    )
    cmd = captured_cmd["cmd"]
    assert cmd[:2] == ["claude", "-p"]
    assert "clean this" in cmd
    assert "--append-system-prompt" in cmd
    assert cmd[cmd.index("--append-system-prompt") + 1] == "you are a cleaner"
    assert "--restricted" in cmd
    assert cmd[cmd.index("--model") + 1] == "sonnet"


def test_complete_via_claude_cli_omits_append_system_prompt_when_no_system(monkeypatch):
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        return _fake_result(json.dumps({"is_error": False, "result": "ok", "usage": {}}))

    monkeypatch.setattr(claude_cli_client.subprocess, "run", fake_run)

    complete_via_claude_cli("p", model="sonnet")
    assert "--append-system-prompt" not in calls["cmd"]


def test_complete_via_claude_cli_reports_max_tokens_stop_as_length_finish_reason(monkeypatch):
    payload = {"is_error": False, "result": "cut off", "stop_reason": "max_tokens", "usage": {}}
    monkeypatch.setattr(claude_cli_client.subprocess, "run", lambda cmd, **kwargs: _fake_result(json.dumps(payload)))

    result = complete_via_claude_cli("p", model="sonnet")
    assert result.finish_reason == "length"


def test_complete_via_claude_cli_raises_on_error_payload(monkeypatch):
    payload = {"is_error": True, "result": "permission denied", "subtype": "error"}
    monkeypatch.setattr(claude_cli_client.subprocess, "run", lambda cmd, **kwargs: _fake_result(json.dumps(payload)))

    with pytest.raises(claude_cli_client.LLMEmptyResponseError):
        complete_via_claude_cli("p", model="sonnet")


def test_complete_via_claude_cli_raises_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(
        claude_cli_client.subprocess, "run", lambda cmd, **kwargs: _fake_result("", returncode=1, stderr="boom")
    )

    with pytest.raises(claude_cli_client.LLMEmptyResponseError, match="boom"):
        complete_via_claude_cli("p", model="sonnet")


def test_complete_via_claude_cli_raises_on_non_json_stdout(monkeypatch):
    monkeypatch.setattr(claude_cli_client.subprocess, "run", lambda cmd, **kwargs: _fake_result("not json"))

    with pytest.raises(claude_cli_client.LLMEmptyResponseError):
        complete_via_claude_cli("p", model="sonnet")


def test_complete_via_claude_cli_raises_not_configured_if_claude_binary_missing(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(claude_cli_client.subprocess, "run", fake_run)

    with pytest.raises(claude_cli_client.LLMNotConfiguredError):
        complete_via_claude_cli("p", model="sonnet")


def test_complete_via_claude_cli_raises_on_timeout(monkeypatch):
    import subprocess

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))

    monkeypatch.setattr(claude_cli_client.subprocess, "run", fake_run)

    with pytest.raises(claude_cli_client.LLMEmptyResponseError):
        complete_via_claude_cli("p", model="sonnet", timeout=1.0)
