"""Unit tests for llm_client's usage/cost tracking. `litellm` itself is
faked out via sys.modules so these run with no network access and no real
API key.
"""

from __future__ import annotations

import sys
import types

import pytest

from ye_olde.ingest import llm_client


class _FakeUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    def __init__(self, content: str, prompt_tokens: int, completion_tokens: int):
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)
        self.choices = [{"message": {"content": content}}]

    def __getitem__(self, key):
        return getattr(self, key)


@pytest.fixture(autouse=True)
def _reset_usage():
    llm_client.reset_usage()
    yield
    llm_client.reset_usage()


def test_call_llm_json_tracks_usage_and_cost(monkeypatch):
    response = _FakeResponse('["a", "b"]', prompt_tokens=100, completion_tokens=20)
    fake_litellm = types.SimpleNamespace(
        completion=lambda **kwargs: response,
        completion_cost=lambda completion_response: 0.0042,
    )
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    result = llm_client.call_llm_json("prompt", model="test-model")
    assert result == ["a", "b"]

    usage = llm_client.get_usage_summary()
    assert usage == {"calls": 1, "prompt_tokens": 100, "completion_tokens": 20, "cost_usd": pytest.approx(0.0042)}


def test_usage_accumulates_across_calls(monkeypatch):
    fake_litellm = types.SimpleNamespace(
        completion=lambda **kwargs: _FakeResponse("[]", prompt_tokens=10, completion_tokens=1),
        completion_cost=lambda completion_response: 0.001,
    )
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    llm_client.call_llm_json("p1", model="test-model")
    llm_client.call_llm_json("p2", model="test-model")

    usage = llm_client.get_usage_summary()
    assert usage["calls"] == 2
    assert usage["prompt_tokens"] == 20
    assert usage["cost_usd"] == pytest.approx(0.002)


def test_unpriced_model_still_tracks_tokens_with_zero_cost(monkeypatch):
    def raise_unknown_pricing(completion_response):
        raise Exception("unknown model pricing")

    fake_litellm = types.SimpleNamespace(
        completion=lambda **kwargs: _FakeResponse("[]", prompt_tokens=5, completion_tokens=2),
        completion_cost=raise_unknown_pricing,
    )
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    llm_client.call_llm_json("p", model="test-model")

    usage = llm_client.get_usage_summary()
    assert usage["prompt_tokens"] == 5
    assert usage["cost_usd"] == 0.0


def test_bad_json_triggers_exactly_one_repair_call(monkeypatch):
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs["messages"])
        return _FakeResponse("not json", prompt_tokens=7, completion_tokens=3)

    fake_litellm = types.SimpleNamespace(completion=fake_completion, completion_cost=lambda **k: 0.0001)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    with pytest.raises(ValueError):
        llm_client.call_llm_json("p", model="test-model")

    # exactly 2 calls total: the original + one repair attempt, not recursive
    assert len(calls) == 2
    usage = llm_client.get_usage_summary()
    assert usage["calls"] == 2
    assert usage["cost_usd"] == pytest.approx(0.0002)


def test_repair_call_recovers_from_bad_first_reply(monkeypatch):
    responses = iter(["not json at all", '["fixed", "output"]'])

    fake_litellm = types.SimpleNamespace(
        completion=lambda **kwargs: _FakeResponse(next(responses), prompt_tokens=1, completion_tokens=1),
        completion_cost=lambda **k: 0.0,
    )
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    result = llm_client.call_llm_json("p", model="test-model")
    assert result == ["fixed", "output"]
    assert llm_client.get_usage_summary()["calls"] == 2


def test_is_local_model_true_for_known_local_providers(monkeypatch):
    from ye_olde.config import Settings
    from ye_olde.ingest import llm_client as lc

    monkeypatch.setattr(lc, "get_settings", lambda: Settings())
    for model in ["ollama/llama3", "ollama_chat/llama3", "vllm/mistral", "lm_studio/model", "llamafile/model"]:
        assert lc.is_local_model(model) is True, model


def test_is_local_model_false_for_hosted_api(monkeypatch):
    from ye_olde.config import Settings
    from ye_olde.ingest import llm_client as lc

    monkeypatch.setattr(lc, "get_settings", lambda: Settings(litellm_model="anthropic/claude-sonnet-5"))
    assert lc.is_local_model() is False
    assert lc.is_local_model("openai/gpt-4") is False


def test_is_local_model_false_for_hosted_api_behind_a_custom_base_url(monkeypatch):
    """A custom LITELLM_API_BASE doesn't by itself mean local — e.g. a
    corporate proxy or gateway in front of a paid hosted API.
    """
    from ye_olde.config import Settings
    from ye_olde.ingest import llm_client as lc

    monkeypatch.setattr(
        lc,
        "get_settings",
        lambda: Settings(litellm_model="anthropic/claude-sonnet-5", litellm_api_base="https://proxy.example.com"),
    )
    assert lc.is_local_model() is False


def test_is_local_model_false_for_unresolvable_model(monkeypatch):
    from ye_olde.config import Settings
    from ye_olde.ingest import llm_client as lc

    monkeypatch.setattr(lc, "get_settings", lambda: Settings(litellm_model=""))
    assert lc.is_local_model() is False
