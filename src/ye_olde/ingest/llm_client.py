"""Thin wrapper around `litellm` for the two LLM-generative steps in the
alignment pipeline (clean.py, align.py) — reads the same pluggable backend
config (`LITELLM_MODEL`/`LITELLM_API_KEY`/`LITELLM_API_BASE`) as
`ye_olde.generation`, since it's the same deliberately-not-hardcoded choice
(spec §8), not a separate one for ingestion.
"""

from __future__ import annotations

import json
import re

from ..config import get_settings

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# Running usage/cost totals for this process, across every call_llm_json()
# call regardless of caller (clean.py, align.py) — a script-lifetime total
# is what "how much did this run cost" actually means, so this is
# deliberately a module-level accumulator rather than something threaded
# through every function signature.
_usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}


def get_usage_summary() -> dict:
    return dict(_usage)


def reset_usage() -> None:
    _usage.update(calls=0, prompt_tokens=0, completion_tokens=0, cost_usd=0.0)


class LLMNotConfiguredError(RuntimeError):
    pass


# Providers litellm resolves a model string to when it's a self-hosted,
# run-it-yourself inference backend, as opposed to a commercial hosted API
# — sourced from litellm's own provider vocabulary (`litellm.LlmProviders`,
# cross-checked against `litellm.get_llm_provider`'s actual behavior for
# each), not guessed from the model string ourselves. These are the ones
# where the caller runs the model, so a call has no marginal cost and a
# second, free self-review pass is worth doing.
_LOCAL_INFERENCE_PROVIDERS = {
    "ollama",
    "ollama_chat",
    "vllm",
    "hosted_vllm",
    "lm_studio",
    "llamafile",
    "oobabooga",
    "petals",
    "xinference",
    "triton",
    "docker_model_runner",
}


def is_local_model(model: str | None = None) -> bool:
    """True if litellm resolves the model to a self-hosted/local-inference
    provider (Ollama, vLLM, LM Studio, ...) rather than a commercial hosted
    API.

    Deliberately asks litellm this directly (`get_llm_provider`, which
    parses the model string the same way `litellm.completion` itself will)
    rather than inferring it from our own project config: checking whether
    `LITELLM_API_BASE` happens to be set would be wrong in both directions
    — it can be set for a hosted API behind a corporate proxy/gateway, and
    it can be left unset for a local provider litellm defaults a base URL
    for automatically (e.g. `ollama/*` -> `http://localhost:11434`).
    """
    import litellm

    settings = get_settings()
    resolved_model = model or settings.litellm_model
    if not resolved_model:
        return False
    try:
        _, provider, _, _ = litellm.get_llm_provider(resolved_model, api_base=settings.litellm_api_base or None)
    except Exception:
        return False  # unresolvable model string -> don't assume it's free to call again
    return provider in _LOCAL_INFERENCE_PROVIDERS


def call_llm_json(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
) -> object:
    """Sends `prompt` to the configured generation LLM and parses the reply
    as JSON. If the reply isn't valid JSON, makes exactly one repair call
    (shows the model its own bad output plus the parse error, asks for a
    corrected reply) — not recursive: if the repair attempt also fails to
    parse, that error propagates rather than trying again.
    """
    settings = get_settings()
    resolved_model = model or settings.litellm_model
    if not resolved_model:
        raise LLMNotConfiguredError(
            "LITELLM_MODEL is not set — copy .env.example to .env and fill it in"
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    content = _complete(resolved_model, messages, settings)
    try:
        return _parse_json(content)
    except (json.JSONDecodeError, ValueError) as exc:
        messages.append({"role": "assistant", "content": content})
        messages.append(
            {
                "role": "user",
                "content": (
                    "That was not valid JSON and could not be parsed "
                    f"({exc}). Reply again with ONLY the corrected JSON, "
                    "no prose, no code fences."
                ),
            }
        )
        repaired_content = _complete(resolved_model, messages, settings)
        return _parse_json(repaired_content)  # a second failure here is not caught/retried again


def _complete(resolved_model: str, messages: list[dict], settings) -> str:
    import litellm

    response = litellm.completion(
        model=resolved_model,
        messages=messages,
        api_key=settings.litellm_api_key or None,
        api_base=settings.litellm_api_base or None,
    )
    _record_usage(response)
    return response["choices"][0]["message"]["content"] or ""


def _parse_json(content: str) -> object:
    cleaned = _JSON_FENCE_RE.sub("", content.strip()).strip()
    return json.loads(cleaned)


def _record_usage(response) -> None:
    import litellm

    _usage["calls"] += 1
    usage = getattr(response, "usage", None)
    if usage is not None:
        _usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        _usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
    try:
        _usage["cost_usd"] += litellm.completion_cost(completion_response=response)
    except Exception:
        pass  # unpriced/unknown model in litellm's cost map — token counts above still tell the story
