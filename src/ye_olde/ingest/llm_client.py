"""Thin wrapper around `litellm` for the two LLM-generative steps in the
alignment pipeline (clean.py, align.py) — reads the same pluggable backend
config (`LITELLM_MODEL`/`LITELLM_API_KEY`/`LITELLM_API_BASE`) as
`ye_olde.generation`, since it's the same deliberately-not-hardcoded choice
(spec §8), not a separate one for ingestion.

Also reads `LITELLM_EXTRA_BODY` (raw JSON, see config.py) and forwards it as
litellm's `extra_body` on every call — left blank by default, which for a
reasoning-capable backend means "thinking" stays on, deliberately: it can
genuinely help this module's close-reading/alignment judgment calls.

The risk with that is a reasoning trace long enough to consume the whole
context window before producing an actual answer, which comes back as an
empty `message.content`. `call_llm_json` handles that with exactly one
retry of the same call using `LITELLM_NO_THINKING_EXTRA_BODY` (also
config.py) instead — a one-shot rescue for that call alone. It does not
change `LITELLM_EXTRA_BODY` or affect any later call: reasoning is back on
for the next call regardless of whether this one needed the fallback.

Every call also passes an explicit `timeout` from `LITELLM_TIMEOUT_SECONDS`
(config.py, default 6000s) rather than trusting litellm's own implicit
default for this code path — that turned out in practice to be far shorter
than an uncapped local reasoning call can need.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..common.errors import LLMEmptyResponseError, LLMNotConfiguredError
from ..common.logging import get_logger
from ..config import Settings, get_settings

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_log = get_logger(__name__)

# Running usage/cost totals for this process, across every call_llm_json()
# call regardless of caller (clean.py, align.py) — a script-lifetime total
# is what "how much did this run cost" actually means, so this is
# deliberately a module-level accumulator rather than something threaded
# through every function signature.
_usage: dict[str, int | float] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}


def get_usage_summary() -> dict[str, int | float]:
    return dict(_usage)


def reset_usage() -> None:
    _usage.update(calls=0, prompt_tokens=0, completion_tokens=0, cost_usd=0.0)


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
    except Exception as exc:
        # Unresolvable model string -> don't assume it's free to call again.
        # Logged (not silent) since this changes downstream behavior (no
        # review pass) even though it's an expected, handled case.
        _log.debug("could not resolve provider for %r: %s", resolved_model, exc)
        return False
    return provider in _LOCAL_INFERENCE_PROVIDERS


def call_llm_json(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
) -> object:
    """Sends `prompt` to the configured generation LLM and parses the reply
    as JSON.

    Two independent retry paths, each exactly one attempt (neither is
    recursive):
    - Empty `content` (a reasoning trace ran the context window out before
      producing an answer): retried once with LITELLM_NO_THINKING_EXTRA_BODY
      in place of LITELLM_EXTRA_BODY, if configured — see config.py and the
      module docstring. If that retry is also empty, or no fallback is
      configured, this raises rather than guessing further.
    - Invalid JSON: retried once by showing the model its own bad output
      plus the parse error and asking for a corrected reply. A second
      failure here propagates rather than trying again.
    """
    settings = get_settings()
    resolved_model = model or settings.litellm_model
    if not resolved_model:
        raise LLMNotConfiguredError("LITELLM_MODEL is not set — copy .env.example to .env and fill it in")

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    content = _complete(resolved_model, messages, settings, _default_extra_body(settings))
    if not content.strip():
        fallback_body = _no_thinking_extra_body(settings)
        if fallback_body is None:
            raise LLMEmptyResponseError(
                "model returned empty content — likely a reasoning trace that filled the "
                "context window before any answer. Set LITELLM_NO_THINKING_EXTRA_BODY to "
                "enable a one-shot retry without reasoning for cases like this."
            )
        content = _complete(resolved_model, messages, settings, fallback_body)
        if not content.strip():
            raise LLMEmptyResponseError(
                "model returned empty content even with LITELLM_NO_THINKING_EXTRA_BODY applied "
                "— not a reasoning-budget issue, something else is wrong."
            )

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
        repaired_content = _complete(resolved_model, messages, settings, _default_extra_body(settings))
        return _parse_json(repaired_content)  # a second failure here is not caught/retried again


def _complete(
    resolved_model: str, messages: list[dict[str, str]], settings: Settings, extra_body: dict[str, Any] | None
) -> str:
    import litellm

    response = litellm.completion(
        model=resolved_model,
        messages=messages,
        api_key=settings.litellm_api_key or None,
        api_base=settings.litellm_api_base or None,
        extra_body=extra_body,
        timeout=settings.litellm_timeout_seconds,
    )
    _record_usage(response)
    return response["choices"][0]["message"]["content"] or ""


def _default_extra_body(settings: Settings) -> dict[str, Any] | None:
    return _parse_extra_body(settings.litellm_extra_body, "LITELLM_EXTRA_BODY")


def _no_thinking_extra_body(settings: Settings) -> dict[str, Any] | None:
    if not settings.litellm_no_thinking_extra_body:
        return None
    return _parse_extra_body(settings.litellm_no_thinking_extra_body, "LITELLM_NO_THINKING_EXTRA_BODY")


def _parse_extra_body(raw: str, var_name: str) -> dict[str, Any] | None:
    """A malformed value raises immediately rather than silently sending
    nothing — this is a setup mistake to fix, not a transient/model-output
    problem worth retrying past.
    """
    if not raw:
        return None
    try:
        parsed: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMNotConfiguredError(f"{var_name} is not valid JSON: {exc}") from exc
    return parsed


def _parse_json(content: str) -> object:
    cleaned = _JSON_FENCE_RE.sub("", content.strip()).strip()
    return json.loads(cleaned)


def _record_usage(response: Any) -> None:
    import litellm

    _usage["calls"] += 1
    usage = getattr(response, "usage", None)
    if usage is not None:
        _usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        _usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
    try:
        _usage["cost_usd"] += litellm.completion_cost(completion_response=response)
    except Exception as exc:
        # Unpriced/unknown model in litellm's cost map — token counts above
        # still tell the story, so this doesn't fail the call, but it's
        # logged rather than silently dropped.
        _log.debug("could not price this call for cost tracking: %s", exc)
