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
context window before producing an actual answer. That surfaces two ways —
`message.content` comes back completely empty, or (observed running a real
Qwen3.5 chunk against a 4096-token local context) non-empty but cut off
mid-answer, reported as `finish_reason="length"`. `call_llm_json` treats
both as the same condition and handles it with exactly one retry of the
same call using `LITELLM_NO_THINKING_EXTRA_BODY` (also config.py) instead —
a one-shot rescue for that call alone. It does not change
`LITELLM_EXTRA_BODY` or affect any later call: reasoning is back on for the
next call regardless of whether this one needed the fallback. Re-sending
the *same* prompt through the ordinary "fix your JSON" repair path instead
would very likely hit the identical token budget and truncate again in
roughly the same place — that repair path is for a genuine formatting
mistake, not a budget problem, so truncation is intercepted before ever
reaching it.

Every call also passes an explicit `timeout` from `LITELLM_TIMEOUT_SECONDS`
(config.py, default 6000s) rather than trusting litellm's own implicit
default for this code path — that turned out in practice to be far shorter
than an uncapped local reasoning call can need.

A third backend, alongside "hosted API" and "local inference server" (both
served through `litellm.completion` below): `claude_cli_client.py`, a
standalone sibling module that shells out to the real `claude` CLI so calls
can draw on a Claude Pro/Max subscription instead of a separate
`ANTHROPIC_API_KEY` — see that module's docstring for what it is and why.
This module only adapts it to fit here: `LITELLM_MODEL=claude_code_cli/
<alias-or-full-name>` (e.g. `claude_code_cli/claude-sonnet-5` — a full model
name pins it exactly, the same guarantee `anthropic/claude-sonnet-5` gives
the hosted-API backend; a bare alias like `claude_code_cli/sonnet` tracks
whatever Claude Code currently calls "sonnet" instead, which can shift
later without anything here changing) is this module's own convention for
picking that backend, recognized by `_is_claude_cli_model`/`_complete`
before anything litellm-specific (provider resolution, API-key lookup) runs
— none of that applies to a CLI invocation. `_render_cli_messages` is the
other half of the adaptation: `claude_cli_client.complete_via_claude_cli`
takes a plain `(prompt, system)` pair, but this module's internal
`messages` list can grow a repair-retry's extra turns (the model's own bad
reply, a follow-up fix-it request) — `claude -p` has no conversation to
append those to (a fresh, stateless process each call), so they get
flattened into one prompt string here before the call.
"""

from __future__ import annotations

import ipaddress
import json
import re
from typing import Any
from urllib.parse import urlparse

from ..common.logging import get_logger
from ..config import Settings, get_settings
from .claude_cli_client import complete_via_claude_cli
from .errors import LLMEmptyResponseError, LLMNotConfiguredError

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


def _is_loopback_or_private_host(url: str) -> bool:
    try:
        host = urlparse(url).hostname
    except ValueError:
        return False
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False  # a real hostname (not an IP/localhost) -> not treated as local
    return addr.is_loopback or addr.is_private


def _resolve_provider(resolved_model: str, settings: Settings) -> tuple[str, str | None] | None:
    """Provider name and litellm-resolved `api_base` for `resolved_model`
    (asked of litellm directly via `get_llm_provider`, which parses the
    model string the same way `litellm.completion` itself will, rather
    than inferred from our own project config), or `None` if the model
    string doesn't resolve. Shared by `is_local_model` and
    `_resolve_api_key`, which both need to know which backend a model
    string actually routes to.
    """
    import litellm

    try:
        _, provider, _, resolved_api_base = litellm.get_llm_provider(
            resolved_model, api_base=settings.litellm_api_base or None
        )
    except Exception as exc:
        # Unresolvable model string -> callers treat this as "unknown/not
        # local" / "use the default key" rather than guessing. Logged (not
        # silent) since this changes downstream behavior even though it's
        # an expected, handled case.
        _log.debug("could not resolve provider for %r: %s", resolved_model, exc)
        return None
    return provider, resolved_api_base


def is_local_model(model: str | None = None) -> bool:
    """True if this call is going to a self-hosted/local-inference backend
    rather than a commercial hosted API, checked two independent ways:

    1. litellm resolves the model string to a known local-inference
       provider (Ollama, vLLM, LM Studio, ...).
    2. The resolved `api_base` points at a loopback/private address. This
       catches the llama.cpp setup this README documents: `llama-server`
       is an OpenAI-*compatible* server, so its recommended
       `LITELLM_MODEL=openai/<any-name>` resolves to provider "openai" —
       indistinguishable from the real hosted OpenAI API by provider name
       alone. Checking whether the request is actually going to a private
       address closes that gap.

    Neither check alone is sufficient: `LITELLM_API_BASE` being set can
    also mean a hosted API behind a corporate proxy/gateway (not local),
    and a local provider can leave it unset since litellm defaults one
    automatically (e.g. `ollama/*` -> `http://localhost:11434`, which is
    why check 2 uses litellm's *resolved* api_base, not the raw setting).
    """
    settings = get_settings()
    resolved_model = model or settings.litellm_model
    if not resolved_model:
        return False
    resolved = _resolve_provider(resolved_model, settings)
    if resolved is None:
        return False
    provider, resolved_api_base = resolved
    if provider in _LOCAL_INFERENCE_PROVIDERS:
        return True
    return resolved_api_base is not None and _is_loopback_or_private_host(resolved_api_base)


def _resolve_api_key(resolved_model: str, settings: Settings) -> str | None:
    """The credential to send for `resolved_model` — `settings.aws_bearer_token_bedrock`
    when this call is going to AWS Bedrock's Mantle endpoint
    (`LITELLM_MODEL=bedrock_mantle/openai.<model>`, e.g. GPT-6 Sol),
    `settings.gemini_api_key` when it's going to Google's Gemini API
    (`LITELLM_MODEL=gemini/<model>`, e.g. `gemini/gemini-3.8-flash`), or
    `settings.litellm_api_key` (today's default backend's credential, e.g.
    Anthropic's) otherwise. This is what lets all three keys stay
    configured side by side in `.env` and be switched between purely via
    which provider `LITELLM_MODEL`/`--model` resolves to. Unlike the
    "openai" provider name (which is ambiguous with a local llama-server's
    own OpenAI-compatible masquerade — see `is_local_model`), neither
    "bedrock_mantle" nor "gemini" is ever ambiguous with a local/self-hosted
    endpoint, so no loopback/local check is needed for either here.
    """
    if not settings.aws_bearer_token_bedrock and not settings.gemini_api_key:
        return settings.litellm_api_key or None
    resolved = _resolve_provider(resolved_model, settings)
    if resolved is None:
        return settings.litellm_api_key or None
    provider, _resolved_api_base = resolved
    if provider == "bedrock_mantle" and settings.aws_bearer_token_bedrock:
        return settings.aws_bearer_token_bedrock
    if provider == "gemini" and settings.gemini_api_key:
        return settings.gemini_api_key
    return settings.litellm_api_key or None


def call_llm_json(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
) -> object:
    """Sends `prompt` to the configured generation LLM and parses the reply
    as JSON.

    Two independent retry paths, each exactly one attempt (neither is
    recursive), applied at *every* completion this function makes — the
    initial call and the JSON-repair retry below both go through
    `_complete_with_truncation_rescue`, since the repair conversation
    (original prompt + bad output + a fix-it instruction) is longer than
    the original and therefore, if anything, more likely to hit the same
    token-budget wall, not less; confirmed in practice when a repair call
    itself came back completely empty and the earlier version of this
    function (which only rescued the initial call) surfaced a confusing
    generic JSONDecodeError instead of the real cause:
    - Empty or truncated content (a reasoning trace ran the context
      window out before producing a complete answer — empty if it consumed
      the whole budget, truncated/`finish_reason="length"` if it left a
      partial answer): retried once with LITELLM_NO_THINKING_EXTRA_BODY in
      place of LITELLM_EXTRA_BODY, if configured — see config.py and the
      module docstring. If that retry is also empty or truncated, or no
      fallback is configured, this raises rather than guessing further.
    - Invalid JSON (content was neither empty nor truncated — a genuine
      formatting mistake, not a budget problem): retried once by showing
      the model its own bad output plus the parse error and asking for a
      corrected reply. A second failure here propagates rather than trying
      again.
    """
    settings = get_settings()
    resolved_model = model or settings.litellm_model
    if not resolved_model:
        raise LLMNotConfiguredError("LITELLM_MODEL is not set — copy .env.example to .env and fill it in")

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    content = _complete_with_truncation_rescue(resolved_model, messages, settings)
    try:
        return _parse_json(content)
    except json.JSONDecodeError as exc:
        if _looks_like_truncation(content, exc):
            return _parse_json(_rescue_truncation(resolved_model, messages, settings))

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
        repaired_content = _complete_with_truncation_rescue(resolved_model, messages, settings)
        try:
            return _parse_json(repaired_content)
        except json.JSONDecodeError as exc2:
            # a second failure here is not caught/retried again — except
            # truncation gets one more rescue even at this point, since
            # nothing about round 2 makes it less likely than round 1
            if _looks_like_truncation(repaired_content, exc2):
                return _parse_json(_rescue_truncation(resolved_model, messages, settings))
            raise


def _complete_with_truncation_rescue(resolved_model: str, messages: list[dict[str, str]], settings: Settings) -> str:
    """One completion, with a single automatic rescue (`_rescue_truncation`)
    if the reply is empty or was cut off mid-answer (`finish_reason="length"`)
    — see `call_llm_json`'s docstring. Shared by both completions it makes
    so neither is exempt from this check.
    """
    content, finish_reason = _complete(resolved_model, messages, settings, _default_extra_body(settings))
    if not content.strip() or finish_reason == "length":
        return _rescue_truncation(resolved_model, messages, settings)
    return content


def _rescue_truncation(resolved_model: str, messages: list[dict[str, str]], settings: Settings) -> str:
    """The shared last resort for anything that looks like the previous
    completion ran out of token budget before finishing: empty content,
    `finish_reason="length"`, or — a corroborating signal that proved
    necessary in practice, since `finish_reason` alone has been observed
    under-reporting truncation on this backend — a JSON parse failure whose
    error position sits at the very end of the content (see
    `_looks_like_truncation`). One completion using
    `LITELLM_NO_THINKING_EXTRA_BODY` in place of the default extra body;
    raises if that's unconfigured or also comes back empty/truncated.
    """
    fallback_body = _no_thinking_extra_body(settings)
    if fallback_body is None:
        raise LLMEmptyResponseError(
            "model response was empty or looked truncated — likely a reasoning trace that "
            "filled the context window before completing an answer. Set "
            "LITELLM_NO_THINKING_EXTRA_BODY to enable a one-shot retry without reasoning for "
            "cases like this."
        )
    content, finish_reason = _complete(resolved_model, messages, settings, fallback_body)
    if not content.strip() or finish_reason == "length":
        raise LLMEmptyResponseError(
            "model response was empty or truncated even with LITELLM_NO_THINKING_EXTRA_BODY applied "
            "— not a reasoning-budget issue, something else is wrong."
        )
    return content


def _looks_like_truncation(content: str, exc: json.JSONDecodeError) -> bool:
    """True if a JSON parse failure looks like the model stopped generating
    before finishing a complete value — the same underlying problem
    `finish_reason="length"` is meant to report, but observed in practice
    (a real local-model run) to sometimes go unreported as such (the API
    said `finish_reason="stop"` on content that was still clearly cut off
    mid-array). Two checks, since one error position convention doesn't
    cover both ways this shows up:
    - "Unterminated string starting at" is raised *only* when the parser
      reaches the end of the entire input while still inside a string with
      no closing quote — never for a mid-string syntax mistake elsewhere in
      an otherwise-complete reply — so this message alone is a reliable
      signal regardless of position.
    - Every other JSONDecodeError reports `.pos` as where the *problem*
      is, which for "Expecting value"/"Expecting ',' delimiter" etc. is
      only truncation if that position is at (or right at) the end of the
      content — the same error can also legitimately occur mid-string for
      a genuine formatting mistake, so position is what distinguishes them.
    """
    if "Unterminated string" in exc.msg:
        return True
    cleaned = _strip_json_fences(content)
    return exc.pos >= len(cleaned) - 1


_CLAUDE_CLI_PREFIX = "claude_code_cli/"


def _is_claude_cli_model(resolved_model: str) -> bool:
    return resolved_model.startswith(_CLAUDE_CLI_PREFIX)


def _complete(
    resolved_model: str, messages: list[dict[str, str]], settings: Settings, extra_body: dict[str, Any] | None
) -> tuple[str, str | None]:
    if _is_claude_cli_model(resolved_model):
        # extra_body (LITELLM_EXTRA_BODY / the no-thinking rescue fallback)
        # has no CLI equivalent -- silently unused for this backend, not an
        # error, since Claude Code's own adaptive thinking already handles
        # the failure mode that setting exists to work around for litellm
        # backends.
        cli_model = resolved_model[len(_CLAUDE_CLI_PREFIX) :] or "claude-sonnet-5"
        system_text, prompt_text = _render_cli_messages(messages)
        result = complete_via_claude_cli(
            prompt_text, system=system_text, model=cli_model, timeout=settings.litellm_timeout_seconds
        )
        _usage["calls"] += 1
        _usage["prompt_tokens"] += result.prompt_tokens
        _usage["completion_tokens"] += result.completion_tokens
        _usage["cost_usd"] += result.cost_usd
        return result.content, result.finish_reason

    import litellm

    response = litellm.completion(
        model=resolved_model,
        messages=messages,
        api_key=_resolve_api_key(resolved_model, settings),
        api_base=settings.litellm_api_base or None,
        extra_body=extra_body,
        timeout=settings.litellm_timeout_seconds,
    )
    _record_usage(response)
    choice = response["choices"][0]
    content: str = choice["message"]["content"] or ""
    finish_reason: str | None = choice.get("finish_reason")
    return content, finish_reason


def _render_cli_messages(messages: list[dict[str, str]]) -> tuple[str | None, str]:
    """Splits a litellm-style `messages` list into `(system_text, prompt_text)`
    for `claude -p`. The system message, if present, is always `messages[0]`
    (see `call_llm_json`'s construction) and goes to `--append-system-prompt`
    — a stable prefix, byte-identical call to call, so it stays eligible for
    Anthropic's server-side prompt cache (see this module's docstring).

    Every remaining message is flattened into one prompt string: the normal
    case is just the one user prompt, but `call_llm_json`'s JSON-repair path
    appends an assistant turn (the bad reply) and a follow-up user turn (the
    fix-it request) to the *same* `messages` list before calling this again
    — `claude -p` is a fresh, stateless process each call with no
    conversation of its own to append those to, so they're rendered into the
    prompt text instead.
    """
    system_text = None
    rest = messages
    if messages and messages[0]["role"] == "system":
        system_text = messages[0]["content"]
        rest = messages[1:]
    if len(rest) == 1:
        return system_text, rest[0]["content"]
    parts = [
        f"Your previous reply:\n{msg['content']}" if msg["role"] == "assistant" else msg["content"] for msg in rest
    ]
    return system_text, "\n\n---\n\n".join(parts)


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


_LENIENT_JSON_DECODER = json.JSONDecoder(strict=False)
_LEADING_WHITESPACE_RE = re.compile(r"[ \t\n\r]*")


def _strip_json_fences(content: str) -> str:
    return _JSON_FENCE_RE.sub("", content.strip()).strip()


def _parse_json(content: str) -> object:
    cleaned = _strip_json_fences(content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Two failure modes observed from a real local model (Qwen3.5 via
        # llama.cpp), both recoverable without spending another billable/
        # slow LLM repair call — the data itself was fine, strict json.loads
        # was just pickier than necessary:
        # - "Invalid control character": a literal control character
        #   (almost always a raw newline from source verse/prose text)
        #   embedded in a JSON string instead of escaped as \n.
        # - "Extra data": a complete, valid JSON value followed by trailing
        #   content the model appended anyway (commentary, a repeated/
        #   duplicated echo) despite being told to reply with only JSON.
        # strict=False tolerates the first; raw_decode (rather than loads,
        # which demands the *entire* string be one value) tolerates the
        # second by parsing just the first complete value and discarding
        # whatever follows it. Re-raises the *original* error unchanged if
        # the content is broken some other way this can't fix either.
        try:
            leading_whitespace = _LEADING_WHITESPACE_RE.match(cleaned)
            start = leading_whitespace.end() if leading_whitespace else 0
            obj, _end = _LENIENT_JSON_DECODER.raw_decode(cleaned, start)
        except json.JSONDecodeError:
            raise exc from None
        return obj


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
