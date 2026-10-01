"""A third way to call an LLM, alongside `llm_client.py`'s hosted-API and
local-inference-server paths (both served through `litellm.completion`):
this module shells out to the real `claude` CLI in `-p`/print (non-
interactive) mode instead. Standalone and independently importable —
`llm_client.call_llm_json` uses it (see that module's `_complete` dispatch)
but nothing here depends on litellm, on `ye_olde.config.Settings`, or on
`llm_client.py`'s internals; any script can call `complete_via_claude_cli`
directly.

**What this is for**: reusing a Claude Pro/Max subscription (`claude
login`'s OAuth session) as a batch LLM backend, instead of a separate
pay-per-token `ANTHROPIC_API_KEY`.

**Decisions and caveats, made explicit rather than left implicit:**
  - **Shares your interactive usage, on purpose.** `claude -p` with no
    `ANTHROPIC_API_KEY` set authenticates via the same OAuth session as
    interactive Claude Code, drawing from the same session/weekly quota —
    a large ingestion run competes with your own coding sessions for it.
    Chosen anyway (a project decision, not a default to reconsider here):
    treat this backend as an option for a contributor's own small/dev-scale
    runs, not the production ingestion path (that stays on the hosted-API
    backend in `llm_client.py`).
  - **Prompt caching works across separate processes for free — no session
    needs to stay open.** Anthropic's prompt cache is keyed by matching
    content prefix within a TTL window, server-side — not by OS process or
    CLI session identity. Confirmed empirically: two cold, unrelated
    `claude -p` invocations already showed a `cache_read_input_tokens` hit
    on Claude Code's own built-in system prompt. `complete_via_claude_cli`
    passes `system` via `--append-system-prompt` verbatim, so a caller that
    keeps its own system text byte-identical across calls gets that same
    caching for free. Every invocation stays a fresh, stateless process (no
    `--continue`/`--resume`) deliberately — resuming a session would resend
    and grow the actual conversation each call, the token-count increase
    this design avoids.
  - **`--restricted` and a neutral `cwd`, not `--bare`.** `--bare` would
    disable CLAUDE.md/settings auto-discovery cleanly, but it also forces
    API-key-only auth (explicitly disables OAuth/keychain reads) — exactly
    the auth path this module exists to avoid. Instead: `--restricted` (no
    Bash/code-exec/WebFetch tools, ignores project/user/local settings
    files) plus running from a neutral temp directory (so whatever repo the
    caller lives in doesn't have its own CLAUDE.md auto-discovered and
    bleed into the prompt) and `--disable-slash-commands` (no skill
    auto-triggering). This narrows the gap to a plain API call but doesn't
    close it entirely — hooks and hosted MCP config still aren't suppressed
    the way `--bare` would, so a call through this backend is not
    behaviorally identical to the same prompt sent via `litellm.completion`.
  - **Not a substitute for the Claude Agent SDK, and not upgraded to it.**
    The `claude-agent-sdk` Python package is a different tool for a
    different purpose here — Anthropic's own docs state it requires
    `ANTHROPIC_API_KEY` auth, not a local OAuth/subscription session — so
    it doesn't achieve what this module exists for. Shelling out to the
    real `claude` CLI is the only path that reuses subscription auth, not
    a stopgap chosen over a nicer SDK.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from typing import Any

from pydantic import BaseModel

from .errors import LLMEmptyResponseError, LLMNotConfiguredError

_DEFAULT_TIMEOUT_SECONDS = 6000.0


class ClaudeCliCompletion(BaseModel):
    """`complete_via_claude_cli`'s return value — a public function's return
    value, so this is a `BaseModel` rather than a bare tuple/dict per
    CLAUDE.md's Pydantic guidance.
    """

    content: str
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


def complete_via_claude_cli(
    prompt: str,
    *,
    system: str | None = None,
    model: str = "claude-sonnet-5",
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> ClaudeCliCompletion:
    """Runs `claude -p --output-format json` once and returns the parsed
    result. `model` is passed straight to `claude -p --model`, so any
    alias/full model name the CLI itself accepts works (`"sonnet"`,
    `"opus"`, a full model ID, ...).

    Raises `LLMNotConfiguredError` if the `claude` binary isn't on PATH,
    `LLMEmptyResponseError` for anything else that went wrong (nonzero
    exit, a timeout, non-JSON stdout, or `is_error: true` in the parsed
    reply) — the same exception types `llm_client.py`'s litellm-backed path
    raises for equivalent failures, so callers of either backend can catch
    one pair of types.
    """
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", model]
    cmd += ["--restricted", "--disable-slash-commands"]
    if system:
        cmd += ["--append-system-prompt", system]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            # A neutral cwd -- not the caller's own repo -- so CLAUDE.md/
            # skill auto-discovery has nothing project-specific to pick up
            # and bleed into the prompt (see module docstring).
            cwd=tempfile.gettempdir(),
        )
    except FileNotFoundError as exc:
        raise LLMNotConfiguredError("the `claude` binary isn't on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise LLMEmptyResponseError(f"claude -p did not finish within {timeout}s") from exc

    if proc.returncode != 0:
        raise LLMEmptyResponseError(f"claude -p exited {proc.returncode}: {(proc.stderr or '').strip()[:500]}")

    try:
        payload: dict[str, Any] = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise LLMEmptyResponseError(f"claude -p produced non-JSON --output-format=json output: {exc}") from exc

    if payload.get("is_error"):
        raise LLMEmptyResponseError(f"claude -p reported an error: {payload.get('result') or payload.get('subtype')}")

    return _to_completion(payload)


def _to_completion(payload: dict[str, Any]) -> ClaudeCliCompletion:
    """`prompt_tokens` sums all three input-side counters `claude -p`
    reports (raw, cache-read, cache-creation) — most of a well-cached run's
    input tokens are the heavily-discounted cache-read kind, and
    undercounting by reporting only the uncached remainder would make this
    figure misleading rather than merely approximate.
    """
    usage = payload.get("usage") or {}
    prompt_tokens = (
        int(usage.get("input_tokens") or 0)
        + int(usage.get("cache_read_input_tokens") or 0)
        + int(usage.get("cache_creation_input_tokens") or 0)
    )
    finish_reason = "length" if payload.get("stop_reason") == "max_tokens" else None
    return ClaudeCliCompletion(
        content=str(payload.get("result") or ""),
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=int(usage.get("output_tokens") or 0),
        cost_usd=float(payload.get("total_cost_usd") or 0.0),
    )
