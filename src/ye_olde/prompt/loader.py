"""Reads prompt text out of this package's `<namespace>.yaml` files."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import yaml

_PROMPT_DIR = Path(__file__).parent


class PromptNotFoundError(KeyError):
    """No `<namespace>.yaml` file, or no `key` within it — raised rather
    than returning an empty prompt an LLM call would otherwise silently
    run with.
    """


@cache
def _load_namespace(namespace: str) -> dict[str, str]:
    path = _PROMPT_DIR / f"{namespace}.yaml"
    if not path.is_file():
        raise PromptNotFoundError(f"no prompt file for namespace {namespace!r} (expected {path})")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_prompt(namespace: str, key: str) -> str:
    """Returns the prompt stored under `key` in
    `src/ye_olde/prompt/<namespace>.yaml` — `namespace` is the main package
    the prompt belongs to (e.g. `"ingest"`).
    """
    prompts = _load_namespace(namespace)
    if key not in prompts:
        raise PromptNotFoundError(f"no prompt {key!r} in {namespace}.yaml")
    return prompts[key]
