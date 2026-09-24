"""YAML-backed LLM prompt templates, one file per top-level package that
prompts an LLM (`ingest.yaml`, and so on as `generation`/`classify`
grow their own). Prompts are data, not code: kept out of the modules that
use them so wording can be read, diffed, and reviewed without touching
control flow. See CLAUDE.md "Prompts".
"""

from __future__ import annotations

from .loader import PromptNotFoundError, load_prompt

__all__ = ["PromptNotFoundError", "load_prompt"]
