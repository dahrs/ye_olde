"""Tiny incremental JSON checkpoint that makes the billable (LLM-calling)
stages of ingestion resumable. Each unit of work — one LLM call — is keyed
by a stable string; its result is flushed to disk immediately after that
call succeeds. A run interrupted mid-way (a lapsed API budget, as happened
partway through aligning Sir Gawayne, is the expected common case, not an
edge case) picks back up at the next uncomputed key instead of re-paying
for every call already made.

`meta` is a small fingerprint of the inputs/settings a checkpoint was built
under (e.g. unit counts, mode, block size). If a resumed run's `meta`
doesn't match what's on disk, the checkpoint is treated as stale and
discarded rather than silently reused against different inputs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


class Checkpoint:
    def __init__(self, path: Path, meta: dict):
        self.path = path
        self.meta = meta
        self.entries: dict[str, object] = {}
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("meta") == meta:
                self.entries = saved.get("entries", {})

    def get_or_compute(self, key: str, compute: Callable[[], T]) -> T:
        """Returns the cached result for `key`, computing and persisting it
        first if it isn't already there. If `compute()` raises, nothing is
        written — that key stays uncomputed and will be retried on the next
        run rather than being marked done with no result.
        """
        if key in self.entries:
            return self.entries[key]  # type: ignore[return-value]
        result = compute()
        self.entries[key] = result
        self._flush()
        return result

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"meta": self.meta, "entries": self.entries}, ensure_ascii=False), encoding="utf-8"
        )

    def clear(self) -> None:
        self.entries = {}
        if self.path.exists():
            self.path.unlink()
