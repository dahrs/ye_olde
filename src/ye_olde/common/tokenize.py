"""Shared deterministic word/punctuation tokenizer — spec §3c/§3d.

Used by `ye_olde.ingest.align` (corpus text, at indexing time) and
`ye_olde.classify` (the live input sentence, at query time) so both sides
produce token boundaries the rest of the pipeline can index/match/highlight
consistently — moved here rather than kept only in `ingest/align.py` once a
second top-level package needed the identical function (see CLAUDE.md's
"Avoid duplicate functions").
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)
