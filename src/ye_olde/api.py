"""Public entrypoint — spec section 1 (docs/diachronic-translation-pipeline-plan.md).

translate(sentence, lang_A, year_A, lang_B, year_B) -> {sentence, annotations}

Not yet implemented: this stub only establishes the contract. See spec section 9
for the scoped first build task (single-language COHA round trip).
"""

from __future__ import annotations

from typing import TypedDict


class Annotation(TypedDict):
    span: str
    status: str  # "attested" | "loan" | "constructed" | "anachronism-passthrough"
    note: str


class Translation(TypedDict):
    sentence: str
    annotations: list[Annotation]


def translate(
    sentence: str,
    lang_A: str,
    year_A: int,
    lang_B: str,
    year_B: int,
) -> Translation:
    raise NotImplementedError("pipeline not yet implemented — see docs/diachronic-translation-pipeline-plan.md §9")
