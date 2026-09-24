"""Public entrypoint — spec section 1 (docs/diachronic-translation-pipeline-plan.md).

translate(sentence, lang_a, year_a, lang_b, year_b) -> Translation

Not yet implemented: this stub only establishes the contract. See spec section 9
for the scoped first build task (single-language COHA round trip).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Annotation(BaseModel):
    span: str
    status: Literal["attested", "loan", "constructed", "anachronism-passthrough"]
    note: str


class Translation(BaseModel):
    sentence: str
    annotations: list[Annotation]


def translate(
    sentence: str,
    lang_a: str,
    year_a: int,
    lang_b: str,
    year_b: int,
) -> Translation:
    raise NotImplementedError("pipeline not yet implemented — see docs/diachronic-translation-pipeline-plan.md §9")
