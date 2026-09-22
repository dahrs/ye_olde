"""FastAPI app — spec §10 endpoints.

Field-mapping note: the /lookup handler below assumes flattened Parquet
columns (source_tokens, target_tokens, alignment_links, ...) for the §3c
aligned-pair record. The exact column layout the ingestion job (§6) actually
writes hasn't been finalized yet — this is provisional and will need
reconciling once real pair shards exist.
"""

from __future__ import annotations

from fastapi import FastAPI

from . import loader
from .config import get_settings
from .schemas import (
    AttestResponse,
    AttestResult,
    HealthResponse,
    HighlightedSpan,
    LookupResponse,
    LookupResult,
)

app = FastAPI(title="ye_olde Search API")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", hf_dataset_repo_id=settings.hf_dataset_repo_id)


@app.get("/attest", response_model=AttestResponse)
def attest(lemma: str, lang: str, year: int, window: int = 50) -> AttestResponse:
    table = loader.load_relational(lang, year - window, year + window)
    results: list[AttestResult] = []
    if table.num_rows:
        for row in table.to_pylist():
            if row.get("lemma") == lemma:
                results.append(AttestResult(**row))
    return AttestResponse(query={"lemma": lemma, "lang": lang, "year": year}, results=results)


@app.get("/lookup", response_model=LookupResponse)
def lookup(text: str, lang: str, year: int, target_lang: str, target_year: int) -> LookupResponse:
    table = loader.load_pairs(lang, target_lang)
    results: list[LookupResult] = []
    if table.num_rows:
        for row in table.to_pylist():
            source_tokens = row.get("source_tokens") or []
            target_tokens = row.get("target_tokens") or []
            match_idx = {i for i, t in enumerate(source_tokens) if t.lower() == text.lower()}
            if not match_idx:
                continue
            for link in row.get("alignment_links") or []:
                if match_idx & set(link["source_idx"]):
                    span_idx = link["target_idx"]
                    results.append(
                        LookupResult(
                            pair_id=row["pair_id"],
                            target_sentence=row["target_text"],
                            highlighted_span=HighlightedSpan(
                                token_idx=span_idx,
                                surface=" ".join(target_tokens[i] for i in span_idx),
                            ),
                            source_sentence=row["source_text"],
                            citation=row.get("citation", ""),
                            confidence=row.get("sentence_confidence", 0.0),
                        )
                    )
    return LookupResponse(
        query={"text": text, "lang": lang, "year": year},
        target={"lang": target_lang, "year": target_year},
        results=results,
    )
