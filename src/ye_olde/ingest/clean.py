"""LLM-generative content cleaning (spec §6 step 1, continued): turns raw
extracted text into an ordered list of content-only units, with editorial
front/back matter (prefaces, introductions, footnotes, glossaries,
sidenotes, folio/manuscript references, transcriber notes, tables of
contents, page furniture) discarded.

This is deliberately LLM-driven rather than regex-driven beyond the cheap
boilerplate strip in extract.py: real scanned/transcribed editions interleave
notes into the body text (sidenotes, footnote markers, bracketed glosses)
in ways that are easy for a reader to disentangle but brittle to hand-write
rules for.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..prompt import load_prompt
from .checkpoint import Checkpoint
from .corpus_files import CorpusFile
from .llm_client import call_llm_json, is_local_model

# Paragraph-based chunking keeps each LLM call well inside context limits
# and lets the model reason about a chunk without losing track of it, at
# the cost of occasionally splitting a note/section across chunk boundaries
# — harmless here since each chunk is judged unit-by-unit, not as a whole.
_DEFAULT_MAX_CHARS = 6000

_SYSTEM_PROMPT = load_prompt("ingest", "clean_system")


def chunk_text(text: str, max_chars: int = _DEFAULT_MAX_CHARS) -> list[str]:
    """Greedily groups paragraphs (blank-line-separated) into chunks no
    larger than `max_chars`. A single paragraph longer than `max_chars` is
    kept whole as its own oversized chunk rather than cut mid-sentence.
    """
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para)
        if current and current_len + para_len + 2 > max_chars:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += para_len + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


_REVIEW_SYSTEM_PROMPT = load_prompt("ingest", "clean_review_system")


def review_units_with_llm(
    units: list[str],
    *,
    lang_code: str,
    work: str,
    model: str | None = None,
    batch_size: int = 40,
    checkpoint: Checkpoint | None = None,
) -> list[str]:
    """Second-opinion pass over already-cleaned units — see `is_local_model`
    for why this only runs for local models (no marginal cost, so it's
    worth the extra call even when nothing turns out to need fixing).
    Batched and checkpointed the same way as align.py's link/review
    batches, under `review:<batch_num>`.
    """
    reviewed: list[str] = []
    n_batches = -(-len(units) // batch_size) if units else 0
    for batch_num, start in enumerate(range(0, len(units), batch_size), start=1):
        print(f"[clean] review: batch {batch_num}/{n_batches}", file=sys.stderr, flush=True)
        batch = units[start : start + batch_size]
        prompt = f'Work: "{work}" ({lang_code}). Review these {len(batch)} content units:\n' + "\n".join(
            f"{i}. {u}" for i, u in enumerate(batch)
        )

        def compute(prompt: str = prompt) -> object:
            return call_llm_json(prompt, system=_REVIEW_SYSTEM_PROMPT, model=model)

        try:
            result = checkpoint.get_or_compute(f"review:{batch_num}", compute) if checkpoint else compute()
        except ValueError:
            reviewed.extend(batch)  # review call itself broke -> keep the originals rather than lose them
            continue
        if not isinstance(result, list) or len(result) != len(batch):
            reviewed.extend(batch)  # malformed/wrong-shape reply -> fail safe to the pre-review units
            continue
        reviewed.extend(str(u).strip() or original for u, original in zip(result, batch, strict=True))
    return reviewed


def clean_chunk_with_llm(
    chunk: str,
    *,
    lang_code: str,
    work: str,
    model: str | None = None,
) -> list[str]:
    prompt = f'Work: "{work}" ({lang_code}). Chunk:\n"""\n{chunk}\n"""'
    result = call_llm_json(prompt, system=_SYSTEM_PROMPT, model=model)
    if not isinstance(result, list) or not all(isinstance(u, str) for u in result):
        raise ValueError(f"expected a JSON array of strings, got {result!r}")
    return [u.strip() for u in result if u.strip()]


def clean_corpus_file(
    cf: CorpusFile,
    raw_text: str,
    *,
    cache_path: Path | None = None,
    model: str | None = None,
    use_cache: bool = True,
) -> list[str]:
    """Cleans a whole file's extracted text into an ordered list of content
    units, one LLM call per chunk. If `cache_path` is given and already
    exists, its contents are reused instead of re-calling the LLM — cleaning
    a long book is the expensive step, and alignment (align.py) is re-run
    far more often than cleaning during development.

    Progress is also checkpointed chunk-by-chunk to a `.partial` file next
    to `cache_path`, so an interrupted run (a long book can be dozens of
    LLM calls) resumes from the last completed chunk instead of re-billing
    every chunk cleaned so far.

    If the configured model is local (`is_local_model`), a second-opinion
    review pass (`review_units_with_llm`) runs once over the fully-cleaned
    unit list before it's cached — free for a self-hosted model, so it's
    worth doing even when nothing turns out to need correcting. Hosted API
    models skip this (it would just be a second bill for the same work).
    """
    if cache_path is not None and use_cache and cache_path.exists():
        print(f"[clean] {cf.path.name}: using cached {cache_path.name}", file=sys.stderr)
        cached: list[str] = json.loads(cache_path.read_text(encoding="utf-8"))
        return cached

    chunks = chunk_text(raw_text)
    partial_path = cache_path.with_name(cache_path.name + ".partial") if cache_path is not None else None

    units: list[str] = []
    start = 0
    if partial_path is not None and use_cache and partial_path.exists():
        saved = json.loads(partial_path.read_text(encoding="utf-8"))
        if saved.get("total_chunks") == len(chunks):
            units, start = saved["units"], saved["next_chunk"]
            print(f"[clean] {cf.path.name}: resuming from chunk {start + 1}/{len(chunks)}", file=sys.stderr)

    for i in range(start, len(chunks)):
        print(f"[clean] {cf.path.name}: chunk {i + 1}/{len(chunks)}", file=sys.stderr, flush=True)
        units.extend(clean_chunk_with_llm(chunks[i], lang_code=cf.lang_code, work=cf.work_label, model=model))
        if partial_path is not None:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(
                json.dumps({"total_chunks": len(chunks), "next_chunk": i + 1, "units": units}, ensure_ascii=False),
                encoding="utf-8",
            )
    print(f"[clean] {cf.path.name}: {len(units)} content units kept", file=sys.stderr)

    if is_local_model(model):
        print(f"[clean] {cf.path.name}: local model, running review pass", file=sys.stderr)
        review_checkpoint = (
            Checkpoint(cache_path.with_name(cache_path.name + ".review_progress.json"), meta={"n_units": len(units)})
            if cache_path is not None
            else None
        )
        units = review_units_with_llm(
            units, lang_code=cf.lang_code, work=cf.work_label, model=model, checkpoint=review_checkpoint
        )
        if review_checkpoint is not None:
            review_checkpoint.clear()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(units, ensure_ascii=False, indent=2), encoding="utf-8")
        if partial_path is not None and partial_path.exists():
            partial_path.unlink()
    return units
