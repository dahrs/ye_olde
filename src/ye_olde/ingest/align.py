"""Data acquisition & alignment pipeline — spec §6, producing §3c aligned
example-pair records from a folder of parallel/comparable diachronic texts
(`data/raw/<work>/`).

Two halves, deliberately split:
  - Algorithmic (this module + extract.py/corpus_files.py/sentence_align.py):
    file discovery and filename parsing, text extraction, embedding-based
    sentence matching, deterministic tokenization, proportional block
    windowing so long books stay within LLM context, span-to-token-index
    resolution, dedup, and output writing.
  - LLM-generative (clean.py + this module's `align_block`/
    `extract_links_batch`): reading and interpreting the text well enough
    to drop non-content material, filling in the sentence matches the
    embedding aligner missed or was unsure of, and proposing word/phrase
    links within a matched pair — the one thing a bare embedding aligner
    can't do at all, and which off-the-shelf statistical/neural word
    aligners (fast_align, awesome-align) handle poorly here because they
    weren't built for text with this much spelling/grammar drift.

Two alignment modes (`mode=` on `align_corpus_pair`/`align_corpus_folder`):
  - "hybrid" (default): `sentence_align.mutual_nearest_neighbor_align` finds
    high-confidence sentence matches cheaply via embeddings; only the gaps
    it leaves unmatched (and the runner-up-margin cases it wasn't sure
    about) go to the LLM for discovery, and word/phrase links for the
    embedding-matched anchors are back-filled with batched LLM calls.
    Far fewer LLM calls than "llm" mode on a long book, at the cost of
    trusting the embedding aligner's matches it doesn't ask the LLM to
    double check.
  - "llm": the LLM proposes sentence matches from scratch over sliding
    windows of the whole text, with no embedding pass at all — simpler,
    slower/costlier at scale, and doesn't depend on `sentence-transformers`
    being installed or the embedding model download succeeding.

Every LLM call this module makes — cleaning (clean.py), gap/block
discovery, word-link batches — is checkpointed to disk as it succeeds (see
checkpoint.py), so an interrupted run (an API billing error partway through
a long book is the expected case, not an edge case) resumes from the next
uncomputed call on retry instead of re-billing everything already done.
Every such call also gets exactly one non-recursive JSON-repair retry if
its reply doesn't parse (see `llm_client.call_llm_json`).

Two more safeguards, applied to every pair regardless of mode:
  - `review_pairs_with_llm`, only for local models (`is_local_model`) — a
    second, free-to-run pass asking the model to re-check its own output,
    changing nothing when there's nothing to fix. Skipped for hosted APIs,
    where it would just be a second bill for the same work.
  - `_verify_pairs_against_source` — always runs, deterministically
    (no LLM): every pair's source_text/target_text must still be a
    verbatim, in-order excerpt of the cleaned units it was aligned from.
    The alignment step (and the review pass above) may choose different
    boundaries, but may never alter the words themselves or their order.
    A pair that fails this is dropped, catching what either step got wrong.

Output rows are flattened to match what `search_api/app/main.py`'s
`/lookup` and `/attest` handlers already expect to read off a Parquet pairs
shard (`pair_id`, `source_text`, `target_text`, `source_tokens`,
`target_tokens`, `alignment_links`, `citation`, `sentence_confidence`),
plus nested `source`/`target` metadata (lang_code/year/work/author) so a
later indexing job (§10) can route each row to its `pairs/<lang_a>_<lang_b>/`
shard without re-deriving that from the pair content.
"""

from __future__ import annotations

import itertools
import json
import re
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

from ..prompt import load_prompt
from .checkpoint import Checkpoint
from .clean import clean_corpus_file
from .corpus_files import CorpusFile, discover_corpus_files
from .extract import extract_text, strip_boilerplate
from .llm_client import call_llm_json, is_local_model
from .sentence_align import SentenceMatch, mutual_nearest_neighbor_align

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# A §3c aligned-pair record in its working (pre-output) shape: heterogeneous
# values (str, list[str], nested alignment_links dicts, float, None) that
# get progressively filled in across align_block/extract_links_batch/
# _finalize_records, so a TypedDict would need every field optional from the
# start — this alias just names what a bare `dict[str, Any]` here *means*.
PairRecord: TypeAlias = dict[str, Any]

_ALIGN_SYSTEM_PROMPT = load_prompt("ingest", "align_system")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().lower()


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "work"


def find_span_token_indices(tokens: list[str], span: str) -> list[int] | None:
    """Locates `span` as a contiguous run inside `tokens` (case-insensitive)
    and returns its token indices, or None if it isn't found verbatim —
    which happens when the LLM's proposed span isn't an exact substring of
    the text it was supposed to quote from, in which case that one link is
    dropped rather than guessed at.
    """
    span_tokens = tokenize(span)
    if not span_tokens:
        return None
    lower_tokens = [t.lower() for t in tokens]
    lower_span = [t.lower() for t in span_tokens]
    n, m = len(lower_tokens), len(lower_span)
    for i in range(n - m + 1):
        if lower_tokens[i : i + m] == lower_span:
            return list(range(i, i + m))
    return None


def make_blocks(
    units_a: list[str],
    units_b: list[str],
    *,
    block_size: int = 40,
    overlap_ratio: float = 0.15,
) -> list[tuple[list[str], list[str]]]:
    """Splits two cleaned-unit lists into corresponding proportional
    windows: window i covers units [i, i+block_size) of `units_a`, and the
    *same fractional position range* of `units_b`. This assumes translated
    editions proceed through the same content in the same order without
    heavy reordering — true for the Bible verses, poem fitts, and book/
    chapter prose this pipeline targets. A small overlap between
    consecutive windows (deduped later) covers pairs whose content straddles
    a window boundary.
    """
    n_a, n_b = len(units_a), len(units_b)
    if n_a == 0 or n_b == 0:
        return []
    step = max(1, round(block_size * (1 - overlap_ratio)))
    blocks: list[tuple[list[str], list[str]]] = []
    i = 0
    while True:
        end_a = min(n_a, i + block_size)
        j = int(i / n_a * n_b)
        end_b = min(n_b, max(j + 1, round(end_a / n_a * n_b)))
        blocks.append((units_a[i:end_a], units_b[j:end_b]))
        if end_a >= n_a:
            break
        i += step
    return blocks


def _format_units(units: list[str]) -> str:
    return "\n".join(f"{i}. {u}" for i, u in enumerate(units))


def align_block(
    block_a: list[str],
    block_b: list[str],
    cf_a: CorpusFile,
    cf_b: CorpusFile,
    *,
    model: str | None = None,
) -> list[PairRecord]:
    """One LLM call: proposes aligned pairs between two corresponding
    windows of cleaned content, then algorithmically resolves each
    proposed word/phrase link to token indices in a deterministic
    tokenization of the pair's own text.
    """
    prompt = (
        f'SOURCE ({cf_a.lang_code}, {cf_a.year}, "{cf_a.work_label}"):\n'
        f"{_format_units(block_a)}\n\n"
        f'TARGET ({cf_b.lang_code}, {cf_b.year}, "{cf_b.work_label}"):\n'
        f"{_format_units(block_b)}"
    )
    raw_pairs = call_llm_json(prompt, system=_ALIGN_SYSTEM_PROMPT, model=model)
    if not isinstance(raw_pairs, list):
        raise ValueError(f"expected a JSON array of pair objects, got {raw_pairs!r}")

    resolved: list[PairRecord] = []
    for rp in raw_pairs:
        if not isinstance(rp, dict):
            continue
        source_text = str(rp.get("source_text") or "").strip()
        target_text = str(rp.get("target_text") or "").strip()
        if not source_text or not target_text:
            continue
        source_tokens = tokenize(source_text)
        target_tokens = tokenize(target_text)
        alignment_links = []
        for link in rp.get("links") or []:
            if not isinstance(link, dict):
                continue
            s_idx = find_span_token_indices(source_tokens, str(link.get("source_span") or ""))
            t_idx = find_span_token_indices(target_tokens, str(link.get("target_span") or ""))
            if s_idx and t_idx:
                alignment_links.append({"source_idx": s_idx, "target_idx": t_idx, "sense_id": None})
        try:
            confidence = float(rp.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        resolved.append(
            {
                "source_text": source_text,
                "target_text": target_text,
                "source_tokens": source_tokens,
                "target_tokens": target_tokens,
                "citation": rp.get("citation") or None,
                "sentence_confidence": confidence,
                "alignment_links": alignment_links,
                "method": "llm",
            }
        )
    return resolved


_LINKS_SYSTEM_PROMPT = load_prompt("ingest", "links_system")


def extract_links_batch(
    pairs: list[PairRecord],
    cf_a: CorpusFile,
    cf_b: CorpusFile,
    *,
    model: str | None = None,
    batch_size: int = 15,
    checkpoint: Checkpoint | None = None,
) -> None:
    """Fills in `alignment_links` (in place) for pairs whose sentence match
    is already fixed — the cheaper of the two LLM prompts this module uses,
    since the model only has to link words, not also find/verify the match.
    Batching amortizes one call across several pairs at once. Each batch is
    checkpointed under `links:<batch_num>` when `checkpoint` is given, so a
    run interrupted (e.g. by an API billing error) partway through doesn't
    re-request links for batches it already got back.
    """
    n_batches = -(-len(pairs) // batch_size) if pairs else 0
    for batch_num, start in enumerate(range(0, len(pairs), batch_size), start=1):
        print(f"[align] links: batch {batch_num}/{n_batches} ({len(pairs)} anchors)", file=sys.stderr, flush=True)
        batch = pairs[start : start + batch_size]
        prompt = "\n".join(
            f"{k}. SOURCE ({cf_a.lang_code}): {p['source_text']}\n   TARGET ({cf_b.lang_code}): {p['target_text']}"
            for k, p in enumerate(batch)
        )

        def compute(prompt: str = prompt) -> object:
            return call_llm_json(prompt, system=_LINKS_SYSTEM_PROMPT, model=model)

        try:
            raw = checkpoint.get_or_compute(f"links:{batch_num}", compute) if checkpoint is not None else compute()
        except ValueError:
            continue  # leave alignment_links empty for this batch rather than failing the whole run
        if not isinstance(raw, list):
            continue
        for k, pair in enumerate(batch):
            links_raw = raw[k] if k < len(raw) and isinstance(raw[k], list) else []
            resolved_links = []
            for link in links_raw:
                if not isinstance(link, dict):
                    continue
                s_idx = find_span_token_indices(pair["source_tokens"], str(link.get("source_span") or ""))
                t_idx = find_span_token_indices(pair["target_tokens"], str(link.get("target_span") or ""))
                if s_idx and t_idx:
                    resolved_links.append({"source_idx": s_idx, "target_idx": t_idx, "sense_id": None})
            pair["alignment_links"] = resolved_links


def _discover_pairs_llm(
    gap_a: list[str],
    gap_b: list[str],
    cf_a: CorpusFile,
    cf_b: CorpusFile,
    *,
    model: str | None,
    block_size: int,
    checkpoint: Checkpoint | None = None,
    key_prefix: str = "gap",
) -> list[PairRecord]:
    """Runs full LLM sentence-discovery (same as "llm" mode) over a small
    unmatched stretch left over by the embedding aligner, instead of the
    whole document — this is what keeps hybrid mode's LLM cost down to
    roughly the fraction of the text the cheap pass couldn't resolve. Each
    LLM call is checkpointed under its own `<key_prefix>:*` key, since a
    single large gap (hundreds of units) can take many calls on its own —
    exactly the case that died mid-gap on Sir Gawayne.
    """

    def call(block_a: list[str], block_b: list[str], key: str) -> list[PairRecord]:
        if checkpoint is not None:
            return checkpoint.get_or_compute(key, lambda: align_block(block_a, block_b, cf_a, cf_b, model=model))
        return align_block(block_a, block_b, cf_a, cf_b, model=model)

    if len(gap_a) <= block_size and len(gap_b) <= block_size:
        return call(gap_a, gap_b, f"{key_prefix}:single")
    pairs = []
    for sub_idx, (block_a, block_b) in enumerate(make_blocks(gap_a, gap_b, block_size=block_size)):
        pairs.extend(call(block_a, block_b, f"{key_prefix}:sub{sub_idx}"))
    return pairs


def _align_pairs_llm_only(
    cf_a: CorpusFile,
    cf_b: CorpusFile,
    units_a: list[str],
    units_b: list[str],
    *,
    model: str | None,
    block_size: int,
    checkpoint: Checkpoint | None = None,
) -> list[PairRecord]:
    blocks = make_blocks(units_a, units_b, block_size=block_size)
    seen: set[tuple[str, str]] = set()
    pairs: list[PairRecord] = []
    for block_num, (block_a, block_b) in enumerate(blocks, start=1):
        print(f"[align] llm mode: block {block_num}/{len(blocks)}", file=sys.stderr, flush=True)

        def compute(ba: list[str] = block_a, bb: list[str] = block_b) -> list[PairRecord]:
            return align_block(ba, bb, cf_a, cf_b, model=model)

        block_pairs = checkpoint.get_or_compute(f"block:{block_num}", compute) if checkpoint is not None else compute()
        for pair in block_pairs:
            key = (_normalize(pair["source_text"]), _normalize(pair["target_text"]))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(pair)
    return pairs


def _align_pairs_hybrid(
    cf_a: CorpusFile,
    cf_b: CorpusFile,
    units_a: list[str],
    units_b: list[str],
    *,
    model: str | None,
    embedding_model: str | None,
    block_size: int,
    margin_threshold: float,
    link_batch_size: int,
    checkpoint: Checkpoint | None = None,
) -> list[PairRecord]:
    n_a, n_b = len(units_a), len(units_b)
    print(f"[align] hybrid: embedding {n_a}+{n_b} units and matching...", file=sys.stderr, flush=True)
    matches = mutual_nearest_neighbor_align(
        units_a, units_b, model_name=embedding_model, margin_threshold=margin_threshold
    )
    print(f"[align] hybrid: {len(matches)} embedding anchors found", file=sys.stderr)

    pairs: list[PairRecord] = []
    anchors: list[PairRecord] = []
    prev_i, prev_j = -1, -1
    gap_index = 0
    for m in [*matches, SentenceMatch(i=n_a, j=n_b, score=0.0)]:  # sentinel: flush the trailing gap
        gap_a, gap_b = units_a[prev_i + 1 : m.i], units_b[prev_j + 1 : m.j]
        if gap_a and gap_b:
            print(
                f"[align] hybrid: gap of {len(gap_a)}+{len(gap_b)} units -> LLM discovery",
                file=sys.stderr,
                flush=True,
            )
            pairs.extend(
                _discover_pairs_llm(
                    gap_a,
                    gap_b,
                    cf_a,
                    cf_b,
                    model=model,
                    block_size=block_size,
                    checkpoint=checkpoint,
                    key_prefix=f"gap{gap_index}",
                )
            )
            gap_index += 1
        if m.i < n_a:  # the sentinel itself isn't a real match
            anchor = {
                "source_text": units_a[m.i],
                "target_text": units_b[m.j],
                "source_tokens": tokenize(units_a[m.i]),
                "target_tokens": tokenize(units_b[m.j]),
                "citation": None,
                "sentence_confidence": m.score,
                "alignment_links": [],
                "method": "embedding",
            }
            pairs.append(anchor)
            anchors.append(anchor)
        prev_i, prev_j = m.i, m.j

    extract_links_batch(anchors, cf_a, cf_b, model=model, batch_size=link_batch_size, checkpoint=checkpoint)

    # Defensive dedup: gap-discovered pairs shouldn't overlap embedding
    # anchors by construction (they cover disjoint unit ranges), but this
    # costs nothing and guards against the LLM re-proposing a pair it was
    # also shown as surrounding context.
    seen: set[tuple[str, str]] = set()
    deduped = []
    for pair in pairs:
        key = (_normalize(pair["source_text"]), _normalize(pair["target_text"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(pair)
    return deduped


def align_corpus_pair(
    cf_a: CorpusFile,
    cf_b: CorpusFile,
    units_a: list[str],
    units_b: list[str],
    *,
    mode: str = "hybrid",
    model: str | None = None,
    embedding_model: str | None = None,
    block_size: int = 40,
    margin_threshold: float = 0.05,
    link_batch_size: int = 15,
    checkpoint_path: Path | None = None,
    use_checkpoint: bool = True,
) -> list[PairRecord]:
    """Aligns two whole cleaned works into deduped §3c-shaped pair records.
    See the module docstring for what `mode="hybrid"` vs. `mode="llm"` mean.

    If `checkpoint_path` is given, every LLM call this makes (gap/block
    discovery, word-link batches) is persisted there as it succeeds; a call
    that raises (an API billing error, most commonly) leaves its own key
    uncomputed rather than losing the calls that already landed. Re-running
    with the same `checkpoint_path` and the same units/mode/block_size
    resumes instead of re-billing from scratch — `use_checkpoint=False`
    (or a mismatched unit count/mode/block_size) starts fresh instead.
    """
    checkpoint = None
    if checkpoint_path is not None:
        meta = {"n_a": len(units_a), "n_b": len(units_b), "mode": mode, "block_size": block_size}
        checkpoint = Checkpoint(checkpoint_path, meta)
        if not use_checkpoint:
            checkpoint.clear()

    if mode == "llm":
        pairs = _align_pairs_llm_only(
            cf_a, cf_b, units_a, units_b, model=model, block_size=block_size, checkpoint=checkpoint
        )
    elif mode == "hybrid":
        pairs = _align_pairs_hybrid(
            cf_a,
            cf_b,
            units_a,
            units_b,
            model=model,
            embedding_model=embedding_model,
            block_size=block_size,
            margin_threshold=margin_threshold,
            link_batch_size=link_batch_size,
            checkpoint=checkpoint,
        )
    else:
        raise ValueError(f"mode must be 'hybrid' or 'llm', got {mode!r}")

    if is_local_model(model):
        print("[align] local model, running pair review pass", file=sys.stderr)
        pairs = review_pairs_with_llm(pairs, cf_a, cf_b, model=model, checkpoint=checkpoint)

    # Mandatory regardless of mode/review: a deterministic (non-LLM) check
    # that every pair — however it was produced — is still a verbatim,
    # in-order excerpt of the cleaned source it claims to quote. Runs after
    # the optional review pass above so it also catches anything *that*
    # pass got wrong, not just what the aligner itself got wrong.
    pairs = _verify_pairs_against_source(pairs, units_a, units_b)

    return _finalize_records(cf_a, cf_b, pairs, model=model, embedding_model=embedding_model)


_PAIR_REVIEW_SYSTEM_PROMPT = load_prompt("ingest", "pair_review_system")


def review_pairs_with_llm(
    pairs: list[PairRecord],
    cf_a: CorpusFile,
    cf_b: CorpusFile,
    *,
    model: str | None = None,
    batch_size: int = 15,
    checkpoint: Checkpoint | None = None,
) -> list[PairRecord]:
    """Second-opinion pass over already-aligned pairs — see `is_local_model`
    for why this only runs for local models. A pair whose text the review
    actually changes has its tokens retokenized and its `alignment_links`
    reset (they were computed against the old text and would otherwise
    point at the wrong words) — `_verify_pairs_against_source` afterwards
    is what actually guards against the review inventing new wording.
    """
    reviewed: list[PairRecord] = []
    n_batches = -(-len(pairs) // batch_size) if pairs else 0
    for batch_num, start in enumerate(range(0, len(pairs), batch_size), start=1):
        print(f"[align] review: batch {batch_num}/{n_batches}", file=sys.stderr, flush=True)
        batch = pairs[start : start + batch_size]
        prompt = "\n".join(
            f"{k}. SOURCE ({cf_a.lang_code}): {p['source_text']}\n"
            f"   TARGET ({cf_b.lang_code}): {p['target_text']}\n"
            f"   citation: {p.get('citation')!r}"
            for k, p in enumerate(batch)
        )

        def compute(prompt: str = prompt) -> object:
            return call_llm_json(prompt, system=_PAIR_REVIEW_SYSTEM_PROMPT, model=model)

        try:
            raw = checkpoint.get_or_compute(f"review:{batch_num}", compute) if checkpoint else compute()
        except ValueError:
            reviewed.extend(batch)  # review call itself broke -> keep the originals rather than lose them
            continue
        if not isinstance(raw, list) or len(raw) != len(batch):
            reviewed.extend(batch)
            continue
        for original, corrected in zip(batch, raw, strict=True):
            if not isinstance(corrected, dict):
                reviewed.append(original)
                continue
            new_source = str(corrected.get("source_text") or "").strip()
            new_target = str(corrected.get("target_text") or "").strip()
            unchanged = new_source == original["source_text"] and new_target == original["target_text"]
            if not new_source or not new_target or unchanged:
                reviewed.append(original)
                continue
            reviewed.append(
                {
                    **original,
                    "source_text": new_source,
                    "target_text": new_target,
                    "source_tokens": tokenize(new_source),
                    "target_tokens": tokenize(new_target),
                    "citation": corrected.get("citation", original.get("citation")),
                    "alignment_links": [],
                }
            )
    return reviewed


def _normalize_for_verification(text: str) -> str:
    return " ".join(text.split()).lower()


def _verify_pairs_against_source(pairs: list[PairRecord], units_a: list[str], units_b: list[str]) -> list[PairRecord]:
    """Deterministic, non-LLM safety net: a pair's source_text/target_text
    must be a verbatim, in-order excerpt of the cleaned units it was
    aligned from. The alignment step (and the optional review pass above)
    may pick different boundaries — which units, or which part of a unit,
    become a pair — but must never alter the words themselves or their
    order. A pair that fails this (paraphrased or hallucinated instead of
    quoted) is dropped rather than shipped into the corpus as a false
    attestation.
    """
    source_ref = _normalize_for_verification(" ".join(units_a))
    target_ref = _normalize_for_verification(" ".join(units_b))
    verified, dropped = [], 0
    for pair in pairs:
        if (
            _normalize_for_verification(pair["source_text"]) in source_ref
            and _normalize_for_verification(pair["target_text"]) in target_ref
        ):
            verified.append(pair)
        else:
            dropped += 1
    if dropped:
        print(
            f"[align] verification: dropped {dropped}/{len(pairs)} pairs not verbatim in cleaned source",
            file=sys.stderr,
        )
    return verified


def _finalize_records(
    cf_a: CorpusFile,
    cf_b: CorpusFile,
    pairs: list[PairRecord],
    *,
    model: str | None,
    embedding_model: str | None,
) -> list[PairRecord]:
    slug = _slugify(cf_a.title)
    resolved_model = model or _default_model_label()
    resolved_embedding_model = embedding_model or _default_embedding_model_label()
    ingested_at = datetime.now(UTC).isoformat()
    records = []
    for i, pair in enumerate(pairs):
        method = pair.pop("method", "llm")
        sentence_method = f"embedding:{resolved_embedding_model}" if method == "embedding" else f"llm:{resolved_model}"
        records.append(
            {
                "pair_id": f"{slug}_{cf_a.year}x{cf_b.year}_{i:05d}",
                "source": {
                    "lang_code": cf_a.lang_code,
                    "year": cf_a.year,
                    "work": cf_a.work_label,
                    "author": cf_a.author,
                },
                "target": {
                    "lang_code": cf_b.lang_code,
                    "year": cf_b.year,
                    "work": cf_b.work_label,
                    "author": cf_b.author,
                },
                **pair,
                "sentence_method": sentence_method,
                "metadata": {"license": None, "ingested_at": ingested_at},
            }
        )
    return records


def _default_model_label() -> str:
    from ..config import get_settings

    return get_settings().litellm_model or "unconfigured"


def _default_embedding_model_label() -> str:
    from ..config import get_settings

    return get_settings().embedding_model


def write_jsonl(records: list[PairRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def align_corpus_folder(
    folder: str | Path,
    *,
    output_dir: str | Path = "data/processed",
    mode: str = "hybrid",
    model: str | None = None,
    embedding_model: str | None = None,
    block_size: int = 40,
    margin_threshold: float = 0.05,
    link_batch_size: int = 15,
    use_cache: bool = True,
) -> list[Path]:
    """End-to-end entrypoint: given `data/raw/<work>/` containing 2+
    parallel-translation files, cleans each with the LLM (cached to
    `<output_dir>/<work>/<file>.cleaned.json`) and writes one `.jsonl`
    bitext file per language pair (`file1-file2`, `file2-file3`,
    `file1-file3`, ... for N files) to `<output_dir>/<work>/`. JSONL is the
    only output format: it's the one that keeps everything downstream needs
    (tokens, word-level `alignment_links`, provenance) — a TMX or plain-text
    export would only ever be a lossy view derived from it, so there's
    nothing to gain from also writing one here.

    Returns the list of bitext file paths written.
    """
    folder = Path(folder)
    # discover_corpus_files() already returns files sorted oldest-year-first;
    # re-sorting here makes that an explicit, local guarantee rather than an
    # incidental side effect of another function's ordering. Every pair
    # itertools.combinations() produces below therefore has cf_a.year <=
    # cf_b.year, so a language pair is only ever written in one direction
    # (e.g. always enm-eng, never also eng-enm) and always chronologically.
    files = sorted(discover_corpus_files(folder), key=lambda cf: cf.year)
    processed_root = Path(output_dir) / folder.name

    cleaned: dict[Path, list[str]] = {}
    for cf in files:
        raw_text = strip_boilerplate(extract_text(cf.path))
        cache_path = processed_root / f"{cf.path.stem}.cleaned.json"
        cleaned[cf.path] = clean_corpus_file(cf, raw_text, cache_path=cache_path, model=model, use_cache=use_cache)

    written: list[Path] = []
    for cf_a, cf_b in itertools.combinations(files, 2):
        print(f"[align] pair {cf_a.lang_code}-{cf_b.lang_code} (mode={mode})", file=sys.stderr, flush=True)
        checkpoint_path = processed_root / f"{cf_a.lang_code}-{cf_b.lang_code}.align_progress.json"
        records = align_corpus_pair(
            cf_a,
            cf_b,
            cleaned[cf_a.path],
            cleaned[cf_b.path],
            mode=mode,
            model=model,
            embedding_model=embedding_model,
            block_size=block_size,
            margin_threshold=margin_threshold,
            link_batch_size=link_batch_size,
            checkpoint_path=checkpoint_path,
            use_checkpoint=use_cache,
        )
        out_path = processed_root / f"{cf_a.lang_code}-{cf_b.lang_code}.bitext.jsonl"
        write_jsonl(records, out_path)
        written.append(out_path)
        checkpoint_path.unlink(missing_ok=True)  # done: no need to keep resumable progress for a finished pair
        print(f"[align] pair {cf_a.lang_code}-{cf_b.lang_code}: {len(records)} pairs -> {out_path}", file=sys.stderr)
    return written
