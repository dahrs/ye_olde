"""Resolves an LLM-proposed (lemma, UPOS, gloss) to a WordNet-anchored
`sense_id` — spec §3a/§3d/§9.

**Matches by embedding similarity of the gloss, not by looking up the
lemma in WordNet.** An earlier version of this module used
`wordnet.synsets(lemma, pos)` to fetch candidate synsets for a lemma, then
compared glosses by word overlap among just those candidates. That has a
real ceiling: WordNet is English-only, so it can only ever return
candidates for a lemma that happens to already be (or resemble) a real
English word — useless for a genuinely distant pair (`eng`↔`lat`, a future
`fro`↔`grc`, ...), where neither side's lemma will ever be an English
WordNet headword, regardless of how the comparison itself works. Matching
on the gloss's *meaning* instead of the lemma's *spelling* sidesteps that
entirely: the gloss is always English (spec §3d), so its embedding can be
compared against a precomputed embedding of every WordNet synset's own
definition — global nearest-neighbor search, not lemma-filtered lookup.
`lemma_hint` still exists in this module's signature, but purely
cosmetically now (naming a minted id), not for matching.

**Two indices, searched together, same embedding space:**
- The WordNet index (`data/sense_index/wordnet_synsets.<model-slug>.{npy,json}`)
  — every open-class synset's definition, embedded once (`build_wordnet_index`,
  ~100k+ definitions) and loaded read-only here. `align.align_corpus_folder`
  checks the index is present *and complete* before a real "hybrid"/"llm"
  mode run starts resolving senses (`ensure_wordnet_index_ready`), building
  or resuming it first if not — this used to be a separately-run,
  easy-to-forget prerequisite (`scripts/build_wordnet_sense_index.py`,
  still usable standalone e.g. to run it ahead of time overnight instead of
  blocking a pipeline run's first real invocation), which meant a run
  starting before the precompute finished silently minted a fresh
  `ye_olde:` id for every sense whose real WordNet synset just hadn't been
  embedded yet — the exact failure this check exists to prevent.
  `resolve_sense_id` itself still soft-fails to mint-only (no auto-build)
  when called directly with no index present — deliberately: triggering a
  many-hour precompute as a side effect of a low-level function a unit
  test calls directly would make testing this module require a full
  WordNet embed first, which defeats the point of a unit test.
- The minted index (`data/sense_index/minted.<model-slug>.json`) — every
  `ye_olde:`-namespaced sense minted so far, growing as this module runs.
  **This is what makes minted ids reusable rather than one-off**: without
  it, the same recurring concept (e.g. "wergild" mentioned five times
  across a corpus) would mint five different ids, defeating the entire
  point of `sense_id` grouping same-meaning attestations together. A new
  gloss is checked against both indices before minting a new id — if it
  matches an *existing* minted entry closely enough, that entry's id is
  reused, not a fresh one.

Both indices are searched restricted to the same coarse UPOS (mapped to
WordNet's `n`/`v`/`a`/`r`) — cheap, and avoids a noun gloss spuriously
matching a verb synset by pure semantic proximity.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_settings
from .embed import embed_units

# Every language's proposed sense maps onto these, spec §3d — WordNet only
# has open-class synsets, so a closed-class UPOS (DET, ADP, ...) mints
# immediately, no embedding/index lookup at all.
_WORDNET_POS = {"NOUN": "n", "PROPN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
# The distinct WordNet POS tags themselves (NOUN and PROPN both map to "n"
# above) -- what scripts/build_wordnet_sense_index.py iterates, so it
# doesn't embed every noun synset twice.
WN_POS_TAGS = ("n", "v", "a", "r")

_SSLUG_RE = re.compile(r"[^a-z0-9]+")

# Cosine-similarity floor (embeddings are L2-normalized, so a plain dot
# product is the cosine similarity) below which a candidate isn't trusted
# as the same sense — open, unvalidated constant, same honesty as
# ye_olde.resolver.LAMBDA_MAX (spec §9): needs empirical tuning once real
# query examples exist.
SIMILARITY_THRESHOLD = 0.80

# WordNet-embedding + minted-sense data, gitignored (large, machine-
# regenerable) -- see `scripts/build_wordnet_sense_index.py` to (re)build it.
DEFAULT_INDEX_ROOT = Path("data/sense_index")

# Per-(lemma, pos) minted-id counters -- a fallback only, for the
# closed-class path (nothing persisted to seed from, see _mint). The
# open-class path derives its counter from the *persisted* minted index
# instead of this dict, specifically so two separate process runs don't
# both mint e.g. "ye_olde:wergild.n.01" for two different senses.
_mint_counters: dict[str, int] = {}


def _slugify(text: str) -> str:
    slug = _SSLUG_RE.sub("_", text.lower()).strip("_")
    return slug or "sense"


def model_slug(model_name: str) -> str:
    return _SSLUG_RE.sub("_", model_name).strip("_")


def _resolve_root(root: Path | None) -> Path:
    # Read at call time, not bound as a `root: Path = DEFAULT_INDEX_ROOT`
    # default -- a default *value* is bound once at import time, so
    # monkeypatching DEFAULT_INDEX_ROOT afterward would silently not affect
    # a call that omits `root=` (the same gotcha common.function_words'
    # `_resolve_root` docstring explains in full).
    return root if root is not None else DEFAULT_INDEX_ROOT


@lru_cache(maxsize=8)
def _load_wordnet_index(model_name: str, root: Path) -> tuple[np.ndarray, list[str], list[str], list[str]] | None:
    """Returns `(embeddings, names, definitions, pos_tags)`, all the same
    length/row-order, or `None` if `scripts/build_wordnet_sense_index.py`
    hasn't been run yet for this `(model_name, root)` — callers treat that
    as "no WordNet candidates available," not an error (this module works,
    just mint-only, before the index is built — the same soft-fail-before-
    real-data posture `search_api` already uses throughout, spec §11).
    """
    slug = model_slug(model_name)
    npy_path = root / f"wordnet_synsets.{slug}.npy"
    json_path = root / f"wordnet_synsets.{slug}.json"
    if not npy_path.is_file() or not json_path.is_file():
        return None
    embeddings = np.load(npy_path)
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    return embeddings, meta["names"], meta["definitions"], meta["pos"]


def _index_paths(model_name: str, root: Path) -> tuple[Path, Path, Path]:
    slug = model_slug(model_name)
    return (
        root / f"wordnet_synsets.{slug}.npy",
        root / f"wordnet_synsets.{slug}.json",
        root / f"wordnet_synsets.{slug}.progress.json",
    )


def _index_is_complete(model_name: str, root: Path) -> bool:
    """True only once every synset is embedded — `build_wordnet_index`
    deletes the `.progress.json` sidecar as its very last step (see that
    function), so that file's mere existence means "still in progress or
    was interrupted partway," regardless of how far it got. A partial
    index still has real npy/json files (`_load_wordnet_index` happily
    reads them for the soft-fail/direct-call path above) — this check is
    stricter on purpose, for `ensure_wordnet_index_ready`'s use only.
    """
    npy_path, json_path, progress_path = _index_paths(model_name, root)
    return npy_path.is_file() and json_path.is_file() and not progress_path.is_file()


def build_wordnet_index(
    *,
    model_name: str | None = None,
    root: Path | None = None,
    batch_size: int = 256,
    min_free_gb: float = 2.0,
) -> None:
    """Embeds every open-class WordNet synset's own definition and writes
    the result to `root` (default `DEFAULT_INDEX_ROOT`) for
    `_load_wordnet_index`/`resolve_sense_id` to read — the one
    implementation of this precompute, called both by
    `scripts/build_wordnet_sense_index.py` (a thin CLI wrapper, for running
    it standalone/ahead of time) and by `ensure_wordnet_index_ready` below
    (automatically, from a real pipeline run that finds the index missing
    or incomplete).

    **Incremental, resumable, disk-aware**: the embedding matrix is written
    to a pre-sized `.npy` file via `numpy.lib.format.open_memmap` and
    flushed to disk after every batch (not accumulated in RAM and written
    once at the end) — the `.progress.json` sidecar records how many rows
    are done, so a killed/interrupted/`min_free_gb`-stopped run resumes
    from the next uncompleted batch on a later call with the same
    `model_name`/`root`, rather than starting over. `min_free_gb` stops the
    run cleanly *before* the destination volume actually fills, leaving
    already-written rows intact and resumable, rather than crashing
    mid-write.

    Prints its own progress; how long a full run actually takes depends
    entirely on the host's hardware (~100k+ definitions to embed) — no
    number is promised here.
    """
    import nltk

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        print("[sense] downloading nltk 'wordnet' corpus...", file=sys.stderr)
        nltk.download("wordnet", quiet=True)
    from nltk.corpus import wordnet

    resolved_model = model_name or get_settings().embedding_model
    resolved_root = _resolve_root(root)
    resolved_root.mkdir(parents=True, exist_ok=True)
    min_free_bytes = min_free_gb * (1024**3)

    # Gathering names/definitions/POS is fast and cheap (plain in-memory
    # WordNet corpus reads, no embedding calls) -- always redone in full
    # even on a resume, so it never itself needs incremental saving.
    names: list[str] = []
    definitions: list[str] = []
    pos_tags: list[str] = []
    for wn_pos in WN_POS_TAGS:
        for syn in wordnet.all_synsets(pos=wn_pos):
            names.append(syn.name())
            definitions.append(syn.definition())
            pos_tags.append(wn_pos)
    total = len(definitions)

    npy_path, json_path, progress_path = _index_paths(resolved_model, resolved_root)

    # Metadata is small (a few MB of text at most for ~120k short
    # definitions) -- written in full up front, not incrementally; the
    # embedding matrix below is the part that's actually large.
    json_path.write_text(
        json.dumps({"names": names, "definitions": definitions, "pos": pos_tags}, ensure_ascii=False),
        encoding="utf-8",
    )

    dim = int(embed_units([definitions[0]], model_name=resolved_model).shape[1])

    resume_from = 0
    if npy_path.is_file() and progress_path.is_file():
        prev = json.loads(progress_path.read_text(encoding="utf-8"))
        if prev.get("total") == total and prev.get("dim") == dim:
            resume_from = int(prev.get("completed", 0))

    if resume_from > 0:
        print(f"[sense] resuming WordNet index build from row {resume_from}/{total}", file=sys.stderr)
        matrix = np.lib.format.open_memmap(npy_path, mode="r+")
    else:
        matrix = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.float32, shape=(total, dim))

    print(f"[sense] embedding {total} synset definitions with {resolved_model}...", file=sys.stderr)
    for i in range(resume_from, total, batch_size):
        batch = definitions[i : i + batch_size]
        matrix[i : i + len(batch)] = embed_units(batch, model_name=resolved_model)
        matrix.flush()  # on disk now -- a kill/crash right after this line loses nothing already done
        completed = i + len(batch)
        progress_path.write_text(json.dumps({"total": total, "dim": dim, "completed": completed}), encoding="utf-8")
        print(f"[sense] {completed}/{total} ({100 * completed / total:.1f}%)", file=sys.stderr, flush=True)

        free_bytes = shutil.disk_usage(resolved_root).free
        if free_bytes < min_free_bytes and completed < total:
            print(
                f"[sense] only {free_bytes / 1024**3:.2f}GB free on {resolved_root} (below "
                f"min_free_gb={min_free_gb}) -- stopping at {completed}/{total}, safe to resume "
                "later by calling this again with the same model_name/root",
                file=sys.stderr,
            )
            return

    progress_path.unlink(missing_ok=True)  # done -- no resume marker needed
    print(f"[sense] wrote {total} synsets -> {resolved_root}", file=sys.stderr)


def ensure_wordnet_index_ready(
    *,
    model_name: str | None = None,
    root: Path | None = None,
    batch_size: int = 256,
    min_free_gb: float = 2.0,
) -> None:
    """Guarantees the WordNet index is present and complete before a real
    pipeline run starts resolving senses with it — building or resuming it
    first (`build_wordnet_index`) if it's missing or was left incomplete by
    an interrupted earlier run (`_index_is_complete`). Called from
    `align.align_corpus_folder` for `mode="hybrid"`/`"llm"` (the two modes
    that ever call `resolve_sense_id`) — never from `resolve_sense_id`
    itself, nor from `align_corpus_pair` directly, so a unit test or a
    direct low-level call keeps today's soft-fail/mint-only behavior
    instead of silently kicking off a many-hour precompute.

    Clears `_load_wordnet_index`'s cache after building, in case something
    in this same process already called it (and got `None`, or a stale
    partial read) before the index was actually complete.
    """
    resolved_model = model_name or get_settings().embedding_model
    resolved_root = _resolve_root(root)
    if _index_is_complete(resolved_model, resolved_root):
        return
    print(
        f"[sense] WordNet index for {resolved_model!r} at {resolved_root} is missing or incomplete -- "
        "building it now before continuing (this can take hours the first time; see build_wordnet_index)",
        file=sys.stderr,
    )
    build_wordnet_index(model_name=resolved_model, root=resolved_root, batch_size=batch_size, min_free_gb=min_free_gb)
    _load_wordnet_index.cache_clear()


def _minted_index_path(model_name: str, root: Path) -> Path:
    return root / f"minted.{model_slug(model_name)}.json"


def _load_minted_index(model_name: str, root: Path) -> list[dict[str, Any]]:
    path = _minted_index_path(model_name, root)
    if not path.is_file():
        return []
    entries = json.loads(path.read_text(encoding="utf-8")).get("entries")
    return entries if isinstance(entries, list) else []


def _write_minted_index(model_name: str, root: Path, entries: list[dict[str, Any]]) -> None:
    path = _minted_index_path(model_name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entries": entries}, ensure_ascii=False, indent=2), encoding="utf-8")


def _embed_gloss(gloss: str, *, model_name: str) -> np.ndarray:
    embedding: np.ndarray = embed_units([gloss], model_name=model_name)[0]
    return embedding


def _best_match(
    query: np.ndarray,
    wn_pos: str,
    upos: str,
    wordnet_data: tuple[np.ndarray, list[str], list[str], list[str]] | None,
    minted_entries: list[dict[str, Any]],
) -> tuple[str, str, float] | None:
    """The single best-matching candidate (by cosine similarity) across
    both indices, restricted to `wn_pos`/`upos` — or `None` if neither
    index has any candidate of that POS at all (an empty pool, not a
    below-threshold miss; the caller applies `SIMILARITY_THRESHOLD`).
    """
    best_id: str | None = None
    best_definition: str | None = None
    best_similarity = -1.0

    if wordnet_data is not None:
        embeddings, names, definitions, pos_tags = wordnet_data
        mask = np.asarray(pos_tags) == wn_pos
        if mask.any():
            similarities = embeddings[mask] @ query
            local_idx = int(np.argmax(similarities))
            similarity = float(similarities[local_idx])
            if similarity > best_similarity:
                matched_name = np.asarray(names, dtype=object)[mask][local_idx]
                matched_def = np.asarray(definitions, dtype=object)[mask][local_idx]
                best_id, best_definition, best_similarity = f"en:{matched_name}", str(matched_def), similarity

    for entry in minted_entries:
        if entry.get("upos") != upos:
            continue
        vector = np.asarray(entry["embedding"], dtype=np.float32)
        similarity = float(vector @ query)
        if similarity > best_similarity:
            best_id, best_definition, best_similarity = entry["id"], entry["gloss"], similarity

    if best_id is None or best_definition is None:
        return None
    return best_id, best_definition, best_similarity


def resolve_sense_id(
    lemma_hint: str, upos: str, gloss: str, *, model_name: str | None = None, root: Path | None = None
) -> tuple[str, str]:
    """Returns `(sense_id, definition)` for `gloss` (an LLM-proposed
    English definition of the sense a link/token expresses — see module
    docstring for why matching is gloss-embedding-based, not lemma-lookup-
    based). `lemma_hint`/`upos` are used only to (a) restrict matching to
    same-POS candidates and (b) name a newly-minted id readably — they do
    not otherwise affect which existing sense gets matched.

    `sense_id` is `en:<synset_name>` when the best-matching candidate
    (across both the WordNet and minted indices, §module docstring) is a
    WordNet synset above `SIMILARITY_THRESHOLD`; the *existing* minted id
    if the best match is an already-minted sense (reuse, not a fresh mint);
    otherwise a newly minted `ye_olde:<slug>.<pos>.<NN>`, persisted to the
    minted index so a later, similar gloss finds and reuses it next time.
    """
    wn_pos = _WORDNET_POS.get(upos.upper())
    if wn_pos is None:
        # Closed-class UPOS: no real WordNet senses to distinguish, and
        # nothing worth persisting an embedding for either -- mint and stop.
        # Nothing is ever persisted for this path (see _mint), so there's
        # no cross-process index to seed a counter from either; unlike the
        # open-class path below, an in-memory-only counter is the best
        # available and the stakes are low (closed-class mints are never
        # matched/reused by anything).
        return _mint(lemma_hint, upos, gloss, embedding=None, model_name=None, root=None, existing_minted=None)

    resolved_root = _resolve_root(root)
    resolved_model = model_name or get_settings().embedding_model
    query = _embed_gloss(gloss, model_name=resolved_model)

    wordnet_data = _load_wordnet_index(resolved_model, resolved_root)
    minted_entries = _load_minted_index(resolved_model, resolved_root)  # one load, reused by _mint below if needed
    match = _best_match(query, wn_pos, upos.upper(), wordnet_data, minted_entries)
    if match is not None and match[2] >= SIMILARITY_THRESHOLD:
        return match[0], match[1]

    return _mint(
        lemma_hint,
        upos,
        gloss,
        embedding=query,
        model_name=resolved_model,
        root=resolved_root,
        existing_minted=minted_entries,
    )


def _mint(
    lemma_hint: str,
    upos: str,
    gloss: str,
    *,
    embedding: np.ndarray | None,
    model_name: str | None,
    root: Path | None,
    existing_minted: list[dict[str, Any]] | None,
) -> tuple[str, str]:
    """Mints a new sense id. The numeric suffix is derived from
    `existing_minted` (the *persisted* minted index, already loaded by the
    caller) when available — never from `_mint_counters` alone, which is
    process-lifetime, in-memory state that restarts at 0 every run. Seeding
    only from that in-memory counter meant a second ingestion process could
    mint `ye_olde:wergild.n.01` again for a genuinely *different* sense,
    colliding with one a prior run already persisted under that same id —
    exactly the collision the persisted index exists to prevent (module
    docstring). `existing_minted` is `None` only for the closed-class path,
    which never persists anything to seed from in the first place.
    """
    pos_tag = _WORDNET_POS.get(upos.upper()) or "x"
    key = f"{_slugify(lemma_hint or 'unknown')}.{pos_tag}"
    if existing_minted is not None:
        prefix = f"ye_olde:{key}."
        suffixes = [
            int(raw_id[len(prefix) :])
            for entry in existing_minted
            if (raw_id := str(entry.get("id", ""))).startswith(prefix) and raw_id[len(prefix) :].isdigit()
        ]
        suffix = max(suffixes, default=0) + 1
    else:
        suffix = _mint_counters.get(key, 0) + 1
    _mint_counters[key] = suffix
    new_id = f"ye_olde:{key}.{suffix:02d}"
    if embedding is not None and model_name is not None and root is not None:
        new_entry = {"id": new_id, "upos": upos.upper(), "gloss": gloss, "embedding": embedding.tolist()}
        _write_minted_index(model_name, root, [*existing_minted, new_entry] if existing_minted else [new_entry])
    return new_id, gloss
