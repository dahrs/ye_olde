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
  — every open-class synset's definition, embedded once by
  `scripts/build_wordnet_sense_index.py` and loaded read-only here. Not
  built automatically by this module — a real precompute over ~100k+
  definitions, run deliberately, not as a side effect of the first
  `resolve_sense_id` call.
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
