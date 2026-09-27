"""Downloads and reads Parquet shards from the HF Dataset repo (spec §10).

Shard layout (spec §10 / §3c):
  relational/<iso_code>/<year_from>-<year_to>.parquet
  pairs/<lang_a>_<lang_b>/<year_from>-<year_to>.parquet
  vectors/pairs/<lang_a>_<lang_b>/<year_from>-<year_to>.faiss
  ngrams/pairs/<lang_a>_<lang_b>/<year_from>-<year_to>.parquet
  bm25/pairs/<lang_a>_<lang_b>/<year_from>-<year_to>.parquet

The vector/n-gram/BM25 shards are all written by ye_olde.ingest.index,
keyed by `row_index` (position in the *same-named* pairs Parquet shard) —
FAISS ids additionally encode `side` (`row_index * 2 (+1 for target)`),
n-gram/BM25 postings carry `row_index`/`side` as plain columns. That keying
only means anything paired with that exact shard's own row order, which is
why `load_pair_shards` below keeps each shard's table, vector index, and
shard path together (so n-gram/BM25 postings can be derived and loaded
per-shard too) instead of concatenating tables the way `load_relational`
does for the simpler exact-match case.

Every lookup here is soft-fail: no dataset repo configured yet, no repo
found, or no shards for a given iso_code/lang-pair yet, all return an empty
table (or []) rather than raising — this service is meant to come up and
serve correctly before any corpus has been gathered (spec §11).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import faiss
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from . import lexical
from .config import get_settings
from .embedding import embed_query
from .schemas import HighlightedSpan, LookupResult, SearchResult

_SHARD_RE = re.compile(r"^(?P<prefix>.+)/(?P<start>\d+)-(?P<end>\d+)\.parquet$")
_PAIRS_DIR_RE = re.compile(r"^pairs/(?P<lang_a>[a-z]+)_(?P<lang_b>[a-z]+)/")

# How many raw FAISS candidates to fetch per shard before hybrid reranking —
# generous on purpose, same reasoning as retrieval/'s own _FETCH_K on the
# ye_olde side: the lexical signals need a wide sample to be meaningful, not
# just the final top_k the caller wants back.
_FETCH_K = 100

# Lexical sub-blend: how much of the "lexical" half of the hybrid score
# comes from n-gram similarity vs. BM25 — open, unvalidated (spec §9),
# starting even.
_NGRAM_BM25_WEIGHT = 0.5

# /lookup: both the minimum n-gram Dice similarity to enter the
# edit-distance-refinement shortlist, and the maximum edit distance still
# accepted as a match, are length-adaptive, not flat numbers — an absolute
# edit count turned out to be just as length-unstable as Dice is (2 edits
# is most of a 3-letter word, negligible for a 10-letter one). See
# lexical.lookup_ngram_threshold / lexical.lookup_max_edit_distance.


@lru_cache(maxsize=1)
def _list_repo_files() -> tuple[str, ...]:
    settings = get_settings()
    if not settings.hf_dataset_repo_id:
        return ()
    try:
        api = HfApi(token=settings.hf_token or None)
        return tuple(api.list_repo_files(settings.hf_dataset_repo_id, repo_type="dataset"))
    except RepositoryNotFoundError:
        return ()


def refresh() -> None:
    """Drop the cached file listing — call after new data shards are pushed."""
    _list_repo_files.cache_clear()


def _matching_shards(prefix: str, year_from: int | None, year_to: int | None) -> list[str]:
    out = []
    for path in _list_repo_files():
        match = _SHARD_RE.match(path)
        if not match or not path.startswith(prefix + "/"):
            continue
        shard_start, shard_end = int(match.group("start")), int(match.group("end"))
        if year_from is not None and shard_end < year_from:
            continue
        if year_to is not None and shard_start > year_to:
            continue
        out.append(path)
    return out


def _download(path: str) -> str | None:
    settings = get_settings()
    try:
        return hf_hub_download(
            repo_id=settings.hf_dataset_repo_id,
            repo_type="dataset",
            filename=path,
            token=settings.hf_token or None,
            cache_dir=settings.hf_cache_dir,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None


def _load_shards(shards: list[str], year_from: int | None, year_to: int | None) -> pa.Table:
    filters = None
    if year_from is not None or year_to is not None:
        filters = []
        if year_from is not None:
            filters.append(("attestation_year", ">=", year_from))
        if year_to is not None:
            filters.append(("attestation_year", "<=", year_to))
    tables = []
    for shard in shards:
        local_path = _download(shard)
        if local_path is None:
            continue
        tables.append(pq.read_table(local_path, filters=filters))
    if not tables:
        return pa.table({})
    return pa.concat_tables(tables)


def load_relational(iso_code: str, year_from: int | None = None, year_to: int | None = None) -> pa.Table:
    shards = _matching_shards(f"relational/{iso_code}", year_from, year_to)
    return _load_shards(shards, year_from, year_to)


def _sibling_shard_path(parquet_shard_path: str, *, prefix: str, suffix: str) -> str:
    """Maps e.g. pairs/enm_eng/1400-1999.parquet to <prefix>/enm_eng/1400-1999<suffix>
    — the naming convention ye_olde.ingest.index writes every artifact of a
    shard under (see that module's docstring)."""
    stem = parquet_shard_path[: -len(".parquet")]
    return f"{prefix}/{stem}{suffix}"


def _vector_shard_path(parquet_shard_path: str) -> str:
    return _sibling_shard_path(parquet_shard_path, prefix="vectors", suffix=".faiss")


def _ngram_shard_path(parquet_shard_path: str) -> str:
    return _sibling_shard_path(parquet_shard_path, prefix="ngrams", suffix=".parquet")


def _bm25_shard_path(parquet_shard_path: str) -> str:
    return _sibling_shard_path(parquet_shard_path, prefix="bm25", suffix=".parquet")


def load_pair_shards(lang_a: str, lang_b: str) -> list[tuple[pa.Table, faiss.Index | None, str]]:
    """Finds every pairs shard for `(lang_a, lang_b)` (or the reverse
    direction, whichever the alignment job actually wrote — same fallback
    as `load_relational`'s exact-match case), keeping each shard's table
    paired with its own FAISS index and its own shard path, rather than
    concatenating tables across shards — a FAISS id, and an n-gram/BM25
    posting's `row_index`, are only meaningful against the exact shard they
    were built from, so shards must never be merged before resolving a hit
    back to a row. The shard path (the `pairs/...` path itself) is returned
    so callers can derive and load that same shard's n-gram/BM25 postings on demand
    (`load_ngram_postings`/`load_bm25_postings`) without re-deriving shard
    discovery themselves. The index is None for a shard that has a pairs
    Parquet file but no matching vector file yet (soft-fail, same
    philosophy as the rest of this module) — that shard's rows are simply
    skipped by semantic search without affecting the others.
    """
    shard_paths = _matching_shards(f"pairs/{lang_a}_{lang_b}", None, None)
    if not shard_paths:
        shard_paths = _matching_shards(f"pairs/{lang_b}_{lang_a}", None, None)
    shards: list[tuple[pa.Table, faiss.Index | None, str]] = []
    for shard_path in shard_paths:
        local_parquet = _download(shard_path)
        if local_parquet is None:
            continue
        table = pq.read_table(local_parquet)
        local_faiss = _download(_vector_shard_path(shard_path))
        vector_index = faiss.read_index(local_faiss) if local_faiss is not None else None
        shards.append((table, vector_index, shard_path))
    return shards


def load_ngram_postings(shard_path: str, ngrams: set[str]) -> pa.Table:
    """Filtered read of the n-gram postings shard matching `shard_path` (a
    `pairs/...` path) for only `ngrams` — the inverted-index lookup step
    (spec §10), same technique as PostgreSQL's `pg_trgm`: the Parquet
    equality filter is applied at read time, so this never loads a posting
    for a trigram the query doesn't contain. Empty table if no n-gram shard
    exists yet for this pairs shard, or `ngrams` is empty (soft-fail).
    """
    if not ngrams:
        return pa.table({})
    local_path = _download(_ngram_shard_path(shard_path))
    if local_path is None:
        return pa.table({})
    return pq.read_table(local_path, filters=[("ngram", "in", list(ngrams))])


def load_bm25_postings(shard_path: str, terms: set[str]) -> pa.Table:
    """Filtered read of the BM25 postings shard matching `shard_path` for
    only `terms` — same technique as `load_ngram_postings`. Empty table if
    no BM25 shard exists yet, or `terms` is empty (soft-fail).
    """
    if not terms:
        return pa.table({})
    local_path = _download(_bm25_shard_path(shard_path))
    if local_path is None:
        return pa.table({})
    return pq.read_table(local_path, filters=[("term", "in", list(terms))])


def _lang_pair_dirs_containing(lang: str) -> list[tuple[str, str]]:
    """Every `(lang_a, lang_b)` pair directory in the repo that has `lang`
    on either side — lets /search look across every indexed pair without
    the caller needing to already know the counterpart language.
    """
    found: set[tuple[str, str]] = set()
    for path in _list_repo_files():
        match = _PAIRS_DIR_RE.match(path)
        if not match:
            continue
        lang_a, lang_b = match.group("lang_a"), match.group("lang_b")
        if lang in (lang_a, lang_b):
            found.add((lang_a, lang_b))
    return sorted(found)


@dataclass
class _Hit:
    """One FAISS candidate's raw score plus everything needed to build a
    SearchResult — explicit typed fields rather than a dict, so building the
    final SearchResult can pass them as real keyword arguments instead of
    `**`-unpacking an untyped dict (mypy strict correctly rejects that
    against a model with per-field types, same reasoning that shaped this
    function's SearchResult construction before hybrid scoring existed).
    """

    row_idx: int
    side: int
    faiss_score: float
    pair_id: str
    lang: str
    year: int
    work: str | None
    text: str
    other_lang: str
    other_year: int
    other_work: str | None
    other_text: str
    citation: str


def _to_search_result(hit: _Hit, score: float) -> SearchResult:
    return SearchResult(
        pair_id=hit.pair_id,
        lang=hit.lang,
        year=hit.year,
        work=hit.work,
        text=hit.text,
        score=score,
        other_lang=hit.other_lang,
        other_year=hit.other_year,
        other_work=hit.other_work,
        other_text=hit.other_text,
        citation=hit.citation,
    )


def search_passages(
    lang: str,
    text: str,
    *,
    year: int | None = None,
    window: int | None = None,
    top_k: int = 5,
) -> list[SearchResult]:
    """Semantic + lexical hybrid passage search (spec §10's "Hybrid
    retrieval scoring"). Embeds `text` with the same model the indexed
    shards were built with (settings.embedding_model), fetches a wide FAISS
    candidate set per shard, and — when `year` is given — reranks it as a
    normalized weighted blend of FAISS similarity, character-trigram
    sentence similarity, and BM25, weighted by how far `year` is from the
    present (`lexical.semantic_weight`): `multilingual-e5-large` has
    essentially no training exposure to `enm`/`ang` orthography, so the
    lexical signals matter more the further back the query's era is.

    `year`/`window` together still apply the original hard include/exclude
    filter (unchanged); `year` alone (no `window`) only feeds the semantic-
    weight calculation — this is a deliberate split: `ye_olde.retrieval`
    always wants the era-aware weighting but must never trigger the hard
    filter (it would bias the local-density count that module's own
    reranking depends on, spec §9), so `window` defaulting to `None` rather
    than a fixed number is what makes that possible. Omitting `year`
    entirely returns pure FAISS similarity, unchanged from before this was
    added — the hybrid blend needs an era to weight against.

    Returns SearchResult directly (unlike load_relational, which returns a
    bare pa.Table for main.py to shape) — the row here is already a
    fixed, known shape, so building the pydantic model here rather than
    passing an untyped dict up keeps this typed end to end. Soft-fails to []
    wherever a shard, vector, n-gram, or BM25 file doesn't exist yet, same
    as the rest of this module.
    """
    settings = get_settings()
    query_vector = embed_query(text, model_name=settings.embedding_model, hf_token=settings.hf_token)
    query_trigrams = lexical.char_trigrams(text)
    query_terms = {t.lower() for t in text.split() if t}

    # (hit, ngram_score, bm25_score) per candidate, across every shard --
    # gathered fully before normalizing, since normalization has to see the
    # whole candidate set to mean anything.
    raw: list[tuple[_Hit, float, float]] = []

    for lang_a, lang_b in _lang_pair_dirs_containing(lang):
        for table, vector_index, shard_path in load_pair_shards(lang_a, lang_b):
            if vector_index is None or table.num_rows == 0:
                continue
            k = min(_FETCH_K, vector_index.ntotal)
            distances, ids = vector_index.search(query_vector, k)
            rows = table.to_pylist()

            shard_hits: list[_Hit] = []
            for score, vec_id in zip(distances[0], ids[0], strict=True):
                if vec_id < 0:
                    continue
                row_idx, side = divmod(int(vec_id), 2)
                if row_idx >= len(rows):
                    continue
                row = rows[row_idx]
                query_side = row["source"] if side == 0 else row["target"]
                other_side = row["target"] if side == 0 else row["source"]
                if query_side["lang_code"] != lang:
                    continue  # defensive: shard's own tag disagrees with the folder name
                if year is not None and window is not None and abs(query_side["year"] - year) > window:
                    continue
                query_text_key = "source_text" if side == 0 else "target_text"
                other_text_key = "target_text" if side == 0 else "source_text"
                shard_hits.append(
                    _Hit(
                        row_idx=row_idx,
                        side=side,
                        faiss_score=float(score),
                        pair_id=row["pair_id"],
                        lang=query_side["lang_code"],
                        year=query_side["year"],
                        work=query_side.get("work"),
                        text=row[query_text_key],
                        other_lang=other_side["lang_code"],
                        other_year=other_side["year"],
                        other_work=other_side.get("work"),
                        other_text=row[other_text_key],
                        citation=row.get("citation") or "",
                    )
                )

            if not shard_hits:
                continue

            if year is None:
                # No era given -- can't weight by trust-in-era, so this
                # stays pure FAISS similarity, same as before hybrid scoring
                # existed.
                raw.extend((hit, 0.0, 0.0) for hit in shard_hits)
                continue

            # Lexical signal 1: n-gram sentence similarity, computed on the
            # fly against just this shard's already-FAISS-narrowed
            # candidates. No need to consult the persisted n-gram index
            # here — that's for /lookup's un-narrowed candidate generation,
            # where nothing has pre-filtered the field yet.
            ngram_scores = [
                lexical.dice_coefficient(query_trigrams, lexical.char_trigrams(hit.text)) for hit in shard_hits
            ]

            # Lexical signal 2: BM25. Document/term frequency aren't
            # derivable from the small FAISS-narrowed set alone, so this is
            # the one signal that does need the persisted postings index.
            side = shard_hits[0].side  # constant across a shard for a fixed lang, see this module's docstring
            tokens_key = "source_tokens" if side == 0 else "target_tokens"
            doc_lens = [len(r.get(tokens_key) or []) for r in rows]
            avg_doc_len = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0
            corpus_size = len(rows)

            postings = load_bm25_postings(shard_path, query_terms).to_pylist()
            tf_by_row: dict[int, dict[str, int]] = {}
            docs_by_term: dict[str, set[int]] = {}
            for posting in postings:
                if posting["side"] != side:
                    continue
                tf_by_row.setdefault(posting["row_index"], {})[posting["term"]] = posting["term_frequency"]
                docs_by_term.setdefault(posting["term"], set()).add(posting["row_index"])
            doc_freqs = {term: len(docs) for term, docs in docs_by_term.items()}

            bm25_scores = [
                lexical.bm25_score(
                    tf_by_row.get(hit.row_idx, {}),
                    doc_freqs,
                    doc_len=doc_lens[hit.row_idx],
                    avg_doc_len=avg_doc_len,
                    corpus_size=corpus_size,
                )
                for hit in shard_hits
            ]

            raw.extend(zip(shard_hits, ngram_scores, bm25_scores, strict=True))

    if not raw:
        return []

    candidates: list[SearchResult] = []
    if year is None:
        for hit, _, _ in raw:
            candidates.append(_to_search_result(hit, hit.faiss_score))
    else:
        faiss_norm = lexical.normalize([hit.faiss_score for hit, _, _ in raw])
        ngram_norm = lexical.normalize([n for _, n, _ in raw])
        bm25_norm = lexical.normalize([b for _, _, b in raw])
        sw = lexical.semantic_weight(year)
        for i, (hit, _, _) in enumerate(raw):
            lexical_component = _NGRAM_BM25_WEIGHT * ngram_norm[i] + (1 - _NGRAM_BM25_WEIGHT) * bm25_norm[i]
            hybrid = sw * faiss_norm[i] + (1 - sw) * lexical_component
            candidates.append(_to_search_result(hit, hybrid))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]


def lookup_passages(text: str, lang: str, target_lang: str) -> list[LookupResult]:
    """Linguee-style aligned lookup (spec §3c): given a word/phrase in
    `lang`, find attested sentences in `target_lang` with the corresponding
    span highlighted. Matching is a character-trigram index (candidate
    generation) + edit-distance refinement (spec §10's "Hybrid retrieval
    scoring") — not exact-token-only. An exact match is just the
    zero-edit-distance case, so this generalizes the original behavior
    rather than replacing it.

    Deliberately no semantic (embedding) signal: the FAISS vectors are
    sentence-level, never per-token, so there's no meaningful "embedding
    score for this specific word" to blend with a lexical one — embedding
    similarity could only ever help pick *which sentence* is relevant,
    never *which word within it* to highlight, and highlighting is this
    endpoint's entire job. Matching is character-only, uniformly across
    every era, for that reason.
    """
    query_trigrams = lexical.char_trigrams(text)
    results: list[LookupResult] = []

    for table, _vector_index, shard_path in load_pair_shards(lang, target_lang):
        if table.num_rows == 0:
            continue
        rows = table.to_pylist()
        # Which side is the query's language is fixed for the whole shard
        # (one align_corpus_pair call per shard, so every row shares the
        # same source/target language roles) — checked once, not per row.
        first = rows[0]
        if (first.get("source") or {}).get("lang_code") == lang:
            query_side = 0
            query_text_key, other_text_key = "source_text", "target_text"
            query_tokens_key, other_tokens_key = "source_tokens", "target_tokens"
            link_query_key, link_other_key = "source_idx", "target_idx"
        elif (first.get("target") or {}).get("lang_code") == lang:
            query_side = 1
            query_text_key, other_text_key = "target_text", "source_text"
            query_tokens_key, other_tokens_key = "target_tokens", "source_tokens"
            link_query_key, link_other_key = "target_idx", "source_idx"
        else:
            continue  # shouldn't happen — load_pair_shards already scoped this to (lang, target_lang)

        # Candidate generation: which (row, token) share at least one
        # trigram with the query at all — the inverted-index step. Filtered
        # to `query_side`: a row where source and target coincidentally
        # share the same word at the same token_index (rare, but not
        # impossible — e.g. a proper noun kept unchanged across periods)
        # would otherwise have its postings from *both* sides merged into
        # one (row_index, token_index) key, since token_index alone doesn't
        # distinguish them.
        postings = load_ngram_postings(shard_path, query_trigrams).to_pylist()
        shared_counts: dict[tuple[int, int], int] = {}
        token_ngram_counts: dict[tuple[int, int], int] = {}
        for posting in postings:
            if posting["side"] != query_side:
                continue
            key = (posting["row_index"], posting["token_index"])
            shared_counts[key] = shared_counts.get(key, 0) + 1
            token_ngram_counts[key] = posting["token_ngram_count"]

        candidates_by_row: dict[int, set[int]] = {}
        for (row_idx, token_idx), shared in shared_counts.items():
            token_ngram_count = token_ngram_counts[(row_idx, token_idx)]
            denom = len(query_trigrams) + token_ngram_count
            dice = (2 * shared / denom) if denom else 0.0
            # Length-adaptive, not a flat number -- see lexical's own
            # comment on why (þ->th and similar substitutions need this).
            pair_length = len(text) + lexical.token_char_length(token_ngram_count)
            if dice >= lexical.lookup_ngram_threshold(pair_length):
                candidates_by_row.setdefault(row_idx, set()).add(token_idx)

        for row_idx, token_candidates in candidates_by_row.items():
            if row_idx >= len(rows):
                continue
            row = rows[row_idx]
            query_tokens = row.get(query_tokens_key) or []
            # Refinement: n-gram alone can be fooled by transpositions and
            # doesn't rank near-misses well; edit distance is precise but
            # too expensive to run un-indexed against a whole corpus, so it
            # only ever runs against this already-narrowed shortlist. The
            # accepted ceiling is length-adaptive here too (see lexical's
            # own comment on why) — the real token string is on hand at
            # this stage, so pair_length just uses len() directly.
            match_idx = {
                idx
                for idx in token_candidates
                if idx < len(query_tokens)
                and lexical.levenshtein(text, query_tokens[idx])
                <= lexical.lookup_max_edit_distance(len(text) + len(query_tokens[idx]))
            }
            if not match_idx:
                continue

            other_tokens = row.get(other_tokens_key) or []
            span = None
            for link in row.get("alignment_links") or []:
                if match_idx & set(link[link_query_key]):
                    span_idx = link[link_other_key]
                    span = HighlightedSpan(token_idx=span_idx, surface=" ".join(other_tokens[i] for i in span_idx))
                    break
            results.append(
                LookupResult(
                    pair_id=row["pair_id"],
                    target_sentence=row[other_text_key],
                    highlighted_span=span,
                    source_sentence=row[query_text_key],
                    citation=row.get("citation") or "",
                    confidence=row.get("sentence_confidence") or 0.0,
                )
            )
    return results
