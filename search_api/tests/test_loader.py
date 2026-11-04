"""Unit tests for app/loader.py's own helper functions.

`lookup_passages`/`search_passages` (the two big public entrypoints) are
already exercised end-to-end through the FastAPI routes in
`test_lookup.py`/`test_search.py`/`test_attest.py` -- those tests mock
`load_pair_shards`/`load_relational`/`_lang_pair_dirs_containing`
*themselves*, which means the shard-discovery/download/caching machinery
those functions are built from (`_list_repo_files`, `_matching_shards`,
`_download`, `_load_shards`, the `_*_shard_path` family, and
`load_pair_shards`'s own directory-fallback) is never actually exercised
by them. This file covers exactly that layer, plus the small pure tag-
matching helpers (`_side_prefix`, `_link_tags`, `_tag_preference_rank`,
`_matching_link_tags`, `_apply_tag_boost`, `_to_search_result`) directly
rather than only incidentally through the bigger functions above.
"""

from __future__ import annotations

import types

import faiss
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from app import loader
from app.schemas import LookupResult


@pytest.fixture(autouse=True)
def _clear_repo_file_cache():
    loader._list_repo_files.cache_clear()
    yield
    loader._list_repo_files.cache_clear()


def _fake_response() -> types.SimpleNamespace:
    return types.SimpleNamespace(headers={}, request=types.SimpleNamespace(url="https://huggingface.co/x"))


# ---------------------------------------------------------------------------
# _list_repo_files / refresh
# ---------------------------------------------------------------------------


def test_list_repo_files_returns_empty_when_no_repo_id_configured(monkeypatch):
    monkeypatch.setattr(loader, "get_settings", lambda: types.SimpleNamespace(hf_dataset_repo_id="", hf_token=""))

    assert loader._list_repo_files() == ()


def test_list_repo_files_returns_the_repos_file_list(monkeypatch):
    monkeypatch.setattr(
        loader, "get_settings", lambda: types.SimpleNamespace(hf_dataset_repo_id="dahrs/idx", hf_token="")
    )

    class _FakeHfApi:
        def __init__(self, token):
            self.token = token

        def list_repo_files(self, repo_id, repo_type):
            return ["pairs/enm_eng/1400-1999.parquet"]

    monkeypatch.setattr(loader, "HfApi", _FakeHfApi)

    assert loader._list_repo_files() == ("pairs/enm_eng/1400-1999.parquet",)


def test_list_repo_files_returns_empty_when_repository_not_found(monkeypatch):
    monkeypatch.setattr(
        loader, "get_settings", lambda: types.SimpleNamespace(hf_dataset_repo_id="dahrs/missing", hf_token="")
    )

    class _FakeHfApi:
        def __init__(self, token):
            pass

        def list_repo_files(self, repo_id, repo_type):
            raise loader.RepositoryNotFoundError("not found", response=_fake_response())

    monkeypatch.setattr(loader, "HfApi", _FakeHfApi)

    assert loader._list_repo_files() == ()


def test_list_repo_files_caches_across_calls(monkeypatch):
    monkeypatch.setattr(
        loader, "get_settings", lambda: types.SimpleNamespace(hf_dataset_repo_id="dahrs/idx", hf_token="")
    )
    call_count = 0

    class _FakeHfApi:
        def __init__(self, token):
            pass

        def list_repo_files(self, repo_id, repo_type):
            nonlocal call_count
            call_count += 1
            return ["a.parquet"]

    monkeypatch.setattr(loader, "HfApi", _FakeHfApi)

    loader._list_repo_files()
    loader._list_repo_files()

    assert call_count == 1


def test_refresh_clears_the_cache(monkeypatch):
    monkeypatch.setattr(
        loader, "get_settings", lambda: types.SimpleNamespace(hf_dataset_repo_id="dahrs/idx", hf_token="")
    )
    call_count = 0

    class _FakeHfApi:
        def __init__(self, token):
            pass

        def list_repo_files(self, repo_id, repo_type):
            nonlocal call_count
            call_count += 1
            return ["a.parquet"]

    monkeypatch.setattr(loader, "HfApi", _FakeHfApi)

    loader._list_repo_files()
    loader.refresh()
    loader._list_repo_files()

    assert call_count == 2


# ---------------------------------------------------------------------------
# _matching_shards
# ---------------------------------------------------------------------------

_SHARD_LIST = (
    "pairs/enm_eng/1400-1999.parquet",
    "pairs/enm_eng/1600-1634.parquet",
    "relational/eng/1600-1634.parquet",
    "pairs/enm_eng/not-a-shard.txt",
)


def test_matching_shards_filters_by_prefix(monkeypatch):
    monkeypatch.setattr(loader, "_list_repo_files", lambda: _SHARD_LIST)

    assert loader._matching_shards("relational/eng", None, None) == ["relational/eng/1600-1634.parquet"]


def test_matching_shards_returns_everything_under_the_prefix_with_no_year_range(monkeypatch):
    monkeypatch.setattr(loader, "_list_repo_files", lambda: _SHARD_LIST)

    result = loader._matching_shards("pairs/enm_eng", None, None)

    assert result == ["pairs/enm_eng/1400-1999.parquet", "pairs/enm_eng/1600-1634.parquet"]


def test_matching_shards_excludes_a_shard_that_ends_before_the_requested_start_year(monkeypatch):
    monkeypatch.setattr(loader, "_list_repo_files", lambda: ("pairs/enm_eng/1000-1300.parquet",))

    assert loader._matching_shards("pairs/enm_eng", 1700, None) == []


def test_matching_shards_keeps_a_shard_whose_range_overlaps_the_requested_window(monkeypatch):
    monkeypatch.setattr(loader, "_list_repo_files", lambda: _SHARD_LIST)

    result = loader._matching_shards("pairs/enm_eng", 1500, 1700)

    # 1400-1999 overlaps (ends after 1500); 1600-1634 overlaps entirely; both kept.
    assert result == ["pairs/enm_eng/1400-1999.parquet", "pairs/enm_eng/1600-1634.parquet"]


def test_matching_shards_excludes_a_shard_entirely_after_the_requested_range(monkeypatch):
    monkeypatch.setattr(loader, "_list_repo_files", lambda: ("pairs/enm_eng/1900-1999.parquet",))

    assert loader._matching_shards("pairs/enm_eng", None, 1500) == []


def test_matching_shards_ignores_a_filename_that_doesnt_match_the_shard_pattern(monkeypatch):
    monkeypatch.setattr(loader, "_list_repo_files", lambda: _SHARD_LIST)

    result = loader._matching_shards("pairs/enm_eng", None, None)

    assert "pairs/enm_eng/not-a-shard.txt" not in result


# ---------------------------------------------------------------------------
# _download
# ---------------------------------------------------------------------------


def test_download_returns_the_downloaded_local_path(monkeypatch):
    monkeypatch.setattr(
        loader,
        "get_settings",
        lambda: types.SimpleNamespace(hf_dataset_repo_id="dahrs/idx", hf_token="", hf_cache_dir="/tmp/hf-cache"),
    )
    monkeypatch.setattr(loader, "hf_hub_download", lambda **kwargs: "/tmp/hf-cache/local/shard.parquet")

    assert loader._download("pairs/enm_eng/1400-1999.parquet") == "/tmp/hf-cache/local/shard.parquet"


def test_download_returns_none_when_the_entry_is_missing(monkeypatch):
    monkeypatch.setattr(
        loader,
        "get_settings",
        lambda: types.SimpleNamespace(hf_dataset_repo_id="dahrs/idx", hf_token="", hf_cache_dir="/tmp/hf-cache"),
    )

    def raise_entry_not_found(**kwargs):
        raise loader.EntryNotFoundError("no such file")

    monkeypatch.setattr(loader, "hf_hub_download", raise_entry_not_found)

    assert loader._download("pairs/enm_eng/missing.parquet") is None


def test_download_returns_none_when_the_repository_is_missing(monkeypatch):
    monkeypatch.setattr(
        loader,
        "get_settings",
        lambda: types.SimpleNamespace(hf_dataset_repo_id="dahrs/missing", hf_token="", hf_cache_dir="/tmp/hf-cache"),
    )

    def raise_repo_not_found(**kwargs):
        raise loader.RepositoryNotFoundError("no such repo", response=_fake_response())

    monkeypatch.setattr(loader, "hf_hub_download", raise_repo_not_found)

    assert loader._download("pairs/enm_eng/1400-1999.parquet") is None


# ---------------------------------------------------------------------------
# _load_shards / load_relational
# ---------------------------------------------------------------------------


def _write_parquet(tmp_path, name: str, rows: list[dict]) -> str:
    path = tmp_path / name
    pq.write_table(pa.Table.from_pylist(rows), path)
    return str(path)


def test_load_shards_returns_an_empty_table_when_every_shard_is_missing(monkeypatch):
    monkeypatch.setattr(loader, "_download", lambda path: None)

    table = loader._load_shards(["relational/eng/1600-1634.parquet"], None, None)

    assert table.num_rows == 0


def test_load_shards_concatenates_multiple_shards(tmp_path, monkeypatch):
    path_a = _write_parquet(tmp_path, "a.parquet", [{"attestation_year": 1600, "form": "a"}])
    path_b = _write_parquet(tmp_path, "b.parquet", [{"attestation_year": 1650, "form": "b"}])
    monkeypatch.setattr(loader, "_download", lambda path: {"a": path_a, "b": path_b}[path])

    table = loader._load_shards(["a", "b"], None, None)

    assert table.num_rows == 2
    assert sorted(table.column("form").to_pylist()) == ["a", "b"]


def test_load_shards_applies_the_year_range_filter(tmp_path, monkeypatch):
    local_path = _write_parquet(
        tmp_path,
        "shard.parquet",
        [
            {"attestation_year": 1500, "form": "too-early"},
            {"attestation_year": 1650, "form": "in-range"},
            {"attestation_year": 1800, "form": "too-late"},
        ],
    )
    monkeypatch.setattr(loader, "_download", lambda path: local_path)

    table = loader._load_shards(["shard"], 1600, 1700)

    assert table.column("form").to_pylist() == ["in-range"]


def test_load_relational_derives_the_prefix_and_delegates(monkeypatch):
    seen: dict[str, object] = {}

    def fake_matching_shards(prefix, year_from, year_to):
        seen["prefix"] = prefix
        return ["s"]

    def fake_load_shards(shards, year_from, year_to):
        seen["shards"] = shards

    monkeypatch.setattr(loader, "_matching_shards", fake_matching_shards)
    monkeypatch.setattr(loader, "_load_shards", fake_load_shards)

    loader.load_relational("eng", 1600, 1700)

    assert seen["prefix"] == "relational/eng"
    assert seen["shards"] == ["s"]


# ---------------------------------------------------------------------------
# shard-path helpers
# ---------------------------------------------------------------------------


def test_vector_shard_path_maps_parquet_to_faiss():
    assert loader._vector_shard_path("pairs/enm_eng/1400-1999.parquet") == "vectors/pairs/enm_eng/1400-1999.faiss"


def test_ngram_shard_path_maps_parquet_to_ngrams_parquet():
    assert loader._ngram_shard_path("pairs/enm_eng/1400-1999.parquet") == "ngrams/pairs/enm_eng/1400-1999.parquet"


def test_bm25_shard_path_maps_parquet_to_bm25_parquet():
    assert loader._bm25_shard_path("pairs/enm_eng/1400-1999.parquet") == "bm25/pairs/enm_eng/1400-1999.parquet"


# ---------------------------------------------------------------------------
# load_pair_shards
# ---------------------------------------------------------------------------


def test_load_pair_shards_falls_back_to_the_reverse_directory_name(monkeypatch):
    # Shard physically lives under pairs/eng_enm/ -- the alignment job ran
    # eng->enm, but a caller asking for (enm, eng) must still find it.
    monkeypatch.setattr(
        loader,
        "_matching_shards",
        lambda prefix, year_from, year_to: ["pairs/eng_enm/1400-1999.parquet"] if prefix == "pairs/eng_enm" else [],
    )
    downloaded_paths: list[str] = []

    def fake_download(path: str) -> None:
        downloaded_paths.append(path)
        return None

    monkeypatch.setattr(loader, "_download", fake_download)

    loader.load_pair_shards("enm", "eng")

    assert downloaded_paths == ["pairs/eng_enm/1400-1999.parquet"]


def test_load_pair_shards_skips_a_shard_whose_parquet_file_is_missing(monkeypatch):
    monkeypatch.setattr(loader, "_matching_shards", lambda prefix, year_from, year_to: ["pairs/enm_eng/x.parquet"])
    monkeypatch.setattr(loader, "_download", lambda path: None)

    assert loader.load_pair_shards("enm", "eng") == []


def test_load_pair_shards_returns_a_none_vector_index_when_no_faiss_shard_exists(tmp_path, monkeypatch):
    parquet_path = _write_parquet(tmp_path, "shard.parquet", [{"pair_id": "p1"}])
    monkeypatch.setattr(loader, "_matching_shards", lambda prefix, year_from, year_to: ["pairs/enm_eng/x.parquet"])
    monkeypatch.setattr(
        loader,
        "_download",
        lambda path: parquet_path if path == "pairs/enm_eng/x.parquet" else None,
    )

    shards = loader.load_pair_shards("enm", "eng")

    assert len(shards) == 1
    table, vector_index, shard_path = shards[0]
    assert table.num_rows == 1
    assert vector_index is None
    assert shard_path == "pairs/enm_eng/x.parquet"


def test_load_pair_shards_loads_a_real_faiss_index_when_present(tmp_path, monkeypatch):
    parquet_path = _write_parquet(tmp_path, "shard.parquet", [{"pair_id": "p1"}])
    faiss_path = str(tmp_path / "shard.faiss")
    faiss.write_index(faiss.IndexIDMap(faiss.IndexFlatIP(2)), faiss_path)

    monkeypatch.setattr(loader, "_matching_shards", lambda prefix, year_from, year_to: ["pairs/enm_eng/x.parquet"])
    monkeypatch.setattr(
        loader,
        "_download",
        lambda path: {"pairs/enm_eng/x.parquet": parquet_path, "vectors/pairs/enm_eng/x.faiss": faiss_path}.get(path),
    )

    shards = loader.load_pair_shards("enm", "eng")

    assert len(shards) == 1
    _, vector_index, _ = shards[0]
    assert vector_index is not None
    assert vector_index.ntotal == 0


# ---------------------------------------------------------------------------
# load_ngram_postings / load_bm25_postings
# ---------------------------------------------------------------------------


def test_load_ngram_postings_returns_empty_without_downloading_for_an_empty_ngram_set(monkeypatch):
    monkeypatch.setattr(loader, "_download", lambda path: pytest.fail("must not download for an empty query"))

    assert loader.load_ngram_postings("pairs/enm_eng/x.parquet", set()).num_rows == 0


def test_load_ngram_postings_returns_empty_when_no_shard_exists(monkeypatch):
    monkeypatch.setattr(loader, "_download", lambda path: None)

    assert loader.load_ngram_postings("pairs/enm_eng/x.parquet", {"abc"}).num_rows == 0


def test_load_ngram_postings_filters_to_the_requested_ngrams(tmp_path, monkeypatch):
    local_path = _write_parquet(
        tmp_path,
        "ngrams.parquet",
        [
            {"ngram": "abc", "row_index": 0, "side": 0, "token_index": 0, "token_ngram_count": 2},
            {"ngram": "xyz", "row_index": 0, "side": 0, "token_index": 1, "token_ngram_count": 2},
        ],
    )
    monkeypatch.setattr(loader, "_download", lambda path: local_path)

    table = loader.load_ngram_postings("pairs/enm_eng/x.parquet", {"abc"})

    assert table.column("ngram").to_pylist() == ["abc"]


def test_load_bm25_postings_returns_empty_without_downloading_for_an_empty_term_set(monkeypatch):
    monkeypatch.setattr(loader, "_download", lambda path: pytest.fail("must not download for an empty query"))

    assert loader.load_bm25_postings("pairs/enm_eng/x.parquet", set()).num_rows == 0


# ---------------------------------------------------------------------------
# _lang_pair_dirs_containing
# ---------------------------------------------------------------------------


def test_lang_pair_dirs_containing_finds_the_language_on_either_side(monkeypatch):
    monkeypatch.setattr(
        loader,
        "_list_repo_files",
        lambda: ("pairs/enm_eng/1400-1999.parquet", "pairs/lat_eng/1-500.parquet", "pairs/ang_enm/900-1100.parquet"),
    )

    assert loader._lang_pair_dirs_containing("eng") == [("enm", "eng"), ("lat", "eng")]


def test_lang_pair_dirs_containing_deduplicates_and_sorts(monkeypatch):
    monkeypatch.setattr(
        loader,
        "_list_repo_files",
        lambda: ("pairs/enm_eng/1.parquet", "pairs/enm_eng/2.parquet", "pairs/ang_enm/3.parquet"),
    )

    assert loader._lang_pair_dirs_containing("enm") == [("ang", "enm"), ("enm", "eng")]


def test_lang_pair_dirs_containing_ignores_non_pairs_paths(monkeypatch):
    monkeypatch.setattr(loader, "_list_repo_files", lambda: ("relational/eng/1-500.parquet",))

    assert loader._lang_pair_dirs_containing("eng") == []


# ---------------------------------------------------------------------------
# _side_prefix / _link_tags
# ---------------------------------------------------------------------------


def test_side_prefix_zero_is_source_and_one_is_target():
    assert loader._side_prefix(0) == "source"
    assert loader._side_prefix(1) == "target"


def test_link_tags_reads_the_prefixed_fields():
    link = {"source_upos": "NOUN", "source_ner": "PER", "source_lemma": "robin"}

    assert loader._link_tags(link, "source") == ("NOUN", "PER", "robin")


def test_link_tags_returns_none_for_fields_not_present():
    assert loader._link_tags({}, "target") == (None, None, None)


# ---------------------------------------------------------------------------
# _tag_preference_rank
# ---------------------------------------------------------------------------


def _lookup_result(matched_upos=None, matched_ner=None) -> LookupResult:
    return LookupResult(
        pair_id="p1",
        target_sentence="x",
        source_sentence="y",
        citation="z",
        confidence=0.9,
        matched_upos=matched_upos,
        matched_ner=matched_ner,
    )


def test_tag_preference_rank_is_zero_when_nothing_was_queried():
    assert loader._tag_preference_rank(_lookup_result(), upos=None, ner=None) == 0


def test_tag_preference_rank_prefers_an_untagged_result_over_a_disagreeing_one():
    untagged = loader._tag_preference_rank(_lookup_result(matched_upos=None), upos="NOUN", ner=None)
    mismatched = loader._tag_preference_rank(_lookup_result(matched_upos="VERB"), upos="NOUN", ner=None)

    assert untagged < mismatched


def test_tag_preference_rank_prefers_a_matching_tag_over_untagged():
    matching = loader._tag_preference_rank(_lookup_result(matched_upos="NOUN"), upos="NOUN", ner=None)
    untagged = loader._tag_preference_rank(_lookup_result(matched_upos=None), upos="NOUN", ner=None)

    assert matching < untagged


def test_tag_preference_rank_treats_empty_string_query_as_not_queried():
    # Same convention as main.attest's own `ner` param -- "" means "no
    # filter," not "require a literal empty-string tag."
    assert loader._tag_preference_rank(_lookup_result(matched_upos=None), upos="", ner=None) == 0


# ---------------------------------------------------------------------------
# _matching_link_tags / _apply_tag_boost / _to_search_result
# ---------------------------------------------------------------------------


def test_matching_link_tags_returns_none_when_neither_param_is_given():
    row = {"alignment_links": [{"source_upos": "NOUN"}]}

    assert loader._matching_link_tags(row, "source", upos=None, ner=None) is None


def test_matching_link_tags_finds_a_link_satisfying_the_query():
    row = {"alignment_links": [{"source_upos": "VERB"}, {"source_upos": "NOUN", "source_ner": "PER"}]}

    assert loader._matching_link_tags(row, "source", upos="NOUN", ner=None) == ("NOUN", "PER", None)


def test_matching_link_tags_returns_none_when_no_link_satisfies():
    row = {"alignment_links": [{"source_upos": "VERB"}]}

    assert loader._matching_link_tags(row, "source", upos="NOUN", ner=None) is None


def test_apply_tag_boost_adds_the_boost_when_a_tag_matched():
    hit = loader._Hit(
        row_idx=0,
        side=0,
        faiss_score=0.5,
        pair_id="p1",
        lang="enm",
        year=1400,
        work=None,
        text="t",
        other_lang="eng",
        other_year=1999,
        other_work=None,
        other_text="t2",
        citation="",
        matched_upos="NOUN",
        matched_ner=None,
        matched_lemma=None,
    )

    assert loader._apply_tag_boost(0.5, hit) == pytest.approx(0.5 + loader._TAG_MATCH_BOOST)


def test_apply_tag_boost_leaves_score_unchanged_when_nothing_matched():
    hit = loader._Hit(
        row_idx=0,
        side=0,
        faiss_score=0.5,
        pair_id="p1",
        lang="enm",
        year=1400,
        work=None,
        text="t",
        other_lang="eng",
        other_year=1999,
        other_work=None,
        other_text="t2",
        citation="",
        matched_upos=None,
        matched_ner=None,
        matched_lemma=None,
    )

    assert loader._apply_tag_boost(0.5, hit) == 0.5


def test_to_search_result_maps_every_hit_field():
    hit = loader._Hit(
        row_idx=0,
        side=0,
        faiss_score=0.5,
        pair_id="p1",
        lang="enm",
        year=1400,
        work="W",
        text="t",
        other_lang="eng",
        other_year=1999,
        other_work="W2",
        other_text="t2",
        citation="c",
        matched_upos="NOUN",
        matched_ner="PER",
        matched_lemma="robin",
    )

    result = loader._to_search_result(hit, 0.75)

    assert result.pair_id == "p1" and result.score == 0.75 and result.matched_lemma == "robin"
