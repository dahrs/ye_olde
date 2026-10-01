import pyarrow as pa
from fastapi.testclient import TestClient

from app import lexical, loader
from app.main import app

client = TestClient(app)

_ROWS = [
    {
        "pair_id": "p1",
        "source": {"lang_code": "enm", "year": 1400},
        "target": {"lang_code": "eng", "year": 1999},
        "source_text": "gladly would I see him",
        "target_text": "Gladly I would see that hero",
        "source_tokens": ["gladly", "would", "I", "see", "him"],
        "target_tokens": ["Gladly", "I", "would", "see", "that", "hero"],
        "citation": None,
        "sentence_confidence": 0.9,
        "alignment_links": [],
    },
    {
        "pair_id": "p2",
        "source": {"lang_code": "enm", "year": 1400},
        "target": {"lang_code": "eng", "year": 1999},
        "source_text": "Gladly sir for sothe",
        "target_text": "Gladly, sir, for sooth",
        "source_tokens": ["Gladly", "sir", "for", "sothe"],
        "target_tokens": ["Gladly", ",", "sir", ",", "for", "sooth"],
        "citation": None,
        "sentence_confidence": 0.95,
        "alignment_links": [
            {"source_idx": [0], "target_idx": [0], "sense_id": None},
            {"source_idx": [3], "target_idx": [5], "sense_id": None},  # sothe <-> sooth
        ],
    },
]

_SHARD_PATH = "pairs/enm_eng/1400-1999.parquet"


def _build_ngram_postings(rows: list[dict]) -> pa.Table:
    """Mirrors ye_olde.ingest.index.build_ngram_table's shape, scoped to
    this test's rows -- reuses the real lexical.char_trigrams rather than
    hand-writing trigram sets, so these tests exercise the same definition
    /lookup itself queries with.
    """
    postings = []
    for row_index, row in enumerate(rows):
        for side, tokens_key in ((0, "source_tokens"), (1, "target_tokens")):
            for token_index, token in enumerate(row[tokens_key]):
                grams = lexical.char_trigrams(token)
                postings.extend(
                    {
                        "ngram": gram,
                        "row_index": row_index,
                        "side": side,
                        "token_index": token_index,
                        "token_ngram_count": len(grams),
                    }
                    for gram in grams
                )
    return pa.Table.from_pylist(postings)


def _patch_shards(monkeypatch, rows: list[dict] = _ROWS) -> None:
    monkeypatch.setattr(loader, "load_pair_shards", lambda *a, **k: [(pa.Table.from_pylist(rows), None, _SHARD_PATH)])

    # Built from *this call's* rows, not the module-level _ROWS/_NGRAM_TABLE
    # -- a test passing its own custom `rows` needs postings that actually
    # match those rows' tokens, not the default fixture's.
    ngram_table = _build_ngram_postings(rows)

    def fake_load_ngram_postings(shard_path: str, ngrams: set[str]) -> pa.Table:
        matches = [p for p in ngram_table.to_pylist() if p["ngram"] in ngrams]
        return pa.Table.from_pylist(matches) if matches else pa.table({})

    monkeypatch.setattr(loader, "load_ngram_postings", fake_load_ngram_postings)


def test_lookup_exact_match_falls_back_to_unhighlighted_without_alignment_links(monkeypatch) -> None:
    _patch_shards(monkeypatch)

    response = client.get(
        "/lookup",
        params={"text": "gladly", "lang": "enm", "year": 1400, "target_lang": "eng", "target_year": 1999},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2

    # p1 has no alignment_links — still returned, but with no highlight.
    p1 = next(r for r in results if r["pair_id"] == "p1")
    assert p1["highlighted_span"] is None
    assert p1["citation"] == ""  # null citation, not the literal None Python would give via .get(k, default)

    # p2 has a real link — the precise span is highlighted.
    p2 = next(r for r in results if r["pair_id"] == "p2")
    assert p2["highlighted_span"] == {"token_idx": [0], "surface": "Gladly"}


def test_lookup_handles_reverse_direction(monkeypatch) -> None:
    # These pairs were stored source=enm/target=eng (as the alignment job ran
    # them), but a query asking eng -> enm (the reverse) still needs to match
    # against the modern-English "sooth" and highlight the enm "sothe" — it
    # must not silently search the enm side for an English spelling.
    _patch_shards(monkeypatch)

    response = client.get(
        "/lookup",
        params={"text": "sooth", "lang": "eng", "year": 1999, "target_lang": "enm", "target_year": 1400},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1

    result = results[0]
    assert result["pair_id"] == "p2"
    # source_sentence is now the query side (eng), target_sentence the enm side.
    assert result["source_sentence"] == "Gladly, sir, for sooth"
    assert result["target_sentence"] == "Gladly sir for sothe"
    assert result["highlighted_span"] == {"token_idx": [3], "surface": "sothe"}


def test_lookup_fuzzy_matches_a_spelling_variant_no_exact_token_exists_for(monkeypatch) -> None:
    # "soothe" is not an exact token anywhere in the corpus (the corpus has
    # "sooth") -- this is the whole point of the n-gram + edit-distance
    # mechanism replacing exact-token-only matching (spec §10): Dice("soothe",
    # "sooth") = 2*5/(8+7) ~= 0.67 (above the 0.5 threshold), and
    # levenshtein("soothe", "sooth") = 1 (within the accepted budget).
    _patch_shards(monkeypatch)

    response = client.get(
        "/lookup",
        params={"text": "soothe", "lang": "eng", "year": 1999, "target_lang": "enm", "target_year": 1400},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["pair_id"] == "p2"
    assert results[0]["highlighted_span"] == {"token_idx": [3], "surface": "sothe"}


def test_lookup_the_fuzzy_matches_that_as_accepted_tradeoff(monkeypatch) -> None:
    # "the" vs "that": dice=0.364 clears the length-adaptive threshold at
    # pair_length=7, edit_distance=2 is within the length-adaptive ceiling
    # at that length too -- an accepted trade-off, not a bug: this design
    # deliberately favors recall over precision for short/common words
    # (spec §9/§10), and this is exactly the kind of borderline-but-
    # defensible match that trade-off lets through. No alignment_links on
    # p1, so no highlight -- just confirms it surfaces at all.
    _patch_shards(monkeypatch)

    response = client.get(
        "/lookup",
        params={"text": "the", "lang": "eng", "year": 1999, "target_lang": "enm", "target_year": 1400},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    p1 = next(r for r in results if r["pair_id"] == "p1")
    assert p1["highlighted_span"] is None


def test_lookup_fuzzy_matches_thorn_th_for_longer_words(monkeypatch) -> None:
    # þat/that (pair_length=7): unlike þe/the, this has no identical-twin
    # short-word collision (see the module-level comment in lexical.py), so
    # it's a case the length-adaptive design is meant to keep. Querying with
    # lang="enm" so the comparison is against source_tokens (which actually
    # has "þat") -- /lookup only ever compares the query against tokens on
    # its *own* language side, then displays the already-linked counterpart
    # via alignment_links; it never compares cross-language directly.
    # target_tokens deliberately doesn't contain "that" or "þat", so there's
    # no exact-match confound on either side.
    rows = [
        {
            "pair_id": "q1",
            "source": {"lang_code": "enm", "year": 1400},
            "target": {"lang_code": "eng", "year": 1999},
            "source_text": "þat knyght rode ferre",
            "target_text": "yon knight rode far",
            "source_tokens": ["þat", "knyght", "rode", "ferre"],
            "target_tokens": ["yon", "knight", "rode", "far"],
            "citation": None,
            "sentence_confidence": 0.9,
            "alignment_links": [{"source_idx": [0], "target_idx": [0], "sense_id": None}],  # þat <-> yon
        }
    ]
    _patch_shards(monkeypatch, rows=rows)

    response = client.get(
        "/lookup",
        params={"text": "that", "lang": "enm", "year": 1400, "target_lang": "eng", "target_year": 1999},
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert results[0]["highlighted_span"] == {"token_idx": [0], "surface": "yon"}


def test_lookup_thorn_th_short_words_are_an_accepted_loss(monkeypatch) -> None:
    # þe/the (pair_length=5) is numerically identical, on every character
    # metric tried, to the/we, the/be, the/to -- genuinely different words.
    # No string-similarity design can accept one without the other (see
    # lexical.py's module comment), and a per-language substitution table
    # was explicitly rejected as unmaintainable across every language this
    # project will eventually support. So this is a deliberate, accepted
    # loss, not an oversight -- this test guards against it silently
    # reopening (which would mean the/we-style false positives are back too).
    # Same direction as the test above: lang="enm" compares against
    # source_tokens, which actually has "þe".
    rows = [
        {
            "pair_id": "q2",
            "source": {"lang_code": "enm", "year": 1400},
            "target": {"lang_code": "eng", "year": 1999},
            "source_text": "þe knyght rode ferre",
            "target_text": "yon knight rode far",
            "source_tokens": ["þe", "knyght", "rode", "ferre"],
            "target_tokens": ["yon", "knight", "rode", "far"],
            "citation": None,
            "sentence_confidence": 0.9,
            "alignment_links": [{"source_idx": [0], "target_idx": [0], "sense_id": None}],
        }
    ]
    _patch_shards(monkeypatch, rows=rows)

    response = client.get(
        "/lookup",
        params={"text": "the", "lang": "enm", "year": 1400, "target_lang": "eng", "target_year": 1999},
    )
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_lookup_upos_ner_params_rank_matching_tag_ahead_of_disagreeing_tag(monkeypatch) -> None:
    # Two rows both match the query "robin" on the source side. r1's link
    # tags it a plain NOUN/non-entity; r2's tags it PROPN/PER — a name, spec
    # §3d/§9's motivating case ("a query for a tagged name should
    # preferentially retrieve identically-tagged indexed tokens"). Without
    # upos/ner params, order follows plain iteration (r1 before r2, same as
    # today); passing upos=PROPN&ner=PER should move r2 to the front without
    # dropping r1 — this is a preference ordering, not a filter.
    rows = [
        {
            "pair_id": "r1",
            "source": {"lang_code": "enm", "year": 1400},
            "target": {"lang_code": "eng", "year": 1999},
            "source_text": "robin the archer",
            "target_text": "robin archer",
            "source_tokens": ["robin", "the", "archer"],
            "target_tokens": ["robin", "archer"],
            "citation": None,
            "sentence_confidence": 0.9,
            "alignment_links": [
                {
                    "source_idx": [0],
                    "target_idx": [0],
                    "sense_id": None,
                    "source_upos": "NOUN",
                    "source_ner": "O",
                }
            ],
        },
        {
            "pair_id": "r2",
            "source": {"lang_code": "enm", "year": 1400},
            "target": {"lang_code": "eng", "year": 1999},
            "source_text": "robin shot the arrow",
            "target_text": "robin shot it",
            "source_tokens": ["robin", "shot", "the", "arrow"],
            "target_tokens": ["robin", "shot", "it"],
            "citation": None,
            "sentence_confidence": 0.9,
            "alignment_links": [
                {
                    "source_idx": [0],
                    "target_idx": [0],
                    "sense_id": None,
                    "source_upos": "PROPN",
                    "source_ner": "PER",
                }
            ],
        },
    ]
    _patch_shards(monkeypatch, rows=rows)

    no_tag_response = client.get(
        "/lookup",
        params={"text": "robin", "lang": "enm", "year": 1400, "target_lang": "eng", "target_year": 1999},
    )
    assert [r["pair_id"] for r in no_tag_response.json()["results"]] == ["r1", "r2"]

    tagged_response = client.get(
        "/lookup",
        params={
            "text": "robin",
            "lang": "enm",
            "year": 1400,
            "target_lang": "eng",
            "target_year": 1999,
            "upos": "PROPN",
            "ner": "PER",
        },
    )
    results = tagged_response.json()["results"]
    assert [r["pair_id"] for r in results] == ["r2", "r1"]
    r2 = next(r for r in results if r["pair_id"] == "r2")
    assert r2["matched_upos"] == "PROPN"
    assert r2["matched_ner"] == "PER"


def test_lookup_empty_string_upos_means_not_queried_same_as_attest_convention(monkeypatch) -> None:
    # A client that always serializes every form field can send `upos=`
    # (empty string) alongside a real `ner=PER` -- main.attest's own `ner`
    # param already documents and enforces that convention ("a client
    # sending `?ner=` means 'no filter'"), and /lookup must honor it the
    # same way: treating "" as a literal value to match against would mean
    # no real tag (never the empty string) can ever satisfy it, so r2's
    # genuine NER agreement must not get cancelled out by the empty upos.
    rows = [
        {
            "pair_id": "r1",
            "source": {"lang_code": "enm", "year": 1400},
            "target": {"lang_code": "eng", "year": 1999},
            "source_text": "robin the archer",
            "target_text": "robin archer",
            "source_tokens": ["robin", "the", "archer"],
            "target_tokens": ["robin", "archer"],
            "citation": None,
            "sentence_confidence": 0.9,
            "alignment_links": [
                {"source_idx": [0], "target_idx": [0], "sense_id": None, "source_upos": "NOUN", "source_ner": "O"}
            ],
        },
        {
            "pair_id": "r2",
            "source": {"lang_code": "enm", "year": 1400},
            "target": {"lang_code": "eng", "year": 1999},
            "source_text": "robin shot the arrow",
            "target_text": "robin shot it",
            "source_tokens": ["robin", "shot", "the", "arrow"],
            "target_tokens": ["robin", "shot", "it"],
            "citation": None,
            "sentence_confidence": 0.9,
            "alignment_links": [
                {"source_idx": [0], "target_idx": [0], "sense_id": None, "source_upos": "PROPN", "source_ner": "PER"}
            ],
        },
    ]
    _patch_shards(monkeypatch, rows=rows)

    response = client.get(
        "/lookup",
        params={
            "text": "robin",
            "lang": "enm",
            "year": 1400,
            "target_lang": "eng",
            "target_year": 1999,
            "upos": "",
            "ner": "PER",
        },
    )
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["pair_id"] for r in results] == ["r2", "r1"]


def test_lookup_no_match_returns_empty_results(monkeypatch) -> None:
    _patch_shards(monkeypatch)

    response = client.get(
        "/lookup",
        params={"text": "xyzzy", "lang": "enm", "year": 1400, "target_lang": "eng", "target_year": 1999},
    )
    assert response.status_code == 200
    assert response.json()["results"] == []
