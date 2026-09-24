"""Unit tests for align.py: tokenization, span resolution, proportional
blocking, dedup, checkpointing/resume, and the jsonl writer. LLM calls are
mocked throughout — none of this should need a live model to be correct.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ye_olde.ingest import align
from ye_olde.ingest.corpus_files import CorpusFile


def test_tokenize_splits_words_and_punctuation():
    assert align.tokenize("Liyt be maad, and liyt was maad.") == [
        "Liyt",
        "be",
        "maad",
        ",",
        "and",
        "liyt",
        "was",
        "maad",
        ".",
    ]


def test_find_span_token_indices_case_insensitive():
    tokens = align.tokenize('Then God said , " Let there be light "')
    idx = align.find_span_token_indices(tokens, "let there be LIGHT")
    assert idx == [tokens.index("Let"), tokens.index("there"), tokens.index("be"), tokens.index("light")]


def test_find_span_token_indices_missing_returns_none():
    tokens = align.tokenize("Then God said light")
    assert align.find_span_token_indices(tokens, "not present here") is None


def test_make_blocks_proportional_windows():
    units_a = [f"a{i}" for i in range(100)]
    units_b = [f"b{i}" for i in range(200)]
    blocks = align.make_blocks(units_a, units_b, block_size=40, overlap_ratio=0.0)
    assert blocks[0][0] == units_a[0:40]
    assert blocks[0][1] == units_b[0:80]
    # every unit of the shorter list is covered by some block
    covered = set()
    for block_a, _ in blocks:
        covered.update(block_a)
    assert covered == set(units_a)


def test_make_blocks_empty_input():
    assert align.make_blocks([], ["x"]) == []
    assert align.make_blocks(["x"], []) == []


def test_align_block_resolves_links(monkeypatch):
    cf_a = CorpusFile(Path("a.txt"), "enm", 1382, "Bible", "Wycliffe", "PG")
    cf_b = CorpusFile(Path("b.txt"), "eng", 1989, "Bible", "NRSV", "PG")

    fake_response = [
        {
            "source_text": "And God seide, Liyt be maad.",
            "target_text": "Then God said, Let there be light.",
            "citation": "Genesis 1:3",
            "confidence": 0.9,
            "links": [
                {"source_span": "Liyt", "target_span": "light"},
                {"source_span": "nonexistent", "target_span": "light"},
            ],
        }
    ]
    monkeypatch.setattr(align, "call_llm_json", lambda *a, **k: fake_response)

    resolved = align.align_block(["source unit"], ["target unit"], cf_a, cf_b)
    assert len(resolved) == 1
    pair = resolved[0]
    assert pair["citation"] == "Genesis 1:3"
    assert pair["sentence_confidence"] == 0.9
    # the good link resolves, the bad one (missing span) is dropped
    assert len(pair["alignment_links"]) == 1
    link = pair["alignment_links"][0]
    assert pair["source_tokens"][link["source_idx"][0]].lower() == "liyt"
    assert pair["target_tokens"][link["target_idx"][0]].lower() == "light"


def test_align_corpus_pair_dedupes_overlapping_blocks(monkeypatch):
    cf_a = CorpusFile(Path("a.txt"), "enm", 1382, "Bible", "Wycliffe", "PG")
    cf_b = CorpusFile(Path("b.txt"), "eng", 1989, "Bible", "NRSV", "PG")

    same_pair = {
        "source_text": "And God seide, Liyt be maad.",
        "target_text": "Then God said, Let there be light.",
        "citation": None,
        "confidence": 0.9,
        "links": [],
    }
    monkeypatch.setattr(align, "call_llm_json", lambda *a, **k: [same_pair])

    # the fake pair's text must be a real (verbatim, in-order) excerpt of
    # the cleaned units for it to survive _verify_pairs_against_source.
    units_a = ["And God seide, Liyt be maad.", *(f"a{i}" for i in range(1, 50))]
    units_b = ["Then God said, Let there be light.", *(f"b{i}" for i in range(1, 50))]
    records = align.align_corpus_pair(cf_a, cf_b, units_a, units_b, mode="llm", block_size=40)

    assert len(records) == 1
    record = records[0]
    assert record["pair_id"].startswith("bible_1382x1989_")
    assert record["source"] == {"lang_code": "enm", "year": 1382, "work": "Bible", "author": "Wycliffe"}
    assert record["target"] == {"lang_code": "eng", "year": 1989, "work": "Bible", "author": "NRSV"}


def test_align_corpus_pair_resumes_after_simulated_crash(tmp_path, monkeypatch):
    cf_a = CorpusFile(Path("a.txt"), "enm", 1400, "Work", "Author", "Src")
    cf_b = CorpusFile(Path("b.txt"), "eng", 1999, "Work", "Author", "Src")

    # block_size=40 over 50 units makes exactly 2 blocks (see make_blocks).
    # Both possible fake pair texts ("pair-from-call-1" / "-3") must be real
    # excerpts of the cleaned units to survive _verify_pairs_against_source.
    units_a = ["pair-from-call-1", "pair-from-call-3", *(f"a{i}" for i in range(2, 50))]
    units_b = ["target-1", "target-3", *(f"b{i}" for i in range(2, 50))]

    call_count = {"n": 0}

    def fake_call_llm_json(prompt, *, system=None, model=None):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise ValueError("simulated API billing failure")
        return [
            {
                "source_text": f"pair-from-call-{call_count['n']}",
                "target_text": f"target-{call_count['n']}",
                "citation": None,
                "confidence": 0.5,
                "links": [],
            }
        ]

    monkeypatch.setattr(align, "call_llm_json", fake_call_llm_json)

    checkpoint_path = tmp_path / "enm-eng.align_progress.json"
    with pytest.raises(ValueError, match="simulated API billing failure"):
        align.align_corpus_pair(
            cf_a, cf_b, units_a, units_b, mode="llm", block_size=40, checkpoint_path=checkpoint_path
        )
    assert call_count["n"] == 2  # block 1 succeeded, block 2 failed

    # resuming must not re-request block 1's already-successful result
    records = align.align_corpus_pair(
        cf_a, cf_b, units_a, units_b, mode="llm", block_size=40, checkpoint_path=checkpoint_path
    )
    assert call_count["n"] == 3  # only one more call: block 2, retried
    texts = {r["source_text"] for r in records}
    assert texts == {"pair-from-call-1", "pair-from-call-3"}


def test_align_corpus_pair_no_checkpoint_starts_fresh(tmp_path, monkeypatch):
    cf_a = CorpusFile(Path("a.txt"), "enm", 1400, "Work", "Author", "Src")
    cf_b = CorpusFile(Path("b.txt"), "eng", 1999, "Work", "Author", "Src")
    units_a, units_b = [f"a{i}" for i in range(50)], [f"b{i}" for i in range(50)]

    calls = []

    def fake_call_llm_json(prompt, *, system=None, model=None):
        calls.append(1)
        return []

    monkeypatch.setattr(align, "call_llm_json", fake_call_llm_json)

    checkpoint_path = tmp_path / "enm-eng.align_progress.json"
    align.align_corpus_pair(cf_a, cf_b, units_a, units_b, mode="llm", block_size=40, checkpoint_path=checkpoint_path)
    assert len(calls) == 2  # 2 blocks

    # use_checkpoint=False must re-request every block even though a
    # (successful, complete) checkpoint file exists from the run above
    align.align_corpus_pair(
        cf_a,
        cf_b,
        units_a,
        units_b,
        mode="llm",
        block_size=40,
        checkpoint_path=checkpoint_path,
        use_checkpoint=False,
    )
    assert len(calls) == 4


def test_align_corpus_folder_cleans_up_checkpoint_on_success(tmp_path, monkeypatch):
    raw_dir = tmp_path / "eng-1810-Work-Author-Source"
    raw_dir.mkdir()
    (raw_dir / "eng-1810-Work-Author-Source.txt").write_text("Old content.")
    (raw_dir / "eng-2026-Work-Author-Source.txt").write_text("New content.")

    monkeypatch.setattr(align, "clean_corpus_file", lambda cf, raw_text, **k: [raw_text])
    monkeypatch.setattr(align, "call_llm_json", lambda *a, **k: [])

    output_dir = tmp_path / "processed"
    align.align_corpus_folder(raw_dir, output_dir=output_dir, mode="llm", use_cache=False)

    leftover_progress = list((output_dir / raw_dir.name).glob("*.align_progress.json"))
    assert leftover_progress == []


def test_align_corpus_folder_orders_pairs_chronologically(tmp_path, monkeypatch):
    raw_dir = tmp_path / "lat-523-Work-Author-Source"
    raw_dir.mkdir()
    # written in a deliberately non-chronological, non-alphabetical order
    (raw_dir / "eng-1897-Work-James-PG.txt").write_text("Modern English content sentence.")
    (raw_dir / "lat-523-Work-Author-Source.txt").write_text("Latin content sentence.")
    (raw_dir / "enm-1380-Work-Chaucer-PG.txt").write_text("Middle English content sentence.")

    # skip the LLM entirely: cleaning returns the raw text as a single unit,
    # alignment returns no pairs — this test only cares about which files
    # get paired into which output filenames, not alignment quality.
    monkeypatch.setattr(align, "clean_corpus_file", lambda cf, raw_text, **k: [raw_text])
    monkeypatch.setattr(align, "call_llm_json", lambda *a, **k: [])

    output_dir = tmp_path / "processed"
    written = align.align_corpus_folder(raw_dir, output_dir=output_dir, mode="llm", use_cache=False)

    # lat (523) < enm (1380) < eng (1897): every pair reads older-first, and
    # each unordered language pair appears exactly once (never both
    # directions).
    names = sorted(p.name for p in written)
    assert names == ["enm-eng.bitext.jsonl", "lat-eng.bitext.jsonl", "lat-enm.bitext.jsonl"]


def test_write_jsonl_round_trip(tmp_path):
    records = [
        {
            "pair_id": "p1",
            "source": {"lang_code": "enm", "year": 1382, "work": "Bible", "author": "Wycliffe"},
            "target": {"lang_code": "eng", "year": 1989, "work": "Bible", "author": "NRSV"},
            "source_text": "Liyt be maad",
            "target_text": "Let there be light",
            "source_tokens": ["Liyt", "be", "maad"],
            "target_tokens": ["Let", "there", "be", "light"],
            "citation": "Genesis 1:3",
            "sentence_confidence": 0.9,
            "alignment_links": [{"source_idx": [0], "target_idx": [3], "sense_id": None}],
            "sentence_method": "llm:test",
            "metadata": {"license": None, "ingested_at": "2026-01-01T00:00:00+00:00"},
        }
    ]

    jsonl_path = tmp_path / "out.jsonl"
    align.write_jsonl(records, jsonl_path)
    round_tripped = json.loads(jsonl_path.read_text().splitlines()[0])
    assert round_tripped["pair_id"] == "p1"
    assert round_tripped["alignment_links"][0]["source_idx"] == [0]


def test_extract_links_batch_resolves_spans(monkeypatch):
    pairs = [
        {
            "source_text": "Liyt be maad",
            "target_text": "let there be light",
            "source_tokens": align.tokenize("Liyt be maad"),
            "target_tokens": align.tokenize("let there be light"),
            "alignment_links": [],
        }
    ]
    monkeypatch.setattr(align, "call_llm_json", lambda *a, **k: [[{"source_span": "Liyt", "target_span": "light"}]])

    cf_a = CorpusFile(Path("a.txt"), "enm", 1382, "Bible", "Wycliffe", "PG")
    cf_b = CorpusFile(Path("b.txt"), "eng", 1989, "Bible", "NRSV", "PG")
    align.extract_links_batch(pairs, cf_a, cf_b)

    assert len(pairs[0]["alignment_links"]) == 1
    link = pairs[0]["alignment_links"][0]
    assert pairs[0]["source_tokens"][link["source_idx"][0]].lower() == "liyt"
    assert pairs[0]["target_tokens"][link["target_idx"][0]].lower() == "light"


def test_align_pairs_hybrid_combines_embedding_anchors_and_llm_gaps(monkeypatch):
    cf_a = CorpusFile(Path("a.txt"), "enm", 1400, "Work", "Author", "Src")
    cf_b = CorpusFile(Path("b.txt"), "eng", 1999, "Work", "Author", "Src")

    units_a = ["a0", "a1", "a2"]
    units_b = ["b0", "b1", "b2"]

    # Pretend the embedding aligner confidently matched a0<->b0 and a2<->b2,
    # leaving a1/b1 as a gap for the LLM to discover.
    fake_matches = [align.SentenceMatch(i=0, j=0, score=0.9), align.SentenceMatch(i=2, j=2, score=0.85)]
    monkeypatch.setattr(align, "mutual_nearest_neighbor_align", lambda *a, **k: fake_matches)

    def fake_call_llm_json(prompt, *, system=None, model=None):
        if system == align._LINKS_SYSTEM_PROMPT:
            return [[], []]  # no word links for either embedding anchor, for simplicity
        return [{"source_text": "a1", "target_text": "b1", "citation": None, "confidence": 0.7, "links": []}]

    monkeypatch.setattr(align, "call_llm_json", fake_call_llm_json)

    records = align.align_corpus_pair(cf_a, cf_b, units_a, units_b, mode="hybrid")

    texts = {(r["source_text"], r["target_text"]) for r in records}
    assert texts == {("a0", "b0"), ("a1", "b1"), ("a2", "b2")}

    methods = {r["source_text"]: r["sentence_method"] for r in records}
    assert methods["a0"].startswith("embedding:")
    assert methods["a2"].startswith("embedding:")
    assert methods["a1"].startswith("llm:")
    # embedding provenance keeps its raw cosine score as sentence_confidence
    assert {r["source_text"]: r["sentence_confidence"] for r in records}["a0"] == 0.9


def test_align_corpus_folder_writes_one_jsonl_file_per_pair(tmp_path, monkeypatch):
    raw_dir = tmp_path / "eng-1810-Work-Author-Source"
    raw_dir.mkdir()
    (raw_dir / "eng-1810-Work-Author-Source.txt").write_text("Old content.")
    (raw_dir / "eng-2026-Work-Author-Source.txt").write_text("New content.")

    monkeypatch.setattr(align, "clean_corpus_file", lambda cf, raw_text, **k: [raw_text])
    monkeypatch.setattr(align, "call_llm_json", lambda *a, **k: [])

    written = align.align_corpus_folder(raw_dir, output_dir=tmp_path / "processed", mode="llm", use_cache=False)

    assert [p.suffix for p in written] == [".jsonl"]


def test_verify_pairs_against_source_keeps_verbatim_pairs():
    units_a = ["And God seide, Liyt be maad.", "And it was so."]
    units_b = ["Then God said, Let there be light.", "And it was so."]
    pairs = [
        {"source_text": "And God seide, Liyt be maad.", "target_text": "Then God said, Let there be light."},
        # a pair spanning exactly one whole unit on both sides, verbatim
        {"source_text": "And it was so.", "target_text": "And it was so."},
    ]
    verified = align._verify_pairs_against_source(pairs, units_a, units_b)
    assert verified == pairs


def test_verify_pairs_against_source_drops_altered_content():
    units_a = ["And God seide, Liyt be maad."]
    units_b = ["Then God said, Let there be light."]
    pairs = [
        {"source_text": "And God seide, Liyt be maad.", "target_text": "Then God said, Let there be light."},
        # reordered words -> not a substring even though same words appear
        {"source_text": "Liyt be maad, and God seide.", "target_text": "Then God said, Let there be light."},
        # invented/paraphrased wording -> not in the cleaned source at all
        {"source_text": "God spoke and made light.", "target_text": "Then God said, Let there be light."},
    ]
    verified = align._verify_pairs_against_source(pairs, units_a, units_b)
    assert verified == [pairs[0]]


def test_verify_pairs_against_source_is_case_and_whitespace_insensitive():
    units_a = ["And   God seide,\nLiyt be maad."]
    units_b = ["Then God said, Let there be light."]
    pairs = [{"source_text": "AND GOD SEIDE, LIYT BE MAAD.", "target_text": "Then God said, Let there be light."}]
    assert align._verify_pairs_against_source(pairs, units_a, units_b) == pairs


def test_review_pairs_with_llm_leaves_correct_pairs_unchanged(monkeypatch):
    cf_a = CorpusFile(Path("a.txt"), "enm", 1382, "Bible", "Wycliffe", "PG")
    cf_b = CorpusFile(Path("b.txt"), "eng", 1989, "Bible", "NRSV", "PG")
    pair = {
        "source_text": "And God seide, Liyt be maad.",
        "target_text": "Then God said, Let there be light.",
        "source_tokens": align.tokenize("And God seide, Liyt be maad."),
        "target_tokens": align.tokenize("Then God said, Let there be light."),
        "citation": None,
        "sentence_confidence": 0.9,
        "alignment_links": [{"source_idx": [0], "target_idx": [0], "sense_id": None}],
    }

    # the review model "agrees" and repeats the pair back unchanged
    monkeypatch.setattr(
        align,
        "call_llm_json",
        lambda *a, **k: [{"source_text": pair["source_text"], "target_text": pair["target_text"], "citation": None}],
    )

    reviewed = align.review_pairs_with_llm([pair], cf_a, cf_b)
    assert reviewed == [pair]  # identical object back out -> links/tokens untouched


def test_review_pairs_with_llm_applies_a_genuine_correction(monkeypatch):
    cf_a = CorpusFile(Path("a.txt"), "enm", 1382, "Bible", "Wycliffe", "PG")
    cf_b = CorpusFile(Path("b.txt"), "eng", 1989, "Bible", "NRSV", "PG")
    pair = {
        "source_text": "And God seide, Liyt be maad.",
        "target_text": "totally wrong unrelated text",
        "source_tokens": align.tokenize("And God seide, Liyt be maad."),
        "target_tokens": align.tokenize("totally wrong unrelated text"),
        "citation": None,
        "sentence_confidence": 0.9,
        "alignment_links": [{"source_idx": [0], "target_idx": [0], "sense_id": None}],
    }

    monkeypatch.setattr(
        align,
        "call_llm_json",
        lambda *a, **k: [
            {
                "source_text": "And God seide, Liyt be maad.",
                "target_text": "Then God said, Let there be light.",
                "citation": "Genesis 1:3",
            }
        ],
    )

    reviewed = align.review_pairs_with_llm([pair], cf_a, cf_b)
    assert len(reviewed) == 1
    corrected = reviewed[0]
    assert corrected["target_text"] == "Then God said, Let there be light."
    assert corrected["citation"] == "Genesis 1:3"
    # links computed against the old (wrong) target text are stale -> reset
    assert corrected["alignment_links"] == []
    assert corrected["target_tokens"] == align.tokenize("Then God said, Let there be light.")


def test_align_corpus_pair_runs_review_pass_only_for_local_models(monkeypatch):
    cf_a = CorpusFile(Path("a.txt"), "enm", 1400, "Work", "Author", "Src")
    cf_b = CorpusFile(Path("b.txt"), "eng", 1999, "Work", "Author", "Src")
    units_a = ["hello world"]
    units_b = ["bonjour monde"]

    review_calls = []

    def fake_call_llm_json(prompt, *, system=None, model=None):
        if system == align._PAIR_REVIEW_SYSTEM_PROMPT:
            review_calls.append(prompt)
            return [{"source_text": "hello world", "target_text": "bonjour monde", "citation": None}]
        return [
            {
                "source_text": "hello world",
                "target_text": "bonjour monde",
                "citation": None,
                "confidence": 0.9,
                "links": [],
            }
        ]

    monkeypatch.setattr(align, "call_llm_json", fake_call_llm_json)

    monkeypatch.setattr(align, "is_local_model", lambda *a, **k: False)
    align.align_corpus_pair(cf_a, cf_b, units_a, units_b, mode="llm", block_size=40)
    assert review_calls == []

    monkeypatch.setattr(align, "is_local_model", lambda *a, **k: True)
    align.align_corpus_pair(cf_a, cf_b, units_a, units_b, mode="llm", block_size=40)
    assert len(review_calls) == 1
