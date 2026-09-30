from __future__ import annotations

from ye_olde.common.tokenize import tokenize


def test_tokenize_splits_words_and_punctuation():
    assert tokenize("Liyt be maad, and liyt was maad.") == [
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


def test_tokenize_empty_string():
    assert tokenize("") == []
