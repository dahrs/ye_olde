"""Unit tests for extract.py's deterministic PDF mojibake repair."""

from __future__ import annotations

from ye_olde.ingest.extract import fix_pdf_mojibake


def test_fix_pdf_mojibake_repairs_macroman_typographic_punctuation():
    mangled = "ÒWhere is the governor?Ó said the kingÕs man, ÔheroÕ, gay gauds of green,Ñthe good."
    fixed = fix_pdf_mojibake(mangled)
    assert fixed == "“Where is the governor?” said the king’s man, ‘hero’, gay gauds of green,—the good."


def test_fix_pdf_mojibake_leaves_clean_text_unchanged():
    text = "No garbled characters here at all."
    assert fix_pdf_mojibake(text) == text


def test_fix_pdf_mojibake_recovers_control_range_macroman_glyphs():
    # confirmed on the real Gawayne 1999 PDF: a font with no /ToUnicode
    # CMap produced these exact bytes for the accented letters in "mêlée".
    # This is a general mechanism (any C1 control char), not a hardcoded
    # word-specific patch — demonstrated here with a different word too.
    assert fix_pdf_mojibake("m\x90l\x8ee") == "mêlée"
    assert fix_pdf_mojibake("caf\x8e") == "café"


def test_fix_pdf_mojibake_control_range_recovery_is_total():
    # mac_roman is a total codec (every byte 0-255 decodes to something),
    # so this must never raise for any control character in range.
    for code in range(0x80, 0xA0):
        fix_pdf_mojibake(chr(code))
