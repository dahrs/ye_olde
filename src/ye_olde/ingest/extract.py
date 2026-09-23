"""Raw text extraction from `data/raw/` source files (`.txt`/`.pdf`) plus
cheap, deterministic boilerplate stripping — the "algorithmic" half of
cleaning, done before the LLM ever sees the text (spec §6 step 1).
"""

from __future__ import annotations

import re
from pathlib import Path

# Project Gutenberg wraps every eBook's actual content between a START and
# END banner line. Stripping this algorithmically means the LLM cleaning
# pass (clean.py) never has to spend context tokens recognizing license
# boilerplate it would otherwise see on every single file.
_GUTENBERG_START_RE = re.compile(
    r"^\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*{3}\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_GUTENBERG_END_RE = re.compile(
    r"^\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*{3}\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_text(path: Path) -> str:
    """Returns the raw text content of a `.txt` or `.pdf` file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        return _extract_pdf_text(path)
    raise ValueError(f"unsupported file type: {path.name!r}")


def _extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    raw = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return fix_pdf_mojibake(raw)


# Known character-level mis-decodings produced by specific PDF-extraction
# failure modes, keyed by a short label for what causes each one. Add a new
# entry here whenever a new corpus surfaces a new pattern — this registry
# is meant to grow, not be exhaustive from the start.
#
# "macroman_typographic_punctuation": a PDF whose embedded font maps
# typographic punctuation (em dash, left/right double quote, left/right
# single quote) to MacRoman code points 0xD1-0xD5, extracted by a reader
# that doesn't resolve the font's actual encoding and instead treats the
# raw byte value as if it were the Unicode code point at that number,
# produces exactly five Latin-1 letters (capital N-tilde, O/O-acute/
# O-circumflex/O-tilde) instead of the intended punctuation.
#
# Confirmed on the Gawayne 1999 PDF two ways: (1) by reading the actual
# context — e.g. an opening quote before "Where" came through as one of
# these letters, "king's" came through with the apostrophe replaced the
# same way, and a mid-sentence dash ("green,Ñthe good") came through as
# Ñ; (2) by diffing pypdf's extraction against poppler's independent
# `pdftotext` on the same file — every one of the 83 Ñ occurrences and all
# Ò/Ó/Õ occurrences were confirmed genuine em-dash/quote usage in context,
# with zero ambiguous cases. All five are vanishingly rare as genuine
# characters in this project's target languages (Old/Middle/Modern
# English, Latin, Old French), so the remap is applied unconditionally.
_KNOWN_PDF_MOJIBAKE: dict[str, dict[str, str]] = {
    "macroman_typographic_punctuation": {
        "Ñ": "—",  # Ñ -> — (em dash)
        "Ò": "“",  # Ò -> “
        "Ó": "”",  # Ó -> ”
        "Ô": "‘",  # Ô -> ‘
        "Õ": "’",  # Õ -> ’
    },
}


def fix_pdf_mojibake(text: str) -> str:
    """Repairs character-level mis-decodings from PDF text extraction,
    deterministically, right after extraction — so neither the LLM
    cleaning pass nor anything downstream ever sees the garbled output.
    Two tiers, in order:

    1. Any C1 control character (U+0080-U+009F) is *always* garbage: no
       correctly-functioning text extraction ever produces a literal
       control character as real content, in this file or any other, so
       its presence unambiguously means the source font had no
       `/ToUnicode` CMap and a byte value got treated as a raw Unicode
       code point. Re-interpreting that same byte under MacRomanEncoding
       (confirmed empirically on the Gawayne 1999 PDF — its font's
       character codes above ASCII decode exactly to MacRoman's
       punctuation/accented-letter layout, e.g. a word whose two accented
       letters were coming through as \\x8e/\\x90 decodes correctly to
       "é"/"ê") recovers the font's actual private glyph assignment. This
       rule is general and needs no per-document or per-character-set
       knowledge, unlike tier 2.
    2. `_KNOWN_PDF_MOJIBAKE`: specific *ordinary* Latin-1-range characters
       (not control characters) that are ambiguous in general — Ñ/Ò/Ó/Ô/Õ
       are genuine letters in some languages — but vanishingly rare as
       real content in this project's current target languages, so are
       remapped unconditionally within this project's scope.
    """
    text = "".join(_recover_macroman_control_char(ch) for ch in text)
    for mapping in _KNOWN_PDF_MOJIBAKE.values():
        for bad, good in mapping.items():
            if bad in text:
                text = text.replace(bad, good)
    return text


def _recover_macroman_control_char(ch: str) -> str:
    if not (0x80 <= ord(ch) <= 0x9F):
        return ch
    return ch.encode("latin-1")[:1].decode("mac_roman")


def strip_boilerplate(text: str) -> str:
    """Cuts a Project Gutenberg license header/footer down to the body
    between its START/END banners. Leaves the text unchanged if no banner
    is found (e.g. Internet Archive / other sources without one) — the LLM
    cleaning pass is the fallback for those.
    """
    start = _GUTENBERG_START_RE.search(text)
    end = _GUTENBERG_END_RE.search(text)
    if start and end and start.end() < end.start():
        return text[start.end() : end.start()].strip()
    if start:
        return text[start.end() :].strip()
    return text.strip()
