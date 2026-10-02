"""Parses `data/raw/<work>/` filenames into the metadata the rest of the
alignment pipeline (spec §6) needs, per the naming convention already used
under `data/raw/` (see the two example works there):

    <lang_code>-<year>-<Title_With_Underscores>-<Author_Or_Editor>-<Source>.<ext>

e.g. `enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg.txt`.
A folder groups two or more files that are translations of one another —
"parallel/comparable diachronic texts" in spec §3c.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..common.logging import get_logger

_log = get_logger(__name__)

SUPPORTED_EXTENSIONS = (".txt", ".pdf")

# Matches any run of characters that are neither a unicode letter/digit nor
# `_` -- this deliberately also catches `-` itself, so a sanitized field can
# never introduce an extra top-level `-` for parse_corpus_filename to trip
# over. `re.UNICODE` (the default for `\w` on a `str` pattern) keeps
# diacritics meaningful to this project's actual target languages (Old
# English æ/þ, Latin macrons, ...) rather than stripping them.
_UNSAFE_CHARS_RE = re.compile(r"[^\w]+", re.UNICODE)


@dataclass(frozen=True)
class CorpusFile:
    path: Path
    lang_code: str
    year: int
    title: str
    author: str
    source: str

    @property
    def work_label(self) -> str:
        return self.title.replace("_", " ")


def parse_corpus_filename(path: Path) -> CorpusFile:
    """Splits a `data/raw/` filename into its five `-`-separated fields.

    The title may itself contain underscores-for-spaces but not the
    top-level `-` separators, so `lang`/`year` are taken from the front,
    `source` from the back, and everything in between is the title —
    `author` is the field immediately before `source`. This tolerates a
    title with an embedded extra `-` (none of the current examples have
    one, but nothing here assumes exactly five fields).
    """
    stem = path.stem
    parts = stem.split("-")
    if len(parts) < 5:
        raise ValueError(f"{path.name!r} doesn't match <lang>-<year>-<title>-<author>-<source>{{.txt,.pdf}}")
    lang_code, year_str = parts[0], parts[1]
    source = parts[-1]
    author = parts[-2]
    title = "-".join(parts[2:-2])
    if not year_str.isdigit():
        raise ValueError(f"{path.name!r}: year field {year_str!r} is not numeric")
    return CorpusFile(
        path=path,
        lang_code=lang_code,
        year=int(year_str),
        title=title,
        author=author.replace("_", " "),
        source=source.replace("_", " "),
    )


def sanitize_name_field(value: str) -> str:
    """Turns an arbitrary human-readable string (a title, author, or source
    name) into something safe to use as one `-`-separated field of a
    `data/raw/` filename: every run of whitespace or other non-word
    character (including a literal `-`, so it can never be mistaken for the
    field separator) becomes a single `_`, matching this project's own
    existing convention (e.g. "Chaucer's" -> "Chaucer_s").
    """
    sanitized = _UNSAFE_CHARS_RE.sub("_", value).strip("_")
    if not sanitized:
        raise ValueError(f"{value!r} has no usable characters left after sanitizing")
    return sanitized


def build_corpus_filename(*, lang_code: str, year: str | int, title: str, author: str, source: str, ext: str) -> str:
    """Assembles a `data/raw/`-ready filename from its five semantic parts
    (as `ye_olde.ingest.acquire.propose_metadata` infers them from scraped
    content), sanitizing each field with `sanitize_name_field` first.

    Round-trips the result through `parse_corpus_filename` before returning
    it -- the one guarantee that every filename this project writes is also
    one `discover_corpus_files`/`align_corpus.py` can read back later.
    """
    year_str = str(year).strip()
    if not year_str.isdigit():
        raise ValueError(f"year {year!r} is not a valid 4-digit numeral")
    stem = "-".join(
        [
            sanitize_name_field(lang_code),
            year_str,
            sanitize_name_field(title),
            sanitize_name_field(author),
            sanitize_name_field(source),
        ]
    )
    filename = f"{stem}{ext}"
    parse_corpus_filename(Path(filename))
    return filename


def oldest_parseable_file(folder: Path) -> CorpusFile | None:
    """The file with the smallest `year` among every supported (`.txt`/
    `.pdf`) file directly inside `folder` — this is what a `data/raw/<work>/`
    folder's own name should match (see the existing `lat-523-...`/
    `enm-1400-...` folders: each is named after its oldest file). Unlike
    `discover_corpus_files`, doesn't require 2+ files (a work can have just
    one source text while awaiting its first translation) and skips a file
    that fails `parse_corpus_filename` instead of raising — a folder can
    have an unrelated, not-yet-fixed non-conformant file in it and this
    should still find the true oldest *conformant* one. Returns `None` if
    `folder` doesn't exist or has no parseable file at all.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return None
    parsed: list[CorpusFile] = []
    for p in sorted(folder.iterdir()):
        if not (p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS):
            continue
        try:
            parsed.append(parse_corpus_filename(p))
        except ValueError as exc:
            _log.debug("skipping unparseable file %s while looking for folder %s's oldest file: %s", p, folder, exc)
    if not parsed:
        return None
    return min(parsed, key=lambda cf: cf.year)


def discover_corpus_files(folder: Path) -> list[CorpusFile]:
    """Finds every supported (`.txt`/`.pdf`) file directly inside `folder`,
    parses each filename, and returns them sorted by year (oldest first —
    conventionally, though not necessarily, the source side of most pairs).
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"{folder} is not a directory")
    files = [
        parse_corpus_filename(p)
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if len(files) < 2:
        raise ValueError(
            f"{folder} has {len(files)} supported file(s); need at least 2 parallel translations to build a bitext"
        )
    return sorted(files, key=lambda cf: cf.year)
