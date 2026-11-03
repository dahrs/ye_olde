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

# A BC/BCE composition date (most of classical Latin/Greek literature) can't be written with a
# literal "-" -- that's already the field separator -- so a BC year is instead the token
# "BC<digits>" (e.g. "BC0019" for 19 BC), parsed to a negative CorpusFile.year. Plain digits parse
# exactly as before; nothing about an existing, non-BC filename changes.
_BC_YEAR_TOKEN_RE = re.compile(r"^BC(\d+)$")


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


def pair_slug(cf_a: CorpusFile, cf_b: CorpusFile) -> str:
    """Filename-safe identifier for the language pair `(cf_a, cf_b)`,
    e.g. `"ang-975_eng-1910"` -- `lang_code` alone (`"ang-eng"`) collides
    whenever a work has two translations into the same language (seen
    live: Beowulf's 1910 and 2021 English translations both aligned
    against Old English wrote to the same `ang-eng.bitext.jsonl`, the
    second silently overwriting the first). Including each side's `year`
    disambiguates without needing the full filename. Shared by
    `ingest.align` (which writes this) and `eval.compare_runs` (which
    needs to construct the identical path to read it back), so the two
    never drift apart.
    """
    return f"{cf_a.lang_code}-{cf_a.year}_{cf_b.lang_code}-{cf_b.year}"


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
    bc_match = _BC_YEAR_TOKEN_RE.match(year_str)
    if bc_match:
        year = -int(bc_match.group(1))
    elif year_str.isdigit():
        year = int(year_str)
    else:
        raise ValueError(f"{path.name!r}: year field {year_str!r} is not numeric or a BC<digits> token")
    return CorpusFile(
        path=path,
        lang_code=lang_code,
        year=year,
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

    `year` is almost always a plain positive numeral, as always. For a BC/BCE
    composition date (most of classical Latin/Greek literature), pass a
    negative `int` (e.g. `-19` for 19 BC) or the equivalent `"BC19"` string
    directly -- either renders as the `BC<digits>` filename token
    `parse_corpus_filename` reads back as a negative year (see that
    function's own comment for why a literal "-" can't be used instead).

    Round-trips the result through `parse_corpus_filename` before returning
    it -- the one guarantee that every filename this project writes is also
    one `discover_corpus_files`/`align_corpus.py` can read back later.
    """
    year_str = str(year).strip()
    if year_str.lstrip("-").isdigit():
        year_str = f"BC{-int(year_str)}" if year_str.startswith("-") else year_str
    elif not _BC_YEAR_TOKEN_RE.match(year_str):
        raise ValueError(f"year {year!r} is not a valid 4-digit numeral, negative int, or BC<digits> token")
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


def rename_folder_to_match_oldest(folder_path: Path, new_file: CorpusFile, *, raw_dir: Path) -> Path:
    """A `data/raw/<work>/` folder is named after its oldest file (see
    `oldest_parseable_file`). When `folder_path` already has files in it and
    `new_file` (about to be added) is older than every one of them, the
    folder's name is stale -- rename it to match this new, now-oldest file
    rather than leaving folder and content mismatched. A brand-new folder
    (nothing in it yet) has nothing to compare against and is left as-is.
    Shared by `acquire.py` (one file at a time) and `split_mixed.py` (several
    files from one source at once, any of which could be the new anchor).
    """
    anchor = oldest_parseable_file(folder_path)
    if anchor is None or new_file.year >= anchor.year:
        return folder_path
    renamed = Path(raw_dir) / new_file.path.stem
    if renamed == folder_path:
        return folder_path
    if renamed.exists():
        raise FileExistsError(
            f"{folder_path} should be renamed to {renamed} ({new_file.path.name} is now the oldest file "
            f"in this work) but that folder name is already taken"
        )
    folder_path.rename(renamed)
    _log.info("renamed %s -> %s (%s is now the oldest file in this work)", folder_path, renamed, new_file.path.name)
    return renamed


def _normalize_for_matching(value: str) -> str:
    # `_UNSAFE_CHARS_RE` (`\w`, unicode-aware) -- not an ASCII-only `[a-z0-9]` class, which
    # collapses any non-Latin-script title (confirmed live: two distinct Greek-titled works,
    # "Ἰδύλλια" and "Ἐπιγράμματα", both normalized to "" and were wrongly treated as the same work).
    return _UNSAFE_CHARS_RE.sub(" ", value.lower()).strip()


def authors_plausibly_match(author_a: str, author_b: str) -> bool:
    """Whether two author/editor/translator strings plausibly name the
    same person -- substring containment after normalization (handles
    "Vergil" vs. "P. Vergilius Maro"), guarded by a minimum length so a
    short/common fragment can't match almost anything. Two placeholder
    "unknown"s count as a match (both sides genuinely don't know); an
    "unknown" never matches a real name, since that would just be
    guessing that an anonymous-attributed text and a named author's text
    are the same work.
    """
    a, b = _normalize_for_matching(author_a), _normalize_for_matching(author_b)
    if not a or not b:
        return False
    if a == "unknown" or b == "unknown":
        return a == b
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return len(shorter) >= 3 and shorter in longer


def find_matching_work_folder(raw_dir: Path, title: str, original_author: str) -> Path | None:
    """Finds an existing `data/raw/<work>/` folder whose OLDEST file --
    by convention, the work's own original-language edition, see
    `oldest_parseable_file` -- carries this same title AND a plausibly-
    the-same original author. Used by a more-reliable re-acquisition of a
    work already covered by an earlier, less reliable pass (e.g.
    `structured_sources.py`'s Perseus path finding a work the generic
    LLM-crawl path already acquired) to add to or correct that existing
    folder instead of creating an accidental duplicate one.

    Matches against the folder's oldest file specifically, not just any
    file in it -- a translation's own `author` field is its translator,
    not the work's original author, so comparing `original_author`
    against an arbitrary sibling file (rather than the one file that's
    reliably the original) would near-never match an existing folder
    whose other translations happen to be by someone else.

    Both title AND original author must match -- title alone is NOT
    enough: many classical titles are conventional genre names reused by
    unrelated authors ("Elegiae", "Epistulae", "Apology", "Symposium",
    "Helen" are all titles two or more different, unrelated authors used
    for two or more different works). Matching on title alone was caught
    live silently merging Tibullus's "Elegiae" with Propertius's
    "Elegiae" (and many similar cases) -- completely different texts, not
    the same work from a different source.

    `None` if nothing already on disk matches both, or `raw_dir` doesn't
    exist yet.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        return None
    normalized_title = _normalize_for_matching(title)
    for folder in sorted(raw_dir.iterdir()):
        if not folder.is_dir():
            continue
        anchor = oldest_parseable_file(folder)
        if anchor is None:
            continue
        if _normalize_for_matching(anchor.work_label) == normalized_title and authors_plausibly_match(
            anchor.author, original_author
        ):
            return folder
    return None


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
