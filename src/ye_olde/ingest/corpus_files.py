"""Parses `data/raw/<work>/` filenames into the metadata the rest of the
alignment pipeline (spec §6) needs, per the naming convention already used
under `data/raw/` (see the two example works there):

    <lang_code>-<year>-<Title_With_Underscores>-<Author_Or_Editor>-<Source>.<ext>

e.g. `enm-1400-Sir_Gawayne_and_the_Green_Knight-Richard_Morris-Project_Gutenberg.txt`.
A folder groups two or more files that are translations of one another —
"parallel/comparable diachronic texts" in spec §3c.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = (".txt", ".pdf")


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
        raise ValueError(
            f"{path.name!r} doesn't match "
            "<lang>-<year>-<title>-<author>-<source>{.txt,.pdf}"
        )
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
            f"{folder} has {len(files)} supported file(s); need at least 2 "
            "parallel translations to build a bitext"
        )
    return sorted(files, key=lambda cf: cf.year)
