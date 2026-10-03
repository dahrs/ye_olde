"""Splits one acquired file that bundles several distinct editions of the
same work (e.g. a Gutenberg compilation containing the original-language
text plus one or more later translations, all in one `.txt`) into separate,
correctly-named `data/raw/` files -- one per edition.

`scripts/acquire_corpus.py` assumes one URL is one edition; this is the
fallback for the sources where that assumption doesn't hold. A human writes
a small manifest (`SplitManifest`, see `load_manifest`) describing each
edition bundled in the file -- its name/year/language and a few free-text
clues -- and `split_mixed_source` asks the configured LLM to assign each
paragraph-chunk of the file to the edition it belongs to, then writes out
one real `data/raw/` file per edition.

Every failure here (a malformed LLM reply, a filename collision, a section
that matched no content) raises rather than being caught -- domain code,
not a boundary; `scripts/split_mixed_source.py`'s `main()` is the boundary
(CLAUDE.md "Error handling").
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from ..common.llm_client import call_llm_json
from ..prompt import load_prompt
from .checkpoint import Checkpoint
from .clean import chunk_text
from .corpus_files import build_corpus_filename, parse_corpus_filename, rename_folder_to_match_oldest

_SYSTEM_PROMPT = load_prompt("ingest", "split_mixed_system")

# Coarser than clean.py's own cleaning chunks: this task only needs the LLM
# to recognize which edition a block belongs to, not closely read it, and a
# smaller chunk gives finer-grained resolution at the actual boundary
# between two editions.
_DEFAULT_CHUNK_MAX_CHARS = 2000
_DEFAULT_BATCH_SIZE = 8


class SectionHint(BaseModel):
    """One edition bundled inside a mixed file -- user-edited external
    input (CLAUDE.md "Pydantic"), same reasoning as `acquire.SourceEntry`.
    """

    name: str
    year: int
    lang_code: str
    clues: list[str] = []
    title: str | None = None


class SplitManifest(BaseModel):
    title: str
    source: str
    sections: list[SectionHint]


def load_manifest(path: Path) -> SplitManifest:
    """Reads a `*.sections.yaml`-style manifest describing every edition
    bundled inside one mixed file.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return SplitManifest.model_validate(raw)


def classify_chunks(
    chunks: list[str],
    sections: list[SectionHint],
    *,
    model: str | None = None,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    checkpoint: Checkpoint | None = None,
) -> list[int | None]:
    """Assigns each of `chunks` to the index of the `sections` entry it
    belongs to, or `None` if it's shared front/back matter that belongs to
    none of them -- one LLM call per `batch_size`-sized batch
    (`split_mixed_system`, see `prompt/ingest.yaml`), checkpointed per batch
    the same way `clean.py`/`align.py` checkpoint theirs.

    A batch whose reply isn't a same-length JSON array raises -- unlike
    `clean.py`'s review pass, there's no safe "keep as-is" fallback for a
    malformed reply here: misclassifying which *edition* a block belongs to
    is a correctness bug, not a stylistic nit, so it should surface loudly
    rather than silently guessing.
    """
    sections_desc = "\n".join(
        f"{i}. {s.name} ({s.year}, {s.lang_code}) -- clues: {', '.join(s.clues) or 'none given'}"
        for i, s in enumerate(sections)
    )
    assignments: list[int | None] = []
    for batch_num, start in enumerate(range(0, len(chunks), batch_size), start=1):
        batch = chunks[start : start + batch_size]
        prompt = f"CANDIDATE EDITIONS:\n{sections_desc}\n\nCHUNKS:\n" + "\n".join(
            f"{i}. {chunk}" for i, chunk in enumerate(batch)
        )

        def compute(prompt: str = prompt) -> object:
            return call_llm_json(prompt, system=_SYSTEM_PROMPT, model=model)

        result = checkpoint.get_or_compute(f"classify:{batch_num}", compute) if checkpoint else compute()
        if not isinstance(result, list) or len(result) != len(batch):
            raise ValueError(f"batch {batch_num}: expected a JSON array of {len(batch)} entries, got {result!r}")
        for entry in result:
            assignments.append(None if entry is None else int(entry))
    return assignments


def split_mixed_source(
    path: Path,
    manifest: SplitManifest,
    *,
    raw_dir: Path,
    model: str | None = None,
    chunk_max_chars: int = _DEFAULT_CHUNK_MAX_CHARS,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    checkpoint: Checkpoint | None = None,
) -> list[Path]:
    """Splits the mixed file at `path` into one `data/raw/` file per
    `manifest.sections`, in the same folder `path` already lives in.

    Never overwrites an existing file at a computed target -- except `path`
    itself, which is explicitly expendable: it's superseded by the files
    this writes, and is deleted once they're all written (unless one of
    them happens to land on `path`'s own name, the common case when the
    original single-file guess matched one of the real editions).
    """
    path = Path(path)
    mixed_filename = path.name
    text = path.read_text(encoding="utf-8")
    chunks = chunk_text(text, max_chars=chunk_max_chars)
    assignments = classify_chunks(chunks, manifest.sections, model=model, batch_size=batch_size, checkpoint=checkpoint)

    # Tracked separately from `path` itself: a section written below can trigger
    # `rename_folder_to_match_oldest` to move the whole folder mid-loop, after which `path`
    # (captured before the loop started) no longer points at the mixed file's real location.
    current_folder = path.parent
    written: list[Path] = []
    for i, section in enumerate(manifest.sections):
        section_text = "\n\n".join(chunk for chunk, idx in zip(chunks, assignments, strict=True) if idx == i)
        if not section_text.strip():
            raise ValueError(f"no content matched section {section.name!r} ({section.year}) -- check its clues")

        filename = build_corpus_filename(
            lang_code=section.lang_code,
            year=section.year,
            title=section.title or manifest.title,
            author=section.name,
            source=manifest.source,
            ext=".txt",
        )
        new_file = parse_corpus_filename(Path(filename))
        current_folder = rename_folder_to_match_oldest(current_folder, new_file, raw_dir=Path(raw_dir))
        target = current_folder / filename
        mixed_path = current_folder / mixed_filename
        if target.exists() and target != mixed_path:
            raise FileExistsError(f"refusing to overwrite existing file {target}")

        target.write_text(section_text, encoding="utf-8")
        written.append(target)

    mixed_path = current_folder / mixed_filename
    if mixed_path.exists() and mixed_path not in written:
        mixed_path.unlink()
    return written
