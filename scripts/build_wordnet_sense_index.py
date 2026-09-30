#!/usr/bin/env python
"""One-time (or rerun-when-changing-`EMBEDDING_MODEL`) precompute for
`ye_olde.ingest.sense.resolve_sense_id`'s embedding-based WordNet matching
(spec §3d/§9): embeds every open-class WordNet synset's own definition and
saves the result to `data/sense_index/` (or `--output-dir`) for that module
to load read-only.

Not run automatically by anything — a real embedding pass over WordNet's
full open-class synset inventory (on the order of 100k+ definitions), run
deliberately once, not as a side effect of the first `resolve_sense_id`
call. Prints its own progress; how long it actually takes depends entirely
on your hardware — no number is promised here, benchmark your own run.

**Incremental, resumable, disk-aware**: the embedding matrix is written to
a pre-sized `.npy` file via `numpy.lib.format.open_memmap` and flushed to
disk after every batch (not accumulated in RAM and written once at the
end) — a `.progress.json` sidecar records how many rows are done, so a
killed/interrupted/`--max-gb`-stopped run resumes from the next
uncompleted batch on a plain rerun with the same arguments, rather than
starting over. `--min-free-gb` (default 2) stops the run cleanly *before*
the destination volume actually fills, leaving already-written rows intact
and resumable, rather than crashing mid-write.

Usage:
    python scripts/build_wordnet_sense_index.py --output-dir /media/dahrs/My_passport/ye_olde-sense-index
    python scripts/build_wordnet_sense_index.py --embedding-model intfloat/multilingual-e5-large
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from ye_olde.config import get_settings
from ye_olde.ingest.embed import embed_units
from ye_olde.ingest.sense import DEFAULT_INDEX_ROOT, WN_POS_TAGS, model_slug


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="litellm/sentence-transformers model id; overrides EMBEDDING_MODEL from .env",
    )
    parser.add_argument("--output-dir", default=None, help="overrides the default data/sense_index root")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=2.0,
        help="stop cleanly (resumable later) once free space on the output volume drops below this",
    )
    args = parser.parse_args(argv)

    import nltk

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        print("[build_wordnet_sense_index] downloading nltk 'wordnet' corpus...", file=sys.stderr)
        nltk.download("wordnet", quiet=True)
    from nltk.corpus import wordnet

    model_name = args.embedding_model or get_settings().embedding_model
    root = Path(args.output_dir) if args.output_dir else DEFAULT_INDEX_ROOT
    root.mkdir(parents=True, exist_ok=True)
    min_free_bytes = args.min_free_gb * (1024**3)

    # Gathering names/definitions/POS is fast and cheap (plain in-memory
    # WordNet corpus reads, no embedding calls) -- always redone in full
    # even on a resume, so it never itself needs incremental saving.
    names: list[str] = []
    definitions: list[str] = []
    pos_tags: list[str] = []
    for wn_pos in WN_POS_TAGS:
        for syn in wordnet.all_synsets(pos=wn_pos):
            names.append(syn.name())
            definitions.append(syn.definition())
            pos_tags.append(wn_pos)
    total = len(definitions)

    slug = model_slug(model_name)
    npy_path = root / f"wordnet_synsets.{slug}.npy"
    json_path = root / f"wordnet_synsets.{slug}.json"
    progress_path = root / f"wordnet_synsets.{slug}.progress.json"

    # Metadata is small (a few MB of text at most for ~120k short
    # definitions) -- written in full up front, not incrementally; the
    # embedding matrix below is the part that's actually large.
    json_path.write_text(
        json.dumps({"names": names, "definitions": definitions, "pos": pos_tags}, ensure_ascii=False),
        encoding="utf-8",
    )

    dim = int(embed_units([definitions[0]], model_name=model_name).shape[1])

    resume_from = 0
    if npy_path.is_file() and progress_path.is_file():
        prev = json.loads(progress_path.read_text(encoding="utf-8"))
        if prev.get("total") == total and prev.get("dim") == dim:
            resume_from = int(prev.get("completed", 0))

    if resume_from > 0:
        print(f"[build_wordnet_sense_index] resuming from row {resume_from}/{total}", file=sys.stderr)
        matrix = np.lib.format.open_memmap(npy_path, mode="r+")
    else:
        matrix = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.float32, shape=(total, dim))

    print(f"[build_wordnet_sense_index] embedding {total} synset definitions with {model_name}...", file=sys.stderr)
    for i in range(resume_from, total, args.batch_size):
        batch = definitions[i : i + args.batch_size]
        matrix[i : i + len(batch)] = embed_units(batch, model_name=model_name)
        matrix.flush()  # on disk now -- a kill/crash right after this line loses nothing already done
        completed = i + len(batch)
        progress_path.write_text(json.dumps({"total": total, "dim": dim, "completed": completed}), encoding="utf-8")
        print(
            f"[build_wordnet_sense_index] {completed}/{total} ({100 * completed / total:.1f}%)",
            file=sys.stderr,
            flush=True,
        )

        free_bytes = shutil.disk_usage(root).free
        if free_bytes < min_free_bytes and completed < total:
            print(
                f"[build_wordnet_sense_index] only {free_bytes / 1024**3:.2f}GB free on {root} (below "
                f"--min-free-gb={args.min_free_gb}) -- stopping at {completed}/{total}, safe to resume "
                "later by rerunning with the same arguments",
                file=sys.stderr,
            )
            return

    progress_path.unlink(missing_ok=True)  # done -- no resume marker needed
    print(f"[build_wordnet_sense_index] wrote {total} synsets -> {root}", file=sys.stderr)


if __name__ == "__main__":
    main()
