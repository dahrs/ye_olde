#!/usr/bin/env python
"""CLI wrapper for `ye_olde.ingest.sense.build_wordnet_index` — see that
function's docstring (and `sense.py`'s module docstring) for the full
picture: on-disk format, resumability, why matching is gloss-embedding-
based rather than lemma lookup.

A real pipeline run (`align.align_corpus_folder`, and so `scripts/
align_corpus.py`) now calls this same logic automatically
(`sense.ensure_wordnet_index_ready`) if the index is missing or was left
incomplete by an earlier interrupted run — so this script is no longer a
required manual step before a real run. It's still useful to run standalone
ahead of time (e.g. overnight, before you actually need a pipeline run to
proceed) rather than block that run's first real invocation for however
long the precompute takes on your hardware.

Usage:
    python scripts/build_wordnet_sense_index.py --output-dir /media/dahrs/My_passport/ye_olde-sense-index
    python scripts/build_wordnet_sense_index.py --embedding-model intfloat/multilingual-e5-large
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ye_olde.ingest.sense import build_wordnet_index


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

    build_wordnet_index(
        model_name=args.embedding_model,
        root=Path(args.output_dir) if args.output_dir else None,
        batch_size=args.batch_size,
        min_free_gb=args.min_free_gb,
    )


if __name__ == "__main__":
    main()
