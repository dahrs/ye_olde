"""Thin smoke test for scripts/build_wordnet_sense_index.py.

`main()` has no branching logic of its own at all -- pure argparse, then
one call straight through to `ye_olde.ingest.sense.build_wordnet_index`
(already covered by `test_sense.py`). Just checks every CLI flag reaches
that call with the right value/type, including the default-vs-override
cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_wordnet_sense_index  # noqa: E402


def test_main_passes_every_argument_through(monkeypatch):
    seen: dict[str, object] = {}

    def fake_build_wordnet_index(*, model_name, root, batch_size, min_free_gb):
        seen.update(model_name=model_name, root=root, batch_size=batch_size, min_free_gb=min_free_gb)

    monkeypatch.setattr(build_wordnet_sense_index, "build_wordnet_index", fake_build_wordnet_index)

    build_wordnet_sense_index.main(
        [
            "--embedding-model",
            "intfloat/multilingual-e5-large",
            "--output-dir",
            "/media/dahrs/My_passport/ye_olde-sense-index",
            "--batch-size",
            "128",
            "--min-free-gb",
            "5.0",
        ]
    )

    assert seen == {
        "model_name": "intfloat/multilingual-e5-large",
        "root": Path("/media/dahrs/My_passport/ye_olde-sense-index"),
        "batch_size": 128,
        "min_free_gb": 5.0,
    }


def test_main_uses_documented_defaults_when_nothing_is_given(monkeypatch):
    seen: dict[str, object] = {}

    def fake_build_wordnet_index(*, model_name, root, batch_size, min_free_gb):
        seen.update(model_name=model_name, root=root, batch_size=batch_size, min_free_gb=min_free_gb)

    monkeypatch.setattr(build_wordnet_sense_index, "build_wordnet_index", fake_build_wordnet_index)

    build_wordnet_sense_index.main([])

    assert seen == {"model_name": None, "root": None, "batch_size": 256, "min_free_gb": 2.0}
