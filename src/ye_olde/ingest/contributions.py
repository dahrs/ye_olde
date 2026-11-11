"""Turns validated §3b community contributions into §3a's indexed
relational schema — the one pipeline leg flagged as unbuilt since this
package's own `__init__.py` docstring, and the reason `/attest` has always
returned `results: []` (search_api §13e): no §3b contribution has ever been
transformed into a row `/attest` can actually query.

Reuses existing, tested infrastructure rather than inventing new LLM calls:
a contribution's headword gets tagged by `ye_olde.classify.classify` (the
exact same per-token lemma/upos/ner tagger `translate()` already applies to
live input text — a contribution's `origin` is just more text that can't be
precomputed, spec §3d) and its `sense_id` is resolved by
`ye_olde.ingest.sense.resolve_sense_id`, reusing the contribution's own
`target.text` as the English gloss — the contributor's translation already
*is* the gloss §3c alignment would otherwise need an LLM to propose fresh.

**Only `search_api/app/schemas.py`'s actual `AttestResult` fields are
populated** — `family_id`/`code_valid_from`/`code_valid_to` from spec §3a's
table aren't, because nothing in the real, deployed schema reads them yet
(same "don't extend for a signal nothing uses" philosophy as every other
deferred field in this codebase — e.g. spec §9 on relational embeddings).

**`doc_type: "grammar_note"` entries are skipped entirely** — spec §3c:
they're prose context for the LLM, `target: null`, no word-level lemma to
index against; nothing in this codebase retrieves them that way yet either.

**A `tokens[].class == "name"` hint bypasses sense-retrieval entirely**,
same as spec §2's "names bypass sense-retrieval" design point for the live
pipeline — a name bypasses `classify`/`resolve_sense_id` and gets
`ner="PER"` directly, with a placeholder `name:<lemma>` sense_id (a third,
clearly-distinct namespace alongside the WordNet-anchored `en:` and minted
`ye_olde:` ones — a name genuinely has no WordNet sense to anchor to).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from jsonschema import validate as jsonschema_validate

from .. import classify
from .sense import resolve_sense_id

_SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema" / "contribution.schema.json"


def load_contributions(contributions_dir: Path) -> list[dict[str, Any]]:
    """Reads and schema-validates every `*.json` file under
    `contributions_dir` (recursively), returning only entries whose
    `metadata.validation.status` is `"validated"` — spec §5: only validated
    contributions feed §3a; a `"pending"`/`"proofread"`/`"rejected"` one is
    real work in progress, not ready to index. A malformed or schema-
    invalid file raises rather than being skipped: by the time a
    contribution is on `main`, CI's own `validate-contributions.yml` has
    already schema-checked it on its PR, so a failure here means something
    regressed, not an expected case to tolerate quietly.
    """
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    validated: list[dict[str, Any]] = []
    for path in sorted(Path(contributions_dir).rglob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        jsonschema_validate(instance=entry, schema=schema)
        if entry["metadata"]["validation"]["status"] == "validated":
            validated.append(entry)
    return validated


def _surface_form(tokens: list[dict[str, Any]]) -> str:
    return " ".join(t["surface"] for t in tokens if not t.get("gap") and t.get("surface"))


def build_relational_row(
    entry: dict[str, Any], *, model: str | None = None, embedding_model: str | None = None
) -> dict[str, Any] | None:
    """Builds one §3a row from one already-validated §3b `entry`, or
    `None` when `entry` has nothing word-level to index (a `grammar_note`,
    a free-text-only `origin` with no `tokens`, an all-gap token list, or a
    `null` `target` — the schema technically allows the last one outside
    `grammar_note`, even though §3b's own prose doesn't describe that case).
    """
    if entry["doc_type"] == "grammar_note":
        return None

    target = entry.get("target")
    if target is None:
        return None

    tokens = entry["origin"].get("tokens") or []
    form = _surface_form(tokens)
    if not form:
        return None

    iso_code = entry["origin"]["lang_code"]
    year = entry["metadata"]["date"]["year"]
    is_name = any(t.get("class") == "name" for t in tokens)

    if is_name:
        lemma = form
        upos = "PROPN"
        ner = "PER"
        sense_id = f"name:{lemma}"
    else:
        tags = classify.classify(form, iso_code, year, model=model)
        content_tag = next((t for t in tags if t.upos not in ("PUNCT", "X")), tags[0] if tags else None)
        lemma = (content_tag.lemma if content_tag else None) or form
        upos = content_tag.upos if content_tag else "X"
        ner = content_tag.ner if content_tag else "O"
        sense_id, _definition = resolve_sense_id(lemma, upos, target["text"], model_name=embedding_model)

    return {
        "iso_code": iso_code,
        "form": form,
        "lemma": lemma,
        "sense_id": sense_id,
        "attestation_year": year,
        "source": entry["metadata"]["source"],
        "doc_type": entry["doc_type"],
        "donor_language": None,
        "first_borrowing_year": None,
        "register": None,
        "dialect": None,
        "ner": ner,
    }


def build_relational_shards(
    entries: list[dict[str, Any]], out_dir: Path, *, model: str | None = None, embedding_model: str | None = None
) -> list[Path]:
    """Groups `entries` (already validated, via `load_contributions`) into
    one §3a relational row per usable contribution (`build_relational_row`),
    and writes one Parquet shard per `iso_code` under
    `<out_dir>/relational/<iso_code>/<year_from>-<year_to>.parquet` (spec
    §10's shard layout — volume-based quantile splitting isn't implemented
    here either, same deferral as `ingest.index`'s own docstring: not yet
    needed at this corpus's scale). Returns the Parquet paths written.
    """
    rows_by_iso: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        row = build_relational_row(entry, model=model, embedding_model=embedding_model)
        if row is None:
            continue
        rows_by_iso.setdefault(row["iso_code"], []).append(row)

    written: list[Path] = []
    for iso_code, rows in rows_by_iso.items():
        years = [r["attestation_year"] for r in rows]
        shard_path = Path(out_dir) / "relational" / iso_code / f"{min(years)}-{max(years)}.parquet"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), shard_path)
        written.append(shard_path)
    return written


def build_relational_index_for_contributions_root(
    contributions_root: Path, out_dir: Path, *, model: str | None = None, embedding_model: str | None = None
) -> list[Path]:
    """CLI entrypoint's own function (`scripts/build_relational_index.py`):
    reads every validated contribution under `contributions_root` and
    writes the relational shards `search_api`'s `/attest` reads. Returns
    `[]` (not an error) when no contribution is validated yet — mirrors
    this service's own soft-fail philosophy rather than treating "nothing
    to index yet" as exceptional.
    """
    entries = load_contributions(Path(contributions_root))
    if not entries:
        return []
    return build_relational_shards(entries, Path(out_dir), model=model, embedding_model=embedding_model)
