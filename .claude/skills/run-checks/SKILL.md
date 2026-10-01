---
name: run-checks
description: Use when asked to run this project's lint/type/test checks, verify a change will pass CI, or set up the dev environment (first clone, pre-commit install). Covers both the root ye_olde package and search_api's separately-installed environment. Not needed for ordinary code edits — ruff/mypy/pytest are run directly via Bash when checking a change, not loaded as context every turn.
---

# Running ye_olde's checks locally

Two independent environments — don't mix their commands up.

## Root package (`ye_olde`)

```
uv sync --extra dev --extra ingest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest tests -q
```

## `search_api` (separately deployed, own deps — see `search_api/app/config.py`'s docstring for why)

```
cd search_api
pip install -r requirements-dev.txt
ruff check .   ruff format --check .    # run from repo root to share config, or from here
mypy app
pytest tests -q
```

**On this reference Pi 5, local dev for both codebases shares the root `.venv`** rather than
`search_api` getting its own — the separation above is about *deployment* independence
(`search_api` ships to Cloud Run from its own `requirements.txt`/`Dockerfile`, never anything
installed here), not a claim that local dev needs two copies of largely-overlapping heavy
dependencies (`torch`/`sentence-transformers`/`pyarrow` are needed by both). Root's `.venv` has
the 3 packages `search_api` needs that the `ye_olde[ingest]` extra doesn't (`fastapi`, `uvicorn`,
`huggingface_hub`) installed ad hoc via `uv pip install --python .venv/bin/python ...` — run
`search_api`'s own test/lint/type-check commands above with `../.venv/bin/python -m <tool>` in
place of a `search_api/.venv`. Local-only convenience, not documented in `search_api/README.md`
(that file is for third-party deployers, who should follow its own separate-venv instructions).

## Pre-commit

`pre-commit install` wires the same `ruff`/`mypy` checks into `git commit` (see `.pre-commit-config.yaml`) — run once per clone, not per change.

A PR that fails `ruff check`, `ruff format --check`, or `mypy --strict` on either environment doesn't merge — see root `CLAUDE.md` for the actual coding standards these checks enforce.
