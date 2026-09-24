# ye_olde coding standards

This file is the source of truth for how code in this repo is written. It's
enforced, not just advisory: `ruff` and `mypy --strict` run in pre-commit and
in CI (`.github/workflows/lint-and-test.yml`) for both the `ye_olde` package
(`src/`) and the `search_api` service, which is deployed separately and has
its own environment (see `search_api/pyproject.toml`, `search_api/app/config.py`'s
docstring for why it doesn't depend on the `ye_olde` package at all).

Two codebases, one rulebook: everything below applies equally to `src/ye_olde/`
and `search_api/app/`, unless a rule explicitly says otherwise.

## Running the checks locally

```
uv sync --extra dev --extra ingest      # root package
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest tests -q

cd search_api
pip install -r requirements-dev.txt     # separate environment, see above
ruff check .   ruff format --check .    # (run from repo root to share config; or from here)
mypy app
pytest tests -q
```

`pre-commit install` wires the same `ruff`/`mypy` checks into `git commit`
(see `.pre-commit-config.yaml`).

## Typing

Every function/method signature (parameters and return type) and every class
attribute is typed. `mypy --strict` is the enforcement mechanism — see
`[tool.mypy]` in `pyproject.toml` (root package) and `search_api/pyproject.toml`
(search_api, checked separately because it has its own, different
dependency set).

- Generics are always parametrized: `dict[str, Any]`, `list[PairRecord]` —
  never bare `dict`/`list`.
- When a `dict`'s shape is heterogeneous and reused across a module (see
  `ye_olde.ingest.align.PairRecord`), name it with a `TypeAlias` instead of
  repeating `dict[str, Any]` everywhere. Reach for a `pydantic.BaseModel`
  instead once the shape is stable and worth validating — see "Pydantic".
- A third-party dependency with no type stubs gets exactly one
  `ignore_missing_imports` entry in `[[tool.mypy.overrides]]`, not scattered
  per-call-site `# type: ignore`s.
- `from __future__ import annotations` at the top of every module (already
  the convention here) — lets modern union syntax (`str | None`) work
  regardless of the target Python version, and means forward references
  never need quoting.
- A function typed to return a concrete type (e.g. `list[str]`) must not
  return an untyped/`Any` expression directly (`mypy`'s `no-any-return`
  under `--strict` catches this) — assign the `Any` expression (a
  `json.loads()` result, most commonly) to an explicitly annotated local
  first, then return that.

## Error handling

**Only a boundary layer catches broad exceptions, and it always logs what it
catches.** A boundary is: a FastAPI route/exception handler in `search_api`,
or a CLI script's `main()` (`scripts/*.py`). Everywhere else — `ye_olde.ingest.*`,
`ye_olde.retrieval`, etc. — let exceptions propagate. Catching broadly inside
domain code to "keep going" is exactly what hides the bug that should have
been fixed; don't do it.

- **Logging**: `ye_olde.common.logging.get_logger(__name__)` (root package)
  or `search_api.app.logging_config.get_logger(__name__)` (search_api —
  deliberately a separate, non-shared implementation of the same pattern,
  for the same reason `search_api/app/config.py` doesn't import
  `ye_olde.config`). Both write to a rotating file under `logs/` (gitignored,
  created on first use) — `logs/ye_olde.log` / `logs/search_api.log`.
  A caught-and-handled error is invisible to whoever runs this later unless
  it's in that file.
- A boundary's catch-all logs the full exception (`log.exception(...)`,
  which includes the traceback) *before* converting it to a response — a
  generic 500 for FastAPI (see `search_api/app/main.py`'s
  `unhandled_exception_handler`), a non-zero exit for a CLI script.
- Inside domain code, use a **specific** exception type when you want
  documented fallback behavior for a known, expected condition (e.g.
  `ye_olde.ingest.llm_client.is_local_model` catching a model-string
  resolution failure) — never a bare `except Exception` to paper over
  something unexpected. Even a fully-handled fallback like this should log
  at `debug`/`info` (via `get_logger`) so it's visible in the log file
  instead of disappearing silently — "no silent errors" applies everywhere,
  not just at the boundary.
- **Which exception type to raise**: a plain builtin (`ValueError`,
  `TypeError`, `KeyError`, `FileNotFoundError`, ...) for "this input/state is
  invalid" — that's most cases, and it's what the codebase already does
  (`corpus_files.parse_corpus_filename`, `checkpoint.Checkpoint`, etc.).
  Reach for `ye_olde.common.errors.YeOldeError` (or a subclass) only when a
  boundary layer genuinely needs to distinguish this failure from an
  arbitrary bug — e.g. `LLMNotConfiguredError` vs. `LLMEmptyResponseError`,
  both things `ingest.llm_client` callers branch on. Don't invent a new
  `YeOldeError` subclass for a one-off "this argument is malformed" check;
  that's what `ValueError` is for.
- Never write `except Exception: pass` (or any bare catch with no logging).
  If you're tempted to, either the exception should propagate, or it's a
  documented, specific, logged fallback per the point above.

## Prompts

Every LLM prompt (system prompt or prompt template) lives in
`src/ye_olde/prompt/<namespace>.yaml`, one YAML file per top-level package
that prompts an LLM — `ingest.yaml` today; `generation.yaml`/`classify.yaml`
etc. as those packages get implemented. Load with
`ye_olde.prompt.load_prompt(namespace, key)`.

- Prompts are data, not code: this keeps wording reviewable/diffable without
  touching control flow, and keeps every prompt in the project discoverable
  in one place instead of buried in whichever module happens to call the LLM.
- A module-level constant (e.g. `_SYSTEM_PROMPT` in `ingest/clean.py`) is
  still the right way to bind a loaded prompt for use within that module —
  only the *literal text* moves to YAML, not the reference to it.
- `load_prompt` raises `PromptNotFoundError` (not a silent empty string) if
  the namespace file or key doesn't exist — an LLM call should never run
  with a missing prompt it didn't notice was missing.
- `search_api` has no LLM calls and no `prompt/` package of its own; this
  section only applies to `ye_olde.*`.

## PEP 8 / formatting

Enforced by `ruff check` (linting: pycodestyle, pyflakes, import order,
pyupgrade, bugbear, PEP 8 naming, comprehensions — see `[tool.ruff.lint]`
`select`) and `ruff format` (formatting — replaces black/isort). Both run in
pre-commit and CI; a PR that fails either doesn't merge. Line length: 120
(see `[tool.ruff]`), chosen to match this codebase's already-dense,
comment-heavy style rather than PEP 8's default 79.

Don't hand-format to work around the formatter — run `ruff format` and
commit what it produces.

## Naming

- **Functions**: `snake_case`, verb-first, named for what they do — short,
  but not at the cost of being unclear (`chunk_text`, `is_local_model`,
  `align_corpus_pair`, not `proc` or `do_the_alignment_thing_for_a_pair`).
  A leading underscore (`_normalize`, `_finalize_records`) means "private to
  this module" — use it for helpers callers outside the module have no
  business calling directly.
- **Classes**: `PascalCase` nouns (`CorpusFile`, `Checkpoint`, `Settings`).
- **Variables**: `snake_case`, and enforced via `ruff`'s `N` (PEP 8 naming)
  rules — this includes function *parameters*: `lang_a`/`year_a`, not
  `lang_A`/`year_A`, even when mirroring a spec doc's notation.
- **Constants**: `UPPER_SNAKE_CASE` at module scope (`_DEFAULT_MAX_CHARS`,
  `SUPPORTED_EXTENSIONS`).
- **Tests**: one test file per module under test, named `test_<module>.py`
  (`ye_olde/ingest/align.py` -> `tests/unit/test_align.py`). Within it, name
  each test `test_<function_or_class>_<scenario>` — describe the behavior
  being verified, not just the function name, so a failing test's name
  alone tells you what broke (`test_is_local_model_false_for_hosted_api`,
  not `test_is_local_model_2`). A test file may cover more than one function
  from the same module, but shouldn't reach across modules — if you're
  testing `corpus_files.py` behavior, it belongs in `test_corpus_files.py`,
  even if the test was written while working on `align.py`.

## Pydantic

Use a `pydantic.BaseModel` (not a `TypedDict`, not a bare `dict`) for any
structured data that:

- crosses a boundary (an API response — see `search_api/app/schemas.py`; a
  public function's return value — see `ye_olde.api.Translation`), or
- is validated on the way in (settings — `Settings` in both `config.py`
  files; a community contribution — see `src/ye_olde/schema/contribution.schema.json`,
  currently plain JSON Schema since it's validated by a non-Python CI step
  too, not just Python code).

Don't reach for pydantic for a purely-internal, freely-mutated working
structure that's built up incrementally across several functions before it's
finalized (`ye_olde.ingest.align.PairRecord` is exactly this case — see the
`TypeAlias` comment in `align.py`) — a `BaseModel` there would need most
fields `Optional` from the start and would fight the mutation pattern for no
real safety gain. Give it a `TypeAlias` for readability instead, and
validate it into a real model only once it's complete, if it then crosses a
boundary.

A `BaseModel` field gets a real type, not `dict`/`list` — nest another
`BaseModel` (see `AttestQuery`/`LookupQuery`/`LookupTarget` in
`search_api/app/schemas.py`) rather than accepting an untyped blob.

## Avoid duplicate functions

Before writing a helper, check whether `ye_olde.common` (cross-cutting:
`errors.py`, `logging.py`) or an existing sibling module already does it. If
two modules within the *same* deployable need the same logic, it belongs in
one shared place they both import — not copy-pasted.

The one deliberate exception: `search_api` reimplements small pieces of the
same *pattern* used in `ye_olde.common` (its own `logging_config.py`, its
own `Settings` in `app/config.py`) rather than importing `ye_olde.common`
directly. That's not duplication of the kind this rule is about — it's two
independently-deployed services that don't share a dependency tree by
design (see `search_api/app/config.py`'s docstring). Don't "fix" that by
adding `ye_olde` as a `search_api` dependency.

## Project layout

```
src/ye_olde/
  common/        exception hierarchy (errors.py), logging setup (logging.py)
  prompt/        <namespace>.yaml prompt files + loader.py
  ingest/        corpus acquisition/cleaning/alignment pipeline (spec §6)
  generation/, classify/, fallback/, resolver/, retrieval/   not yet implemented (spec §2)
  config.py      Settings (pydantic-settings), env-driven
  api.py         public translate() entrypoint (spec §1) — not yet implemented
search_api/      separately deployed FastAPI service (spec §10) — own deps, own config, own logging
scripts/         CLI entrypoints (e.g. align_corpus.py) — a boundary layer, see "Error handling"
eval/            manual evaluation harness, not run in CI (see eval/README.md)
tests/unit/, tests/integration/
```
