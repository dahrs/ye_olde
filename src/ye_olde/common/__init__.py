"""Cross-cutting utilities shared across every `ye_olde` package: the
project's exception hierarchy (`errors.py`) and its logging setup
(`logging.py`). Anything one module needs and another would otherwise
reimplement belongs here — see CLAUDE.md "Avoid duplicate functions".
"""

from __future__ import annotations
