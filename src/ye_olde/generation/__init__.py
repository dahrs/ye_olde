"""Grounded LLM generation + self-check — spec §2.

Generates the output sentence constrained to retrieved attested forms, then
self-checks the result against them before returning. LLM backend (API
provider or local/OpenAI-compatible endpoint) is read from
ye_olde.config.Settings via litellm — not yet implemented.
"""
