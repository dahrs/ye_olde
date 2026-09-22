"""Settings for the search_api service.

Separate from the main ye_olde package's config.py — this runs as its own
deployable (spec §10), not inside the translation pipeline's process, so it
has its own env vars and its own .env for local dev (see .env.example here).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # The Hugging Face Dataset repo holding the Parquet shards (spec §10),
    # e.g. "dahrs/ye-olde-index". No default — a misconfigured deployment
    # should fail to find data loudly, not silently serve empty results
    # forever without anyone noticing why.
    hf_dataset_repo_id: str = ""

    # Only needed if the dataset repo is private. v1 assumes a public repo
    # (spec §9: auth deliberately deferred), so this stays empty by default.
    hf_token: str = ""

    # Local cache for downloaded shards (huggingface_hub caches by content
    # hash, so a shard is only re-downloaded when it actually changes).
    # /tmp is the one reliably-writable path in a Spaces Docker container.
    hf_cache_dir: str = "/tmp/hf-cache"


def get_settings() -> Settings:
    return Settings()
