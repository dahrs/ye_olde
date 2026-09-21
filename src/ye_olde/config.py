"""Central settings, loaded from environment / .env (see .env.example).

Keeps the deliberately-deferred choices (embedding model, generation backend,
vector store) as config values rather than hard-coded imports, so picking them
later is a settings change, not a code change.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Embedding backend — self-hosted via sentence-transformers (see spec §4, §8).
    embedding_model: str = ""

    # Generation LLM — via litellm, so this can be an API model string
    # (e.g. "anthropic/claude-...") or a local/OpenAI-compatible endpoint
    # (e.g. "ollama/<model>" with litellm_api_base set).
    litellm_model: str = ""
    litellm_api_key: str = ""
    litellm_api_base: str = ""

    # Vector store backend (see spec §4).
    vector_store: str = ""


def get_settings() -> Settings:
    return Settings()
