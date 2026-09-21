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
    # multilingual-e5-large: best-quality e5 variant our target hardware (Pi 5,
    # 16GB RAM, CPU-only) can run — RAM isn't the constraint for a single
    # forward-pass encoder, so no need to drop to the base/small variants.
    embedding_model: str = "intfloat/multilingual-e5-large"

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
