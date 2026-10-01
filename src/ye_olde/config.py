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
    # (e.g. "ollama/<model>" with litellm_api_base set). litellm_api_key is
    # this default backend's credential (today: the Anthropic key, for
    # LITELLM_MODEL=anthropic/claude-...).
    litellm_model: str = ""
    litellm_api_key: str = ""
    litellm_api_base: str = ""

    # A second, independent credential for OpenAI models hosted on AWS
    # Bedrock's Mantle endpoint (e.g. GPT-6 Sol/Luna/Astra, LITELLM_MODEL=
    # "bedrock_mantle/openai.gpt-6-sol") -- kept side by side with
    # litellm_api_key rather than replacing it, so both Claude's and this
    # Bedrock key stay configured at once, switched between purely via
    # which provider LITELLM_MODEL (or a --model override) resolves to.
    # This is AWS's own long-term Bedrock API key (bearer-token auth, not
    # the classic AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY SigV4 pair) --
    # used when litellm_client._resolve_api_key determines the call is
    # going to provider "bedrock_mantle".
    aws_bearer_token_bedrock: str = ""

    # Optional raw JSON object merged into every litellm.completion() call as
    # extra_body — for whatever provider-specific parameter your chosen
    # backend always needs. Left blank, nothing extra is sent, which for a
    # reasoning-capable backend (Qwen3-family served locally, for instance)
    # means "thinking" stays on by default — deliberate: reasoning can
    # genuinely help this pipeline's close-reading/alignment judgment calls,
    # so it isn't switched off just to be safe (see common/llm_client.py).
    litellm_extra_body: str = ""

    # Optional raw JSON object used only as a one-shot RETRY on a single
    # call whose content came back empty — the signature of a reasoning
    # trace that consumed the whole context window before producing an
    # actual answer. This does not change LITELLM_EXTRA_BODY or any later
    # call: it rescues the one call that hit the ceiling, then reasoning is
    # back on next time. Provider-specific, same reasoning as above for why
    # it's a blank passthrough rather than a hardcoded convention: for a
    # llama.cpp/vLLM-served Qwen3-family model this is
    # {"chat_template_kwargs": {"enable_thinking": false}}; Anthropic's
    # equivalent is {"thinking": {"type": "disabled"}}; OpenAI reasoning
    # models use {"reasoning_effort": "low"}. Left blank, a call that comes
    # back empty raises instead of being silently retried.
    litellm_no_thinking_extra_body: str = ""

    # Request timeout (seconds) for every litellm.completion() call. Set
    # explicitly rather than left to litellm's own implicit default — that
    # default turned out, in practice, to be far shorter than a slow local
    # reasoning-enabled call can need (an unmodified request timed out
    # around ~10 minutes into an uncapped local generation). Generous by
    # design: a deployer who explicitly wants reasoning enabled on slow
    # hardware has already accepted long calls as the cost of that (see
    # common/llm_client.py); a timeout here should only fire on a genuinely
    # hung connection, not on a model that's still legitimately working.
    litellm_timeout_seconds: float = 6000.0

    # Search API base URL (spec §10) — the pipeline is an HTTP client of it,
    # never holds the corpus/index locally. No default: fail loudly rather
    # than silently retrieving nothing if this isn't set.
    search_api_url: str = ""

    # Request timeout (seconds) for every call to the Search API. Generous
    # by design, same reasoning as litellm_timeout_seconds above: /search
    # embeds its query with sentence-transformers at request time, so the
    # first call after the service scales to zero re-downloads its ~2.2GB
    # model from the Hugging Face Hub before it can answer — measured live
    # at ~70-77s (search_api/README.md). A 30s timeout was tried first and
    # reliably failed on exactly this, the common case (any call after a
    # ~15-minute idle gap), not a rare edge case. Warm calls return in ~1-2s
    # regardless of this setting.
    search_api_timeout_seconds: float = 120.0

    # The Hugging Face Dataset repo search_api serves from (spec §10/§13d)
    # -- read by scripts/push_index_to_hf.py to push scripts/build_index.py's
    # local output there. Same repo search_api/app/config.py's own
    # hf_dataset_repo_id points at in production; kept as a separate .env
    # value here rather than shared, same reasoning as everywhere else this
    # project keeps the two deployables' config independent (search_api/app/
    # config.py's docstring).
    hf_dataset_repo_id: str = ""

    # A write-scoped HF token -- needed to push, unlike search_api's own
    # hf_token (read-only access is enough to serve). Never committed;
    # .env only, same as every other credential in this file.
    hf_token: str = ""


def get_settings() -> Settings:
    return Settings()
