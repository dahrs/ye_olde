# ye_olde
A language translator using highly modern tools where the focus is on old and ancient languages.

Translates `(sentence, lang_code, year) → (sentence, lang_code, year)` — between any two points in a language's own history — using retrieval over a corpus partitioned by `(lang_code, year)`, not fine-tuning. Full design: [docs/diachronic-translation-pipeline-plan.md](docs/diachronic-translation-pipeline-plan.md).

## Status

Translation pipeline (`src/ye_olde/`): scaffolding only, no logic implemented yet.
Search API (`search_api/`): stood up and deployable, no real data yet. See the spec's §11 for
the first scoped build task (gather + align a corpus, then it has something to serve).

## Project layout

```
src/ye_olde/
├── api.py          # translate() — the public entrypoint (spec §1)
├── config.py       # settings: embedding model, generation LLM, Search API URL
├── resolver/       # (lang_code, year) -> weighted corpus partitions (§2, §3a)
├── classify/       # name vs. common-word vs. anachronism token classification (§2)
├── retrieval/      # thin HTTP client for the Search API (§10) + temporal rerank
├── fallback/       # native -> loan -> constructed -> temporal-loan chain (§2, §3a)
├── generation/      # grounded LLM generation + self-check (§2)
├── ingest/         # corpus ingestion into the queryable index (§3a, §5, §6)
└── schema/         # JSON Schema for community contributions (§3b)

search_api/         # standalone service (spec §10) — deployed to Google Cloud Run,
                     # separately from the pipeline above; see search_api/README.md
contributions/      # community-submitted JSON entries (§3b), validated by CI on PR
data/               # gitignored corpora/index artifacts (raw/processed/index)
scripts/            # CLI entrypoints (ingestion runner, translate demo)
tests/              # unit/ and integration/
```

## Setup

```
uv sync
cp .env.example .env   # fill in EMBEDDING_MODEL / LITELLM_MODEL / SEARCH_API_URL
```

## Local LLM (llama.cpp)

The generation LLM (`LITELLM_MODEL`) doesn't have to be a hosted API — this repo is public, and
anyone cloning it can point it at their own local model instead. This section documents a real,
tested setup (built and run on the reference Pi 5), not just a suggestion.

**1. Build `llama-server`** (llama.cpp's own OpenAI-compatible server), into a shared location
outside any repo so other projects can reuse the same binary/models:
```
mkdir -p ~/tools && cd ~/tools
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install cmake   # only if your system's package manager isn't available (e.g. no sudo) —
                     # PyPI ships a prebuilt cmake binary for common platforms including aarch64
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
cmake --build build --config Release --target llama-server -j$(nproc)
```

**2. Download a GGUF model** into a shared directory (also outside any repo, e.g. an external
drive if local storage is tight):
```
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='unsloth/Qwen3.5-9B-GGUF', filename='Qwen3.5-9B-Q4_K_M.gguf',
                 local_dir='/path/to/shared/models/qwen3.5-9b-gguf')
"
```
Pick a size/quantization your machine can actually hold **with real headroom to spare, not just
barely fit** — check `free -h`'s `available` column against the file size before assuming it'll
work, and be aware other things already running (a full desktop environment, in the reference
case) can permanently claim several GB. A model that technically fits but leaves no margin will
get its memory-mapped pages evicted and re-read from disk during inference, which is far slower
than compute-bound generation and easy to mistake for "this hardware is just slow." If a model
you download turns out to be split into several `-00001-of-000NN.gguf` shards, `llama-server`
loads them natively by pointing `-m` at the first shard — no merge step needed.

**3. Run the server:**
```
~/tools/llama.cpp/build/bin/llama-server -m /path/to/shared/models/.../<file>.gguf \
  --port 8080 --host 127.0.0.1 -c 4096
```

**4. Point `.env` at it:**
```
LITELLM_MODEL=openai/<any-name>            # llama-server ignores the exact string, but litellm needs the openai/ prefix
LITELLM_API_BASE=http://localhost:8080/v1
LITELLM_API_KEY=sk-local                   # llama-server doesn't check it, litellm just needs something non-empty
LITELLM_TIMEOUT_SECONDS=6000               # litellm's own implicit default is far too short for a slow local reasoning call
LITELLM_NO_THINKING_EXTRA_BODY={"chat_template_kwargs": {"enable_thinking": false}}
```
Leave `LITELLM_EXTRA_BODY` itself blank. Qwen3(.5)-family models default to emitting a
"thinking" preamble before the real answer — left alone, deliberately, since reasoning can
genuinely help `ingest/llm_client.py`'s close-reading/alignment judgment calls. The risk is a
reasoning trace long enough to fill the whole context window before producing an answer, which
comes back as empty `content`. `call_llm_json` handles that itself: one retry of the same call
using `LITELLM_NO_THINKING_EXTRA_BODY` instead, then reasoning is back on for the next call
regardless. Leave that setting blank and a call like that raises instead of being rescued. Other
model families disable their equivalent reasoning mode differently (Anthropic:
`{"thinking": {"type": "disabled"}}`; OpenAI reasoning models: `{"reasoning_effort": "low"}`) —
both extra-body settings are deliberately raw passthroughs rather than code guessing at one
provider's convention.

**Honest performance note**: on the reference Pi 5 (16GB, CPU-only, desktop environment already
using ~7.5GB), Qwen3.5-9B-Q4_K_M generates at roughly **1.5–1.7 tokens/second** once properly
cached in RAM — and with reasoning left on, that's tokens spent thinking *and* answering. A real
test of a single one-sentence chunk took **~15 minutes** (1,472 tokens, entirely on the primary
reasoning-enabled call). Reasoning length isn't fixed either: the same prompt exhausted the full
4096-token context and came back empty in one run, then finished fine within budget on the next
attempt — which is exactly the scenario the empty-content fallback above exists for. Given
`clean.py`'s real chunks run up to 6000 characters, expect real calls to take considerably
longer than 15 minutes — this is overnight-batch-job territory, not something to expect
chat-speed responsiveness from, and worth raising `-c` above 4096 if reasoning is regularly
getting cut off. This number is specific to that one machine, not a property of the model: it's
entirely gated by whatever hardware you're actually running it on, and anyone cloning this repo
with a stronger machine (more free RAM, an SSD instead of a spinning disk, more CPU cores)
should expect meaningfully better throughput. Benchmark your own setup before assuming either
way.

## Search API (spec §10)

```
cd search_api
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in HF_DATASET_REPO_ID once a dataset repo exists
uvicorn app.main:app --reload --port 8000
```

Deploys automatically to Google Cloud Run on push to `main` — see
`.github/workflows/deploy-search-api.yml`. Data (Parquet/FAISS shards) stays on the free
Hugging Face Dataset repo regardless; only compute runs on Cloud Run. One-time setup (GCP
project/billing/Workload Identity Federation + GitHub secrets) isn't automatable from here —
see the steps given alongside this scaffold.
