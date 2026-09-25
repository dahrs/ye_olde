# ye_olde
A language translator using highly modern tools where the focus is on old and ancient languages.

Translates `(sentence, lang_code, year) → (sentence, lang_code, year)` — between any two points in a language's own history — using retrieval over a corpus partitioned by `(lang_code, year)`, not fine-tuning. Full design: [docs/diachronic-translation-pipeline-plan.md](docs/diachronic-translation-pipeline-plan.md).

## Status

Translation pipeline (`src/ye_olde/`): scaffolding only, no logic implemented yet.
Search API (`search_api/`): deployed and serving real data — `/lookup` and `/search` both work
against *Sir Gawayne and the Green Knight* (enm 1400 ↔ eng 1999, 704 aligned pairs). `/attest`
has no data yet (spec §3a — community contributions aren't ingested yet). See spec §13e for the
current state of each endpoint, and §7 for the next build-order step.

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

llama.cpp's own server is one way to do this, not the only one — e.g. Ollama works too, with its
own native `LITELLM_MODEL=ollama/<name>` prefix instead of an OpenAI-compatible endpoint (litellm
resolves that prefix to a genuinely different code path, not just a naming choice). This section
documents llama.cpp specifically because that's the setup actually built and benchmarked here;
`ye_olde.ingest.llm_client.is_local_model()` recognizes both patterns (and others — see that
function's docstring) when deciding whether a call is free to re-check with a second pass.

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
**This does not call the real OpenAI API.** The `openai/` prefix is only litellm's way of picking which wire
protocol to speak — llama-server deliberately implements that same request/response shape so any
OpenAI-compatible client can talk to it, but every request still goes to `LITELLM_API_BASE` above (your own
machine), never to `api.openai.com`. `LITELLM_API_KEY` is a throwaway placeholder for the same reason:
llama-server doesn't check it, litellm just requires the field to be non-empty. If this ever looks
suspicious, two easy ways to confirm you're actually hitting the local server: response latency (real OpenAI
answers in ~1-2s; this setup is ~1.5-1.7 tok/s, so anything more than a couple words takes minutes) and the
fact that a fake API key like `sk-local` works at all (the real API would reject it immediately).
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
`.github/workflows/deploy-search-api.yml`. Data (Parquet/FAISS shards) stays on a free
Hugging Face Dataset repo regardless; only compute runs on Cloud Run.

### Reference deployment

The maintainer's own instance, so third-party deployers have a working example to compare
against — none of this is secret, all of it is safe to be public (Workload Identity Federation,
below, means knowing these values alone doesn't grant access to anything):

| | |
|---|---|
| Search API URL | `https://ye-olde-search-api-say7jittea-uc.a.run.app` |
| GCP project | `ye-olde-search-api` |
| Cloud Run service | `ye-olde-search-api`, region `us-central1` |
| HF Dataset repo | [`dahrs/ye_olde_data-index`](https://huggingface.co/datasets/dahrs/ye_olde_data-index) (public) |

Smoke test:
```
curl "https://ye-olde-search-api-say7jittea-uc.a.run.app/health"
curl "https://ye-olde-search-api-say7jittea-uc.a.run.app/lookup?text=gladly&lang=enm&year=1400&target_lang=eng&target_year=1999"
curl "https://ye-olde-search-api-say7jittea-uc.a.run.app/search?text=greetings,%20sir&lang=enm&top_k=3"
```
`/search` embeds the query with `sentence-transformers` at request time, so the first call after the
service scales to zero re-downloads the ~2.2GB model (~60–80s measured); warm calls are ~1–2s. This is
also why the service runs at `--memory=4Gi --cpu=2` rather than the Cloud Run default.

### Setting up your own deployment

Two accounts, both free tier: a Google Cloud project for compute, a Hugging Face account for
data storage. High level (a coding assistant with shell access can follow the fuller version of
this in the project's spec doc/artifact §13, which has exact commands):

1. **Hugging Face**: create a Dataset repo (public) to hold the Parquet/FAISS shards — this is
   `HF_DATASET_REPO_ID` below. Create a write-scoped access token (huggingface.co/settings/tokens)
   — needed even for a public repo, since Hugging Face rate-limits fully anonymous API reads (see
   spec §10).
2. **Google Cloud**: create a project, link billing (Cloud Run's free tier needs billing enabled
   even though it won't be charged at this scale — see spec §10 for the actual numbers), enable
   `run.googleapis.com`, `artifactregistry.googleapis.com`, `cloudbuild.googleapis.com`,
   `iamcredentials.googleapis.com`, `secretmanager.googleapis.com`.
3. **Store the HF token in Secret Manager** (`gcloud secrets create hf-token --data-file=...`),
   never as a GitHub secret or a plain env var — see spec §10 for why.
4. **Create a deploy service account** and grant it `roles/run.admin`,
   `roles/artifactregistry.admin`, `roles/cloudbuild.builds.editor`, `roles/iam.serviceAccountUser`,
   `roles/storage.admin`, plus `roles/secretmanager.secretAccessor` on the `hf-token` secret.
5. **Set up Workload Identity Federation** between GitHub Actions and that service account,
   scoped to your fork's `owner/repo` — this is what lets `.github/workflows/deploy-search-api.yml`
   deploy without a downloadable, leakable GCP key ever existing.
6. **Add to your fork's GitHub Actions secrets/variables**:

   | Name | Kind | Value |
   |---|---|---|
   | `GCP_WORKLOAD_IDENTITY_PROVIDER` | secret | full WIF provider resource name from step 5 |
   | `GCP_SERVICE_ACCOUNT` | secret | the deploy service account's email from step 4 |
   | `GCP_PROJECT_ID` | variable | your GCP project ID |
   | `GCP_REGION` | variable | a Cloud Run region, e.g. `us-central1` |
   | `HF_DATASET_REPO_ID` | variable | your HF dataset repo ID from step 1 |

   `HF_TOKEN` is deliberately not in this table — the workflow references the Secret Manager
   secret from step 3 directly (`hf-token:latest`), so the raw token value never touches GitHub
   at all.
7. Push to `main` — the workflow builds and deploys. First deploy takes a few minutes.

The full version of these steps, with exact `gcloud` commands, is in the project's spec
doc/artifact — see its §13.
