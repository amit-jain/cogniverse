# Models and Inference Deployment

Every served model, the container image that serves it, and how the
deployment differs between CPU, ROCm, and CUDA. The chart's
`values.yaml`, `values.rocm.yaml`, `values.cuda.yaml`, and the
`deploy/` sidecars are the underlying source of truth; this page
flattens them into one reference.

---

## Overview

Cogniverse runs four classes of inference services:

| Class | Purpose |
|---|---|
| **LLMs** (chat / generation) | Agent reasoning, query enhancement, distillation. Two tiers: a small **student** model used at runtime, and a larger **teacher** model used only during DSPy MIPROv2 optimization. |
| **Visual / multimodal embeddings** | ColPali + ColQwen for video/image patch embeddings, VideoPrism for chunk embeddings. |
| **Text embeddings** | ColBERT-style late-interaction (LateOn, served by vLLM) for documents/code, DenseOn (ModernBERT) for query/single-vector text. |
| **Audio (ASR)** | Whisper transcription of audio files. |

Each service lives in `charts/cogniverse/values.yaml` under the
`inference:` block (or under top-level `llm:` for the LLM serving
path). The chart renders one Deployment + Service per enabled service.

---

## LLM serving

### Student (default chat / generation LLM)

| Field | Value |
|---|---|
| Chart key | `inference.vllm_llm_student` (and the `llm.*` block when `engine: vllm`) |
| Model | `google/gemma-4-e4b-it` |
| Image (CPU) | `vllm/vllm-openai-cpu:latest` (official) |
| Image (ROCm) | `vllm/vllm-openai-rocm:v0.23.0` (official) |
| NodePort | 29010 |
| Default state | enabled |
| ROCm GPU memory | `--gpu-memory-utilization 0.30` (≈19 GiB on 62 GiB unified memory) |

The student is the primary chat LLM used by every agent for
DSPy/litellm calls. The Helm template (`cogniverse.primaryLLMModel` in
`templates/_helpers.tpl`) always prepends the `openai/` provider prefix
and writes the resulting model id verbatim into `config.json`;
`create_dspy_lm()` passes it through unchanged. The actual destination
is determined by `api_base`, not the prefix.

### Teacher (DSPy MIPROv2 optimization only)

| Field | Value |
|---|---|
| Chart key | `inference.vllm_llm_teacher` |
| Model | `cyankiwi/Qwen3.6-27B-AWQ-INT4` (AWQ-INT4, ~14 GiB) |
| Image (CPU) | `vllm/vllm-openai-cpu:latest` (official) |
| Image (ROCm) | `vllm/vllm-openai-rocm:v0.23.0` (official) |
| NodePort | 29011 |
| Default state | **`replicaCount: 0`** — scale-to-zero |
| `--max-model-len` | 262144 |

The teacher is **scaled to zero by default** because it's only needed
when `cogniverse_runtime.optimization_cli` runs prompt optimization with
a teacher configured (the larger model proposes few-shot demos for the
optimizer to evaluate). Bring it up before optimization runs:

```bash
kubectl scale deployment/cogniverse-vllm-llm-teacher -n cogniverse --replicas=1
kubectl rollout status deployment/cogniverse-vllm-llm-teacher -n cogniverse --timeout=600s
# ... run optimization ...
kubectl scale deployment/cogniverse-vllm-llm-teacher -n cogniverse --replicas=0
```

The agent code wires it via `LLMConfig.teacher` → `create_dspy_lm()` →
`teacher_settings` on `BootstrapFewShot` in
`libs/agents/cogniverse_agents/optimizer/dspy_agent_optimizer.py`, driven
by `cogniverse_runtime.optimization_cli`. See the
[ColPali/ColQwen embedding flow](#visual-embeddings) for the runtime
side that consumes the optimized program.

### Optional: Ollama instead of vLLM

| Field | Value |
|---|---|
| Chart key | `llm.ollama` (only when `llm.engine: ollama`) |
| Model | `gemma3:4b` (configurable via `llm.model`) |
| Image | `ollama/ollama:0.20.5` (official) |
| Deployment style | StatefulSet + PVC for the model cache |
| Default state | opt-in via `llm.engine: ollama`, `llm.builtin.enabled: true` |

Use Ollama for local development on machines without a vLLM-ready GPU.
The Helm template (`cogniverse.primaryLLMModel`) writes `openai/<model>`
into `config.json` regardless of engine; `llm.engine` only selects the
`api_base` URL (pointing at the in-cluster Ollama `/v1` endpoint in this
case). Modern Ollama exposes `/v1/chat/completions`, so the OpenAI-compat
wire contract routes to it unchanged.

### Optional: external LLM endpoint

Set `llm.engine: external` and `llm.external.url: <your-endpoint>` to
deploy nothing and route runtime LLM calls to a host-side or
third-party endpoint (e.g. `http://host.k3d.internal:11434` for a
host-running Ollama).

### Route through the vLLM Semantic Router

`cogniverse up` deploys the vLLM Semantic Router (Envoy front-end + the router)
in front of the LLM backend, and the runtime routes every agent's LLM call
through it. The router forwards to the same in-cluster LLM the runtime would
otherwise call directly — the chart's `srUpstream*` helpers derive the upstream
host/port from `primaryLLMEndpoint`, so it tracks the `llm.engine` in use
(ollama → the `-llm` service, vllm → the `-vllm-llm-student` service, external →
the configured URL). The division of labor:

- **cogniverse** sends only *who* the tenant is — the tenant identity
  (`x-authz-user-id` = `tenant_id`) and its tier (`x-authz-user-groups`,
  resolved from `tenant_tiers`). It does **not** classify the request.
- **the router** gates the tenant's allowed model set by tier (its authz
  signal — which requires the identity header and refuses to evaluate role
  bindings without it) and classifies the request content itself
  (domain/complexity) to pick the model + reasoning mode.

The router's own policy — model catalog, tier→role bindings, and the
content-driven decisions — lives in the chart at
`charts/cogniverse/files/semantic-router/config.yaml` (v0.3 schema). The default
ships a `pro`/`free` tier split and a "technical domain → reasoning model"
decision; extend the `providers.models` / `routing.decisions` there as you add
models.

#### `SemanticRouterConfig` — the cogniverse side

`SystemConfig.semantic_router` (a `SemanticRouterConfig`) controls what
cogniverse sends. The helper
`cogniverse_foundation.config.semantic_router.apply_semantic_routing(endpoint,
config, tenant_id)` returns a copy of the endpoint config with `api_base`
rewritten to the router and the two authz headers merged onto `extra_headers`:

| Field | Meaning |
|---|---|
| `enabled` | Master switch. `False` ⇒ endpoint passes through untouched. |
| `semantic_router_url` | The router's OpenAI-compatible endpoint. Enabled with an empty value raises. |
| `tenant_tiers` | `tenant_id → tier` map; unknown tenants fall back to `default_tier`. |
| `default_tier` | Tier for tenants not in `tenant_tiers`. |
| `tier_header` / `user_id_header` | Header names for the tier / identity (default `x-authz-user-groups` / `x-authz-user-id`). |

The resolved headers win on a key collision with any pre-existing
`extra_headers`. The block is part of `SystemConfig`, which the runtime reads
from the config store (Vespa) — **not** from `config.json`. A deployed runtime
receives it from the chart via the `SEMANTIC_ROUTER_ENABLED` /
`SEMANTIC_ROUTER_URL` / `SEMANTIC_ROUTER_TENANT_TIERS` env vars, which
`main.py` folds into `SystemConfig.semantic_router` at boot (a malformed tier
map raises rather than silently emptying).

Agents build a router-aware LM through one shared helper,
`semantic_router.create_routed_lm(endpoint, config, tenant_id)`
(`apply_semantic_routing` + `create_dspy_lm`); `resolve_semantic_router_config(...)`
reads the block from a `ConfigUtils`-like accessor (a broken config store
raises — no silent bypass). `DynamicDSPyMixin` uses these at LM-construction
time, and the per-request paths (the orchestrator and the direct-build
execution agents) build their LM the same way with the request's `tenant_id`.

The `SemanticRouterConfig` dataclass defaults to disabled, so unconfigured
library use is a no-op — but `cogniverse up` turns routing on by default
(`semanticRouter.enabled: true`). Opt out with
`cogniverse up ... --set semanticRouter.enabled=false`. The router downloads
its classifier bundle on first boot into a model-cache PVC, so allow the
startup probe time. Coverage:
`tests/foundation/integration/test_semantic_router_e2e.py` self-launches the
real router+Envoy+stub via `docker run` and asserts the tier/content decisions;
`tests/e2e/deployment/test_semantic_router_deploy_e2e.py` rides the
`deployed_stack` fixture (its own isolated k3d cluster running the full chart)
and asserts the routing *decision* per tenant tier + content against the
deployed router's `llm_decision_match_total` metric; `tests/charts/test_semantic_router_chart.py`
pins the rendered upstream endpoint and served model per `llm.engine`.

---

## Visual embeddings

### ColPali (per-tenant visual retrieval)

| Field | Value |
|---|---|
| Chart key | `inference.vllm_colpali` |
| Model | `TomoroAI/tomoro-colqwen3-embed-4b` |
| Image (CPU / k3s default) | **`cogniverse/colpali:dev` (CUSTOM, built from `deploy/colpali/`)** |
| Image (ROCm 7.12+) | `vllm/vllm-openai-rocm:v0.23.0` (official) |
| Engine flag | `colpali_native` (CPU) or `vllm_token_embed` (ROCm) |
| NodePort | 29001 |
| Default state | enabled |

**Why custom on CPU**: vLLM's `ColPaliForRetrieval` registers
`/v1/score` (cross-encoder scoring) instead of `/v1/embeddings`.
The runtime needs per-token multi-vector embeddings so we ship a
custom FastAPI sidecar (`deploy/colpali/server.py`) that wraps
colpali-engine directly.

**On ROCm 7.12+ with gfx1151**: the `vllm_token_embed` path serves
the multi-vector contract correctly, so the chart's ROCm overlay
overrides `engine: vllm_token_embed` and uses the official vLLM image.

### ColQwen (alternative multi-vector visual encoder)

| Field | Value |
|---|---|
| Chart key | `inference.vllm_colqwen` |
| Model | `TomoroAI/tomoro-colqwen3-embed-4b` |
| Image | `vllm/vllm-openai-cpu:latest` / `vllm/vllm-openai-rocm` |
| Default state | disabled |

Used by the `video_colqwen_omni_mv_chunk_30s` profile when enabled.

### VideoPrism (chunk-level video embeddings)

| Field | Value |
|---|---|
| Chart key | `inference.videoprism` |
| Model | `videoprism_public_v1_base_hf` |
| Image | **`cogniverse/videoprism:dev` (CUSTOM, built from `deploy/videoprism/`)** |
| Engine | `videoprism_jax` |
| Default state | disabled |

Custom JAX sidecar — no upstream vLLM equivalent. Used by the
`video_videoprism_*` family of profiles. Build with
`docker build -t cogniverse/videoprism:dev deploy/videoprism/`; see
[`deploy/videoprism/README.md`](../../deploy/videoprism/README.md) for the
endpoint, supported models, and the video-only scope.

---

## Text embeddings

### ColBERT (late-interaction, multi-vector text)

| Field | Value |
|---|---|
| Chart key | `inference.colbert_pylate` |
| Model | `lightonai/LateOn` (ColBERT-style, late-interaction) |
| Image | `vllm/vllm-openai-cpu` / `vllm/vllm-openai-rocm` (official) |
| Engine | `vllm_token_embed` |
| Endpoint | `POST /pooling` (per-token multi-vector) |
| NodePort | 29002 |
| Default state | enabled |

**Serving**: vLLM's pooling runner (`--runner pooling --convert embed`)
serves the per-token `/pooling` contract. The
`--hf-overrides '{"architectures": ["ColBERTModernBertModel"]}'` flag
forces the multi-vector architecture; without it vLLM serves a plain
dense ModernBert and the per-token outputs LateOn retrieval needs vanish.
Query vs document is distinguished client-side by a `[Q] `/`[D] ` prefix,
never an `is_query` field.

### Code search (ColBERT variant)

| Field | Value |
|---|---|
| Chart key | `inference.code_colbert_pylate` |
| Model | `lightonai/LateOn-Code-edge` |
| Image | `vllm/vllm-openai-cpu` / `vllm/vllm-openai-rocm` (official) |
| Engine | `vllm_token_embed` |
| Default state | disabled |

### DenseOn (single-vector dense text)

| Field | Value |
|---|---|
| Chart key | `inference.denseon` |
| Model | `lightonai/DenseOn` (ModernBERT-base, 768-dim, CLS pooling, 512 ctx) |
| Image | `vllm/vllm-openai-cpu` / `vllm/vllm-openai-rocm` (official) |
| Engine | `vllm_embed` |
| Endpoint | `POST /v1/embeddings` (OpenAI-compatible, single dense vector) |
| Default state | enabled |

DenseOn uses the same vLLM pooling runner as the ColBERT path, but
`vllm_embed` pools to a single dense vector per input (no per-token
reshape), matching DenseOn's dense-retrieval semantics.

---

## Audio (ASR)

### Whisper

| Field | Value |
|---|---|
| Chart key | `inference.vllm_asr` |
| Model | `openai/whisper-large-v3-turbo` |
| Image | `vllm/vllm-openai-cpu:latest` / `vllm/vllm-openai-rocm:v0.23.0` (official) |
| Engine | `vllm_transcription` |
| NodePort | 29005 |
| Default state | enabled |

vLLM's stock CPU and ROCm images don't ship the `[audio]` extras, so
the chart's pod template runs `pip install soundfile librosa` at
startup before exec-ing `vllm serve`. The endpoint is
`/v1/audio/transcriptions` (OpenAI-compatible multipart upload).

### CLAP acoustic embeddings (`clap_embed` sidecar)

| Field | Value |
|---|---|
| Chart key | `inference.clap_embed` |
| Model | `laion/clap-htsat-unfused` (~1.7 GiB) |
| Image | `cogniverse/clap-embed` (CUSTOM, `deploy/clap_embed/Dockerfile`, module `cogniverse_runtime/sidecars/clap_embed.py`) |
| Endpoints | `POST /embed/audio`, `POST /embed/text` (one joint space) |
| NodePort | 29008 |
| Default state | disabled |

When the sidecar is deployed, `AudioEmbeddingGenerator` routes acoustic
embeddings to it via `inference_service_urls["clap_embed"]` (ingestion
side injected by the embedding-generator factory; query side via
`AudioAnalysisDeps.clap_endpoint` filled by the dispatcher). Without it,
CLAP loads in-process — which requires torch and therefore only works in
dev environments, never in the deployed runtime image; in that case the
acoustic vector is skipped (best-effort) and audio chunks carry only
transcript + semantic embeddings.

The runtime pods also need `NUMBA_CACHE_DIR` writable (set by the
chart) or librosa's numba JIT crashes with "no locator available".

---

## Deployment style summary

| Service | Image source | Custom build? |
|---|---|---|
| `vllm_llm_student` | `vllm/vllm-openai-cpu` / `vllm/vllm-openai-rocm` | No (official) |
| `vllm_llm_teacher` | same as student | No |
| `vllm_colpali` (CPU/k3s) | `cogniverse/colpali:dev` | **Yes** (`deploy/colpali/`) |
| `vllm_colpali` (ROCm 7.12+) | `vllm/vllm-openai-rocm` | No |
| `vllm_colqwen` | `vllm/vllm-openai-cpu` / `vllm/vllm-openai-rocm` | No |
| `vllm_asr` | `vllm/vllm-openai-cpu` / `vllm/vllm-openai-rocm` | No |
| `colbert_pylate` | `vllm/vllm-openai-cpu` / `vllm/vllm-openai-rocm` | No (official) |
| `code_colbert_pylate` | `vllm/vllm-openai-cpu` / `vllm/vllm-openai-rocm` | No (official) |
| `denseon` | `vllm/vllm-openai-cpu` / `vllm/vllm-openai-rocm` | No (official) |
| `videoprism` | `cogniverse/videoprism:dev` | **Yes** (`deploy/videoprism/`) |
| `llm.builtin` (Ollama) | `ollama/ollama` | No (official) |

Custom images are built locally by `cogniverse up` (which calls
`build_images()` in `libs/cli/cogniverse_cli/images.py`) and imported
into the k3d cluster. They are NOT published to a public registry —
they're loaded from the host docker daemon into the cluster's
containerd via `k3d image import`.

---

## Device selection (`device:` per service)

Each `inference.<svc>` block has a `device:` key. Values:

| Value | Meaning |
|---|---|
| `cpu` | (default in `values.yaml`) — CPU-only execution. |
| `rocm` | AMD GPU via ROCm. Chart adds `amd.com/gpu: 1` resource limit, the `amd.com/gpu.present=true` nodeSelector, the `/dev/kfd` + `/dev/dri` hostPath mounts, and `supplementalGroups: [992, 44]` for the host's render and video group ids. See [ROCm GPU passthrough](./kubernetes-deployment.md#gpu-passthrough-rocm--cuda) for the device-mount specifics. |
| `cuda` | NVIDIA GPU. Chart adds `nvidia.com/gpu: 1` resource limit and the `nvidia.com/gpu.present=true` nodeSelector. |

`cogniverse up` auto-applies the right values overlay
(`values.rocm.yaml` or `values.cuda.yaml`) and the node label when
the host has the corresponding device — see
[scripts-operations.md](../development/scripts-operations.md).

For Strix Halo (`gfx1151`) iGPU specifically, the GPU "VRAM" IS host
RAM (unified memory). The chart's ROCm overlay tunes
`--gpu-memory-utilization` per-service so three concurrent vLLM pods
can coexist (see `values.rocm.yaml` comments).

### GEMM auto-tuning on ROCm (`runtime.tunableOp`)

`runtime.tunableOp` (`false` in `values.yaml`, `true` in
`values.rocm.yaml`) enables PyTorch TunableOp on every rocm-device
inference pod. On gfx1151 the default hipBLASLt kernel heuristic
mistunes many GEMM shapes; TunableOp benchmarks the candidate kernels
once per shape and reuses the fastest. The key only takes effect when a
pod's `device` is `rocm` — the `cogniverse.tunableOpEnv` helper gates on
both the device and the toggle, so CPU/CUDA pods are unaffected.

Each rocm pod gets `PYTORCH_TUNABLEOP_ENABLED=1`,
`PYTORCH_TUNABLEOP_TUNING=1`, and a per-service
`PYTORCH_TUNABLEOP_FILENAME=/root/.cache/huggingface/tunableop_<svc>_%d.csv`.
The results file lives in the persistent `model-cache` volume, so tuning
survives pod restarts and rollouts — a shape is benchmarked once over the
file's lifetime. The first request hitting a not-yet-tuned shape pays a
one-time tuning latency; the persisted file means later pods skip it.

---

## Per-tenant Vespa schemas (separate from inference)

The seven schemas you'll see in Vespa per tenant — `video_colpali_*`,
`audio_content_*`, `document_text_*`, `image_colpali_mv_*`,
`knowledge_graph_*`, `wiki_pages_*`, `agent_memories_*` — are NOT
served by the inference services. They're Vespa document schemas the
runtime feeds into directly. The inference services produce the
embeddings; Vespa stores and ranks them. See
[architecture/multi-tenant.md](../architecture/multi-tenant.md) for
the schema lifecycle.

---

## See also

- [`docs/operations/setup-installation.md`](./setup-installation.md) — local docker-style setup with port table
- [`docs/operations/kubernetes-deployment.md`](./kubernetes-deployment.md) — chart structure, GPU passthrough, manual `helm install`
- [`docs/architecture/overview.md`](../architecture/overview.md) — service graph
- [`charts/cogniverse/values.yaml`](../../charts/cogniverse/values.yaml) — canonical defaults
- [`charts/cogniverse/values.rocm.yaml`](../../charts/cogniverse/values.rocm.yaml) — ROCm overlay
- [`charts/cogniverse/values.cuda.yaml`](../../charts/cogniverse/values.cuda.yaml) — CUDA overlay
