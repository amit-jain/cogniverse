# Models and Inference Deployment

Every served model, the container image that serves it, and how the
deployment differs between CPU, ROCm, and CUDA. The chart's
`values.yaml` (chart defaults), `values.k3s.yaml` (the local k3d dev
overlay `cogniverse up` applies by default), `values.rocm.yaml`,
`values.cuda.yaml`, and the `deploy/` sidecars are the underlying
source of truth; this page flattens them into one reference.

---

## Overview

Cogniverse runs five classes of inference services:

| Class | Purpose |
|---|---|
| **LLMs** (chat / generation) | Agent reasoning, query enhancement, distillation. Two tiers: a small **student** model used at runtime, and a larger **teacher** model used only during DSPy optimization (`BootstrapFewShot`'s `teacher_settings`). |
| **Visual / multimodal embeddings** | ColPali/ColQwen (one pod, one model, shared by both retrieval families) for video/image patch embeddings, VideoPrism for chunk embeddings. |
| **Text embeddings** | ColBERT-style late-interaction (LateOn, served by the PyLate sidecar) for documents/code, DenseOn (ModernBERT, served by vLLM) for query/single-vector text. |
| **Audio (ASR + acoustic embeddings)** | Whisper transcription of audio files; CLAP for a joint audio/text acoustic embedding space. |
| **Vision/NLP sidecars** | GLiNER zero-shot entity extraction (gateway routing + entity extraction agents), InsightFace face embeddings (knowledge-graph face clustering). |

Each service lives in `charts/cogniverse/values.yaml` under the
`inference:` block (or under top-level `llm:` for the LLM serving
path). The chart renders one Deployment + Service per enabled service.

---

## LLM serving

### Student (vLLM chat / generation LLM)

| Field | Value |
|---|---|
| Chart key | `inference.vllm_llm_student` |
| Model | `google/gemma-4-e4b-it` |
| Image (CPU / CUDA) | `vllm/vllm-openai-cpu:v0.23.0` / `vllm/vllm-openai:v0.23.0` (official) |
| Image (ROCm) | `vllm/vllm-openai-rocm:v0.23.0` (official) |
| NodePort | 29010 |
| Default state | **`enabled: false`** in base `values.yaml` — turned on by the ROCm overlay (`values.rocm.yaml` sets `enabled: true`, `llm.engine: vllm`, `llm.builtin.enabled: false`) |
| ROCm GPU memory | `--gpu-memory-utilization 0.30` (≈19 GiB on 62 GiB unified memory) |

The **base default LLM engine is Ollama**, not the vLLM student (see
[Optional: Ollama instead of vLLM](#optional-ollama-instead-of-vllm) below —
despite the name, Ollama is what a fresh CPU install and the local k3d dev
overlay (`values.k3s.yaml`) actually run). The vLLM student pod is the
production path used once a ROCm (or manually configured CUDA) GPU is
available: when enabled, it becomes the primary chat LLM used by every agent
for DSPy/litellm calls. The Helm template (`cogniverse.primaryLLMModel` in
`templates/_helpers.tpl`) always prepends the `openai/` provider prefix
and writes the resulting model id verbatim into `config.json`;
`create_dspy_lm()` passes it through unchanged. The actual destination
is determined by `api_base`, not the prefix.

### Teacher (DSPy optimization only)

| Field | Value |
|---|---|
| Chart key | `inference.vllm_llm_teacher` |
| Model | `cyankiwi/Qwen3.6-27B-AWQ-INT4` (AWQ-INT4, ~14 GiB) |
| Image (CPU) | `vllm/vllm-openai-cpu:v0.23.0` (official) |
| Image (ROCm) | `vllm/vllm-openai-rocm:v0.23.0` (official) |
| NodePort | 29011 |
| Default state | **`enabled: false`, `replicaCount: 0`** — scale-to-zero |
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

The agent code wires it via `LLMConfig.resolve_teacher()` → `create_dspy_lm()` →
`teacher_settings` on `BootstrapFewShot` in
`libs/agents/cogniverse_agents/optimizer/dspy_agent_optimizer.py`, driven
by `cogniverse_runtime.optimization_cli`. See the
[ColPali/ColQwen embedding flow](#visual-embeddings) for the runtime
side that consumes the optimized program.

### Optional: Ollama instead of vLLM

| Field | Value |
|---|---|
| Chart key | `llm.ollama` (active when `llm.engine: ollama`) |
| Model | `gemma3:4b` (configurable via `llm.model`) |
| Image | `ollama/ollama:0.20.5` (official) |
| Deployment style | StatefulSet + PVC for the model cache |
| Default state | **the base default** — `values.yaml` and the local k3d dev overlay (`values.k3s.yaml`) both ship `llm.engine: ollama`, `llm.builtin.enabled: true` (implicit default) |

Ollama is what a fresh CPU install and local development actually run;
"instead of vLLM" here means instead of the `vllm_llm_student` pod described
above, which is disabled until the ROCm (or a manually configured CUDA)
overlay turns it on. Use Ollama on machines without a vLLM-ready GPU.
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
rewritten to the router, `model` replaced by the router's auto alias, and the
two authz headers merged onto `extra_headers`:

| Field | Meaning |
|---|---|
| `enabled` | Master switch. `False` ⇒ endpoint passes through untouched. |
| `semantic_router_url` | The router's OpenAI-compatible endpoint. Enabled with an empty value raises. |
| `tenant_tiers` | `tenant_id → tier` map; unknown tenants fall back to `default_tier`. |
| `default_tier` | Tier for tenants not in `tenant_tiers`. |
| `tier_header` / `user_id_header` | Header names for the tier / identity (default `x-authz-user-groups` / `x-authz-user-id`). |
| `routed_model` | Model sent on routed requests (default `openai/auto`). The router resolves models by its own catalog names / auto alias and rejects raw provider model ids with a 400, so the endpoint's model is replaced, not forwarded. |

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
its classifier bundle on first boot into a model-cache PVC. Its startup probe
allows 30 minutes for that cold download because model pulls can be serialized
on a shared development host; warm starts reuse the PVC. Coverage:
`tests/foundation/integration/test_semantic_router_e2e.py` self-launches the
real router+Envoy+stub via `docker run` and asserts the tier/content decisions;
`tests/e2e/deployment/test_semantic_router_deploy_e2e.py` rides the
`deployed_stack` fixture (its own isolated k3d cluster running the full chart)
and asserts the routing *decision* per tenant tier + content against the
deployed router's `llm_decision_match_total` metric; `tests/charts/test_semantic_router_chart.py`
pins the rendered upstream endpoint and served model per `llm.engine`.

---

## Visual embeddings

### ColPali / ColQwen (visual multi-vector retrieval — one pod)

There is a single chart entry, `inference.vllm_colpali`, that serves both the
ColPali and ColQwen retrieval families — both route to the same
`TomoroAI/tomoro-colqwen3-embed-4b` (ColQwen3, 320-dim per-patch multi-vector)
checkpoint. The `video_colpali_smol500_mv_frame`, `image_colpali_mv`, and
`video_colqwen_omni_mv_chunk_30s` profiles all set
`inference_service: "vllm_colpali"` — there is no separate `vllm_colqwen`
chart key.

| Field | Value |
|---|---|
| Chart key | `inference.vllm_colpali` |
| Model | `TomoroAI/tomoro-colqwen3-embed-4b` |
| Image (base `values.yaml`, no device overlay) | `vllm/vllm-openai-cpu:v0.23.0` (official), `engine: vllm_token_embed`, `enabled: false` |
| Image (k3d local dev, `values.k3s.yaml`) | No override: inherits the disabled official CPU vLLM definition from `values.yaml` |
| Image (ROCm 7.12+) | `vllm/vllm-openai-rocm:v0.23.0` (official), `engine: vllm_token_embed` |
| Image (CUDA) | `vllm/vllm-openai:v0.23.0` (official), `engine: vllm_token_embed`, `replicaCount: 3` |
| NodePort | 29001 |
| Default state | Disabled in base and k3d-only composition; the ROCm and CUDA overlays enable it |

The k3s overlay changes neither this service's image nor its enabled state, so
a CPU-only `cogniverse up` does not allocate a ColPali/ColQwen pod. On ROCm
7.12+ with gfx1151 and on CUDA, the device overlay enables the service and the
official vLLM pooling runner serves the `vllm_token_embed` multi-vector
contract.

### VideoPrism (chunk-level video embeddings)

| Field | Value |
|---|---|
| Chart key | `inference.videoprism_jax` |
| Model | `videoprism_public_v1_base_hf` |
| Image | **`cogniverse/videoprism:0.1.0-dev` (CUSTOM, `deploy/videoprism/Dockerfile`)** |
| Engine | `videoprism_jax` |
| NodePort | 29003 |
| Default state | disabled |

Custom JAX sidecar — no upstream vLLM equivalent. Used by the
`video_videoprism_*` family of profiles. Build with
`docker build -f deploy/videoprism/Dockerfile -t cogniverse/videoprism:0.1.0-dev .`; see
[`deploy/videoprism/README.md`](../../deploy/videoprism/README.md) for the
endpoint, supported models, and the video-only scope.

---

## Text embeddings

### ColBERT (late-interaction, multi-vector text)

| Field | Value |
|---|---|
| Chart key | `inference.colbert_pylate` |
| Model | `lightonai/LateOn` (ColBERT-style, late-interaction), pinned revision `c01907b7…` |
| Image | **`cogniverse/pylate` (CUSTOM, `deploy/pylate/Dockerfile`)** |
| Engine | `pylate` |
| Endpoint | `POST /pooling` (per-token multi-vector, `{"input", "model", "is_query"}`) |
| NodePort | 29002 |
| Default state | enabled |

**Serving**: the PyLate sidecar
(`libs/cli/cogniverse_cli/modal_inference/servers/pylate.py`) runs
`pylate.models.ColBERT` itself and owns both encode directions: queries
get PyLate's exact expansion — mask-token padding excluded from attention
— and documents get the marker plus punctuation skiplist. Stock vLLM
cannot serve LateOn exactly: its `/pooling` request accepts text or token
IDs but no attention mask, so it attends to all query positions and every
per-token vector drifts from the trained model's output. Clients send raw
text plus `is_query`; no prefixes or token IDs are built client-side.

### Code search (ColBERT variant)

| Field | Value |
|---|---|
| Chart key | `inference.code_colbert_pylate` |
| Model | `lightonai/LateOn-Code-edge` (48-dim), pinned revision `07ef20f4…` |
| Image | **`cogniverse/pylate` (CUSTOM, `deploy/pylate/Dockerfile`)** |
| Engine | `pylate` |
| Default state | disabled |

### DenseOn (single-vector dense text)

| Field | Value |
|---|---|
| Chart key | `inference.denseon` |
| Model | `lightonai/DenseOn` (ModernBERT-base, 768-dim, CLS pooling, 512 ctx) |
| Image | `vllm/vllm-openai-cpu` / `vllm/vllm-openai-rocm` (official) |
| Engine | `vllm_embed` |
| Endpoint | `POST /v1/embeddings` (OpenAI-compatible, single dense vector) |
| NodePort | 29006 |
| Default state | enabled |

DenseOn uses the same vLLM pooling runner as the ColBERT path, but
`vllm_embed` pools to a single dense vector per input (no per-token
reshape), matching DenseOn's dense-retrieval semantics.

---

## Audio (ASR + acoustic embeddings)

### Whisper

| Field | Value |
|---|---|
| Chart key | `inference.vllm_asr` |
| Model | `openai/whisper-large-v3-turbo` |
| Image | `vllm/vllm-openai-cpu:v0.23.0` / `vllm/vllm-openai-rocm:v0.23.0` (official) |
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
| Revision | `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a` |
| Image | `cogniverse/clap-embed` (CUSTOM, `deploy/clap_embed/Dockerfile`, module `cogniverse_cli.modal_inference.servers.clap`) |
| Endpoints | `POST /embed/audio`, `POST /embed/text` (one joint space) |
| Health | `GET /health` loads the pinned model, then returns `{"status": "ready", "model": "laion/clap-htsat-unfused", "model_revision": "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"}` |
| Kubernetes probes | HTTP `GET /health` readiness on port 8000; TCP liveness on port 8000 |
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
If the pinned CLAP model cannot load, `/health` returns HTTP 503 and its
`detail` matches
`clap_embed: model laion/clap-htsat-unfused load failed (<ExceptionType>): <cause>`.
Readiness therefore keeps the pod out of the
Service while the TCP liveness probe leaves the running process available for
diagnosis and recovery.

---

## Vision/NLP sidecars

These two sidecars keep heavy ML dependencies (torch + gliner,
torch + insightface) out of the slim runtime image. Both are FastAPI
services with a model-backed `/health` readiness and model-identity endpoint
plus one predict route. Their liveness probes check only the TCP serving
socket, so a model load fault cannot create a Kubernetes restart loop.

### GLiNER (zero-shot entity extraction)

| Field | Value |
|---|---|
| Chart key | `inference.gliner` |
| Model | `urchade/gliner_large-v2.1` at revision `abd49a1f1ebc12af1be84d06f6848221cf96dcad` (pinned; other request model IDs are rejected) |
| Image | `cogniverse/gliner` (CUSTOM, `deploy/gliner/Dockerfile` + `cogniverse_cli.modal_inference.servers.gliner`) |
| Endpoint | `POST /predict_entities` (mirrors the in-process `model.predict_entities(text, labels, threshold)` shape) |
| Health | `GET /health` loads the pinned model, then returns `{"status": "ready", "default_model": "urchade/gliner_large-v2.1", "model_revision": "abd49a1f1ebc12af1be84d06f6848221cf96dcad", "loaded_models": ["urchade/gliner_large-v2.1"]}` |
| Kubernetes probes | HTTP `GET /health` readiness on port 8080; TCP liveness on port 8080 |
| NodePort | 29007 |
| Default state | enabled |

`GatewayAgent` classifies queries by modality/generation-type using GLiNER
zero-shot NER (`urchade/gliner_large-v2.1`), and `EntityExtractionAgent` /
the knowledge-graph doc extractor use the same sidecar for entity extraction.
The sidecar accepts only the canonical large model and its immutable revision;
the optional request `model` field must match that identifier exactly.
`RemoteGlinerClient` is the client-side loader that replaces the in-process
GLiNER loader transparently. A pinned-model load failure returns HTTP 503; its
`detail` matches
`gliner: model urchade/gliner_large-v2.1 load failed (<ExceptionType>): <cause>`.

### InsightFace (face embeddings, `face_embed` sidecar)

| Field | Value |
|---|---|
| Chart key | `inference.face_embed` |
| Model | `buffalo_l` (InsightFace bundle: RetinaFace detector + ArcFace `w600k_r50` recognizer, 512-dim) |
| Artifact | [InsightFace v0.7 `buffalo_l.zip`](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip), SHA-256 `80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f` |
| Image | `cogniverse/face-embed` (CUSTOM, `deploy/face_embed/Dockerfile`, module `cogniverse_cli.modal_inference.servers.face`) |
| Endpoint | `POST /embed` (`{"image_url": ...}` or `{"image_b64": ...}` → `{"faces": [{bbox, vec, det_score}], "n": int}`); `det_score` is the RetinaFace confidence in `[0, 1]` |
| Health | `GET /health` loads the pinned model, then returns `{"status": "ready", "model": "buffalo_l", "model_revision": "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"}` |
| Kubernetes probes | HTTP `GET /health` readiness on port 8000; TCP liveness on port 8000 |
| NodePort | 29009 |
| Default state | disabled |

The Modal image downloads that official archive during its image build,
verifies the digest before unpacking, and installs the required ONNX files at
`/opt/insightface/models/buffalo_l`. Runtime initialization passes that root
to InsightFace and fails if any required file is missing; it never downloads
model data on the first inference request. A load failure returns HTTP 503; its
`detail` matches
`face_embed: model buffalo_l load failed (<ExceptionType>): <cause>`.

The service is stateless — no persistence. The knowledge-graph face pipeline
(`libs/agents/cogniverse_agents/graph/face_extractor.py`) POSTs one keyframe
per request and clusters the returned 512-dim L2-normalized ArcFace vectors
per `source_doc_id` to discover anonymous identity groups, attributing each
cluster to a temporally-aligned named person where possible. It's wired the
same way as CLAP: opt-in via `inference_service_urls["face_embed"]`
(`_lookup_face_embed_endpoint` in `libs/runtime/cogniverse_runtime/routers/ingestion.py`),
additive — an unreachable sidecar degrades the pipeline rather than failing
the whole ingest.

---

## Deployment style summary

| Service | Image source | Custom build? |
|---|---|---|
| `vllm_llm_student` | `vllm/vllm-openai-cpu` / `vllm/vllm-openai` / `vllm/vllm-openai-rocm` | No (official) |
| `vllm_llm_teacher` | same as student | No |
| `vllm_colpali` (base and CPU/k3d dev composition; disabled) | `vllm/vllm-openai-cpu` | No |
| `vllm_colpali` (ROCm 7.12+ / CUDA) | `vllm/vllm-openai-rocm` / `vllm/vllm-openai` | No |
| `vllm_asr` | `vllm/vllm-openai-cpu` / `vllm/vllm-openai-rocm` | No |
| `colbert_pylate` | `cogniverse/pylate` | **Yes** (`deploy/pylate/Dockerfile`) |
| `code_colbert_pylate` | `cogniverse/pylate` (shared image, built once) | **Yes** (`deploy/pylate/Dockerfile`) |
| `denseon` | `vllm/vllm-openai-cpu` / `vllm/vllm-openai-rocm` | No (official) |
| `videoprism_jax` | `cogniverse/videoprism:0.1.0-dev` | **Yes** (`deploy/videoprism/Dockerfile`) |
| `clap_embed` | `cogniverse/clap-embed` | **Yes** (`deploy/clap_embed/`) |
| `gliner` | `cogniverse/gliner` | **Yes** (`deploy/gliner/Dockerfile`) |
| `face_embed` | `cogniverse/face-embed` | **Yes** (`deploy/face_embed/`) |
| `llm.builtin` (Ollama) | `ollama/ollama` | No (official) |

Custom images are built locally by `cogniverse up` (which calls
`build_images()` in `libs/cli/cogniverse_cli/images.py`) and imported
into the k3d cluster. They are NOT published to a public registry —
they're loaded from the host docker daemon into the cluster's
containerd via `k3d image import`.

---

## Device selection (`device:` per service)

Each vLLM-backed `inference.<svc>` block has a `device:` key (the FastAPI
sidecars that are always CPU-only — `clap_embed`, `face_embed` — omit it
entirely; `gliner` sets `device: cpu` for documentation even though nothing
reads it for GPU scheduling). Values:

| Value | Meaning |
|---|---|
| `cpu` | (default in `values.yaml`) — CPU-only execution. |
| `rocm` | AMD GPU via ROCm. Chart adds `amd.com/gpu: 1` resource limit, the `amd.com/gpu.present=true` nodeSelector, the `/dev/kfd` + `/dev/dri` hostPath mounts, and `supplementalGroups: [992, 44]` for the host's render and video group ids. See [ROCm GPU passthrough](./kubernetes-deployment.md#gpu-passthrough-rocm-cuda) for the device-mount specifics. |
| `cuda` | NVIDIA GPU. Chart adds `nvidia.com/gpu: 1` resource limit and the `nvidia.com/gpu.present=true` nodeSelector. |

`cogniverse up` auto-applies the right values overlay
(`values.rocm.yaml` or `values.cuda.yaml`) and the node label when
the host has the corresponding device — see
[scripts-operations.md](../development/scripts-operations.md).

For Strix Halo (`gfx1151`) iGPU specifically, the GPU "VRAM" IS host
RAM (unified memory). The chart's ROCm overlay gives generation services
bounded `--gpu-memory-utilization` fractions. The Tomoro pooling service uses
an explicit 1 GiB KV cache because its transient image-profile allocation is
larger than its steady-state cache budget; it also caps preprocessing at
1,048,576 pixels, matching the 1,024-patch Vespa document contract. See
`values.rocm.yaml` for the complete per-service allocation. Its accompanying
0.45 utilization value is retained only for vLLM's initial free-memory guard;
the explicit byte value controls the actual cache reservation.

Inference readiness probes begin immediately. A failed readiness probe only
keeps the Service endpoint out of rotation; it does not restart the container.
Cold-start protection remains on the separately budgeted liveness probe, so a
model becomes routable on the first successful `/health` response instead of
waiting through an additional fixed delay.

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
- [`charts/cogniverse/values.k3s.yaml`](../../charts/cogniverse/values.k3s.yaml) — local k3d dev overlay (`cogniverse up`)
- [`charts/cogniverse/values.rocm.yaml`](../../charts/cogniverse/values.rocm.yaml) — ROCm overlay
- [`charts/cogniverse/values.cuda.yaml`](../../charts/cogniverse/values.cuda.yaml) — CUDA overlay
