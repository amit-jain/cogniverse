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
| **Text embeddings** | ColBERT-style late-interaction (PyLate) for documents/code, DenseOn (ModernBERT) for query/single-vector text. |
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
| Image (ROCm) | `vllm/vllm-openai-rocm:v0.20.0` (official) |
| NodePort | 29010 |
| Default state | enabled |
| ROCm GPU memory | `--gpu-memory-utilization 0.40` (≈25 GiB on 62 GiB unified memory) |

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
| Image (ROCm) | `vllm/vllm-openai-rocm:v0.20.0` (official) |
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

---

## Visual embeddings

### ColPali (per-tenant visual retrieval)

| Field | Value |
|---|---|
| Chart key | `inference.vllm_colpali` |
| Model | `vidore/colpali-v1.3-hf` |
| Image (CPU / k3s default) | **`cogniverse/colpali:dev` (CUSTOM, built from `deploy/colpali/`)** |
| Image (ROCm 7.12+) | `vllm/vllm-openai-rocm:v0.20.0` (official) |
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
| Model | `vidore/colqwen2-v1.0` |
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
| Image | **`cogniverse/pylate:dev` (CUSTOM, `deploy/pylate/Dockerfile`, module `cogniverse_runtime/sidecars/colbert_pylate.py`)** |
| Engine | `pylate` |
| NodePort | 29002 |
| Default state | enabled |

**Why custom**: PyLate (the late-interaction backbone the chart uses)
has no upstream containerized release. The cogniverse build ships the
right torch flavour for the host (CPU / ROCm / CUDA) per the
`TORCH_BACKEND` build-arg.

### Code search (ColBERT variant)

| Field | Value |
|---|---|
| Chart key | `inference.code_colbert_pylate` |
| Model | `lightonai/LateOn-Code-edge` |
| Image | `cogniverse/pylate:dev` (same custom image as above) |
| Default state | disabled |

### DenseOn (single-vector dense text)

| Field | Value |
|---|---|
| Chart key | `inference.denseon` |
| Model | `lightonai/DenseOn` (ModernBERT-base, 768-dim, CLS pooling, 512 ctx) |
| Image | `cogniverse/pylate:dev` (same image, `MODE=dense`) |
| Engine | `pylate` |
| Default state | enabled |

The same `cogniverse/pylate` image serves both the multi-vector
ColBERT path and the single-vector DenseOn path; the
`MODE` env var picks the head.

---

## Audio (ASR)

### Whisper

| Field | Value |
|---|---|
| Chart key | `inference.vllm_asr` |
| Model | `openai/whisper-large-v3-turbo` |
| Image | `vllm/vllm-openai-cpu:latest` / `vllm/vllm-openai-rocm:v0.20.0` (official) |
| Engine | `vllm_transcription` |
| NodePort | 29005 |
| Default state | enabled |

vLLM's stock CPU and ROCm images don't ship the `[audio]` extras, so
the chart's pod template runs `pip install soundfile librosa` at
startup before exec-ing `vllm serve`. The endpoint is
`/v1/audio/transcriptions` (OpenAI-compatible multipart upload).

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
| `colbert_pylate` | `cogniverse/pylate:dev` | **Yes** (`deploy/pylate/Dockerfile`) |
| `code_colbert_pylate` | `cogniverse/pylate:dev` | **Yes** (same image) |
| `denseon` | `cogniverse/pylate:dev` | **Yes** (same image, `MODE=dense`) |
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
