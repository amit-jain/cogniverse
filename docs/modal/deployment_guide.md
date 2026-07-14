# Modal Deployment Guide - Multi-Agent System

Complete guide for deploying LLMs and VLMs on Modal for the Cogniverse multi-agent video search system.

## Overview

Modal provides serverless GPU infrastructure for:

- **DSPy Optimization**: Teacher/student models via `BootstrapFewShot`'s `teacher_settings`

- **Video Processing**: VLMs for frame description generation during ingestion

- **Model Fine-tuning**: SFT and DPO training on cloud GPUs

- **Agent LLMs**: Deploy agent models on GPUs instead of local Ollama

- **LLM/VLM-as-Judge Evaluation**: Run experiment evaluators against a Modal-hosted vision model

## Use Cases in Cogniverse

### 1. DSPy Optimization (Teacher/Student)

The optimizer package uses teacher/student patterns with DSPy optimizers:

```python
from cogniverse_agents.optimizer.dspy_agent_optimizer import (
    DSPyAgentPromptOptimizer,
    DSPyAgentOptimizerPipeline,
)
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

# Student model on Modal GPU endpoint
endpoint_config = LLMEndpointConfig(
    model="openai/HuggingFaceTB/SmolLM3-3B",
    api_base="https://username--general-inference-service-serve.modal.run",
)

# Teacher model — drives BootstrapFewShot's demo generation
teacher_endpoint_config = LLMEndpointConfig(
    model="anthropic/claude-3-5-sonnet-20241022",
    api_key="sk-ant-...",
)

optimizer = DSPyAgentPromptOptimizer(config={
    "optimization": {"max_bootstrapped_demos": 8, "max_labeled_demos": 16}
})
optimizer.initialize_language_model(
    endpoint_config=endpoint_config,
    teacher_endpoint_config=teacher_endpoint_config,
)

# Run optimization pipeline (teacher: Claude/GPT-4, student: Modal endpoint)
pipeline = DSPyAgentOptimizerPipeline(optimizer)
await pipeline.optimize_all_modules()

# Optimized prompts saved via ArtifactManager → telemetry DatasetStore
await pipeline.save_optimized_prompts(
    tenant_id="production",
    telemetry_provider=telemetry_provider,
)
```

### 2. VLM for Video Processing

The VLM service uses Qwen3-VL-8B for video frame descriptions:

```python
# Actual location: scripts/modal_vlm_service.py
# Deploy with: modal deploy scripts/modal_vlm_service.py

# Call via HTTP endpoint after deployment
import requests

response = requests.post(
    "https://your-username--cogniverse-vlm-vlmmodel-generate-description.modal.run",
    json={
        "frame_base64": frame_base64_data,
        "prompt": "Describe this video frame in detail"
    }
)
description = response.json()["description"]
```

### 3. Embedding Model Training

Embedding fine-tuning on Modal GPUs via `train_embedding_remote()` in `modal_app.py`:

```python
from cogniverse_finetuning.training.modal_runner import ModalTrainingRunner, ModalJobConfig

config = ModalJobConfig(gpu="A10G", timeout=3600)
runner = ModalTrainingRunner(config)

# Train embedding model with triplet loss on Modal GPU
result = await runner.run_embedding(
    dataset=triplet_data,          # List of {anchor, positive, negative} dicts
    base_model="jinaai/jina-embeddings-v3",
    output_dir="./adapters",
    embedding_config={"learning_rate": 2e-5, "epochs": 3, "triplet_margin": 0.5},
)
# result.adapter_path contains the trained adapter
```

Deploy:
```bash
modal deploy libs/finetuning/cogniverse_finetuning/training/modal_app.py
```

### 4. LLM/VLM-as-Judge Evaluation

Experiment evaluators can score results with a vision-capable judge model hosted on Modal
instead of a local LM. `LLMJudgeCore` (`libs/evaluation/cogniverse_evaluation/evaluators/llm_judge.py`)
is provider-agnostic — it POSTs to any OpenAI-compatible `/v1/chat/completions` endpoint —
so pointing it at a Modal deployment only requires a config entry:

```json
{
  "evaluators": {
    "modal_visual_judge": {
      "provider": "modal",
      "model": "qwen2-vl",
      "base_url": "https://your-modal-endpoint.modal.run/",
      "api_key": "your-modal-api-key"
    }
  }
}
```

Select it when running experiments:
```bash
uv run python scripts/run_experiments_with_visualization.py \
    --dataset-path data/testset/evaluation/video_search_queries.csv \
    --dataset-name golden_eval_v1 \
    --profiles frame_based_colpali \
    --evaluator modal_visual_judge
```

## Modal Deployments

### LLM Inference Service

A separate Modal LLM inference service can provide OpenAI-compatible endpoints for DSPy
student models. Deploy a custom vLLM-based Modal app for this purpose (not included in
the repository). Once deployed, configure the endpoint in `llm_config.primary.api_base`.

Typical app characteristics:
- App name: `general-inference-service`
- Endpoints: serve (vLLM OpenAI-compatible), `/health`, `/models`
- Configurable model (e.g., `HuggingFaceTB/SmolLM3-3B`) and GPU via env vars

### DSPy with Modal (via create_dspy_lm)

Use the centralized LLM config to run DSPy optimization with Modal:

```python
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.config.llm_factory import create_dspy_lm

# Create LLM endpoint config pointing to your Modal deployment
endpoint = LLMEndpointConfig(
    model="openai/HuggingFaceTB/SmolLM3-3B",  # Modal serves OpenAI-compatible API
    api_base="https://username--general-inference-service-serve.modal.run",
)

# Create DSPy LM via the factory
lm = create_dspy_lm(endpoint)

# Use with DSPy modules via scoped context
import dspy
with dspy.context(lm=lm):
    result = module(query="...")
```

### VLM Service Deployment

- Actual location: `scripts/modal_vlm_service.py`
- App name: `cogniverse-vlm`
- Uses: SGLang + Qwen3-VL-8B-Instruct
- GPU: `h100` by default (configurable via `GPU_TYPE` env var)

The `VLMModel` class (on `@app.cls`) exposes two `@modal.fastapi_endpoint` methods, each
its own web URL of the form `https://<user>--cogniverse-vlm-vlmmodel-<method-name>.modal.run`
(underscores in the method name become dashes in the URL):

- `generate_description` (`POST .../cogniverse-vlm-vlmmodel-generate-description.modal.run`):
  single frame description (accepts `frame_base64`, `frame_path`, or `remote_frame_path`)
- `upload_and_process_frames` (`POST .../cogniverse-vlm-vlmmodel-upload-and-process-frames.modal.run`):
  batch processing — upload a zip of frames, get back a description per frame

There is also a standalone `upload_app` ASGI function (`@app.function` + `@modal.asgi_app()`)
that extracts an uploaded zip into the shared `cogniverse-frames` volume for later processing.
It is not currently invoked by the ingestion pipeline — `VLMDescriptor` calls
`upload_and_process_frames` directly instead.

`VLMProcessor`/`VLMDescriptor` (`libs/runtime/cogniverse_runtime/ingestion/processors/`)
default `auto_start=True`: if the configured `vlm_endpoint` isn't reachable, the ingestion
pipeline runs `modal deploy scripts/modal_vlm_service.py` for you before the first batch.

Deploy:
```bash
modal deploy scripts/modal_vlm_service.py
```

### Fine-tuning Service

Train models on Modal GPUs with the finetuning module:

- Actual location: `libs/finetuning/cogniverse_finetuning/training/modal_app.py`
- App name: `cogniverse-finetuning`

The Modal app (`modal_app.py`) provides three GPU functions:
- `train_sft_remote` — SFT (Supervised Fine-Tuning)
- `train_dpo_remote` — DPO (Direct Preference Optimization)
- `train_embedding_remote` — triplet-loss embedding training
- Configurable GPU selection (T4, A10G, A100-40GB, A100-80GB, H100)
- Direct dataset passing (no upload needed) and adapter download as bytes

Deploy:
```bash
modal deploy libs/finetuning/cogniverse_finetuning/training/modal_app.py
```

`ModalTrainingRunner` (`libs/finetuning/cogniverse_finetuning/training/modal_runner.py`) wraps
those three functions and is the low-level client shown below. The finetuning module also
provides a higher-level `RemoteTrainingBackend` (`libs/finetuning/cogniverse_finetuning/training/backend.py`)
that wraps `ModalTrainingRunner` behind the same `TrainingBackend` interface used by
`LocalTrainingBackend`, so orchestration code can switch between local and Modal training by
changing one config field. See [High-Level Orchestration](#high-level-orchestration) below.

Usage:
```python
from cogniverse_finetuning.training.modal_runner import ModalTrainingRunner, ModalJobConfig

config = ModalJobConfig(gpu="A10G", timeout=3600)
runner = ModalTrainingRunner(config)

# SFT (Supervised Fine-Tuning)
result = await runner.run_sft(
    dataset=training_data,
    base_model="HuggingFaceTB/SmolLM3-3B",  # Or any HuggingFace model
    output_dir="./adapters",
    sft_config={"learning_rate": 2e-5}
)

# DPO (Direct Preference Optimization) also available
result = await runner.run_dpo(dataset, base_model, output_dir, dpo_config)
```

### High-Level Orchestration

Rather than calling `ModalTrainingRunner` directly, most callers use the module-level
`finetune()` convenience function (`libs/finetuning/cogniverse_finetuning/orchestrator.py`),
which extracts a dataset from telemetry, auto-selects SFT/DPO/embedding, generates synthetic
data if needed, and trains via whichever backend you pick — `backend="remote"` with
`backend_provider="modal"` routes through `RemoteTrainingBackend` → `ModalTrainingRunner`
under the hood:

```python
from cogniverse_finetuning.orchestrator import finetune

result = await finetune(
    telemetry_provider=provider,
    tenant_id="tenant1",
    project="cogniverse-tenant1",
    model_type="llm",
    agent_type="routing",
    backend="remote",
    backend_provider="modal",
    gpu="A10G",  # T4, A10G, A100-40GB, A100-80GB, H100
)
# result.adapter_path / result.metrics — same OrchestrationResult shape as backend="local"
```

### Embedding Model Training on Modal

Embedding fine-tuning runs on Modal GPUs using triplet loss (sentence-transformers). The `train_embedding_remote()` Modal function in `modal_app.py` supports:

- Contrastive learning with triplet datasets (anchor/positive/negative)
- Configurable distance metrics (cosine, euclidean) and margins
- Automatic adapter serialization and transfer back to local

## Integration with Cogniverse

### Configuration

Update your configuration file with your Modal endpoints. LLM configuration is centralized in `llm_config` (see configs/config.json):

```json
{
  "llm_config": {
    "primary": {
      "model": "openai/HuggingFaceTB/SmolLM3-3B",
      "api_base": "https://username--general-inference-service-serve.modal.run"
    },
    "teacher": {
      "model": "anthropic/claude-3-5-sonnet-20241022",
      "api_key": "sk-ant-..."
    }
  },
  "vlm_endpoint_url": "https://username--cogniverse-vlm-vlmmodel-generate-description.modal.run/"
}
```

The teacher/student optimization workflow:
1. **Teacher** (Claude/GPT-4) generates high-quality bootstrap demonstrations via `BootstrapFewShot`'s `teacher_settings`
2. **Student** (SmolLM3-3B on Modal) is the model the compiled module actually runs on in production
3. Optimized prompts are saved via `ArtifactManager` → telemetry `DatasetStore`

### Using Modal Services

#### For Video Ingestion

The VLM service can be used for frame descriptions during ingestion:

```bash
# Deploy the VLM service first
modal deploy scripts/modal_vlm_service.py

# Run ingestion with the standard pipeline
# The Modal VLM endpoint can be configured via vlm_endpoint_url in config.json
uv run python scripts/run_ingestion.py \
    --video_dir path/to/your/videos \
    --backend vespa \
    --profile video_colpali_smol500_mv_frame
```

> **Note**: Embedding *fine-tuning* runs on Modal GPUs (see `ModalTrainingRunner.run_embedding()`). Embedding *inference* during ingestion runs locally.

## Performance & Cost

### DSPy Optimization
```text
Teacher (Claude/GPT-4):
- Cost: ~$0.01-0.10 per example
- Total: ~$5-50 for full optimization

Student (Modal SmolLM3-3B):
- GPU: A10G (24GB) - chosen when you deploy your custom vLLM inference
  app (see "LLM Inference Service" above); not an in-repo config key
- Duration: 15-30 minutes
- Cost: ~$1.10/hour = ~$0.55
```

### Video Processing
```text
VLM (Qwen3-VL-8B):
- GPU: H100 by default (configurable via GPU_TYPE env var)
- Speed: 2-4 seconds per frame
- Cost: ~$0.01-0.05 per frame

Embeddings (ColPali):
- GPU: A10G
- Speed: 100ms per frame
- Cost: ~$0.001 per frame
```

### Production Inference
```text
Agents with Modal LLMs:
- GPU: T4 (16GB) minimum
- Scaling: 1-10 instances auto-scale
- Cost: $0.60-1.10/hour when active
- Latency: 50-200ms per query
```

## Deployment Commands

### Full System Deployment

```bash
# 1. Deploy available Modal services
modal deploy scripts/modal_vlm_service.py
modal deploy libs/finetuning/cogniverse_finetuning/training/modal_app.py

# 2. Get endpoints
modal app list

# 3. Update your configuration file with endpoints
# Edit config.json: set llm_config.primary.api_base to your Modal endpoint

# 4. Test integration
uv run python -c "
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.config.llm_factory import create_dspy_lm
lm = create_dspy_lm(LLMEndpointConfig(model='openai/test', api_base='YOUR_ENDPOINT'))
print('LM created:', type(lm).__name__)
"
```

### Development Workflow

```bash
# 1. Get your Modal endpoint (after deploying a vLLM inference service)
modal app list  # Note the general-inference-service endpoint

# 2. Update your configuration with the Modal endpoint
# Set llm_config.primary.api_base in config.json to your deployed endpoint

# 3. Run DSPy optimization with teacher/student
python -m cogniverse_runtime.optimization_cli --mode workflow --tenant-id default

# 4. Run video ingestion with Modal VLM
modal deploy scripts/modal_vlm_service.py
uv run python scripts/run_ingestion.py \
    --video_dir path/to/your/videos \
    --backend vespa \
    --profile video_colpali_smol500_mv_frame
```

## Hybrid Configurations

### Current Recommended Setup
```yaml
# What works today:
VLM: Modal (Qwen3-VL-8B via scripts/modal_vlm_service.py)
LLM Inference: Modal (vLLM on a separate Modal inference service)
Embeddings: Local (ColPali, VideoPrism computed locally)
DSPy Student: Modal (SmolLM3-3B on general-inference-service)
DSPy Teacher: API (Claude/GPT-4)
```

### Full Local (Development)
```yaml
# For local development without Modal:
VLM: Local (Ollama LLaVA)
LLM: Local (Ollama)
Embeddings: Local
DSPy: Local only
```

### Cost-Optimized
```yaml
# Use Modal only where needed:
VLM: Modal (only for batch processing)
LLM: Local (Ollama)
Embeddings: Local
DSPy: Teacher API (Claude/GPT-4), Student Modal (SmolLM3-3B)
```

## Benefits vs Local

### For DSPy Optimization
- **Parallel Training**: Run multiple optimizers simultaneously
- **No Local GPU Required**: Train student models without local hardware
- **Experiment Tracking**: Modal provides built-in monitoring

### For Video Processing
- **10x Faster**: GPU acceleration for VLM and embeddings
- **Scalable**: Process multiple videos concurrently
- **Cost-Effective**: Pay only for processing time

### For Production
- **Auto-scaling**: Handle traffic spikes automatically
- **High Availability**: No single point of failure
- **Global Deployment**: Deploy close to users

## Testing

### Test Modal Services
```bash
# Test VLM service
curl -X POST https://your-username--cogniverse-vlm-vlmmodel-generate-description.modal.run \
  -H "Content-Type: application/json" \
  -d '{"frame_base64": "...", "prompt": "Describe this frame"}'

# Test LLM inference service (OpenAI-compatible)
# Note: The serve() endpoint provides vLLM's OpenAI-compatible API
curl -X POST https://your-username--general-inference-service-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "HuggingFaceTB/SmolLM3-3B", "messages": [{"role": "user", "content": "Hello"}]}'

# Health check
curl https://your-username--general-inference-service-health.modal.run
```

### Integration Tests
```bash
# Test the local/remote training-backend dispatch (RemoteTrainingBackend
# wraps ModalTrainingRunner behind the same TrainingBackend interface)
uv run pytest tests/finetuning/test_training_backend.py tests/finetuning/test_orchestrator.py -v

# Test VLM service locally (requires Modal CLI)
modal run scripts/modal_vlm_service.py::test_vlm --frame-path path/to/frame.jpg
```

## Troubleshooting

### Authentication Issues
```bash
# Refresh Modal credentials
modal token new
```

`ROUTER_OPTIMIZER_TEACHER_KEY` and `ANNOTATION_API_KEY` (used by the local DSPy
optimizer/annotation processes, not by any Modal-deployed function) are plain local
environment variables — `export ROUTER_OPTIMIZER_TEACHER_KEY=your-api-key`. Neither
`scripts/modal_vlm_service.py` nor `libs/finetuning/cogniverse_finetuning/training/modal_app.py`
attaches a `modal.Secret` today; `modal secret create` only matters if you extend one of
those apps yourself with `secrets=[modal.Secret.from_name("...")]`.

### GPU Availability

Modal's CLI has no `gpu list` command. Check current GPU pricing/availability on the
[Modal pricing page](https://modal.com/pricing), then change the GPU in the deployment
code itself, e.g. `gpu="T4"` instead of `gpu="A10G"` in `modal_app.py`, or
`GPU_TYPE=t4 modal deploy scripts/modal_vlm_service.py` for the VLM service.

### Cost Management
```bash
# Monitor usage (billing has no "current" subcommand — use report with --for)
modal billing report --for today

# Detailed report for a date range
modal billing report --start 2026-06-01 --end 2026-07-01 --csv > report.csv
```

Spending limits are configured in the Modal dashboard (Settings → Billing) — there is no
CLI command for setting them.

## Monitoring

### Modal Dashboard
- View all deployments: `modal app list`
- Check logs: `modal app logs <app-name>`
- Monitor costs: `modal billing report --for "this month"`

### Phoenix Integration
```python
# Track Modal service calls in Phoenix
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.config.utils import create_default_config_manager

# Get telemetry config
config_manager = create_default_config_manager()
telemetry_config = config_manager.get_telemetry_config("default")
telemetry = TelemetryManager(telemetry_config)

# Note: span() requires tenant_id parameter
with telemetry.span("modal.vlm.describe", tenant_id="your_org:production") as span:
    span.set_attribute("provider", "modal")
    span.set_attribute("model", "qwen3-vl-8b")
    result = await modal_vlm.describe(frame)
```

## When to Use Modal

### Use Modal When:

- Running DSPy optimization with large student models

- Processing large video batches (>100 videos)

- Need GPU acceleration without local hardware

- Scaling to multiple concurrent users

- Running experiments with different models

### Use Local When:

- Developing and testing

- Processing small batches (<10 videos)

- Have powerful local GPU

- Need lowest latency

- Cost-sensitive for small workloads

---

