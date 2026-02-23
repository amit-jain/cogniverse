# Modal Deployment Guide - Multi-Agent System

Complete guide for deploying LLMs and VLMs on Modal for the Cogniverse multi-agent video search system.

## Overview

Modal provides serverless GPU infrastructure for:

- **DSPy Optimization**: Teacher/student models using MIPROv2 optimizer

- **Video Processing**: VLMs for frame description generation during ingestion

- **Model Fine-tuning**: SFT and DPO training on cloud GPUs

- **Agent LLMs**: Deploy agent models on GPUs instead of local Ollama

## Use Cases in Cogniverse

### 1. DSPy Optimization (Teacher/Student)

The routing optimizer uses teacher/student patterns with DSPy optimizers:

```python
# Run the optimization orchestrator
from cogniverse_agents.optimizer.orchestrator import OptimizationOrchestrator

# Config loaded from COGNIVERSE_CONFIG env var or configs/config.json
orchestrator = OptimizationOrchestrator()
result = orchestrator.run_optimization()

# Available DSPy optimizers (MIPROv2 is used in the orchestrator):
from dspy.teleprompt import MIPROv2

# Teacher (Claude/GPT-4) generates training examples
# Student (SmolLM3-3B on Modal) learns from teacher examples via MIPROv2
# Optimized prompts saved to get_output_manager().get_optimization_dir()
```

### 2. VLM for Video Processing

The VLM service uses Qwen2-VL-7B for video frame descriptions:

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

## Modal Deployments

### LLM Inference Service

The main inference service provides OpenAI-compatible endpoints:

# Actual location: libs/runtime/cogniverse_runtime/inference/modal_inference_service.py
# App name: general-inference-service
# Endpoints: serve (vLLM OpenAI-compatible), /generate, /chat-completions, /health, /models

# Uses vLLM with configurable models (default: HuggingFaceTB/SmolLM3-3B via env var DEFAULT_MODEL)
# GPU: A100-80GB by default (configurable via DEFAULT_GPU env var)

Deploy:
```bash
modal deploy libs/runtime/cogniverse_runtime/inference/modal_inference_service.py
```

### DSPy with Modal (via ProviderFactory)

Use the existing provider infrastructure to run DSPy optimization with Modal:

```python
from cogniverse_agents.optimizer.providers.base_provider import (
    ProviderFactory,
    DSPyLMProvider
)

# Create Modal model provider
modal_provider = ProviderFactory.create_model_provider(
    provider_type="modal",
    config={}  # Provider config from config.json
)

# Wrap for DSPy compatibility
dspy_lm = DSPyLMProvider(
    model_provider=modal_provider,
    model_id="HuggingFaceTB/SmolLM3-3B",  # Or any model deployed to Modal
    model_type="modal"
)

# Use with DSPy modules via scoped context
import dspy
with dspy.context(lm=dspy_lm):
    result = module(query="...")
```

### VLM Service Deployment

# Actual location: scripts/modal_vlm_service.py
# App name: cogniverse-vlm
# Uses: SGLang + Qwen2-VL-7B-Instruct
# GPU: H100 by default (configurable via GPU_TYPE env var)

# The VLMModel class provides:
# - generate_description: Single frame description (supports frame_base64, frame_path, remote_frame_path)
# - upload_and_process_frames: Batch processing via zip upload

# Key endpoints after deployment:
# POST /generate_description - Single frame analysis
# POST /upload_and_process_frames - Batch frame processing

Deploy:
```bash
modal deploy scripts/modal_vlm_service.py
```

### Fine-tuning Service

Train models on Modal GPUs with the finetuning module:

# Actual location: libs/finetuning/cogniverse_finetuning/training/modal_app.py
# App name: cogniverse-finetuning

# The ModalTrainingRunner provides:
# - SFT (Supervised Fine-Tuning) on Modal GPUs
# - DPO (Direct Preference Optimization) on Modal GPUs
# - Configurable GPU selection (T4, A10G, A100-40GB, A100-80GB, H100)
# - Direct dataset passing (no upload needed) and adapter download

Deploy:
```bash
modal deploy libs/finetuning/cogniverse_finetuning/training/modal_app.py
```

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
      "provider": "modal",
      "model": "HuggingFaceTB/SmolLM3-3B",
      "api_base": "https://username--general-inference-service-serve.modal.run"
    },
    "teacher": {
      "provider": "anthropic",
      "model": "claude-3-5-sonnet-20241022",
      "api_key_env": "ROUTER_OPTIMIZER_TEACHER_KEY"
    }
  },
  "vlm_endpoint_url": "https://username--cogniverse-vlm-vlmmodel-generate-description.modal.run/"
}
```

The teacher/student optimization workflow:
1. **Teacher** (Claude/GPT-4) generates high-quality training examples
2. **Student** (SmolLM3-3B on Modal) learns from these examples via MIPROv2
3. Optimized prompts are saved to `get_output_manager().get_optimization_dir()`

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
- GPU: A10G (24GB) - configured in optimization.providers.modal
- Duration: 15-30 minutes
- Cost: ~$1.10/hour = ~$0.55
```

### Video Processing
```text
VLM (Qwen2-VL-7B):
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
modal deploy libs/runtime/cogniverse_runtime/inference/modal_inference_service.py
modal deploy scripts/modal_vlm_service.py
modal deploy libs/finetuning/cogniverse_finetuning/training/modal_app.py

# 2. Get endpoints
modal app list

# 3. Update your configuration file with endpoints
# Edit the config to include Modal endpoint in inference.modal_endpoint

# 4. Test integration
uv run python -c "
from cogniverse_agents.optimizer.providers.base_provider import ProviderFactory
provider = ProviderFactory.create_model_provider('modal', {'endpoint': 'YOUR_ENDPOINT'})
print(provider.health_check())
"
```

### Development Workflow

```bash
# 1. Deploy Modal inference service
modal deploy libs/runtime/cogniverse_runtime/inference/modal_inference_service.py

# 2. Get your Modal endpoint
modal app list  # Note the general-inference-service endpoint

# 3. Update your configuration with the Modal endpoint
# Set inference.modal_endpoint in config.json to your deployed endpoint

# 4. Run DSPy optimization with teacher/student
# Note: Orchestrator requires config_manager injected at startup via llm_config
uv run python -m cogniverse_agents.optimizer.orchestrator

# 5. Run video ingestion with Modal VLM
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
VLM: Modal (Qwen2-VL-7B via scripts/modal_vlm_service.py)
LLM Inference: Modal (vLLM via modal_inference_service.py)
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
# Test Modal provider factory
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/ -k "provider" -v

# Test VLM service locally (requires Modal CLI)
modal run scripts/modal_vlm_service.py::test_vlm --frame-path path/to/frame.jpg
```

## Troubleshooting

### Authentication Issues
```bash
# Refresh Modal credentials
modal token new

# Set secrets for API keys
modal secret create teacher-key ROUTER_OPTIMIZER_TEACHER_KEY=your-api-key
modal secret create annotation-key ANNOTATION_API_KEY=your-api-key
```

### GPU Availability
```bash
# Check available GPUs
modal gpu list

# Use different GPU if needed
# Change in deployment: gpu="T4" instead of gpu="A10G"
```

### Cost Management
```bash
# Monitor usage
modal billing current

# Set spending limits
modal billing limit set --monthly 100
```

## Monitoring

### Modal Dashboard
- View all deployments: `modal app list`
- Check logs: `modal logs app-name`
- Monitor costs: `modal billing`

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
with telemetry.span("modal.vlm.describe", tenant_id="default") as span:
    span.set_attribute("provider", "modal")
    span.set_attribute("model", "qwen2-vl-7b")
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

