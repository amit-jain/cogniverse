# Modal Deployment Guide - Multi-Agent System

Complete guide for deploying LLMs and VLMs on Modal for the Cogniverse multi-agent video search system.

## 🎯 Overview

Modal provides serverless GPU infrastructure for:
- **DSPy Optimization**: Teacher/student models for Bootstrap, SIMBA, MIPRO, GEPA optimizers
- **Video Processing**: VLMs for frame description generation during ingestion
- **Embedding Models**: ColPali, VideoPrism, ColQwen for video embeddings
- **Agent LLMs**: Deploy agent models on GPUs instead of local Ollama

## 🚀 Use Cases in Cogniverse

### 1. DSPy Optimization (Teacher/Student)

Our routing optimizer uses teacher/student patterns:

```python
# GEPA Optimizer with teacher/student
from cogniverse_agents.routing.gepa_optimizer import GEPAOptimizer

optimizer = GEPAOptimizer(
    teacher_model="claude-3-opus",  # API or Modal
    student_model="gemma-7b",       # Modal deployment
    experience_buffer=buffer
)

# Bootstrap/SIMBA/MIPRO also use teacher/student
from dspy import BootstrapFewShot, SIMBA, MIPRO

# Teacher generates examples, student learns routing
bootstrap = BootstrapFewShot(
    teacher_model=gpt4_model,    # Strong teacher
    student_model=modal_gemma,   # Efficient student on Modal
)
```

### 2. VLM for Video Processing

Deploy Qwen2-VL or LLaVA for frame descriptions:

```python
# Modal VLM for frame descriptions
from scripts.modal_vlm_service import VLMModel

vlm = VLMModel()
description = vlm.generate_description(frame_image)
```

### 3. Embedding Models

Deploy heavy embedding models on Modal GPUs:

```python
# ColPali on Modal instead of local
from cogniverse_runtime.embeddings.modal_colpali import ModalColPali

colpali = ModalColPali(endpoint="modal://colpali-deployment")
embeddings = await colpali.encode_frames(frames)
```

## 📦 Modal Deployments

### DSPy Optimizer Deployment

```python
# cogniverse_agents/modal/dspy_optimizer_service.py
import modal

app = modal.App("cogniverse-dspy-optimizer")

@app.cls(
    gpu="A10G",
    image=modal.Image.debian_slim()
        .pip_install("dspy-ai", "torch", "transformers")
)
class DSPyOptimizerService:
    def __init__(self):
        # Load student model (Gemma, Llama, etc.)
        self.student_model = load_model("google/gemma-7b")

    @modal.method()
    def optimize_routing(self,
                        teacher_examples: list,
                        routing_data: dict) -> dict:
        """Run DSPy optimization with teacher examples"""
        # Bootstrap, SIMBA, MIPRO, or GEPA optimization
        optimizer = self.select_optimizer(len(teacher_examples))
        return optimizer.compile(
            student=self.student_model,
            trainset=teacher_examples
        )
```

Deploy:
```bash
modal deploy cogniverse_agents/modal/dspy_optimizer_service.py
```

### VLM Service Deployment

```python
# cogniverse_vlm/modal_vlm_service.py
import modal

app = modal.App("cogniverse-vlm")

@app.cls(
    gpu="L40S",  # or T4 for cost savings
    image=modal.Image.debian_slim()
        .pip_install("transformers", "qwen-vl-utils")
)
class VLMModel:
    def __init__(self):
        self.model = AutoModelForVision2Seq.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct"
        )

    @modal.method()
    def generate_description(self, image_base64: str) -> str:
        """Generate description for video frame"""
        return self.model.generate(image_base64)
```

### Embedding Model Deployment

```python
# cogniverse_processing/modal/embedding_service.py
import modal

app = modal.App("cogniverse-embeddings")

@app.cls(
    gpu="A10G",
    image=modal.Image.debian_slim()
        .pip_install("colpali", "videopris", "torch")
)
class EmbeddingService:
    def __init__(self):
        self.colpali = ColPali.from_pretrained("vidore/colsmol-500m")
        self.videoprism = VideoPrism.from_pretrained("google/videoprism-base")

    @modal.method()
    def encode_frames(self, frames: list, model: str = "colpali") -> np.array:
        """Generate embeddings for video frames"""
        if model == "colpali":
            return self.colpali.encode(frames)
        elif model == "videoprism":
            return self.videoprism.encode(frames)
```

## 🔧 Integration with Cogniverse

### Configuration

Update `configs/config.json`:

```json
{
  "optimization": {
    "dspy": {
      "teacher": {
        "provider": "anthropic",
        "model": "claude-3-opus"
      },
      "student": {
        "provider": "modal",
        "endpoint": "https://username--cogniverse-dspy-optimizer.modal.run",
        "model": "gemma-7b"
      }
    }
  },
  "ingestion": {
    "vlm": {
      "provider": "modal",
      "endpoint": "https://username--cogniverse-vlm.modal.run"
    },
    "embeddings": {
      "provider": "modal",
      "endpoint": "https://username--cogniverse-embeddings.modal.run"
    }
  }
}
```

### Using Modal Services

#### For DSPy Optimization

```python
from cogniverse_agents.routing.optimizer_factory import OptimizerFactory

factory = OptimizerFactory()

# Modal student, API teacher
optimizer = factory.create_gepa_optimizer(
    teacher_provider="anthropic",
    student_provider="modal"
)

# Run optimization
optimized_router = optimizer.optimize(
    routing_examples=experience_buffer.sample(1000)
)
```

#### For Video Ingestion

```python
from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline

pipeline = VideoIngestionPipeline(
    vlm_provider="modal",  # Use Modal VLM
    embedding_provider="modal",  # Use Modal embeddings
    profile="video_colpali_smol500_mv_frame"
)

# Process videos with Modal acceleration
await pipeline.process_video(video_path)
```

## 📊 Performance & Cost

### DSPy Optimization
```
Teacher (Claude/GPT-4):
- Cost: ~$0.01-0.10 per example
- Total: ~$5-50 for full optimization

Student (Modal Gemma-7B):
- GPU: A10G (24GB)
- Duration: 15-30 minutes
- Cost: ~$1.10/hour = ~$0.55
```

### Video Processing
```
VLM (Qwen2-VL-7B):
- GPU: L40S or T4
- Speed: 2-4 seconds per frame
- Cost: ~$0.01-0.05 per frame

Embeddings (ColPali):
- GPU: A10G
- Speed: 100ms per frame
- Cost: ~$0.001 per frame
```

### Production Inference
```
Agents with Modal LLMs:
- GPU: T4 (16GB) minimum
- Scaling: 1-10 instances auto-scale
- Cost: $0.60-1.10/hour when active
- Latency: 50-200ms per query
```

## 🚀 Deployment Commands

### Full System Deployment

```bash
# 1. Deploy all Modal services
modal deploy cogniverse_agents/modal/dspy_optimizer_service.py
modal deploy cogniverse_vlm/modal_vlm_service.py
modal deploy cogniverse_processing/modal/embedding_service.py

# 2. Get endpoints
modal app list

# 3. Update config.json with endpoints
vim configs/config.json

# 4. Test integration
python tests/test_modal_integration.py
```

### Development Workflow

```bash
# Local development with Modal services
export MODAL_ENDPOINTS=true

# Run video ingestion with Modal
uv run python scripts/run_ingestion.py \
    --video_dir data/videos \
    --use-modal-vlm \
    --use-modal-embeddings

# Run DSPy optimization with Modal
uv run python scripts/run_optimization.py \
    --optimizer GEPA \
    --use-modal-student
```

## 🔄 Hybrid Configurations

### Option 1: Modal for Heavy Compute
```yaml
# Heavy models on Modal, light models local
VLM: Modal (Qwen2-VL-7B)
Embeddings: Modal (ColPali, VideoPrism)
Agents: Local (Ollama Llama3)
DSPy Student: Modal (Gemma-7B)
DSPy Teacher: API (Claude/GPT-4)
```

### Option 2: Full Modal
```yaml
# Everything on Modal
VLM: Modal
Embeddings: Modal
Agents: Modal (deployed LLMs)
DSPy: Modal (both teacher and student)
```

### Option 3: Cost-Optimized
```yaml
# Balance cost and performance
VLM: Local (Ollama LLaVA)
Embeddings: Modal (only for batch processing)
Agents: Local
DSPy: Teacher API, Student Modal
```

## 🎯 Benefits vs Local

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

## 🧪 Testing

### Test Modal Services
```bash
# Test VLM
curl -X POST https://your-vlm.modal.run/describe \
  -d '{"image_base64": "..."}'

# Test embeddings
curl -X POST https://your-embeddings.modal.run/encode \
  -d '{"frames": [...], "model": "colpali"}'

# Test DSPy optimizer
curl -X POST https://your-optimizer.modal.run/optimize \
  -d '{"examples": [...], "config": {...}}'
```

### Integration Tests
```bash
# Full pipeline test with Modal
uv run pytest tests/integration/test_modal_pipeline.py

# DSPy optimization test
uv run pytest tests/routing/test_modal_optimization.py
```

## 🚨 Troubleshooting

### Authentication Issues
```bash
# Refresh Modal credentials
modal token new

# Set secrets for API keys
modal secret create anthropic-key ANTHROPIC_API_KEY=sk-...
modal secret create openai-key OPENAI_API_KEY=sk-...
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

## 📈 Monitoring

### Modal Dashboard
- View all deployments: `modal app list`
- Check logs: `modal logs app-name`
- Monitor costs: `modal billing`

### Phoenix Integration
```python
# Track Modal service calls in Phoenix
from cogniverse_foundation.telemetry.multi_tenant_manager import MultiTenantTelemetryManager

telemetry = MultiTenantTelemetryManager()

with telemetry.span("modal.vlm.describe", tenant_id) as span:
    span.set_attribute("provider", "modal")
    span.set_attribute("model", "qwen2-vl-7b")
    result = await modal_vlm.describe(frame)
```

## 🎯 When to Use Modal

### Use Modal When:
- 🎯 Running DSPy optimization with large student models
- 🎯 Processing large video batches (>100 videos)
- 🎯 Need GPU acceleration without local hardware
- 🎯 Scaling to multiple concurrent users
- 🎯 Running experiments with different models

### Use Local When:
- 💻 Developing and testing
- 💻 Processing small batches (<10 videos)
- 💻 Have powerful local GPU
- 💻 Need lowest latency
- 💻 Cost-sensitive for small workloads

---

**Last Updated**: 2025-11-13
**Status**: Production Ready - Integrated with 11-Package Architecture