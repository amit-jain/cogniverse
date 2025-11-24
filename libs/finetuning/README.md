# Cogniverse Fine-Tuning Infrastructure

End-to-end fine-tuning pipeline for LLM agents and embedding models with automatic method selection, synthetic data generation, and Modal GPU training.

## Features

- **Auto-Selection**: Automatically chooses SFT vs DPO vs embedding training based on available data
- **Dataset Extraction**: Extracts training data from Phoenix telemetry (spans + annotations)
- **Synthetic Integration**: Generates synthetic data via existing `cogniverse-synthetic` service with **mandatory** human approval
- **LoRA Training**: Parameter-efficient fine-tuning with PEFT/LoRA adapters
- **Modal GPU**: Cloud GPU training on Modal (T4, A10G, A100, H100)
- **Multiple Formats**: Alpaca, ShareGPT, ChatML for LLM, triplet loss for embeddings

## Quick Start

```python
from cogniverse_finetuning import finetune
from cogniverse_foundation.telemetry import TelemetryManager

# Get telemetry provider
manager = TelemetryManager()
provider = manager.get_provider("tenant1", "cogniverse-tenant1")

# Example 1: Local Training (CPU/GPU)
result = await finetune(
    telemetry_provider=provider,
    tenant_id="tenant1",
    project="cogniverse-tenant1",
    model_type="llm",  # or "embedding"
    agent_type="routing",  # routing, profile_selection, entity_extraction
    base_model="HuggingFaceTB/SmolLM-135M",
    backend="local"  # Train on local machine
)

# Example 2: Remote GPU Training (Modal)
result = await finetune(
    telemetry_provider=provider,
    tenant_id="tenant1",
    project="cogniverse-tenant1",
    model_type="llm",
    agent_type="routing",
    base_model="HuggingFaceTB/SmolLM-135M",
    backend="remote",  # Train on remote GPU
    backend_provider="modal",  # Cloud provider
    gpu="A100-40GB",  # GPU type
    epochs=5
)

print(f"Method: {result.training_method}")  # sft or dpo
print(f"Adapter: {result.adapter_path}")
```

### Remote GPU Setup (Modal)

To use remote GPU training with Modal:

```bash
# Install Modal CLI
pip install modal

# Login to Modal
modal token new

# Deploy training functions
modal deploy libs/finetuning/cogniverse_finetuning/training/modal_app.py
```

## Modal GPU Options

| GPU | VRAM | Use Case | Cost (approx) |
|-----|------|----------|---------------|
| T4 | 16GB | SmolLM 135M/360M | $0.50/hr |
| A10G | 24GB | SmolLM 1.7B, Qwen 3B | $1.10/hr |
| A100-40GB | 40GB | Qwen 7B | $3.00/hr |
| A100-80GB | 80GB | Large models | $5.00/hr |
| H100 | 80GB | Fastest training | $8.00/hr |

## Training Methods

| Method | Use Case | Data Requirements |
|--------|----------|-------------------|
| **SFT** | Instruction tuning | 50+ approved examples |
| **DPO** | Preference learning | 20+ preference pairs |
| **Embedding** | Semantic search | 100+ search triplets |

