# Fine-Tuning Module

**Package**: `cogniverse-finetuning`
**Location**: `libs/finetuning/`

The Fine-Tuning module provides end-to-end infrastructure for adapting LLM and embedding models using telemetry data from production systems. It supports Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and embedding model fine-tuning with LoRA adapters.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Components](#components)
- [Data Flow](#data-flow)
- [Usage](#usage)
- [Configuration](#configuration)
- [Backend Support](#backend-support)
- [Training Methods](#training-methods)
- [Validation & Error Handling](#validation--error-handling)
- [Best Practices](#best-practices)

---

## Overview

### Purpose

The Fine-Tuning module enables continuous improvement of AI agents by learning from production telemetry data:

1. **Extract Training Data** from Phoenix telemetry (spans + annotations)
2. **Auto-Select Method** (SFT vs DPO) based on available data
3. **Generate Synthetic Data** with human approval when needed
4. **Train LoRA Adapters** on local or remote GPUs
5. **Return Adapters** for deployment

### Key Features

- **Telemetry Integration**: Direct extraction from Phoenix spans and annotations
- **Auto-Selection**: Automatically chooses SFT or DPO based on data availability
- **Synthetic Data**: Generates additional training data with mandatory human approval
- **LoRA Adapters**: Parameter-efficient fine-tuning (trains <1% of parameters)
- **Multi-Backend**: Local GPU or remote cloud (Modal, SageMaker, Azure ML)
- **Three Training Types**: LLM (SFT/DPO), Embedding (triplet loss)
- **Validation**: Dataset validation, schema checking, graceful error handling
- **Industry Standard**: Uses HuggingFace TRL/PEFT libraries

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Fine-Tuning Orchestrator                     │
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Dataset     │    │   Method     │    │  Training    │      │
│  │  Extraction  │───▶│  Selection   │───▶│  Execution   │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                     │                     │            │
│         ▼                     ▼                     ▼            │
│  Phoenix Telemetry    Auto SFT/DPO     Local/Remote GPU        │
└─────────────────────────────────────────────────────────────────┘
         │                     │                     │
         ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Phoenix    │      │  Synthetic   │      │   Backend    │
│   Provider   │      │   Service    │      │   Provider   │
│ (Spans+Anno) │      │   (DSPy)     │      │ (Local/Modal)│
└──────────────┘      └──────────────┘      └──────────────┘
```

### Component Architecture

```
cogniverse-finetuning/
│
├── orchestrator.py          # High-level coordination
│   ├── FinetuningOrchestrator
│   ├── validate_sft_dataset()
│   ├── validate_dpo_dataset()
│   └── validate_embedding_dataset()
│
├── dataset/                 # Data extraction & formatting
│   ├── trace_converter.py      # Phoenix → SFT examples
│   ├── preference_extractor.py # Phoenix → DPO pairs
│   ├── embedding_extractor.py  # Search logs → triplets
│   ├── method_selector.py      # Auto SFT/DPO selection
│   └── formatters.py            # TRL format conversion
│
├── training/                # Model training
│   ├── backend.py               # Backend abstraction
│   ├── sft_trainer.py           # SFT with TRL
│   ├── dpo_trainer.py           # DPO with TRL
│   ├── embedding_finetuner.py   # Triplet loss
│   └── modal_runner.py          # Remote GPU execution
│
└── __init__.py             # Public API
```

---

## Components

### 1. Orchestrator (`orchestrator.py`)

**Purpose**: High-level coordination of fine-tuning pipeline.

**Key Classes**:
- `FinetuningOrchestrator`: Main entry point
- `OrchestrationConfig`: Configuration dataclass
- `OrchestrationResult`: Training result

**Responsibilities**:
1. Coordinate dataset extraction
2. Auto-select training method
3. Trigger synthetic generation if needed
4. Execute training via backend
5. Return adapter and metrics

**Public API**:
```python
async def finetune(
    telemetry_provider: TelemetryProvider,
    tenant_id: str,
    project: str,
    model_type: Literal["llm", "embedding"],
    agent_type: Optional[str] = None,
    modality: Optional[str] = None,
    backend: Literal["local", "remote"] = "local",
    **kwargs
) -> OrchestrationResult
```

---

### 2. Dataset Extraction (`dataset/`)

#### TraceToInstructionConverter (`trace_converter.py`)

**Purpose**: Extract SFT training data from Phoenix spans with approved annotations.

**Input**: Phoenix spans + annotations
**Output**: `InstructionDataset` with instruction-response pairs

**Data Format**:
```python
InstructionExample(
    instruction="Route the following query to the appropriate agent",
    input="What's the weather?",
    output="weather_agent",
    metadata={"span_id": "...", "confidence": 0.95}
)
```

**Usage**:
```python
converter = TraceToInstructionConverter(provider)
dataset = await converter.convert(
    project="cogniverse-tenant1",
    agent_type="routing"
)
```

---

#### PreferencePairExtractor (`preference_extractor.py`)

**Purpose**: Extract DPO preference pairs from spans with both approved and rejected annotations.

**Input**: Phoenix spans with multiple annotations per span
**Output**: `PreferenceDataset` with chosen/rejected pairs

**Data Format**:
```python
PreferencePair(
    prompt="Route: What's the weather?",
    chosen="weather_agent",      # Approved annotation
    rejected="general_agent",    # Rejected annotation
    metadata={"span_id": "...", "chosen_score": 0.9}
)
```

**Usage**:
```python
extractor = PreferencePairExtractor(provider)
dataset = await extractor.extract(
    project="cogniverse-tenant1",
    agent_type="routing"
)
```

---

#### TripletExtractor (`embedding_extractor.py`)

**Purpose**: Extract triplets from search logs for embedding fine-tuning.

**Input**: Phoenix search spans with relevance feedback
**Output**: `TripletDataset` with anchor/positive/negative triplets

**Data Format**:
```python
Triplet(
    anchor="query text",
    positive="relevant document",
    negative="irrelevant document",
    metadata={"search_id": "..."}
)
```

---

#### TrainingMethodSelector (`method_selector.py`)

**Purpose**: Auto-select SFT vs DPO based on available data.

**Decision Logic**:
```
IF preference_pairs >= min_dpo_pairs (20):
    ✅ Recommend DPO (more sample-efficient)
ELIF approved_examples >= min_sft_examples (50):
    ✅ Recommend SFT
ELSE:
    ⚠️ Insufficient data → Generate synthetic
```

**Usage**:
```python
selector = TrainingMethodSelector(
    synthetic_service=synthetic_svc,
    approval_orchestrator=approval_orch
)

analysis, approved_batch = await selector.analyze_and_prepare(
    provider=provider,
    project="cogniverse-tenant1",
    agent_type="routing",
    generate_synthetic=True
)

print(f"Method: {analysis.recommended_method}")
print(f"Confidence: {analysis.confidence}")
```

---

### 3. Training Backends (`training/`)

#### Backend Abstraction (`backend.py`)

**Purpose**: Unified interface for local and remote training.

**Architecture**:
```
TrainingBackend (abstract)
    ├── LocalTrainingBackend    # Local GPU/CPU
    └── RemoteTrainingBackend   # Modal/SageMaker/Azure ML
```

**Interface**:
```python
class TrainingBackend(ABC):
    async def train_sft(...) -> TrainingJobResult
    async def train_dpo(...) -> TrainingJobResult
    async def train_embedding(...) -> TrainingJobResult
```

---

#### SFTFinetuner (`sft_trainer.py`)

**Purpose**: Supervised fine-tuning with HuggingFace TRL.

**Technology Stack**:
- **Library**: TRL `SFTTrainer`
- **Format**: Alpaca text format (`{"text": "..."}`)
- **PEFT**: LoRA adapters (r=8, alpha=16)
- **Validation**: Automatic 90/10 split for datasets >100 examples

**Training Process**:
1. Load base model (e.g., SmolLM-135M)
2. Apply LoRA adapters (graceful fallback if fails)
3. Create train/val split if dataset >100
4. Train with TRL SFTTrainer
5. Save LoRA adapter
6. Return metrics

**Memory**: ~2-3x model size (e.g., 3B model = 6-9GB GPU)

---

#### DPOFinetuner (`dpo_trainer.py`)

**Purpose**: Direct Preference Optimization with HuggingFace TRL.

**Technology Stack**:
- **Library**: TRL `DPOTrainer`
- **Format**: `{"prompt": "...", "chosen": "...", "rejected": "..."}`
- **PEFT**: LoRA adapters
- **Validation**: Automatic 90/10 split for datasets >100 pairs

**Training Process**:
1. Load base model + reference model (frozen copy)
2. Apply LoRA to policy model only
3. Create train/val split if dataset >100
4. Train with DPO loss: `-log σ(β * (log π(chosen) - log π(rejected)))`
5. Save LoRA adapter
6. Return metrics (loss, reward accuracy, margins)

**Memory**: **2x model size** due to reference model (e.g., 3B model = 12GB GPU)

---

#### EmbeddingFinetuner (`embedding_finetuner.py`)

**Purpose**: Fine-tune embedding models with contrastive learning.

**Technology Stack**:
- **Library**: sentence-transformers
- **Loss**: Triplet loss with margin
- **PEFT**: LoRA adapters (optional)

**Training Process**:
1. Load base embedding model
2. Apply LoRA (graceful fallback if fails)
3. Train with triplet loss
4. Save adapter
5. Return metrics

---

### 4. Formatters (`dataset/formatters.py`)

**Purpose**: Convert extracted data to TRL-compatible formats.

**Supported Formats**:
- **Alpaca Text**: `{"text": "### Instruction:\n...\n### Response:\n..."}`
- **DPO**: `{"prompt": "...", "chosen": "...", "rejected": "..."}`
- **Triplets**: `{"anchor": "...", "positive": "...", "negative": "..."}`

---

## Data Flow

### End-to-End Flow (LLM Fine-Tuning)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Data Extraction                                               │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
Phoenix Telemetry Provider
         │
         ├─▶ TraceStore.get_spans(project="cogniverse-tenant1")
         │       └─▶ Returns: spans_df
         │
         └─▶ AnnotationStore.get_annotations(spans_df)
                 └─▶ Returns: annotations_df (with approved/rejected)

┌─────────────────────────────────────────────────────────────────┐
│ 2. Method Selection                                              │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
TrainingMethodSelector.analyze_data()
         │
         ├─▶ Count approved annotations → SFT examples
         ├─▶ Count preference pairs (approved + rejected) → DPO pairs
         │
         └─▶ Decision:
             ├─▶ preference_pairs >= 20? → DPO ✅
             ├─▶ approved >= 50? → SFT ✅
             └─▶ Else → Generate Synthetic ⚠️

┌─────────────────────────────────────────────────────────────────┐
│ 3. Dataset Formatting                                            │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
IF DPO:
    PreferencePairExtractor.extract()
         └─▶ InstructionFormatter.format_dpo()
              └─▶ [{"prompt": "...", "chosen": "...", "rejected": "..."}]
ELSE (SFT):
    TraceToInstructionConverter.convert()
         └─▶ InstructionFormatter.format_alpaca_text()
              └─▶ [{"text": "### Instruction:\n...\n### Response:\n..."}]

┌─────────────────────────────────────────────────────────────────┐
│ 4. Validation                                                    │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
validate_dpo_dataset(dataset) / validate_sft_dataset(dataset)
         │
         ├─▶ Check: len(dataset) > 0?
         ├─▶ Check: Required fields present?
         └─▶ Raise ValueError if invalid

┌─────────────────────────────────────────────────────────────────┐
│ 5. Training                                                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
Backend.train_dpo() / Backend.train_sft()
         │
         ├─▶ Load base model
         ├─▶ Apply LoRA (try/except fallback)
         ├─▶ Split train/val if >100 examples
         ├─▶ Train with TRL
         └─▶ Save adapter

┌─────────────────────────────────────────────────────────────────┐
│ 6. Return Result                                                 │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
OrchestrationResult(
    model_type="llm",
    training_method="dpo",
    adapter_path="outputs/adapters/dpo_routing_20251124_143052/",
    metrics={"train_loss": 0.42, "eval_reward_accuracy": 0.78},
    base_model="HuggingFaceTB/SmolLM-135M",
    lora_config={"use_lora": True},
    used_synthetic=False
)
```

---

## Usage

### Basic Usage (Local GPU)

```python
from cogniverse_telemetry_phoenix import PhoenixProvider
from cogniverse_finetuning import finetune

# Initialize provider
provider = PhoenixProvider(
    tenant_id="tenant1",
    project="cogniverse-tenant1"
)

# Fine-tune routing agent
result = await finetune(
    telemetry_provider=provider,
    tenant_id="tenant1",
    project="cogniverse-tenant1",
    model_type="llm",
    agent_type="routing",
    base_model="HuggingFaceTB/SmolLM-135M",
    backend="local",
    epochs=3,
    batch_size=4,
    learning_rate=2e-4
)

print(f"Adapter saved to: {result.adapter_path}")
print(f"Training method: {result.training_method}")
print(f"Metrics: {result.metrics}")
```

---

### Remote GPU Training (Modal)

```python
result = await finetune(
    telemetry_provider=provider,
    tenant_id="tenant1",
    project="cogniverse-tenant1",
    model_type="llm",
    agent_type="routing",
    backend="remote",
    backend_provider="modal",
    gpu="A100-40GB",
    gpu_count=1,
    epochs=5,
    batch_size=8,
    learning_rate=1e-4,
    timeout=7200  # 2 hours
)
```

---

### Embedding Fine-Tuning

```python
result = await finetune(
    telemetry_provider=provider,
    tenant_id="tenant1",
    project="cogniverse-tenant1",
    model_type="embedding",
    modality="video",
    base_model="jinaai/jina-embeddings-v3",
    backend="local",
    epochs=3,
    batch_size=16,
    learning_rate=2e-5
)
```

---

### Using the Orchestrator Directly

```python
from cogniverse_finetuning import FinetuningOrchestrator, OrchestrationConfig

config = OrchestrationConfig(
    tenant_id="tenant1",
    project="cogniverse-tenant1",
    model_type="llm",
    agent_type="routing",
    base_model="HuggingFaceTB/SmolLM-135M",
    min_sft_examples=50,
    min_dpo_pairs=20,
    epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    backend="local",
    generate_synthetic=True,
    output_dir="outputs/adapters"
)

orchestrator = FinetuningOrchestrator(
    telemetry_provider=provider,
    synthetic_service=synthetic_svc,
    approval_orchestrator=approval_orch
)

result = await orchestrator.run(config)
```

---

### Loading and Using Adapters

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceTB/SmolLM-135M"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    result.adapter_path
)

# Use for inference
inputs = tokenizer("What's the weather?", return_tensors="pt")
outputs = model.generate(**inputs)
```

---

## Configuration

### OrchestrationConfig

```python
@dataclass
class OrchestrationConfig:
    # Tenant and project
    tenant_id: str
    project: str

    # Model type
    model_type: Literal["llm", "embedding"]

    # Agent type (for LLM)
    agent_type: Optional[Literal["routing", "profile_selection", "entity_extraction"]] = None

    # Modality (for embedding)
    modality: Optional[Literal["video", "image", "text"]] = None

    # Base model
    base_model: str = "HuggingFaceTB/SmolLM-135M"

    # Auto-selection thresholds
    min_sft_examples: int = 50
    min_dpo_pairs: int = 20
    min_triplets: int = 100  # For embeddings

    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    use_lora: bool = True

    # Backend configuration
    backend: Literal["local", "remote"] = "local"
    backend_provider: str = "modal"  # modal, sagemaker, azure_ml

    # Remote backend config
    gpu: str = "A10G"
    gpu_count: int = 1
    cpu: int = 4
    memory: int = 16384  # MB
    timeout: int = 3600  # seconds

    # Synthetic generation
    generate_synthetic: bool = True

    # Output
    output_dir: str = "outputs/adapters"
```

---

### Hyperparameter Recommendations

| Model Size | Batch Size | Learning Rate | GPU Memory | Training Time |
|------------|------------|---------------|------------|---------------|
| SmolLM-135M | 4-8 | 2e-4 | 2-4GB | Fast (~10 min) |
| Qwen2.5-3B | 2-4 | 5e-5 | 12-16GB | Medium (~1 hour) |
| Llama-3.1-8B | 1-2 | 2e-5 | 24-40GB | Slow (~3 hours) |

**DPO Memory**: Multiply by **2x** (loads reference model)

---

## Backend Support

### Local Backend

**When to Use**:
- Development/testing
- Small models (SmolLM-135M)
- Have local GPU

**Configuration**:
```python
backend="local"
```

**Pros**:
- ✅ No network latency
- ✅ No additional costs
- ✅ Easy debugging

**Cons**:
- ❌ GPU availability
- ❌ Manual resource management
- ❌ No automatic scaling

---

### Remote Backend (Modal)

**When to Use**:
- Production training
- Large models (>3B)
- Need powerful GPUs (A100, H100)
- Want automatic scaling

**Configuration**:
```python
backend="remote",
backend_provider="modal",
gpu="A100-40GB",
gpu_count=1,
timeout=7200
```

**Pros**:
- ✅ Powerful GPUs on-demand
- ✅ Automatic resource management
- ✅ Scalable
- ✅ Pay per use

**Cons**:
- ❌ Network overhead
- ❌ Usage costs
- ❌ Requires Modal setup

**Modal Setup**:
```bash
# Deploy Modal app
modal deploy libs/finetuning/cogniverse_finetuning/training/modal_app.py
```

---

## Training Methods

### Supervised Fine-Tuning (SFT)

**When to Use**:
- Have approved examples (instruction → response)
- No preference data
- Bootstrap new agent behavior

**Data Requirement**: ≥50 approved examples

**Format**:
```python
{
    "text": "### Instruction:\nRoute the query\n### Input:\nWhat's weather?\n### Response:\nweather_agent"
}
```

**Advantages**:
- ✅ Simple data requirements
- ✅ Fast training
- ✅ Lower memory (1x model)

**Disadvantages**:
- ❌ Less sample-efficient than DPO
- ❌ No explicit preference learning

---

### Direct Preference Optimization (DPO)

**When to Use**:
- Have preference pairs (approved + rejected)
- Want sample-efficient training
- Align agent with human preferences

**Data Requirement**: ≥20 preference pairs

**Format**:
```python
{
    "prompt": "Route: What's weather?",
    "chosen": "weather_agent",
    "rejected": "general_agent"
}
```

**Advantages**:
- ✅ More sample-efficient (learns from preferences)
- ✅ No reward model needed
- ✅ Industry standard (Anthropic, OpenAI)

**Disadvantages**:
- ❌ Higher memory (2x model)
- ❌ Requires preference annotations

**Algorithm**:
```
Loss = -log σ(β * (log π_θ(chosen|prompt) - log π_θ(rejected|prompt)))
```

Where:
- `π_θ`: Policy model (being trained)
- `β`: KL penalty coefficient (default 0.1)
- `σ`: Sigmoid function

---

### Embedding Fine-Tuning

**When to Use**:
- Improve search/retrieval quality
- Domain adaptation for embeddings
- Learn from search feedback

**Data Requirement**: ≥100 triplets

**Format**:
```python
{
    "anchor": "query text",
    "positive": "relevant document",
    "negative": "irrelevant document"
}
```

**Loss**: Triplet loss with margin
```
Loss = max(0, margin + sim(anchor, neg) - sim(anchor, pos))
```

---

## Validation & Error Handling

### Dataset Validation

All datasets are validated before training:

**Empty Dataset Check**:
```python
if len(dataset) == 0:
    raise ValueError("Cannot train with empty dataset")
```

**Schema Validation**:
```python
# SFT requires "text" field
required_fields = ["text"]

# DPO requires "prompt", "chosen", "rejected"
required_fields = ["prompt", "chosen", "rejected"]

for idx, item in enumerate(dataset):
    missing = [f for f in required_fields if f not in item]
    if missing:
        raise ValueError(f"Invalid dataset at index {idx}: missing {missing}")
```

**Preference Deduplication**:
```python
# Skip pairs where chosen == rejected (no learning signal)
if chosen_response == rejected_response:
    logger.warning(f"Skipping identical pair for span {span_id}")
    continue
```

---

### Graceful LoRA Fallback

LoRA application includes graceful fallback:

```python
try:
    lora_config = LoraConfig(...)
    model = get_peft_model(model, lora_config)
    logger.info("LoRA applied successfully")
except Exception as e:
    logger.warning(
        f"Failed to apply LoRA: {e}. "
        "Training will proceed with full model fine-tuning."
    )
    # Continue with full fine-tuning
```

**Impact**: Training continues even if model architecture doesn't support LoRA.

---

### Automatic Validation Split

For datasets >100 examples, automatic 90/10 train/val split:

```python
if len(dataset) > 100:
    split_idx = int(len(dataset) * 0.9)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    logger.info(f"Split: {len(train_data)} train, {len(val_data)} val")
```

**Benefits**:
- Early stopping to prevent overfitting
- Eval metrics during training
- Better training quality

---

### Concurrent Training Safety

Output directories include timestamps to prevent conflicts:

```python
timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
output_dir = f"{config.output_dir}/dpo_{config.agent_type}_{timestamp}"
```

**Result**: Multiple concurrent jobs don't overwrite each other.

---

## Best Practices

### 1. Data Preparation

**Minimum Data Requirements**:
- SFT: 50+ approved examples
- DPO: 20+ preference pairs (40+ annotations)
- Embedding: 100+ triplets

**Data Quality**:
- ✅ Review annotations before training
- ✅ Use approval workflow for synthetic data
- ✅ Ensure diversity in training examples
- ❌ Don't train on identical preference pairs

---

### 2. Model Selection

**Start Small**:
- Begin with SmolLM-135M for testing
- Validate pipeline works end-to-end
- Scale up to larger models

**GPU Memory Guide**:
- SmolLM-135M: 2-4GB (any GPU)
- Qwen2.5-3B: 12-16GB (RTX 3090, A10G)
- Llama-3.1-8B: 24-40GB (A100)

**DPO Considerations**:
- Requires 2x GPU memory (loads reference model)
- Use remote backend for large models

---

### 3. Training Configuration

**Learning Rate**:
- Small models (135M): 2e-4
- Medium models (3B): 5e-5
- Large models (8B+): 2e-5

**Batch Size**:
- Limited GPU memory: Start with batch_size=1-2
- Ample GPU memory: Use batch_size=4-8
- Use gradient accumulation if OOM

**Epochs**:
- Small datasets (<100): 3-5 epochs
- Large datasets (>1000): 1-3 epochs
- Monitor validation loss to prevent overfitting

---

### 4. Backend Selection

**Use Local Backend When**:
- Testing/development
- Small models (SmolLM-135M)
- Have available GPU
- Frequent iterations

**Use Remote Backend When**:
- Production training
- Large models (>3B)
- Need powerful GPUs (A100, H100)
- Infrequent training runs

---

### 5. Monitoring & Debugging

**Training Logs**:
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)
```

**Metrics to Monitor**:
- Train loss (should decrease)
- Eval loss (should decrease, watch for overfitting)
- DPO reward accuracy (should increase)
- DPO reward margin (should increase)

**Common Issues**:

| Issue | Cause | Solution |
|-------|-------|----------|
| Empty dataset error | No approved annotations | Generate synthetic or collect more data |
| GPU OOM (DPO) | Model too large | Use smaller model or remote backend |
| LoRA failure | Unsupported architecture | Check logs, training continues with full fine-tuning |
| Low accuracy | Insufficient data | Need more examples or better quality |
| Overfitting | Too many epochs | Reduce epochs or use validation split |

---

### 6. Adapter Management

**Adapter Naming**:
- Timestamped by default: `dpo_routing_20251124_143052/`
- Prevents overwriting previous adapters
- Enables A/B testing

**Adapter Storage**:
```
outputs/adapters/
├── sft_routing_20251124_120000/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer_config.json
├── dpo_routing_20251124_143052/
│   └── ... (same structure)
└── embedding_video_20251124_150000/
    └── ... (same structure)
```

**Loading Adapters**:
```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
model = PeftModel.from_pretrained(base_model, adapter_path)
```

---

### 7. Production Deployment

**Testing Checklist**:
- [ ] Validate with small test dataset (3-5 examples)
- [ ] Check GPU memory usage
- [ ] Verify adapter loads correctly
- [ ] Test inference performance
- [ ] A/B test against base model

**Deployment Strategy**:
1. Train adapter on production data
2. Validate adapter quality offline
3. Deploy adapter to staging
4. A/B test vs current model
5. Gradual rollout to production

---

### 8. Cost Optimization

**Local Training**:
- No cloud costs
- Use for small models and testing

**Remote Training**:
- Modal charges per GPU-hour
- Choose smallest GPU that fits model
- Use timeout to prevent runaway costs
- Batch multiple training runs

**GPU Selection Guide**:
| Model Size | Recommended GPU | Cost/Hour (Modal) |
|------------|-----------------|-------------------|
| <1B | T4 | ~$0.50 |
| 1-3B | A10G | ~$1.10 |
| 3-8B | A100-40GB | ~$2.21 |
| 8B+ | A100-80GB | ~$3.67 |

---

## Error Messages & Troubleshooting

### Common Errors

**1. Empty Dataset**:
```
ValueError: Cannot train with empty dataset. No training examples available after formatting.
```
**Solution**: Ensure you have approved annotations in Phoenix.

---

**2. Missing Fields**:
```
ValueError: Invalid DPO dataset at index 5: missing required fields ['chosen', 'rejected'].
Expected fields: ['prompt', 'chosen', 'rejected'], got: ['prompt', 'text']
```
**Solution**: Check dataset formatting logic.

---

**3. Insufficient Data**:
```
ValueError: Insufficient preference pairs: 15 < 20. Need spans with both approved and rejected annotations.
```
**Solution**: Generate synthetic data or collect more annotations.

---

**4. GPU Out of Memory**:
```
RuntimeError: CUDA out of memory. Tried to allocate 12.00 GiB
```
**Solution**: Use smaller model, reduce batch size, or use remote backend with larger GPU.

---

**5. LoRA Application Failure**:
```
WARNING: Failed to apply LoRA (continuing with full fine-tuning): KeyError: 'q_proj'
```
**Impact**: Training continues with full fine-tuning (higher memory usage).
**Solution**: Check if model architecture supports LoRA target modules.

---

## Integration with Other Modules

### Phoenix Telemetry

Fine-tuning extracts data directly from Phoenix:

```python
# Phoenix provides TelemetryProvider interface
provider = PhoenixProvider(tenant_id, project)

# Fine-tuning uses provider to query spans and annotations
spans = await provider.traces.get_spans(project)
annotations = await provider.annotations.get_annotations(spans_df, project)
```

---

### Synthetic Data Service

When insufficient data available:

```python
# Fine-tuning triggers synthetic generation
synthetic_svc.generate(SyntheticDataRequest(
    optimizer="routing",
    count=50
))

# Sends through approval workflow (mandatory)
approval_orch.submit_for_review(batch)
```

---

### Approval Workflow

All synthetic data requires human approval:

```python
# Create approval batch
batch = ApprovalBatch(
    batch_id="synthetic_routing_...",
    items=[ReviewItem(...) for item in synthetic_data]
)

# Submit for review
approved_batch = await approval_orchestrator.submit_for_review(batch)

# Only approved items used for training
```

---

## Package Structure

```
libs/finetuning/
├── cogniverse_finetuning/
│   ├── __init__.py              # Public API
│   ├── orchestrator.py          # Main orchestrator
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── trace_converter.py       # SFT extraction
│   │   ├── preference_extractor.py  # DPO extraction
│   │   ├── embedding_extractor.py   # Triplet extraction
│   │   ├── method_selector.py       # Auto-selection
│   │   ├── formatters.py            # Format conversion
│   │   └── utils.py                 # Utilities
│   └── training/
│       ├── __init__.py
│       ├── backend.py               # Backend abstraction
│       ├── sft_trainer.py           # SFT training
│       ├── dpo_trainer.py           # DPO training
│       ├── embedding_finetuner.py   # Embedding training
│       ├── modal_runner.py          # Modal integration
│       └── modal_app.py             # Modal deployment
├── tests/
│   └── ... (test files)
├── pyproject.toml
└── README.md
```

---

## Dependencies

**Core**:
- `transformers` - HuggingFace model loading
- `trl` - Supervised fine-tuning and DPO
- `peft` - LoRA adapters
- `torch` - Deep learning framework
- `datasets` - Dataset management

**Optional**:
- `sentence-transformers` - Embedding fine-tuning
- `modal` - Remote GPU training

**Integration**:
- `cogniverse-foundation` - Telemetry provider interface
- `cogniverse-telemetry-phoenix` - Phoenix implementation
- `cogniverse-agents` - Approval workflow
- `cogniverse-synthetic` - Synthetic data generation

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-11-24 | Initial release with SFT, DPO, embedding fine-tuning |

---

## References

### Papers

- **DPO**: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023)
- **LoRA**: [Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
- **Triplet Loss**: [FaceNet](https://arxiv.org/abs/1503.03832) (Schroff et al., 2015)

### Libraries

- [HuggingFace TRL](https://github.com/huggingface/trl) - Transformer Reinforcement Learning
- [HuggingFace PEFT](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning
- [sentence-transformers](https://www.sbert.net/) - Embedding models

### Related Documentation

- [Phoenix Telemetry](telemetry.md) - Telemetry provider integration
- [Approval Workflow](approval-workflow.md) - Human-in-the-loop approval
- [Optimization](optimization.md) - DSPy module optimization
- [Implementation Plan](../plan/fine-tuning-implementation.md) - Future enhancements
