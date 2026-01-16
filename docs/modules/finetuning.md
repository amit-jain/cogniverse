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

#### Trajectory Extraction (`trace_converter.py`)

**Purpose**: Extract multi-turn conversation trajectories from Phoenix sessions for fine-tuning.

**Data Structures**:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    turn_id: int
    query: str
    response: str
    timestamp: datetime
    span_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationTrajectory:
    """A complete conversation session."""
    session_id: str
    turns: List[ConversationTurn]
    session_outcome: Optional[str] = None  # "success", "partial", "failure"
    session_score: Optional[float] = None  # 0.0-1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrajectoryDataset:
    """Collection of conversation trajectories."""
    trajectories: List[ConversationTrajectory]
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**TraceToTrajectoryConverter**:

```python
from cogniverse_finetuning.dataset.trace_converter import TraceToTrajectoryConverter

converter = TraceToTrajectoryConverter(telemetry_provider)

# Extract trajectories from Phoenix
dataset = await converter.extract_trajectories(
    project="cogniverse-tenant1",
    min_turns_per_session=2,      # Minimum turns for multi-turn
    include_annotations=True,     # Include session annotations
    start_time=datetime.now() - timedelta(days=7)
)

print(f"Extracted {len(dataset.trajectories)} trajectories")
for traj in dataset.trajectories[:3]:
    print(f"Session {traj.session_id}: {len(traj.turns)} turns, outcome={traj.session_outcome}")
```

**Conversion Process**:
1. Query Phoenix for spans with `session.id` attribute
2. Group spans by `session_id`
3. Order turns chronologically within each session
4. Filter sessions by `min_turns_per_session`
5. Attach session-level annotations (outcome, score)
6. Return `TrajectoryDataset`

**Export Formats**:

```python
# Save to JSONL
dataset.save_jsonl("trajectories.jsonl")

# Save to Parquet
dataset.save_parquet("trajectories.parquet")

# Load from file
loaded = TrajectoryDataset.load_jsonl("trajectories.jsonl")
```

**JSONL Format**:
```json
{"session_id": "sess_123", "turns": [{"turn_id": 0, "query": "...", "response": "...", "timestamp": "..."}], "session_outcome": "success", "session_score": 0.9}
{"session_id": "sess_456", "turns": [...], "session_outcome": "partial", "session_score": 0.6}
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

### 5. Adapter Registry (`registry/`)

**Purpose**: Manage trained LoRA adapters with versioning, activation, and deployment capabilities.

The Adapter Registry provides complete lifecycle management for trained adapters, storing metadata in Vespa for multi-tenant support and enabling seamless integration with inference systems.

#### Key Components

| Component | File | Description |
|-----------|------|-------------|
| `AdapterRegistry` | `adapter_registry.py` | Main interface for adapter lifecycle management |
| `AdapterMetadata` | `models.py` | Data model for adapter metadata |
| `VespaAdapterStore` | `libs/vespa/.../adapter_store.py` | Vespa-backed storage for adapter metadata |
| `LocalStorage` | `storage.py` | Local filesystem storage backend |
| `HuggingFaceStorage` | `storage.py` | HuggingFace Hub storage backend |
| Inference Helpers | `inference.py` | vLLM integration utilities |

#### Core Features

**Registration & Versioning**:
```python
from cogniverse_finetuning.registry import AdapterRegistry, AdapterMetadata

registry = AdapterRegistry()

# Register a trained adapter
adapter_id = registry.register_adapter(
    tenant_id="acme_corp",
    name="routing_sft",
    version="1.0.0",
    base_model="SmolLM-135M",
    model_type="llm",
    training_method="sft",
    adapter_path="outputs/adapters/sft_routing_20251116/",
    agent_type="routing",
    metrics={"eval_loss": 0.15, "accuracy": 0.92}
)
```

**Activation & Deployment**:
```python
# Activate adapter for use by agents
registry.activate_adapter(adapter_id)

# Get active adapter for a specific agent type
active = registry.get_active_adapter("acme_corp", "routing")
print(f"Active adapter: {active.adapter_path}")
```

**Storage Backends**:
```python
from cogniverse_finetuning.registry import upload_adapter, download_adapter

# Upload to HuggingFace Hub
upload_adapter(
    local_path="outputs/adapters/sft_routing/",
    destination_uri="hf://myorg/my-adapter",
    hf_token="hf_xxx"
)

# Download from HuggingFace Hub
local_path = download_adapter(
    source_uri="hf://myorg/my-adapter",
    local_dir="/tmp/adapters/"
)
```

**Inference Integration** (vLLM):
```python
from cogniverse_finetuning.registry import (
    get_active_adapter_for_inference,
    resolve_adapter_path
)

# Get active adapter info for vLLM
adapter_info = get_active_adapter_for_inference("acme_corp", "routing")
if adapter_info:
    # Configure vLLM with the adapter
    engine = LLMEngine(
        model=adapter_info.base_model,
        enable_lora=True,
        lora_modules=[LoRARequest(
            lora_name=adapter_info.name,
            lora_path=resolve_adapter_path(adapter_info.adapter_uri)
        )]
    )
```

#### Orchestrator Integration

When `enable_registry=True`, the orchestrator automatically registers adapters after training:

```python
config = OrchestrationConfig(
    tenant_id="acme_corp",
    agent_type="routing",
    enable_registry=True,           # Auto-register after training
    adapter_version="1.0.0",        # Version for registered adapter
    adapter_storage_uri="hf://org/repo"  # Optional: upload to HuggingFace
)

result = orchestrator.train_sft(config)
print(f"Registered adapter: {result.adapter_id}")
print(f"Storage URI: {result.adapter_uri}")
```

#### Agent Integration

Agents can use `AdapterAwareMixin` to automatically load fine-tuned adapters:

```python
from cogniverse_agents import AdapterAwareMixin, get_active_adapter_path

class RoutingAgent(AdapterAwareMixin):
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        adapter_path = self.load_adapter_if_available("routing")
        if adapter_path:
            self.model = self._load_with_adapter(adapter_path)
```

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

### Multi-Turn Fine-Tuning (Planned)

**Status**: Phase 5 - Pending Implementation

Multi-turn fine-tuning will enable training on conversation trajectories extracted from Phoenix sessions.

**Planned Features**:
- Extract trajectories using `TraceToTrajectoryConverter`
- Format for multi-turn SFT (each turn includes conversation history as context)
- Support `multi_turn: bool` and `min_turns_per_session: int` in `OrchestrationConfig`
- Train using existing SFT backend with conversation-aware formatting

**Planned Configuration**:
```python
@dataclass
class OrchestrationConfig:
    # ... existing fields ...
    multi_turn: bool = False
    min_turns_per_session: int = 2
```

**Planned Usage**:
```python
result = await finetune(
    telemetry_provider=provider,
    tenant_id="tenant1",
    project="cogniverse-tenant1",
    model_type="llm",
    agent_type="routing",
    multi_turn=True,
    min_turns_per_session=2,
    backend="local"
)
```

**Future Enhancements**:
- **DiaTool-DPO**: Trajectory-level DPO when algorithm is released
- **GRPO with SkyRL**: Online RL for multi-turn agents
- **Turn Attribution**: Per-turn quality scoring and credit assignment
- **Reward Models**: Session-level reward prediction from history

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

## Phoenix Experiment Tracking

### Overview

All fine-tuning experiments are automatically logged to Phoenix as EXPERIMENT spans. This provides a unified system for telemetry, evaluation, AND experiment tracking—no MLflow needed.

### Experiment Span Schema

Each training run creates an EXPERIMENT span with the following attributes:

**Span Identification**:
- `openinference.span.kind`: "EXPERIMENT"
- `operation.name`: "fine_tuning"
- `experiment.run_id`: Unique run ID (e.g., "run_2025-11-24T10:00:00")
- `experiment.agent_type`: Agent type or modality (e.g., "routing", "video")

**Hyperparameters** (`params.*`):
- `params.base_model`: Base model name (e.g., "HuggingFaceTB/SmolLM-135M")
- `params.method`: Training method ("sft", "dpo", "embedding")
- `params.backend`: Backend type ("local" or "remote")
- `params.backend_provider`: Provider for remote backend ("modal", "sagemaker", etc.)
- `params.epochs`: Number of training epochs
- `params.batch_size`: Batch size
- `params.learning_rate`: Learning rate
- `params.use_lora`: Whether LoRA was used
- `params.lora_r`: LoRA rank (default: 8)
- `params.lora_alpha`: LoRA alpha (default: 16)

**Dataset Info** (`data.*`):
- `data.total_spans`: Total spans in Phoenix
- `data.approved_count`: Number of approved annotations
- `data.rejected_count`: Number of rejected annotations
- `data.preference_pairs`: Number of preference pairs (for DPO)
- `data.dataset_size`: Final training dataset size
- `data.used_synthetic`: Whether synthetic data was used
- `data.synthetic_approved_count`: Number of approved synthetic examples

**Results** (`metrics.*`):
- `metrics.train_loss`: Final training loss
- `metrics.train_samples`: Number of training samples processed
- `metrics.epochs_completed`: Actual epochs completed

**Output** (`output.*`):
- `output.adapter_path`: Path to saved adapter

### Querying Experiments

Use the helper functions to query experiments from Python:

```python
from cogniverse_finetuning.orchestrator import (
    list_experiments,
    get_experiment_details,
    compare_experiments
)

# List all routing experiments
experiments = await list_experiments(
    telemetry_provider=provider,
    project="cogniverse-tenant1",
    agent_type="routing",
    method="sft",  # Optional: filter by method
    limit=50
)

# Get specific experiment details
details = await get_experiment_details(
    telemetry_provider=provider,
    project="cogniverse-tenant1",
    run_id="run_2025-11-24T10:00:00"
)

# Compare multiple experiments
comparison = await compare_experiments(
    telemetry_provider=provider,
    project="cogniverse-tenant1",
    run_ids=["run_001", "run_002", "run_003"]
)
```

### Dashboard Integration

The Phoenix dashboard includes a **Fine-Tuning** tab (Monitoring > Fine-Tuning) that provides:

- **Experiment History Table**: All training runs with hyperparameters and metrics
- **Summary Metrics**: Total runs, SFT/DPO counts, best loss
- **Experiment Details**: View hyperparameters, dataset info, and output paths
- **Side-by-Side Comparison**: Compare multiple experiments with loss charts
- **Filtering**: Filter by agent type, modality, or training method

Access the dashboard at `http://localhost:8501` (default Streamlit port).

---

## Automatic Adapter Evaluation

### Overview

After training completes, adapters are automatically evaluated against the base model on a held-out test set. This provides objective metrics on adapter quality and improvement over baseline.

### Evaluation Metrics

**Accuracy Metrics**:
- `accuracy`: Percentage of correct predictions (0-1)
- `top_k_accuracy`: Correct prediction in top-k results

**Confidence Metrics**:
- `avg_confidence`: Average confidence score (0-1)
- `confidence_calibration`: How well confidence matches accuracy

**Error Metrics**:
- `error_rate`: Percentage of incorrect predictions (0-1)
- `hallucination_rate`: Predictions not in valid output space

**Latency**:
- `avg_latency_ms`: Average inference latency in milliseconds
- `latency_overhead`: Additional latency from adapter (ms)

### Configuration

Enable/disable evaluation in `OrchestrationConfig`:

```python
config = OrchestrationConfig(
    tenant_id="tenant1",
    project="cogniverse-tenant1",
    model_type="llm",
    agent_type="routing",
    evaluate_after_training=True,  # Enable auto-evaluation
    test_set_size=50,  # Number of test examples
    ...
)
```

### Test Set Creation

Evaluation uses a time-based split to ensure test data is NOT in the training set:
- **Training Data**: Older annotations (used for training)
- **Test Data**: Recent annotations (last 7 days)
- **Size**: Configurable (default: 50 examples)
- **Sampling**: Random sample from recent data

### Evaluation Process

1. **Create Test Set**: Extract recent examples from telemetry (last 7 days)
2. **Load Base Model**: Load the base model without adapter
3. **Evaluate Base**: Run base model on test set, compute metrics
4. **Load Adapter**: Load the trained LoRA adapter
5. **Evaluate Adapter**: Run adapter model on test set, compute metrics
6. **Compare**: Calculate improvements and statistical significance

### Evaluation Span Schema

Each evaluation creates an EVALUATION span with the following attributes:

**Span Identification**:
- `openinference.span.kind`: "EVALUATION"
- `operation.name`: "adapter_evaluation"
- `evaluation.adapter_path`: Path to evaluated adapter
- `evaluation.agent_type`: Agent type (e.g., "routing")
- `evaluation.test_size`: Number of test examples

**Base Model Metrics** (`metrics.base.*`):
- `metrics.base.accuracy`: Base model accuracy
- `metrics.base.confidence`: Base model average confidence
- `metrics.base.error_rate`: Base model error rate
- `metrics.base.hallucination_rate`: Base model hallucination rate
- `metrics.base.latency_ms`: Base model average latency

**Adapter Metrics** (`metrics.adapter.*`):
- `metrics.adapter.accuracy`: Adapter accuracy
- `metrics.adapter.confidence`: Adapter average confidence
- `metrics.adapter.error_rate`: Adapter error rate
- `metrics.adapter.hallucination_rate`: Adapter hallucination rate
- `metrics.adapter.latency_ms`: Adapter average latency

**Improvements** (`improvement.*`):
- `improvement.accuracy`: Accuracy improvement (e.g., 0.18 = 18% improvement)
- `improvement.confidence`: Confidence improvement
- `improvement.error_reduction`: Error rate reduction
- `improvement.latency_overhead`: Additional latency from adapter (ms)
- `improvement.significant`: Statistical significance (boolean)
- `improvement.p_value`: P-value for significance test

### Statistical Significance

Improvements are tested for statistical significance:
- **Threshold**: Accuracy improvement >5%
- **p-value**: Computed for significance test
- **Result**: `improvement.significant` = true if p < 0.05

### Dashboard Integration

Evaluation results are displayed in the Fine-Tuning dashboard:

**Experiment Details View**:
- **Base Model Metrics**: Accuracy, confidence, error rate, hallucination rate, latency
- **Adapter Model Metrics**: Same metrics for adapter
- **Improvements**: Side-by-side comparison with delta indicators
- **Statistical Significance**: Visual indicator (✅/ℹ️) with p-value

**Example**:
```
Base Model Metrics          Adapter Model Metrics
Accuracy: 65.0%            Accuracy: 83.0%
Confidence: 72.0%          Confidence: 88.0%
Error Rate: 35.0%          Error Rate: 17.0%
Hallucination Rate: 8.0%   Hallucination Rate: 2.0%
Latency: 145.3 ms          Latency: 152.1 ms

Improvements
Accuracy Δ: +18.0% ↑
Confidence Δ: +16.0% ↑
Error Reduction: 18.0% ↑
Latency Overhead: 6.8 ms ↓

✅ Statistically significant improvement (p=0.01)
```

### Usage Example

```python
from cogniverse_finetuning import finetune

# Run fine-tuning with automatic evaluation
result = await finetune(
    telemetry_provider=provider,
    tenant_id="tenant1",
    project="cogniverse-tenant1",
    model_type="llm",
    agent_type="routing",
    base_model="HuggingFaceTB/SmolLM-135M",
    backend="local",
    evaluate_after_training=True,  # Enable evaluation
    test_set_size=50  # Number of test examples
)

# Access evaluation results
if result.evaluation_result:
    print(f"Accuracy improvement: {result.evaluation_result.accuracy_improvement:.2%}")
    print(f"Base accuracy: {result.evaluation_result.base_metrics.accuracy:.2%}")
    print(f"Adapter accuracy: {result.evaluation_result.adapter_metrics.accuracy:.2%}")
    print(f"Significant: {result.evaluation_result.improvement_significant}")
```

### Supported Agent Types

Evaluation currently supports:
- **routing**: Agent routing decisions
- **profile_selection**: Backend profile selection
- **entity_extraction**: Not yet implemented (falls back to exact match)

### Error Handling

Evaluation failures do NOT fail the training pipeline:
- **No Test Data**: Logs warning, continues without evaluation
- **Model Load Failure**: Logs error, continues without evaluation
- **Inference Errors**: Logs error, continues without evaluation

This ensures training can complete even if evaluation fails.

### Best Practices

**Test Set Quality**:
- ✅ Ensure recent telemetry data available (last 7 days)
- ✅ Use realistic test set size (50-100 examples)
- ✅ Review test set coverage (different query types)

**Interpreting Results**:
- ✅ Focus on statistically significant improvements
- ✅ Consider latency overhead vs accuracy gains
- ✅ Compare multiple training runs
- ❌ Don't over-interpret single-run results
- ❌ Don't ignore high hallucination rates

**When to Retrain**:
- Accuracy improvement <5%: May need more data or better hyperparameters
- High hallucination rate (>10%): Check dataset quality
- Large latency overhead (>50ms): Consider distillation or quantization

---

## Validation Split & Early Stopping

### Overview

For larger datasets (>100 examples/pairs), training automatically uses a 90/10 train/validation split with early stopping to prevent overfitting and improve generalization.

### Automatic Validation Split

**Trigger Condition**:
- Dataset size > 100 examples (SFT) or pairs (DPO)

**Split Ratio**:
- 90% training data
- 10% validation data

**Example**:
```
Dataset: 150 examples
→ Train: 135 examples (90%)
→ Val: 15 examples (10%)
```

### Early Stopping

**Configuration**:
- **Metric**: Validation loss (`eval_loss`)
- **Patience**: 3 evaluations
- **Threshold**: 0.0 (any improvement counts)
- **Goal**: Lower is better

**How It Works**:
1. Model evaluates on validation set every 500 steps (default `eval_steps`)
2. If `eval_loss` doesn't improve for 3 consecutive evaluations, training stops
3. Best checkpoint (lowest `eval_loss`) is loaded at end

**Benefits**:
- Prevents overfitting on training data
- Saves compute time (stops when no longer improving)
- Automatically selects best model checkpoint

### Validation Metrics

**Training Metrics** (logged to Phoenix):
- `metrics.train_loss`: Final training loss
- `metrics.train_samples`: Number of training samples
- `metrics.train_examples`: Number of training examples/pairs

**Validation Metrics** (logged to Phoenix):
- `metrics.used_validation_split`: Boolean (true if validation used)
- `metrics.eval_loss`: Validation loss
- `metrics.eval_samples`: Number of validation samples
- `metrics.val_examples`: Number of validation examples/pairs

**DPO-Specific Validation Metrics**:
- `metrics.eval_reward_accuracy`: Percentage of times chosen > rejected
- `metrics.eval_reward_margin`: Average reward margin (chosen - rejected)

### Dashboard Display

**Experiment Details → Validation Metrics Section**:
- **Train Examples**: Number of training examples
- **Val Examples**: Number of validation examples
- **Val Loss**: Final validation loss
- **Overfit**: Percentage difference between val loss and train loss
  - Positive (red ↑): Model is overfitting
  - Negative (green ↓): Model is generalizing well
- **Early Stopping Indicator**: "✅ Validation split used with early stopping (patience=3)"

**DPO Experiments Also Show**:
- **Reward Accuracy**: Percentage of preference pairs correctly ranked
- **Reward Margin**: Average difference in rewards between chosen and rejected

### Example Output

```
Training HuggingFaceTB/SmolLM-135M with 120 examples...
Split dataset: 108 train, 12 validation examples
Early stopping enabled (patience=3)
Starting training...
[Epoch 1] train_loss=0.8234, eval_loss=0.7891
[Epoch 2] train_loss=0.6543, eval_loss=0.7234
[Epoch 3] train_loss=0.5123, eval_loss=0.7456  # Validation loss increased
[Epoch 4] train_loss=0.4234, eval_loss=0.7567  # No improvement for 3 evals
Early stopping triggered at epoch 4
Loading best model (eval_loss=0.7234 from epoch 2)
Training complete. Adapter saved to outputs/sft_adapters
Validation loss: 0.7234
```

### Configuration

Validation split and early stopping are **automatic** and cannot be disabled when dataset size > 100. This is by design to ensure high-quality adapters.

**Customization** (advanced users can modify trainer code):
- `eval_steps`: Frequency of validation evaluation (default: 500)
- `early_stopping_patience`: Number of evaluations to wait (default: 3)
- `metric_for_best_model`: Metric to optimize (default: "eval_loss")

### Best Practices

**Interpreting Validation Metrics**:
- ✅ `eval_loss` < `train_loss`: Model generalizing well
- ⚠️ `eval_loss` ≈ `train_loss`: Model learning appropriately
- ❌ `eval_loss` >> `train_loss`: Model overfitting (>10% difference)

**When Overfitting Occurs**:
- Reduce `epochs` (try 2 instead of 3)
- Increase `batch_size` (try 8 instead of 4)
- Reduce `learning_rate` (try 1e-4 instead of 2e-4)
- Add more training data

**When Validation Loss is High**:
- Increase training data quality (review annotations)
- Check for data distribution mismatch (train vs val)
- Consider using DPO if you have preference data
- Increase model capacity (try larger base model)

### Comparison with No Validation

| Feature | No Validation (<100 examples) | With Validation (>100 examples) |
|---------|-------------------------------|----------------------------------|
| Training Data | All data | 90% of data |
| Validation Data | None | 10% of data |
| Early Stopping | No | Yes (patience=3) |
| Best Checkpoint | Last checkpoint | Best checkpoint by val loss |
| Overfitting Risk | Higher | Lower |
| Compute Time | May be longer | Shorter (stops early) |
| Metrics Logged | `train_loss` only | `train_loss` + `eval_loss` |

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
