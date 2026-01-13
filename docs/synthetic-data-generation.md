# Synthetic Data Generation System

**Package**: `cogniverse-synthetic` (Implementation Layer)
**Location**: `/home/user/cogniverse/packages/cogniverse-synthetic`

The synthetic data generation system creates high-quality training examples for all Cogniverse optimizers by sampling real content from backend storage and generating realistic queries using DSPy-driven LLM modules with validation.

## Overview

The system extends DSPy optimization to all routing and orchestration components by automatically generating diverse, realistic training datasets. It uses agent-based profile selection and backend schema introspection to create synthetic examples that match production patterns.

### Supported Optimizers

1. **ModalityOptimizer** - Per-modality routing (VIDEO, DOCUMENT, IMAGE, AUDIO)
2. **CrossModalOptimizer** - Multi-modal fusion decisions
3. **RoutingOptimizer** - Entity-based advanced routing
4. **WorkflowOptimizer** - Multi-agent workflow orchestration
5. **UnifiedOptimizer** - Combined routing and workflow planning

## Architecture

### System Overview

```mermaid
graph TB
    subgraph "Synthetic Data Generation Service"
        Service[SyntheticDataService<br/>Main Orchestrator]

        Service --> ProfileSelector
        Service --> BackendQuerier
        Service --> Generators

        ProfileSelector[ProfileSelector<br/>LLM or Rule-based]
        BackendQuerier[BackendQuerier<br/>Backend Sampling]

        subgraph "Generators"
            ModalityGen[ModalityGenerator]
            CrossModalGen[CrossModalGenerator]
            RoutingGen[RoutingGenerator]
            WorkflowGen[WorkflowGenerator]
        end

        subgraph "Utilities"
            PatternExtractor[PatternExtractor<br/>Topics, Entities]
            AgentInferrer[AgentInferrer<br/>Agent Mapping]
        end

        Generators --> PatternExtractor
        Generators --> AgentInferrer
    end

    subgraph "External Systems"
        Backend[(Backend<br/>Vespa/Other)]
        LLM[LLM Client<br/>Optional]
        BackendConfig[Backend Config<br/>Profiles]
    end

    ProfileSelector -.-> LLM
    ProfileSelector --> BackendConfig
    BackendQuerier --> Backend

    subgraph "Data Flow"
        Request[SyntheticDataRequest] --> Service
        Service --> Response[SyntheticDataResponse<br/>Generated Examples]
    end

    style Service fill:#4A90E2,stroke:#2E5C8A,color:#fff
    style Generators fill:#7ED321,stroke:#5FA319,color:#000
    style Utilities fill:#F5A623,stroke:#C4841D,color:#000
    style Backend fill:#BD10E0,stroke:#8B0CA6,color:#fff
```

### Generation Pipeline

```mermaid
sequenceDiagram
    participant API as REST API / Python
    participant Service as SyntheticDataService
    participant PS as ProfileSelector
    participant BQ as BackendQuerier
    participant Gen as Generator
    participant Utils as PatternExtractor
    participant Backend as Backend DB
    participant DSPy as DSPy Modules

    API->>Service: generate(request)

    Note over Service: Step 1: Profile Selection
    Service->>PS: select_profiles(optimizer, config)
    PS->>PS: LLM reasoning or<br/>rule-based scoring
    PS-->>Service: [profiles, reasoning]

    Note over Service: Step 2: Content Sampling
    Service->>BQ: query_profiles(profiles, strategy)
    BQ->>Backend: Query via Backend interface
    Backend-->>BQ: sampled documents
    BQ-->>Service: sampled_content

    Note over Service: Step 3: Data Generation
    Service->>Gen: generate(content, count)
    Gen->>Utils: extract_topics(content)
    Gen->>Utils: extract_entities(content)
    Gen->>Utils: infer_agents(characteristics)
    Utils-->>Gen: patterns
    Gen->>DSPy: Generate queries with<br/>validation & reasoning
    DSPy-->>Gen: validated queries
    Gen-->>Service: List[BaseModel]

    Note over Service: Step 4: Response
    Service->>Service: build response with<br/>metadata
    Service-->>API: SyntheticDataResponse
```

### Core Components

#### 1. Registry (`registry.py`)
Central configuration mapping optimizers to generators and schemas:

```python
from cogniverse_synthetic import OPTIMIZER_REGISTRY, get_optimizer_config

# Get optimizer configuration
config = get_optimizer_config("cross_modal")
print(config.schema_class)  # FusionHistorySchema
print(config.backend_query_strategy)  # "cross_modal_pairs"
```

#### 2. Schemas (`schemas.py`)
Pydantic models for all optimizer training data:

- `ModalityExampleSchema` - Modality routing examples
- `FusionHistorySchema` - Cross-modal fusion results
- `RoutingExperienceSchema` - Entity-based routing
- `WorkflowExecutionSchema` - Workflow execution patterns
- `SyntheticDataRequest` / `SyntheticDataResponse` - API contracts

#### 3. ProfileSelector (`profile_selector.py`)
Selects optimal backend profiles for data generation:

```python
from cogniverse_synthetic.profile_selector import ProfileSelector

selector = ProfileSelector(llm_client=llm)  # or None for rule-based
profiles, reasoning = await selector.select_profiles(
    optimizer_name="modality",
    optimizer_task="Per-modality routing optimization",
    available_profiles=backend_config["video_processing_profiles"],
    max_profiles=3
)
# Returns: (["video_colpali_smol500_mv_frame", ...], "reasoning...")
```

**Selection Strategies**:
- **LLM-based**: Uses reasoning to match profile characteristics to optimizer needs
- **Rule-based**: Heuristic scoring with diversity selection (fallback)

#### 4. BackendQuerier (`backend_querier.py`)
Samples content from backend storage (Vespa or other) using Backend interface:

```python
from cogniverse_synthetic.backend_querier import BackendQuerier
from cogniverse_retrieval.vespa_backend import VespaBackend

# Initialize with backend instance
backend = VespaBackend(config=backend_config)
querier = BackendQuerier(backend=backend)

samples = await querier.query_profiles(
    profile_configs=[{"profile_name": "video_colpali_smol500_mv_frame"}],
    sample_size=200,
    strategy="diverse"  # or "entity_rich", "temporal_recent", etc.
)
```

**Sampling Strategies**:
- `diverse` - Random sampling across all content
- `temporal_recent` - Recent content (time-based)
- `entity_rich` - Content with many named entities
- `multi_modal_sequences` - Content from different modalities
- `by_modality` - Specific modality filtering
- `cross_modal_pairs` - Paired content from different modalities

**Backend Abstraction**: Uses `Backend` interface to support any vector database (Vespa, Pinecone, Weaviate, etc.)

#### 5. Generators (`generators/`)
Four concrete generators implementing the `BaseGenerator` interface:

**ModalityGenerator** (`generators/modality.py`):
```python
# Uses DSPy modules to generate modality-specific queries
# Example DSPy-generated queries:
# "show me TensorFlow videos"
# "find machine learning documents"

from cogniverse_sdk.config.types import OptimizerGenerationConfig, DSPyModuleConfig

modality_config = OptimizerGenerationConfig(
    optimizer_type="modality",
    dspy_modules={
        "query_generator": DSPyModuleConfig(
            signature_class="cogniverse_synthetic.dspy_signatures.GenerateModalityQuery",
            module_type="ChainOfThought"
        )
    }
)

modality_gen = ModalityGenerator(optimizer_config=modality_config)
examples = await modality_gen.generate(
    sampled_content=backend_samples,
    target_count=100,
    modality="VIDEO"
)
# Returns: List[ModalityExampleSchema]
```

**CrossModalGenerator** (`generators/cross_modal.py`):
```python
# Generates fusion scenarios:
# Primary: VIDEO results, Secondary: DOCUMENT results
# fusion_context: {"agreement": 0.8, "ambiguity": 0.2, ...}

examples = await cross_modal_gen.generate(
    sampled_content=backend_samples,
    target_count=100
)
# Returns: List[FusionHistorySchema]
```

**RoutingGenerator** (`generators/routing.py`):
```python
# Uses ValidatedEntityQueryGenerator with ChainOfThought and retry logic
# Generates entity-annotated queries guaranteed to contain entities:
# Query: "find TensorFlow object detection tutorial"
# Enhanced: "find TensorFlow(TECHNOLOGY) object detection tutorial"
# entities: [{"text": "TensorFlow", "type": "TECHNOLOGY"}]

from cogniverse_sdk.config.types import OptimizerGenerationConfig, DSPyModuleConfig

routing_config = OptimizerGenerationConfig(
    optimizer_type="routing",
    dspy_modules={
        "query_generator": DSPyModuleConfig(
            signature_class="cogniverse_synthetic.dspy_signatures.GenerateEntityQuery",
            module_type="ChainOfThought"  # LLM reasons about which entities to include
        )
    }
)

routing_gen = RoutingGenerator(optimizer_config=routing_config)
examples = await routing_gen.generate(
    sampled_content=backend_samples,
    target_count=100
)
# Returns: List[RoutingExperienceSchema]
# Each example validated to contain at least one entity (3 retry attempts)
```

**WorkflowGenerator** (`generators/workflow.py`):
```python
# Generates workflow patterns:
# Simple: ["video_search_agent"]
# Complex: ["video_search_agent", "summarizer", "detailed_report"]

examples = await workflow_gen.generate(
    sampled_content=backend_samples,
    target_count=100
)
# Returns: List[WorkflowExecutionSchema]
```

#### 6. Utilities (`utils/`)

**PatternExtractor** (`utils/pattern_extraction.py`):
- Extract topics (bigrams, trigrams)
- Extract entities (capitalized terms)
- Extract temporal patterns (years, dates)
- Extract content types (tutorial, guide, overview)
- Extract relationships (co-occurrence)

**AgentInferrer** (`utils/agent_inference.py`):
- Map modality → agent
- Infer agents from content characteristics
- Generate workflow sequences
- Validate agent sequences

#### 7. DSPy Signatures and Modules

**DSPy Signatures** (`dspy_signatures.py`):

Defines the interface between generators and LLMs for query generation. Signatures guide LLM behavior through field descriptions.

```python
class GenerateEntityQuery(dspy.Signature):
    """Generate search query that MUST include at least one of the provided entities"""

    topics: str = dspy.InputField(
        desc="Comma-separated topics from content"
    )
    entities: str = dspy.InputField(
        desc="Comma-separated named entities - YOUR QUERY MUST MENTION AT LEAST ONE OF THESE"
    )
    entity_types: str = dspy.InputField(
        desc="Comma-separated entity types (TECHNOLOGY, ORGANIZATION, CONCEPT)"
    )

    reasoning: str = dspy.OutputField(
        desc="Brief explanation of which entity/entities you're including and why"
    )
    query: str = dspy.OutputField(
        desc="Natural query that explicitly mentions at least one entity"
    )
```

**Available Signatures**:
- `GenerateModalityQuery` - Generate modality-specific queries
- `GenerateEntityQuery` - Generate entity-rich queries with reasoning
- `InferAgentFromModality` - Infer correct agent for modality/query

**DSPy Modules** (`dspy_modules.py`):

Validated query generators with built-in quality checks and retry logic.

```python
class ValidatedEntityQueryGenerator(dspy.Module):
    """
    Entity query generator with validation.
    Uses ChainOfThought for better quality - LLM reasons about which entities to include.
    Validates output and retries if needed (max 3 attempts).
    """

    def __init__(self, max_retries: int = 3):
        super().__init__()
        self.max_retries = max_retries
        self.generate = dspy.ChainOfThought(GenerateEntityQuery)

    def forward(self, topics: str, entities: str, entity_types: str) -> dspy.Prediction:
        # Retry loop with validation
        for attempt in range(self.max_retries):
            result = self.generate(topics=topics, entities=entities, entity_types=entity_types)

            # Validate: at least one entity must appear in query (case-insensitive)
            query_lower = result.query.lower()
            if any(entity.lower() in query_lower for entity in entity_list):
                return result  # Valid query found!

        # After max retries, raise error (no arbitrary fallbacks)
        raise ValueError(f"Failed to generate valid query after {self.max_retries} attempts")
```

**Key Features**:
- **ChainOfThought**: LLM reasons before generating (better quality)
- **Validation**: Ensures output meets requirements (e.g., entity presence)
- **Retry Logic**: Up to 3 attempts to generate valid output
- **No Fallbacks**: Raises exception if validation fails (no arbitrary defaults)
- **Optimization Ready**: Can be compiled with DSPy optimizers (BootstrapFewShot, MIPRO, etc.)

## Usage

### Python API

```python
from cogniverse_synthetic import SyntheticDataService
from cogniverse_synthetic.schemas import SyntheticDataRequest
from cogniverse_retrieval.vespa_backend import VespaBackend

# Initialize backend
backend = VespaBackend(config=backend_config)

# Initialize service
service = SyntheticDataService(
    backend=backend,            # Backend interface (Vespa, Pinecone, etc.)
    llm_client=llm_client,      # Optional for profile selection
)

# Generate training data
request = SyntheticDataRequest(
    optimizer="cross_modal",
    count=100,
    sample_size=200,            # Number of documents to sample from backend
    strategies=["diverse"],
    max_profiles=3,
    tenant_id="default"
)

response = await service.generate(request)

print(f"Generated {response.count} examples")
print(f"Used profiles: {response.selected_profiles}")
print(f"Reasoning: {response.profile_selection_reasoning}")

# Access generated data
for example in response.data:
    print(example["query"])
```

### REST API

```python
from fastapi import FastAPI
from cogniverse_synthetic import router, configure_service
from cogniverse_retrieval.vespa_backend import VespaBackend

app = FastAPI()

# Configure service (optional)
backend = VespaBackend(config=backend_config)
configure_service(backend=backend)

# Mount router
app.include_router(router)
```

**Endpoints**:

**POST /synthetic/generate**
```bash
curl -X POST http://localhost:8000/synthetic/generate \
  -H "Content-Type: application/json" \
  -d '{
    "optimizer": "modality",
    "count": 50,
    "vespa_sample_size": 100,
    "max_profiles": 2
  }'
```

**GET /synthetic/optimizers**
```bash
curl http://localhost:8000/synthetic/optimizers
# Returns: {"modality": "Per-modality routing...", ...}
```

**GET /synthetic/optimizers/{name}**
```bash
curl http://localhost:8000/synthetic/optimizers/cross_modal
# Returns: Detailed optimizer info with schema, generator, etc.
```

**GET /synthetic/health**
```bash
curl http://localhost:8000/synthetic/health
# Returns: {"status": "healthy", "generators": 4, ...}
```

**POST /synthetic/batch/generate**
```bash
curl -X POST "http://localhost:8000/synthetic/batch/generate?optimizer=routing&count_per_batch=100&num_batches=5"
# Generates 500 examples across 5 batches
```

## Integration with Optimizers

### ModalityOptimizer

```python
from src.app.routing.modality_optimizer import ModalityOptimizer
from cogniverse_synthetic import SyntheticDataService
from cogniverse_synthetic.schemas import SyntheticDataRequest

# Generate training data
service = SyntheticDataService()
request = SyntheticDataRequest(optimizer="modality", count=200)
response = await service.generate(request)

# Convert to ModalityExample objects
from src.app.routing.modality_optimizer import ModalityExample
examples = [ModalityExample(**ex) for ex in response.data]

# Train optimizer
optimizer = ModalityOptimizer()
optimizer.compile(trainset=examples)
```

### CrossModalOptimizer

```python
from src.app.routing.cross_modal_optimizer import CrossModalOptimizer
from cogniverse_synthetic import SyntheticDataService
from cogniverse_synthetic.schemas import SyntheticDataRequest

# Generate fusion training data
service = SyntheticDataService()
request = SyntheticDataRequest(optimizer="cross_modal", count=200)
response = await service.generate(request)

# Convert to FusionHistory objects
from src.app.routing.cross_modal_optimizer import FusionHistory
fusion_histories = [FusionHistory(**ex) for ex in response.data]

# Train optimizer
optimizer = CrossModalOptimizer()
optimizer.compile(trainset=fusion_histories)
```

### Advanced Routing

```python
from src.app.routing.advanced_optimizer import AdvancedOptimizer
from cogniverse_synthetic import SyntheticDataService
from cogniverse_synthetic.schemas import SyntheticDataRequest

# Generate entity-based routing data
service = SyntheticDataService()
request = SyntheticDataRequest(optimizer="routing", count=200)
response = await service.generate(request)

# Convert to RoutingExperience
from src.app.routing.advanced_optimizer import RoutingExperience
experiences = [RoutingExperience(**ex) for ex in response.data]

# Train optimizer
optimizer = AdvancedOptimizer()
optimizer.compile(trainset=experiences)
```

### Workflow Intelligence

```python
from src.app.agents.workflow_intelligence import WorkflowIntelligence
from cogniverse_synthetic import SyntheticDataService
from cogniverse_synthetic.schemas import SyntheticDataRequest

# Generate workflow execution patterns
service = SyntheticDataService()
request = SyntheticDataRequest(optimizer="workflow", count=200)
response = await service.generate(request)

# Convert to WorkflowExecution
from src.app.agents.workflow_intelligence import WorkflowExecution
executions = [WorkflowExecution(**ex) for ex in response.data]

# Train workflow optimizer
workflow_intel = WorkflowIntelligence()
workflow_intel.compile(trainset=executions)
```

## Configuration

### Backend Configuration

The service accepts a backend configuration dictionary with video processing profiles:

```python
backend_config = {
    "video_processing_profiles": {
        "video_colpali_smol500_mv_frame": {
            "model": "vidore/colpali-v1.2",
            "embedding_dimension": 128,
            "chunk_strategy": "frame",
            # ... other config
        },
        "video_videoprism_base_mv_chunk_30s": {
            "model": "google/videoprism-base",
            "embedding_dimension": 768,
            "chunk_strategy": "temporal",
            # ... other config
        }
    }
}

service = SyntheticDataService(backend_config=backend_config)
```

### Profile Selection

**With LLM** (better quality, slower):
```python
from openai import AsyncOpenAI

llm_client = AsyncOpenAI(api_key="...")
service = SyntheticDataService(llm_client=llm_client)
```

**Rule-based** (faster, good quality):
```python
service = SyntheticDataService()  # No llm_client = rule-based
```

## Testing

```bash
# Run all synthetic data tests
uv run pytest tests/routing/unit/synthetic/ -v

# Run specific test file
uv run pytest tests/routing/unit/synthetic/test_service.py -v

# Test generator
uv run pytest tests/routing/unit/synthetic/test_generators_integration.py -v
```

**Test Coverage**:
- 11 base generator tests
- 7 generator integration tests
- 23 registry tests
- 22 schema tests
- 19 service tests
- **Total: 82 tests**

## Development

### Adding a New Optimizer

1. **Define Schema** in `schemas.py`:
```python
class NewOptimizerSchema(BaseModel):
    query: str
    # ... optimizer-specific fields
```

2. **Register in** `registry.py`:
```python
OPTIMIZER_REGISTRY["new_optimizer"] = OptimizerConfig(
    name="new_optimizer",
    description="What it does",
    schema_class=NewOptimizerSchema,
    generator_class_name="NewOptimizerGenerator",
    backend_query_strategy="diverse",
    agent_mapping_required=True,
)
```

3. **Create Generator** in `generators/new_optimizer.py`:
```python
from cogniverse_synthetic.generators.base import BaseGenerator

class NewOptimizerGenerator(BaseGenerator):
    async def generate(
        self, sampled_content, target_count, **kwargs
    ) -> List[BaseModel]:
        # Implementation
        return examples
```

4. **Add to Service** in `service.py`:
```python
self.generators = {
    # ... existing
    "NewOptimizerGenerator": NewOptimizerGenerator(),
}
```

5. **Write Tests**:
```python
@pytest.mark.asyncio
async def test_new_optimizer_generator():
    gen = NewOptimizerGenerator()
    examples = await gen.generate(mock_data, 10)
    assert len(examples) == 10
```

## Performance Considerations

- **Batch Size**: Use `batch/generate` endpoint for large datasets
- **Profile Selection**: Rule-based is faster; LLM-based is higher quality
- **Backend Sampling**: Larger `sample_size` = more diverse patterns
- **Caching**: Profile selection reasoning is not cached (stateless)
- **Concurrency**: All generators are async-ready
- **DSPy Optimization**: Compiled modules faster than uncompiled (use `compiled_path` in config)

## Troubleshooting

**Issue**: `ValueError: Unknown optimizer 'xyz'`
- **Fix**: Check `OPTIMIZER_REGISTRY.keys()` for valid names

**Issue**: Empty `sampled_content` from BackendQuerier
- **Fix**: Ensure `Backend` instance is configured and passed to service
- **Note**: Falls back to mock data if no backend provided

**Issue**: Profile selection returns unexpected profiles
- **Fix**: Provide `backend_config` with actual profile definitions
- **Note**: System uses defaults if no config provided

**Issue**: `ValueError: RoutingGenerator requires optimizer_config`
- **Fix**: Provide `OptimizerGenerationConfig` with DSPy modules configuration
- **Note**: Configuration is required - no defaults or fallbacks

**Issue**: `ValueError: Failed to generate query containing entities after 3 attempts`
- **Fix**: Check DSPy LM is configured correctly (`dspy.configure(lm=...)`)
- **Note**: ValidatedEntityQueryGenerator retries 3 times before raising error

**Issue**: Tests fail with import errors
- **Fix**: Reinstall package: `uv pip install -e libs/synthetic`

## Package Location

The synthetic data generation package is part of the Implementation Layer:

```
packages/
└── cogniverse-synthetic/
    ├── cogniverse_synthetic/
    │   ├── service.py              # Main service orchestrator
    │   ├── api.py                  # FastAPI router
    │   ├── schemas.py              # Pydantic models
    │   ├── registry.py             # Optimizer registry
    │   ├── profile_selector.py     # Profile selection logic
    │   ├── backend_querier.py      # Backend content sampling
    │   ├── dspy_signatures.py      # DSPy signature definitions
    │   ├── dspy_modules.py         # Validated query generators
    │   ├── generators/             # Concrete generators
    │   │   ├── base.py
    │   │   ├── modality.py
    │   │   ├── cross_modal.py
    │   │   ├── routing.py
    │   │   └── workflow.py
    │   └── utils/                  # Pattern extraction utilities
    │       ├── pattern_extraction.py
    │       └── agent_inference.py
    └── tests/
```

## Related Documentation

- [11-Package Architecture](./architecture/overview.md) - Overall system architecture
- [Routing Module](./modules/routing.md) - Query routing module (uses this system)
- [Optimization Module](./modules/optimization.md) - DSPy optimization integration

## API Reference

See `packages/cogniverse-synthetic/cogniverse_synthetic/` for detailed docstrings:
- `service.py` - SyntheticDataService class
- `api.py` - FastAPI router
- `schemas.py` - All Pydantic models
- `registry.py` - Optimizer registry
- `generators/` - All generator implementations
