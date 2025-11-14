# Cogniverse Core

**Last Updated:** 2025-11-13
**Layer:** Core
**Dependencies:** cogniverse-sdk, cogniverse-foundation, cogniverse-evaluation

Core functionality, base classes, and registries for the Cogniverse platform.

## Overview

The Core package sits at the **Core Layer** of the Cogniverse architecture, providing fundamental building blocks for the entire platform. It includes base agent classes, component registries, memory management, backend abstractions, configuration management, and shared utilities.

This package bridges the foundation layers (`sdk`, `foundation`, `evaluation`) and provides services to implementation layers (`agents`, `vespa`, `synthetic`). All concrete agent implementations and backend integrations build on top of Core.

## Package Structure

```
cogniverse_core/
├── __init__.py
├── agents/                  # Agent base classes and mixins
│   ├── a2a_mixin.py         # Agent-to-Agent communication mixin
│   ├── dspy_a2a_base.py     # DSPy A2A base class
│   ├── dspy_integration_mixin.py  # DSPy integration
│   ├── dynamic_dspy_mixin.py      # Dynamic DSPy capabilities
│   ├── health_mixin.py      # Health check mixin
│   ├── memory_aware_mixin.py      # Memory integration mixin
│   └── tenant_aware_mixin.py      # Multi-tenancy mixin
├── backends/                # Backend abstractions
├── common/                  # Shared utilities
│   ├── caching/             # Caching utilities
│   ├── memory/              # Memory management
│   └── tenant/              # Tenant utilities
├── config/                  # System configuration
├── factories/               # Factory patterns for components
├── interfaces/              # Core interfaces
├── memory/                  # Memory system interfaces
├── registries/              # Component registries
│   ├── agent_registry.py    # Agent registration
│   ├── backend_registry.py  # Backend registration
│   ├── dspy_registry.py     # DSPy module registration
│   ├── schema_registry.py   # Schema registration
│   └── registry.py          # Base registry
├── schemas/                 # Data schemas
├── telemetry/               # Telemetry integration
└── validation/              # Validation utilities
```

## Key Modules

### Agent Base Classes (`cogniverse_core.agents`)

Foundation classes and mixins for all agent implementations:

**Base Classes:**
- `BaseAgent`: Abstract base class for all agents
- `DSPyA2ABase`: Base class for DSPy-powered A2A agents

**Mixins:**
- `MemoryAwareMixin`: Memory integration for agents
- `TenantAwareMixin`: Multi-tenancy support
- `HealthCheckMixin`: Health monitoring capabilities
- `A2AMixin`: Agent-to-Agent communication
- `DSPyIntegrationMixin`: DSPy framework integration
- `DynamicDSPyMixin`: Dynamic DSPy module loading

**Key Features:**
- **Composable Design**: Mix and match capabilities via mixins
- **Multi-Tenancy**: Built-in tenant isolation
- **Memory Support**: Persistent agent memory
- **Health Monitoring**: Built-in health checks
- **A2A Protocol**: Agent-to-agent communication
- **DSPy Integration**: Native DSPy support

### Registries (`cogniverse_core.registries`)

Component registration and discovery system:

**Core Registries:**
- `AgentRegistry`: Agent class registration and lookup
- `BackendRegistry`: Backend provider registration
- `DSPyRegistry`: DSPy module registration and management
- `SchemaRegistry`: Schema template registration
- `BaseRegistry`: Abstract base registry

**Features:**
- **Plugin Discovery**: Automatic plugin detection via entry points
- **Type Safety**: Type-checked registry operations
- **Namespace Support**: Namespaced component registration
- **Dynamic Loading**: Lazy loading of components
- **Validation**: Component validation on registration

### Memory Management (`cogniverse_core.memory`)

Memory interfaces and integration:

**Components:**
- Memory interface definitions
- Mem0 integration (`Mem0MemoryManager`)
- Memory lifecycle management
- Tenant-aware memory isolation
- Query and retrieval APIs

**Features:**
- **Provider Agnostic**: Works with multiple memory providers
- **Tenant Isolation**: Separate memory per tenant
- **Persistent Storage**: Long-term memory persistence
- **Semantic Search**: Vector-based memory retrieval
- **Context Management**: Automatic context tracking

### Backend Abstractions (`cogniverse_core.backends`)

Backend provider abstractions and utilities:
- Backend factory patterns
- Connection pooling
- Provider-specific optimizations
- Error handling and retries

### Configuration (`cogniverse_core.config`)

System-wide configuration management:
- `SystemConfig`: Unified system configuration
- `AgentConfig`: Agent-specific configuration
- Environment-based config loading
- Configuration validation
- Tenant-specific overrides

### Common Utilities (`cogniverse_core.common`)

Shared functionality across the platform:

**Caching:**
- Query cache
- Result cache
- Embedding cache
- TTL-based expiration

**Memory:**
- `Mem0MemoryManager`: Mem0 integration wrapper
- Memory utilities and helpers

**Tenant:**
- `TenantUtils`: Tenant context management
- Tenant isolation utilities
- Multi-tenant data partitioning

### Telemetry (`cogniverse_core.telemetry`)

Telemetry and observability integration:
- OpenTelemetry span creation
- Trace context propagation
- Custom metrics emission
- Performance monitoring

### Validation (`cogniverse_core.validation`)

Data validation utilities:
- Schema validation
- Input sanitization
- Output validation
- Type checking

## Installation

```bash
uv add cogniverse-core
```

Or with pip:
```bash
pip install cogniverse-core
```

## Dependencies

**Internal:**
- `cogniverse-sdk`: Pure backend interfaces
- `cogniverse-foundation`: Configuration and telemetry base
- `cogniverse-evaluation`: Evaluation framework

**External:**
- `dspy-ai>=3.0.2`: DSPy framework for LLM programming
- `litellm>=1.73.0`: Multi-provider LLM interface
- `opentelemetry-api>=1.20.0`: Telemetry interfaces
- `opentelemetry-sdk>=1.20.0`: Telemetry SDK
- `pydantic>=2.0.0`: Data validation
- `mem0ai>=0.1.118`: Memory management
- `pyyaml>=6.0`: YAML configuration
- `aiofiles>=24.1.0`: Async file I/O

## Usage Examples

### Creating Custom Agents

```python
from cogniverse_core.agents import (
    BaseAgent,
    MemoryAwareMixin,
    TenantAwareMixin,
    HealthCheckMixin
)

class MyCustomAgent(
    BaseAgent,
    MemoryAwareMixin,
    TenantAwareMixin,
    HealthCheckMixin
):
    """Custom agent with memory, multi-tenancy, and health checks."""

    def __init__(self, tenant_id: str, config):
        BaseAgent.__init__(self)
        TenantAwareMixin.__init__(self, tenant_id=tenant_id)
        MemoryAwareMixin.__init__(self, tenant_id=tenant_id)
        self.config = config

    async def execute(self, query: str):
        """Execute agent logic with memory and tenant isolation."""
        # Use tenant-aware memory
        memories = await self.memory.search(
            query=query,
            user_id=self.tenant_id
        )

        # Your business logic here
        result = await self._process_with_context(query, memories)

        # Store result in memory
        await self.memory.add(
            messages=[
                {"role": "user", "content": query},
                {"role": "assistant", "content": result}
            ],
            user_id=self.tenant_id
        )

        return result

    async def health_check(self) -> dict:
        """Health check implementation."""
        return {
            "status": "healthy",
            "tenant_id": self.tenant_id,
            "memory_available": self.memory is not None
        }
```

### Using Agent Registry

```python
from cogniverse_core.registries import AgentRegistry

# Register an agent class
AgentRegistry.register("my_custom_agent", MyCustomAgent)

# List registered agents
available_agents = AgentRegistry.list()
print(f"Available agents: {available_agents}")

# Get and instantiate registered agent
agent_class = AgentRegistry.get("my_custom_agent")
agent = agent_class(tenant_id="acme", config=config)

# Execute agent
result = await agent.execute("What is machine learning?")
```

### DSPy Integration

```python
from cogniverse_core.agents import DSPyIntegrationMixin
from cogniverse_core.registries import DSPyRegistry
import dspy

class MyDSPyAgent(DSPyIntegrationMixin):
    """Agent with DSPy optimization support."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setup_dspy()

    def setup_dspy(self):
        """Setup DSPy modules."""
        # Configure LLM
        lm = dspy.OpenAI(model="gpt-4", max_tokens=2000)
        dspy.settings.configure(lm=lm)

        # Register DSPy signature
        class QASignature(dspy.Signature):
            """Answer questions based on context."""
            context = dspy.InputField(desc="background context")
            question = dspy.InputField(desc="question to answer")
            answer = dspy.OutputField(desc="concise answer")

        self.qa_module = dspy.ChainOfThought(QASignature)

    async def answer(self, question: str, context: str):
        """Answer question using DSPy."""
        result = self.qa_module(context=context, question=question)
        return result.answer
```

### Memory Management

```python
from cogniverse_core.common.memory import Mem0MemoryManager

# Get memory manager for tenant (singleton pattern)
memory = Mem0MemoryManager.get_instance(tenant_id="acme")

# Add conversation to memory
memory.add(
    messages=[
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation..."}
    ],
    user_id="user_123"
)

# Search memories
relevant_memories = memory.search(
    query="retrieval augmented generation",
    user_id="user_123",
    limit=5
)

# Get all memories for user
all_memories = memory.get_all(user_id="user_123")

# Delete specific memory
memory.delete(memory_id="mem_xyz")
```

### Backend Registry

```python
from cogniverse_core.registries import BackendRegistry
from cogniverse_sdk.interfaces.backend import Backend

class MyCustomBackend(Backend):
    """Custom backend implementation."""

    def _initialize_backend(self, config):
        self.client = CustomVectorDB(config)

    async def search(self, query, embedding=None, **kwargs):
        # Implementation
        pass

# Register backend
BackendRegistry.register("my_backend", MyCustomBackend)

# Get backend
backend = BackendRegistry.get("my_backend")(config=backend_config)
```

### Multi-Tenant Configuration

```python
from cogniverse_core.config import SystemConfig
from cogniverse_core.common.tenant import TenantUtils

# Load base configuration
base_config = SystemConfig.from_file("config.yaml")

# Get tenant-specific configuration
tenant_config = TenantUtils.get_tenant_config(
    base_config=base_config,
    tenant_id="acme"
)

# Apply tenant overrides
tenant_config.update_overrides({
    "max_concurrent_requests": 100,
    "embedding_model": "custom-model-v2"
})
```

### A2A (Agent-to-Agent) Communication

```python
from cogniverse_core.agents import A2AMixin

class CollaborativeAgent(BaseAgent, A2AMixin):
    """Agent that can communicate with other agents."""

    async def execute(self, query: str):
        # Invoke another agent
        search_results = await self.invoke_agent(
            agent_name="video_search_agent",
            method="search",
            params={"query": query, "top_k": 10}
        )

        # Process results
        summary = await self.invoke_agent(
            agent_name="summarizer_agent",
            method="summarize",
            params={"content": search_results}
        )

        return summary
```

## Architecture Position

```
Foundation Layer:
  cogniverse-sdk → cogniverse-foundation
    ↓
Core Layer:
  cogniverse-core ← YOU ARE HERE
  cogniverse-evaluation
  cogniverse-telemetry-phoenix (plugin)
    ↓
Implementation Layer:
  cogniverse-agents (routing, search, orchestration)
  cogniverse-vespa (Vespa backend implementation)
  cogniverse-synthetic (synthetic data generation)
    ↓
Application Layer:
  cogniverse-runtime (FastAPI runtime)
  cogniverse-dashboard (Streamlit UI)
```

## Design Principles

1. **Composability**: Mix and match capabilities via mixins
2. **Extensibility**: Easy to extend for custom needs
3. **Type Safety**: Full type hints and validation
4. **Multi-Tenancy**: Built-in tenant isolation
5. **Observability**: Comprehensive telemetry support
6. **Provider Agnostic**: Works with multiple backends and providers

## Development

```bash
# Install in editable mode
cd libs/core
uv pip install -e .

# Run tests
pytest tests/core/ tests/common/ tests/memory/

# Run specific test categories
pytest tests/core/test_agents.py
pytest tests/core/test_registries.py
pytest tests/core/test_memory.py
```

## Testing

The core package includes:
- Unit tests for all base classes
- Registry tests with mock components
- Memory management tests
- Integration tests with agents
- Multi-tenancy tests
- DSPy integration tests

## Performance

- **Agent Registration**: O(1) lookup via registry
- **Memory Operations**: <100ms P95 for search
- **Health Checks**: <10ms P95 latency
- **Registry Lookups**: <1ms average

## License

MIT
