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
│   ├── a2a_agent.py         # A2A protocol agent
│   ├── a2a_mixin.py         # A2A endpoints mixin
│   ├── base.py              # AgentBase, AgentInput, AgentOutput, AgentDeps
│   ├── rails.py             # Input/output rail chains
│   ├── rlm_options.py       # RLM configuration
│   └── tenant_aware_mixin.py      # Multi-tenancy mixin (TenantAwareAgentMixin)
├── backends/                # Backend abstractions
├── common/                  # Shared utilities
│   ├── agent_models.py      # Shared agent data models
│   ├── dspy_module_registry.py    # DSPy module registry
│   ├── dynamic_dspy_mixin.py      # Dynamic DSPy capabilities
│   ├── health_mixin.py      # Health check mixin (HealthCheckMixin)
│   ├── tenant_utils.py      # Tenant parsing and validation utilities
│   └── vlm_interface.py     # Vision-language model interface
├── config/                  # Backward-compat shim → cogniverse_foundation.config
├── factories/               # Factory patterns for components
├── interfaces/              # Core interfaces
├── memory/                  # Memory system (Mem0MemoryManager)
├── registries/              # Component registries
│   ├── agent_registry.py    # Agent registration (AgentRegistry)
│   ├── backend_registry.py  # Backend registration (BackendRegistry)
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
- `AgentBase`: Generic type-safe base class for all agents (in `cogniverse_core.agents.base`)
- `A2AAgent`: A2A protocol + DSPy integration on top of `AgentBase` (in `cogniverse_core.agents.a2a_agent`)
- `AgentInput`, `AgentOutput`, `AgentDeps`: Pydantic base models for typed I/O

**Mixins (separate modules):**
- `MemoryAwareMixin`: Memory integration for agents (in `cogniverse_agents.memory_aware_mixin`)
- `TenantAwareAgentMixin`: Multi-tenancy support (in `cogniverse_core.agents.tenant_aware_mixin`)
- `HealthCheckMixin`: Health monitoring / FastAPI endpoint setup (in `cogniverse_core.common.health_mixin`)
- `A2AEndpointsMixin`: A2A endpoint helpers (in `cogniverse_core.agents.a2a_mixin`)
- `DynamicDSPyMixin`: Dynamic DSPy module loading (in `cogniverse_core.common.dynamic_dspy_mixin`)

**Key Features:**
- **Composable Design**: Mix and match capabilities via mixins
- **Multi-Tenancy**: Built-in tenant isolation
- **Memory Support**: Persistent agent memory via `MemoryAwareMixin`
- **Health Monitoring**: Built-in health checks via `HealthCheckMixin`
- **A2A Protocol**: Agent-to-agent communication
- **DSPy Integration**: Native DSPy support via `A2AAgent`

### Registries (`cogniverse_core.registries`)

Component registration and discovery system:

**Core Registries:**
- `AgentRegistry`: Agent endpoint registration and health monitoring (in `cogniverse_core.registries.agent_registry`)
- `BackendRegistry`: Backend provider registration (in `cogniverse_core.registries.backend_registry`)
- `SchemaRegistry`: Schema template registration (in `cogniverse_core.registries.schema_registry`)
- `WorkflowStoreRegistry`: Workflow store backend registration (in `cogniverse_core.registries.workflow_store_registry`)

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

### Configuration (`cogniverse_foundation.config`)

System-wide configuration management:
- `SystemConfig`: Unified system configuration (in `cogniverse_foundation.config.unified_config`)
- `LLMEndpointConfig`, `LLMConfig`: LLM endpoint and model configuration
- `create_default_config_manager`: Build a default `ConfigManager` (in `cogniverse_foundation.config.utils`)
- Environment-based config loading
- Configuration validation

### Common Utilities (`cogniverse_core.common`)

Shared functionality across the platform:

**Caching:**
- Query cache
- Result cache
- Embedding cache
- TTL-based expiration

**Tenant:**
- `parse_tenant_id`: Parse `org:tenant` or simple `tenant` format (in `cogniverse_core.common.tenant_utils`)
- `require_tenant_id`: Enforce an explicit `tenant_id` on a request (in `cogniverse_core.common.tenant_utils`)
- `validate_tenant_id`: Validate tenant ID format
- `get_tenant_storage_path`: Build tenant-scoped storage paths

### Telemetry (`cogniverse_foundation.telemetry`)

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
from cogniverse_core.agents import AgentBase, AgentInput, AgentOutput, AgentDeps
from cogniverse_core.agents.tenant_aware_mixin import TenantAwareAgentMixin
from cogniverse_core.common.health_mixin import HealthCheckMixin
from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin


class MyInput(AgentInput):
    query: str
    tenant_id: str


class MyOutput(AgentOutput):
    result: str


class MyDeps(AgentDeps):
    pass


class MyCustomAgent(
    AgentBase[MyInput, MyOutput, MyDeps],
    TenantAwareAgentMixin,
    HealthCheckMixin,
):
    """Custom agent with multi-tenancy and health checks."""

    def __init__(self, tenant_id: str, deps: MyDeps):
        AgentBase.__init__(self, deps=deps)
        TenantAwareAgentMixin.__init__(self, tenant_id=tenant_id)

    async def _process_impl(self, input: MyInput) -> MyOutput:
        """Core agent logic."""
        # Your business logic here
        result = f"Processed: {input.query}"
        return MyOutput(result=result)

    def get_health_status(self) -> dict:
        """Health check implementation (HealthCheckMixin)."""
        return {
            "status": "healthy",
            "tenant_id": self.tenant_id,
        }
```

### Using Agent Registry

`AgentRegistry` manages live agent endpoints and is instantiated per-tenant.

```python
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.utils import create_default_config_manager

config_manager = create_default_config_manager()
registry = AgentRegistry(tenant_id="acme:production", config_manager=config_manager)

# List registered agent names
available_agents = registry.list_agents()
print(f"Available agents: {available_agents}")

# Look up a specific agent endpoint
agent_endpoint = registry.get_agent("video_search_agent")

# Find agents by capability
search_agents = registry.find_agents_by_capability("video_search")
```

### Memory Management

```python
from cogniverse_core.memory.manager import Mem0MemoryManager

# Per-tenant singleton (LRU-bounded); call initialize() before first use
memory = Mem0MemoryManager(tenant_id="acme:production")

# Add content to agent memory
memory.add_memory(
    content="RAG stands for Retrieval-Augmented Generation.",
    tenant_id="acme:production",
    agent_name="search_agent",
)

# Search memories
relevant_memories = memory.search_memory(
    query="retrieval augmented generation",
    tenant_id="acme:production",
    agent_name="search_agent",
    top_k=5,
)

# Get all memories for an agent
all_memories = memory.get_all_memories(
    tenant_id="acme:production",
    agent_name="search_agent",
)

# Delete a specific memory by ID
memory.delete_memory(memory_id="mem_xyz")
```

### Backend Registry

```python
from cogniverse_core.registries import BackendRegistry
from cogniverse_sdk.interfaces.backend import Backend

class MyCustomBackend(Backend):
    """Custom backend implementation."""

    async def search(self, query, embedding=None, **kwargs):
        # Implementation
        pass

# Register a backend class
BackendRegistry.register_backend("my_backend", MyCustomBackend)

# Retrieve the registry singleton and look up a backend class
registry = BackendRegistry.get_instance()
```

### Multi-Tenant Configuration

```python
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_core.common.tenant_utils import parse_tenant_id, require_tenant_id

# Build a SystemConfig from a dict (e.g. loaded from YAML)
config = SystemConfig.from_dict({
    "llm": {"model": "ollama/qwen3:4b"},
})

# Parse an org:tenant identifier
org_id, tenant_name = parse_tenant_id("acme:production")
# → ("acme", "production")

# Enforce that a request carries an explicit tenant_id
# (raises ValueError if missing or empty)
tenant_id = require_tenant_id("acme:production", source="MyRequest")
```

### A2A (Agent-to-Agent) Communication

`A2AAgent` (in `cogniverse_core.agents.a2a_agent`) combines `AgentBase` with DSPy integration and A2A endpoint helpers. Subclass it when your agent needs to receive or dispatch A2A protocol messages.

```python
from cogniverse_core.agents import AgentBase, AgentInput, AgentOutput, AgentDeps, A2AAgent, A2AAgentConfig
from cogniverse_agents.orchestrator_agent import OrchestratorAgent, OrchestratorDeps, OrchestratorInput
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.utils import create_default_config_manager

config_manager = create_default_config_manager()
registry = AgentRegistry(tenant_id="acme:production", config_manager=config_manager)
orchestrator = OrchestratorAgent(deps=OrchestratorDeps(), registry=registry)

result = await orchestrator.process(OrchestratorInput(
    query="Find and summarize videos about AI",
    tenant_id="acme:production",
))
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
