# Core Module

**Package:** `cogniverse_core`
**Location:** `libs/core/cogniverse_core/`
**Purpose:** Type-safe agent base classes, registries, and core infrastructure
**Last Updated:** 2026-01-01

---

## Table of Contents

1. [Overview](#overview)
2. [Package Structure](#package-structure)
3. [Type-Safe Agent System](#type-safe-agent-system)
   - [AgentBase](#agentbase)
   - [AgentInput / AgentOutput / AgentDeps](#agentinput--agentoutput--agentdeps)
   - [A2AAgent](#a2aagent)
4. [Agent Mixins](#agent-mixins)
5. [Registries](#registries)
6. [Memory Management](#memory-management)
7. [Configuration](#configuration)
8. [Usage Examples](#usage-examples)
9. [Architecture Position](#architecture-position)
10. [Testing](#testing)

---

## Overview

The Core package is the **foundation layer** for all agent implementations in Cogniverse. It provides:

- **Type-Safe Agents**: Generic base classes with compile-time type checking and runtime Pydantic validation
- **A2A Protocol Support**: Google's Agent-to-Agent protocol for inter-agent communication
- **DSPy Integration**: Native DSPy module support for AI-powered agents
- **Multi-Tenancy**: Built-in tenant isolation for enterprise deployments
- **Component Registries**: Dynamic registration and discovery of agents, backends, and schemas
- **Memory System**: Mem0-based persistent agent memory

All concrete agent implementations (RoutingAgent, SearchAgent, etc.) inherit from these base classes.

---

## Package Structure

```
cogniverse_core/
├── agents/                      # Agent base classes and mixins
│   ├── base.py                  # AgentBase[InputT, OutputT, DepsT]
│   ├── a2a_agent.py             # A2AAgent with A2A protocol + DSPy
│   ├── memory_aware_mixin.py    # Memory integration mixin
│   ├── tenant_aware_mixin.py    # Multi-tenancy mixin
│   ├── health_mixin.py          # Health check mixin
│   ├── a2a_mixin.py             # A2A communication mixin
│   ├── dspy_integration_mixin.py # DSPy module support
│   └── dynamic_dspy_mixin.py    # Dynamic DSPy loading
├── registries/                  # Component registries
│   ├── agent_registry.py        # Agent class registration
│   ├── backend_registry.py      # Backend provider registration
│   ├── dspy_registry.py         # DSPy module registration
│   ├── schema_registry.py       # Schema template registration
│   └── registry.py              # Base registry class
├── memory/                      # Memory system
│   ├── manager.py               # Mem0MemoryManager
│   ├── backend_config.py        # Memory backend configuration
│   └── backend_vector_store.py  # Vector store integration
├── common/                      # Shared utilities
│   ├── cache/                   # Caching subsystem
│   ├── tenant_utils.py          # Tenant utilities
│   ├── dspy_module_registry.py  # DSPy module management
│   └── a2a_utils.py             # A2A protocol utilities
├── config/                      # Configuration management
├── schemas/                     # Data schemas
│   └── filesystem_loader.py     # Schema loading from files
└── backends/                    # Backend abstractions
```

---

## Type-Safe Agent System

The type-safe agent system uses Python generics to provide **compile-time type checking** and **runtime validation**. This is the foundation of all agents in Cogniverse.

### AgentBase

`AgentBase[InputT, OutputT, DepsT]` is the abstract base class for all agents.

**Type Parameters:**
- `InputT`: Agent input type (must extend `AgentInput`)
- `OutputT`: Agent output type (must extend `AgentOutput`)
- `DepsT`: Agent dependencies type (must extend `AgentDeps`)

**Key Features:**
- Generic type parameters extracted at class definition time
- Automatic Pydantic validation for inputs and outputs
- Runtime type checking
- Statistics tracking (process count, error count)

```python
from cogniverse_core.agents.base import AgentBase, AgentInput, AgentOutput, AgentDeps

class SearchInput(AgentInput):
    query: str
    top_k: int = 10

class SearchOutput(AgentOutput):
    results: List[Dict[str, Any]]
    total_count: int

class SearchDeps(AgentDeps):
    vespa_client: VespaClient
    embedding_model: str = "colpali"

class SearchAgent(AgentBase[SearchInput, SearchOutput, SearchDeps]):
    async def process(self, input: SearchInput) -> SearchOutput:
        # IDE autocomplete works here - input.query, input.top_k
        results = await self.deps.vespa_client.search(
            query=input.query,
            limit=input.top_k
        )
        return SearchOutput(results=results, total_count=len(results))
```

**API:**

| Method | Description |
|--------|-------------|
| `__init__(deps: DepsT)` | Initialize agent with typed dependencies |
| `process(input: InputT) -> OutputT` | **Abstract** - Implement agent logic |
| `run(raw_input: Dict) -> OutputT` | Run with raw dict, validates input/output |
| `validate_input(raw: Dict) -> InputT` | Validate and convert to typed input |
| `validate_output(raw: Dict) -> OutputT` | Validate and convert to typed output |
| `get_input_schema() -> Dict` | Get JSON schema for input type |
| `get_output_schema() -> Dict` | Get JSON schema for output type |
| `get_stats() -> Dict` | Get processing statistics |

### AgentInput / AgentOutput / AgentDeps

These are Pydantic BaseModel subclasses that define agent interfaces:

```python
from cogniverse_core.agents.base import AgentInput, AgentOutput, AgentDeps

class AgentInput(BaseModel):
    """Base class for all agent inputs. Strict - no extra fields allowed."""
    model_config = ConfigDict(extra="forbid")

class AgentOutput(BaseModel):
    """Base class for all agent outputs. Strict - no extra fields allowed."""
    model_config = ConfigDict(extra="forbid")

class AgentDeps(BaseModel):
    """Base class for agent dependencies. tenant_id is required."""
    tenant_id: str
    model_config = ConfigDict(extra="allow")  # Dependencies can have extra fields
```

**Important:** `tenant_id` is **required** in all AgentDeps - this enforces multi-tenancy.

### A2AAgent

`A2AAgent[InputT, OutputT, DepsT]` extends `AgentBase` with:

- **A2A Protocol Endpoints**: Standard endpoints per Google A2A spec
- **DSPy Integration**: Optional DSPy module for AI processing
- **FastAPI Server**: Built-in HTTP server with A2A endpoints
- **Inter-Agent Communication**: Call other A2A agents

```python
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig

class RoutingAgent(A2AAgent[RoutingInput, RoutingOutput, RoutingDeps]):
    async def process(self, input: RoutingInput) -> RoutingOutput:
        # Use DSPy module if available
        if self.dspy_module:
            result = self.dspy_module(query=input.query)
            return RoutingOutput(
                recommended_agent=result.agent,
                confidence=result.confidence
            )
        return RoutingOutput(recommended_agent="search", confidence=0.5)

# Create and run agent
deps = RoutingDeps(tenant_id="acme", model_name="smollm3:3b")
config = A2AAgentConfig(
    agent_name="routing_agent",
    agent_description="Routes queries to appropriate agents",
    capabilities=["query_routing", "agent_selection"],
    port=8001
)
agent = RoutingAgent(deps=deps, config=config)
agent.start()  # Starts FastAPI server
```

**A2A Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/.well-known/agent-card.json` | GET | Agent card per A2A spec |
| `/agent.json` | GET | Legacy agent card |
| `/tasks/send` | POST | Process A2A task |
| `/health` | GET | Health check with metrics |
| `/metrics` | GET | Detailed performance metrics |
| `/schema` | GET | Input/output JSON schemas |

---

## Agent Mixins

Mixins provide composable functionality that can be added to any agent:

### MemoryAwareMixin

Adds Mem0-based persistent memory to agents:

```python
from cogniverse_core.agents.memory_aware_mixin import MemoryAwareMixin

class MyAgent(AgentBase[...], MemoryAwareMixin):
    async def process(self, input: MyInput) -> MyOutput:
        # Search memories
        memories = await self.memory.search(
            query=input.query,
            user_id=self.tenant_id
        )

        # Store new memory
        await self.memory.add(
            messages=[{"role": "user", "content": input.query}],
            user_id=self.tenant_id
        )
```

### TenantAwareMixin

Provides tenant context and isolation:

```python
from cogniverse_core.agents.tenant_aware_mixin import TenantAwareMixin

class MyAgent(AgentBase[...], TenantAwareMixin):
    def __init__(self, deps, ...):
        super().__init__(deps)
        TenantAwareMixin.__init__(self, tenant_id=deps.tenant_id)

    async def process(self, input: MyInput) -> MyOutput:
        # Access tenant-specific configuration
        tenant_config = self.get_tenant_config()
```

### HealthMixin

Adds health check capabilities:

```python
from cogniverse_core.agents.health_mixin import HealthMixin

class MyAgent(AgentBase[...], HealthMixin):
    async def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "tenant_id": self.tenant_id,
            "custom_metric": self.get_custom_metric()
        }
```

### DSPyIntegrationMixin

Integrates DSPy modules for AI processing:

```python
from cogniverse_core.agents.dspy_integration_mixin import DSPyIntegrationMixin
import dspy

class MyAgent(AgentBase[...], DSPyIntegrationMixin):
    def setup_dspy(self):
        class MySignature(dspy.Signature):
            query = dspy.InputField()
            answer = dspy.OutputField()

        self.dspy_module = dspy.ChainOfThought(MySignature)
```

---

## Registries

Registries provide dynamic component registration and discovery:

### AgentRegistry

```python
from cogniverse_core.registries import AgentRegistry

# Register agent class
AgentRegistry.register("search_agent", SearchAgent)

# List registered agents
agents = AgentRegistry.list()  # ["search_agent", "routing_agent", ...]

# Get agent class
agent_class = AgentRegistry.get("search_agent")
agent = agent_class(deps=deps, config=config)
```

### BackendRegistry

```python
from cogniverse_core.registries import BackendRegistry

# Register custom backend
BackendRegistry.register("my_backend", MyBackendClass)

# Get backend
backend = BackendRegistry.get("my_backend")(config=config)
```

### DSPyRegistry

```python
from cogniverse_core.registries import DSPyRegistry

# Register DSPy module
DSPyRegistry.register("qa_module", qa_signature, dspy.ChainOfThought)

# Get module
module = DSPyRegistry.get("qa_module")
```

### SchemaRegistry

```python
from cogniverse_core.registries import SchemaRegistry

# Register schema template
SchemaRegistry.register("video_content", video_schema)

# Get schema
schema = SchemaRegistry.get("video_content")
```

---

## Memory Management

The memory system uses Mem0 for persistent, tenant-isolated agent memory:

```python
from cogniverse_core.memory.manager import Mem0MemoryManager

# Get memory manager (singleton per tenant)
memory = Mem0MemoryManager.get_instance(tenant_id="acme")

# Add memories
memory.add(
    messages=[
        {"role": "user", "content": "What is RAG?"},
        {"role": "assistant", "content": "RAG is Retrieval-Augmented Generation..."}
    ],
    user_id="user_123"
)

# Search memories
results = memory.search(
    query="retrieval augmented generation",
    user_id="user_123",
    limit=5
)

# Get all memories for user
all_memories = memory.get_all(user_id="user_123")

# Delete memory
memory.delete(memory_id="mem_xyz")
```

---

## Configuration

Configuration management for agents and system settings:

```python
from cogniverse_core.config import SystemConfig
from cogniverse_core.common.tenant_utils import TenantUtils

# Load base configuration
config = SystemConfig.from_file("config/base.yaml")

# Get tenant-specific configuration
tenant_config = TenantUtils.get_tenant_config(
    base_config=config,
    tenant_id="acme"
)

# Apply overrides
tenant_config.update({
    "max_concurrent_requests": 100,
    "embedding_model": "colpali-v2"
})
```

---

## Usage Examples

### Creating a Complete Agent

```python
from cogniverse_core.agents.base import AgentBase, AgentInput, AgentOutput, AgentDeps
from cogniverse_core.agents.a2a_agent import A2AAgent, A2AAgentConfig
from typing import List, Optional

# 1. Define types
class SummarizerInput(AgentInput):
    text: str
    max_length: int = 100
    style: str = "concise"

class SummarizerOutput(AgentOutput):
    summary: str
    word_count: int
    key_points: List[str]

class SummarizerDeps(AgentDeps):
    model_name: str = "gpt-4"
    temperature: float = 0.7

# 2. Implement agent
class SummarizerAgent(A2AAgent[SummarizerInput, SummarizerOutput, SummarizerDeps]):
    async def process(self, input: SummarizerInput) -> SummarizerOutput:
        # Use DSPy module if available
        if self.dspy_module:
            result = self.dspy_module(
                text=input.text,
                max_length=input.max_length,
                style=input.style
            )
            return SummarizerOutput(
                summary=result.summary,
                word_count=len(result.summary.split()),
                key_points=result.key_points
            )

        # Fallback logic
        summary = input.text[:input.max_length] + "..."
        return SummarizerOutput(
            summary=summary,
            word_count=len(summary.split()),
            key_points=[]
        )

# 3. Run agent
if __name__ == "__main__":
    deps = SummarizerDeps(tenant_id="acme")
    config = A2AAgentConfig(
        agent_name="summarizer_agent",
        agent_description="Summarizes text content",
        capabilities=["text_summarization"],
        port=8002
    )
    agent = SummarizerAgent(deps=deps, config=config)
    agent.start()
```

### Calling Between Agents

```python
class OrchestratorAgent(A2AAgent[OrchestratorInput, OrchestratorOutput, OrchestratorDeps]):
    async def process(self, input: OrchestratorInput) -> OrchestratorOutput:
        # Call search agent
        search_result = await self.call_agent(
            agent_url="http://localhost:8001",
            query=input.query,
            top_k=10
        )

        # Call summarizer agent
        summary_result = await self.call_agent(
            agent_url="http://localhost:8002",
            text=str(search_result["results"]),
            max_length=200
        )

        return OrchestratorOutput(
            search_results=search_result["results"],
            summary=summary_result["summary"]
        )
```

---

## Architecture Position

```
┌─────────────────────────────────────────────────────────────────┐
│                      Application Layer                          │
│  cogniverse-runtime (FastAPI) │ cogniverse-dashboard (Streamlit)│
└─────────────────────────────────────────────────────────────────┘
                                ↑
┌─────────────────────────────────────────────────────────────────┐
│                    Implementation Layer                          │
│   cogniverse-agents    │   cogniverse-vespa   │  cogniverse-    │
│   (RoutingAgent,       │   (Vespa backend)    │  synthetic      │
│    SearchAgent, etc.)  │                      │  (data gen)     │
└─────────────────────────────────────────────────────────────────┘
                                ↑
┌─────────────────────────────────────────────────────────────────┐
│                        Core Layer                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   cogniverse-core ◄─── YOU ARE HERE         ││
│  │  AgentBase, A2AAgent, Registries, Memory, Config            ││
│  └─────────────────────────────────────────────────────────────┘│
│  cogniverse-evaluation  │  cogniverse-telemetry-phoenix         │
└─────────────────────────────────────────────────────────────────┘
                                ↑
┌─────────────────────────────────────────────────────────────────┐
│                      Foundation Layer                            │
│     cogniverse-sdk (interfaces)  │  cogniverse-foundation       │
│                                  │  (config, telemetry base)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Testing

```bash
# Run all core tests
JAX_PLATFORM_NAME=cpu uv run pytest tests/core/ -v

# Run specific test categories
uv run pytest tests/core/unit/test_agent_base.py -v
uv run pytest tests/core/unit/test_registries.py -v
uv run pytest tests/core/integration/ -v

# Run with coverage
uv run pytest tests/core/ --cov=cogniverse_core --cov-report=html
```

**Test Categories:**
- `tests/core/unit/` - Unit tests for base classes, mixins, registries
- `tests/core/integration/` - Integration tests with multiple components
- `tests/common/` - Tests for shared utilities

---

## Related Documentation

- [Agents Module](./agents.md) - Concrete agent implementations (RoutingAgent, SearchAgent, etc.)
- [Multi-Agent Interactions](../architecture/multi-agent-interactions.md) - A2A protocol flows
- [SDK Architecture](../architecture/sdk-architecture.md) - Package structure
- [Creating Agents Tutorial](../tutorials/creating-agents.md) - Step-by-step agent creation

---

**Summary:** The Core module provides the type-safe foundation for all agents in Cogniverse. `AgentBase[InputT, OutputT, DepsT]` ensures compile-time type checking and runtime validation, while `A2AAgent` adds A2A protocol support, DSPy integration, and FastAPI endpoints. Mixins provide composable functionality for memory, multi-tenancy, and health checks.
