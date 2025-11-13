# Cogniverse Core

Core functionality, base classes, and registries for the Cogniverse platform.

## Overview

This package provides the core functionality of Cogniverse, including base agent classes, component registries, memory management, and shared utilities. It sits at the heart of the system architecture, depending on the foundation layers (`sdk`, `foundation`, `evaluation`) and providing services to implementation layers (`agents`, `vespa`, `synthetic`).

## Key Components

### Agent Base Classes (`cogniverse_core.agents`)

Foundation classes for all agent implementations:
- `BaseAgent`: Abstract base class for all agents
- `MemoryAwareMixin`: Memory integration for agents
- `HealthCheckMixin`: Health monitoring capabilities
- Agent lifecycle management

### Registries (`cogniverse_core.registries`)

Component registration and discovery:
- `AgentRegistry`: Agent registration and lookup
- `BackendRegistry`: Backend provider registration
- `DSPyRegistry`: DSPy module registration
- Plugin discovery via entry points

### Memory Management (`cogniverse_core.memory`)

Memory interfaces and utilities:
- Memory interface definitions
- Integration with memory providers (e.g., Mem0)
- Memory lifecycle management

### Common Utilities (`cogniverse_core.common`)

Shared functionality across the platform:
- `Mem0MemoryManager`: Mem0 integration wrapper
- `TenantUtils`: Tenant context management
- Caching utilities
- Configuration helpers

## Installation

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
- `pydantic>=2.0.0`: Data validation
- `mem0ai>=0.1.118`: Memory management

## Usage

### Creating Custom Agents

```python
from cogniverse_core.agents import BaseAgent, MemoryAwareMixin

class MyCustomAgent(BaseAgent, MemoryAwareMixin):
    """Custom agent with memory support."""

    def __init__(self, tenant_id: str, config):
        super().__init__(tenant_id=tenant_id)
        self.config = config

    async def execute(self, query: str):
        # Implement agent logic
        return {"result": "success"}
```

### Using Registries

```python
from cogniverse_core.registries import AgentRegistry

# Register an agent
AgentRegistry.register("my_agent", MyCustomAgent)

# Get registered agent
agent_class = AgentRegistry.get("my_agent")
agent = agent_class(tenant_id="acme", config=config)
```

### Memory Management

```python
from cogniverse_core.common import Mem0MemoryManager

# Get memory manager for tenant
memory = Mem0MemoryManager.get_instance(tenant_id="acme")

# Add memory
memory.add(
    messages=[{"role": "user", "content": "test"}],
    user_id="acme"
)

# Search memories
results = memory.search(query="test", user_id="acme")
```

## Architecture Position

Core sits in the **Core Layer** of the Cogniverse architecture:

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
  cogniverse-agents, cogniverse-vespa, cogniverse-synthetic
```

## Development

```bash
# Install in editable mode
cd libs/core
pip install -e .

# Run tests
pytest tests/core/ tests/common/ tests/memory/
```

## Testing

The core package includes:
- Unit tests for all base classes
- Registry tests
- Memory management tests
- Integration tests with agents

## License

MIT
