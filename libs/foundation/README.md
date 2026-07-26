# Cogniverse Foundation

**Last Updated:** 2025-11-13
**Layer:** Foundation
**Dependencies:** cogniverse-sdk

Cross-cutting concerns and shared infrastructure for the Cogniverse platform.

## Overview

The Foundation package sits in the **Foundation Layer**, bridging the pure interfaces (`cogniverse-sdk`) and core implementations. It provides reusable infrastructure for configuration management and telemetry abstractions that are used throughout the platform.

This package is designed to be lightweight and dependency-minimal, focusing on base classes and interfaces rather than concrete implementations.

## Package Structure

```
cogniverse_foundation/
├── __init__.py
├── config/
│   ├── agent_config.py      # Agent configuration base classes
│   ├── api_mixin.py         # API configuration mixins
│   ├── bootstrap.py         # Bootstrap configuration helpers
│   ├── config.py            # Configuration base classes
│   ├── llm_factory.py       # LLM endpoint factory
│   ├── manager.py           # Configuration manager
│   ├── unified_config.py    # Unified config interface
│   └── utils.py             # Config utilities
└── telemetry/
    ├── config.py            # Telemetry configuration
    ├── context.py           # Span-helper functions (search_span, etc.)
    ├── manager.py           # Telemetry manager
    ├── registry.py          # Telemetry provider registry
    └── providers/           # Provider implementations
```

## Key Modules

### Configuration System (`cogniverse_foundation.config`)

Base classes and utilities for configuration management across the platform:

**Core Components:**
- `AgentConfig` (`cogniverse_foundation.config.agent_config`): Base configuration class for agent settings
- `SystemConfig` (`cogniverse_foundation.config.unified_config`): System-level configuration (agent URLs, backends, telemetry)
- `ConfigManager` (`cogniverse_foundation.config.manager`): Centralized configuration management
- `APIMixin` (`cogniverse_foundation.config.api_mixin`): API configuration integration mixin

**Key Features:**
- **Type-Safe**: Full Pydantic validation
- **Multi-Tenant**: Tenant-aware configuration isolation
- **Extensible**: Easy to extend for custom config types

### Telemetry System (`cogniverse_foundation.telemetry`)

Provider-agnostic telemetry and observability infrastructure:

**Core Components:**
- `TelemetryConfig`: Base configuration for telemetry providers
- `TelemetryManager`: Centralized telemetry management
- `TelemetryRegistry` (`cogniverse_foundation.telemetry.registry`): Provider registration and discovery
- `context` module: Span-helper functions for tenant-aware instrumentation (e.g., `search_span`)

**Provider Support:**
- **OpenTelemetry**: Built-in OpenTelemetry integration
- **Plugin-Based**: Support for custom telemetry providers
- **Provider Registry**: Dynamic provider discovery and registration

**Key Features:**
- **Provider-Agnostic**: Works with Phoenix, Jaeger, Zipkin, etc.
- **OpenTelemetry-First**: Native OpenTelemetry support
- **Distributed Tracing**: Trace propagation across services
- **Metrics & Spans**: Comprehensive observability primitives

## Installation

```bash
uv add cogniverse-foundation
```

Or with pip:
```bash
pip install cogniverse-foundation
```

## Dependencies

**Internal:**
- `cogniverse-sdk`: Pure backend interfaces

**External:**
- `fastapi==0.135.3`: Dynamic configuration REST endpoints
- `opentelemetry-api==1.41.0`: OpenTelemetry interfaces
- `opentelemetry-sdk==1.41.0`: OpenTelemetry SDK
- `pydantic==2.12.5`: Data validation
- `sqlalchemy==2.0.49`: Database support
- `pandas==2.3.3`: Data manipulation

## Usage Examples

### Configuration Management

```python
from cogniverse_foundation.config.agent_config import (
    AgentConfig,
    DSPyModuleType,
    ModuleConfig,
)
from cogniverse_foundation.config.manager import ConfigManager

# ConfigStore is an injected cogniverse-sdk implementation.
config_manager = ConfigManager(store=config_store)
agent_config = AgentConfig(
    agent_name="summarizer",
    agent_version="1.0.0",
    agent_description="Summarizes documents",
    agent_url="http://summarizer:8000",
    capabilities=["summarization"],
    skills=[],
    module_config=ModuleConfig(
        module_type=DSPyModuleType.PREDICT,
        signature="document -> summary",
    ),
    llm_model="openai/gpt-4.1",
)
config_manager.set_agent_config(
    tenant_id="acme",
    agent_name="summarizer",
    agent_config=agent_config,
)

config = config_manager.get_agent_config("acme", "summarizer")
```

### Telemetry Integration

```python
from cogniverse_foundation.telemetry import TelemetryManager, TelemetryConfig

# Initialize telemetry
telemetry_config = TelemetryConfig(
    service_name="my-service",
    environment="production",
    otlp_endpoint="localhost:4317",
)

telemetry_manager = TelemetryManager(telemetry_config)

# Create spans for distributed tracing
with telemetry_manager.span("process_request", tenant_id="acme") as span:
    # Your business logic here
    result = process_query(query)
```

### Telemetry Provider Discovery

```python
from cogniverse_foundation.telemetry.registry import TelemetryRegistry

provider = TelemetryRegistry.get(
    name="phoenix",
    tenant_id="acme",
    config={
        "project_name": "search",
        "http_endpoint": "http://phoenix:6006",
        "grpc_endpoint": "phoenix:4317",
    },
)
```

### System Configuration

```python
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_foundation.config.utils import create_default_config_manager

# Load configuration via ConfigManager
config_manager = create_default_config_manager()
system_config = config_manager.get_system_config()

# Or create directly
system_config = SystemConfig(
    backend_url="http://localhost",
    backend_port=8080,
    environment="production",
)
```

## Architecture Position

```
Foundation Layer:
  cogniverse-sdk (zero dependencies)
    ↓
  cogniverse-foundation ← YOU ARE HERE
    ↓
Core Layer:
  cogniverse-core (agents, registries, memory)
  cogniverse-evaluation (metrics, experiments)
    ↓
Implementation Layer:
  cogniverse-agents, cogniverse-vespa, cogniverse-synthetic
    ↓
Application Layer:
  cogniverse-runtime, cogniverse-dashboard
```

## Design Principles

1. **Lightweight**: Minimal dependencies, focused on infrastructure
2. **Reusable**: Shared by all layers above
3. **Extensible**: Easy to extend for custom needs
4. **Provider-Agnostic**: Works with multiple backends and providers
5. **Type-Safe**: Full type hints and validation

## Use Cases

### Configuration Management
- Agent configuration persistence
- Multi-tenant configuration isolation
- Configuration versioning and rollback
- Schema validation and defaults

### Telemetry
- Distributed tracing across agents
- Performance monitoring
- Error tracking and debugging
- Custom metrics collection
- Integration with Phoenix, Jaeger, Zipkin

## Development

```bash
# Install in editable mode
cd libs/foundation
uv pip install -e .

# Run tests
pytest tests/foundation/
```

## Testing

The foundation package includes:
- Unit tests for configuration management
- Telemetry integration tests
- Provider registry tests

## License

MIT
