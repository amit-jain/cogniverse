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
│   ├── manager.py           # Configuration manager
│   ├── schema.py            # Configuration schemas
│   ├── unified_config.py    # Unified config interface
│   ├── utils.py             # Config utilities
│   └── sqlite/              # SQLite-based config storage
└── telemetry/
    ├── config.py            # Telemetry configuration
    ├── context.py           # Telemetry context management
    ├── exporter.py          # Telemetry exporters
    ├── manager.py           # Telemetry manager
    ├── registry.py          # Telemetry provider registry
    └── providers/           # Provider implementations
```

## Key Modules

### Configuration System (`cogniverse_foundation.config`)

Base classes and utilities for configuration management across the platform:

**Core Components:**
- `AgentConfig`: Base configuration class for agent settings
- `UnifiedConfig`: Unified configuration interface across all components
- `ConfigManager`: Centralized configuration management
- `APIMixin`: API configuration integration mixin
- `ConfigSchema`: Pydantic schemas for configuration validation
- `ConfigUtils`: Helper utilities for config loading and validation

**SQLite Storage:**
- Persistent configuration storage using SQLite
- Tenant-aware configuration management
- Configuration versioning and history

**Key Features:**
- **Type-Safe**: Full Pydantic validation
- **Multi-Tenant**: Tenant-aware configuration isolation
- **Extensible**: Easy to extend for custom config types
- **Persistent**: SQLite-backed configuration storage

### Telemetry System (`cogniverse_foundation.telemetry`)

Provider-agnostic telemetry and observability infrastructure:

**Core Components:**
- `TelemetryConfig`: Base configuration for telemetry providers
- `TelemetryContext`: Context management for distributed tracing
- `TelemetryManager`: Centralized telemetry management
- `TelemetryRegistry`: Provider registration and discovery
- `TelemetryExporter`: Export telemetry to various backends

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
- `opentelemetry-api>=1.20.0`: OpenTelemetry interfaces
- `opentelemetry-sdk>=1.20.0`: OpenTelemetry SDK
- `pydantic>=2.0.0`: Data validation
- `sqlalchemy>=2.0.0`: Database support
- `pandas>=2.0.0`: Data manipulation

## Usage Examples

### Configuration Management

```python
from cogniverse_foundation.config import AgentConfig, ConfigManager

# Define custom agent configuration
class MyAgentConfig(AgentConfig):
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000

    class Config:
        extra = "allow"

# Use configuration manager
config_manager = ConfigManager()
config_manager.save_config(
    tenant_id="acme",
    config_key="my_agent_config",
    config=MyAgentConfig(
        model_name="claude-sonnet-4.5",
        temperature=0.9
    )
)

# Load configuration
config = config_manager.load_config(
    tenant_id="acme",
    config_key="my_agent_config",
    config_class=MyAgentConfig
)
```

### Telemetry Integration

```python
from cogniverse_foundation.telemetry import (
    TelemetryManager,
    TelemetryConfig,
    TelemetryContext
)

# Initialize telemetry
telemetry_config = TelemetryConfig(
    service_name="my-service",
    environment="production",
    endpoint="http://localhost:4317"
)

telemetry_manager = TelemetryManager(telemetry_config)
tracer = telemetry_manager.get_tracer("my-service")

# Create spans for distributed tracing
with tracer.start_as_current_span("process_request") as span:
    span.set_attribute("user.id", "user_123")
    span.set_attribute("tenant.id", "acme")

    # Your business logic here
    result = process_query(query)

    span.set_attribute("result.count", len(result))
```

### Custom Telemetry Provider

```python
from cogniverse_foundation.telemetry import TelemetryRegistry
from opentelemetry import trace

class CustomTelemetryProvider:
    """Custom telemetry provider implementation."""

    def __init__(self, config: dict):
        self.config = config
        self.tracer = trace.get_tracer(__name__)

    def get_tracer(self, name: str):
        return self.tracer

    def export_span(self, span):
        # Custom export logic
        pass

# Register custom provider
TelemetryRegistry.register("custom", CustomTelemetryProvider)

# Use custom provider
telemetry = TelemetryRegistry.get("custom")(config={"endpoint": "..."})
```

### Unified Configuration

```python
from cogniverse_foundation.config import UnifiedConfig

# Load unified configuration for a component
unified_config = UnifiedConfig.from_file("config.yaml")

# Access different configuration sections
agent_config = unified_config.agent
telemetry_config = unified_config.telemetry
backend_config = unified_config.backend

# Validate configuration
unified_config.validate()
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
- SQLite storage tests

## License

MIT
