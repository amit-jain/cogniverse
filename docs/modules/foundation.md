# Foundation Module

**Package:** `cogniverse_foundation`
**Location:** `libs/foundation/cogniverse_foundation/`
**Purpose:** Configuration management and telemetry infrastructure with multi-tenant support
**Last Updated:** 2026-01-25

---

## Table of Contents

1. [Overview](#overview)
2. [Package Structure](#package-structure)
3. [Configuration System](#configuration-system)
   - [ConfigManager](#configmanager)
   - [Configuration Types](#configuration-types)
   - [Configuration Scopes](#configuration-scopes)
4. [Telemetry System](#telemetry-system)
   - [TelemetryManager](#telemetrymanager)
   - [Span Context](#span-context)
   - [Session Tracking](#session-tracking)
5. [Usage Examples](#usage-examples)
6. [Architecture Position](#architecture-position)
7. [Testing](#testing)

---

## Overview

The Foundation module provides **infrastructure services** that all other modules depend on:

- **Configuration Management**: Multi-tenant, versioned configuration with SQLite persistence
- **Telemetry Infrastructure**: OpenTelemetry-based tracing with tenant isolation
- **Provider Abstraction**: Pluggable backends for telemetry (Phoenix, etc.)

All configuration and telemetry operations are **tenant-aware** - `tenant_id` is required for all operations.

---

## Package Structure

```
cogniverse_foundation/
├── config/                      # Configuration system
│   ├── manager.py               # ConfigManager - central config API
│   ├── unified_config.py        # SystemConfig, BackendConfig, etc.
│   ├── agent_config.py          # AgentConfig for DSPy/LLM settings
│   ├── schema.py                # Configuration schemas
│   ├── utils.py                 # Configuration utilities
│   ├── api_mixin.py             # API configuration mixin
│   └── sqlite/                  # SQLite-based config storage
│       └── config_store.py      # ConfigStore implementation
├── telemetry/                   # Telemetry system
│   ├── manager.py               # TelemetryManager - central telemetry API
│   ├── config.py                # TelemetryConfig
│   ├── registry.py              # Telemetry provider registry
│   ├── exporter.py              # Span exporters
│   ├── context.py               # Trace context management
│   └── providers/               # Telemetry provider implementations
│       ├── base.py              # Base provider interface
│       └── __init__.py
└── __init__.py
```

---

## Configuration System

### ConfigManager

`ConfigManager` is the central configuration API. All configuration operations go through this class.

**Key Features:**
- Multi-tenant configuration with tenant isolation
- Version history tracking for all configurations
- LRU caching for performance
- SQLite persistence via `ConfigStore`

```python
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.sqlite.config_store import SQLiteConfigStore

# Initialize with SQLite store
store = SQLiteConfigStore(db_path="config/cogniverse.db")
config_manager = ConfigManager(store=store, cache_size=100)

# Get system configuration for tenant
system_config = config_manager.get_system_config(tenant_id="acme")

# Set agent configuration
config_manager.set_agent_config(
    tenant_id="acme",
    agent_name="routing_agent",
    agent_config=agent_config
)
```

**API Reference:**

| Method | Description |
|--------|-------------|
| `get_system_config(tenant_id)` | Get system configuration |
| `set_system_config(config, tenant_id)` | Set system configuration |
| `get_agent_config(tenant_id, agent_name)` | Get agent configuration |
| `set_agent_config(tenant_id, agent_name, config)` | Set agent configuration |
| `get_agent_config_history(tenant_id, agent_name, limit)` | Get config version history |
| `get_routing_config(tenant_id, service)` | Get routing configuration |
| `set_routing_config(config, tenant_id, service)` | Set routing configuration |
| `get_telemetry_config(tenant_id, service)` | Get telemetry configuration |
| `set_telemetry_config(config, tenant_id, service)` | Set telemetry configuration |
| `get_backend_config(tenant_id, service)` | Get backend configuration |
| `set_backend_config(config, tenant_id, service)` | Set backend configuration |
| `get_backend_profile(profile_name, tenant_id)` | Get specific backend profile |
| `add_backend_profile(profile, tenant_id)` | Add/update backend profile |
| `update_backend_profile(name, overrides, base_tenant, target_tenant)` | Partial profile update |
| `list_backend_profiles(tenant_id)` | List all backend profiles |
| `delete_backend_profile(name, tenant_id)` | Delete backend profile |
| `export_configs(tenant_id, output_path)` | Export all configs to JSON |
| `get_stats()` | Get configuration statistics |

### Configuration Types

**SystemConfig** - Infrastructure settings:
```python
from cogniverse_foundation.config.unified_config import SystemConfig

system_config = SystemConfig(
    tenant_id="acme",
    default_backend="vespa",
    agent_urls={
        "routing_agent": "http://localhost:8001",
        "search_agent": "http://localhost:8002",
    },
    max_concurrent_requests=100,
    timeout_seconds=30
)
```

**AgentConfig** - Agent-specific settings:
```python
from cogniverse_foundation.config.agent_config import AgentConfig

agent_config = AgentConfig(
    agent_name="routing_agent",
    dspy_module="ChainOfThought",
    llm_model="gpt-4",
    temperature=0.7,
    max_tokens=2000,
    optimizer="GEPA",
    optimizer_config={"num_trials": 50}
)
```

**BackendConfig** - Backend and profile settings:
```python
from cogniverse_foundation.config.unified_config import BackendConfig, BackendProfileConfig

backend_config = BackendConfig(
    tenant_id="acme",
    default_profile="video_colpali_mv_frame",
    profiles={
        "video_colpali_mv_frame": BackendProfileConfig(
            profile_name="video_colpali_mv_frame",
            embedding_model="colpali",
            chunk_strategy="frame",
            top_k=10
        )
    }
)
```

**RoutingConfigUnified** - Routing agent settings:
```python
from cogniverse_foundation.config.unified_config import RoutingConfigUnified

routing_config = RoutingConfigUnified(
    tenant_id="acme",
    tiers=["fast", "accurate", "comprehensive"],
    default_tier="accurate",
    optimization_enabled=True
)
```

**TelemetryConfigUnified** - Telemetry settings:
```python
from cogniverse_foundation.config.unified_config import TelemetryConfigUnified

telemetry_config = TelemetryConfigUnified(
    tenant_id="acme",
    enabled=True,
    phoenix_endpoint="http://localhost:6006",
    batch_export=True,
    sample_rate=1.0
)
```

### Configuration Scopes

Configurations are organized by scope for isolation:

| Scope | Description | Example Keys |
|-------|-------------|--------------|
| `SYSTEM` | Infrastructure settings | agent_urls, backends, timeouts |
| `AGENT` | Per-agent settings | DSPy config, LLM settings |
| `ROUTING` | Routing agent settings | tiers, strategies, optimization |
| `TELEMETRY` | Telemetry settings | Phoenix endpoint, sampling |
| `BACKEND` | Backend profiles | embedding models, search params |

```python
from cogniverse_sdk.interfaces.config_store import ConfigScope

# Get arbitrary config value by scope
value = config_manager.get_config_value(
    tenant_id="acme",
    scope=ConfigScope.AGENT,
    service="routing_agent",
    config_key="optimizer_config"
)
```

### Configuration Inheritance

The configuration system uses a layered inheritance model where tenant-specific settings override system defaults:

```mermaid
flowchart TB
    subgraph Sources["Configuration Sources"]
        EnvVars[Environment Variables<br/>COGNIVERSE_CONFIG, etc.]
        ConfigFile[config.json<br/>Auto-discovered]
        SQLite[SQLite Store<br/>Persisted configs]
    end

    subgraph Layers["Configuration Layers"]
        direction TB
        SystemDefaults[System Defaults<br/>Hardcoded fallbacks]
        GlobalConfig[Global Configuration<br/>config.json profiles]
        TenantOverlay[Tenant Overlay<br/>Per-tenant overrides]
        RuntimeOverride[Runtime Override<br/>API/query-time params]
    end

    subgraph Resolution["Resolution Order (Bottom Wins)"]
        Final[Final Configuration<br/>Merged result]
    end

    EnvVars --> GlobalConfig
    ConfigFile --> GlobalConfig
    SQLite --> TenantOverlay

    SystemDefaults --> Final
    GlobalConfig --> Final
    TenantOverlay --> Final
    RuntimeOverride --> Final

    style Sources fill:#e1f5ff
    style Layers fill:#fff4e1
    style Resolution fill:#e1ffe1
```

**Configuration Resolution Example:**

```python
# System default (hardcoded)
max_frames = 50

# Global config (config.json) - overrides default
"profiles": {
    "video_colpali_mv_frame": {
        "max_frames": 100
    }
}

# Tenant overlay (SQLite) - overrides global
config_manager.set_backend_config(
    tenant_id="premium_tenant",
    config=BackendConfig(profiles={
        "video_colpali_mv_frame": {"max_frames": 200}
    })
)

# Runtime override (query param) - overrides all
result = await search(query, max_frames=300)

# Final: premium_tenant gets max_frames=300 for this query
```

**Resolution Priority (highest to lowest):**

| Priority | Source | Scope | Example |
|----------|--------|-------|---------|
| 1 (highest) | Runtime Override | Per-request | Query params, API args |
| 2 | Tenant Overlay | Per-tenant | `ConfigManager.set_*_config()` |
| 3 | Global Config | All tenants | `config.json` profiles |
| 4 (lowest) | System Defaults | Fallback | Hardcoded in classes |

---

## Telemetry System

### TelemetryManager

`TelemetryManager` is a **singleton** that manages OpenTelemetry tracing with multi-tenant isolation.

**Key Features:**
- Tenant-isolated tracer providers
- LRU caching of tracers
- Graceful degradation when telemetry unavailable
- Session tracking for multi-turn conversations
- Phoenix integration for trace visualization

```python
from cogniverse_foundation.telemetry.manager import TelemetryManager, get_telemetry_manager

# Get global singleton
telemetry = get_telemetry_manager()

# Create span with tenant isolation
with telemetry.span("search.execute", tenant_id="acme") as span:
    span.set_attribute("query", "find videos about cats")
    # ... search logic ...
```

**API Reference:**

| Method | Description |
|--------|-------------|
| `span(name, tenant_id, project_name, attributes)` | Create tenant-isolated span |
| `session(tenant_id, session_id, project_name)` | Session context for multi-turn |
| `session_span(name, tenant_id, session_id, ...)` | Span within session context |
| `get_tracer(tenant_id, project_name)` | Get tracer (legacy) |
| `get_provider(tenant_id, project_name)` | Get telemetry provider for queries |
| `register_project(tenant_id, project_name, **kwargs)` | Register project with config |
| `unregister_project(tenant_id, project_name)` | Unregister and shutdown project |
| `force_flush(timeout_millis)` | Flush all pending spans |
| `shutdown()` | Graceful shutdown |
| `get_stats()` | Get telemetry statistics |

### Span Context

Creating spans with tenant isolation:

```python
from cogniverse_foundation.telemetry.manager import get_telemetry_manager

telemetry = get_telemetry_manager()

# Basic span
with telemetry.span("agent.process", tenant_id="acme") as span:
    span.set_attribute("agent.name", "routing_agent")
    span.set_attribute("query.length", len(query))
    result = await process_query(query)

# Span with project isolation (for management operations)
with telemetry.span(
    "experiment.run",
    tenant_id="acme",
    project_name="experiments",  # Separate Phoenix project
    attributes={"experiment.name": "optimizer_v2"}
) as span:
    await run_experiment()
```

### Session Tracking

Track multi-turn conversations across requests:

```python
# At API entry point - establish session context
with telemetry.session_span(
    "api.search.request",
    tenant_id="acme",
    session_id="user-session-abc123",
    attributes={"query": query, "turn": 3}
) as span:
    # All child spans inherit session_id
    result = await search_service.search(query)

# Alternative: wrap multiple operations in session
with telemetry.session(tenant_id="acme", session_id="session-xyz"):
    with telemetry.span("operation1", tenant_id="acme") as span1:
        pass
    with telemetry.span("operation2", tenant_id="acme") as span2:
        pass
    # Both spans share session_id
```

### Project Registration

Register projects with custom endpoints (useful for tests):

```python
# Register with default config
telemetry.register_project(
    tenant_id="acme",
    project_name="search"
)

# Register with custom endpoints (for tests)
telemetry.register_project(
    tenant_id="test-tenant",
    project_name="synthetic_data",
    otlp_endpoint="http://localhost:24317",
    http_endpoint="http://localhost:26006",
    use_sync_export=True  # Sync export for tests
)
```

---

## Usage Examples

### Complete Configuration Setup

```python
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.sqlite.config_store import SQLiteConfigStore
from cogniverse_foundation.config.unified_config import (
    SystemConfig,
    BackendConfig,
    BackendProfileConfig,
)
from cogniverse_foundation.config.agent_config import AgentConfig

# Initialize config manager
store = SQLiteConfigStore(db_path="config/cogniverse.db")
config_manager = ConfigManager(store=store)

# Set system config for tenant
system_config = SystemConfig(
    tenant_id="acme",
    default_backend="vespa",
    agent_urls={
        "routing_agent": "http://localhost:8001",
        "search_agent": "http://localhost:8002",
    }
)
config_manager.set_system_config(system_config)

# Add backend profile for tenant
profile = BackendProfileConfig(
    profile_name="custom_colpali",
    embedding_model="colpali-v2",
    chunk_strategy="frame",
    top_k=20
)
config_manager.add_backend_profile(profile, tenant_id="acme")

# Set agent config
agent_config = AgentConfig(
    agent_name="routing_agent",
    dspy_module="ChainOfThought",
    llm_model="gpt-4",
    temperature=0.7
)
config_manager.set_agent_config(
    tenant_id="acme",
    agent_name="routing_agent",
    agent_config=agent_config
)
```

### Tenant-Specific Profile Overrides

```python
# Start with system profile, customize for tenant
config_manager.update_backend_profile(
    profile_name="video_colpali_mv_frame",
    overrides={
        "embedding_model": "colpali-custom",
        "top_k": 25
    },
    base_tenant_id="default",  # Get from system
    target_tenant_id="acme"    # Save to tenant
)
```

### Telemetry with Phoenix

```python
from cogniverse_foundation.telemetry.manager import get_telemetry_manager
from cogniverse_foundation.telemetry.config import TelemetryConfig

# Initialize with config
config = TelemetryConfig(
    enabled=True,
    otlp_endpoint="http://localhost:4317",
    service_name="cogniverse",
    environment="production"
)
telemetry = TelemetryManager(config=config)

# Use in agent processing
async def process_query(query: str, tenant_id: str):
    with telemetry.span(
        "agent.routing",
        tenant_id=tenant_id,
        attributes={"query": query}
    ) as span:
        # Route query
        route = await route_query(query)
        span.set_attribute("route.agent", route.agent_name)
        span.set_attribute("route.confidence", route.confidence)

        # Execute with nested span
        with telemetry.span(
            f"agent.{route.agent_name}",
            tenant_id=tenant_id
        ) as child_span:
            result = await execute_agent(route.agent_name, query)
            child_span.set_attribute("result.count", len(result.items))

        return result
```

### Querying Telemetry Data

```python
# Get provider for querying spans
provider = telemetry.get_provider(tenant_id="acme")

# Query spans from Phoenix
spans_df = await provider.traces.get_spans(
    project="cogniverse-acme",
    start_time=datetime(2025, 1, 1),
    limit=1000
)

# Add annotation
await provider.annotations.add_annotation(
    span_id="abc123",
    name="human_review",
    label="approved",
    score=1.0,
    metadata={"reviewer": "alice"}
)
```

---

## Architecture Position

```
┌─────────────────────────────────────────────────────────────────┐
│                        Core Layer                                │
│  cogniverse-core (agents) │ cogniverse-evaluation               │
└─────────────────────────────────────────────────────────────────┘
                                ↑
┌─────────────────────────────────────────────────────────────────┐
│                    Foundation Layer                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              cogniverse-foundation ◄─── YOU ARE HERE        ││
│  │  ConfigManager, TelemetryManager, Provider Registry         ││
│  └─────────────────────────────────────────────────────────────┘│
│                     cogniverse-sdk (interfaces)                  │
└─────────────────────────────────────────────────────────────────┘
```

**Dependencies:**
- `cogniverse-sdk`: Pure interfaces (ConfigStore, Backend, etc.)
- `opentelemetry-api/sdk`: Telemetry infrastructure
- `pydantic`: Configuration validation
- `sqlite3`: Configuration persistence

**Dependents:**
- `cogniverse-core`: Uses ConfigManager, TelemetryManager
- `cogniverse-agents`: Uses configuration and telemetry
- `cogniverse-telemetry-phoenix`: Implements telemetry provider

---

## Testing

```bash
# Run foundation tests
JAX_PLATFORM_NAME=cpu uv run pytest tests/foundation/ -v

# Test configuration
uv run pytest tests/foundation/test_config_manager.py -v

# Test telemetry
uv run pytest tests/foundation/test_telemetry_manager.py -v

# Test with coverage
uv run pytest tests/foundation/ --cov=cogniverse_foundation --cov-report=html
```

**Test Categories:**
- `tests/foundation/unit/` - Unit tests for config and telemetry
- `tests/foundation/integration/` - Integration tests with SQLite

---

## Related Documentation

- [Core Module](./core.md) - Agent base classes that use configuration and telemetry
- [Configuration System Guide](../CONFIGURATION_SYSTEM.md) - Detailed configuration guide
- [Telemetry Module](./telemetry.md) - Phoenix provider implementation
- [Multi-Tenant Architecture](../architecture/multi-tenant.md) - Tenant isolation patterns

---

**Summary:** The Foundation module provides the infrastructure layer for Cogniverse. `ConfigManager` handles multi-tenant, versioned configuration with SQLite persistence. `TelemetryManager` provides OpenTelemetry tracing with tenant isolation and Phoenix integration. All operations require `tenant_id` to ensure proper multi-tenant isolation.
