# Configuration Management System

**Spans**: SDK (interfaces), Foundation (base), Core (system configuration)

Multi-tenant, versioned configuration system with pluggable storage backends for the Cogniverse layered architecture.

## Overview

The configuration system provides centralized management for all system configurations with:

- **Multi-tenant isolation**: Complete configuration separation per tenant
- **Versioning**: Full history tracking with rollback capability
- **Pluggable backends**: VespaConfigStore (default), or custom implementations via ConfigStore interface
- **Hot reload**: Configuration changes apply immediately without restart
- **Type-safe schemas**: Strongly typed configuration dataclasses
- **DSPy integration**: Dynamic optimizer and module configuration

## Architecture

See [System Architecture](architecture/overview.md) for the complete package structure.

The configuration system spans:

- **SDK Layer**: ConfigStore interface, type definitions
- **Foundation Layer**: Base config classes, serialization
- **Core Layer**: ConfigManager for system-wide configuration
- **Implementation Layer**: VespaConfigStore (default backend)

## Configuration Scopes

### System Configuration

Global infrastructure settings shared across all agents:

- LLM providers and models
- Backend URLs and connection settings
- Telemetry endpoints
- Memory configuration
- Default resource limits

### Agent Configuration

Per-agent DSPy module and optimizer settings:

- DSPy module types (ChainOfThought, ReAct, etc.)
- Optimizer selection (Bootstrap, SIMBA, MIPRO, GEPA)
- Model-specific parameters
- Prompt templates and signatures
- Resource allocations

### Routing Configuration

Routing optimizer and strategy settings:

- Routing tiers (FAST, BALANCED, COMPREHENSIVE)
- Strategy weights and thresholds
- Experience buffer configuration
- GEPA optimizer parameters
- Multi-tenant routing rules

### Telemetry Configuration

Observability settings:

- Project isolation per tenant
- Span export configuration
- Experiment tracking settings
- Metric collection intervals
- Dashboard customization

### Backend Configuration

Backend-specific settings for video processing and storage:

- Backend type (vespa, elasticsearch, etc.)
- Backend connection parameters (URL, port)
- Profile-based video processing configuration
- Pipeline settings (frame extraction, transcription)
- Embedding strategies and models
- Per-tenant backend overrides

### ConfigScope Enum

Configuration scopes are defined by the `ConfigScope` enum:

```python
from cogniverse_sdk.interfaces.config_store import ConfigScope

class ConfigScope(Enum):
    SYSTEM = "system"        # Global infrastructure settings
    AGENT = "agent"          # Per-agent DSPy configuration
    ROUTING = "routing"      # Routing optimizer settings
    TELEMETRY = "telemetry"  # Observability settings
    SCHEMA = "schema"        # Schema configuration
    BACKEND = "backend"      # Backend and profile configuration
```

**Usage Example**:
```python
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_sdk.interfaces.config_store import ConfigScope

manager = create_default_config_manager()

# Set backend configuration for tenant
from cogniverse_foundation.config.unified_config import BackendConfig

backend_config = BackendConfig(
    tenant_id="acme",
    backend_type="vespa",
    url="http://vespa-cluster",
    port=8080
)

manager.set_backend_config(backend_config, tenant_id="acme")

# Retrieve backend configuration
config = manager.get_backend_config(tenant_id="acme")
print(f"Backend: {config.url}:{config.port}")
```

## Storage Backends

The configuration system uses a pluggable backend architecture. Any storage backend implementing the `ConfigStore` interface can be used.

### Backend Implementation (Vespa Example)

The default implementation uses Vespa for unified configuration storage alongside application data:

```python
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_vespa.config.config_store import VespaConfigStore

# Initialize backend store
store = VespaConfigStore(
    vespa_app=None,  # Optional: pass existing Vespa app instance
    backend_url="http://localhost",
    backend_port=8080,
    schema_name="config_metadata"
)

# Use with ConfigManager
manager = ConfigManager(store=store)

# Get system configuration
system_config = manager.get_system_config(tenant_id="default")
print(f"LLM: {system_config.llm_model}")
print(f"Backend: {system_config.backend_url}")
```

**Default Initialization:**
```python
from cogniverse_foundation.config.utils import create_default_config_manager

# Automatically uses default backend with settings from environment
# Reads BACKEND_URL and BACKEND_PORT from environment variables
manager = create_default_config_manager()
```

**Schema Deployment (Vespa):**

The `config_metadata` schema is automatically deployed as part of the metadata schemas:

```python
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

schema_manager = VespaSchemaManager(
    backend_endpoint="http://localhost",
    backend_port=19071  # Config server port
)

# Deploys organization_metadata, tenant_metadata, config_metadata, adapter_registry
schema_manager.upload_metadata_schemas(app_name="videosearch")
```

**Backend Features:**

- High availability and replication
- Unified storage with application data (no separate database)
- Real-time configuration sync
- Horizontal scaling
- Multi-tenant isolation via tenant_id field
- Version tracking for configuration history

### Custom Backend Implementation

Create custom storage backends by implementing the ConfigStore interface:

```python
from cogniverse_sdk.interfaces.config_store import ConfigStore, ConfigEntry, ConfigScope
from typing import Dict, Any, Optional, List
from datetime import datetime

class RedisConfigStore(ConfigStore):
    """Redis-based configuration storage"""

    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.initialize()

    def initialize(self) -> None:
        """Setup Redis indexes and structures"""
        # Create sorted sets for versioning
        # Setup pub/sub for hot reload
        pass

    def set_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        config_value: Dict[str, Any],
    ) -> ConfigEntry:
        """Store configuration with versioning"""
        # Generate version number
        # Store in Redis with TTL
        # Publish update event
        pass

    def get_config(
        self,
        tenant_id: str,
        scope: ConfigScope,
        service: str,
        config_key: str,
        version: Optional[int] = None,
    ) -> Optional[ConfigEntry]:
        """Retrieve configuration by version"""
        # Get from Redis cache
        # Deserialize JSON
        # Return ConfigEntry
        pass

    # Implement remaining abstract methods...
```

## Multi-Tenant Configuration

### Tenant Isolation

Each tenant has completely isolated configuration:

```python
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_foundation.config.unified_config import SystemConfig

manager = create_default_config_manager()

# Configure Tenant A
tenant_a_config = SystemConfig(
    tenant_id="tenant_a",
    llm_model="gpt-4",
    base_url="https://api.openai.com/v1",
    backend_url="http://backend-tenant-a:8080",
    telemetry_url="http://phoenix-tenant-a:6006"
)
manager.set_system_config(tenant_a_config, tenant_id="tenant_a")

# Configure Tenant B
tenant_b_config = SystemConfig(
    tenant_id="tenant_b",
    llm_model="claude-3-opus",
    base_url="https://api.anthropic.com",
    backend_url="http://backend-tenant-b:8080",
    telemetry_url="http://phoenix-tenant-b:6006"
)
manager.set_system_config(tenant_b_config, tenant_id="tenant_b")

# Configurations are completely isolated
config_a = manager.get_system_config("tenant_a")
config_b = manager.get_system_config("tenant_b")
assert config_a.llm_model != config_b.llm_model
```

### Tenant Lifecycle Management

Tenant creation is handled via the runtime admin API:

```bash
# Create a new tenant via admin API
curl -X POST http://localhost:8000/admin/tenants \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "acme:production", "created_by": "admin"}'
```

For schema cleanup, use VespaSchemaManager with SchemaRegistry:

```python
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
from cogniverse_core.registries.schema_registry import SchemaRegistry
from cogniverse_foundation.config.utils import create_default_config_manager

# SchemaRegistry is required for tenant schema operations
config_manager = create_default_config_manager()
schema_registry = SchemaRegistry(config_manager, backend, schema_loader)

schema_manager = VespaSchemaManager(
    backend_endpoint="http://localhost",
    backend_port=19071,
    schema_registry=schema_registry  # Required for delete_tenant_schemas and tenant_schema_exists
)

# Delete all schemas for a tenant
deleted = schema_manager.delete_tenant_schemas(tenant_id="old_tenant")
print(f"Deleted schemas: {deleted}")

# Check if tenant schema exists
exists = schema_manager.tenant_schema_exists(
    tenant_id="acme",
    base_schema_name="video_frames"
)
```

For programmatic tenant configuration, create a new SystemConfig for each tenant:

```python
from cogniverse_foundation.config.unified_config import SystemConfig

# Clone configuration from existing tenant
source_config = manager.get_system_config("tenant_a")

# Create new config based on source
import dataclasses
new_config_dict = dataclasses.asdict(source_config)
new_config_dict["tenant_id"] = "tenant_a_staging"
new_config = SystemConfig(**new_config_dict)

manager.set_system_config(new_config, tenant_id="tenant_a_staging")
```

## DSPy Integration

### Dynamic Module Configuration

```python
from cogniverse_foundation.config.agent_config import (
    AgentConfig, ModuleConfig, DSPyModuleType, OptimizerConfig, OptimizerType
)
from cogniverse_foundation.config.utils import create_default_config_manager

manager = create_default_config_manager()

# Configure Video Search Agent with ReAct and GEPA optimizer
video_agent_config = AgentConfig(
    agent_name="video_search_agent",
    agent_version="1.0.0",
    agent_description="Video search and analysis agent",
    agent_url="http://localhost:8002",
    capabilities=["video_search", "visual_analysis"],
    skills=[],
    module_config=ModuleConfig(
        module_type=DSPyModuleType.REACT,  # Available: PREDICT, CHAIN_OF_THOUGHT, REACT, MULTI_CHAIN_COMPARISON, PROGRAM_OF_THOUGHT
        signature="Question -> Answer",
        max_retries=3,
        temperature=0.7
    ),
    optimizer_config=OptimizerConfig(
        optimizer_type=OptimizerType.GEPA,  # Available: BOOTSTRAP_FEW_SHOT, LABELED_FEW_SHOT, BOOTSTRAP_FEW_SHOT_WITH_RANDOM_SEARCH, COPRO, MIPRO_V2, GEPA, SIMBA
        num_trials=20,
        max_bootstrapped_demos=4
    ),
    llm_model="gpt-4",
    llm_temperature=0.7
)

manager.set_agent_config(
    tenant_id="default",
    agent_name="video_search_agent",
    agent_config=video_agent_config
)
```

### Centralized LLM Configuration

All DSPy-based agents and optimizers use a centralized LLM configuration system instead of reading environment variables or configuring `dspy.settings` globally.

**Config structure** (`config.json`):

```json
{
  "llm_config": {
    "primary": {
      "provider": "ollama",
      "model": "ollama_chat/smollm3:3b",
      "api_base": "http://localhost:11434"
    },
    "teacher": {
      "provider": "anthropic",
      "model": "claude-3-5-sonnet-20241022",
      "api_key_env": "ROUTER_OPTIMIZER_TEACHER_KEY"
    },
    "overrides": {
      "orchestrator_agent": {
        "model": "ollama_chat/qwen3:8b"
      }
    }
  }
}
```

**Key classes and factory**:

```python
from cogniverse_foundation.config.unified_config import LLMConfig, LLMEndpointConfig
from cogniverse_foundation.config.llm_factory import create_dspy_lm

# Load from config.json
llm_config = LLMConfig.from_dict(config.get("llm_config", {}))

# Resolve endpoint for a specific component (checks overrides, falls back to primary)
endpoint = llm_config.resolve("orchestrator_agent")  # Returns LLMEndpointConfig

# Create a scoped DSPy LM instance via the factory
lm = create_dspy_lm(endpoint)

# Use with scoped context (never global dspy.settings.configure)
import dspy
with dspy.context(lm=lm):
    result = module(query="machine learning videos")
```

- `LLMEndpointConfig`: Dataclass with `provider`, `model`, `api_base`, `api_key`, `api_key_env`, `temperature`, `max_tokens`
- `LLMConfig`: Holds `primary`, `teacher`, and `overrides` dict. `resolve(component_name)` returns the override if present, else `primary`
- `create_dspy_lm(config: LLMEndpointConfig) -> dspy.LM`: Factory that creates a DSPy LM from endpoint config. Resolves `api_key_env` to actual key. All DSPy LM creation goes through this factory

## Backend Configuration API

The ConfigManager provides methods for managing backend and profile configurations:

### Get/Set Backend Configuration

```python
from cogniverse_foundation.config.utils import create_default_config_manager

manager = create_default_config_manager()

# Get backend configuration for a tenant
backend_config = manager.get_backend_config(tenant_id="acme")
print(f"Backend URL: {backend_config.url}")
print(f"Backend Port: {backend_config.port}")

# Set backend configuration
from cogniverse_foundation.config.unified_config import BackendConfig

new_config = BackendConfig(
    backend_type="vespa",
    url="http://vespa-cluster",
    port=8080
)
manager.set_backend_config(new_config, tenant_id="acme")
```

### Profile Management

```python
# List all profiles for a tenant
profiles = manager.list_backend_profiles(tenant_id="acme")
for name, profile in profiles.items():
    print(f"Profile: {name}, Schema: {profile.schema_name}")

# Get a specific profile
profile = manager.get_backend_profile(
    profile_name="video_colpali_smol500_mv_frame",
    tenant_id="acme"
)

# Add a new profile
from cogniverse_foundation.config.unified_config import BackendProfileConfig

new_profile = BackendProfileConfig(
    profile_name="custom_profile",
    schema_name="custom_schema_acme",
    embedding_model="colpali",
    model_specific={"dimensions": 128}
)
manager.add_backend_profile(new_profile, tenant_id="acme")

# Update an existing profile
manager.update_backend_profile(
    profile_name="custom_profile",
    overrides={"model_specific": {"dimensions": 256}},
    base_tenant_id="acme"
)

# Delete a profile
manager.delete_backend_profile(profile_name="custom_profile", tenant_id="acme")
```

## Configuration Versioning

### Version Tracking

Every configuration change creates a new version:

```python
from cogniverse_sdk.interfaces.config_store import ConfigScope

# Get configuration history
history = manager.store.get_config_history(
    tenant_id="default",
    scope=ConfigScope.SYSTEM,
    service="system",
    config_key="system_config",
    limit=10
)

for entry in history:
    print(f"Version {entry.version}:")
    print(f"  Updated: {entry.updated_at}")
    print(f"  Changes: {entry.config_value}")
```

### Rollback Capability

Rollback is achieved by retrieving a previous version from history and re-applying it:

```python
from cogniverse_sdk.interfaces.config_store import ConfigScope

# Get current version
current = manager.get_system_config("default")
print(f"Current LLM: {current.llm_model}")

# Get configuration history to find version to restore
history = manager.store.get_config_history(
    tenant_id="default",
    scope=ConfigScope.SYSTEM,
    service="system",
    config_key="system_config",
    limit=10
)

# Find the target version (e.g., version 5)
target_entry = next((e for e in history if e.version == 5), None)
if target_entry:
    # Re-apply the historical configuration
    manager.store.set_config(
        tenant_id="default",
        scope=ConfigScope.SYSTEM,
        service="system",
        config_key="system_config",
        config_value=target_entry.config_value
    )
    print(f"Rolled back to version {target_entry.version}")

# Verify rollback
rolled_back = manager.get_system_config("default")
print(f"Rolled back LLM: {rolled_back.llm_model}")
```

## Export/Import

### Backup Configuration

```python
import json
from datetime import datetime

# Export all configurations
export_data = manager.store.export_configs(
    tenant_id="production",
    include_history=True  # Include version history
)

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"config_backup_{timestamp}.json", "w") as f:
    json.dump(export_data, f, indent=2, default=str)

print(f"Exported {len(export_data['configs'])} configurations")
```

### Restore Configuration

```python
# Load backup
with open("config_backup_20250104_120000.json", "r") as f:
    backup_data = json.load(f)

# Import to new environment
imported_count = manager.store.import_configs(
    tenant_id="staging",
    configs=backup_data
)

print(f"Imported {imported_count} configurations")
```

## Monitoring and Health

### Configuration Health Checks

```python
# Check storage backend health
if manager.store.health_check():
    print("✓ Configuration storage healthy")
else:
    print("✗ Configuration storage unavailable")

# Get storage statistics
stats = manager.store.get_stats()
print(f"Total configurations: {stats['total_configs']}")
print(f"Total versions: {stats['total_versions']}")
print(f"Total tenants: {stats['total_tenants']}")
print(f"Configs by scope: {stats['configs_per_scope']}")
```

## Testing

### Unit Tests

```bash
# Run all configuration tests
JAX_PLATFORM_NAME=cpu uv run pytest tests/common/unit/test_config_*.py -v

# Test specific backend
JAX_PLATFORM_NAME=cpu uv run pytest tests/common/unit/test_vespa_config_store.py -v
```

### Integration Tests

```bash
# Test with real backends
docker compose -f deployment/docker-compose.yml up -d vespa

# Run integration tests
JAX_PLATFORM_NAME=cpu uv run pytest tests/common/integration/ -v
```

### Load Testing

Load testing can be implemented using standard Python tools:

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

async def load_test_config(manager, tenant_id: str, iterations: int = 100):
    """Simple load test for configuration reads."""
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(manager.get_system_config, tenant_id)
            for _ in range(iterations)
        ]
        results = [f.result() for f in futures]

    elapsed = time.perf_counter() - start
    print(f"Completed {iterations} reads in {elapsed:.2f}s")
    print(f"Read QPS: {iterations / elapsed:.1f}")
```

## Best Practices

### 1. Always Use Tenant Context
```python
# Good: Explicit tenant ID
config = manager.get_system_config(tenant_id="production")

# Bad: Implicit default tenant
config = manager.get_system_config()  # Uses "default"
```

### 2. Version Critical Changes
```python
# Before major changes
backup = manager.store.export_configs(
    tenant_id="production",
    include_history=True
)

# Make changes with audit trail (metadata is a field on SystemConfig)
new_config.metadata = {"changed_by": "admin", "reason": "Performance tuning"}
manager.set_system_config(new_config, tenant_id="production")
```

### 3. Use Type-Safe Configurations
```python
# Good: Type-safe dataclass
from cogniverse_foundation.config.unified_config import SystemConfig
config = SystemConfig(
    tenant_id="prod",
    llm_model="gpt-4",
    backend_url="http://backend:8080"
)

# Bad: Raw dictionaries
config = {"tenant_id": "prod", "llm_model": "gpt-4"}  # No validation
```

### 4. Implement Configuration Templates
```python
from cogniverse_foundation.config.unified_config import SystemConfig

# Define reusable templates
TEMPLATES = {
    "development": SystemConfig(
        tenant_id="",  # Set per-tenant
        llm_model="gpt-3.5-turbo",
        backend_url="http://localhost:8080"
    ),
    "production": SystemConfig(
        tenant_id="",
        llm_model="gpt-4",
        backend_url="http://backend-cluster:8080"
    )
}

# Apply template with overrides
def apply_template(manager, tenant_id: str, template_name: str, **overrides):
    """Apply a configuration template with optional overrides."""
    import dataclasses
    template = TEMPLATES[template_name]
    config_dict = dataclasses.asdict(template)
    config_dict["tenant_id"] = tenant_id
    config_dict.update(overrides)
    new_config = SystemConfig(**config_dict)
    manager.set_system_config(new_config, tenant_id=tenant_id)

# Usage
apply_template(manager, "new_customer", "production", llm_model="claude-3-opus")
```

## Migration Guide

### From Environment Variables

```python
# Old: Environment variables
import os
llm_model = os.getenv("LLM_MODEL", "gpt-4")
backend_url = os.getenv("BACKEND_URL", "http://localhost:8080")

# New: ConfigManager
from cogniverse_foundation.config.utils import create_default_config_manager
manager = create_default_config_manager()
config = manager.get_system_config("default")
llm_model = config.llm_model
backend_url = config.backend_url
```

### From Static Config Files

```python
# Old: Static YAML/JSON
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# New: Dynamic configuration
manager = create_default_config_manager()
config = manager.get_system_config("default")
# Hot reload supported automatically
```

## Troubleshooting

### Configuration Not Found

```python
from cogniverse_sdk.interfaces.config_store import ConfigScope
from cogniverse_foundation.config.unified_config import SystemConfig

# Check if configuration exists
configs = manager.store.list_configs(
    tenant_id="default",
    scope=ConfigScope.SYSTEM
)
print(f"Available configs: {configs}")

# Initialize missing configuration if needed
try:
    config = manager.get_system_config("default")
except Exception:
    manager.set_system_config(SystemConfig(tenant_id="default"), tenant_id="default")
```

### Version Conflicts

Configuration versioning is tracked automatically via the `get_config_history` method:

```python
# Check version history before updates
history = manager.store.get_config_history(
    tenant_id="default",
    scope=ConfigScope.SYSTEM,
    service="system",
    config_key="system_config",
    limit=5
)

# Log current version before update
if history:
    print(f"Current version: {history[0].version}")

# Make update (creates new version automatically)
manager.set_system_config(config, tenant_id="default")
```

### Storage Backend Issues

```python
from cogniverse_vespa.config.config_store import VespaConfigStore
from cogniverse_foundation.config.manager import ConfigManager

# Ensure backend is available before creating ConfigManager
try:
    store = VespaConfigStore(
        vespa_app=None,
        backend_url="http://localhost",
        backend_port=8080,
        schema_name="config_metadata"
    )
    manager = ConfigManager(store=store)
except ConnectionError as e:
    raise RuntimeError(
        f"Backend unavailable: {e}. "
        "Ensure the backend is running and metadata schemas are deployed."
    )
```

## Configuration Layer Details

### SDK Layer (cogniverse-sdk)

- Defines configuration interfaces and type contracts
- No implementation, just pure interfaces
- Used by all other layers for type safety

### Foundation Layer (cogniverse-foundation)

- Implements base ConfigStore interface
- Provides common configuration utilities
- Handles serialization/deserialization

### Core Layer (cogniverse-core)

- Implements ConfigManager for system-wide configuration
- Orchestrates configuration across all components
- Manages tenant isolation and versioning

## Related Documentation

- [Architecture Overview](architecture/overview.md) - System design
- [Multi-Tenant Architecture](architecture/multi-tenant.md) - Tenant isolation
- [Agents Module](modules/agents.md) - Agent configuration
- [Optimization Module](modules/optimization.md) - DSPy optimizer configuration

