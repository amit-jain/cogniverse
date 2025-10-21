# Configuration Management System

Multi-tenant, versioned configuration system with pluggable storage backends for the Cogniverse multi-agent architecture.

## Overview

The configuration system provides centralized management for all system configurations with:
- **Multi-tenant isolation**: Complete configuration separation per tenant
- **Versioning**: Full history tracking with rollback capability
- **Pluggable backends**: SQLite (default), Vespa, PostgreSQL, or custom
- **Hot reload**: Configuration changes apply immediately without restart
- **Type-safe schemas**: Strongly typed configuration dataclasses
- **DSPy integration**: Dynamic optimizer and module configuration

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Applications                       │
├─────────────┬──────────────┬────────────┬──────────┤
│ Composing   │ Video Search │  Routing   │ Phoenix  │
│   Agent     │    Agent     │ Optimizer  │ Telemetry│
└──────┬──────┴──────┬───────┴──────┬─────┴─────┬────┘
       │             │              │            │
       └─────────────┼──────────────┼────────────┘
                     ▼              ▼
            ┌─────────────────────────────┐
            │     ConfigManager           │
            │    (Singleton Access)       │
            └────────────┬────────────────┘
                        │
                        ▼
            ┌─────────────────────────────┐
            │     ConfigStore Interface   │
            └────────────┬────────────────┘
                        │
       ┌────────────────┼────────────────┬───────────────┐
       ▼                ▼                ▼               ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   SQLite     │ │    Vespa     │ │  PostgreSQL  │ │   Custom     │
│ ConfigStore  │ │ ConfigStore  │ │ ConfigStore  │ │   Backend    │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

## Configuration Scopes

### System Configuration
Global infrastructure settings shared across all agents:
- LLM providers and models
- Vespa backend URLs
- Phoenix telemetry endpoints
- Mem0 memory configuration
- Default resource limits

### Agent Configuration
Per-agent DSPy module and optimizer settings:
- DSPy module types (CoT, ReAct, etc.)
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
Phoenix observability settings:
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
from cogniverse_core.config.store_interface import ConfigScope

class ConfigScope(Enum):
    SYSTEM = "system"        # Global infrastructure settings
    AGENT = "agent"          # Per-agent DSPy configuration
    ROUTING = "routing"      # Routing optimizer settings
    TELEMETRY = "telemetry"  # Phoenix observability settings
    BACKEND = "backend"      # Backend and profile configuration
```

**Usage Example**:
```python
from cogniverse_core.config.config_manager import get_config_manager
from cogniverse_core.config.store_interface import ConfigScope

manager = get_config_manager()

# Set backend configuration for tenant
backend_config = {
    "backend": {
        "profiles": {
            "video_colpali_smol500_mv_frame": {
                "pipeline_config": {
                    "max_frames": 200
                }
            }
        }
    }
}

manager.set_config(
    tenant_id="acme",
    scope=ConfigScope.BACKEND,  # Use BACKEND scope
    config=backend_config
)

# Retrieve backend configuration
config = manager.get_config(
    tenant_id="acme",
    scope=ConfigScope.BACKEND
)
```

## Storage Backends

### SQLiteConfigStore (Default)

Local SQLite database for development and single-instance deployments.

```python
from cogniverse_core.config.config_manager import get_config_manager

# Automatically uses SQLite at data/config/config.db
manager = get_config_manager()

# Get system configuration
system_config = manager.get_system_config(tenant_id="default")
print(f"LLM: {system_config.llm_model}")
print(f"Vespa: {system_config.vespa_url}")
```

**Features:**
- Zero configuration setup
- File-based persistence
- Full ACID compliance
- Migration path to PostgreSQL

### VespaConfigStore

Unified configuration storage in Vespa alongside application data.

```python
from vespa.application import Vespa
from cogniverse_core.config.store import VespaConfigStore
from cogniverse_core.config.config_manager import ConfigManager

# Initialize Vespa store
vespa_app = Vespa(url="http://localhost:8080")
vespa_store = VespaConfigStore(
    vespa_app=vespa_app,
    namespace="config_metadata",
    document_type="configuration"
)

# Use with ConfigManager
manager = ConfigManager(store=vespa_store)
```

**Schema Deployment:**
```bash
# Deploy configuration schema
vespa deploy --wait 300 schemas/config_metadata/

# Verify deployment
curl http://localhost:8080/ApplicationStatus | jq '.schemas'
```

**Features:**
- Leverages Vespa's HA/replication
- Unified storage management
- Real-time configuration sync
- Scales with Vespa cluster

### Custom Backend Implementation

Create custom storage backends by implementing the ConfigStore interface:

```python
from cogniverse_core.config.store_interface import ConfigStore, ConfigEntry, ConfigScope
from typing import Dict, Any, Optional, List
import datetime

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
from cogniverse_core.config.config_manager import get_config_manager
from cogniverse_core.config.unified_config import SystemConfig

manager = get_config_manager()

# Configure Tenant A
tenant_a_config = SystemConfig(
    tenant_id="tenant_a",
    llm_model="gpt-4",
    llm_base_url="https://api.openai.com/v1",
    vespa_url="http://vespa-tenant-a:8080",
    phoenix_project_name="tenant_a_project"
)
manager.set_system_config(tenant_a_config)

# Configure Tenant B
tenant_b_config = SystemConfig(
    tenant_id="tenant_b",
    llm_model="claude-3-opus",
    llm_base_url="https://api.anthropic.com",
    vespa_url="http://vespa-tenant-b:8080",
    phoenix_project_name="tenant_b_project"
)
manager.set_system_config(tenant_b_config)

# Configurations are completely isolated
config_a = manager.get_system_config("tenant_a")
config_b = manager.get_system_config("tenant_b")
assert config_a.llm_model != config_b.llm_model
```

### Tenant Lifecycle Management

```python
# Create new tenant
manager.create_tenant(
    tenant_id="new_tenant",
    config_template="enterprise",  # Use predefined template
    overrides={
        "llm_model": "gpt-4-turbo",
        "max_qps": 100
    }
)

# Clone tenant configuration
manager.clone_tenant(
    source_tenant_id="tenant_a",
    target_tenant_id="tenant_a_staging"
)

# Delete tenant (soft delete, keeps history)
manager.delete_tenant("old_tenant", hard_delete=False)
```

## DSPy Integration

### Dynamic Module Configuration

```python
from cogniverse_core.config.agent_config import AgentConfig, ModuleConfig, DSPyModuleType
from cogniverse_core.config.config_manager import get_config_manager

manager = get_config_manager()

# Configure Video Search Agent with ReAct
video_agent_config = AgentConfig(
    agent_name="video_search_agent",
    module_config=ModuleConfig(
        module_type=DSPyModuleType.REACT,
        signature="Question -> Answer",
        max_iterations=5,
        tools=["vespa_search", "rerank", "summarize"]
    ),
    optimizer_config={
        "type": "GEPA",
        "learning_rate": 0.001,
        "buffer_size": 10000
    },
    llm_model="gpt-4",
    temperature=0.7
)

manager.set_agent_config(
    tenant_id="default",
    agent_name="video_search_agent",
    agent_config=video_agent_config
)
```

### Optimizer Selection

```python
from cogniverse_agents.routing.optimizer_factory import OptimizerFactory
from cogniverse_core.config.config_manager import get_config_manager

manager = get_config_manager()
factory = OptimizerFactory()

# Get routing configuration
routing_config = manager.get_routing_config("default")

# Select optimizer based on data availability
optimizer = factory.get_optimizer(
    config=routing_config,
    training_data_size=len(experience_buffer)
)

# Optimizer selection logic:
# < 100 samples: Bootstrap
# 100-1000 samples: SIMBA
# 1000-10000 samples: MIPRO
# > 10000 samples: GEPA
```

## Configuration Versioning

### Version Tracking

Every configuration change creates a new version:

```python
from cogniverse_core.config.store_interface import ConfigScope

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

```python
# Get current version
current = manager.get_system_config("default")
print(f"Current LLM: {current.llm_model}")

# Rollback to specific version
manager.rollback_config(
    tenant_id="default",
    scope=ConfigScope.SYSTEM,
    service="system",
    config_key="system_config",
    target_version=5
)

# Verify rollback
rolled_back = manager.get_system_config("default")
print(f"Rolled back LLM: {rolled_back.llm_model}")
```

## Hot Reload Support

Configuration changes apply immediately without restart:

```python
from cogniverse_core.config.config_watcher import ConfigWatcher

# Setup configuration watcher
watcher = ConfigWatcher(manager)

# Register callback for configuration changes
def on_config_change(tenant_id: str, scope: str, config: dict):
    print(f"Configuration updated for {tenant_id}/{scope}")
    # Reload affected components
    if scope == "AGENT":
        reload_agent(config)
    elif scope == "ROUTING":
        update_routing_strategy(config)

watcher.register_callback(on_config_change)
watcher.start()

# Changes are detected and applied automatically
manager.set_system_config(updated_config)
# Callback fires immediately
```

## Export/Import

### Backup Configuration

```python
import json
from datetime import datetime

# Export all configurations
export_data = manager.store.export_configs(
    tenant_id="production",
    include_history=True,
    include_versions=True
)

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"config_backup_{timestamp}.json", "w") as f:
    json.dump(export_data, f, indent=2, default=str)

print(f"Exported {len(export_data['configs'])} configurations")
print(f"Total versions: {export_data['total_versions']}")
```

### Restore Configuration

```python
# Load backup
with open("config_backup_20250104_120000.json", "r") as f:
    backup_data = json.load(f)

# Import to new environment
imported_count = manager.store.import_configs(
    tenant_id="staging",
    configs=backup_data,
    overwrite_existing=False,  # Skip existing configs
    preserve_versions=True     # Keep version numbers
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
print(f"Total tenants: {stats['total_tenants']}")
print(f"Storage size: {stats['storage_size_mb']} MB")
print(f"Configs by scope: {stats['configs_per_scope']}")
```

### Configuration Metrics

```python
from cogniverse_core.telemetry.metrics_manager import MetricsManager

metrics = MetricsManager()

# Track configuration changes
@metrics.track_config_change
def update_config(tenant_id: str, config: dict):
    manager.set_system_config(config)

# Metrics exposed:
# - config_changes_total
# - config_rollbacks_total
# - config_errors_total
# - config_latency_seconds
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
docker compose up -d vespa postgres redis

# Run integration tests
JAX_PLATFORM_NAME=cpu uv run pytest tests/common/integration/ -v
```

### Load Testing

```python
# Test configuration performance under load
from tests.load.config_load_test import ConfigLoadTest

test = ConfigLoadTest(manager)
results = test.run(
    concurrent_readers=100,
    concurrent_writers=10,
    duration_seconds=60
)

print(f"Read QPS: {results['read_qps']}")
print(f"Write QPS: {results['write_qps']}")
print(f"P95 latency: {results['p95_latency_ms']}ms")
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

# Make changes with audit trail
manager.set_system_config(
    new_config,
    metadata={"changed_by": "admin", "reason": "Performance tuning"}
)
```

### 3. Use Type-Safe Configurations
```python
# Good: Type-safe dataclass
from cogniverse_core.config.unified_config import SystemConfig
config = SystemConfig(
    tenant_id="prod",
    llm_model="gpt-4",
    vespa_url="http://vespa:8080"
)

# Bad: Raw dictionaries
config = {"tenant_id": "prod", "llm_model": "gpt-4"}  # No validation
```

### 4. Implement Configuration Templates
```python
# Define reusable templates
TEMPLATES = {
    "development": {
        "llm_model": "gpt-3.5-turbo",
        "max_qps": 10,
        "cache_ttl": 300
    },
    "production": {
        "llm_model": "gpt-4",
        "max_qps": 100,
        "cache_ttl": 3600
    }
}

# Apply template with overrides
manager.apply_template(
    tenant_id="new_customer",
    template_name="production",
    overrides={"llm_model": "claude-3-opus"}
)
```

## Migration Guide

### From Environment Variables

```python
# Old: Environment variables
import os
llm_model = os.getenv("LLM_MODEL", "gpt-4")
vespa_url = os.getenv("VESPA_URL", "http://localhost:8080")

# New: ConfigManager
from cogniverse_core.config.config_manager import get_config_manager
manager = get_config_manager()
config = manager.get_system_config("default")
llm_model = config.llm_model
vespa_url = config.vespa_url
```

### From Static Config Files

```python
# Old: Static YAML/JSON
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# New: Dynamic configuration
manager = get_config_manager()
config = manager.get_system_config("default")
# Hot reload supported automatically
```

## Troubleshooting

### Configuration Not Found

```python
# Check if configuration exists
configs = manager.store.list_configs(
    tenant_id="default",
    scope=ConfigScope.SYSTEM
)
print(f"Available configs: {configs}")

# Initialize missing configuration
if not manager.has_system_config("default"):
    manager.set_system_config(SystemConfig(tenant_id="default"))
```

### Version Conflicts

```python
# Handle concurrent updates
try:
    manager.set_system_config(
        config,
        expected_version=current_version,  # Optimistic locking
        force=False
    )
except VersionConflictError:
    # Reload and retry
    latest = manager.get_system_config("default")
    # Merge changes and retry
```

### Storage Backend Issues

```python
# Fallback to local storage
try:
    manager = ConfigManager(store=VespaConfigStore())
except ConnectionError:
    print("Vespa unavailable, using local SQLite")
    manager = ConfigManager(store=SQLiteConfigStore())
```

## Related Documentation

- [Architecture Overview](architecture.md) - System design
- [Multi-Tenant System](multi-tenant-system.md) - Tenant isolation
- [Agent Orchestration](agent-orchestration.md) - Agent configuration
- [Optimization System](optimization-system.md) - DSPy optimizer configuration

---

**Last Updated**: 2025-10-04
**Status**: Production - Version 2.0