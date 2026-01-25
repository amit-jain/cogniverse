# Common Module Study Guide

**Package:** `cogniverse_core` (Core Layer)
**Location:** `libs/core/cogniverse_core/common/` and `libs/core/cogniverse_core/config/`
**Last Updated:** 2026-01-25
**Purpose:** Foundational infrastructure for configuration, memory, DSPy integration, and tenant utilities

---

## Table of Contents
1. [Module Overview](#module-overview)
2. [Package Architecture](#package-architecture)
3. [Configuration System](#configuration-system)
4. [Memory Management](#memory-management)
5. [Tenant Utilities](#tenant-utilities)
6. [Dynamic DSPy Integration](#dynamic-dspy-integration)
7. [Usage Examples](#usage-examples)
8. [Production Considerations](#production-considerations)
9. [Testing](#testing)

---

## Module Overview

### Purpose
The Common module provides foundational infrastructure shared across all system components in the SDK architecture. It includes multi-tenant configuration management, persistent memory systems, DSPy integration, and tenant isolation utilities.

### Key Features
- **Multi-Tenant Configuration**: SQLite-based versioned config with per-tenant isolation
- **Memory Management**: Mem0-based memory with schema-per-tenant isolation via Vespa
- **Tenant Utilities**: Org:tenant ID parsing and storage path management
- **DSPy Integration**: Runtime DSPy module/optimizer configuration
- **Config Store**: Versioned configuration persistence with history tracking
- **Type Definitions**: Shared data models and configuration schemas

### Package Structure
```
libs/core/cogniverse_core/
├── common/                           # Shared utilities (15 files)
│   ├── tenant_utils.py              # Tenant ID parsing and storage paths
│   ├── config_store.py              # SQLite config storage
│   ├── mem0_memory_manager.py       # Per-tenant memory manager
│   ├── dynamic_dspy_mixin.py        # DSPy runtime configuration
│   ├── dspy_module_registry.py      # DSPy module/optimizer registry
│   ├── a2a_mixin.py                 # Agent-to-agent communication
│   ├── a2a_utils.py                 # A2A utilities
│   ├── health_mixin.py              # Health check mixin
│   ├── vlm_interface.py             # Vision-language model interface
│   ├── agent_models.py              # Agent data models
│   ├── document.py                  # Document models
│   └── config_store_interface.py    # Config store interface
└── config/                           # Configuration schemas (10 files)
    ├── unified_config.py            # System/routing/telemetry configs
    ├── agent_config.py              # Agent-specific configuration
    ├── config_manager.py            # Configuration management
    ├── store.py                     # SQLite config store implementation
    ├── store_interface.py           # Storage abstraction
    ├── api_mixin.py                 # Config API mixin
    ├── utils.py                     # Config utilities
    └── schema.py                    # Config schemas
```

---

## Package Architecture

### Configuration System Architecture

```mermaid
graph TB
    subgraph "cogniverse_agents Package"
        AgentLayer[Agent Layer<br/>RoutingAgent, VideoSearchAgent, etc.]
    end

    subgraph "cogniverse_core Package"
        subgraph "Config Module"
            ConfigManager[Config Manager<br/>• SystemConfig - global settings<br/>• RoutingConfigUnified - per-tenant<br/>• TelemetryConfigUnified - per-tenant<br/>• AgentConfigUnified - per-agent]
        end

        subgraph "Common Module"
            ConfigStore[SQLite Config Store<br/>• Versioned persistence<br/>• History tracking<br/>• Tenant isolation]
        end
    end

    AgentLayer --> ConfigManager
    ConfigManager --> ConfigStore
    ConfigStore -.-> DB[(SQLite Database<br/>data/config/config.db)]

    style AgentLayer fill:#e1f5ff
    style ConfigManager fill:#fff4e1
    style ConfigStore fill:#ffe1e1
    style DB fill:#e1ffe1
```

### Memory Management Architecture

```mermaid
graph TB
    subgraph "cogniverse_agents Package"
        AgentLayer[Agents<br/>with MemoryAwareMixin]
    end

    subgraph "cogniverse_core Package"
        MemoryManager[Mem0MemoryManager<br/>Per-tenant singleton<br/>• add_memory<br/>• search_memory<br/>• get_all_memories<br/>• clear_agent_memory]
    end

    subgraph "cogniverse_vespa Package"
        SchemaManager[TenantSchemaManager<br/>Schema: agent_memories_tenant_id]
    end

    subgraph External
        Mem0Lib[Mem0 Library<br/>• LLM: Ollama llama3.2<br/>• Embedder: nomic-embed-text<br/>• Vector Store: Vespa]
        VespaDB[(Vespa<br/>Schema per tenant<br/>agent_memories_acme)]
    end

    AgentLayer --> MemoryManager
    MemoryManager --> SchemaManager
    MemoryManager --> Mem0Lib
    SchemaManager --> VespaDB
    Mem0Lib --> VespaDB

    style AgentLayer fill:#e1f5ff
    style MemoryManager fill:#fff4e1
    style SchemaManager fill:#ffe1e1
    style Mem0Lib fill:#e1ffe1
    style VespaDB fill:#d4edda
```

### Tenant Utilities Flow

```mermaid
graph LR
    TenantID["Tenant ID<br/>acme:production"]

    subgraph "cogniverse_core Package"
        Parse[tenant_utils.parse_tenant_id]
        StoragePath[tenant_utils.get_tenant_storage_path]
    end

    OrgTenant["org_id: acme<br/>tenant_name: production"]
    Path["Path:<br/>data/optimization/acme/production"]

    TenantID --> Parse --> OrgTenant
    TenantID --> StoragePath --> Path

    style TenantID fill:#e1f5ff
    style Parse fill:#fff4e1
    style StoragePath fill:#fff4e1
    style OrgTenant fill:#ffe1e1
    style Path fill:#d4edda
```

---

## Configuration System

### SystemConfig

**Location:** `libs/core/cogniverse_core/config/unified_config.py:29-115`

**Purpose:** System-level configuration for global settings

**Import:**
```python
from cogniverse_core.config.unified_config import SystemConfig
```

**Key Attributes:**
```python
@dataclass
class SystemConfig:
    tenant_id: str = "default"

    # Agent service URLs
    routing_agent_url: str = "http://localhost:8001"
    video_agent_url: str = "http://localhost:8002"
    text_agent_url: str = "http://localhost:8003"
    summarizer_agent_url: str = "http://localhost:8004"
    text_analysis_agent_url: str = "http://localhost:8005"

    # Search backend
    search_backend: str = "vespa"
    vespa_url: str = "http://localhost"
    vespa_port: int = 8080

    # LLM configuration
    llm_model: str = "ollama/gemma3:4b"
    base_url: str = "http://localhost:11434"
    llm_api_key: Optional[str] = None

    # Phoenix/Telemetry
    phoenix_url: str = "http://localhost:6006"
    phoenix_collector_endpoint: str = "localhost:4317"

    # Metadata
    environment: str = "development"
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Methods:**
- `to_dict() -> Dict[str, Any]` - Convert to dictionary
- `from_dict(data: Dict) -> SystemConfig` - Create from dictionary

**Usage:**
```python
from cogniverse_core.config.unified_config import SystemConfig

# Create system config
config = SystemConfig(
    tenant_id="acme",
    vespa_url="http://prod-vespa.example.com",
    vespa_port=8080,
    environment="production"
)

# Serialize
config_dict = config.to_dict()

# Deserialize
loaded_config = SystemConfig.from_dict(config_dict)
```

---

### RoutingConfigUnified

**Location:** `libs/core/cogniverse_core/config/unified_config.py:118-236`

**Purpose:** Per-tenant routing configuration

**Import:**
```python
from cogniverse_core.config.unified_config import RoutingConfigUnified
```

**Key Attributes:**
```python
@dataclass
class RoutingConfigUnified:
    tenant_id: str = "default"

    # Routing strategy
    routing_mode: str = "tiered"  # tiered, ensemble, hybrid

    # Tier thresholds
    enable_fast_path: bool = True
    enable_slow_path: bool = True
    fast_path_confidence_threshold: float = 0.7
    slow_path_confidence_threshold: float = 0.6
    max_routing_time_ms: int = 1000

    # GLiNER configuration (Fast Path)
    gliner_model: str = "urchade/gliner_large-v2.1"
    gliner_threshold: float = 0.3
    gliner_device: str = "cpu"
    gliner_labels: List[str] = field(default_factory=list)

    # LLM configuration (Slow Path)
    llm_provider: str = "local"
    llm_routing_model: str = "ollama/gemma3:4b"
    llm_endpoint: str = "http://localhost:11434"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 150
    use_chain_of_thought: bool = True

    # Optimization
    enable_auto_optimization: bool = True
    optimization_interval_seconds: int = 3600
    min_samples_for_optimization: int = 100

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    max_cache_size: int = 1000
```

**Usage:**
```python
from cogniverse_core.config.unified_config import RoutingConfigUnified

# Create tenant-specific routing config
config = RoutingConfigUnified(
    tenant_id="acme",
    routing_mode="ensemble",
    fast_path_confidence_threshold=0.8,
    enable_auto_optimization=True,
    cache_ttl_seconds=7200
)
```

---

### TelemetryConfigUnified

**Location:** `libs/core/cogniverse_core/config/unified_config.py:239-321`

**Purpose:** Per-tenant telemetry configuration

**Import:**
```python
from cogniverse_core.config.unified_config import TelemetryConfigUnified
```

**Key Attributes:**
```python
@dataclass
class TelemetryConfigUnified:
    tenant_id: str = "default"

    # Core settings
    enabled: bool = True
    level: str = "detailed"  # disabled, basic, detailed, verbose
    environment: str = "development"

    # Phoenix settings
    phoenix_enabled: bool = True
    phoenix_endpoint: str = "localhost:4317"
    phoenix_use_tls: bool = False

    # Multi-tenant settings
    tenant_project_template: str = "cogniverse-{tenant_id}-{service}"
    max_cached_tenants: int = 100
    tenant_cache_ttl_seconds: int = 3600

    # Batch export settings
    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    export_timeout_millis: int = 30000
    schedule_delay_millis: int = 500
    use_sync_export: bool = False
```

**Usage:**
```python
from cogniverse_core.config.unified_config import TelemetryConfigUnified

# Create tenant-specific telemetry config
config = TelemetryConfigUnified(
    tenant_id="acme",
    level="detailed",
    phoenix_enabled=True,
    phoenix_endpoint="prod-phoenix.internal:4317",
    tenant_project_template="cogniverse-{tenant_id}-{service}"
)
```

---

### SQLiteConfigStore

**Location:** `libs/core/cogniverse_core/common/config_store.py:18-605`

**Purpose:** Versioned configuration storage with multi-tenant support

**Import:**
```python
from cogniverse_core.common.config_store import SQLiteConfigStore
from cogniverse_core.common.config_store_interface import ConfigScope
```

**Key Methods:**

#### set_config()
```python
def set_config(
    self,
    tenant_id: str,
    scope: ConfigScope,
    service: str,
    config_key: str,
    config_value: Dict[str, Any],
) -> ConfigEntry:
    """
    Set configuration value (creates new version).

    Args:
        tenant_id: Tenant identifier
        scope: Configuration scope (SYSTEM, AGENT, SERVICE)
        service: Service name (e.g., "routing_agent")
        config_key: Configuration key (e.g., "routing_config")
        config_value: Configuration value as dictionary

    Returns:
        ConfigEntry with new version
    """
```

#### get_config()
```python
def get_config(
    self,
    tenant_id: str,
    scope: ConfigScope,
    service: str,
    config_key: str,
    version: Optional[int] = None,
) -> Optional[ConfigEntry]:
    """
    Get configuration value.

    Args:
        tenant_id: Tenant identifier
        scope: Configuration scope
        service: Service name
        config_key: Configuration key
        version: Specific version (None for latest)

    Returns:
        ConfigEntry or None if not found
    """
```

#### get_config_history()
```python
def get_config_history(
    self,
    tenant_id: str,
    scope: ConfigScope,
    service: str,
    config_key: str,
    limit: int = 10,
) -> List[ConfigEntry]:
    """
    Get configuration history.

    Returns:
        List of ConfigEntry ordered by version descending
    """
```

**Usage:**
```python
from cogniverse_core.common.config_store import SQLiteConfigStore
from cogniverse_core.common.config_store_interface import ConfigScope
from cogniverse_core.config.unified_config import RoutingConfigUnified

# Initialize store
store = SQLiteConfigStore()

# Create routing config
routing_config = RoutingConfigUnified(
    tenant_id="acme",
    routing_mode="ensemble"
)

# Save config (creates version 1)
entry = store.set_config(
    tenant_id="acme",
    scope=ConfigScope.SERVICE,
    service="routing_agent",
    config_key="routing_config",
    config_value=routing_config.to_dict()
)

print(f"Saved config version: {entry.version}")

# Load latest config
entry = store.get_config(
    tenant_id="acme",
    scope=ConfigScope.SERVICE,
    service="routing_agent",
    config_key="routing_config"
)

if entry:
    routing_config = RoutingConfigUnified.from_dict(entry.config_value)
    print(f"Loaded routing mode: {routing_config.routing_mode}")

# Get config history
history = store.get_config_history(
    tenant_id="acme",
    scope=ConfigScope.SERVICE,
    service="routing_agent",
    config_key="routing_config",
    limit=10
)

print(f"Config has {len(history)} versions")
```

---

## Memory Management

### Mem0MemoryManager

**Location:** `libs/core/cogniverse_core/common/mem0_memory_manager.py:53-494`

**Purpose:** Per-tenant memory management using Mem0 with Vespa backend

**Import:**
```python
from cogniverse_core.common.mem0_memory_manager import Mem0MemoryManager
```

**Architecture:**
- **Per-Tenant Singleton**: One instance per tenant_id
- **Schema Isolation**: Each tenant gets dedicated Vespa schema: `agent_memories_{tenant_id}`
- **Agent Namespacing**: Within tenant, memories are namespaced by agent_name
- **Mem0 Integration**: Uses Mem0 for LLM-processed memories with semantic search

**Key Methods:**

#### initialize()
```python
def initialize(
    self,
    vespa_host: str = "localhost",
    vespa_port: int = 8080,
    vespa_config_port: Optional[int] = None,
    base_schema_name: str = "agent_memories",
    llm_model: str = "llama3.2",
    embedding_model: str = "nomic-embed-text",
    ollama_base_url: str = "http://localhost:11434/v1",
    auto_create_schema: bool = True,
) -> None:
    """
    Initialize Mem0 with Vespa backend using tenant-specific schema.

    Configuration:
    - LLM: Ollama llama3.2 for memory processing
    - Embedder: Ollama nomic-embed-text (768-dim)
    - Vector Store: Vespa with schema-per-tenant

    Example:
        manager = Mem0MemoryManager(tenant_id="acme")
        manager.initialize(
            vespa_host="localhost",
            vespa_port=8080,
            llm_model="llama3.2",
            embedding_model="nomic-embed-text"
        )
    """
```

#### add_memory()
```python
def add_memory(
    self,
    content: str,
    tenant_id: str,
    agent_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Add content to agent's memory.

    Process:
    1. Mem0 processes content with LLM (llama3.2)
    2. Generates embedding (nomic-embed-text, 768-dim)
    3. Stores in tenant-specific Vespa schema

    Args:
        content: Memory content (natural language)
        tenant_id: Tenant identifier
        agent_name: Agent name (e.g., "routing_agent")
        metadata: Optional metadata dict

    Returns:
        Memory ID (string)

    Example:
        memory_id = manager.add_memory(
            content="User prefers detailed technical explanations",
            tenant_id="acme",
            agent_name="routing_agent",
            metadata={"source": "user_feedback"}
        )
    """
```

#### search_memory()
```python
def search_memory(
    self,
    query: str,
    tenant_id: str,
    agent_name: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Search agent's memory for relevant content.

    Process:
    1. Encode query to embedding (nomic-embed-text)
    2. Semantic search in tenant's Vespa schema
    3. Return top_k most similar memories

    Args:
        query: Search query (natural language)
        tenant_id: Tenant identifier
        agent_name: Agent name
        top_k: Number of results

    Returns:
        List of memories with scores:
        [
            {
                "id": "mem_123",
                "memory": "User prefers detailed explanations",
                "score": 0.92,
                "metadata": {...}
            },
            ...
        ]

    Example:
        memories = manager.search_memory(
            query="What are user's preferences?",
            tenant_id="acme",
            agent_name="routing_agent",
            top_k=5
        )

        for mem in memories:
            print(f"{mem['memory']} (score: {mem['score']:.2f})")
    """
```

#### get_all_memories()
```python
def get_all_memories(
    self,
    tenant_id: str,
    agent_name: str,
) -> List[Dict[str, Any]]:
    """
    Get all memories for an agent.

    Returns:
        List of all memories (no filtering)
    """
```

#### clear_agent_memory()
```python
def clear_agent_memory(
    self,
    tenant_id: str,
    agent_name: str,
) -> bool:
    """
    Clear all memory for an agent.

    Use case: Reset agent memory for testing or tenant offboarding
    """
```

**Memory Storage Format** (in Vespa):
```json
{
  "id": "mem_abc123",
  "user_id": "acme",
  "agent_id": "routing_agent",
  "memory": "User prefers detailed technical explanations with code examples",
  "embedding": [0.23, -0.15, 0.87, ...],  // 768-dim vector
  "metadata": {
    "source": "user_feedback",
    "timestamp": "2025-10-15T10:30:00Z"
  },
  "hash": "abc123def456"
}
```

**Multi-Tenant Flow:**

```mermaid
sequenceDiagram
    participant Agent as RoutingAgent<br/>cogniverse_agents
    participant MemMgr as Mem0MemoryManager<br/>cogniverse_core
    participant SchMgr as TenantSchemaManager<br/>cogniverse_vespa
    participant Vespa as Vespa<br/>agent_memories_acme

    Agent->>MemMgr: Mem0MemoryManager(tenant_id="acme")
    Note over MemMgr: Per-tenant singleton created

    Agent->>MemMgr: initialize(vespa_host, vespa_port)
    MemMgr->>SchMgr: get_tenant_schema_name("acme", "agent_memories")
    SchMgr-->>MemMgr: "agent_memories_acme"
    MemMgr->>SchMgr: ensure_tenant_schema_exists("acme", "agent_memories")
    SchMgr->>Vespa: Deploy schema agent_memories_acme
    Vespa-->>SchMgr: Schema deployed
    MemMgr-->>Agent: Initialized

    Agent->>MemMgr: add_memory("User prefers videos", "acme", "routing_agent")
    MemMgr->>Vespa: Store in agent_memories_acme
    Vespa-->>MemMgr: memory_id
    MemMgr-->>Agent: memory_id

    Agent->>MemMgr: search_memory("preferences?", "acme", "routing_agent")
    MemMgr->>Vespa: Search agent_memories_acme
    Vespa-->>MemMgr: [{"memory": "...", "score": 0.92}]
    MemMgr-->>Agent: memories
```

---

## Tenant Utilities

### parse_tenant_id()

**Location:** `libs/core/cogniverse_core/common/tenant_utils.py:12-53`

**Purpose:** Parse tenant_id into org_id and tenant_name

**Import:**
```python
from cogniverse_core.common.tenant_utils import parse_tenant_id
```

**Function:**
```python
def parse_tenant_id(tenant_id: str) -> tuple[str, str]:
    """
    Parse tenant_id into org_id and tenant_name.

    Supports two formats:
    - Simple: "acme" → ("acme", "acme")
    - Org:tenant: "acme:production" → ("acme", "production")

    Args:
        tenant_id: Tenant identifier

    Returns:
        Tuple of (org_id, tenant_name)

    Raises:
        ValueError: If tenant_id is empty or has invalid format

    Examples:
        >>> parse_tenant_id("acme")
        ('acme', 'acme')
        >>> parse_tenant_id("acme:production")
        ('acme', 'production')
    """
```

**Usage:**
```python
from cogniverse_core.common.tenant_utils import parse_tenant_id

# Simple format
org_id, tenant_name = parse_tenant_id("acme")
print(f"Org: {org_id}, Tenant: {tenant_name}")
# Output: Org: acme, Tenant: acme

# Org:tenant format
org_id, tenant_name = parse_tenant_id("acme:production")
print(f"Org: {org_id}, Tenant: {tenant_name}")
# Output: Org: acme, Tenant: production
```

---

### get_tenant_storage_path()

**Location:** `libs/core/cogniverse_core/common/tenant_utils.py:55-85`

**Purpose:** Get tenant-specific storage path with org/tenant structure

**Import:**
```python
from cogniverse_core.common.tenant_utils import get_tenant_storage_path
```

**Function:**
```python
def get_tenant_storage_path(base_dir: Path | str, tenant_id: str) -> Path:
    """
    Get tenant-specific storage path with proper org/tenant structure.

    Supports two formats:
    - Simple: "acme" → base_dir/acme/
    - Org:tenant: "acme:production" → base_dir/acme/production/

    Args:
        base_dir: Base storage directory
        tenant_id: Tenant identifier

    Returns:
        Path to tenant-specific storage directory

    Examples:
        >>> get_tenant_storage_path("data/optimization", "acme")
        Path('data/optimization/acme')
        >>> get_tenant_storage_path("data/optimization", "acme:production")
        Path('data/optimization/acme/production')
    """
```

**Usage:**
```python
from pathlib import Path
from cogniverse_core.common.tenant_utils import get_tenant_storage_path

# Simple format
path = get_tenant_storage_path("data/optimization", "acme")
print(path)
# Output: data/optimization/acme

# Org:tenant format
path = get_tenant_storage_path("data/optimization", "acme:production")
print(path)
# Output: data/optimization/acme/production

# Create tenant-specific directory
path.mkdir(parents=True, exist_ok=True)
```

---

### validate_tenant_id()

**Location:** `libs/core/cogniverse_core/common/tenant_utils.py:87-119`

**Purpose:** Validate tenant ID format

**Import:**
```python
from cogniverse_core.common.tenant_utils import validate_tenant_id
```

**Function:**
```python
def validate_tenant_id(tenant_id: str) -> None:
    """
    Validate tenant ID format.

    Raises:
        ValueError: If tenant_id is invalid
    """
```

**Usage:**
```python
from cogniverse_core.common.tenant_utils import validate_tenant_id

# Valid IDs
validate_tenant_id("acme")                  # OK
validate_tenant_id("acme-corp")             # OK
validate_tenant_id("acme_corp")             # OK
validate_tenant_id("acme:production")       # OK
validate_tenant_id("acme-corp:prod-env")    # OK

# Invalid IDs
try:
    validate_tenant_id("")                  # ValueError: empty
except ValueError as e:
    print(f"Error: {e}")

try:
    validate_tenant_id("acme:prod:env")     # ValueError: multiple colons
except ValueError as e:
    print(f"Error: {e}")

try:
    validate_tenant_id("acme@corp")         # ValueError: invalid chars
except ValueError as e:
    print(f"Error: {e}")
```

---

## Dynamic DSPy Integration

### DynamicDSPyMixin

**Location:** `libs/core/cogniverse_core/common/dynamic_dspy_mixin.py:16-247`

**Purpose:** Mixin for runtime DSPy module and optimizer configuration

**Import:**
```python
from cogniverse_core.common.dynamic_dspy_mixin import DynamicDSPyMixin
from cogniverse_core.config.agent_config import AgentConfig, ModuleConfig, OptimizerConfig
```

**Key Methods:**

#### initialize_dynamic_dspy()
```python
def initialize_dynamic_dspy(self, config: AgentConfig):
    """
    Initialize DSPy with dynamic configuration.

    Sets up:
    - DSPy LM (language model)
    - Signature registry (for module creation)
    - Module cache
    - Optimizer instance

    Example:
        class MyAgent(DynamicDSPyMixin):
            def __init__(self):
                config = AgentConfig(
                    agent_name="my_agent",
                    llm_model="ollama/llama3.2",
                    module_config=ModuleConfig(
                        module_type=ModuleType.CHAIN_OF_THOUGHT
                    ),
                    optimizer_config=OptimizerConfig(
                        optimizer_type=OptimizerType.MIPROV2
                    )
                )
                self.initialize_dynamic_dspy(config)
    """
```

#### register_signature()
```python
def register_signature(self, name: str, signature: Type[dspy.Signature]):
    """
    Register a DSPy signature for module creation.

    Example:
        class MySignature(dspy.Signature):
            query = dspy.InputField(desc="User query")
            answer = dspy.OutputField(desc="Answer")

        self.register_signature("my_query", MySignature)
    """
```

#### create_module()
```python
def create_module(
    self,
    signature_name: str,
    module_config: Optional[ModuleConfig] = None
) -> dspy.Module:
    """
    Create DSPy module dynamically.

    Module types:
    - Predict: Direct prediction
    - ChainOfThought: Reasoning before answer
    - ReAct: Reasoning + Action + Observation loop
    - ProgramOfThought: Code generation

    Example:
        # Register signature
        self.register_signature("analyze_query", QueryAnalysisSignature)

        # Create ChainOfThought module
        module = self.create_module("analyze_query")

        # Use module
        result = module(query="What is machine learning?")
        print(result.answer)
    """
```

#### create_optimizer()
```python
def create_optimizer(
    self,
    optimizer_config: Optional[OptimizerConfig] = None
) -> Any:
    """
    Create DSPy optimizer dynamically.

    Optimizer types:
    - BootstrapFewShot: Basic few-shot learning
    - SIMBA: Similarity-based memory augmentation
    - MIPROv2: Metric-aware instruction optimization
    - GEPA: Reflective prompt evolution

    Example:
        optimizer = self.create_optimizer()

        # Compile module with optimizer
        optimized_module = optimizer.compile(
            module,
            trainset=training_examples,
            max_bootstrapped_demos=4
        )
    """
```

**Complete Usage Example:**
```python
from cogniverse_core.common.dynamic_dspy_mixin import DynamicDSPyMixin
from cogniverse_core.config.agent_config import AgentConfig, ModuleConfig, OptimizerConfig
from cogniverse_core.config.agent_config import ModuleType, OptimizerType
import dspy

class SmartAgent(DynamicDSPyMixin):
    def __init__(self):
        # Configure agent
        config = AgentConfig(
            agent_name="smart_agent",
            llm_model="ollama/llama3.2",
            llm_base_url="http://localhost:11434",
            module_config=ModuleConfig(
                module_type=ModuleType.CHAIN_OF_THOUGHT,
                max_retries=3
            ),
            optimizer_config=OptimizerConfig(
                optimizer_type=OptimizerType.MIPROV2,
                max_bootstrapped_demos=4,
                max_labeled_demos=8
            )
        )

        # Initialize dynamic DSPy
        self.initialize_dynamic_dspy(config)

        # Register signatures
        class QueryAnalysisSignature(dspy.Signature):
            query = dspy.InputField(desc="User query to analyze")
            intent = dspy.OutputField(desc="Detected intent")
            confidence = dspy.OutputField(desc="Confidence score")

        self.register_signature("analyze", QueryAnalysisSignature)

    def analyze_query(self, query: str):
        # Get or create module
        module = self.get_or_create_module("analyze")

        # Run inference
        result = module(query=query)
        return result

# Usage
agent = SmartAgent()
result = agent.analyze_query("Find videos about machine learning")
print(f"Intent: {result.intent}, Confidence: {result.confidence}")
```

---

## Usage Examples

### Example 1: Multi-Tenant Configuration

```python
from cogniverse_core.config.unified_config import SystemConfig, RoutingConfigUnified
from cogniverse_core.common.config_store import SQLiteConfigStore
from cogniverse_core.common.config_store_interface import ConfigScope

# Initialize system
system_config = SystemConfig(
    tenant_id="acme",
    vespa_url="http://localhost",
    vespa_port=8080,
    environment="production"
)

store = SQLiteConfigStore()

# Create configs for multiple tenants
tenants = ["acme", "acme:production", "acme:staging"]

for tenant_id in tenants:
    # Tenant-specific routing config
    routing_config = RoutingConfigUnified(
        tenant_id=tenant_id,
        routing_mode="tiered",
        fast_path_confidence_threshold=0.75,
        enable_auto_optimization=True,
        cache_ttl_seconds=3600 if "production" in tenant_id else 300
    )

    # Save to store
    entry = store.set_config(
        tenant_id=tenant_id,
        scope=ConfigScope.SERVICE,
        service="routing_agent",
        config_key="routing_config",
        config_value=routing_config.to_dict()
    )
    print(f"Saved routing config for {tenant_id} (version {entry.version})")

# Load tenant config in agent
tenant_id = "acme:production"
entry = store.get_config(
    tenant_id=tenant_id,
    scope=ConfigScope.SERVICE,
    service="routing_agent",
    config_key="routing_config"
)

if entry:
    routing_config = RoutingConfigUnified.from_dict(entry.config_value)
    print(f"Loaded config for {tenant_id}:")
    print(f"  Routing mode: {routing_config.routing_mode}")
    print(f"  Cache TTL: {routing_config.cache_ttl_seconds}s")
```

---

### Example 2: Memory-Aware Agent

```python
from cogniverse_core.common.mem0_memory_manager import Mem0MemoryManager

# Initialize memory manager (per-tenant singleton)
tenant_id = "acme"
memory = Mem0MemoryManager(tenant_id=tenant_id)
memory.initialize(
    vespa_host="localhost",
    vespa_port=8080,
    llm_model="llama3.2",
    embedding_model="nomic-embed-text"
)

agent_name = "routing_agent"

# Add memories from user interactions
memory.add_memory(
    content="User prefers video results over text documents",
    tenant_id=tenant_id,
    agent_name=agent_name,
    metadata={"source": "user_feedback"}
)

memory.add_memory(
    content="User is interested in machine learning tutorials",
    tenant_id=tenant_id,
    agent_name=agent_name,
    metadata={"source": "query_history"}
)

memory.add_memory(
    content="User's technical level: advanced",
    tenant_id=tenant_id,
    agent_name=agent_name,
    metadata={"source": "user_profile"}
)

# Search relevant memories for new query
query = "Find content about neural networks"
relevant_memories = memory.search_memory(
    query=query,
    tenant_id=tenant_id,
    agent_name=agent_name,
    top_k=3
)

print(f"Relevant memories for: '{query}'")
for mem in relevant_memories:
    print(f"  • {mem['memory']} (score: {mem['score']:.2f})")

# Output:
# Relevant memories for: 'Find content about neural networks'
#   • User is interested in machine learning tutorials (score: 0.89)
#   • User's technical level: advanced (score: 0.72)
#   • User prefers video results over text documents (score: 0.65)

# Use memories to enhance routing
preferences = " ".join([m['memory'] for m in relevant_memories])
enhanced_query = f"{query}. Context: {preferences}"
```

---

### Example 3: Tenant Storage Paths

```python
from pathlib import Path
from cogniverse_core.common.tenant_utils import (
    parse_tenant_id,
    get_tenant_storage_path,
    validate_tenant_id
)

# Parse tenant IDs
simple_tenant = "acme"
org_tenant = "acme:production"

org_id, tenant_name = parse_tenant_id(simple_tenant)
print(f"Simple: org={org_id}, tenant={tenant_name}")
# Output: Simple: org=acme, tenant=acme

org_id, tenant_name = parse_tenant_id(org_tenant)
print(f"Org:tenant: org={org_id}, tenant={tenant_name}")
# Output: Org:tenant: org=acme, tenant=production

# Get storage paths
base_dir = "data/optimization"

path1 = get_tenant_storage_path(base_dir, simple_tenant)
print(f"Simple path: {path1}")
# Output: Simple path: data/optimization/acme

path2 = get_tenant_storage_path(base_dir, org_tenant)
print(f"Org:tenant path: {path2}")
# Output: Org:tenant path: data/optimization/acme/production

# Create directories
path1.mkdir(parents=True, exist_ok=True)
path2.mkdir(parents=True, exist_ok=True)

# Validate tenant IDs
try:
    validate_tenant_id("acme")                  # Valid
    validate_tenant_id("acme:production")       # Valid
    validate_tenant_id("acme:prod:env")         # Invalid - multiple colons
except ValueError as e:
    print(f"Validation error: {e}")
```

---

### Example 4: Configuration Versioning

```python
from cogniverse_core.common.config_store import SQLiteConfigStore
from cogniverse_core.common.config_store_interface import ConfigScope
from cogniverse_core.config.unified_config import RoutingConfigUnified

store = SQLiteConfigStore()
tenant_id = "acme"

# Version 1: Initial config
config_v1 = RoutingConfigUnified(
    tenant_id=tenant_id,
    routing_mode="tiered",
    fast_path_confidence_threshold=0.7
)

entry = store.set_config(
    tenant_id=tenant_id,
    scope=ConfigScope.SERVICE,
    service="routing_agent",
    config_key="routing_config",
    config_value=config_v1.to_dict()
)
print(f"Created version {entry.version}")

# Version 2: Update threshold
config_v2 = RoutingConfigUnified(
    tenant_id=tenant_id,
    routing_mode="tiered",
    fast_path_confidence_threshold=0.8  # Increased threshold
)

entry = store.set_config(
    tenant_id=tenant_id,
    scope=ConfigScope.SERVICE,
    service="routing_agent",
    config_key="routing_config",
    config_value=config_v2.to_dict()
)
print(f"Created version {entry.version}")

# Version 3: Switch to ensemble mode
config_v3 = RoutingConfigUnified(
    tenant_id=tenant_id,
    routing_mode="ensemble",  # Changed mode
    fast_path_confidence_threshold=0.8
)

entry = store.set_config(
    tenant_id=tenant_id,
    scope=ConfigScope.SERVICE,
    service="routing_agent",
    config_key="routing_config",
    config_value=config_v3.to_dict()
)
print(f"Created version {entry.version}")

# Get latest version
latest = store.get_config(
    tenant_id=tenant_id,
    scope=ConfigScope.SERVICE,
    service="routing_agent",
    config_key="routing_config"
)
print(f"Latest version: {latest.version}, mode: {latest.config_value['routing_mode']}")

# Get specific version
v1 = store.get_config(
    tenant_id=tenant_id,
    scope=ConfigScope.SERVICE,
    service="routing_agent",
    config_key="routing_config",
    version=1
)
print(f"Version 1: mode={v1.config_value['routing_mode']}, threshold={v1.config_value['fast_path_confidence_threshold']}")

# Get history
history = store.get_config_history(
    tenant_id=tenant_id,
    scope=ConfigScope.SERVICE,
    service="routing_agent",
    config_key="routing_config",
    limit=10
)

print(f"\nConfiguration history ({len(history)} versions):")
for entry in history:
    config = RoutingConfigUnified.from_dict(entry.config_value)
    print(f"  v{entry.version}: mode={config.routing_mode}, "
          f"threshold={config.fast_path_confidence_threshold}, "
          f"updated={entry.updated_at.isoformat()}")
```

---

## Production Considerations

### Configuration Management Best Practices

**1. Environment-Specific Configs:**
```python
from cogniverse_core.config.unified_config import SystemConfig

environments = {
    "development": SystemConfig(
        tenant_id="dev",
        vespa_url="http://localhost",
        vespa_port=8080,
        environment="development"
    ),
    "staging": SystemConfig(
        tenant_id="staging",
        vespa_url="http://staging-vespa.internal",
        vespa_port=8080,
        environment="staging"
    ),
    "production": SystemConfig(
        tenant_id="prod",
        vespa_url="http://prod-vespa.example.com",
        vespa_port=8080,
        environment="production"
    )
}

import os
env = os.getenv("ENVIRONMENT", "development")
config = environments[env]
```

**2. Validate Configs Before Use:**
```python
from cogniverse_core.config.unified_config import SystemConfig

def validate_config(config: SystemConfig):
    assert config.vespa_url, "Vespa URL required"
    assert config.vespa_port > 0, "Valid Vespa port required"
    assert config.llm_model, "LLM model required"

    # Test connectivity
    import requests
    try:
        response = requests.get(f"{config.vespa_url}:{config.vespa_port}/ApplicationStatus")
        assert response.status_code == 200
    except:
        raise RuntimeError("Cannot connect to Vespa")

config = SystemConfig.from_dict(config_dict)
validate_config(config)
```

**3. Use Config Versioning:**
```python
from cogniverse_core.common.config_store import SQLiteConfigStore
from cogniverse_core.common.config_store_interface import ConfigScope

store = SQLiteConfigStore()

# Always create new version on update
entry = store.set_config(
    tenant_id="acme",
    scope=ConfigScope.SERVICE,
    service="routing_agent",
    config_key="routing_config",
    config_value=new_config.to_dict()
)

# Get history for rollback if needed
history = store.get_config_history(
    tenant_id="acme",
    scope=ConfigScope.SERVICE,
    service="routing_agent",
    config_key="routing_config"
)

# Rollback to previous version
if performance_degraded:
    previous_version = history[1]  # Get version before latest
    store.set_config(
        tenant_id="acme",
        scope=ConfigScope.SERVICE,
        service="routing_agent",
        config_key="routing_config",
        config_value=previous_version.config_value
    )
```

### Memory Management Best Practices

**1. Initialize Once at Startup:**
```python
from functools import lru_cache
from cogniverse_core.common.mem0_memory_manager import Mem0MemoryManager

@lru_cache(maxsize=100)  # Cache per tenant
def get_memory_manager(tenant_id: str) -> Mem0MemoryManager:
    manager = Mem0MemoryManager(tenant_id=tenant_id)
    manager.initialize(
        vespa_host=config.vespa_url,
        vespa_port=config.vespa_port
    )
    return manager

# Use in agents
memory = get_memory_manager("acme")
```

**2. Memory Storage Policy:**
```python
from cogniverse_core.common.mem0_memory_manager import Mem0MemoryManager

class MemoryPolicy:
    def should_store(self, content: str) -> bool:
        # Don't store short or generic content
        if len(content) < 20:
            return False

        # Don't store common phrases
        generic_phrases = ["hello", "thanks", "ok", "yes", "no"]
        if content.lower() in generic_phrases:
            return False

        return True

memory = Mem0MemoryManager(tenant_id="acme")
memory.initialize()

policy = MemoryPolicy()
content = "User prefers technical content"

if policy.should_store(content):
    memory.add_memory(
        content=content,
        tenant_id="acme",
        agent_name="routing_agent"
    )
```

**3. Periodic Memory Cleanup:**
```python
from datetime import datetime, timedelta
from cogniverse_core.common.mem0_memory_manager import Mem0MemoryManager

async def cleanup_old_memories(tenant_id: str, agent_name: str, days: int = 90):
    """Remove memories older than specified days"""
    memory = Mem0MemoryManager(tenant_id=tenant_id)
    memory.initialize()

    cutoff_date = datetime.now() - timedelta(days=days)

    memories = memory.get_all_memories(tenant_id, agent_name)

    for mem in memories:
        if isinstance(mem, dict):
            timestamp = mem.get("metadata", {}).get("timestamp")
            if timestamp:
                mem_date = datetime.fromisoformat(timestamp)
                if mem_date < cutoff_date:
                    memory.delete_memory(mem["id"], tenant_id, agent_name)
                    print(f"Deleted old memory: {mem['id']}")

# Run cleanup for all tenants
for tenant_id in get_active_tenants():
    await cleanup_old_memories(tenant_id, "routing_agent", days=90)
```

### Tenant Isolation Verification

**1. Verify Schema Isolation:**
```python
from cogniverse_vespa.tenant_schema_manager import TenantSchemaManager
from cogniverse_core.common.mem0_memory_manager import Mem0MemoryManager

schema_manager = TenantSchemaManager()

# Each tenant should have dedicated schema
tenant1 = "acme"
tenant2 = "globex"

schema1 = schema_manager.get_tenant_schema_name(tenant1, "agent_memories")
schema2 = schema_manager.get_tenant_schema_name(tenant2, "agent_memories")

assert schema1 != schema2, "Schemas must be different for different tenants"
assert schema1 == "agent_memories_acme"
assert schema2 == "agent_memories_globex"

print(f"✓ Schema isolation verified: {schema1} vs {schema2}")
```

**2. Verify Memory Isolation:**
```python
from cogniverse_core.common.mem0_memory_manager import Mem0MemoryManager

# Add memory for tenant 1
memory1 = Mem0MemoryManager(tenant_id="acme")
memory1.initialize()
memory1.add_memory(
    content="Secret information for ACME",
    tenant_id="acme",
    agent_name="routing_agent"
)

# Try to search from tenant 2
memory2 = Mem0MemoryManager(tenant_id="globex")
memory2.initialize()
results = memory2.search_memory(
    query="secret information",
    tenant_id="globex",
    agent_name="routing_agent"
)

# Should NOT find tenant 1's memory
assert len(results) == 0, "Cross-tenant memory leak detected!"
print("✓ Memory isolation verified")
```

---

## Testing

### Unit Tests
**Location:** `tests/common/unit/`

**Key Test Files:**
- `test_config_store.py` - Configuration storage tests
- `test_mem0_memory_manager.py` - Memory manager tests
- `test_tenant_utils.py` - Tenant utilities tests
- `test_dynamic_dspy_mixin.py` - DSPy mixin tests

### Integration Tests
**Location:** `tests/common/integration/`

**Key Test Files:**
- `test_config_store_integration.py` - Config store with SQLite
- `test_mem0_vespa_integration.py` - Memory manager with Vespa
- `test_tenant_isolation.py` - Multi-tenant isolation verification

### Example Test

```python
import pytest
from cogniverse_core.common.mem0_memory_manager import Mem0MemoryManager
from cogniverse_core.common.tenant_utils import parse_tenant_id, get_tenant_storage_path

@pytest.mark.integration
async def test_mem0_memory_integration():
    """Test Mem0 memory manager integration"""
    tenant_id = "test-tenant"
    agent_name = "test-agent"

    manager = Mem0MemoryManager(tenant_id=tenant_id)
    manager.initialize()

    # Add memory
    memory_id = manager.add_memory(
        content="User prefers technical explanations",
        tenant_id=tenant_id,
        agent_name=agent_name
    )

    assert memory_id

    # Search memory
    results = manager.search_memory(
        query="What does user prefer?",
        tenant_id=tenant_id,
        agent_name=agent_name
    )

    assert len(results) > 0
    assert "technical" in results[0]["memory"].lower()

    # Cleanup
    manager.clear_agent_memory(tenant_id, agent_name)

@pytest.mark.unit
def test_tenant_id_parsing():
    """Test tenant ID parsing"""
    # Simple format
    org_id, tenant_name = parse_tenant_id("acme")
    assert org_id == "acme"
    assert tenant_name == "acme"

    # Org:tenant format
    org_id, tenant_name = parse_tenant_id("acme:production")
    assert org_id == "acme"
    assert tenant_name == "production"

    # Invalid format
    with pytest.raises(ValueError):
        parse_tenant_id("")

    with pytest.raises(ValueError):
        parse_tenant_id("acme:prod:env")

@pytest.mark.unit
def test_tenant_storage_path():
    """Test tenant storage path generation"""
    from pathlib import Path

    # Simple format
    path = get_tenant_storage_path("data/optimization", "acme")
    assert path == Path("data/optimization/acme")

    # Org:tenant format
    path = get_tenant_storage_path("data/optimization", "acme:production")
    assert path == Path("data/optimization/acme/production")
```

---

## Next Steps

For related modules:
- **Agents Module** (`agents.md`) - Uses Common module for config and memory (libs/agents/cogniverse_agents/)
- **Backends Module** (`backends.md`) - Vespa integration details (libs/vespa/cogniverse_vespa/)
- **Telemetry Module** (`telemetry.md`) - Multi-tenant telemetry (libs/core/cogniverse_core/telemetry/)
- **SDK Architecture** (`../architecture/sdk-architecture.md`) - UV workspace and package structure
- **Multi-Tenant Architecture** (`../architecture/multi-tenant.md`) - Tenant isolation patterns

---

## Key Takeaways

1. **SDK Package Structure**
   - Common utilities in `libs/core/cogniverse_core/common/`
   - Configuration in `libs/core/cogniverse_core/config/`
   - Import from `cogniverse_core` package

2. **Multi-Tenant Configuration**
   - SQLite-based versioned configuration storage
   - Per-tenant configuration with history tracking
   - ConfigScope for service/agent/system level configs

3. **Memory Management**
   - Per-tenant singleton pattern: `Mem0MemoryManager(tenant_id="acme")`
   - Schema-per-tenant isolation: `agent_memories_{tenant_id}`
   - Mem0 integration with Vespa backend

4. **Tenant Utilities**
   - Two formats: simple ("acme") and org:tenant ("acme:production")
   - Storage path management with org/tenant structure
   - Validation utilities for tenant IDs

5. **DSPy Integration**
   - Runtime module/optimizer configuration
   - DynamicDSPyMixin for agent integration
   - Module types: Predict, ChainOfThought, ReAct, ProgramOfThought

6. **Production Readiness**
   - Configuration versioning with rollback support
   - Memory cleanup policies
   - Tenant isolation verification
   - Health checks and monitoring
