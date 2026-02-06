# Schema-Driven Multi-Tenant Processing Flow

## Overview

The Cogniverse system uses a schema-driven architecture where backend schemas (e.g., Vespa) define the data model, ranking strategies, and tenant isolation boundaries. Each tenant gets dedicated schemas deployed per embedding profile.

## Complete Flow Summary

1. **Schemas define everything** → `configs/schemas/*.json`
2. **Deployment extracts strategies** → `ranking_strategies.json`
3. **Processing uses schema mapping** → profile → schema
4. **Query loads strategies** → validates → executes
5. **No manual steps** → Everything is automatic!

## Multi-Tenant Schema Architecture

```mermaid
flowchart TD
    A[<span style='color:#000'>Tenant Request</span>] --> B[<span style='color:#000'>Schema Manager</span>]
    B --> C{<span style='color:#000'>Schema Exists?</span>}

    C -->|No| D[<span style='color:#000'>Generate Tenant Schema</span>]
    D --> E[<span style='color:#000'>Deploy to Backend</span>]
    E --> F[<span style='color:#000'>Register in Metadata</span>]

    C -->|Yes| G[<span style='color:#000'>Get Schema Name</span>]

    F --> H[<span style='color:#000'>Process with Schema</span>]
    G --> H

    H --> I[<span style='color:#000'>Tenant-Isolated Data</span>]

    D --> L
    G --> L

    subgraph SchemaNaming[<span style='color:#000'>Schema Naming Convention</span>]
        J[<span style='color:#000'>Profile</span>]
        K[<span style='color:#000'>Tenant ID</span>]
        L[<span style='color:#000'>profile_tenantid</span>]
        J --> L
        K --> L
    end

    style A fill:#90caf9,stroke:#1565c0,color:#000
    style B fill:#ffcc80,stroke:#ef6c00,color:#000
    style C fill:#ffcc80,stroke:#ef6c00,color:#000
    style D fill:#ffcc80,stroke:#ef6c00,color:#000
    style E fill:#a5d6a7,stroke:#388e3c,color:#000
    style F fill:#a5d6a7,stroke:#388e3c,color:#000
    style G fill:#a5d6a7,stroke:#388e3c,color:#000
    style H fill:#ce93d8,stroke:#7b1fa2,color:#000
    style I fill:#a5d6a7,stroke:#388e3c,color:#000
    style SchemaNaming fill:#b0bec5,stroke:#546e7a,color:#000
    style J fill:#90caf9,stroke:#1565c0,color:#000
    style K fill:#90caf9,stroke:#1565c0,color:#000
    style L fill:#a5d6a7,stroke:#388e3c,color:#000
```

## 1. Processing/Ingestion Flow

```mermaid
flowchart TD
    A[<span style='color:#000'>Start Ingestion</span>] --> B[<span style='color:#000'>Load Config</span>]
    B --> C{<span style='color:#000'>First Time?</span>}

    C -->|Yes| D[<span style='color:#000'>Deploy All Schemas</span>]
    C -->|No| E[<span style='color:#000'>VideoIngestionPipeline</span>]

    D --> D1[<span style='color:#000'>Read Schema JSONs</span>]
    D1 --> D2[<span style='color:#000'>Deploy to Backend</span>]
    D2 --> D3[<span style='color:#000'>Extract Ranking Strategies</span>]
    D3 --> D4[<span style='color:#000'>Save ranking_strategies.json</span>]
    D4 --> E

    E --> F[<span style='color:#000'>Process Videos by Profile</span>]
    F --> G[<span style='color:#000'>Extract Keyframes</span>]
    G --> H[<span style='color:#000'>Generate Embeddings</span>]
    H --> I[<span style='color:#000'>Get Schema for Profile</span>]

    I --> I1[<span style='color:#000'>Profile to Schema Mapping</span>]
    I1 --> I2[<span style='color:#000'>Load Ranking Strategies</span>]
    I2 --> I3[<span style='color:#000'>Determine Required Tensors</span>]

    I3 --> J[<span style='color:#000'>Format Data by Strategy</span>]
    J --> J1{<span style='color:#000'>Needs Float?</span>}
    J1 -->|Yes| J2[<span style='color:#000'>Add embedding field</span>]
    J1 -->|No| K
    J2 --> J3{<span style='color:#000'>Needs Binary?</span>}
    J3 -->|Yes| J4[<span style='color:#000'>Add embedding_binary</span>]
    J3 -->|No| K
    J4 --> K[<span style='color:#000'>Send to Backend</span>]
    K --> L[<span style='color:#000'>Next Video/Profile</span>]
    L -->|More Videos| F
    L -->|Done| M[<span style='color:#000'>End</span>]

    style A fill:#90caf9,stroke:#1565c0,color:#000
    style B fill:#b0bec5,stroke:#546e7a,color:#000
    style C fill:#ffcc80,stroke:#ef6c00,color:#000
    style D fill:#ffcc80,stroke:#ef6c00,color:#000
    style D1 fill:#ffcc80,stroke:#ef6c00,color:#000
    style D2 fill:#a5d6a7,stroke:#388e3c,color:#000
    style D3 fill:#ffcc80,stroke:#ef6c00,color:#000
    style D4 fill:#a5d6a7,stroke:#388e3c,color:#000
    style E fill:#ce93d8,stroke:#7b1fa2,color:#000
    style F fill:#ce93d8,stroke:#7b1fa2,color:#000
    style G fill:#ffcc80,stroke:#ef6c00,color:#000
    style H fill:#81d4fa,stroke:#0288d1,color:#000
    style I fill:#ffcc80,stroke:#ef6c00,color:#000
    style I1 fill:#b0bec5,stroke:#546e7a,color:#000
    style I2 fill:#b0bec5,stroke:#546e7a,color:#000
    style I3 fill:#ffcc80,stroke:#ef6c00,color:#000
    style J fill:#ffcc80,stroke:#ef6c00,color:#000
    style J1 fill:#ffcc80,stroke:#ef6c00,color:#000
    style J2 fill:#a5d6a7,stroke:#388e3c,color:#000
    style J3 fill:#ffcc80,stroke:#ef6c00,color:#000
    style J4 fill:#a5d6a7,stroke:#388e3c,color:#000
    style K fill:#a5d6a7,stroke:#388e3c,color:#000
    style L fill:#ffcc80,stroke:#ef6c00,color:#000
    style M fill:#a5d6a7,stroke:#388e3c,color:#000
```

### Detailed Processing Steps:

1. **Schema Deployment** (First time or schema updates):
   ```bash
   uv run python scripts/deploy_all_schemas.py
   ```
   - Reads all schema JSON files from `configs/schemas/`
   - Deploys schemas to backend as one application package
   - Automatically extracts ranking strategies → `ranking_strategies.json`

2. **Profile Processing**:
   ```python
   from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
   from pathlib import Path

   # Initialize schema loader
   loader = FilesystemSchemaLoader(Path("configs/schemas"))

   # Load ranking strategies from ranking_strategies.json
   strategies = loader.load_ranking_strategies()

   # Get strategies for a specific schema (e.g., "video_colpali_smol500_mv_frame")
   schema_strategies = strategies.get("video_colpali_smol500_mv_frame", {})

   # Determine what embeddings we need based on strategy config
   needs_float = any(s.get("needs_float_embeddings") for s in schema_strategies.values())
   needs_binary = any(s.get("needs_binary_embeddings") for s in schema_strategies.values())
   ```

3. **Data Formatting**:
   ```python
   document = {
       "video_id": video_id,
       "audio_transcript": transcript,  # BM25-indexed text field
       "segment_description": description,  # BM25-indexed text field
   }

   if needs_float:
       document["embedding"] = float_embeddings

   if needs_binary:
       document["embedding_binary"] = binary_embeddings
   ```

## 2. Query Flow

```mermaid
flowchart TD
    A[<span style='color:#000'>Query Request</span>] --> B[<span style='color:#000'>Load Profile Config</span>]
    B --> C[<span style='color:#000'>Get Schema Name</span>]
    C --> D[<span style='color:#000'>Load Ranking Strategies</span>]

    D --> E{<span style='color:#000'>Validate Strategy</span>}
    E -->|Invalid| F[<span style='color:#000'>Error: Strategy not available</span>]
    E -->|Valid| G[<span style='color:#000'>Get Strategy Config</span>]

    G --> H{<span style='color:#000'>Strategy Type?</span>}

    H -->|Pure Text| I[<span style='color:#000'>BM25 Only</span>]
    H -->|Pure Visual| J[<span style='color:#000'>Embeddings Only</span>]
    H -->|Hybrid| K[<span style='color:#000'>BM25 + Embeddings</span>]

    I --> L[<span style='color:#000'>Generate Query</span>]
    J --> L
    K --> L

    L --> M[<span style='color:#000'>Add Tenant Filter</span>]
    M --> N[<span style='color:#000'>Execute Backend Query</span>]

    N --> O[<span style='color:#000'>Return Results</span>]

    style A fill:#90caf9,stroke:#1565c0,color:#000
    style B fill:#b0bec5,stroke:#546e7a,color:#000
    style C fill:#ffcc80,stroke:#ef6c00,color:#000
    style D fill:#b0bec5,stroke:#546e7a,color:#000
    style E fill:#ffcc80,stroke:#ef6c00,color:#000
    style F fill:#e53935,stroke:#c62828,color:#000
    style G fill:#b0bec5,stroke:#546e7a,color:#000
    style H fill:#ffcc80,stroke:#ef6c00,color:#000
    style I fill:#a5d6a7,stroke:#388e3c,color:#000
    style J fill:#81d4fa,stroke:#0288d1,color:#000
    style K fill:#ce93d8,stroke:#7b1fa2,color:#000
    style L fill:#ffcc80,stroke:#ef6c00,color:#000
    style M fill:#ffcc80,stroke:#ef6c00,color:#000
    style N fill:#a5d6a7,stroke:#388e3c,color:#000
    style O fill:#a5d6a7,stroke:#388e3c,color:#000
```

### Query Types and Embedding Requirements

- **Pure Visual** (float_float, binary_binary):
  - Global models: Use `nearestNeighbor`
  - Patch models: Use tensor ranking

- **Hybrid** (hybrid_binary_bm25):
  - Combines `userInput` for text with embeddings

- **Text-Only** (bm25_only):
  - Only uses `userInput`, no embeddings needed

### How Embeddings Flow
1. SearchService generates embeddings upfront
2. Backend receives both text and embeddings
3. Strategy determines what to use:
   - `needs_float_embeddings: true` → Use float embeddings
   - `needs_binary_embeddings: true` → Use binary embeddings
   - Both false → Text-only search, ignore embeddings
4. Future optimization: Make embeddings optional, generate lazily

## 3. Ranking Strategy Details

### Strategy Requirements Table

| Strategy | Text | Float | Binary | nearestNeighbor | Notes |
|----------|------|-------|--------|-----------------|-------|
| bm25_only | ✓ | - | - | - | Pure text search |
| float_float | - | ✓ | - | ✓* | Pure visual, float embeddings |
| binary_binary | - | - | ✓ | ✓* | Pure visual, binary embeddings |
| float_binary | - | ✓/✓ | ✓/✓ | ✓* | Float primary, binary fallback |
| hybrid_float_bm25 | ✓ | ✓ | - | ✓* | Text + float embeddings |
| hybrid_binary_bm25 | ✓ | - | ✓ | ✓* | Text + binary embeddings |
| phased | ✓ | ✓ | ✓ | ✓* | Two-phase: binary → float |

*nearestNeighbor used by single-vector schemas (detected via `_sv_` in schema name, e.g., `video_videoprism_lvt_base_sv_chunk_6s`)

### Schema-Specific Behavior

The search method is determined by the `RankingStrategyInfo` extracted from schemas. Key fields:

```python
from pathlib import Path
from cogniverse_vespa.ranking_strategy_extractor import RankingStrategyExtractor

extractor = RankingStrategyExtractor()
strategies = extractor.extract_from_schema(Path("configs/schemas/video_videoprism_base_mv_chunk_30s_schema.json"))

# Example: Get float_float strategy info
float_strategy = strategies["float_float"]

# RankingStrategyInfo provides:
print(f"Use nearestNeighbor: {float_strategy.use_nearestneighbor}")  # False for mv schemas, True for _sv_ schemas
print(f"NN field: {float_strategy.nearestneighbor_field}")          # "embedding" for sv float strategies
print(f"NN tensor: {float_strategy.nearestneighbor_tensor}")        # "qt" for sv float strategies
print(f"Strategy type: {float_strategy.strategy_type}")             # SearchStrategyType.PURE_VISUAL
```

**Search Method Logic:**
- **Single-vector embeddings** (sv schemas containing `_sv_` like `video_videoprism_lvt_base_sv_chunk_6s`): `use_nearestneighbor=True` for visual/hybrid strategies → uses `nearestNeighbor(embedding, qt)` YQL for efficient ANN search - embeddings are `tensor<float>(v[N])` with no sparse dimensions
- **Multi-vector embeddings** (mv schemas like `video_colpali_smol500_mv_frame`, `video_videoprism_base_mv_chunk_30s`): `use_nearestneighbor=False` → uses tensor ranking expression - embeddings have `patch{}` or similar sparse dimensions
- **Text-only** (`bm25_only`): Uses `userInput` YQL function

## 4. Multi-Tenant Schema Deployment

### Tenant Schema Management

**Package**: cogniverse-vespa (Implementation Layer)
**Location**: `libs/vespa/cogniverse_vespa/vespa_schema_manager.py`

```python
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

# Initialize with backend connection and optional registry
manager = VespaSchemaManager(
    backend_endpoint="http://localhost",
    backend_port=19071,
    schema_loader=schema_loader,  # Optional, needed for tenant schema operations
    schema_registry=schema_registry  # Optional, needed for tenant schema operations
)

# Generate tenant-specific schema name
schema_name = manager.get_tenant_schema_name(
    tenant_id="acme",
    base_schema_name="video_colpali_smol500_mv_frame"
)
# Returns: "video_colpali_smol500_mv_frame_acme"

# Check if tenant schema exists
exists = manager.tenant_schema_exists(
    tenant_id="acme",
    base_schema_name="video_colpali_smol500_mv_frame"
)

# Deploy schema from JSON definition
manager.deploy_schema_from_json(
    schema_json=schema_definition,
    app_name="videosearch"
)

# Or deploy from JSON file
manager.upload_schema_from_json_file(
    json_file_path="configs/schemas/video_colpali_smol500_mv_frame_schema.json",
    app_name="videosearch"
)

# Delete all schemas for a tenant (unregisters from registry)
deleted = manager.delete_tenant_schemas(tenant_id="acme")
# Returns: ["video_colpali_smol500_mv_frame_acme", ...]
```

**Key Methods:**
- `get_tenant_schema_name(tenant_id, base_schema_name)` - Generate tenant schema name
- `tenant_schema_exists(tenant_id, base_schema_name)` - Check schema existence
- `delete_tenant_schemas(tenant_id)` - Unregister tenant schemas
- `deploy_schema_from_json(schema_json, app_name)` - Deploy from dict
- `upload_schema_from_json_file(json_file_path, app_name)` - Deploy from file

### Profile to Schema Mapping

**Package**: cogniverse-vespa (Implementation Layer)
**Location**: Each profile has a dedicated schema in `configs/schemas/`

Each embedding profile has its own Vespa schema file. The schema name matches the profile name:

| Profile | Schema File | Embedding Type |
|---------|-------------|----------------|
| `video_colpali_smol500_mv_frame` | `video_colpali_smol500_mv_frame_schema.json` | Patch (multi-vector) |
| `video_colqwen_omni_mv_chunk_30s` | `video_colqwen_omni_mv_chunk_30s_schema.json` | Patch (multi-vector) |
| `video_videoprism_base_mv_chunk_30s` | `video_videoprism_base_mv_chunk_30s_schema.json` | Chunk (multi-vector) |
| `video_videoprism_large_mv_chunk_30s` | `video_videoprism_large_mv_chunk_30s_schema.json` | Chunk (multi-vector) |
| `video_videoprism_lvt_base_sv_chunk_6s` | `video_videoprism_lvt_base_sv_chunk_6s_schema.json` | Chunk (single-vector) |
| `video_videoprism_lvt_large_sv_chunk_6s` | `video_videoprism_lvt_large_sv_chunk_6s_schema.json` | Chunk (single-vector) |

**Schema Structure Example** (`configs/schemas/video_colpali_smol500_mv_frame_schema.json`):
```json
{
  "name": "video_colpali_smol500_mv_frame",
  "document": {
    "name": "video_colpali_smol500_mv_frame",
    "fields": [
      {"name": "video_id", "type": "string", ...},
      {"name": "embedding", "type": "tensor<bfloat16>(patch{}, v[128])", ...},
      {"name": "embedding_binary", "type": "tensor<int8>(patch{}, v[16])", ...}
    ]
  },
  "rank_profiles": [
    {"name": "default", ...},
    {"name": "binary_binary", ...},
    {"name": "hybrid_float_bm25", ...}
  ]
}
```

## 5. Strategy Extraction and Validation

### Automatic Strategy Extraction

**Package**: cogniverse-vespa (Implementation Layer)
**Location**: `libs/vespa/cogniverse_vespa/ranking_strategy_extractor.py`

```python
from cogniverse_vespa.ranking_strategy_extractor import (
    RankingStrategyExtractor,
    RankingStrategyInfo,
    SearchStrategyType,
    extract_all_ranking_strategies,
    save_ranking_strategies,
)
from pathlib import Path

# Extract strategies from a single schema
extractor = RankingStrategyExtractor()
strategies = extractor.extract_from_schema(
    Path("configs/schemas/video_colpali_smol500_mv_frame_schema.json")
)

# Each strategy is a RankingStrategyInfo dataclass with:
# - name: str
# - strategy_type: SearchStrategyType (PURE_VISUAL, PURE_TEXT, HYBRID)
# - needs_float_embeddings: bool
# - needs_binary_embeddings: bool
# - needs_text_query: bool
# - use_nearestneighbor: bool
# - nearestneighbor_field: Optional[str]
# - nearestneighbor_tensor: Optional[str]
# - embedding_field: Optional[str]
# - query_tensor_name: Optional[str]
# - timeout: float
# - description: str
# - inputs: Dict[str, str]  # Full input definitions
# - query_tensors_needed: List[str]  # List of tensor names needed
# - schema_name: str  # Schema this strategy belongs to

# Extract from all schemas in directory
all_strategies = extract_all_ranking_strategies(Path("configs/schemas"))

# Save to ranking_strategies.json
save_ranking_strategies(all_strategies, Path("configs/schemas/ranking_strategies.json"))
```

### Strategy Validation

Strategy validation is done by checking if the requested strategy exists in the schema's `ranking_strategies.json`:

```python
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

loader = FilesystemSchemaLoader(Path("configs/schemas"))
strategies = loader.load_ranking_strategies()

def validate_strategy_for_schema(schema_name: str, strategy_name: str) -> bool:
    """Check if a strategy is available for a schema"""
    schema_strategies = strategies.get(schema_name, {})
    return strategy_name in schema_strategies

# Example usage
if not validate_strategy_for_schema("video_colpali_smol500_mv_frame", "float_float"):
    raise ValueError("Strategy not available for this schema")
```

## 6. Optimization Points

### Current Implementation
- Embeddings generated for ALL queries upfront
- Backend receives both text and embeddings
- Strategy decides what to use

### Future Optimizations
```python
# Lazy embedding generation
class OptimizedSearchService:
    async def search(self, query: str, strategy: str):
        # Only generate embeddings if strategy needs them
        strategy_config = self.get_strategy_config(strategy)

        float_embeddings = None
        binary_embeddings = None

        if strategy_config.get("needs_float_embeddings"):
            float_embeddings = await self.generate_float_embeddings(query)

        if strategy_config.get("needs_binary_embeddings"):
            binary_embeddings = await self.generate_binary_embeddings(query)

        return await self.backend.search(
            text=query,
            float_embeddings=float_embeddings,  # May be None
            binary_embeddings=binary_embeddings,  # May be None
            strategy=strategy
        )
```

## 7. Configuration Storage in Vespa

Configuration is stored directly in Vespa via `VespaConfigStore`, providing unified storage with multi-tenant support.

**Package**: cogniverse-vespa (Implementation Layer)
**Location**: `libs/vespa/cogniverse_vespa/config/config_store.py`

### VespaConfigStore Schema

Uses the `config_metadata` schema with document structure:
```python
{
    "fields": {
        "config_id": "tenant_id:scope:service:config_key",
        "tenant_id": "default",
        "scope": "system",  # ConfigScope: system, agent, routing, telemetry, schema, backend
        "service": "video_search",
        "config_key": "search_settings",
        "config_value": {...},  # JSON serialized
        "version": 1,
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00"
    }
}
```

### Usage
```python
from cogniverse_vespa.config.config_store import VespaConfigStore
from cogniverse_sdk.interfaces.config_store import ConfigScope

# Initialize with URL and port (Vespa app created internally)
config_store = VespaConfigStore(
    vespa_url="http://localhost",
    vespa_port=8080,
    schema_name="config_metadata"
)
config_store.initialize()

# Or initialize with existing Vespa app instance
from vespa.application import Vespa
vespa_app = Vespa(url="http://localhost:8080")
config_store = VespaConfigStore(
    vespa_app=vespa_app,
    schema_name="config_metadata"
)
config_store.initialize()

# Store configuration (creates new version)
entry = config_store.set_config(
    tenant_id="acme",
    scope=ConfigScope.SYSTEM,  # Valid scopes: SYSTEM, AGENT, ROUTING, TELEMETRY, SCHEMA, BACKEND
    service="video_search",
    config_key="search_settings",
    config_value={"default_limit": 10, "max_results": 100}
)

# Retrieve configuration (latest version)
config = config_store.get_config(
    tenant_id="acme",
    scope=ConfigScope.SYSTEM,
    service="video_search",
    config_key="search_settings"
)

# Get configuration history
history = config_store.get_config_history(
    tenant_id="acme",
    scope=ConfigScope.SYSTEM,
    service="video_search",
    config_key="search_settings",
    limit=10
)

# Export/import for tenant
exported = config_store.export_configs(tenant_id="acme")
config_store.import_configs(tenant_id="new_tenant", configs=exported)
```

**Key Features:**
- Versioned configurations (all updates create new versions)
- Multi-tenant isolation via tenant_id
- Scoped configurations (system, agent, routing, telemetry, schema, backend)
- Export/import for tenant migration
- Health checks and statistics

## 8. Complete Configuration Flow

### Configuration Loading Order (Layered Architecture)

1. **config.json** (cogniverse-foundation) → Main configuration with profiles
   - Location: `configs/config.json`
   - Package: cogniverse-foundation (config module)

2. **schemas/*.json** (cogniverse-vespa) → Backend schema definitions
   - Location: `configs/schemas/*.json`
   - Package: cogniverse-vespa (schema manager)

3. **ranking_strategies.json** (auto-generated) → Extracted strategies
   - Location: `configs/schemas/ranking_strategies.json`
   - Package: cogniverse-vespa (strategy extractor)

4. **Profile-Schema Mapping** (Naming convention) → Profile to schema mapping
   - Convention: Each profile has a dedicated schema file named `{profile_name}_schema.json`
   - Location: `configs/schemas/` directory
   - Example: Profile `video_colpali_smol500_mv_frame` → `video_colpali_smol500_mv_frame_schema.json`

### Runtime Configuration

Configuration is loaded via `ConfigManager` from cogniverse-foundation:

```python
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_vespa.config.config_store import VespaConfigStore
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

# Initialize config store and manager
config_store = VespaConfigStore(vespa_url="http://localhost", vespa_port=8080)
config_manager = ConfigManager(store=config_store)

# Load system configuration for tenant
system_config = config_manager.get_system_config(tenant_id="acme")

# Load schema and strategy information
loader = FilesystemSchemaLoader(Path("configs/schemas"))
strategies = loader.load_ranking_strategies()

# Get strategy config for a specific schema
schema_name = "video_colpali_smol500_mv_frame"
strategy_name = "hybrid_float_bm25"
strategy_config = strategies.get(schema_name, {}).get(strategy_name)

if strategy_config:
    print(f"Strategy type: {strategy_config['strategy_type']}")
    print(f"Needs float: {strategy_config['needs_float_embeddings']}")
    print(f"Uses nearestNeighbor: {strategy_config['use_nearestneighbor']}")
```

## 9. Monitoring & Validation

### Schema Health Checks

Use VespaSchemaManager to validate tenant schemas:

```python
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

manager = VespaSchemaManager(
    backend_endpoint="http://localhost",
    backend_port=19071,
    schema_loader=schema_loader,  # Optional, needed for tenant operations
    schema_registry=schema_registry  # Optional, needed for tenant operations
)

# Check if a specific tenant schema exists
exists = manager.tenant_schema_exists(
    tenant_id="acme",
    base_schema_name="video_colpali_smol500_mv_frame"
)

# Get the tenant-specific schema name
schema_name = manager.get_tenant_schema_name(
    tenant_id="acme",
    base_schema_name="video_colpali_smol500_mv_frame"
)
# Returns: "video_colpali_smol500_mv_frame_acme"
```

### Health Check via Vespa Query

```python
from vespa.application import Vespa

vespa = Vespa(url="http://localhost:8080")

# Check schema accessibility
try:
    response = vespa.query(
        yql=f"select * from {schema_name} where true limit 1"
    )
    doc_count = response.number_documents_retrieved
    print(f"Schema {schema_name}: {doc_count} documents accessible")
except Exception as e:
    print(f"Schema {schema_name}: NOT ACCESSIBLE - {e}")
```

## Troubleshooting

### Common Issues

**Strategy Not Available**
```python
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

loader = FilesystemSchemaLoader(Path("configs/schemas"))
strategies = loader.load_ranking_strategies()

# Check if strategy exists for schema
schema_name = "video_colpali_smol500_mv_frame"
strategy_name = "float_float"

if schema_name not in strategies or strategy_name not in strategies[schema_name]:
    raise ValueError(f"Strategy {strategy_name} not available for schema {schema_name}")
```

**nearestNeighbor vs Tensor Ranking**

Use the extracted `RankingStrategyInfo` to determine the correct search method:
```python
from cogniverse_vespa.ranking_strategy_extractor import RankingStrategyExtractor

extractor = RankingStrategyExtractor()
strategies = extractor.extract_from_schema(schema_path)
strategy_info = strategies.get(strategy_name)

if strategy_info.use_nearestneighbor:
    # Global embedding - use nearestNeighbor
    field = strategy_info.nearestneighbor_field  # "embedding" or "embedding_binary"
    tensor = strategy_info.nearestneighbor_tensor  # "qt" or "qtb"
    yql = f"where ({{targetHits:100}}nearestNeighbor({field}, {tensor}))"
elif strategy_info.strategy_type.value == "pure_visual":
    # Patch embedding - use tensor ranking expression
    yql = "where true"  # Ranking handled by first-phase expression
else:
    # Text search
    yql = "where userInput(@userQuery)"
```

**Dimension Mismatch**

Check tensor dimensions from the schema JSON:
```python
import json

with open("configs/schemas/video_colpali_smol500_mv_frame_schema.json") as f:
    schema = json.load(f)

# Find embedding field dimension
for field in schema["document"]["fields"]:
    if field["name"] == "embedding":
        # Parse dimension from type like "tensor<bfloat16>(patch{}, v[128])"
        field_type = field["type"]
        # Extract dimension - e.g., v[128] → 128
        print(f"Schema embedding dimension: {field_type}")

# Compare with actual embedding
if embedding_dim != expected_dim:
    raise ValueError(f"Dimension mismatch: expected {expected_dim}, got {embedding_dim}")
```

## Best Practices

1. **Always deploy schemas first** before ingesting data
2. **Use extracted strategies** from ranking_strategies.json
3. **Validate strategy compatibility** before querying
4. **Monitor schema health** regularly
5. **Keep schema naming consistent**: `{profile}_{tenant_id}`
6. **Test strategy changes** in staging first
7. **Document custom modifications** to schemas

---

**Package Architecture Note**: Schema-driven processing spans multiple packages in the layered architecture:

- **Foundation Layer**: cogniverse-sdk (schema interfaces), cogniverse-foundation (config)
- **Core Layer**: cogniverse-core (schema registries, base processing)
- **Implementation Layer**: cogniverse-vespa (schema management, deployment), cogniverse-agents (schema-based agents)
- **Application Layer**: cogniverse-runtime (ingestion pipelines using schemas)

**Key Locations:**

- Schema files: `configs/schemas/*.json`
- Schema processing: `libs/vespa/cogniverse_vespa/vespa_schema_manager.py`
- Tenant schema routing: `libs/vespa/cogniverse_vespa/vespa_schema_manager.py`
- Configuration storage: `libs/vespa/cogniverse_vespa/config/config_store.py` (VespaConfigStore)
