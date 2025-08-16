# Architecture Refactoring Plan

## Overview
This document outlines the refactoring of the Cogniverse codebase into a clean, modular architecture with pluggable backend support.

## Target Architecture

### Top-Level Structure
```
src/
├── common/                # Foundation module - shared utilities & contracts
├── app/                   # Main application module (ingestion, search, agents)
├── evaluation/            # Evaluation and metrics module
├── visualization/         # Visualization and reporting (optional)
└── backends/              # Backend implementations (vespa, elasticsearch, etc.)
```

Only 4-5 top-level modules for maximum clarity!

### Detailed Module Structure

#### 1. Common Module (Foundation)
```
src/common/
├── __init__.py
├── config.py                      # Global configuration loader
├── constants.py                   # Shared constants
│
├── core/                          # Core contracts and structures
│   ├── __init__.py
│   ├── documents.py              # Document, MediaType, TemporalInfo classes
│   ├── interfaces.py             # All ABCs (IngestionBackend, SearchBackend)
│   ├── schemas.py                # Schema definitions and validators
│   ├── exceptions.py             # Custom exceptions (IncompatibleBackendError, etc.)
│   └── registry.py               # Backend registry for auto-discovery
│
├── models/                        # Model loaders
│   ├── __init__.py
│   ├── base.py                   # Abstract ModelLoader
│   ├── colpali.py                # ColPali model implementation
│   ├── colqwen.py                # ColQwen model implementation
│   ├── videoprism.py             # VideoPrism model implementation
│   └── factory.py                # Model factory/registry
│
├── cache/                         # Caching utilities
│   ├── __init__.py
│   ├── manager.py                # Generic cache manager
│   ├── pipeline_cache.py         # Pipeline-specific caching
│   └── lru_cache.py             # LRU cache implementation
│
└── utils/                         # Other utilities
    ├── __init__.py
    ├── logging.py                # Logging configuration
    ├── file_utils.py             # File operations
    ├── retry.py                  # Retry logic with backoff
    └── validators.py             # Input validators
```

#### 2. App Module (Main Application)
```
src/app/
├── __init__.py
│
├── ingestion/                     # Ingestion subsystem
│   ├── __init__.py
│   ├── pipeline.py               # Main ingestion pipeline
│   ├── orchestrator.py           # Pipeline orchestration
│   └── processors/               # Processing steps
│       ├── __init__.py
│       ├── keyframe_extractor.py
│       ├── embedding_generator.py
│       ├── audio_transcriber.py
│       └── description_generator.py
│
├── search/                        # Search subsystem
│   ├── __init__.py
│   ├── service.py                # Main search service
│   ├── query_builder.py          # Query construction
│   └── result_processor.py       # Result processing
│
└── agents/                        # Agent subsystem
    ├── __init__.py
    ├── video_agent.py            # Video search agent
    ├── query_encoder.py          # Query encoding logic
    └── server.py                 # Agent API server
```

#### 3. Evaluation Module
```
src/evaluation/
├── __init__.py
├── evaluator.py                  # Main evaluator class
├── metrics.py                    # Metric definitions
├── datasets.py                   # Dataset management
├── inspect_tasks/
│   └── video_retrieval.py       # Video retrieval evaluation
└── scorers/
    ├── ragas_scorer.py           # RAGAS scoring
    └── llm_judge.py             # LLM-based evaluation
```

#### 4. Visualization Module (Optional)
```
src/visualization/
├── __init__.py
├── dashboard.py                  # Streamlit/Gradio dashboards
├── plots.py                      # Plotting utilities
└── reports.py                    # Report generation
```

#### 5. Backend Modules
```
src/backends/vespa/               # Vespa backend
├── __init__.py                   # Auto-registers with registry
├── backend.py                    # Implements IngestionBackend & SearchBackend
├── client.py                     # Vespa-specific client
├── schema_translator.py          # Schema translation
└── schema_deployer.py            # Schema deployment

src/backends_elasticsearch/       # Future: Elasticsearch (can be separate package)
├── __init__.py
├── backend.py
├── client.py
└── schema_translator.py

src/backends_pinecone/            # Future: Pinecone (can be separate package)
├── __init__.py
├── backend.py
└── client.py
```

## Dependency Rules

```python
# Clean dependency hierarchy (dependencies flow downward only):
common → (none)                    # Foundation - no dependencies
backends/* → common                # Backends only depend on common
app → common                       # App only depends on common (gets backends via registry)
evaluation → common, app.search    # Can evaluate search functionality
visualization → common, evaluation # Can visualize evaluation results

# Key insight: app doesn't directly import backends!
# It discovers them through the registry in common/core/registry.py
```

## Key Interfaces

### Backend Interfaces (in common/core/interfaces.py)
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class SchemaRequirements:
    """Requirements extracted from schema"""
    requires_multi_vector: bool = False
    requires_binary_vectors: bool = False
    requires_text_search: bool = False
    requires_custom_ranking: bool = False
    vector_dimensions: Optional[int] = None

class IngestionBackend(ABC):
    """Interface for ingestion backends"""
    
    @abstractmethod
    def validate_schema(self, schema_path: str) -> None:
        """Validate backend can support schema
        Raises: IncompatibleBackendError if not supported
        """
        pass
    
    @abstractmethod
    def translate_schema(self, schema_path: str) -> Any:
        """Translate unified schema to backend format"""
        pass
    
    @abstractmethod
    def deploy_schema(self, schema: Any) -> None:
        """Deploy schema to backend"""
        pass
    
    @abstractmethod
    def format_documents(self, documents: List[Dict]) -> List[Dict]:
        """Format documents for backend-specific ingestion"""
        pass
    
    @abstractmethod
    def ingest_documents(self, documents: List[Dict]) -> Dict[str, Any]:
        """Ingest formatted documents"""
        pass

class SearchBackend(ABC):
    """Interface for search backends"""
    
    @abstractmethod
    def validate_schema(self, schema_path: str) -> None:
        """Validate backend can support schema for search"""
        pass
    
    @abstractmethod
    def format_query(self, query: Dict) -> Any:
        """Format query for backend-specific search"""
        pass
    
    @abstractmethod
    def search(self, query: Any) -> Dict[str, Any]:
        """Execute search query"""
        pass
```

### Backend Registry (in common/core/registry.py)
```python
class BackendRegistry:
    """Central registry for backend auto-discovery"""
    
    _ingestion_backends = {}
    _search_backends = {}
    
    @classmethod
    def register_ingestion(cls, name: str, backend_class):
        """Called by backends to self-register"""
        cls._ingestion_backends[name] = backend_class
    
    @classmethod
    def get_ingestion_backend(cls, name: str):
        """Used by app to discover backends"""
        if name not in cls._ingestion_backends:
            cls._try_import_backend(name)
        return cls._ingestion_backends.get(name)
```

## Migration Plan

### Phase 1: Preparation (Week 1)
**Goal**: Set up new structure without breaking existing code

1. **Create new directory structure**
   ```bash
   mkdir -p src/common/core src/common/models src/common/cache src/common/utils
   mkdir -p src/app/ingestion/processors src/app/search src/app/agents
   mkdir -p src/backends/vespa
   ```

2. **Create base interfaces and registry**
   - Create `src/common/core/interfaces.py` with all ABCs
   - Create `src/common/core/registry.py` with BackendRegistry
   - Create `src/common/core/exceptions.py` with custom exceptions

3. **Set up compatibility layer**
   - Create temporary import shims to maintain backward compatibility
   - Add deprecation warnings to old imports

### Phase 2: Common Module Consolidation (Week 1-2)
**Goal**: Consolidate core, models, utils into common

1. **Move core components**
   ```bash
   # Move existing core to common/core
   mv src/core/documents.py src/common/core/
   mv src/core/strategy.py src/common/core/schemas.py
   ```

2. **Move models**
   ```bash
   # Move model loaders to common/models
   mv src/models/* src/common/models/
   ```

3. **Move utilities**
   ```bash
   # Move utils to common/utils
   mv src/utils/* src/common/utils/
   # Move cache to common/cache
   mv src/cache/* src/common/cache/
   ```

4. **Update imports**
   ```python
   # Find and replace across codebase:
   # "from src.core" → "from src.common.core"
   # "from src.models" → "from src.common.models"
   # "from src.utils" → "from src.common.utils"
   # "from src.cache" → "from src.common.cache"
   ```

### Phase 3: Create App Module (Week 2)
**Goal**: Organize main application logic

1. **Move ingestion components**
   ```bash
   # Move pipeline
   mv src/processing/unified_video_pipeline.py src/app/ingestion/pipeline.py
   
   # Move processors
   mv src/processing/pipeline_steps/keyframe_extractor.py src/app/ingestion/processors/
   mv src/processing/pipeline_steps/embedding_generator/* src/app/ingestion/processors/
   ```

2. **Move search components**
   ```bash
   # Move search services
   mv src/search/search_service.py src/app/search/service.py
   mv src/search/search.py src/app/search/query_builder.py
   ```

3. **Move agent components**
   ```bash
   # Move agents
   mv src/agents/* src/app/agents/
   ```

4. **Update internal imports**
   - Update all imports within app module
   - Ensure app only imports from common, not from backends directly

### Phase 4: Refactor Backend Module (Week 2-3)
**Goal**: Create pluggable backend architecture

1. **Create Vespa backend module**
   ```bash
   # Move Vespa-specific code
   mv src/processing/vespa/* src/backends/vespa/
   mv src/search/vespa_search_backend.py src/backends/vespa/
   ```

2. **Implement backend interfaces**
   - Create `src/backends/vespa/backend.py` implementing both IngestionBackend and SearchBackend
   - Refactor existing Vespa code to fit new interfaces

3. **Add self-registration**
   ```python
   # src/backends/vespa/__init__.py
   from src.common.core.registry import BackendRegistry
   from .backend import VespaBackend
   
   BackendRegistry.register_ingestion("vespa", VespaBackend)
   BackendRegistry.register_search("vespa", VespaBackend)
   ```

### Phase 5: Update Scripts and Tests (Week 3)
**Goal**: Update all scripts and tests to use new structure

1. **Update scripts**
   ```python
   # scripts/ingestion.py
   from src.app.ingestion import IngestionPipeline
   from src.common.config import get_config
   
   # scripts/search.py
   from src.app.search import SearchService
   ```

2. **Update tests**
   - Update all test imports
   - Add integration tests for backend registry
   - Add tests for schema validation

3. **Update configuration**
   - Migrate from JSON to YAML for main config
   - Update config paths in code

### Phase 6: Cleanup (Week 4)
**Goal**: Remove old code and finalize migration

1. **Remove deprecated modules**
   ```bash
   rm -rf src/processing
   rm -rf src/core
   rm -rf src/models
   rm -rf src/utils
   rm -rf src/cache
   ```

2. **Remove compatibility shims**
   - Remove temporary import redirects
   - Clean up deprecation warnings

3. **Documentation**
   - Update README with new structure
   - Update development documentation
   - Create backend development guide

### Phase 7: Validation (Week 4)
**Goal**: Ensure everything works correctly

1. **Run comprehensive tests**
   ```bash
   # Run all tests
   pytest tests/ -v
   
   # Run integration tests
   python scripts/test_integration.py
   ```

2. **Test each profile**
   ```bash
   # Test ingestion with each profile
   for profile in video_colpali_smol500_mv_frame video_colqwen_omni_mv_chunk_30s; do
     python scripts/ingestion.py --profile $profile --backend vespa
   done
   ```

3. **Performance validation**
   - Compare performance before/after refactoring
   - Ensure no regression in processing speed

## Success Criteria

1. **Clean Architecture**
   - [ ] Only 4-5 top-level modules
   - [ ] Clear dependency hierarchy
   - [ ] No circular dependencies

2. **Pluggable Backends**
   - [ ] Backends self-register on import
   - [ ] App doesn't directly import backends
   - [ ] Easy to add new backends

3. **Backward Compatibility**
   - [ ] All existing scripts work
   - [ ] All tests pass
   - [ ] No performance regression

4. **Developer Experience**
   - [ ] Clear module boundaries
   - [ ] Intuitive import paths
   - [ ] Good error messages

## Risk Mitigation

1. **Gradual Migration**: Use compatibility shims to avoid breaking changes
2. **Comprehensive Testing**: Test at each phase before proceeding
3. **Version Control**: Create a branch for refactoring, merge only when complete
4. **Rollback Plan**: Keep old structure temporarily, can revert if issues arise

## Timeline

- **Week 1**: Preparation and Common module consolidation
- **Week 2**: Create App module and refactor backends
- **Week 3**: Update scripts, tests, and configuration
- **Week 4**: Cleanup, validation, and documentation

Total estimated time: 4 weeks (part-time) or 1-2 weeks (full-time)

## Next Steps

1. Review and approve this plan
2. Create feature branch: `git checkout -b refactor/modular-architecture`
3. Start Phase 1: Create new directory structure
4. Set up CI/CD to run tests on the refactoring branch