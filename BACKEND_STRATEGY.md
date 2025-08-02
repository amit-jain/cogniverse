# Multi-Backend Architecture Strategy for Cogniverse

## Executive Summary

This document outlines a comprehensive strategy for evolving Cogniverse to support multiple pluggable backends with profile-based selection through a UI. After analyzing Haystack's architecture, we recommend building our own lightweight abstraction layer that leverages Cogniverse's existing strengths while maintaining simplicity.

## Current State Analysis

### Strengths
- **Profile System**: Already supports 6 different video processing profiles
- **Backend Abstraction**: Partial abstraction with Vespa and Byaldi backends
- **Factory Pattern**: Used for query encoders and embedding generators
- **Configuration-Driven**: Centralized config with runtime profile switching
- **Modular Pipeline**: Configurable processing steps

### Limitations
- Backend selection tied to profile configuration
- No unified backend interface
- Limited to Vespa for production use
- Profile switching requires config changes

## Haystack Comparison

### Haystack Advantages
- Mature component abstraction
- Extensive integrations (30+ vector DBs)
- Pipeline serialization
- Built-in async support

### Haystack Disadvantages
- Overhead for our specific use case
- Learning curve for team
- May conflict with existing architecture
- Generic design vs. our video-specific optimizations

### Decision: Build Lightweight Abstraction Layer
Given our specific video RAG requirements and existing architecture, we recommend building a lightweight backend abstraction layer rather than adopting Haystack.

## Proposed Architecture

### 1. Backend Abstraction Layer

```python
# src/core/backends/base.py
class BaseSearchBackend(ABC):
    """Abstract base class for all search backends"""
    
    @abstractmethod
    async def index(self, documents: List[VideoDocument]) -> bool:
        """Index video documents"""
        pass
    
    @abstractmethod
    async def search(self, query: SearchQuery) -> SearchResults:
        """Execute search query"""
        pass
    
    @abstractmethod
    async def delete(self, doc_ids: List[str]) -> bool:
        """Delete documents"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> BackendCapabilities:
        """Return backend capabilities"""
        pass

class BackendCapabilities:
    supports_vector_search: bool
    supports_text_search: bool
    supports_hybrid_search: bool
    max_embedding_dimensions: int
    supports_binary_embeddings: bool
    supports_filtering: bool
    supports_faceting: bool
```

### 2. Backend Registry System

```python
# src/core/backends/registry.py
class BackendRegistry:
    """Registry for managing available backends"""
    
    def register(self, name: str, backend_class: Type[BaseSearchBackend]):
        """Register a new backend"""
        
    def get(self, name: str, config: Dict) -> BaseSearchBackend:
        """Get configured backend instance"""
        
    def list_available(self) -> List[BackendInfo]:
        """List all registered backends with capabilities"""

# Auto-registration
@backend_registry.register("vespa")
class VespaSearchBackend(BaseSearchBackend):
    pass

@backend_registry.register("elasticsearch")  
class ElasticsearchBackend(BaseSearchBackend):
    pass

@backend_registry.register("weaviate")
class WeaviateBackend(BaseSearchBackend):
    pass
```

### 3. Enhanced Profile System

```python
# src/core/profiles/profile_manager.py
class ProfileManager:
    """Manages video processing profiles"""
    
    def create_profile(self, profile_def: ProfileDefinition) -> Profile:
        """Create new profile"""
        
    def get_profile(self, name: str) -> Profile:
        """Get profile by name"""
        
    def list_profiles(self, backend: Optional[str] = None) -> List[Profile]:
        """List profiles, optionally filtered by backend compatibility"""
        
    def validate_profile_backend_compatibility(self, profile: Profile, backend: BaseSearchBackend) -> bool:
        """Check if profile is compatible with backend"""

class Profile:
    name: str
    display_name: str
    description: str
    model_config: ModelConfig
    pipeline_config: PipelineConfig
    compatible_backends: List[str]
    search_strategies: List[SearchStrategy]
    ui_metadata: UIMetadata  # For UI display
```

### 4. Dynamic Configuration System

```python
# src/core/config/dynamic_config.py
class DynamicConfig:
    """Runtime configuration management"""
    
    def set_backend(self, backend_name: str, backend_config: Dict):
        """Set active backend"""
        
    def set_profile(self, profile_name: str):
        """Set active profile"""
        
    def get_valid_profiles_for_backend(self, backend_name: str) -> List[str]:
        """Get profiles compatible with backend"""
        
    def validate_configuration(self) -> ConfigValidation:
        """Validate current configuration"""
```

### 5. Backend-Specific Implementations

#### Vespa Backend (Enhanced)
```python
# src/backends/vespa/vespa_backend.py
class VespaBackend(BaseSearchBackend):
    def __init__(self, config: VespaConfig):
        self.client = Vespa(url=config.url, port=config.port)
        self.schema_manager = VespaSchemaManager()
        
    async def index(self, documents: List[VideoDocument]) -> bool:
        # Use existing Vespa implementation
        pass
```

#### Elasticsearch Backend
```python
# src/backends/elasticsearch/es_backend.py
class ElasticsearchBackend(BaseSearchBackend):
    def __init__(self, config: ESConfig):
        self.client = AsyncElasticsearch(hosts=config.hosts)
        
    async def index(self, documents: List[VideoDocument]) -> bool:
        # Implement ES-specific indexing
        pass
```

#### Weaviate Backend
```python
# src/backends/weaviate/weaviate_backend.py
class WeaviateBackend(BaseSearchBackend):
    def __init__(self, config: WeaviateConfig):
        self.client = weaviate.Client(url=config.url)
        
    async def index(self, documents: List[VideoDocument]) -> bool:
        # Implement Weaviate-specific indexing
        pass
```

### 6. UI Integration Layer

```python
# src/api/config_api.py
class ConfigurationAPI:
    """API endpoints for UI configuration"""
    
    @app.get("/api/backends")
    async def list_backends():
        """List available backends with capabilities"""
        
    @app.get("/api/profiles")
    async def list_profiles(backend: Optional[str] = None):
        """List profiles, optionally filtered by backend"""
        
    @app.post("/api/config/backend")
    async def set_backend(backend_name: str, config: Dict):
        """Set active backend"""
        
    @app.post("/api/config/profile")
    async def set_profile(profile_name: str):
        """Set active profile"""
        
    @app.get("/api/config/validate")
    async def validate_config():
        """Validate current configuration"""
```

### 7. Profile-Backend Compatibility Matrix

| Profile | Vespa | Elasticsearch | Weaviate | Pinecone | Qdrant |
|---------|-------|---------------|----------|----------|---------|
| frame_based_colpali | ✅ | ✅ | ✅ | ❌ | ✅ |
| direct_video_colqwen | ✅ | ✅ | ✅ | ✅ | ✅ |
| direct_video_global | ✅ | ✅ | ✅ | ✅ | ✅ |
| videoprism_base | ✅ | ✅ | ✅ | ❌ | ✅ |
| videoprism_large | ✅ | ✅ | ✅ | ❌ | ✅ |

### 8. UI Design Mockup

```
┌─────────────────────────────────────────────────────────┐
│                  Cogniverse Configuration                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Backend Selection:                                     │
│  ┌─────────────────────────────────────┐              │
│  │ ▼ Vespa (Active)                    │              │
│  └─────────────────────────────────────┘              │
│    • Elasticsearch                                      │
│    • Weaviate                                          │
│    • Pinecone                                          │
│                                                         │
│  Profile Selection:                                     │
│  ┌─────────────────────────────────────┐              │
│  │ ▼ frame_based_colpali               │              │
│  └─────────────────────────────────────┘              │
│    • direct_video_global                               │
│    • direct_video_global_large                         │
│    • videoprism_base                                   │
│                                                         │
│  [Apply Configuration]  [Test Connection]              │
│                                                         │
│  Status: ✅ Connected to Vespa                         │
│  Profile: frame_based_colpali (768-dim embeddings)    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Implementation Roadmap

### Phase 1: Backend Abstraction (Week 1-2)
1. Create base backend interface
2. Implement backend registry
3. Refactor existing Vespa backend
4. Add comprehensive testing

### Phase 2: Profile Enhancement (Week 3)
1. Enhance profile system with backend compatibility
2. Create profile validation system
3. Update configuration management
4. Migration existing profiles

### Phase 3: New Backend Implementation (Week 4-5)
1. Implement Elasticsearch backend
2. Implement Weaviate backend
3. Add backend-specific optimizations
4. Performance testing

### Phase 4: UI Integration (Week 6)
1. Create configuration API endpoints
2. Build React configuration UI
3. Add real-time validation
4. Integration testing

### Phase 5: Production Readiness (Week 7-8)
1. Add monitoring and metrics
2. Create migration tools
3. Documentation
4. Load testing

## Migration Strategy

### 1. Backward Compatibility
- Maintain existing config format
- Add adapter layer for legacy code
- Gradual migration path

### 2. Data Migration
```python
# src/migration/backend_migrator.py
class BackendMigrator:
    async def migrate(self, source: BaseSearchBackend, target: BaseSearchBackend, profile: str):
        """Migrate data between backends"""
```

### 3. Zero-Downtime Migration
1. Deploy new backend
2. Dual-write to both backends
3. Verify data consistency
4. Switch read traffic
5. Decommission old backend

## Benefits of This Approach

### 1. Flexibility
- Easy to add new backends
- Profile-backend decoupling
- Runtime configuration

### 2. Maintainability
- Clear separation of concerns
- Testable components
- Self-documenting code

### 3. User Experience
- Simple UI for configuration
- Real-time validation
- Clear compatibility matrix

### 4. Performance
- Backend-specific optimizations
- Async-first design
- Efficient resource usage

## Conclusion

This strategy provides a clear path to evolve Cogniverse into a truly pluggable multi-backend system while maintaining its current strengths. The lightweight abstraction layer approach balances flexibility with simplicity, avoiding the overhead of adopting a full framework like Haystack while gaining the benefits of a modular architecture.

The phased implementation allows for incremental progress with minimal disruption to existing functionality, and the migration strategy ensures smooth transitions between backends in production environments.