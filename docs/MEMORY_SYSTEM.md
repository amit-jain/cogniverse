# Agent Memory System

The agent memory system uses **Mem0** with **Vespa** as the persistent backend to provide multi-tenant, per-agent memory capabilities.

## Architecture

```
┌─────────────────┐
│  Agent Code     │
│  (RoutingAgent, │
│   VideoAgent,   │
│   etc.)         │
└────────┬────────┘
         │ uses MemoryAwareMixin
         ↓
┌─────────────────┐
│ MemoryAware     │
│ Mixin           │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Mem0Memory      │
│ Manager         │
│ (Singleton)     │
└────────┬────────┘
         │
         ↓
┌─────────────────┐      ┌─────────────────┐
│ Mem0 Framework  │◄────►│ VespaVectorStore│
│                 │      │ (Custom Backend)│
└─────────────────┘      └────────┬────────┘
                                  │
                                  ↓
                         ┌─────────────────┐
                         │ Vespa Instance  │
                         │ (localhost:8080)│
                         └─────────────────┘
```

## Features

- **Multi-tenant isolation**: Each tenant's memories are completely isolated
- **Per-agent namespacing**: Each agent has its own memory space within a tenant
- **Semantic search**: Uses embeddings for relevant memory retrieval
- **Persistent storage**: All memories stored in Vespa for durability
- **Simple API**: Clean interface for add/search/delete operations

## Setup

### 1. Deploy Vespa Schema

```bash
# Deploy the agent_memories schema to Vespa
uv run python scripts/deploy_memory_schema.py
```

This creates the `agent_memories` schema in Vespa with:
- Vector embeddings (1536 dimensions for text-embedding-3-small)
- Text fields with BM25 indexing
- Multi-tenant fields (user_id, agent_id)
- Metadata storage
- Hybrid ranking (semantic + BM25)

### 2. Initialize Memory in Agents

```python
from src.app.agents.memory_aware_mixin import MemoryAwareMixin

class MyAgent(MemoryAwareMixin):
    def __init__(self):
        super().__init__()

        # Initialize memory
        self.initialize_memory(
            agent_name="my_agent",
            tenant_id="tenant_123",
            vespa_host="localhost",  # optional
            vespa_port=8080,         # optional
        )
```

### 3. Use Memory Operations

```python
# Add memory
self.update_memory("User prefers video content")

# Search memory
context = self.get_relevant_context("user preferences", top_k=5)

# Remember success/failure
self.remember_success(query="search cats", result="found 10 videos")
self.remember_failure(query="search xyz", error="no results")

# Get memory stats
stats = self.get_memory_stats()

# Clear all memory
self.clear_memory()
```

## Memory Manager API

The `Mem0MemoryManager` provides a lower-level API:

```python
from src.common.mem0_memory_manager import Mem0MemoryManager

manager = Mem0MemoryManager()

# Initialize (singleton, only once)
manager.initialize(
    vespa_host="localhost",
    vespa_port=8080,
    collection_name="agent_memories",
)

# Add memory
memory_id = manager.add_memory(
    content="Memory content",
    tenant_id="tenant_123",
    agent_name="my_agent",
    metadata={"key": "value"},
)

# Search memory
results = manager.search_memory(
    query="search query",
    tenant_id="tenant_123",
    agent_name="my_agent",
    top_k=5,
)

# Get all memories
all_memories = manager.get_all_memories(
    tenant_id="tenant_123",
    agent_name="my_agent",
)

# Delete specific memory
manager.delete_memory(memory_id, tenant_id, agent_name)

# Clear all agent memories
manager.clear_agent_memory(tenant_id, agent_name)

# Health check
is_healthy = manager.health_check()

# Get stats
stats = manager.get_memory_stats(tenant_id, agent_name)
```

## Configuration

The memory system uses local Ollama models by default:

- **LLM Model**: `llama3.2:latest` (for memory management operations)
- **Embedding Model**: `nomic-embed-text` (768 dimensions)
- **Vespa Endpoint**: `localhost:8080`
- **Collection Name**: `agent_memories`
- **Ollama Endpoint**: `localhost:11434`

### Required Ollama Models

Pull these models before using the memory system:

```bash
# Embedding model (required)
ollama pull nomic-embed-text

# LLM for memory management (required)
# llama3.2:latest should already be available
ollama pull llama3.2
```

Override via initialization:

```python
manager.initialize(
    vespa_host="localhost",
    vespa_port=8080,
    llm_model="llama3.2:latest",
    embedding_model="nomic-embed-text",
    ollama_base_url="http://localhost:11434",
)
```

## Testing

### Unit Tests

```bash
# Test Mem0MemoryManager
JAX_PLATFORM_NAME=cpu PYTHONPATH=$PWD uv run pytest tests/memory/unit/test_mem0_memory_manager.py -v

# Test MemoryAwareMixin
JAX_PLATFORM_NAME=cpu PYTHONPATH=$PWD uv run pytest tests/memory/unit/test_memory_aware_mixin.py -v
```

### Integration Tests (requires Vespa)

```bash
# Deploy schema first
uv run python scripts/deploy_memory_schema.py

# Run integration tests
JAX_PLATFORM_NAME=cpu PYTHONPATH=$PWD uv run pytest tests/memory/integration/test_mem0_vespa_integration.py -v -s
```

## Vespa Schema Details

The `agent_memories` schema includes:

### Fields

- `id` (string): Unique memory identifier
- `embedding` (tensor): 1536-dimensional embedding vector
- `text` (string): Memory content with BM25 indexing
- `user_id` (string): Tenant identifier (fast-search)
- `agent_id` (string): Agent name (fast-search)
- `metadata_` (string): JSON metadata
- `created_at` (long): Timestamp

### Ranking Profiles

1. **semantic_search**: Pure vector similarity (default)
2. **bm25**: Pure keyword matching
3. **hybrid**: 70% semantic + 30% BM25

### Example Vespa Query

```python
# Search with tenant/agent filtering
params = {
    "yql": 'select * from agent_memories where user_id contains "tenant_123" and agent_id contains "my_agent"',
    "hits": 5,
    "ranking.profile": "semantic_search",
    "input.query(q)": str(embedding_vector),
}
```

## Migration from Letta

The system **no longer supports Letta**. All Letta code has been removed:

- ❌ Removed: `letta_memory_manager.py`
- ❌ Removed: `letta_memory_config.py`
- ❌ Removed: Letta integration tests
- ❌ Uninstalled: `letta` and `letta-client` packages

There is **no migration utility** and **no backward compatibility**. Start fresh with Mem0.

## Troubleshooting

### Vespa Not Running

```bash
# Check Vespa status
curl http://localhost:8080/ApplicationStatus

# If not running, start Vespa (Docker example)
docker run --detach --name vespa --hostname vespa-container \
  --publish 8080:8080 --publish 19071:19071 \
  vespaengine/vespa
```

### Schema Not Deployed

```bash
# Deploy schema
uv run python scripts/deploy_memory_schema.py

# Verify deployment
curl http://localhost:8080/document/v1/agent_memories/agent_memories/docid/test
# Should return 404 (document not found) but confirms schema exists
```

### Memory Not Initializing

Check that:
1. Vespa is running
2. Schema is deployed
3. OpenAI API key is set: `export OPENAI_API_KEY=sk-...`
4. Mem0 is installed: `uv add mem0ai`

### Search Returns No Results

- Memories need time to index (2-5 seconds)
- Semantic search requires relevant embeddings
- Check tenant_id and agent_name match exactly

## Performance Considerations

- **Indexing latency**: 2-5 seconds for new memories to be searchable
- **Search latency**: <100ms for vector search with HNSW index
- **Embedding cost**: ~$0.0001 per 1000 tokens (text-embedding-3-small)
- **LLM cost**: Minimal, only for memory deduplication/updates

## Security

- **Tenant isolation**: Enforced at query level via `user_id` filtering
- **Agent isolation**: Enforced at query level via `agent_id` filtering
- **No authentication**: Vespa endpoint should be secured separately
- **Metadata**: Stored as plain text, do not put sensitive data

## Limitations

- **No core memory**: Mem0 doesn't have Letta's "always in context" core memory
- **No conversational memory**: Can't "talk to" memory like Letta agents
- **Deduplication**: Mem0 handles this automatically, no control
- **Schema changes**: Require Vespa restart/redeployment
