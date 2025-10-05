# Memory System Integration

## Overview

Cogniverse uses **Mem0** with **Vespa vector store backend** to provide multi-tenant, per-agent persistent memory. This enables agents to remember past interactions, learn from successes/failures, and maintain context across sessions.

## Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Layer                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ Routing    │  │ Video      │  │ Summarizer │            │
│  │ Agent      │  │ Search     │  │ Agent      │            │
│  │            │  │ Agent      │  │            │            │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘            │
│         │ inherits      │ inherits      │ inherits         │
│         └───────────────┴───────────────┘                  │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                  MemoryAwareMixin                            │
│  ┌────────────────────────────────────────────────┐         │
│  │ Standard memory interface for all agents       │         │
│  │ - initialize_memory(agent_name, tenant_id)     │         │
│  │ - get_relevant_context(query, top_k)           │         │
│  │ - update_memory(content, metadata)             │         │
│  │ - remember_success(query, result)              │         │
│  │ - remember_failure(query, error)               │         │
│  │ - clear_memory()                               │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│              Mem0MemoryManager (Singleton)                   │
│  ┌────────────────────────────────────────────────┐         │
│  │ Multi-tenant memory management                 │         │
│  │ - add_memory(content, tenant, agent, metadata) │         │
│  │ - search_memory(query, tenant, agent, top_k)   │         │
│  │ - get_all_memories(tenant, agent)              │         │
│  │ - delete_memory(id, tenant, agent)             │         │
│  │ - clear_agent_memory(tenant, agent)            │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                    Mem0 Framework                            │
│  ┌────────────────────────────────────────────────┐         │
│  │ LLM: Ollama (llama3.2) via OpenAI API          │         │
│  │ - Memory deduplication                          │         │
│  │ - Memory updates                                │         │
│  │ - Relevance scoring                             │         │
│  └────────────────────────────────────────────────┘         │
│  ┌────────────────────────────────────────────────┐         │
│  │ Embedder: Ollama (nomic-embed-text)             │         │
│  │ - 768-dimensional embeddings                    │         │
│  │ - Semantic search                               │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│              VespaVectorStore (Custom Backend)               │
│  ┌────────────────────────────────────────────────┐         │
│  │ Implements Mem0 VectorStore interface          │         │
│  │ - insert(vectors, payloads, ids)               │         │
│  │ - search(query_vector, limit, filters)         │         │
│  │ - delete(vector_id)                            │         │
│  │ - update(vector_id, vector, payload)           │         │
│  │ - get(vector_id)                               │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                Vespa Backend (localhost:8080)                │
│  ┌────────────────────────────────────────────────┐         │
│  │ Schema: agent_memories                          │         │
│  │ - id (string): Memory identifier                │         │
│  │ - embedding (tensor<float>(x[768]))             │         │
│  │ - text (string): Memory content                 │         │
│  │ - user_id (string): Tenant identifier           │         │
│  │ - agent_id (string): Agent name                 │         │
│  │ - metadata_ (string): JSON metadata             │         │
│  │ - created_at (long): Timestamp                  │         │
│  └────────────────────────────────────────────────┘         │
│  ┌────────────────────────────────────────────────┐         │
│  │ Ranking Profiles:                               │         │
│  │ - semantic_search (vector similarity)           │         │
│  │ - bm25 (keyword matching)                       │         │
│  │ - hybrid (70% semantic + 30% BM25)              │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Multi-Tenant Memory Isolation

### Tenant-Agent Namespacing

Each memory is scoped to a specific **tenant + agent** combination:

```
Memory ID: mem_123
├── user_id: "acme"                    (tenant)
├── agent_id: "routing_agent"          (agent name)
├── text: "User prefers video format"  (content)
└── embedding: [0.12, -0.34, ...]     (768 dims)

Memory ID: mem_456
├── user_id: "acme"                    (same tenant)
├── agent_id: "multi_modal_agent"      (different agent)
├── text: "Preferred strategy: hybrid"
└── embedding: [0.45, -0.21, ...]

Memory ID: mem_789
├── user_id: "globex"                  (different tenant)
├── agent_id: "routing_agent"          (same agent name)
├── text: "User prefers text summaries"
└── embedding: [-0.33, 0.67, ...]
```

### Isolation Guarantees

- **Tenant isolation**: Memories for tenant "acme" never visible to "globex"
- **Agent isolation**: Routing agent can't access video search agent's memories
- **Query filtering**: All searches automatically scoped to tenant+agent
- **No cross-tenant leakage**: Enforced at Vespa query level

Example Vespa query with filtering:

```python
# Search routing_agent memories for tenant "acme"
yql = '''
    select * from agent_memories
    where user_id contains "acme"
    and agent_id contains "routing_agent"
'''

# Returns only memories matching both filters
# Cannot access other tenants or agents
```

## MemoryAwareMixin API

### Agent Integration

All agents can inherit memory capabilities:

```python
from src.app.agents.memory_aware_mixin import MemoryAwareMixin

class RoutingAgent(MemoryAwareMixin):
    def __init__(self, tenant_id: str = "default"):
        super().__init__()

        # Initialize memory for this agent
        self.initialize_memory(
            agent_name="routing_agent",
            tenant_id=tenant_id,
            vespa_host="localhost",
            vespa_port=8080
        )

    def route_query(self, query: str):
        # Get relevant context from memory
        context = self.get_relevant_context(query, top_k=5)

        if context:
            # Inject context into routing decision
            enhanced_prompt = self.inject_context_into_prompt(
                prompt=self.base_prompt,
                query=query
            )
            decision = self._route_with_context(enhanced_prompt)
        else:
            decision = self._route_without_context(query)

        # Remember successful routing
        self.remember_success(
            query=query,
            result=f"Routed to {decision.chosen_agent}"
        )

        return decision
```

### Memory Operations

**Initialize Memory:**
```python
success = self.initialize_memory(
    agent_name="routing_agent",
    tenant_id="acme",
    vespa_host="localhost",
    vespa_port=8080
)

if success:
    print("Memory initialized")
```

**Get Relevant Context:**
```python
# Search memory for relevant content
context = self.get_relevant_context(
    query="user preferences for video search",
    top_k=5  # Return top 5 most relevant memories
)

# Returns formatted string:
# 1. User prefers ColPali for visual search
# 2. User typically searches for ML tutorials
# 3. User likes hybrid_float_bm25 strategy
```

**Update Memory:**
```python
# Add new memory
self.update_memory(
    content="User prefers ColPali for visual search queries",
    metadata={"category": "preference", "importance": "high"}
)
```

**Remember Success/Failure:**
```python
# Remember successful interaction
self.remember_success(
    query="machine learning tutorial",
    result="Found 10 relevant videos with ColPali",
    metadata={"strategy": "hybrid_float_bm25"}
)

# Remember failed interaction
self.remember_failure(
    query="xyz non-existent content",
    error="No results found after trying 3 strategies",
    metadata={"attempted_strategies": ["bm25", "float_float", "hybrid"]}
)
```

**Inject Context into Prompts:**
```python
# Automatically inject memory context
enhanced_prompt = self.inject_context_into_prompt(
    prompt="Route this query to the best agent:",
    query="find ML videos"
)

# Result:
# """
# Route this query to the best agent:
#
# ## Relevant Context from Memory:
# 1. User prefers ColPali for visual search
# 2. Previously successful: ML queries → video_search_agent
#
# ## Current Query:
# find ML videos
# """
```

**Check Memory Status:**
```python
# Get memory state
state = self.get_memory_state()
# {
#   "total_memories": 45,
#   "enabled": True,
#   "tenant_id": "acme",
#   "agent_name": "routing_agent"
# }

# Get summary
summary = self.get_memory_summary()
# {
#   "enabled": True,
#   "agent_name": "routing_agent",
#   "tenant_id": "acme",
#   "initialized": True,
#   "total_memories": 45
# }
```

**Clear Memory:**
```python
# Clear all memories for this agent
success = self.clear_memory()

if success:
    print("All memories cleared")
```

## Mem0MemoryManager API

### Direct Memory Management

For direct access to memory operations without using agents:

```python
from src.common.mem0_memory_manager import Mem0MemoryManager

# Get singleton instance
manager = Mem0MemoryManager()

# Initialize (only once)
manager.initialize(
    vespa_host="localhost",
    vespa_port=8080,
    collection_name="agent_memories",
    llm_model="llama3.2",                    # Ollama model
    embedding_model="nomic-embed-text",      # Ollama embedding
    ollama_base_url="http://localhost:11434/v1"
)
```

### Core Operations

**Add Memory:**
```python
memory_id = manager.add_memory(
    content="User prefers video format",
    tenant_id="acme",
    agent_name="routing_agent",
    metadata={"category": "preference"}
)

print(f"Memory added: {memory_id}")
```

**Search Memory:**
```python
results = manager.search_memory(
    query="user video preferences",
    tenant_id="acme",
    agent_name="routing_agent",
    top_k=5
)

for result in results:
    print(f"Memory: {result['memory']}")
    print(f"Score: {result.get('score', 'N/A')}")
```

**Get All Memories:**
```python
all_memories = manager.get_all_memories(
    tenant_id="acme",
    agent_name="routing_agent"
)

print(f"Total: {len(all_memories)} memories")
```

**Delete Memory:**
```python
success = manager.delete_memory(
    memory_id="mem_123",
    tenant_id="acme",
    agent_name="routing_agent"
)
```

**Update Memory:**
```python
success = manager.update_memory(
    memory_id="mem_123",
    content="Updated: User strongly prefers video format",
    tenant_id="acme",
    agent_name="routing_agent",
    metadata={"category": "preference", "importance": "critical"}
)
```

**Clear Agent Memory:**
```python
success = manager.clear_agent_memory(
    tenant_id="acme",
    agent_name="routing_agent"
)
```

**Health Check:**
```python
is_healthy = manager.health_check()

if not is_healthy:
    print("Memory system unhealthy - check Vespa connection")
```

**Get Statistics:**
```python
stats = manager.get_memory_stats(
    tenant_id="acme",
    agent_name="routing_agent"
)

# {
#   "total_memories": 45,
#   "enabled": True,
#   "tenant_id": "acme",
#   "agent_name": "routing_agent"
# }
```

## Configuration

### Ollama Models

Mem0 uses local Ollama models for embeddings and LLM operations:

**Required Models:**
```bash
# Embedding model (required for semantic search)
ollama pull nomic-embed-text

# LLM model (required for memory management)
ollama pull llama3.2
```

**Model Specifications:**
- **nomic-embed-text**: 768-dimensional embeddings, optimized for retrieval
- **llama3.2**: 3B parameter model, handles memory deduplication and updates

### Environment Variables

```bash
# Disable Mem0 telemetry (set automatically)
export MEM0_TELEMETRY=False

# Ollama endpoint
export OLLAMA_BASE_URL=http://localhost:11434

# Vespa endpoint
export VESPA_HOST=localhost
export VESPA_PORT=8080
```

### Initialization Parameters

```python
manager.initialize(
    vespa_host="localhost",           # Vespa host
    vespa_port=8080,                  # Vespa port
    collection_name="agent_memories", # Vespa schema name
    llm_model="llama3.2",            # Ollama LLM
    embedding_model="nomic-embed-text", # Ollama embeddings
    ollama_base_url="http://localhost:11434/v1"  # Ollama API
)
```

## Vespa Schema

### Schema Deployment

```bash
# Deploy agent_memories schema to Vespa
uv run python scripts/deploy_memory_schema.py
```

### Schema Structure

```
Document Type: agent_memories

Fields:
├── id (string)
│   └── indexing: summary | attribute
│
├── embedding (tensor<float>(x[768]))
│   └── indexing: attribute | index
│   └── attribute: distance-metric: angular
│   └── index: hnsw (for fast ANN search)
│
├── text (string)
│   └── indexing: summary | index
│   └── index: enable-bm25
│
├── user_id (string)
│   └── indexing: summary | attribute
│   └── attribute: fast-search
│
├── agent_id (string)
│   └── indexing: summary | attribute
│   └── attribute: fast-search
│
├── metadata_ (string)
│   └── indexing: summary
│
└── created_at (long)
    └── indexing: summary | attribute
```

### Ranking Profiles

**1. semantic_search (default):**
```
Pure vector similarity using cosine distance:
- Uses HNSW index for fast ANN
- Returns most semantically similar memories
```

**2. bm25:**
```
Pure keyword matching:
- Traditional BM25 text scoring
- Good for exact phrase matches
```

**3. hybrid:**
```
Combined semantic + keyword:
- 70% semantic similarity
- 30% BM25 text score
- Best overall performance
```

## Usage Examples

### Example 1: Routing Agent with Memory

```python
from src.app.agents.routing_agent import RoutingAgent

# Initialize with memory
routing_agent = RoutingAgent(tenant_id="acme")

# First query - no context
result1 = routing_agent.route_query("find ML videos")
# Routed to: video_search_agent

# Agent remembers success
routing_agent.remember_success(
    query="find ML videos",
    result="video_search_agent with hybrid strategy worked well"
)

# Second query - similar to first
result2 = routing_agent.route_query("show machine learning content")

# Agent retrieves context from first query
# Context: "Previously successful: ML videos → video_search_agent"
# Routes to same agent with higher confidence
```

### Example 2: Video Search Agent Learning Preferences

```python
from src.app.agents.video_search_agent import VideoSearchAgent

agent = VideoSearchAgent(tenant_id="acme")

# User searches with different strategies
agent.search(query="ML tutorial", strategy="hybrid_float_bm25")
agent.remember_success(
    query="ML tutorial",
    result="hybrid_float_bm25 returned excellent results"
)

agent.search(query="Python basics", strategy="float_float")
agent.remember_failure(
    query="Python basics",
    error="float_float returned poor results"
)

# Later query - agent learns from memory
context = agent.get_relevant_context("ML programming tutorial")
# Returns: "hybrid_float_bm25 worked well for ML content"
# Agent prioritizes hybrid strategy
```

### Example 3: Multi-Tenant Isolation

```python
# Tenant A
agent_a = RoutingAgent(tenant_id="acme")
agent_a.update_memory("User prefers ColPali model")

# Tenant B
agent_b = RoutingAgent(tenant_id="globex")
agent_b.update_memory("User prefers VideoPrism model")

# Search memories
context_a = agent_a.get_relevant_context("which model to use")
# Returns: "User prefers ColPali model"

context_b = agent_b.get_relevant_context("which model to use")
# Returns: "User prefers VideoPrism model"

# Complete isolation - no cross-tenant visibility
```

## Performance Considerations

### Indexing Latency

- **Write latency**: ~50-100ms per memory
- **Indexing delay**: 2-5 seconds for searchability
- **Deduplication**: Automatic by Mem0 LLM

**Best Practice**: Add memories asynchronously, don't block on indexing

### Search Performance

- **Vector search**: <100ms with HNSW index
- **Hybrid search**: <150ms (vector + BM25)
- **Scaling**: Linear with document count up to 1M memories

**Best Practice**: Use top_k=5 for balance between relevance and speed

### Embedding Costs

- **Ollama (local)**: Free, ~50ms per embedding
- **nomic-embed-text**: 768 dimensions, optimized for retrieval
- **Batch embeddings**: Not supported by Mem0, sequential only

**Best Practice**: Use Ollama for local development, consider cloud embeddings for production scale

### Memory Usage

- **Per memory**: ~3KB (text + embedding + metadata)
- **1000 memories**: ~3MB
- **10000 memories**: ~30MB

**Best Practice**: Clear old memories periodically for inactive agents

## Monitoring

### Memory Statistics

```python
# Get memory stats for dashboard
stats = manager.get_memory_stats("acme", "routing_agent")

print(f"Total memories: {stats['total_memories']}")
print(f"Enabled: {stats['enabled']}")
```

### Health Checks

```python
# Check memory system health
if not manager.health_check():
    logger.error("Memory system unhealthy")
    # Fall back to stateless operation
```

### Phoenix Integration

Memory operations can be traced with Phoenix:

```python
from src.app.telemetry.manager import get_telemetry_manager

telemetry = get_telemetry_manager()

with telemetry.span("memory.search", tenant_id="acme") as span:
    results = agent.get_relevant_context(query)
    span.set_attribute("memory.results_count", len(results))
    span.set_attribute("memory.agent_name", "routing_agent")
```

## Troubleshooting

### Memory Not Initializing

**Symptom**: `initialize_memory()` returns False

**Causes:**
1. Vespa not running
2. Schema not deployed
3. Ollama not running

**Solutions:**
```bash
# Check Vespa
curl http://localhost:8080/ApplicationStatus

# Check Ollama
curl http://localhost:11434/api/tags

# Deploy schema
uv run python scripts/deploy_memory_schema.py
```

### Search Returns No Results

**Symptom**: `get_relevant_context()` returns None

**Causes:**
1. No memories added yet
2. Indexing delay (2-5 seconds)
3. Query not semantically similar

**Solutions:**
```python
# Check if memories exist
all_memories = agent.memory_manager.get_all_memories("acme", "routing_agent")
print(f"Total memories: {len(all_memories)}")

# Wait for indexing
import time
time.sleep(5)

# Try exact text match
results = agent.memory_manager.search_memory(
    query="exact text from memory",
    tenant_id="acme",
    agent_name="routing_agent",
    top_k=10
)
```

### Ollama Model Not Found

**Symptom**: "Model not found: nomic-embed-text"

**Solution:**
```bash
# Pull required models
ollama pull nomic-embed-text
ollama pull llama3.2

# Verify models available
ollama list
```

### Memory Deduplication Issues

**Symptom**: Similar memories not being deduplicated

**Cause**: Mem0 uses LLM for deduplication, may miss subtle duplicates

**Solution**: Manually clear duplicates or adjust content to be more distinct

## Best Practices

### Memory Content

1. **Be specific**: "User prefers hybrid_float_bm25 strategy" not "User likes hybrid"
2. **Include context**: "For ML queries, ColPali performs best" not "ColPali is best"
3. **Factual statements**: Mem0 works best with clear facts, not instructions
4. **Avoid meta-commentary**: "User prefers X" not "Remember that user prefers X"

### Memory Lifecycle

1. **Initialize early**: Call `initialize_memory()` in agent `__init__`
2. **Remember successes**: Track what works for future optimization
3. **Remember failures**: Avoid repeating mistakes
4. **Clear periodically**: Remove old memories for inactive tenants/agents

### Search Strategy

1. **Start with top_k=5**: Balance relevance and speed
2. **Use semantic search**: Better for conceptual matches
3. **Fall back gracefully**: Handle empty context (memory disabled or no results)
4. **Inject context wisely**: Don't overwhelm prompts with too much context

### Multi-Tenancy

1. **Always pass tenant_id**: Never use global/shared memory
2. **Validate tenant access**: Ensure user authorized for tenant
3. **Namespace agents**: Use clear agent names ("routing_agent", not "agent1")
4. **Monitor per-tenant**: Track memory usage by tenant

## Migration Notes

### From Letta

Cogniverse **no longer supports Letta**. The Letta integration has been completely removed:

- ❌ **Removed files**: `letta_memory_manager.py`, `letta_memory_config.py`
- ❌ **Uninstalled packages**: `letta`, `letta-client`
- ❌ **No migration path**: Start fresh with Mem0

**Why Mem0?**
- Simpler API (no agents, just memory storage)
- Persistent Vespa backend
- Multi-tenant support built-in
- Better semantic search
- Lower complexity

## Related Documentation

- [Architecture Overview](architecture.md) - System architecture
- [Agent Orchestration](agent-orchestration.md) - Multi-agent coordination
- [Multi-Tenant System](multi-tenant-system.md) - Tenant isolation
- [Phoenix Integration](phoenix-integration.md) - Telemetry and tracing

**Last Updated**: 2025-10-04
