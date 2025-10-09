# Agents Module - Deep Dive Study Guide

**Module:** `src/app/agents/`
**Purpose:** Core agent implementations for multi-tenant multi-agent RAG system
**Last Updated:** 2025-10-09

---

## Table of Contents
1. [Module Overview](#module-overview)
2. [Multi-Tenant Requirements](#multi-tenant-requirements)
3. [Architecture Diagrams](#architecture-diagrams)
4. [Core Classes](#core-classes)
5. [Data Flow](#data-flow)
6. [Usage Examples](#usage-examples)
7. [Production Considerations](#production-considerations)

---

## Module Overview

The Agents Module contains the intelligent components of the system. Each agent is specialized for a specific task, operates within tenant boundaries, and communicates via the A2A (Agent-to-Agent) protocol.

### Key Agents
1. **MultiAgentOrchestrator** - Coordinates multi-agent workflows
2. **RoutingAgent** - Routes queries with DSPy + relationship extraction
3. **VideoSearchAgent** - Multi-modal video search (text/video/image)
4. **SummarizerAgent** - Intelligent content summarization with VLM
5. **DetailedReportAgent** - Comprehensive analysis and reporting
6. **TextAnalysisAgent** - Text processing and analysis
7. **MemoryAwareMixin** - Mem0 conversation memory integration

### Design Principles
- **Multi-Tenant Isolation**: All agents require tenant_id for complete data isolation
- **Separation of Concerns**: Each agent has a single, well-defined responsibility
- **A2A Protocol**: Standard message format for inter-agent communication
- **DSPy Integration**: Declarative LLM programming for optimization
- **Memory-Aware**: All agents can leverage conversation memory
- **Production-Ready**: Health checks, graceful degradation, comprehensive logging

---

## Multi-Tenant Requirements

### Tenant ID Requirement

**All agents REQUIRE tenant_id** - there are no default values. This ensures complete data isolation between tenants.

### Tenant ID Format

```
{organization}:{tenant_name}

Examples:
- acme:production
- acme:staging
- initech:production
```

### Agent Factory Pattern

All agents implement a factory function for tenant-aware instantiation:

```python
# Get tenant-specific agent instance
from src.app.agents.routing_agent import get_agent

agent = get_agent(tenant_id="acme:production")
```

**Benefits:**
- Per-tenant agent instances are cached
- Automatic tenant validation on initialization
- Consistent initialization across all agents

### Validation

Agents raise `ValueError` if tenant_id is empty or None:

```python
# Valid initialization
agent = RoutingAgent(tenant_id="acme:production")

# Invalid - raises ValueError
agent = RoutingAgent(tenant_id="")        # ValueError: tenant_id is required
agent = RoutingAgent(tenant_id=None)      # ValueError: tenant_id is required
agent = RoutingAgent()                    # ValueError: tenant_id is required
```

### Tenant Isolation Guarantees

- **Data Isolation**: Each tenant's data stored in dedicated Vespa schemas
- **Memory Isolation**: Conversation memory scoped to tenant_id
- **Configuration Isolation**: Agent configs persisted per tenant
- **Search Isolation**: All search operations limited to tenant schemas
- **No Cross-Tenant Access**: Zero possibility of accessing other tenant data

---

## Architecture Diagrams

### Multi-Agent Workflow

```mermaid
graph TB
    UserQuery[User Query] --> Orchestrator[MultiAgentOrchestrator]

    Orchestrator --> Analysis[1. Analyze Query Complexity<br/>2. Build Dependency Graph<br/>3. Select Required Agents<br/>4. Plan Execution Order]

    Analysis --> RoutingAgent[Routing Agent<br/>Port 8001]
    Analysis --> VideoAgent[Video Search Agent<br/>Port 8002]
    Analysis --> SummarizerAgent[Summarizer Agent<br/>Port 8004]

    RoutingAgent --> A2A[A2A Task Messages<br/>JSON + Binary Data]
    VideoAgent --> A2A
    SummarizerAgent --> A2A

    A2A --> Response[Synthesized Response]

    style UserQuery fill:#e1f5ff
    style Orchestrator fill:#fff4e1
    style Analysis fill:#f5f5f5
    style RoutingAgent fill:#ffe1e1
    style VideoAgent fill:#ffe1e1
    style SummarizerAgent fill:#ffe1e1
    style A2A fill:#f0f0f0
    style Response fill:#e1ffe1
```

### Routing Agent Decision Flow

```mermaid
graph TB
    UserQuery[User Query] --> ProcessQuery[RoutingAgent.process_query]

    ProcessQuery --> Phase1[Phase 1: Entity Extraction<br/>• GLiNER for named entities<br/>• Confidence scoring]
    Phase1 --> Phase2[Phase 2: Relationship Extraction<br/>• Pattern matching<br/>• Semantic analysis]
    Phase2 --> Phase3[Phase 3: Query Enhancement<br/>• Context enrichment<br/>• Relationship-aware expansion]
    Phase3 --> Phase4[Phase 4: Agent Selection<br/>• Video vs Text vs Both<br/>• GRPO optimization]

    Phase4 --> Decision[RoutingDecision<br/>• recommended_agent<br/>• enhanced_query<br/>• entities + relationships<br/>• confidence_score]

    style UserQuery fill:#e1f5ff
    style ProcessQuery fill:#fff4e1
    style Phase1 fill:#f5f5f5
    style Phase2 fill:#f5f5f5
    style Phase3 fill:#f5f5f5
    style Phase4 fill:#f5f5f5
    style Decision fill:#e1ffe1
```

### Video Search Agent Flow

```mermaid
graph TB
    Input[Input: Query<br/>text/video/image] --> Step1[1. Memory Context Retrieval<br/>if memory enabled]
    Step1 --> Step2[2. Query Encoding<br/>• Text: ColPali/VideoPrism<br/>• Video: Frame extraction<br/>• Image: Direct encoding]
    Step2 --> Step3[3. Vespa Vector Search<br/>• Hybrid ranking BM25 + neural<br/>• Top-K results]
    Step3 --> Step4[4. Relationship Boost<br/>if enhanced<br/>• Entity matching<br/>• Relationship scoring<br/>• Result reranking]
    Step4 --> Step5[5. Memory Update<br/>• Store success/failure<br/>• Learn from patterns]
    Step5 --> Output[Return: Ranked search results]

    style Input fill:#e1f5ff
    style Step1 fill:#f5f5f5
    style Step2 fill:#f5f5f5
    style Step3 fill:#fff4e1
    style Step4 fill:#f5f5f5
    style Step5 fill:#f5f5f5
    style Output fill:#e1ffe1
```

---

## Core Classes

### 1. RoutingAgent

**File:** `src/app/agents/routing_agent.py`
**Lines:** 50-600
**Purpose:** Intelligent query routing with relationship extraction

#### Class Hierarchy
```python
class RoutingAgent(DSPyRoutingMixin, A2AEndpointsMixin, HealthCheckMixin):
    # Inherits DSPy integration, A2A protocol, and health checking
```

#### Key Methods

**`process_query(query: str, context: dict) -> RoutingDecision`**
- **Purpose:** Main entry point for routing decisions
- **Input:** User query string + optional context
- **Output:** RoutingDecision with agent selection + enhanced query
- **Process:**
  1. Extract entities using GLiNER/LangExtract
  2. Identify relationships between entities
  3. Enhance query with relationship context
  4. Select optimal agent using DSPy module
  5. Apply GRPO optimization (if enabled)

**Example:**
```python
# Initialize with tenant_id (REQUIRED)
routing_agent = RoutingAgent(tenant_id="acme:production")

decision = await routing_agent.process_query(
    query="Show me videos where Einstein discusses quantum physics",
    context={"user_id": "user_123"}
)

# RoutingDecision:
# - recommended_agent: "video_search"
# - enhanced_query: "Videos featuring Albert Einstein discussing quantum mechanics and wave-particle duality"
# - entities: [{"text": "Einstein", "type": "PERSON"}, {"text": "quantum physics", "type": "TOPIC"}]
# - relationships: [{"subject": "Einstein", "relation": "discusses", "object": "quantum physics"}]
# - confidence: 0.89
```

**`extract_entities(query: str) -> List[Dict]`**
- **Purpose:** Named entity recognition
- **Strategy:** Tiered approach
  - Tier 1: GLiNER (fast, entity-based)
  - Tier 2: LLM (relationship-aware)
  - Tier 3: LangExtract (structured output)

**`enhance_query(query: str, entities: List, relationships: List) -> str`**
- **Purpose:** Enrich query with relationship context
- **Technique:** Relationship-aware query expansion

#### Configuration
```python
{
    "dspy_enabled": true,
    "grpo_enabled": true,
    "confidence_threshold": 0.7,
    "memory_enabled": true,
    "entity_extraction": "gliner",  # or "llm", "langextract"
    "relationship_extraction": true
}
```

---

### 2. VideoSearchAgent

**File:** `src/app/agents/video_search_agent.py`
**Lines:** 236-1273
**Purpose:** Multi-modal video search with memory integration

#### Key Features
- **Multi-Modal Support**: Text, video files, image files
- **Memory-Aware**: Learns from search patterns
- **Relationship-Enhanced**: Boosts results based on entity/relationship matches
- **Production-Ready**: Health checks, error handling, metrics

#### Key Methods

**`search_by_text(query: str, top_k: int) -> List[Dict]`**
- **Purpose:** Text-to-video search
- **Encoding:** ColPali or VideoPrism text encoder
- **Ranking:** Hybrid (BM25 + neural embeddings)

**Example:**
```python
# Initialize with tenant_id (REQUIRED)
agent = VideoSearchAgent(
    tenant_id="acme:production",
    vespa_url="http://localhost",
    vespa_port=8080,
    profile="video_colpali_smol500_mv_frame"
)

# Or use factory pattern
from src.app.agents.video_search_agent import get_agent
agent = get_agent(tenant_id="acme:production")

results = agent.search_by_text(
    query="Machine learning tutorial",
    top_k=10,
    ranking="hybrid_binary_bm25_no_description"
)

# Results are isolated to acme:production tenant only
# [
#   {
#     "video_id": "vid_123",
#     "title": "ML Basics Tutorial",
#     "score": 0.87,
#     "keyframes": [...],
#     "metadata": {...}
#   },
#   ...
# ]
```

**`search_by_video(video_data: bytes, filename: str, top_k: int) -> List[Dict]`**
- **Purpose:** Video-to-video similarity search
- **Processing:**
  1. Save video to temp file
  2. Extract frames (1 FPS or every 30 frames)
  3. Encode frames using query encoder
  4. Search Vespa with frame embeddings
  5. Clean up temp files

**`search_with_routing_decision(decision: RoutingDecision) -> Dict`**
- **Purpose:** Enhanced search using routing context
- **Enhancement:**
  - Uses enhanced query from routing
  - Applies entity matching boost
  - Scores relationship relevance
  - Reranks results

#### Memory Integration
```python
# Agent initialized with tenant_id
agent = VideoSearchAgent(tenant_id="acme:production")

# Memory is automatically initialized for the tenant
# Memory scope: acme:production only

# Search automatically:
# 1. Retrieves relevant context from past searches (tenant-isolated)
# 2. Stores successful patterns (for this tenant)
# 3. Learns from failures (tenant-specific)
```

---

### 3. SummarizerAgent

**File:** `src/app/agents/summarizer_agent.py`
**Lines:** 107-400
**Purpose:** Intelligent summarization with VLM and thinking phase

#### DSPy Signature
```python
class SummaryGenerationSignature(dspy.Signature):
    """Generate structured summaries with key insights."""

    content = dspy.InputField(desc="Search results content to summarize")
    query = dspy.InputField(desc="Original user query")
    summary_type = dspy.InputField(desc="Type: brief, comprehensive, detailed")

    summary = dspy.OutputField(desc="Generated summary text")
    key_points = dspy.OutputField(desc="List of key points (comma-separated)")
    confidence_score = dspy.OutputField(desc="Confidence in summary (0.0-1.0)")
```

#### Key Methods

**`summarize(request: SummaryRequest) -> SummaryResult`**
- **Purpose:** Multi-phase intelligent summarization
- **Phases:**
  1. **Thinking Phase**: Analyze content, identify themes
  2. **Visual Analysis**: Extract visual insights (if enabled)
  3. **DSPy Generation**: Generate structured summary
  4. **Confidence Assessment**: Evaluate summary quality

**Example:**
```python
# Initialize with tenant_id (REQUIRED)
summarizer = SummarizerAgent(tenant_id="acme:production")

request = SummaryRequest(
    query="AI developments",
    search_results=[...],  # Search results from VideoSearchAgent (same tenant)
    summary_type="comprehensive",
    include_visual_analysis=True,
    max_results_to_analyze=10
)

result = await summarizer.summarize(request)

# SummaryResult:
# - summary: "Comprehensive overview..."
# - key_points: ["Point 1", "Point 2", ...]
# - visual_insights: ["Visual element 1", ...]
# - confidence_score: 0.92
# - thinking_phase: ThinkingPhase(...)
```

**`summarize_with_routing_context(enhanced_request) -> SummaryResult`**
- **Purpose:** Relationship-aware summarization
- **Enhancement:** Includes entity/relationship analysis in summary

---

### 4. DetailedReportAgent

**File:** `src/app/agents/detailed_report_agent.py`
**Lines:** 99-450
**Purpose:** Comprehensive analysis and reporting

#### Report Structure
```python
@dataclass
class ReportResult:
    executive_summary: str            # High-level overview
    detailed_findings: List[Dict]     # Granular analysis
    visual_analysis: List[Dict]       # Visual content insights
    technical_details: List[Dict]     # Technical specifications
    recommendations: List[str]        # Actionable recommendations
    confidence_assessment: Dict       # Quality metrics
    thinking_phase: ThinkingPhase     # Agent's reasoning process
```

#### Key Methods

**`generate_report(request: ReportRequest) -> ReportResult`**
- **Purpose:** Generate comprehensive detailed report
- **Phases:**
  1. **Thinking Phase**: Content analysis, pattern identification
  2. **Visual Analysis**: Deep dive into visual elements
  3. **Executive Summary**: High-level synthesis
  4. **Detailed Findings**: Granular analysis
  5. **Technical Details**: Specifications and metadata
  6. **Recommendations**: Actionable next steps

**Example:**
```python
# Initialize with tenant_id (REQUIRED)
report_agent = DetailedReportAgent(tenant_id="acme:production")

request = ReportRequest(
    query="AI safety research trends",
    search_results=[...],  # Results from acme:production tenant
    report_type="comprehensive",
    include_visual_analysis=True,
    include_technical_details=True,
    include_recommendations=True
)

report = await report_agent.generate_report(request)

# Uses DSPy for structured report generation
# Includes VLM analysis for visual content
# Provides thinking phase for transparency
# All analysis scoped to acme:production tenant
```

---

### 5. MemoryAwareMixin

**File:** `src/app/agents/memory_aware_mixin.py`
**Lines:** 16-327
**Purpose:** Standard memory interface for all agents

#### Core Concept
All agents inherit from `MemoryAwareMixin` to access conversation memory powered by Mem0 + Vespa.

#### Key Methods

**`initialize_memory(agent_name, tenant_id, vespa_port) -> bool`**
- **Purpose:** Setup memory for agent
- **Backend:** Mem0 with Vespa vector store

**`get_relevant_context(query: str, top_k: int) -> str`**
- **Purpose:** Retrieve relevant memories for query
- **Process:**
  1. Embed query using Mem0
  2. Search vector store
  3. Return formatted context

**`update_memory(content: str, metadata: dict) -> bool`**
- **Purpose:** Store new memory
- **LLM Processing:** Mem0 uses LLM to condense and structure memories

**`remember_success(query, result, metadata) -> bool`**
- **Purpose:** Store successful interactions for learning

**`remember_failure(query, error, metadata) -> bool`**
- **Purpose:** Store failures to avoid repeating mistakes

#### Usage Pattern
```python
class MyAgent(MemoryAwareMixin, BaseAgent):
    def __init__(self, tenant_id: str):  # REQUIRED parameter
        if not tenant_id:
            raise ValueError("tenant_id is required")

        super().__init__()
        self.tenant_id = tenant_id

        # Initialize memory (automatically tenant-scoped)
        self.initialize_memory(
            agent_name="my_agent",
            tenant_id=tenant_id  # Mem0 memory isolated to this tenant
        )

    async def process(self, query):
        # Get relevant memories (tenant-isolated)
        context = self.get_relevant_context(query, top_k=5)

        # Use context in processing
        if context:
            enhanced_prompt = f"Context: {context}\n\nQuery: {query}"

        # Process...
        result = await self._process(enhanced_prompt)

        # Store success (tenant-isolated)
        if result:
            self.remember_success(query, result)
        else:
            self.remember_failure(query, "Processing failed")

        return result

# Usage
agent = MyAgent(tenant_id="acme:production")
```

---

## Data Flow

### End-to-End Query Processing

```
1. USER SUBMITS QUERY
   ↓
2. MultiAgentOrchestrator receives query
   ↓
3. Orchestrator → RoutingAgent.process_query()
   │
   ├─→ Extract entities (GLiNER)
   ├─→ Extract relationships
   ├─→ Enhance query
   └─→ Return RoutingDecision
   ↓
4. Based on decision:
   │
   ├─→ Video Search: Orchestrator → VideoSearchAgent
   │   ├─→ Memory: Get context
   │   ├─→ Encode query (ColPali/VideoPrism)
   │   ├─→ Search Vespa
   │   ├─→ Apply relationship boost
   │   ├─→ Memory: Store pattern
   │   └─→ Return search results
   │
   ├─→ Text Search: Orchestrator → TextAgent
   │   └─→ [Similar flow]
   │
   └─→ Both: Execute parallel search
       └─→ Aggregate results
   ↓
5. If summarization requested:
   Orchestrator → SummarizerAgent.summarize()
   ├─→ Thinking phase (analyze results)
   ├─→ Visual analysis (VLM)
   ├─→ DSPy summary generation
   └─→ Return SummaryResult
   ↓
6. Orchestrator synthesizes final response
   ↓
7. Return to USER
```

---

## Usage Examples

### Example 1: Simple Video Search (Multi-Tenant)

```python
from src.app.agents.video_search_agent import get_agent

# Get tenant-specific agent instance
agent = get_agent(tenant_id="acme:production")

# Or direct initialization
from src.app.agents.video_search_agent import VideoSearchAgent
agent = VideoSearchAgent(
    tenant_id="acme:production",  # REQUIRED
    vespa_url="http://localhost",
    vespa_port=8080,
    profile="video_colpali_smol500_mv_frame"
)

# Search (results isolated to acme:production)
results = agent.search_by_text(
    query="Python tutorial for beginners",
    top_k=5
)

# Process results (only from acme:production tenant)
for result in results:
    print(f"Video: {result['title']}")
    print(f"Score: {result['score']}")
    print(f"ID: {result['video_id']}")
```

### Example 2: Routing with Enhancement (Multi-Tenant)

```python
from src.app.agents.routing_agent import get_agent

# Get tenant-specific routing agent
routing_agent = get_agent(tenant_id="acme:production")

# Or direct initialization
from src.app.agents.routing_agent import RoutingAgent
routing_agent = RoutingAgent(tenant_id="acme:production")  # REQUIRED

# Process complex query (tenant-isolated)
decision = await routing_agent.process_query(
    query="Show me videos where Marie Curie discusses radioactivity and its applications",
    context={"user_preference": "educational"}
)

# Access routing details
print(f"Agent: {decision.recommended_agent}")
print(f"Enhanced query: {decision.enhanced_query}")
print(f"Entities: {decision.extracted_entities}")
print(f"Relationships: {decision.extracted_relationships}")
print(f"Confidence: {decision.confidence}")
```

### Example 3: Multi-Agent Workflow (Multi-Tenant)

```python
from src.app.agents.multi_agent_orchestrator import MultiAgentOrchestrator
from src.app.agents.routing_agent import get_agent as get_routing_agent
from src.app.agents.video_search_agent import get_agent as get_video_agent
from src.app.agents.summarizer_agent import get_agent as get_summarizer_agent

# All agents must use the same tenant_id
tenant_id = "acme:production"

# Initialize orchestrator with tenant-specific agents
orchestrator = MultiAgentOrchestrator(
    routing_agent=get_routing_agent(tenant_id=tenant_id),
    video_agent=get_video_agent(tenant_id=tenant_id),
    summarizer=get_summarizer_agent(tenant_id=tenant_id)
)

# Process complex query (all operations isolated to acme:production)
result = await orchestrator.process_query(
    query="Summarize recent AI research videos",
    options={
        "include_summary": True,
        "max_results": 20
    }
)

# Result includes (all from acme:production tenant):
# - Routing decision
# - Search results (tenant-isolated)
# - Summary
# - Confidence scores
```

---

## Production Considerations

### Multi-Tenancy

**Tenant Isolation**:
- All agents require `tenant_id` parameter (no defaults)
- Agent instances cached per tenant using factory pattern
- Memory (Mem0) automatically scoped to tenant
- Search operations limited to tenant-specific Vespa schemas
- Zero possibility of cross-tenant data access

**Factory Pattern Benefits**:
```python
# Recommended: Use factory functions
from src.app.agents.routing_agent import get_agent
agent = get_agent(tenant_id="acme:production")

# Agent instances cached per tenant
agent1 = get_agent("acme:production")   # Creates new instance
agent2 = get_agent("acme:production")   # Returns cached instance
agent3 = get_agent("initech:production")  # Creates new instance (different tenant)
```

**Validation**:
- `tenant_id` validated on initialization (org:tenant format)
- `ValueError` raised if tenant_id is empty or None
- All tenant operations logged with tenant_id for auditing

### Performance
- **Memory Initialization**: One-time setup per agent per tenant, reused across requests
- **Query Encoding**: Cached for repeated queries within tenant
- **Parallel Execution**: Use MultiAgentOrchestrator for concurrent agent calls
- **Batch Processing**: VideoSearchAgent supports batch encoding per tenant

### Scalability
- **Stateless Agents**: Can be horizontally scaled
- **Per-Tenant Caching**: Agent instances cached per tenant_id
- **Shared Memory Backend**: Mem0 backend shared across instances (tenant-isolated)
- **Vector Cache**: Per-tenant caching in Vespa schemas

### Monitoring
- **Health Checks**: All agents expose `/health` endpoint
- **Telemetry**: Integrated with Phoenix (see Telemetry Module)
- **Metrics**: Request count, latency, success rate
- **Logging**: Structured logging with correlation IDs

### Error Handling
- **Graceful Degradation**: Fallback strategies at every level
- **Circuit Breaker**: Prevents cascading failures
- **Retry Logic**: Exponential backoff for transient failures
- **Error Memory**: Stores failures to avoid repetition

### Configuration
```yaml
agents:
  routing_agent:
    port: 8001
    dspy_enabled: true
    grpo_enabled: true
    confidence_threshold: 0.7

  video_agent:
    port: 8002
    profile: "video_colpali_smol500_mv_frame"
    memory_enabled: true
    cache_ttl: 300

  summarizer_agent:
    port: 8004
    max_summary_length: 500
    thinking_enabled: true
    visual_analysis_enabled: true
```

---

## Testing

### Unit Tests
- `tests/agents/unit/test_routing_agent.py`
- `tests/agents/unit/test_video_search_agent.py`
- `tests/agents/unit/test_summarizer_agent.py`
- `tests/agents/unit/test_detailed_report_agent.py`

### Integration Tests
- `tests/agents/integration/test_routing_agent_integration.py`
- `tests/agents/integration/test_video_search_agent_integration.py`

### End-to-End Tests
- `tests/agents/e2e/test_real_multi_agent_integration.py`

---

## Next Steps

For related documentation, see:
- **Multi-Tenant Management** (`../operations/multi-tenant-management.md`) - Complete tenant lifecycle guide
- **Backends Module** (`backends.md`) - TenantSchemaManager and tenant-aware search
- **Routing Module** (`02_ROUTING_MODULE.md`) - Routing strategies and optimization
- **Memory Module** (`07_MEMORY_MODULE.md`) - Mem0 integration details
- **Telemetry Module** (`05_TELEMETRY_MODULE.md`) - Observability and metrics

---

**Study Tips:**
1. Start with Multi-Tenant Requirements to understand tenant_id usage
2. Explore MemoryAwareMixin to understand the base pattern
3. Review VideoSearchAgent for the most complex multi-modal example
4. Check RoutingAgent for DSPy integration patterns
5. Review integration tests for real-world multi-tenant usage examples
