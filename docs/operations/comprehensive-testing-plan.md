# Comprehensive System Testing Plan

**Last Updated:** 2026-01-25
**Architecture:** UV Workspace with 11 packages in layered architecture
**Purpose**: Bottom-up manual testing and learning guide for Cogniverse
**Approach**: Start with foundational subsystems, validate each layer, build upward
**Goal**: Thorough understanding and validation of the complete system

---

## Testing Philosophy

**Bottom-Up Approach:**
1. Validate foundation before building on it
2. Each layer depends on layers below it
3. Fix issues at lowest level first
4. Document learnings and discoveries

**For Each Subsystem:**
- ✅ **Purpose**: Understand what it does and why
- ✅ **Basic Tests**: Core functionality works
- ✅ **Advanced Tests**: Edge cases and features
- ✅ **Integration Tests**: Works with dependent layers
- ✅ **Learnings**: Document key insights

---

## Layer 1: Storage Foundation (Vespa)

**Purpose**: Vector database for embeddings and metadata storage

### 1.1 Vespa Service Health

**Test Vespa is running:**
```bash
# Check Vespa service status
curl http://localhost:8080/state/v1/health

# Expected: {"status": {"code": "up"}}
```

**Verify Vespa configuration endpoint:**
```bash
curl http://localhost:8080/ApplicationStatus

# Should show application status
```

**Learning Points:**
- Vespa is the foundation - everything else needs it
- Default port: 8080
- State API provides health checks

### 1.2 Schema Validation

**Check deployed schemas:**
```bash
# List all schemas
curl "http://localhost:8080/search/?yql=select+*+from+sources+*+where+true+limit+0"

# Check specific tenant schema exists
curl "http://localhost:8080/search/?yql=select+*+from+video_colpali_smol500_mv_frame_default+where+true+limit+1"
```

**Test schema structure:**
```bash
# Get document count per schema
curl "http://localhost:8080/search/?yql=select+*+from+video_colpali_smol500_mv_frame_default+where+true+limit+0" | jq '.root.fields.totalCount'
```

**Learning Points:**
- Schemas are tenant-isolated: `{profile}_{tenant_id}`
- Multiple profiles can exist per tenant
- Default tenant: "default"

### 1.3 Document Ingestion Test

**Ingest a test video:**
```bash
# Create test video directory
mkdir -p /tmp/test_video

# Download or copy a sample video (MP4)
# For testing: use any short video file

# Run ingestion for single profile
JAX_PLATFORM_NAME=cpu uv run python scripts/run_ingestion.py \
    --video_dir /tmp/test_video \
    --backend vespa \
    --profile video_colpali_smol500_mv_frame \
    --tenant-id test_basic

# Monitor logs
tail -f outputs/logs/ingestion_*.log
```

**Verify ingestion:**
```bash
# Check document count increased
curl "http://localhost:8080/search/?yql=select+*+from+video_colpali_smol500_mv_frame_test_basic+where+true+limit+10" | jq '.root.fields.totalCount'

# Inspect a document
curl "http://localhost:8080/search/?yql=select+*+from+video_colpali_smol500_mv_frame_test_basic+where+true+limit+1" | jq '.root.children[0].fields'
```

**Learning Points:**
- Ingestion creates embeddings and metadata
- Documents have video_id, frame_number, embeddings, metadata
- Each profile has different embedding dimensions

### 1.4 Basic Search Test

**Test vector search:**
```bash
# Simple text query
curl "http://localhost:8080/search/?yql=select+*+from+video_colpali_smol500_mv_frame_test_basic+where+userQuery()&query=test+video&hits=5"

# Check results structure
curl "http://localhost:8080/search/?yql=select+video_id,video_title,frame_number+from+video_colpali_smol500_mv_frame_test_basic+where+userQuery()&query=test&hits=3" | jq '.root.children[] | {video_id: .fields.video_id, title: .fields.video_title}'
```

**Learning Points:**
- Vespa provides vector search via embeddings
- Results ranked by relevance
- Can filter by fields (video_id, timestamp, etc.)

**✅ Layer 1 Complete**: Vespa storage working, schemas deployed, documents ingested and searchable

---

## Layer 2: Telemetry Foundation (Phoenix)

**Purpose**: Observability and tracing for all operations

### 2.1 Phoenix Service Health

**Check Phoenix server:**
```bash
# Verify Phoenix is running
curl http://localhost:6006/healthz

# Expected: 200 OK
```

**Access Phoenix UI:**
```bash
# Open Phoenix dashboard
open http://localhost:6006

# Verify UI loads
```

**Learning Points:**
- Phoenix: observability platform for LLM apps
- Port 6006
- UI provides span visualization

### 2.2 Project Structure

**List Phoenix projects:**
```python
# Run in Python REPL
import phoenix as px

client = px.Client(endpoint='http://localhost:6006')

# List projects
projects = client.list_projects()
for project in projects:
    print(f"Project: {project.name}")
```

**Expected projects:**
- `cogniverse-default-search` (search telemetry)
- `cogniverse-default-routing` (routing decisions)
- `cogniverse-default-orchestration` (multi-agent workflows)

**Learning Points:**
- Projects isolate telemetry by tenant and function
- Naming: `cogniverse-{tenant_id}-{function}`

### 2.3 Span Collection Test

**Generate telemetry by running a search:**
```bash
# Run comprehensive test (generates spans)
JAX_PLATFORM_NAME=cpu uv run python tests/comprehensive_video_query_test_v2.py \
    --profiles video_colpali_smol500_mv_frame \
    --test-multiple-strategies \
    --max-queries 3

# Wait for execution to complete
```

**Query spans from Phoenix:**
```python
import phoenix as px
from datetime import datetime, timedelta

client = px.Client()

# Get search spans
spans_df = client.get_spans_dataframe(
    project_name="cogniverse-default-search",
    start_time=datetime.now() - timedelta(hours=1)
)

print(f"Total spans collected: {len(spans_df)}")
print(f"Columns: {spans_df.columns.tolist()}")

# Inspect a span
if len(spans_df) > 0:
    print("\nSample span:")
    print(spans_df.iloc[0][['name', 'attributes.query', 'latency_ms']])
```

**Learning Points:**
- Spans capture operation traces
- Attributes store metadata (query, results, latency, etc.)
- Can query by time range and project

### 2.4 Span Attributes Validation

**Check span structure:**
```python
# Get span with all attributes
span = spans_df.iloc[0]

# Key attributes for search spans
search_attributes = [
    'attributes.query',
    'attributes.results',
    'attributes.profile',
    'attributes.strategy',
    'attributes.latency_ms',
    'attributes.result_count',
    'attributes.tenant_id'
]

for attr in search_attributes:
    if attr in span.index:
        print(f"{attr}: {span[attr]}")
```

**Learning Points:**
- Search spans include query, results, profile, strategy
- Can filter spans by attributes
- Used for optimization and analytics

**✅ Layer 2 Complete**: Phoenix collecting telemetry, spans queryable, attributes validated

---

## Layer 3: Memory Foundation (Mem0)

**Purpose**: Conversation memory and context for agents

### 3.1 Memory Service Health

**Check Mem0 initialization:**
```python
from cogniverse_core.common.mem0_memory_manager import Mem0MemoryManager

# Initialize manager
manager = Mem0MemoryManager()
manager.initialize()

print(f"Memory manager initialized: {manager.client is not None}")
```

**Learning Points:**
- Mem0 stores agent memories and user preferences
- Initialized on first use
- Isolated by tenant_id + agent_id

### 3.2 Memory Storage Test

**Add a memory:**
```python
# Add memory for routing agent
manager.add_memory(
    messages="User prefers video results for cooking queries",
    user_id="test_user",
    agent_id="routing_agent",
    metadata={"context": "preference_learning", "tenant_id": "default"}
)

print("Memory added successfully")
```

**Retrieve memory:**
```python
# Search for memory
results = manager.search_memory(
    query="cooking preferences",
    user_id="test_user",
    agent_id="routing_agent",
    limit=5
)

for i, result in enumerate(results, 1):
    print(f"\nMemory {i}:")
    print(f"  Content: {result['memory']}")
    print(f"  Score: {result['score']:.3f}")
    print(f"  Metadata: {result.get('metadata', {})}")
```

**Learning Points:**
- Memories are searchable (semantic search)
- Score indicates relevance
- Metadata provides context

### 3.3 Memory Lifecycle

**Get all memories:**
```python
# Get all memories for user
all_memories = manager.get_all_memories(
    user_id="test_user",
    agent_id="routing_agent"
)

print(f"Total memories: {len(all_memories)}")
```

**Delete a memory:**
```python
# Get memory ID
memory_id = all_memories[0]['id'] if all_memories else None

if memory_id:
    manager.delete_memory(memory_id=memory_id)
    print(f"Deleted memory: {memory_id}")
```

**Learning Points:**
- Memories persist across sessions
- Can be deleted individually or bulk cleared
- Memory manager handles CRUD operations

**✅ Layer 3 Complete**: Mem0 storing and retrieving memories, lifecycle operations working

---

## Layer 4: Core Library (cogniverse_core)

**Purpose**: Configuration, telemetry, and memory management abstractions

### 4.1 Configuration Management

**Test ConfigStore:**
```python
from cogniverse_foundation.config.unified_config import BaseConfig
from cogniverse_core.config.unified_config import SystemConfig

# Get system config for tenant
# SystemConfig inherits from BaseConfig (foundation layer)
system_config = SystemConfig(
    tenant_id="default",
    llm_model="gpt-4",
    vespa_url="http://localhost",
    vespa_port=8080
)

print(f"LLM Model: {system_config.llm_model}")
print(f"Search Backend: vespa")
print(f"Vespa URL: {system_config.vespa_url}")
print(f"Vespa Port: {system_config.vespa_port}")
```

**Test config override:**
```python
# Create tenant-specific override
tenant_config = SystemConfig(
    tenant_id="test_tenant",
    llm_model="gpt-3.5-turbo",
    vespa_url="http://localhost",
    vespa_port=8080
)

# Config would be saved via ConfigManager
print(f"Tenant config LLM: {tenant_config.llm_model}")
```

**Learning Points:**
- SystemConfig (core layer) builds on BaseConfig (foundation layer)
- Tenant-specific configurations supported
- Default configs in libs/core/ and libs/foundation/

### 4.2 Telemetry Manager

**Test TelemetryManager:**
```python
from cogniverse_foundation.telemetry.providers.base import TelemetryProvider
from cogniverse_telemetry_phoenix.provider import PhoenixProvider

# Initialize Phoenix provider (from telemetry-phoenix package)
telemetry = PhoenixProvider(
    tenant_id="default",
    project_name="default_project"
)

# Check Phoenix connection
print(f"Phoenix project: {telemetry.project_name}")
print(f"Telemetry enabled: {telemetry.enabled}")
```

**Learning Points:**
- PhoenixProvider (telemetry-phoenix package) implements TelemetryProvider interface (foundation)
- Auto-creates projects per tenant
- Provides span export utilities via foundation layer interfaces

### 4.3 Tenant-Aware Components

**Test TenantAwareMixin:**
```python
from cogniverse_core.common.tenant_aware_mixin import TenantAwareMixin

class TestComponent(TenantAwareMixin):
    def __init__(self, tenant_id: str):
        super().__init__(tenant_id=tenant_id)

    def get_info(self):
        return f"Tenant: {self.tenant_id}, Qualified: {self.qualified_tenant_id}"

# Create instance
component = TestComponent(tenant_id="acme:production")
print(component.get_info())
```

**Learning Points:**
- TenantAwareMixin provides tenant isolation
- qualified_tenant_id includes org prefix
- All agents/components inherit this

**✅ Layer 4 Complete**: Core abstractions working, config/telemetry/memory managers functional

---

## Layer 5: Backend Layer (cogniverse_vespa)

**Purpose**: Search backend implementation using sdk interfaces

### 5.1 Backend Implementation

**Test backend usage:**
```python
from cogniverse_sdk.interfaces.backend import Backend
from cogniverse_vespa.backends.vespa_search_client import VespaSearchClient

# VespaSearchClient implements Backend interface from sdk layer
backend = VespaSearchClient(
    vespa_url="http://localhost",
    vespa_port=8080,
    tenant_id="default"
)

print(f"Backend type: {type(backend).__name__}")
print(f"Vespa URL: {backend.vespa_url}")
print(f"Vespa Port: {backend.vespa_port}")
```

**Learning Points:**
- Backend implementations (vespa package) use interfaces from sdk layer
- Provides tenant-aware search operations
- Handles schema management and routing

### 5.2 Profile Management

**Test profile loading:**
```python
# List available profiles
profiles = backend.get_available_profiles()
print(f"Available profiles: {list(profiles.keys())}")

# Get profile config
profile_config = backend.get_profile_config("video_colpali_smol500_mv_frame")
print(f"\nProfile config:")
print(f"  Model: {profile_config.get('model')}")
print(f"  Embedding dim: {profile_config.get('embedding_dimension')}")
print(f"  Chunk strategy: {profile_config.get('chunk_strategy')}")
```

**Learning Points:**
- Profiles define processing pipelines
- Each profile: model + embedding dim + chunk strategy
- Stored in backend.video_processing_profiles

### 5.3 Tenant Schema Management

**Test schema deployment:**
```python
from cogniverse_vespa.schema.json_schema_parser import JSONSchemaParser
from cogniverse_vespa.backends.vespa_search_client import VespaSearchClient

# Initialize schema parser
parser = JSONSchemaParser()
schema = parser.load_schema_from_json_file("configs/schemas/video_colpali_smol500_mv_frame.json")

# Initialize Vespa client for tenant
client = VespaSearchClient(
    vespa_url="http://localhost",
    vespa_port=8080,
    tenant_id="test_tenant"
)

# Deploy schema
# Schema will be tenant-isolated
print(f"Schema deployed for tenant: test_tenant")
print(f"Schema name includes tenant suffix")
```

**Learning Points:**
- Schema management in vespa package
- Tenant-specific schema isolation
- Auto-creates schemas on first use

### 5.4 Search Execution

**Test search via backend:**
```python
# Simple search
results = backend.search(
    query="test video",
    tenant_id="default",
    profile="video_colpali_smol500_mv_frame",
    limit=5
)

print(f"Search results: {len(results)} found")
for i, result in enumerate(results[:3], 1):
    print(f"\n{i}. {result.get('video_title', 'Unknown')}")
    print(f"   Video ID: {result.get('video_id')}")
    print(f"   Frame: {result.get('frame_number')}")
```

**Learning Points:**
- Backend abstracts Vespa queries
- Returns normalized result format
- Handles tenant-specific schema routing

**✅ Layer 5 Complete**: Backend abstraction working, profile management functional, schema deployment automated

---

## Layer 6: Synthetic Data Layer (cogniverse_synthetic)

**Purpose**: Generate training data for optimizer training (implementation layer)

### 6.1 Service Initialization

**Test SyntheticDataService:**
```python
from cogniverse_synthetic.service import SyntheticDataService

# Initialize service
service = SyntheticDataService(
    vespa_url="http://localhost",
    vespa_port=8080,
    tenant_id="default"
)

print(f"Service initialized")
print(f"Synthetic data generators available")
```

**Learning Points:**
- Part of implementation layer
- Service coordinates data generation for optimization
- Integrates with vespa package for data sampling

### 6.2 Profile Selection

**Test profile selector:**
```python
from cogniverse_synthetic.schemas import SyntheticDataRequest

# Create request
request = SyntheticDataRequest(
    optimizer="modality",
    count=10,
    vespa_sample_size=50,
    strategies=["diverse"],
    max_profiles=2,
    tenant_id="default"
)

# Generate data
response = await service.generate(request)

print(f"Generated {response.count} examples")
print(f"Selected profiles: {response.selected_profiles}")
print(f"Reasoning: {response.profile_selection_reasoning}")
```

**Learning Points:**
- Profile selector chooses best profiles for optimizer
- Rule-based (heuristic) or LLM-based
- Returns reasoning for transparency

### 6.3 Data Generation

**Inspect generated data:**
```python
# Check first example
if response.data:
    example = response.data[0]
    print(f"\nSample example:")
    print(f"  Query: {example.get('query')}")
    print(f"  Modality: {example.get('modality')}")
    print(f"  Agent: {example.get('recommended_agent')}")
```

**Test different optimizers:**
```python
# Test cross_modal optimizer
request_cm = SyntheticDataRequest(
    optimizer="cross_modal",
    count=5,
    vespa_sample_size=20
)

response_cm = await service.generate(request_cm)
print(f"\nCross-modal examples: {response_cm.count}")
print(f"Schema: {response_cm.schema_name}")
```

**Learning Points:**
- Each optimizer has dedicated generator
- Schema defines example structure
- Data sampled from real Vespa content

### 6.4 Optimizer Registry

**Test optimizer registry:**
```python
from cogniverse_synthetic import OPTIMIZER_REGISTRY, get_optimizer_config

# List all optimizers
for name in OPTIMIZER_REGISTRY.keys():
    config = get_optimizer_config(name)
    print(f"\n{name}:")
    print(f"  Description: {config.description}")
    print(f"  Schema: {config.schema_class.__name__}")
    print(f"  Generator: {config.generator_class_name}")
```

**Learning Points:**
- Registry maps optimizer → generator + schema
- Supports: modality, cross_modal, routing, workflow, unified
- Extensible for new optimizers

**✅ Layer 6 Complete**: Synthetic data generation working, profile selection functional, optimizer-specific generators validated

---

## Layer 7: Agent Layer (cogniverse_agents)

**Purpose**: Agent implementations (implementation layer)

### 7.1 Individual Agent Testing

**Test VideoSearchAgent:**
```python
from cogniverse_agents.search.video_search_agent import VideoSearchAgent

# Initialize agent (inherits from core layer base classes)
agent = VideoSearchAgent(
    tenant_id="default",
    profile="video_colpali_smol500_mv_frame"
)

# Run search
result = await agent.search(
    query="machine learning tutorials",
    top_k=10
)

print(f"Agent result:")
print(f"  Videos found: {len(result)}")
print(f"  Profile used: video_colpali_smol500_mv_frame")
```

**Test routing agents:**
```python
from cogniverse_agents.routing.routing_agent import RoutingAgent

# Routing Agent
router = RoutingAgent(tenant_id="default")
decision = await router.route(query="Find videos about Python")

print(f"Routing decision: {decision}")
```

**Learning Points:**
- Agents in implementation layer use core layer base classes
- Integrate with vespa package for backend operations
- Tenant-aware by default

### 7.2 Routing Agent

**Test RoutingAgent:**
```python
from cogniverse_agents.routing_agent import RoutingAgent

# Initialize routing agent
router = RoutingAgent(tenant_id="default")

# Test routing decision
routing_result = await router.route_query(
    query="Show me videos about Python programming"
)

print(f"Routing decision:")
print(f"  Recommended workflow: {routing_result.get('recommended_workflow')}")
print(f"  Primary agent: {routing_result.get('primary_agent')}")
print(f"  Confidence: {routing_result.get('routing_confidence')}")
print(f"  Reasoning: {routing_result.get('reasoning')}")
```

**Test routing with different query types:**
```python
# Video search query
video_routing = await router.route_query("Find cooking videos")

# Report query
report_routing = await router.route_query("Create a detailed analysis of climate change")

# Comparison queries
compare_routing = await router.route_query("Compare Python and Java for web development")

print(f"Video query → {video_routing['primary_agent']}")
print(f"Report query → {report_routing['primary_agent']}")
print(f"Compare query → {compare_routing['primary_agent']}")
```

**Learning Points:**
- RoutingAgent decides which agent to use
- Uses tiered routing: keyword → GLiNER → LLM
- Returns workflow + confidence + reasoning

### 7.3 Modality Optimizer

**Test ModalityOptimizer:**
```python
from cogniverse_agents.routing.modality_optimizer import ModalityOptimizer
from cogniverse_agents.search.multi_modal_reranker import QueryModality

# Initialize optimizer
optimizer = ModalityOptimizer(tenant_id="default")

# Optimize single modality
result = await optimizer.optimize_modality(
    modality=QueryModality.VIDEO,
    lookback_hours=24,
    min_confidence=0.7,
    force_training=True  # Force for testing
)

print(f"Optimization result:")
print(f"  Modality: {result['modality']}")
print(f"  Trained: {result['trained']}")
print(f"  Strategy: {result.get('strategy')}")
print(f"  Examples: {result.get('examples_count')}")
```

**Learning Points:**
- ModalityOptimizer trains per-modality routing
- Uses XGBoost meta-models for training decisions
- Auto-generates synthetic data if needed

### 7.4 Cross-Modal Optimizer

**Test CrossModalOptimizer:**
```python
from cogniverse_agents.routing.cross_modal_optimizer import CrossModalOptimizer

# Initialize optimizer
cm_optimizer = CrossModalOptimizer(tenant_id="default")

# Run optimization
result = await cm_optimizer.optimize()

print(f"Cross-modal optimization:")
print(f"  Status: {result.get('status')}")
print(f"  Fusion patterns learned: {result.get('patterns_learned')}")
```

**Learning Points:**
- CrossModalOptimizer learns fusion decisions
- When to combine multiple modalities
- When single modality is sufficient

### 7.5 Advanced Routing Optimizer (GRPO)

**Test AdvancedRoutingOptimizer:**
```python
from cogniverse_agents.routing.advanced_optimizer import AdvancedRoutingOptimizer

# Initialize GRPO optimizer
grpo = AdvancedRoutingOptimizer(tenant_id="default")

# Record routing experience
reward = await grpo.record_routing_experience(
    query="Show me machine learning videos",
    entities=[{"text": "machine learning", "label": "topic"}],
    relationships=[],
    enhanced_query="Show me machine learning videos",
    chosen_agent="video_search_agent",
    routing_confidence=0.9,
    search_quality=0.85,
    agent_success=True,
    user_satisfaction=0.9,
    processing_time=1.5
)

print(f"Reward computed: {reward:.3f}")

# Get optimization status
status = grpo.get_optimization_status()
print(f"Total experiences: {status['total_experiences']}")
print(f"Avg reward: {status['metrics']['avg_reward']:.3f}")
```

**Learning Points:**
- GRPO uses experience replay for learning
- Computes reward from multiple signals
- Auto-selects DSPy optimizer (Bootstrap/SIMBA/MIPRO/GEPA)

**✅ Layer 7 Complete**: Individual agents working, routing decisions functional, optimizers trainable

---

## Layer 8: Runtime Layer (cogniverse_runtime)

**Purpose**: FastAPI server (application layer) exposing all functionality via REST API

### 8.1 Runtime Service Startup

**Start runtime server:**
```bash
# Start server (application layer)
JAX_PLATFORM_NAME=cpu uv run python -m cogniverse_runtime.server.main

# Verify startup in logs
# Expected: "Uvicorn running on http://0.0.0.0:8000"
```

**Check health endpoint:**
```bash
curl http://localhost:8000/health

# Expected: {"status": "healthy", "services": {...}}
```

**Learning Points:**
- Runtime is in application layer
- Integrates agents, vespa, synthetic, and evaluation packages
- Port 8000 (default)
- Health check validates all services

### 8.2 Search Endpoints

**Test search API:**
```bash
# Text search
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning tutorials",
    "tenant_id": "default",
    "profile": "video_colpali_smol500_mv_frame",
    "limit": 5
  }'

# Should return JSON with results
```

**Test multi-profile search:**
```bash
curl -X POST http://localhost:8000/search/multi-profile \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python programming",
    "tenant_id": "default",
    "profiles": [
      "video_colpali_smol500_mv_frame",
      "video_videoprism_base_mv_chunk_30s"
    ],
    "limit": 10
  }'
```

**Learning Points:**
- /search/text for single profile
- /search/multi-profile for multiple profiles
- Results aggregated and ranked

### 8.3 Routing Endpoints

**Test routing API:**
```bash
curl -X POST http://localhost:8000/routing/route \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me cooking videos",
    "tenant_id": "default"
  }'

# Returns: recommended_workflow, primary_agent, confidence, reasoning
```

**Learning Points:**
- /routing/route for query routing
- Returns workflow recommendation
- Includes confidence and reasoning

### 8.4 Synthetic Data Endpoints

**Test synthetic data generation:**
```bash
curl -X POST http://localhost:8000/synthetic/generate \
  -H "Content-Type: application/json" \
  -d '{
    "optimizer": "modality",
    "count": 10,
    "vespa_sample_size": 50,
    "strategies": ["diverse"],
    "max_profiles": 2,
    "tenant_id": "default"
  }'

# Returns: generated examples with metadata
```

**List available optimizers:**
```bash
curl http://localhost:8000/synthetic/optimizers

# Returns: {"modality": "Per-modality routing...", ...}
```

**Learning Points:**
- /synthetic/generate creates training data
- /synthetic/optimizers lists available types
- Integrated with backend auto-discovery

### 8.5 Admin Endpoints

**Create organization:**
```bash
curl -X POST http://localhost:8000/admin/organizations \
  -H "Content-Type: application/json" \
  -d '{
    "org_id": "acme",
    "org_name": "Acme Corporation",
    "contact_email": "admin@acme.com"
  }'
```

**Create tenant:**
```bash
curl -X POST http://localhost:8000/admin/tenants \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "acme:production",
    "tenant_name": "Acme Production",
    "org_id": "acme"
  }'
```

**Learning Points:**
- Admin endpoints for tenant management
- Organization → Tenant hierarchy
- Tenant format: {org}:{name}

**✅ Layer 8 Complete**: Runtime API functional, all endpoints responding, multi-tenant support working

---

## Layer 9: Dashboard Layer (cogniverse_dashboard)

**Purpose**: Streamlit UI (application layer) for monitoring, configuration, and optimization

### 9.1 Dashboard Startup

**Start Phoenix dashboard:**
```bash
# Dashboard from application layer package
uv run streamlit run libs/dashboard/cogniverse_dashboard/app.py --server.port 8501

# Open browser
open http://localhost:8501
```

**Learning Points:**
- Dashboard in application layer (10th package)
- Integrates with evaluation, telemetry-phoenix, and core packages
- Tabs for analytics, evaluation, config, memory, and optimization

### 9.2 Analytics Tab

**Test Analytics:**
1. Navigate to "📊 Analytics" tab
2. Select time range (1h, 24h, 7d)
3. Select tenant: "default"
4. Click "🔄 Refresh Metrics"

**Verify metrics displayed:**
- Total queries
- Avg latency
- Success rate
- Query distribution chart
- Latency trends

**Learning Points:**
- Analytics pulls from Phoenix spans
- Real-time metrics
- Filterable by tenant and time range

### 9.3 Evaluation Tab

**Test Evaluation:**
1. Navigate to "📈 Evaluation" tab
2. Select experiments to compare
3. View metrics comparison

**Verify evaluation data:**
- NDCG@10 scores
- Precision/Recall
- Profile comparison
- Query-level breakdown

**Learning Points:**
- Evaluation uses Phoenix experiments
- Compares different profiles/strategies
- Visualizes performance differences

### 9.4 Config Management Tab

**Test Config Management:**
1. Navigate to "⚙️ Config Management" tab
2. Select tenant: "default"
3. View System Config
4. Make a change (e.g., routing threshold)
5. Click "💾 Save"
6. Verify change persisted

**Test import/export:**
1. Navigate to "💾 Import/Export" sub-tab
2. Click "📥 Download JSON"
3. Modify JSON
4. Upload modified JSON
5. Verify imported

**Learning Points:**
- Full CRUD for all config types
- Import/Export for backup
- Version history tracked

### 9.5 Memory Management Tab

**Test Memory Management:**
1. Navigate to "🧠 Memory Management" tab
2. Enter tenant: "default", agent: "routing_agent"
3. Click "📈 Refresh Stats"

**Add a memory:**
1. Navigate to "📝 Add Memory" sub-tab
2. Enter memory content
3. Add metadata
4. Click "💾 Add Memory"

**Search memories:**
1. Navigate to "🔍 Search Memories" sub-tab
2. Enter search query
3. View results with scores

**Learning Points:**
- UI for Mem0 operations
- Semantic search interface
- Metadata management

### 9.6 Optimization Tab

**Test Synthetic Data Generation:**
1. Navigate to "🔧 Optimization Framework" tab
2. Go to "🔬 Synthetic Data" sub-tab
3. Select optimizer: "modality"
4. Set count: 10
5. Click "🚀 Generate Synthetic Data"
6. View generated examples

**Test Module Optimization:**
1. Go to "🎯 Module Optimization" sub-tab
2. Select module: "modality"
3. Set max iterations: 100
4. Check "Use Synthetic Data"
5. Click "🚀 Submit Routing Optimization Workflow"
6. Verify Argo workflow submitted

**Learning Points:**
- UI integrates with Argo Workflows
- Synthetic data preview
- Workflow submission from UI

**✅ Layer 9 Complete**: Dashboard functional (application layer), all tabs working, UI integrations validated

---

## Layer 10: End-to-End Integration Tests

**All 11 Packages Working Together**

**Purpose**: Validate complete workflows across all layers

### 10.1 Complete Search Workflow

**Test full search pipeline:**
```bash
# 1. Ingest test video
JAX_PLATFORM_NAME=cpu uv run python scripts/run_ingestion.py \
    --video_dir /tmp/test_video \
    --backend vespa \
    --profile video_colpali_smol500_mv_frame \
    --tenant-id e2e_test

# 2. Run comprehensive search test
JAX_PLATFORM_NAME=cpu uv run python tests/comprehensive_video_query_test_v2.py \
    --profiles video_colpali_smol500_mv_frame \
    --test-multiple-strategies \
    --max-queries 5

# 3. Verify Phoenix captured spans
```

**Validate in Phoenix:**
```python
import phoenix as px

client = px.Client()
spans_df = client.get_spans_dataframe(
    project_name="cogniverse-e2e_test-search"
)

print(f"Search spans collected: {len(spans_df)}")
print(f"Avg latency: {spans_df['latency_ms'].mean():.2f}ms")
```

**Learning Points:**
- Complete flow: Ingest → Search → Telemetry
- All layers working together
- End-to-end latency tracking

### 10.2 Routing + Search Workflow

**Test routing to search:**
```bash
# Use routing API to get agent recommendation
curl -X POST http://localhost:8000/routing/route \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find videos about machine learning",
    "tenant_id": "default"
  }' > /tmp/routing_result.json

# Extract recommended agent
cat /tmp/routing_result.json | jq '.primary_agent'

# If video_search_agent, run search
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning",
    "tenant_id": "default",
    "profile": "video_colpali_smol500_mv_frame",
    "limit": 5
  }'
```

**Learning Points:**
- Routing determines agent
- Agent executes search
- Results returned to user

### 10.3 Optimization Workflow

**Test complete optimization cycle:**
```bash
# 1. Generate synthetic data
curl -X POST http://localhost:8000/synthetic/generate \
  -H "Content-Type: application/json" \
  -d '{
    "optimizer": "modality",
    "count": 50,
    "vespa_sample_size": 100,
    "tenant_id": "default"
  }' > /tmp/synthetic_data.json

# 2. Run module optimization
JAX_PLATFORM_NAME=cpu uv run python scripts/run_module_optimization.py \
  --module modality \
  --tenant-id default \
  --use-synthetic-data \
  --force-training \
  --output /tmp/optimization_results.json

# 3. Check results
cat /tmp/optimization_results.json | jq '.results'
```

**Learning Points:**
- Synthetic data → Training → Optimized model
- Complete optimization pipeline
- Results include improvement metrics

### 10.4 Multi-Tenant Workflow

**Test tenant isolation:**
```bash
# 1. Create organization
curl -X POST http://localhost:8000/admin/organizations \
  -H "Content-Type: application/json" \
  -d '{"org_id": "test_org", "org_name": "Test Org"}'

# 2. Create tenant
curl -X POST http://localhost:8000/admin/tenants \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "test_org:dev", "org_id": "test_org"}'

# 3. Ingest video for tenant
JAX_PLATFORM_NAME=cpu uv run python scripts/run_ingestion.py \
    --video_dir /tmp/test_video \
    --backend vespa \
    --profile video_colpali_smol500_mv_frame \
    --tenant-id test_org:dev

# 4. Search with tenant
curl -X POST http://localhost:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test",
    "tenant_id": "test_org:dev",
    "profile": "video_colpali_smol500_mv_frame"
  }'

# 5. Verify data isolation
# Should not see test_org:dev data when searching with "default" tenant
```

**Learning Points:**
- Complete tenant isolation
- Separate schemas per tenant
- Config/telemetry/memory isolated

### 10.5 Argo Workflow Integration

**Test scheduled optimization:**
```bash
# 1. Deploy workflows
kubectl apply -f workflows/scheduled-optimization.yaml

# 2. Check CronWorkflow created
kubectl get cronworkflow weekly-optimization -n cogniverse

# 3. Trigger manually
argo submit --from cronwf/weekly-optimization -n cogniverse

# 4. Monitor workflow
argo list -n cogniverse
argo logs <workflow-name> -n cogniverse --follow

# 5. Check results
argo get <workflow-name> -n cogniverse -o json | \
  jq '.status.outputs.parameters'
```

**Learning Points:**
- Argo Workflows for batch jobs
- Scheduled optimization automation
- Kubernetes integration

**✅ Layer 10 Complete**: End-to-end workflows validated, multi-tenant isolation verified, all integrations working

---

## Final Validation Checklist

### Infrastructure
- [ ] Vespa running and healthy
- [ ] Phoenix running and collecting spans
- [ ] Mem0 initialized and storing memories
- [ ] Runtime API responding on all endpoints

### Data Layer
- [ ] Videos ingested across multiple profiles
- [ ] Documents searchable via Vespa
- [ ] Embeddings generated correctly
- [ ] Search results ranked properly

### Telemetry
- [ ] Phoenix projects created per tenant
- [ ] Spans collected for search/routing/orchestration
- [ ] Span attributes populated correctly
- [ ] Analytics dashboard showing metrics

### Configuration
- [ ] System config loaded
- [ ] Tenant-specific overrides working
- [ ] Backend auto-discovery functional
- [ ] Profile management operational

### Agents
- [ ] Individual agents (video, summarizer, report) working
- [ ] Routing agent making correct decisions
- [ ] Agent memory integration functional
- [ ] Multi-agent orchestration working

### Optimization
- [ ] Synthetic data generation producing quality examples
- [ ] Modality optimizer training successfully
- [ ] Cross-modal optimizer learning fusion patterns
- [ ] GRPO optimizer improving routing decisions
- [ ] Argo workflows submitting and executing

### Multi-Tenant
- [ ] Tenant isolation validated
- [ ] Separate schemas per tenant
- [ ] Config/telemetry/memory isolated
- [ ] Cross-tenant data leakage prevented

### UI/UX
- [ ] Dashboard loading all tabs
- [ ] Analytics showing real data
- [ ] Config management CRUD working
- [ ] Optimization workflows submittable from UI

---

## Troubleshooting Guide

### Common Issues

**Vespa Connection Failed:**
- Check: `curl http://localhost:8080/state/v1/health`
- Fix: Start Vespa service
- Verify: Port 8080 accessible

**Phoenix Not Collecting Spans:**
- Check: Phoenix server running on port 6006
- Fix: Set PHOENIX_ENDPOINT env var
- Verify: Spans appear in Phoenix UI

**Memory Manager Initialization Failed:**
- Check: Mem0 dependencies installed
- Fix: `uv pip install mem0`
- Verify: Can create Mem0MemoryManager()

**Backend Config Not Found:**
- Check: config.json exists in standard locations
- Fix: Create config.json with backend section
- Verify: get_backend() succeeds

**Optimization Training Failed:**
- Check: Sufficient training data (>50 examples)
- Fix: Use --force-training or generate synthetic data
- Verify: /tmp/optimization_results.json created

**Argo Workflow Submission Failed:**
- Check: kubectl or argo CLI configured
- Fix: Install argo CLI, configure kubectl
- Verify: `argo version` succeeds

---

## Learning Outcomes

After completing this testing plan, you will understand:

1. **Architecture**: How all layers interact bottom-up
2. **Data Flow**: From ingestion → storage → search → telemetry
3. **Tenant Isolation**: How multi-tenancy works across layers
4. **Configuration**: Auto-discovery and override mechanisms
5. **Optimization**: Complete optimization lifecycle
6. **Deployment**: Argo Workflows for production automation
7. **Observability**: Phoenix telemetry and analytics
8. **APIs**: REST endpoints for all functionality
9. **UI Integration**: Dashboard for all operations
10. **Troubleshooting**: Common issues and solutions

---

## Next Steps

After completing all layers:

1. **Run full test suite:** `JAX_PLATFORM_NAME=cpu timeout 7200 uv run pytest`
2. **Deploy to staging:** Test with real workloads
3. **Performance testing:** Benchmark search latency, optimization throughput
4. **Security audit:** Verify tenant isolation, auth mechanisms
5. **Documentation review:** Update any gaps discovered during testing
6. **Production deployment:** Deploy with monitoring and alerting

---

**Testing Tips:**
- Document issues as you find them
- Take notes on architecture insights
- Save successful commands for future reference
- Test edge cases (empty queries, large documents, etc.)
- Validate error handling (network failures, invalid inputs)

Good luck with your comprehensive testing! 🚀
