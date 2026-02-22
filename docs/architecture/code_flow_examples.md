# Code Flow Examples

Real-world code execution flows through the Cogniverse multi-agent system.

## 1. Video Ingestion Flow

### Deploy Tenant Schema
```bash
# Deploy schema for a new tenant using deploy_all_schemas.py
# Note: This script deploys schemas from configs/schemas directory
uv run python scripts/deploy_all_schemas.py
```

**Code Flow:**
```python
# 1. Schema Manager creates tenant-specific schema
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

schema_manager = VespaSchemaManager(
    backend_endpoint="http://localhost",
    backend_port=8080
)
# Note: VespaSchemaManager provides read_sd_file() and parse_sd_schema() methods
# for parsing .sd schema files. Tenant-specific schema naming is handled at
# the application level when deploying schemas.
schema_name = "video_colpali_smol500_mv_frame_customer_a"

# 2. Deploy to Vespa
from vespa.package import ApplicationPackage, Field, Schema
from vespa.application import Vespa

# Initialize Vespa application client (application-specific setup)
vespa_app = Vespa(url="http://localhost:8080")  # Example initialization

app_package = ApplicationPackage(name=schema_name)
app_package.schema.add_fields(
    Field("embedding", "tensor<float>(patch{}, v[768])"),
    Field("binary_embedding", "tensor<int8>(patch{}, v[96])"),
    Field("text", "string", indexing=["index", "summary"])
)

# 3. Add ranking profiles
# Note: create_ranking_profile() is application-specific helper function
for strategy in ["hybrid_float_bm25", "float_float", "phased"]:
    app_package.add_rank_profile(
        create_ranking_profile(strategy)  # Application-defined helper
    )

# Deploy the application package
vespa_app.deploy(app_package)
```

### Process Videos
```bash
# Ingest videos for tenant
uv run python scripts/run_ingestion.py \
    --video_dir /path/to/videos \
    --tenant-id customer_a \
    --profile video_colpali_smol500_mv_frame
```

**Code Flow:**
```python
# 1. Pipeline initialization
from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline, PipelineConfig
from cogniverse_foundation.config.utils import create_default_config_manager

config_manager = create_default_config_manager()
pipeline_config = PipelineConfig.from_config(
    tenant_id="customer_a",
    config_manager=config_manager
)

pipeline = VideoIngestionPipeline(
    tenant_id="customer_a",
    config=pipeline_config,
    config_manager=config_manager
)

# 2. Process videos using the unified pipeline
# The pipeline handles all steps internally: frame extraction, embedding generation, storage
# video_paths is a list of Path objects to video files
for video_path in video_paths:
    # Process video through the complete pipeline (async)
    result = await pipeline.process_video_async(video_path)
```

## 2. Multi-Agent Search Flow

### User Query Request
```bash
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning tutorial", "tenant_id": "customer_a"}'
```

**Code Flow:**
```python
# 1. Receive and parse request
from cogniverse_agents.video_agent_refactored import VideoSearchAgent
from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_foundation.telemetry.config import TelemetryConfig
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

config_manager = create_default_config_manager()
schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

# 2. Extract tenant from header
# request is the incoming HTTP request object (e.g., FastAPI Request)
tenant_id = request.headers.get("X-Tenant-ID", "default")

# 3. Initialize routing agent with typed dependencies
deps = RoutingDeps(
    tenant_id=tenant_id,
    telemetry_config=TelemetryConfig(),
    model_name="smollm3:3b",
    base_url="http://localhost:11434/v1"
)
routing_agent = RoutingAgent(deps=deps, port=8001)

# 4. Route query with DSPy optimization (async)
routing_decision = await routing_agent.route_query(
    query="machine learning tutorial"
)
# Decision: Route to video agent for tutorial content

# 5. Video Search Agent executes search (synchronous)
video_agent = VideoSearchAgent(
    config_manager=config_manager,
    schema_loader=schema_loader
)
results = video_agent.search(
    query="machine learning tutorial",
    profile="video_colpali_smol500_mv_frame",
    tenant_id=tenant_id,
    top_k=10
)

# 6. Backend automatically handles tenant-scoped schema
# Schema name is constructed from profile and tenant_id
# Search service routes to correct tenant schema internally
```

## 3. DSPy Routing Optimization Flow

### Advanced Multi-Stage Optimizer

The routing system uses DSPy's advanced optimization techniques (GEPA, MIPROv2, SIMBA, BootstrapFewShot) to continuously improve routing decisions based on experience.

**Location**: `libs/agents/cogniverse_agents/routing/advanced_optimizer.py`

```python
# 1. Initialize optimizer for tenant
from cogniverse_agents.routing.advanced_optimizer import (
    AdvancedRoutingOptimizer,
    AdvancedOptimizerConfig
)

config = AdvancedOptimizerConfig(
    optimizer_strategy="adaptive",  # Auto-selects based on data size
    experience_replay_size=1000,
    gepa_threshold=200,             # Use GEPA when 200+ examples
    mipro_threshold=100,            # Use MIPROv2 when 100+ examples
    simba_threshold=50,             # Use SIMBA when 50+ examples
    bootstrap_threshold=20,         # Use Bootstrap when 20+ examples
    min_experiences_for_training=50
)

optimizer = AdvancedRoutingOptimizer(
    tenant_id="customer_a",
    config=config
)
```

### Recording Routing Experiences

```python
# 2. After each routing decision, record the outcome
from cogniverse_agents.routing.advanced_optimizer import RoutingExperience

reward = await optimizer.record_routing_experience(
    query="machine learning tutorial",
    entities=[{"text": "machine learning", "type": "TOPIC"}],
    relationships=[],
    enhanced_query="machine learning tutorial videos",
    chosen_agent="video_search",
    routing_confidence=0.92,

    # Outcome metrics (collected after agent execution)
    search_quality=0.85,        # Quality of search results (0-1)
    agent_success=True,         # Did agent complete successfully
    user_satisfaction=0.9,      # Explicit user feedback (optional)
    processing_time=1.2         # Seconds
)

# Reward computed from weighted combination:
# reward = (search_quality * 0.4) + (agent_success * 0.3) + (user_satisfaction * 0.3) - (time_penalty)
# Result: 0.855
```

### Experience Replay and Learning

```python
# 3. Experience stored in replay buffer (simple list)
# libs/agents/cogniverse_agents/routing/advanced_optimizer.py:652-655

# Add to experience replay buffer
self.experience_replay.append(experience)
if len(self.experience_replay) > self.config.experience_replay_size:
    self.experience_replay.pop(0)  # FIFO queue

# 4. Automatic optimization triggers when conditions met
def _should_trigger_optimization(self) -> bool:
    if len(self.experiences) < self.config.min_experiences_for_training:
        return False
    # Trigger every N experiences
    if len(self.experiences) % self.config.update_frequency == 0:
        return True
    # Trigger if performance is declining
    recent_rewards = [exp.reward for exp in self.experiences[-10:]]
    if len(recent_rewards) >= 10:
        if np.mean(recent_rewards) < self.metrics.avg_reward - 0.1:
            return True
    return False
```

### Multi-Stage DSPy Optimization

```python
# 5. When optimization triggers, adaptive algorithm selection
# libs/agents/cogniverse_agents/routing/advanced_optimizer.py:342-363

from dspy.teleprompt import GEPA, MIPROv2, SIMBA, BootstrapFewShot

# Optimizer initialized with DSPy's actual optimizers
self.gepa_optimizer = GEPA(
    metric=routing_accuracy_metric,
    auto="light",
    reflection_lm=current_lm  # LLM for reflective prompt evolution
)
self.mipro_optimizer = MIPROv2(metric=routing_accuracy_metric)
self.simba_optimizer = SIMBA(metric=routing_accuracy_metric)
self.bootstrap_optimizer = BootstrapFewShot(metric=routing_accuracy_metric)

# Adaptive selection based on dataset size
optimization_stages = [
    ("bootstrap", bootstrap_optimizer, 20),   # 20+ examples
    ("simba", simba_optimizer, 50),           # 50+ examples
    ("mipro", mipro_optimizer, 100),          # 100+ examples
    ("gepa", gepa_optimizer, 200),            # 200+ examples (most advanced)
]

# Select optimizer based on current experience count
dataset_size = len(self.experience_replay)
if dataset_size >= 200:
    selected_optimizer = gepa_optimizer      # Reflective prompt evolution
    optimizer_name = "gepa"
elif dataset_size >= 100:
    selected_optimizer = mipro_optimizer     # Metric-aware optimization
    optimizer_name = "mipro"
elif dataset_size >= 50:
    selected_optimizer = simba_optimizer     # Similarity-based memory
    optimizer_name = "simba"
else:
    selected_optimizer = bootstrap_optimizer # Few-shot learning
    optimizer_name = "bootstrap"
```

### Optimization Execution

```python
# 6. Run optimization on routing module
import dspy

# Prepare training data from experiences
trainset = []
for exp in self.experience_replay[-config.batch_size:]:
    trainset.append(
        dspy.Example(
            query=exp.query,
            enhanced_query=exp.enhanced_query,
            agent_type=exp.chosen_agent
        ).with_inputs("query", "enhanced_query")
    )

# Compile routing module with selected optimizer
optimized_module = selected_optimizer.compile(
    student=routing_module,
    trainset=trainset
)

# 7. Update routing module with optimized version
self.routing_module = optimized_module

# 8. Track optimization metrics
self.metrics = OptimizationMetrics(
    total_experiences=len(self.experiences),
    avg_reward=np.mean([e.reward for e in self.experiences]),
    successful_routes=sum(1 for e in self.experiences if e.agent_success),
    failed_routes=sum(1 for e in self.experiences if not e.agent_success),
    confidence_accuracy=self._compute_confidence_accuracy(),
    last_updated=datetime.now()
)
```

### Complete Workflow

```python
# Full routing optimization workflow
from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps
from cogniverse_agents.video_agent_refactored import VideoSearchAgent
from cogniverse_agents.routing.advanced_optimizer import AdvancedRoutingOptimizer
from cogniverse_foundation.telemetry.config import TelemetryConfig
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

# 1. Initialize routing agent with optimizer
deps = RoutingDeps(
    tenant_id="customer_a",
    telemetry_config=TelemetryConfig(),
    model_name="smollm3:3b",
    base_url="http://localhost:11434/v1"
)
agent = RoutingAgent(deps=deps, port=8001)
optimizer = AdvancedRoutingOptimizer(tenant_id="customer_a")

# 2. Process user query (async)
decision = await agent.route_query(query="cooking videos")

# 3. Execute search with chosen agent (synchronous)
config_manager = create_default_config_manager()
schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
video_agent = VideoSearchAgent(
    config_manager=config_manager,
    schema_loader=schema_loader
)
results = video_agent.search(query="cooking videos", profile="video_colpali_smol500_mv_frame", tenant_id="customer_a", top_k=10)

# 4. Record experience with outcome
# compute_search_quality() is application-defined evaluation function
reward = await optimizer.record_routing_experience(
    query="cooking videos",
    entities=decision.entities,
    relationships=[],
    enhanced_query=decision.enhanced_query,
    chosen_agent=decision.recommended_agent,
    routing_confidence=decision.confidence,
    search_quality=compute_search_quality(results),  # Application-defined
    agent_success=True,
    processing_time=1.5
)

# 5. Optimizer automatically triggers training when thresholds met
# After 200+ experiences, GEPA reflective optimization runs
# Routing module continuously improves based on feedback
```

### Key Architecture Differences from Documentation

**What Actually Exists**:

- ✅ `AdvancedRoutingOptimizer` orchestrates everything
- ✅ DSPy's GEPA/MIPROv2/SIMBA/Bootstrap (not custom implementations)
- ✅ `RoutingExperience` dataclass for tracking
- ✅ Simple list-based experience replay (not separate buffer class)
- ✅ Adaptive algorithm selection based on dataset size
- ✅ Continuous learning from routing outcomes

**What Doesn't Exist**:

- ❌ Standalone `GEPAOptimizer` class
- ❌ Separate `ExperienceBuffer` class
- ❌ Custom GEPA implementation

## 4. Memory-Augmented Search Flow

### Search with Context
```python
# 1. Initialize video search agent with memory
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_agents.video_agent_refactored import VideoSearchAgent
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

# Initialize agent
config_manager = create_default_config_manager()
schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
tenant_id = "customer_a"
agent = VideoSearchAgent(
    config_manager=config_manager,
    schema_loader=schema_loader
)

# Initialize memory manager separately (memory is typically initialized via MemoryAwareMixin)
memory_manager = Mem0MemoryManager(tenant_id=tenant_id)

# 2. Retrieve user context
user_memories = memory_manager.search_memory(
    query="machine learning",
    tenant_id="customer_a",
    agent_name="SearchAgent",
    top_k=5
)

# 3. Use context to enhance query (application-level logic)
# Query enhancement is done by the application based on retrieved memories
context_items = [m.get("content", "") for m in user_memories]
augmented_query = f"machine learning tutorial {' '.join(context_items)}"
# Result: "machine learning tutorial python sklearn beginner"

# 4. Store interaction in memory
memory_manager.add_memory(
    content="User searched for ML tutorial, showed interest in sklearn",
    tenant_id="customer_a",
    agent_name="SearchAgent",
    metadata={
        "query": "machine learning tutorial",
        "selected_result": "sklearn_basics.mp4"
    }
)
```

## 5. Experiment Flow

### Run A/B Test
```python
# 1. Initialize evaluation provider
from cogniverse_telemetry_phoenix.evaluation.evaluation_provider import PhoenixEvaluationProvider

provider = PhoenixEvaluationProvider()
provider.initialize({
    "tenant_id": "customer_a",
    "http_endpoint": "http://localhost:6006",
    "grpc_endpoint": "http://localhost:4317",
    "project_name": "evaluation"
})

# 2. Create experiment
# create_experiment returns a dict-like experiment object
experiment = provider.create_experiment(
    name="ml_queries_v1",
    description="Compare search strategies",
    metadata={
        "strategies": ["hybrid_float_bm25", "float_float"],
        "tenant_id": "customer_a"
    }
)

# 3. Run evaluations for each strategy
# queries is a list of query strings to evaluate
for query in queries:
    for strategy in ["hybrid_float_bm25", "float_float"]:
        # Execute search (application-defined function)
        results = search(query, strategy)

        # Evaluate quality (application-defined function)
        score = evaluate_quality(results)

        # Log evaluation result
        provider.log_evaluation(
            experiment_id=experiment["id"],
            evaluation_name="search_quality",
            score=score,
            label="pass" if score > 0.7 else "fail",
            explanation=f"Search quality for {strategy}",
            metadata={"strategy": strategy, "query": query}
        )

# 4. Compare experiment results through Phoenix UI at http://localhost:6006
# Or retrieve programmatically through provider.telemetry
```

## 6. Error Recovery Flow

### Graceful Degradation
```python
import logging
from cogniverse_agents.video_agent_refactored import VideoSearchAgent
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from pathlib import Path

logger = logging.getLogger(__name__)

# 1. Primary search fails
# video_agent, tenant_id, config_manager, schema_loader are defined in calling context
try:
    results = video_agent.search(
        query="tutorial",
        top_k=10
    )
except Exception as e:
    logger.error(f"Search failed: {e}")

    # 2. Try alternate agent with different profile
    try:
        fallback_agent = VideoSearchAgent(
            config_manager=config_manager,
            schema_loader=schema_loader
        )
        results = fallback_agent.search(query="tutorial", profile="frame_based_colpali", tenant_id=tenant_id, top_k=10)
    except Exception as fallback_error:
        # 3. Return cached results if available (requires cache manager setup)
        # Note: In production, cache_mgr would be initialized at application startup
        # This example shows the basic pattern for cache retrieval
        logger.error(f"Fallback failed: {fallback_error}")

        # If cache manager is available (initialized separately)
        # from cogniverse_core.common.cache import CacheManager
        # cache_key = f"search_results_{hash(query)}"
        # cached = await cache_mgr.get(cache_key)  # get() is async
        # if cached:
        #     return cached

        # 4. Return error with helpful message
        logger.error("No fallback options available")
        raise RuntimeError(
            "Search temporarily unavailable. Please try again later."
        )
```

## Performance Monitoring

### Request Tracing
```python
# Complete request trace using OpenTelemetry
from opentelemetry import trace
from cogniverse_foundation.telemetry.registry import TelemetryRegistry

# Get telemetry provider for this tenant
# tenant_id is defined in calling context
registry = TelemetryRegistry()
telemetry = registry.get_telemetry_provider(
    name="phoenix",
    tenant_id=tenant_id,
    config={
        "project_name": "search",
        "http_endpoint": "http://localhost:6006",
        "grpc_endpoint": "http://localhost:4317"
    }
)

tracer = trace.get_tracer(__name__)

# Create trace with nested spans
# router, agent, query, rerank are defined in calling context
with tracer.start_as_current_span("search_request") as trace_span:
    trace_span.set_attribute("tenant_id", tenant_id)

    # Routing: 10ms
    with tracer.start_as_current_span("routing"):
        decision = await router.route_query(query)

    # Search: 200ms
    with tracer.start_as_current_span("search"):
        results = agent.search(query, top_k=10)

    # Reranking: 50ms (rerank is application-defined function)
    with tracer.start_as_current_span("rerank"):
        results = rerank(results)

    # Total: 260ms
    trace_span.set_attribute("latency_ms", 260)
    trace_span.set_attribute("cache_hit", True)
```

---

**Package Architecture Note**: Code examples use Cogniverse's layered architecture:

- **Foundation Layer**: cogniverse-sdk, cogniverse-foundation (config, telemetry)
- **Core Layer**: cogniverse-core (base agents, memory), cogniverse-evaluation (experiments), cogniverse-telemetry-phoenix (Phoenix integration)
- **Implementation Layer**: cogniverse-agents (routing, search), cogniverse-vespa (backends), cogniverse-synthetic (data generation)
- **Application Layer**: cogniverse-runtime (ingestion, API), cogniverse-dashboard (UI)

All code examples follow the correct import paths for the layered structure located in `libs/`.
