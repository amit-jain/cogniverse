# Code Flow Examples

Real-world code execution flows through the Cogniverse multi-agent system.

## 1. Video Ingestion Flow

### Deploy Tenant Schema
```bash
# Deploy schema for a new tenant
uv run python scripts/deploy_tenant_schema.py \
    --tenant customer_a \
    --profile video_colpali_smol500_mv_frame
```

**Code Flow:**
```python
# 1. Schema Manager creates tenant-specific schema
from cogniverse_vespa.tenant_schema_manager import TenantSchemaManager

schema_manager = TenantSchemaManager()
schema_name = schema_manager.deploy_tenant_schema(
    tenant_id="customer_a",
    profile="video_colpali_smol500_mv_frame"
)
# Result: "video_colpali_smol500_mv_frame_customer_a"

# 2. Deploy to Vespa
from vespa.package import ApplicationPackage

app_package = ApplicationPackage(name=schema_name)
app_package.schema.add_fields(
    Field("embedding", "tensor<float>(patch{}, v[768])"),
    Field("binary_embedding", "tensor<int8>(patch{}, v[96])"),
    Field("text", "string", indexing=["index", "summary"])
)

# 3. Add ranking profiles
for strategy in ["hybrid_float_bm25", "float_float", "phased"]:
    app_package.add_rank_profile(
        create_ranking_profile(strategy)
    )

vespa_app.deploy(app_package)
```

### Process Videos
```bash
# Ingest videos for tenant
uv run python scripts/run_ingestion.py \
    --video_dir /path/to/videos \
    --tenant customer_a \
    --profile video_colpali_smol500_mv_frame
```

**Code Flow:**
```python
# 1. Pipeline initialization with Phoenix telemetry
from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline
from cogniverse_foundation.telemetry import TelemetryProvider
from cogniverse_foundation.config import SystemConfig

config = SystemConfig(tenant_id="customer_a")
telemetry = TelemetryProvider(config)
pipeline = VideoIngestionPipeline(
    profile="video_colpali_smol500_mv_frame",
    tenant_id="customer_a",
    backend="vespa"
)

# 2. Process each video with tracing
for video_path in video_paths:
    with telemetry.span("ingestion.process_video", "customer_a") as span:
        span.set_attribute("video.path", str(video_path))

        # Extract frames
        frames = await pipeline.extract_frames(video_path)

        # Generate embeddings
        embeddings = await pipeline.generate_embeddings(frames)

        # Store in Vespa
        await pipeline.store_in_vespa(embeddings)
```

## 2. Multi-Agent Search Flow

### User Query Request
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-Tenant-ID: customer_a" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning tutorial"}'
```

**Code Flow:**
```python
# 1. Composing Agent receives request
from cogniverse_agents.composing_agent import ComposingAgent
from cogniverse_agents.video_search_agent import VideoSearchAgent
from cogniverse_agents.text_analysis_agent import TextAnalysisAgent

composing_agent = ComposingAgent()

# 2. Extract tenant from header
tenant_id = request.headers.get("X-Tenant-ID", "default")

# 3. Route query with DSPy optimization
with telemetry.span("agent.route", tenant_id) as span:
    routing_decision = composing_agent.route_query(
        query="machine learning tutorial",
        tenant_id=tenant_id
    )
    # Decision: Route to video agent for tutorial content

# 4. Orchestrate agents via A2A protocol
from cogniverse_agents.tools.a2a_utils import create_text_message

# Create search message for video agent
search_msg = create_text_message(
    text="machine learning tutorial",
    role="user"
)

# 5. Video Search Agent executes search
from cogniverse_agents.video_search_agent import VideoSearchAgent

video_agent = VideoSearchAgent(
    config=config,
    profile="video_colpali_smol500_mv_frame"
)
results = video_agent.search_by_text(
    query="machine learning tutorial",
    top_k=10,
    ranking="hybrid_binary_bm25_no_description"
)

# 6. Backend automatically handles tenant-scoped schema
# Schema name: "video_colpali_smol500_mv_frame_customer_a"
# VespaBackend routes to correct tenant schema internally
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
def _should_trigger_optimization(self):
    return (
        len(self.experiences) >= self.config.min_experiences_for_training
        and len(self.experiences) % self.config.update_frequency == 0
        and len(self.experience_replay) >= self.config.batch_size
    )
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
from cogniverse_agents.routing_agent import RoutingAgent
from cogniverse_agents.routing.advanced_optimizer import AdvancedRoutingOptimizer

# 1. Initialize routing agent with optimizer
agent = RoutingAgent(tenant_id="customer_a")
optimizer = AdvancedRoutingOptimizer(tenant_id="customer_a")

# 2. Process user query
decision = agent.route_query(query="cooking videos")

# 3. Execute search with chosen agent
results = await video_agent.search_by_text(query="cooking videos")

# 4. Record experience with outcome
reward = await optimizer.record_routing_experience(
    query="cooking videos",
    entities=decision.entities,
    relationships=[],
    enhanced_query=decision.enhanced_query,
    chosen_agent=decision.recommended_agent,
    routing_confidence=decision.confidence,
    search_quality=compute_search_quality(results),
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
# 1. Initialize memory-aware agent
from cogniverse_core.memory import Mem0MemoryManager
from cogniverse_core.agents import MemoryAwareMixin
from cogniverse_agents.search import VideoSearchAgent

class MemoryAwareVideoAgent(VideoSearchAgent, MemoryAwareMixin):
    def __init__(self, config):
        super().__init__(config)
        self.memory = Mem0MemoryManager(config)

# 2. Retrieve user context
user_memories = await agent.memory.search(
    query="machine learning",
    user_id="user_123",
    tenant_id="customer_a",
    limit=5
)

# 3. Augment query with context
augmented_query = agent.augment_with_memory(
    original_query="machine learning tutorial",
    memories=user_memories
)
# Result: "machine learning tutorial python sklearn beginner"

# 4. Store interaction in memory
await agent.memory.add(
    content="User searched for ML tutorial, showed interest in sklearn",
    user_id="user_123",
    tenant_id="customer_a",
    metadata={
        "query": "machine learning tutorial",
        "selected_result": "sklearn_basics.mp4"
    }
)
```

## 5. Phoenix Experiment Flow

### Run A/B Test
```python
# 1. Create experiment
from cogniverse_telemetry_phoenix.evaluation import PhoenixExperimentProvider
import phoenix as px

provider = PhoenixExperimentProvider()
client = px.Client()

# 2. Setup dataset
dataset = client.upload_dataset(
    dataset_name="ml_queries_v1",
    dataframe=queries_df
)

# 3. Run experiment with multiple strategies
experiment_id = provider.run_experiment(
    dataset_name="ml_queries_v1",
    tenant_id="customer_a",
    strategies=["hybrid_float_bm25", "float_float"],
    evaluators=[quality_scorer, visual_judge]
)

# 4. Track results in Phoenix
with telemetry.span("experiment.evaluate", tenant_id) as span:
    for query in dataset:
        for strategy in strategies:
            # Execute search
            results = await search(query, strategy)

            # Evaluate quality
            scores = evaluate(query, results)

            # Record in Phoenix
            client.log_evaluation(
                experiment_id=experiment_id,
                query=query,
                strategy=strategy,
                scores=scores
            )

# 5. Compare results
comparison = client.compare_experiments(
    baseline_id=exp_1,
    treatment_id=exp_2
)
print(f"Improvement: {comparison['delta_quality']:.2%}")
```

## 6. Error Recovery Flow

### Graceful Degradation
```python
# 1. Primary search fails
try:
    results = await video_agent.search(
        query="tutorial",
        strategy="hybrid_float_bm25"
    )
except VespaConnectionError:
    logger.error("Vespa unavailable")

    # 2. Try simpler strategy
    try:
        results = await video_agent.search(
            query="tutorial",
            strategy="bm25_only"  # Text-only, no embeddings
        )
    except Exception as e:
        # 3. Return cached results if available
        cache_key = f"search_results/{hash(query)}/latest"
        cached = await cache.get(cache_key, tenant_id)

        if cached:
            logger.warning("Serving stale cached results")
            return cached
        else:
            # 4. Return error with helpful message
            raise ServiceUnavailableError(
                "Search temporarily unavailable. Please try again."
            )
```

## Performance Monitoring

### Request Tracing
```python
# Complete request trace in Phoenix
with telemetry.trace("search_request", tenant_id) as trace:
    # Routing: 10ms
    with telemetry.span("routing"):
        tier = router.route(query)

    # Search: 200ms
    with telemetry.span("search"):
        results = await search(query)

    # Reranking: 50ms
    with telemetry.span("rerank"):
        results = rerank(results)

    # Total: 260ms
    trace.set_attribute("latency_ms", 260)
    trace.set_attribute("cache_hit", True)
```

---

**Last Updated:** 2026-01-25
**Status**: Production Ready

**Package Architecture Note**: Code examples use Cogniverse's 11-package layered architecture:
- **Foundation Layer**: cogniverse-sdk, cogniverse-foundation (config, telemetry)
- **Core Layer**: cogniverse-core (base agents, memory), cogniverse-evaluation (experiments), cogniverse-telemetry-phoenix (Phoenix integration)
- **Implementation Layer**: cogniverse-agents (routing, search), cogniverse-vespa (backends), cogniverse-synthetic (data generation)
- **Application Layer**: cogniverse-runtime (ingestion, API), cogniverse-dashboard (UI)

All code examples follow the correct import paths for the 11-package structure located in `/home/user/cogniverse/libs/*/`.