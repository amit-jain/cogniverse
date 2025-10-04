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
from src.backends.vespa.schema_manager import SchemaManager

schema_manager = SchemaManager()
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
from src.app.ingestion.pipeline import VideoIngestionPipeline
from src.telemetry.multi_tenant_manager import MultiTenantTelemetryManager

telemetry = MultiTenantTelemetryManager()
pipeline = VideoIngestionPipeline(
    profile="video_colpali_smol500_mv_frame",
    tenant_id="customer_a",
    telemetry=telemetry
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
from src.app.agents.composing_agent import ComposingAgent
from src.app.agents.enhanced_video_search_agent import EnhancedVideoSearchAgent
from src.app.agents.text_analysis_agent import TextAnalysisAgent

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
from src.app.agents.a2a_protocol import A2AMessage, MessageType

# Create search message for video agent
search_msg = A2AMessage(
    type=MessageType.REQUEST,
    source="composing_agent",
    target="video_search_agent",
    payload={
        "query": "machine learning tutorial",
        "strategy": "hybrid_float_bm25",
        "tenant_id": tenant_id
    }
)

# 5. Video Search Agent executes search
video_agent = EnhancedVideoSearchAgent()
results = await video_agent.search(
    query=search_msg.payload["query"],
    tenant_id=tenant_id,
    strategy=search_msg.payload["strategy"]
)

# 6. Query Vespa with tenant schema
from src.backends.vespa.query_builder import VespaQueryBuilder

query_builder = VespaQueryBuilder()
vespa_query = query_builder.build_hybrid_query(
    text="machine learning tutorial",
    embedding=query_embedding,
    schema=f"video_colpali_smol500_mv_frame_{tenant_id}"
)

vespa_results = await vespa_app.query(vespa_query)
```

## 3. DSPy Optimization Flow

### GEPA Optimizer Training
```python
# 1. Collect routing experiences
from src.app.routing.gepa_optimizer import GEPAOptimizer
from src.app.routing.experience_buffer import ExperienceBuffer

buffer = ExperienceBuffer(capacity=10000)
optimizer = GEPAOptimizer(buffer=buffer)

# 2. Record routing decision
experience = {
    "query": "machine learning tutorial",
    "features": extract_query_features(query),
    "decision": RoutingTier.BALANCED,
    "reward": 0.85,  # Based on user feedback
    "tenant_id": "customer_a"
}
buffer.add(experience)

# 3. Periodic optimization (every 5 minutes)
if buffer.size() > 1000:
    with telemetry.span("optimization.gepa", tenant_id) as span:
        # Sample batch for training
        batch = buffer.sample(batch_size=100)

        # Update routing model
        optimizer.update(batch)

        # Save optimized model
        model_path = f"models/routing/{tenant_id}/gepa_model.pkl"
        optimizer.save(model_path)
```

### Optimizer Selection Based on Data
```python
from src.routing.optimizer_factory import OptimizerFactory

factory = OptimizerFactory()
data_size = len(buffer)

# Select appropriate optimizer
if data_size < 100:
    # Bootstrap for cold start
    optimizer = factory.create_bootstrap_optimizer()
elif data_size < 1000:
    # SIMBA for moderate data
    optimizer = factory.create_simba_optimizer()
elif data_size < 10000:
    # MIPRO for substantial data
    optimizer = factory.create_mipro_optimizer()
else:
    # GEPA for large-scale continuous learning
    optimizer = factory.create_gepa_optimizer()
```

## 4. Memory-Augmented Search Flow

### Search with Context
```python
# 1. Initialize memory-aware agent
from src.memory.mem0_manager import Mem0Manager
from src.app.agents.memory_aware_mixin import MemoryAwareMixin

class MemoryAwareVideoAgent(EnhancedVideoSearchAgent, MemoryAwareMixin):
    def __init__(self):
        super().__init__()
        self.memory = Mem0Manager()

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
from src.evaluation.plugins.phoenix_experiment import PhoenixExperimentPlugin
import phoenix as px

plugin = PhoenixExperimentPlugin()
client = px.Client()

# 2. Setup dataset
dataset = client.upload_dataset(
    dataset_name="ml_queries_v1",
    dataframe=queries_df
)

# 3. Run experiment with multiple strategies
experiment_id = plugin.run_experiment(
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

## 6. Configuration Hot Reload Flow

### Dynamic Configuration Update
```python
# 1. Update configuration
from src.common.config_manager import get_config_manager

manager = get_config_manager()

# 2. Change LLM model for tenant
new_config = SystemConfig(
    tenant_id="customer_a",
    llm_model="gpt-4-turbo",  # Upgrade from gpt-4
    temperature=0.7
)
manager.set_system_config(new_config)

# 3. Configuration watcher detects change
from src.common.config_watcher import ConfigWatcher

watcher = ConfigWatcher(manager)

@watcher.on_change("customer_a", "SYSTEM")
def reload_llm(config):
    # Reinitialize LLM with new model
    global llm_client
    llm_client = create_llm_client(
        model=config["llm_model"],
        temperature=config["temperature"]
    )
    logger.info(f"Reloaded LLM: {config['llm_model']}")

# 4. Changes apply immediately without restart
# Next query uses gpt-4-turbo automatically
```

## 7. Tiered Cache Flow

### Cache-Aware Processing
```python
# 1. Setup tiered cache
from src.cache.tiered_manager import TieredCacheManager

cache = TieredCacheManager(
    hot=RedisBackend(),   # 1 hour TTL
    warm=LocalFSBackend(), # 24 hour TTL
    cold=S3Backend()      # 30 day TTL
)

# 2. Check cache before processing
async def process_video(video_id: str, tenant_id: str):
    cache_key = f"embeddings/colpali/{video_id}"

    # Try to get from cache
    embeddings = await cache.get_or_compute(
        key=cache_key,
        tenant_id=tenant_id,
        compute_fn=lambda: generate_embeddings(video_id),
        cache_hot=False,  # Too large for Redis
        cache_warm=True,   # Keep locally
        cache_cold=True    # Long-term storage
    )

    return embeddings

# 3. Cache hit flow
# Hot (Redis) → Miss
# Warm (Local) → Miss
# Cold (S3) → Hit!
# Promote to Warm cache
# Return without computation

# 4. Invalidate on update
await cache.invalidate_pattern(
    pattern=f"embeddings/*/{video_id}/*",
    tenant_id=tenant_id
)
```

## 8. Error Recovery Flow

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

**Last Updated**: 2025-10-04
**Status**: Production Ready