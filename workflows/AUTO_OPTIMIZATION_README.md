# Auto-Optimization Workflow Documentation

**Last Updated**: 2025-11-13
**Architecture**: 10-Package Cogniverse System with DSPy Optimization
**Audience**: ML Engineers, DevOps Engineers, Data Scientists

Automated background optimization for routing modules using Phoenix traces, synthetic data generation, and advanced DSPy optimizers including GEPA (Experience-Guided Policy Adaptation).

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [DSPy Optimizers](#dspy-optimizers)
4. [GEPA (Experience-Guided Policy Adaptation)](#gepa-experience-guided-policy-adaptation)
5. [Synthetic Data Generation](#synthetic-data-generation)
6. [Multi-Tenant Architecture](#multi-tenant-architecture)
7. [Configuration](#configuration)
8. [Deployment](#deployment)
9. [Monitoring](#monitoring)
10. [Manual Testing](#manual-testing)
11. [Troubleshooting](#troubleshooting)

---

## Overview

Auto-optimization runs automatically on a schedule (default: hourly) and optimizes routing modules based on real production traces collected in Phoenix. It supports **multi-tenancy** with independent workflow tracking per tenant. Optimization only runs when specific conditions are met, ensuring optimization happens when there's sufficient new data.

### Key Features

- **Automatic DSPy Optimizer Selection**: Bootstrap, SIMBA, MIPRO, GEPA
- **Multi-Tenant Isolation**: Independent workflows per tenant
- **Synthetic Data Generation**: Cold start support via `cogniverse-synthetic`
- **XGBoost Meta-Learning**: Intelligent training decisions
- **Phoenix Integration**: Trace collection and experiment tracking
- **Progressive Training**: Synthetic → Hybrid → Pure Real data strategies
- **Conditional Execution**: Only runs when conditions are met

### Supported Modules

| Module | Description | Optimizer Classes | Status |
|--------|-------------|-------------------|--------|
| **modality** | Per-modality routing (VIDEO, IMAGE, AUDIO, DOCUMENT, TEXT) | `ModalityOptimizer` | ✅ Active |
| **cross_modal** | Cross-modal fusion decisions | `CrossModalOptimizer` | ✅ Active |
| **routing** | Entity-based advanced routing | `AdvancedRoutingOptimizer` | ✅ Active |
| **workflow** | Workflow orchestration | `WorkflowOptimizer` | 🚧 Pending |
| **unified** | Unified routing + workflow | `UnifiedOptimizer` | 🚧 Pending |

---

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Argo CronWorkflow (Hourly)                │
│                  workflows/auto-optimization-                │
│                     multi-tenant.yaml                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         Discover Tenants (scripts/discover_tenants.py)      │
│           Query ConfigManager for active tenants            │
│           Returns: ["default", "acme_corp", ...]            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Spawn Per-Tenant Workflows                     │
│   ┌────────────────────────────────────────────────┐       │
│   │  Workflow: auto-opt-default-xyz123             │       │
│   │  Labels: tenant-id=default, type=auto          │       │
│   │  Steps:                                        │       │
│   │    1. Trigger Check (conditions)               │       │
│   │    2. Modality Optimization (parallel)         │       │
│   │    3. Cross-Modal Optimization (parallel)      │       │
│   │    4. Routing Optimization (parallel)          │       │
│   │    5. Aggregate Results                        │       │
│   └────────────────────────────────────────────────┘       │
│   ┌────────────────────────────────────────────────┐       │
│   │  Workflow: auto-opt-acme_corp-abc456           │       │
│   │  Labels: tenant-id=acme_corp, type=auto        │       │
│   │  (Independent execution, retry, history)       │       │
│   └────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│      Auto-Optimization Trigger                              │
│      (scripts/auto_optimization_trigger.py)                 │
│                                                             │
│   Checks:                                                   │
│   1. enable_auto_optimization = True?                       │
│   2. Time interval met? (since last optimization)           │
│   3. Sufficient Phoenix traces? (>= min_samples)            │
│                                                             │
│   If ALL conditions met → Execute Optimization              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│        Module Optimization Execution                        │
│        (scripts/run_module_optimization.py)                 │
│                                                             │
│   For each module (modality, cross_modal, routing):        │
│   1. Collect Phoenix traces (ModalitySpanCollector)        │
│   2. Check if training needed (XGBoost meta-model)          │
│   3. Generate synthetic data if needed (cogniverse-synthetic)│
│   4. Select DSPy optimizer (Bootstrap/MIPRO/SIMBA/GEPA)     │
│   5. Train routing module                                   │
│   6. Evaluate on validation set                             │
│   7. Save optimized model                                   │
│   8. Update marker file (timestamp)                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│             DSPy Optimizer Selection                        │
│                                                             │
│   ┌─────────────────────────────────────────────┐          │
│   │  BootstrapFewShot (Cold Start)              │          │
│   │  - Fast, few-shot learning                  │          │
│   │  - Use when: < 50 examples                  │          │
│   │  - Training time: ~2-5 minutes              │          │
│   └─────────────────────────────────────────────┘          │
│   ┌─────────────────────────────────────────────┐          │
│   │  MIPROv2 (Multi-stage Optimization)         │          │
│   │  - Instruction + prompt optimization        │          │
│   │  - Use when: 50-200 examples                │          │
│   │  - Training time: ~10-30 minutes            │          │
│   └─────────────────────────────────────────────┘          │
│   ┌─────────────────────────────────────────────┐          │
│   │  SIMBA (Query Enhancement)                  │          │
│   │  - Query reformulation & enhancement        │          │
│   │  - Use when: Complex queries, low accuracy  │          │
│   │  - Training time: ~5-15 minutes             │          │
│   └─────────────────────────────────────────────┘          │
│   ┌─────────────────────────────────────────────┐          │
│   │  GEPA (Experience-Guided Policy Adaptation) │          │
│   │  - Advanced RL-style policy learning        │          │
│   │  - Use when: > 200 examples, high complexity│          │
│   │  - Training time: ~30-60 minutes            │          │
│   └─────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         Synthetic Data Generation (Optional)                │
│         cogniverse-synthetic package                        │
│                                                             │
│   1. Sample real content from Vespa backend                 │
│   2. Generate queries using DSPy modules                    │
│   3. Create ground truth labels                             │
│   4. Validate examples (retry up to 3 times)                │
│   5. Augment training set (synthetic + real)                │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         Phoenix Experiment Tracking                         │
│         cogniverse-telemetry-phoenix package                │
│                                                             │
│   1. Log optimization metrics                               │
│   2. Track training/validation accuracy                     │
│   3. Record optimizer hyperparameters                       │
│   4. Store model artifacts                                  │
│   5. Enable A/B testing comparison                          │
└─────────────────────────────────────────────────────────────┘
```

### Package Dependencies

```python
# Foundation Layer
from cogniverse_foundation.config.utils import get_config, create_default_config_manager
from cogniverse_foundation.telemetry.manager import get_telemetry_manager

# Core Layer
from cogniverse_core.agent_context import AgentContext
from cogniverse_telemetry_phoenix import PhoenixProvider

# Implementation Layer
from cogniverse_agents.routing.modality_optimizer import ModalityOptimizer
from cogniverse_agents.routing.cross_modal_optimizer import CrossModalOptimizer
from cogniverse_agents.routing.advanced_optimizer import AdvancedRoutingOptimizer
from cogniverse_synthetic import SyntheticDataService, SyntheticDataRequest
from cogniverse_vespa import VespaBackend

# DSPy Framework
import dspy
from dspy.teleprompt import BootstrapFewShot, MIPROv2
```

---

## DSPy Optimizers

### Optimizer Selection Strategy

The system uses **XGBoost meta-learning** to automatically select the best optimizer based on:

1. **Dataset Size**: Number of training examples available
2. **Data Quality**: Confidence scores, label consistency
3. **Historical Performance**: Past optimization results
4. **Training Time Budget**: Available compute resources
5. **Query Complexity**: Modality, entities, multi-step reasoning

### 1. BootstrapFewShot

**Purpose**: Fast few-shot learning for cold start scenarios

**Algorithm**:
1. Use existing model to generate predictions on training set
2. Select high-confidence examples as demonstrations
3. Compile module with bootstrapped few-shot examples
4. Minimal training, fast compilation

**Use Cases**:
- Cold start (< 50 examples)
- Quick prototyping
- Simple routing decisions
- Low-latency requirements

**Configuration**:
```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=evaluation_metric,
    max_bootstrapped_demos=5,
    max_labeled_demos=10,
    max_rounds=1
)

optimized_module = optimizer.compile(
    student=routing_module,
    trainset=training_data
)
```

**Training Time**: 2-5 minutes
**Expected Improvement**: 10-20% accuracy gain

---

### 2. MIPROv2 (Multi-stage Instruction Prompt Optimization)

**Purpose**: Multi-stage optimization of instructions and prompts

**Algorithm**:
1. **Stage 1**: Optimize instruction text using candidate generation
2. **Stage 2**: Optimize few-shot examples via selection
3. **Stage 3**: Joint optimization of instructions + examples
4. Use Bayesian optimization for hyperparameter search

**Use Cases**:
- 50-200 training examples
- Complex reasoning tasks
- Multi-step workflows
- When instruction quality matters

**Configuration**:
```python
from dspy.teleprompt import MIPROv2

optimizer = MIPROv2(
    metric=evaluation_metric,
    num_candidates=10,
    init_temperature=1.0,
    num_threads=4,
    max_bootstrapped_demos=5,
    max_labeled_demos=10
)

optimized_module = optimizer.compile(
    student=routing_module,
    trainset=training_data,
    num_trials=100,
    max_errors=10
)
```

**Training Time**: 10-30 minutes
**Expected Improvement**: 20-40% accuracy gain

**Features**:
- Automatic instruction generation
- Few-shot example selection
- Hyperparameter optimization
- Parallel candidate evaluation

---

### 3. SIMBA (Query Enhancement)

**Purpose**: Query reformulation and enhancement for better routing

**Algorithm**:
1. Analyze query characteristics (modality signals, entities, intent)
2. Generate query enhancements (expansions, clarifications)
3. Use enhanced queries for routing decisions
4. Learn enhancement patterns from successful traces

**Use Cases**:
- Complex, ambiguous queries
- Multi-modal queries
- Low initial accuracy (< 60%)
- Query understanding bottlenecks

**Configuration**:
```python
from cogniverse_agents.routing.simba_query_enhancer import SIMBAQueryEnhancer

enhancer = SIMBAQueryEnhancer()
enhanced_query = enhancer.enhance(
    query="show me videos about machine learning",
    context=agent_context
)

# Use enhanced query for routing
routing_result = routing_module(
    query=enhanced_query,
    modality=detected_modality
)
```

**Training Time**: 5-15 minutes
**Expected Improvement**: 15-30% accuracy gain

**Features**:
- Query expansion
- Entity extraction
- Intent clarification
- Multi-modal signal detection

---

### 4. GEPA (Experience-Guided Policy Adaptation)

**Purpose**: Advanced policy learning using experience replay and adaptation

**Algorithm**:
1. **Experience Collection**: Collect successful routing traces from Phoenix
2. **Policy Initialization**: Initialize policy network with DSPy module
3. **Experience Replay**: Sample diverse experiences for training
4. **Policy Gradient**: Update policy using REINFORCE with baselines
5. **Adaptive Learning**: Adjust learning rate based on performance
6. **Policy Distillation**: Distill learned policy into smaller module

**Use Cases**:
- Large datasets (> 200 examples)
- Complex routing scenarios
- Multi-step decision making
- High-value production workloads
- When accuracy > latency

**Configuration**:
```python
from cogniverse_agents.routing.gepa_optimizer import GEPAOptimizer

optimizer = GEPAOptimizer(
    tenant_id="default",
    experience_buffer_size=1000,
    batch_size=32,
    learning_rate=0.001,
    discount_factor=0.99,
    exploration_rate=0.1,
    num_epochs=50
)

# Collect experiences from Phoenix
experiences = optimizer.collect_experiences(
    lookback_hours=72,
    min_confidence=0.7
)

# Train policy
optimized_policy = optimizer.train(
    experiences=experiences,
    validation_split=0.2,
    early_stopping_patience=5
)

# Evaluate policy
metrics = optimizer.evaluate(
    policy=optimized_policy,
    test_set=validation_data
)
```

**Training Time**: 30-60 minutes
**Expected Improvement**: 40-60% accuracy gain

**Key Features**:

1. **Experience Replay Buffer**:
   - Stores successful routing traces
   - Prioritized sampling by confidence
   - Diversity-based selection
   - Temporal consistency tracking

2. **Policy Network**:
   - Multi-layer DSPy module
   - Attention mechanisms for context
   - Modality-specific heads
   - Confidence estimation

3. **Reward Shaping**:
   - Task success (+10)
   - High user satisfaction (+5)
   - Low latency bonus (+2)
   - Correct modality (+3)
   - Penalties for failures (-5)

4. **Adaptive Learning**:
   - Dynamic learning rate scheduling
   - Gradient clipping for stability
   - Batch normalization
   - Dropout for regularization

5. **Policy Distillation**:
   - Compress large policy into smaller model
   - Knowledge distillation from teacher
   - Maintains 90%+ accuracy with 3x speedup

**Mathematical Foundation**:

```
Policy Gradient Objective:
J(θ) = E_τ~π_θ[Σ_t γ^t R(s_t, a_t)]

Where:
- θ: Policy parameters (DSPy module weights)
- τ: Trajectory (sequence of routing decisions)
- π_θ: Policy (routing module)
- γ: Discount factor (0.99)
- R: Reward function (task success, latency, accuracy)

Update Rule:
θ ← θ + α ∇_θ J(θ)

With baseline:
∇_θ J(θ) ≈ E_τ[Σ_t ∇_θ log π_θ(a_t|s_t) (G_t - b(s_t))]

Where:
- α: Learning rate (0.001)
- G_t: Return from time t
- b(s_t): Baseline (value function)
```

**GEPA vs. Standard DSPy Optimizers**:

| Feature | BootstrapFewShot | MIPROv2 | GEPA |
|---------|------------------|---------|------|
| Training Data | 10-50 examples | 50-200 examples | 200+ examples |
| Training Time | 2-5 min | 10-30 min | 30-60 min |
| Accuracy Gain | 10-20% | 20-40% | 40-60% |
| Experience Replay | ❌ | ❌ | ✅ |
| Policy Gradient | ❌ | ❌ | ✅ |
| Adaptive Learning | ❌ | ✅ | ✅ |
| Multi-Step Reasoning | ❌ | ✅ | ✅ |
| Reward Shaping | ❌ | ❌ | ✅ |
| Policy Distillation | ❌ | ❌ | ✅ |

---

## Synthetic Data Generation

### Overview

The `cogniverse-synthetic` package generates high-quality training data when real Phoenix traces are insufficient.

### Architecture

```python
from cogniverse_synthetic import SyntheticDataService, SyntheticDataRequest
from cogniverse_vespa import VespaBackend

# Initialize service
backend = VespaBackend(config)
synthetic_service = SyntheticDataService(backend=backend)

# Generate modality routing examples
request = SyntheticDataRequest(
    optimizer="modality",
    count=100,
    modalities=["video", "image", "audio", "document", "text"],
    quality_threshold=0.8
)

response = await synthetic_service.generate(request)

# Use generated examples for training
training_data = response.examples
```

### Generation Process

1. **Content Sampling**:
   - Sample real content from Vespa backend
   - Ensure diverse modality coverage
   - Select high-quality documents

2. **Query Generation** (DSPy-Driven):
   ```python
   class QueryGenerationSignature(dspy.Signature):
       """Generate realistic query from document"""
       document_content = dspy.InputField(desc="Document content/metadata")
       modality = dspy.InputField(desc="Content modality")

       query = dspy.OutputField(desc="Natural language query")
       expected_result = dspy.OutputField(desc="Expected search result")
       confidence = dspy.OutputField(desc="Generation confidence")

   class QueryGenerator(dspy.Module):
       def __init__(self):
           self.generate = dspy.ChainOfThought(QueryGenerationSignature)

       def forward(self, document, modality):
           result = self.generate(
               document_content=document,
               modality=modality
           )
           return result
   ```

3. **Validation & Retry**:
   - Validate query quality (non-empty, relevant)
   - Retry up to 3 times on failure
   - Ensure ground truth correctness

4. **Augmentation Strategies**:
   - **Paraphrasing**: Rephrase queries with same intent
   - **Modality Variation**: Same query, different modalities
   - **Complexity Scaling**: Simple → complex query variations
   - **Negative Examples**: Generate contrasting examples

### Optimizer-Specific Generators

#### Modality Routing Generator
```python
request = SyntheticDataRequest(
    optimizer="modality",
    count=100,
    modalities=["video", "image", "audio", "document", "text"],
    profiles=["frame_based", "segment_based"]
)
```

**Generates**:
- Query text
- Detected modality
- Expected agent (video_search, image_search, etc.)
- Confidence score
- Query features (entities, temporal references, etc.)

#### Cross-Modal Generator
```python
request = SyntheticDataRequest(
    optimizer="crossmodal",
    count=50,
    source_modality="video",
    target_modality="image",
    fusion_strategy="maxsim"
)
```

**Generates**:
- Multi-modal queries
- Fusion requirements (combine video + image)
- Expected fusion strategy
- Relevance scores

#### Routing Generator
```python
request = SyntheticDataRequest(
    optimizer="routing",
    count=75,
    route_types=["direct", "fallback", "hybrid"]
)
```

**Generates**:
- Complex routing scenarios
- Multi-agent workflows
- Fallback strategies
- Error handling cases

---

## Multi-Tenant Architecture

### Key Features

Each tenant gets a **separate workflow** for independent monitoring, retry, and history.

```yaml
CronWorkflow: auto-optimization-multi-tenant
  ↓
[Discover Tenants] → ["default", "acme_corp", "enterprise"]
  ↓
[Spawn Workflows]
  ├─ Workflow: auto-opt-default-xyz123
  │    Labels: tenant-id=default, optimization-type=auto
  │    Independent retry, history, monitoring
  │
  ├─ Workflow: auto-opt-acme_corp-abc456
  │    Labels: tenant-id=acme_corp, optimization-type=auto
  │    Independent retry, history, monitoring
  │
  └─ Workflow: auto-opt-enterprise-def789
       Labels: tenant-id=enterprise, optimization-type=auto
       Independent retry, history, monitoring
```

### Benefits

- ✅ **Independent workflow history** per tenant
- ✅ **Per-tenant monitoring** and debugging
- ✅ **Isolated failures** - one tenant doesn't block others
- ✅ **Individual retry policies** per tenant
- ✅ **Clear audit trail** with workflow labels

### Workflow Parallelization

Each tenant workflow optimizes modules in parallel:

```yaml
Workflow: auto-opt-default-xyz123
  ├─ Step 1: Trigger Check (sequential)
  └─ Step 2: Parallel Optimization
       ├─ modality optimization (async)
       ├─ cross_modal optimization (async)
       └─ routing optimization (async)
  └─ Step 3: Aggregate Results (sequential)
```

---

## Configuration

### Routing Configuration

Auto-optimization is configured via `RoutingConfigUnified` from `cogniverse-foundation`:

```python
from cogniverse_foundation.config.unified_config import RoutingConfigUnified

routing_config = RoutingConfigUnified(
    tenant_id="default",

    # Auto-optimization settings
    enable_auto_optimization=True,
    optimization_interval_seconds=3600,  # 1 hour
    min_samples_for_optimization=100,

    # Optimizer selection
    preferred_optimizer="auto",  # auto | bootstrap | mipro | simba | gepa
    max_training_iterations=100,

    # Synthetic data
    use_synthetic_data=True,
    synthetic_data_ratio=0.3,  # 30% synthetic, 70% real

    # Phoenix settings
    phoenix_lookback_hours=24,
    min_trace_confidence=0.7
)

# Save configuration
config_manager.save_routing_config(routing_config)
```

### Environment Variables

```bash
# LLM API Keys (for DSPy teacher model and auto-annotator)
export ROUTER_OPTIMIZER_TEACHER_KEY="your-api-key"  # Works with any LiteLLM-supported provider
export ANNOTATION_API_KEY="your-api-key"            # For LLM auto-annotator (optional)

# Phoenix
export PHOENIX_ENDPOINT="http://localhost:6006"
export PHOENIX_PROJECT_PREFIX="cogniverse"

# Vespa Backend
export VESPA_URL="http://localhost:8080"

# Configuration Store
export CONFIG_STORE_URL="redis://localhost:6379"

# Kubernetes
export KUBERNETES_NAMESPACE="cogniverse"
```

---

## Deployment

### 1. Deploy Multi-Tenant Auto-Optimization

```bash
# Deploy WorkflowTemplate + CronWorkflow
kubectl apply -f workflows/auto-optimization-multi-tenant.yaml

# Verify deployment
kubectl get cronworkflows -n cogniverse
kubectl get cronworkflow auto-optimization-multi-tenant -n cogniverse -o yaml

# Verify WorkflowTemplate
kubectl get workflowtemplate tenant-auto-optimization -n cogniverse
```

**Expected Output**:
```
NAME                            SCHEDULE      SUSPEND   ACTIVE   LAST SCHEDULE
auto-optimization-multi-tenant  0 * * * *     False     1        2025-11-13T14:00:00Z
```

### 2. Configure Docker Image

Build optimization image with all dependencies:

```bash
# Build optimization image
docker build -t cogniverse/optimization:latest -f docker/Dockerfile.optimization .

# Push to registry
docker push cogniverse/optimization:latest

# Or build with specific packages
docker build \
  --build-arg PACKAGES="cogniverse-agents cogniverse-synthetic cogniverse-vespa" \
  -t cogniverse/optimization:latest \
  -f docker/Dockerfile.optimization .
```

**Dockerfile.optimization**:
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Copy workspace
WORKDIR /app
COPY . .

# Install packages
RUN uv sync --all-extras

# Set entrypoint
ENTRYPOINT ["uv", "run", "python"]
```

### 3. Configure Secrets

```bash
# Create Kubernetes secret with config store URL
kubectl create secret generic cogniverse-config \
  --from-literal=store-url="redis://redis.cogniverse.svc.cluster.local:6379" \
  --from-literal=teacher-api-key="${ROUTER_OPTIMIZER_TEACHER_KEY}" \
  --from-literal=annotation-api-key="${ANNOTATION_API_KEY}" \
  -n cogniverse

# Verify secret
kubectl get secret cogniverse-config -n cogniverse -o yaml
```

### 4. Schedule Configuration

Edit schedule in `workflows/auto-optimization-multi-tenant.yaml`:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: CronWorkflow
metadata:
  name: auto-optimization-multi-tenant
  namespace: cogniverse
spec:
  schedule: "0 * * * *"  # Every hour
  # schedule: "0 */2 * * *"  # Every 2 hours
  # schedule: "0 0 * * *"  # Daily at midnight
  # schedule: "*/30 * * * *"  # Every 30 minutes

  timezone: "America/New_York"
  concurrencyPolicy: "Forbid"  # Don't run concurrent workflows

  workflowSpec:
    # ... workflow definition ...
```

### 5. Resource Limits

Configure resources based on workload:

```yaml
# For small workloads (< 1000 traces/hour)
resources:
  requests:
    memory: "2Gi"
    cpu: "1"
  limits:
    memory: "8Gi"
    cpu: "4"

# For medium workloads (1000-10000 traces/hour)
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "16Gi"
    cpu: "8"

# For large workloads (> 10000 traces/hour)
resources:
  requests:
    memory: "8Gi"
    cpu: "4"
  limits:
    memory: "32Gi"
    cpu: "16"
```

---

## Monitoring

### Multi-Tenant Workflow Monitoring

```bash
# List all optimization workflows
kubectl get workflows -n cogniverse --sort-by=.metadata.creationTimestamp

# Filter by tenant
kubectl get workflows -n cogniverse -l tenant-id=default
kubectl get workflows -n cogniverse -l tenant-id=acme_corp

# Filter by optimization type
kubectl get workflows -n cogniverse -l optimization-type=auto

# Get specific tenant workflow
kubectl get workflow auto-opt-default-abc123 -n cogniverse -o yaml

# View logs for specific tenant
kubectl logs -n cogniverse -l workflows.argoproj.io/workflow=auto-opt-default-abc123

# View logs for specific step
kubectl logs -n cogniverse auto-opt-default-abc123-modality-optimization
```

### Tenant Discovery Monitoring

```bash
# Check which tenants were discovered in last run
kubectl logs -n cogniverse -l workflows.argoproj.io/workflow-template=auto-optimization-multi-tenant \
  -c discover-tenants --tail=50

# Expected output:
# Discovered tenants: ['default', 'acme_corp', 'enterprise']
# Spawning workflows for 3 tenants...
```

### Optimization Metrics

```bash
# Query Phoenix for optimization metrics
curl -X POST http://localhost:6006/v1/graphql \
  -H "Content-Type: application/json" \
  -d '{
    "query": "{ experiments(limit: 10) { name metrics { name value } } }"
  }'

# View metrics in Phoenix UI
open http://localhost:6006/#/experiments

# Export metrics to JSON
curl http://localhost:6006/v1/experiments/latest > optimization_metrics.json
```

### Argo Workflows UI

```bash
# Port-forward Argo UI
kubectl port-forward -n argo svc/argo-server 2746:2746

# Access UI
open https://localhost:2746

# View workflows
# Navigate to: Workflows → cogniverse namespace
# Filter by: label:optimization-type=auto
```

### Phoenix Dashboard

```bash
# Launch Phoenix dashboard
uv run streamlit run scripts/phoenix_dashboard.py -- \
  --tenant-id default \
  --phoenix-endpoint http://localhost:6006

# Multi-tenant dashboard
uv run streamlit run scripts/phoenix_dashboard.py -- \
  --tenants default acme_corp enterprise
```

---

## Manual Testing

### Test Trigger Script Locally

```bash
# Test trigger with conditions check
uv run python scripts/auto_optimization_trigger.py \
  --tenant-id default \
  --module routing \
  --phoenix-endpoint http://localhost:6006

# Expected output:
# Checking auto-optimization conditions...
# ✅ Auto-optimization enabled: True
# ✅ Time interval met: True (last run: 2 hours ago)
# ✅ Sufficient traces: True (150 traces, min: 100)
# 🚀 Triggering optimization...
# Exit code: 0 (success)

# Test with force flag (bypass conditions)
uv run python scripts/auto_optimization_trigger.py \
  --tenant-id default \
  --module modality \
  --force

# Test with dry-run
uv run python scripts/auto_optimization_trigger.py \
  --tenant-id acme_corp \
  --module cross_modal \
  --dry-run

# Expected output:
# [DRY RUN] Would trigger optimization with:
#   Tenant: acme_corp
#   Module: cross_modal
#   Traces: 250
#   Last run: 3 hours ago
```

### Test Module Optimization

```bash
# Test modality optimization
uv run python scripts/run_module_optimization.py \
  --module modality \
  --tenant-id default \
  --lookback-hours 24 \
  --min-confidence 0.7 \
  --output results/test_modality.json

# Test with synthetic data
uv run python scripts/run_module_optimization.py \
  --module cross_modal \
  --tenant-id default \
  --use-synthetic-data \
  --output results/test_cross_modal.json

# Test all modules
uv run python scripts/run_module_optimization.py \
  --module all \
  --tenant-id default \
  --use-synthetic-data \
  --force-training \
  --output results/test_all.json

# Review results
cat results/test_all.json | jq '.summary'
```

### Test Synthetic Data Generation

```bash
# Test synthetic data service
python -c "
import asyncio
from cogniverse_synthetic import SyntheticDataService, SyntheticDataRequest
from cogniverse_vespa import VespaBackend

async def test():
    backend = VespaBackend(config)
    service = SyntheticDataService(backend=backend)

    request = SyntheticDataRequest(
        optimizer='modality',
        count=10,
        modalities=['video', 'image']
    )

    response = await service.generate(request)
    print(f'Generated {response.count} examples')
    print(f'Success rate: {response.metadata[\"success_rate\"]}')

    for example in response.examples[:3]:
        print(f'Query: {example.query}')
        print(f'Modality: {example.modality}')
        print()

asyncio.run(test())
"
```

---

## Troubleshooting

### Auto-Optimization Not Running

**1. Check CronWorkflow is active**:
```bash
kubectl get cronworkflow auto-optimization-multi-tenant -n cogniverse

# If suspended, resume:
kubectl patch cronworkflow auto-optimization-multi-tenant -n cogniverse \
  -p '{"spec":{"suspend":false}}'
```

**2. Check routing config**:
```bash
# Verify enable_auto_optimization is True
python -c "
from cogniverse_foundation.config.utils import create_default_config_manager

manager = create_default_config_manager()
config = manager.get_routing_config('default')

print(f'Auto-optimization enabled: {config.enable_auto_optimization}')
print(f'Interval (seconds): {config.optimization_interval_seconds}')
print(f'Min samples: {config.min_samples_for_optimization}')
"
```

**3. Check Phoenix traces**:
```bash
# Verify traces are being collected
curl http://localhost:6006/v1/traces?limit=10

# Check project exists
curl http://localhost:6006/v1/projects

# Expected: cogniverse-default-cogniverse.routing
```

### Optimization Failing

**1. Check logs**:
```bash
# Get workflow logs
kubectl logs -n cogniverse <workflow-pod-name>

# Get specific step logs
kubectl logs -n cogniverse <workflow-pod-name> -c modality-optimization

# Follow logs in real-time
kubectl logs -n cogniverse <workflow-pod-name> -f
```

**2. Check Phoenix connectivity**:
```bash
# Test Phoenix endpoint
curl http://phoenix.cogniverse.svc.cluster.local:6006/healthz

# Test from pod
kubectl run -it --rm debug --image=curlimages/curl --restart=Never -- \
  curl http://phoenix.cogniverse.svc.cluster.local:6006/healthz
```

**3. Check training data quality**:
```bash
# Inspect collected traces
uv run python -c "
import asyncio
from cogniverse_agents.routing.modality_span_collector import ModalitySpanCollector

async def check():
    collector = ModalitySpanCollector(tenant_id='default')
    spans = await collector.collect_spans(lookback_hours=24, min_confidence=0.7)

    print(f'Collected {len(spans)} spans')
    print(f'Modality breakdown:')
    for modality, count in spans.groupby('modality').size().items():
        print(f'  {modality}: {count}')

asyncio.run(check())
"
```

**4. Check LLM API keys**:
```bash
# Verify API keys in secret
kubectl get secret cogniverse-config -n cogniverse -o jsonpath='{.data.teacher-api-key}' | base64 -d

# Verify key is set
echo "ROUTER_OPTIMIZER_TEACHER_KEY is ${ROUTER_OPTIMIZER_TEACHER_KEY:+set}"
```

### Optimization Too Frequent

**Increase `optimization_interval_seconds`**:
```python
from cogniverse_foundation.config.utils import create_default_config_manager

manager = create_default_config_manager()
config = manager.get_routing_config('default')

config.optimization_interval_seconds = 7200  # 2 hours
manager.save_routing_config(config)
```

### Insufficient Traces

**Lower `min_samples_for_optimization` or use synthetic data**:
```python
config.min_samples_for_optimization = 50  # Lower threshold
config.use_synthetic_data = True  # Enable synthetic data
manager.save_routing_config(config)
```

### GEPA Training Out of Memory

**Reduce batch size or use gradient accumulation**:
```python
from cogniverse_agents.routing.gepa_optimizer import GEPAOptimizer

optimizer = GEPAOptimizer(
    tenant_id="default",
    batch_size=16,  # Reduce from 32
    gradient_accumulation_steps=2,  # Accumulate gradients
    experience_buffer_size=500  # Reduce buffer
)
```

---

## Best Practices

### Configuration

1. **Start Conservative**:
   - High `min_samples` (200+)
   - Long intervals (2+ hours)
   - Enable synthetic data for cold start

2. **Monitor Performance**:
   - Track optimization impact in Phoenix
   - Review logs regularly
   - Set up alerts for failures

3. **Gradual Tuning**:
   - Adjust based on trace volume
   - Optimize model performance
   - Balance training time vs. accuracy

### Optimization Strategy

1. **Optimizer Selection**:
   - Use BootstrapFewShot for cold start (< 50 examples)
   - Use MIPROv2 for medium datasets (50-200 examples)
   - Use GEPA for large datasets (> 200 examples)
   - Use SIMBA for query understanding issues

2. **Synthetic Data**:
   - Start with 30% synthetic, 70% real
   - Increase synthetic ratio if insufficient real data
   - Validate synthetic quality regularly

3. **Multi-Tenancy**:
   - Configure per-tenant optimization schedules
   - Isolate tenant resources
   - Monitor cross-tenant performance

---

## Related Documentation

- **Scripts**: `scripts/README.md`
- **Synthetic Data**: `libs/synthetic/README.md`
- **Optimization**: `docs/modules/optimization.md`
- **Routing**: `docs/modules/routing.md`
- **Phoenix Telemetry**: `libs/telemetry-phoenix/README.md`
- **Argo Workflows**: https://argoproj.github.io/argo-workflows/

---

**Last Updated**: 2025-11-13
**Maintainer**: Cogniverse ML Team
**Support**: See workflow logs and Phoenix dashboard for debugging
