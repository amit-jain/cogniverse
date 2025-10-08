# Optimization Module Study Guide

**Module:** `src/app/routing/` (optimization components)
**Purpose:** DSPy-based multi-stage optimization for routing, modality-specific, and cross-modal decision-making
**Last Updated:** 2025-10-07

---

## Table of Contents
1. [Module Overview](#module-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Usage Examples](#usage-examples)
5. [Production Considerations](#production-considerations)
6. [Testing](#testing)

---

## Module Overview

### Purpose
The Optimization Module provides sophisticated multi-stage optimization for routing decisions, modality-specific routing, and cross-modal result fusion using DSPy 3.0 advanced optimizers (GEPA, MIPROv2, SIMBA, BootstrapFewShot).

### Key Features
- **Advanced DSPy Optimization**: GEPA, MIPROv2, SIMBA, BootstrapFewShot optimizers
- **GRPO (Gradient-based Reward Policy Optimization)**: Experience replay and reward signals
- **Modality-Specific Optimization**: Per-modality routing with XGBoost meta-learning
- **Unified Optimization**: Bidirectional learning between routing and orchestration
- **Optimizer Coordination**: Facade pattern for routing optimization requests
- **Complete Optimization Pipeline**: Automatic span evaluation, annotation, and feedback loops

### Dependencies
```python
from src.app.routing.advanced_optimizer import AdvancedRoutingOptimizer
from src.app.routing.modality_optimizer import ModalityOptimizer
from src.app.routing.cross_modal_optimizer import CrossModalOptimizer
from src.app/routing.unified_optimizer import UnifiedOptimizer
from src.app.routing.optimizer_coordinator import OptimizerCoordinator
from src.app.routing.optimization_orchestrator import OptimizationOrchestrator
```

**Module Location:** `src/app/routing/` (optimization components)

---

## Architecture

### 1. Multi-Stage Optimization Architecture

```mermaid
graph TB
    Orchestrator[Optimization Orchestrator<br/>• Continuous span evaluation every 15m<br/>• Annotation workflow every 30m<br/>• Feedback loop integration every 15m<br/>• Automatic optimization triggering]

    Orchestrator --> Unified[Unified Optimizer Bidirectional Learning<br/>• Routing Optimization ←→ Orchestration Outcomes<br/>• Workflow Intelligence integration<br/>• Cross-pollination of insights]

    Unified --> AdvRouting[Advanced<br/>Routing<br/>Optimizer]
    Unified --> Modality[Modality<br/>Optimizer]
    Unified --> CrossModal[Cross-Modal<br/>Optimizer]
    Unified --> Coordinator[Optimizer<br/>Coordinator<br/>Facade]

    style Orchestrator fill:#e1f5ff
    style Unified fill:#fff4e1
    style AdvRouting fill:#ffe1e1
    style Modality fill:#ffe1e1
    style CrossModal fill:#ffe1e1
    style Coordinator fill:#ffe1e1
```

### 2. Advanced Routing Optimizer Architecture (GRPO)

```mermaid
graph TB
    Collection[Routing Experience Collection<br/>• Query, entities, relationships, enhanced_query<br/>• Chosen agent, routing confidence<br/>• Search quality, agent success, user satisfaction<br/>• Processing time → Computed reward]

    Collection --> Buffer[Experience Replay Buffer<br/>• Store routing experiences max 1000<br/>• Sample batches for training batch_size=32<br/>• Track metrics: avg_reward, success_rate, confidence_accuracy]

    Buffer --> Pipeline[Multi-Stage DSPy Optimization Pipeline<br/>Dataset Size < 20: BootstrapFewShot<br/>Dataset Size >= 50: SIMBA similarity-based memory<br/>Dataset Size >= 100: MIPROv2 metric-aware instruction<br/>Dataset Size >= 200: GEPA reflective prompt evolution<br/>Optimizer Selection: adaptive, gepa, mipro, simba]

    Pipeline --> Policy[Optimized Routing Policy Module<br/>• RoutingPolicySignature query → recommended_agent + confidence<br/>• ChainOfThought reasoning<br/>• Confidence calibration<br/>• Exploration vs Exploitation epsilon-greedy]

    style Collection fill:#e1f5ff
    style Buffer fill:#fff4e1
    style Pipeline fill:#ffe1e1
    style Policy fill:#e1ffe1
```

### 3. Modality Optimizer Architecture (XGBoost Meta-Learning)

```mermaid
graph TB
    SpanCollection[Modality Span Collection<br/>• Collect spans per modality VIDEO, IMAGE, AUDIO, DOCUMENT<br/>• Extract modality features from telemetry<br/>• Filter by confidence threshold]

    SpanCollection --> XGBoost[XGBoost Meta-Models Decision Making<br/>1. TrainingDecisionModel:<br/>• should_train context → bool + expected_improvement<br/>• Features: sample_count, success_rate, days_since_training<br/><br/>2. TrainingStrategyModel:<br/>• select_strategy context → SKIP / SYNTHETIC / HYBRID / REAL<br/>• Progressive strategies based on data availability]

    XGBoost --> Synthetic[Synthetic Data Generation<br/>• Generate training data when real data < 20 examples<br/>• Query Vespa for ingested content<br/>• Create synthetic queries from video metadata<br/>• HYBRID: Mix real + synthetic 1:1 ratio]

    Synthetic --> Training[Modality-Specific DSPy Module Training<br/>• ModalityRoutingSignature query, modality → agent + confidence<br/>• ChainOfThought reasoning<br/>• MIPROv2 if ≥50 examples or BootstrapFewShot if <50<br/>• Save trained models per modality]

    style SpanCollection fill:#e1f5ff
    style XGBoost fill:#fff4e1
    style Synthetic fill:#ffe1e1
    style Training fill:#e1ffe1
```

---

## Core Components

### 1. **OptimizationOrchestrator** (`optimization_orchestrator.py`)

Complete end-to-end optimization pipeline orchestrator.

**Key Methods:**

```python
async def start(self) -> None:
    """
    Start continuous optimization processes:
    - Span evaluation (every span_eval_interval_minutes)
    - Annotation workflow (every annotation_interval_minutes)
    - Feedback loop (every feedback_interval_minutes)
    - Metrics reporting (every 5 minutes)
    """

async def run_once(self) -> Dict[str, Any]:
    """
    Run single optimization cycle (for testing):
    1. Evaluate spans from Phoenix
    2. Identify spans needing annotation
    3. Generate LLM annotations
    4. Process feedback loop

    Returns optimization results
    """

def get_metrics(self) -> Dict[str, Any]:
    """Get orchestrator metrics including uptime and component stats"""
```

**Constructor Parameters:**
```python
__init__(
    tenant_id: str = "default",
    span_eval_interval_minutes: int = 15,
    annotation_interval_minutes: int = 30,
    feedback_interval_minutes: int = 15,
    confidence_threshold: float = 0.6,
    min_annotations_for_optimization: int = 50,
    optimization_improvement_threshold: float = 0.05
)
```

**Internal Components:**
- `PhoenixSpanEvaluator` - Extract routing experiences from telemetry
- `AnnotationAgent` - Identify low-quality spans
- `LLMAutoAnnotator` - Generate annotations automatically
- `AnnotationStorage` - Store annotations in Phoenix
- `AnnotationFeedbackLoop` - Feed annotations to optimizer
- `AdvancedRoutingOptimizer` - GRPO optimization

**File:** `src/app/routing/optimization_orchestrator.py`

---

### 2. **AdvancedRoutingOptimizer** (`advanced_optimizer.py:136-1273`)

GRPO-based routing optimizer with multi-stage DSPy optimization.

**Key Methods:**

```python
async def record_routing_experience(
    query: str,
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    enhanced_query: str,
    chosen_agent: str,
    routing_confidence: float,
    search_quality: float,
    agent_success: bool,
    processing_time: float = 0.0,
    user_satisfaction: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> float:
    """
    Record routing experience and compute reward signal

    Reward Computation:
    reward = search_quality * weight_sq
           + agent_success * weight_success
           + user_satisfaction * weight_us
           - processing_time_penalty

    Returns computed reward (0-1)
    """

async def get_routing_recommendations(
    query: str,
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Get routing recommendations using optimized policy

    Returns:
        {
            "recommended_agent": str,
            "confidence": float,
            "reasoning": str,
            "optimization_ready": bool,
            "experiences_count": int,
            "training_step": int
        }
    """

async def optimize_routing_decision(
    query: str,
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    enhanced_query: str,
    baseline_prediction: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply GRPO optimization to improve routing decision

    Uses epsilon-greedy exploration:
    - Exploration (ε): Random agent selection for learning
    - Exploitation (1-ε): Use optimized policy

    Returns optimized routing decision with confidence calibration
    """

def get_optimization_status(self) -> Dict[str, Any]:
    """
    Get optimization status and metrics:
    - Optimizer readiness
    - Experience counts
    - Training step
    - Metrics: avg_reward, success_rate, confidence_accuracy
    """
```

**Configuration (AdvancedOptimizerConfig):**
```python
@dataclass
class AdvancedOptimizerConfig:
    # Learning parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    experience_replay_size: int = 1000
    update_frequency: int = 10

    # Optimizer selection
    optimizer_strategy: str = "adaptive"  # "gepa", "mipro", "simba", "bootstrap"
    force_optimizer: Optional[str] = None
    enable_multi_stage: bool = True

    # Optimizer thresholds
    bootstrap_threshold: int = 20
    simba_threshold: int = 50
    mipro_threshold: int = 100
    gepa_threshold: int = 200

    # Reward computation
    search_quality_weight: float = 0.4
    agent_success_weight: float = 0.3
    user_satisfaction_weight: float = 0.3
    processing_time_penalty: float = 0.1

    # Exploration
    exploration_epsilon: float = 0.1
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05

    # Minimum experiences before training
    min_experiences_for_training: int = 50
```

**Multi-Stage Optimizer Selection:**
```python
# Adaptive strategy selects best optimizer based on dataset size:
- Dataset < 20:   BootstrapFewShot (few-shot learning)
- Dataset >= 50:  SIMBA (similarity-based memory augmentation)
- Dataset >= 100: MIPROv2 (metric-aware instruction optimization)
- Dataset >= 200: GEPA (reflective prompt evolution)
```

**File:** `src/app/routing/advanced_optimizer.py:136-1273`

---

### 3. **UnifiedOptimizer** (`unified_optimizer.py:23-260`)

Bidirectional learning between routing and orchestration.

**Key Methods:**

```python
async def integrate_orchestration_outcomes(
    workflow_executions: List[WorkflowExecution]
) -> Dict[str, Any]:
    """
    Convert orchestration workflows → routing experiences

    Learns from successful orchestration patterns:
    - When parallel orchestration outperforms sequential
    - When multi-agent synergy produces better results
    - When orchestration is beneficial vs single agent

    Returns:
        {
            "workflows_processed": int,
            "routing_experiences_created": int,
            "patterns_learned": Dict[str, int],
            "total_workflows_integrated": int
        }
    """

async def optimize_unified_policy(self) -> Dict[str, Any]:
    """
    Trigger unified optimization across routing + orchestration:

    1. Optimize orchestration workflows (WorkflowIntelligence)
    2. Optimize routing decisions (AdvancedRoutingOptimizer)
    3. Cross-pollinate (orchestration insights → routing knowledge)

    Returns combined optimization results
    """

def _workflow_to_routing_experience(
    workflow: WorkflowExecution
) -> RoutingExperience:
    """
    Convert WorkflowExecution to RoutingExperience

    Mappings:
    - chosen_agent: First agent in workflow sequence
    - search_quality: Based on user_satisfaction or success
    - agent_success: workflow.success
    - metadata: Captures orchestration_pattern, agent_sequence
    """
```

**Constructor:**
```python
__init__(
    routing_optimizer: AdvancedRoutingOptimizer,
    workflow_intelligence: WorkflowIntelligence
)
```

**Learning Signals:**
- `orchestration_was_beneficial`: success + user_satisfaction > 0.7
- `multi_agent_synergy`: len(agent_sequence) > 1
- `parallel_efficiency`: From workflow metadata
- `orchestration_pattern`: Sequential, parallel, or conditional

**File:** `src/app/routing/unified_optimizer.py:23-260`

---

### 4. **ModalityOptimizer** (`modality_optimizer.py:71-769`)

Per-modality routing optimization with XGBoost meta-learning.

**Key Methods:**

```python
async def optimize_all_modalities(
    lookback_hours: int = 24,
    min_confidence: float = 0.7
) -> Dict[QueryModality, Dict[str, Any]]:
    """
    Evaluate and optimize all modalities (VIDEO, IMAGE, AUDIO, DOCUMENT)

    For each modality:
    1. Collect training examples from Phoenix spans
    2. Build ModelingContext
    3. Use XGBoost to decide: should_train?
    4. Select training strategy (SYNTHETIC/HYBRID/PURE_REAL)
    5. Train modality-specific DSPy module

    Returns optimization results per modality
    """

async def optimize_modality(
    modality: QueryModality,
    lookback_hours: int = 24,
    min_confidence: float = 0.7,
    force_training: bool = False
) -> Dict[str, Any]:
    """
    Optimize single modality:

    1. Collect training examples
    2. Build ModelingContext
    3. TrainingDecisionModel.should_train(context)
    4. TrainingStrategyModel.select_strategy(context)
    5. Prepare training data (synthetic/hybrid/real)
    6. Train ModalityRoutingModule (DSPy)
    7. Record training history

    Returns:
        {
            "modality": str,
            "trained": bool,
            "strategy": str,
            "examples_count": int,
            "expected_improvement": float,
            "training_result": dict
        }
    """

def predict_agent(
    query: str,
    modality: QueryModality,
    query_features: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Predict best agent using trained modality model

    Returns:
        {
            "recommended_agent": str,
            "confidence": float,
            "reasoning": str,
            "modality": str
        }
    or None if no model trained
    """

def get_optimization_summary(self) -> Dict[str, Any]:
    """
    Get summary of all modality optimizations

    Returns training history, meta-model status, last training details
    """
```

**XGBoost Meta-Models:**
```python
# 1. TrainingDecisionModel
should_train(context: ModelingContext) -> Tuple[bool, float]:
    """
    Decide whether to train based on:
    - real_sample_count (sufficient data?)
    - success_rate (performance degradation?)
    - days_since_last_training (stale model?)
    - current_performance_score
    - data_quality_score

    Returns: (should_train: bool, expected_improvement: float)
    """

# 2. TrainingStrategyModel
select_strategy(context: ModelingContext) -> TrainingStrategy:
    """
    Select strategy based on data availability:
    - SKIP: Not enough benefit
    - SYNTHETIC: < 20 real examples
    - HYBRID: 20-50 real examples (mix synthetic + real)
    - PURE_REAL: >= 50 real examples

    Returns: TrainingStrategy enum
    """
```

**Modality-Specific DSPy Module:**
```python
class ModalityRoutingSignature(dspy.Signature):
    query = dspy.InputField(desc="User query")
    modality = dspy.InputField(desc="Query modality (video, image, audio, document, text)")
    query_features = dspy.InputField(desc="Extracted query features as JSON")

    recommended_agent = dspy.OutputField(desc="Recommended agent")
    confidence = dspy.OutputField(desc="Confidence (0-1)")
    reasoning = dspy.OutputField(desc="Reasoning for routing choice")

class ModalityRoutingModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.route = dspy.ChainOfThought(ModalityRoutingSignature)
```

**Training:**
- Uses **MIPROv2** if ≥50 examples (metric-aware instruction optimization)
- Uses **BootstrapFewShot** if <50 examples (few-shot learning)
- Saves trained models per modality to `model_dir/{modality}_routing_module.json`

**File:** `src/app/routing/modality_optimizer.py:71-769`

---

### 5. **OptimizerCoordinator** (`optimizer_coordinator.py:25-207`)

Facade pattern for routing optimization requests to appropriate optimizers.

**Key Methods:**

```python
def optimize(
    type: OptimizationType,
    training_data: List[Dict[str, Any]],
    **kwargs
) -> Dict[str, Any]:
    """
    Route optimization request to appropriate optimizer:

    - ROUTING → AdvancedRoutingOptimizer
    - MODALITY → ModalityOptimizer
    - CROSS_MODAL → CrossModalOptimizer
    - UNIFIED → UnifiedOptimizer

    Returns optimization results
    """

def get_optimizer(
    type: OptimizationType
):
    """
    Get direct access to specific optimizer

    Use when you need optimizer-specific methods not exposed via coordinator
    """

def get_optimization_status(self) -> Dict[str, Any]:
    """
    Get status of all loaded optimizers

    Returns:
        {
            "tenant_id": str,
            "optimization_dir": str,
            "loaded_optimizers": List[str]
        }
    """
```

**Lazy Loading:**
```python
# Optimizers loaded on-demand to minimize memory usage
_get_routing_optimizer()     # AdvancedRoutingOptimizer
_get_modality_optimizer()    # ModalityOptimizer
_get_cross_modal_optimizer() # CrossModalOptimizer
_get_unified_optimizer()     # UnifiedOptimizer
```

**OptimizationType Enum:**
```python
class OptimizationType(Enum):
    ROUTING = "routing"          # AdvancedRoutingOptimizer
    MODALITY = "modality"        # ModalityOptimizer
    CROSS_MODAL = "cross_modal"  # CrossModalOptimizer
    UNIFIED = "unified"          # UnifiedOptimizer
    ORCHESTRATION = "orchestration"
```

**File:** `src/app/routing/optimizer_coordinator.py:25-207`

---

### 6. **RoutingOptimizer** (`optimizer.py:87-771`)

Base optimizer for routing strategies with auto-tuning.

**Key Methods:**

```python
def track_performance(
    query: str,
    predicted: RoutingDecision,
    actual: RoutingDecision | None = None,
    user_feedback: dict[str, Any] | None = None
):
    """
    Track routing performance for single query

    Triggers optimization if conditions met:
    - Time since last optimization > interval
    - Samples >= min_samples_for_optimization
    - Performance degradation detected
    """

async def optimize(self):
    """
    Run optimization process (to be overridden)

    Base implementation:
    - Calculates current metrics
    - Updates baseline if improved
    - Exports metrics to file
    """

def _calculate_current_metrics(self) -> OptimizationMetrics:
    """
    Calculate performance metrics from history:
    - Accuracy, precision, recall, F1 score
    - Average latency
    - Confidence correlation (alignment with success)
    - Error rate
    """
```

**AutoTuningOptimizer Subclass:**
```python
class AutoTuningOptimizer(RoutingOptimizer):
    """
    Auto-tuning for specific routing strategies:
    - GLiNER: Optimize threshold and labels
    - LLM: Optimize temperature, DSPy compilation
    - Keyword: Optimize keyword effectiveness
    - Composite: Optimize ensemble weights, confidence thresholds
    """

    async def _optimize_gliner(self):
        """Optimize GLiNER threshold (0.1-0.9 grid search)"""

    async def _optimize_llm(self):
        """Optimize LLM temperature or use DSPy BootstrapFewShot"""

    async def _optimize_keyword(self):
        """Analyze keyword effectiveness, update keyword lists"""

    async def _optimize_composite(self):
        """Optimize ensemble weights and confidence thresholds"""
```

**Configuration (OptimizationConfig):**
```python
@dataclass
class OptimizationConfig:
    # Triggers
    min_samples_for_optimization: int = 100
    optimization_interval_seconds: int = 3600  # 1 hour
    performance_degradation_threshold: float = 0.1  # 10% drop

    # Thresholds
    min_accuracy: float = 0.8
    min_precision: float = 0.75
    min_recall: float = 0.75
    max_acceptable_latency_ms: float = 100

    # Learning
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.01

    # DSPy
    dspy_enabled: bool = True
    dspy_max_bootstrapped_demos: int = 10
    dspy_max_labeled_demos: int = 50
    dspy_metric: str = "f1"

    # GLiNER
    gliner_threshold_optimization: bool = True
    gliner_label_optimization: bool = True
    gliner_threshold_step: float = 0.05

    # Storage
    max_history_size: int = 10000
    checkpoint_dir: Path = Path("outputs/routing_checkpoints")
    metrics_export_dir: Path = Path("outputs/routing_metrics")
```

**File:** `src/app/routing/optimizer.py:87-771`

---

## Usage Examples

### Example 1: Complete Optimization Orchestration (Production)

```python
from src.app.routing.optimization_orchestrator import OptimizationOrchestrator

# Initialize orchestrator with production config
orchestrator = OptimizationOrchestrator(
    tenant_id="production",
    span_eval_interval_minutes=15,      # Evaluate spans every 15 minutes
    annotation_interval_minutes=30,     # Identify spans for annotation every 30 minutes
    feedback_interval_minutes=15,       # Process feedback every 15 minutes
    confidence_threshold=0.6,           # Annotate spans with confidence < 0.6
    min_annotations_for_optimization=50, # Trigger optimization at 50 annotations
    optimization_improvement_threshold=0.05  # Accept if improvement > 5%
)

# Start continuous optimization (runs indefinitely)
await orchestrator.start()

# Get metrics
metrics = orchestrator.get_metrics()
print(f"Spans evaluated: {metrics['spans_evaluated']}")
print(f"Experiences created: {metrics['experiences_created']}")
print(f"Annotations completed: {metrics['annotations_completed']}")
print(f"Optimizations triggered: {metrics['optimizations_triggered']}")
```

---

### Example 2: Advanced Routing Optimizer with GRPO

```python
from src.app.routing.advanced_optimizer import (
    AdvancedRoutingOptimizer,
    AdvancedOptimizerConfig
)

# Configure advanced optimizer
config = AdvancedOptimizerConfig(
    optimizer_strategy="adaptive",  # Auto-select best optimizer
    learning_rate=0.001,
    batch_size=32,
    experience_replay_size=1000,
    min_experiences_for_training=50,
    exploration_epsilon=0.1,  # 10% exploration
    epsilon_decay=0.995,
    # Reward weights
    search_quality_weight=0.4,
    agent_success_weight=0.3,
    user_satisfaction_weight=0.3,
    processing_time_penalty=0.1
)

optimizer = AdvancedRoutingOptimizer(
    config=config,
    storage_dir="data/optimization"
)

# Record routing experience
reward = await optimizer.record_routing_experience(
    query="Show me videos where Marie Curie discusses radioactivity",
    entities=[{"text": "Marie Curie", "label": "person"}],
    relationships=[{"head": "Marie Curie", "relation": "discusses", "tail": "radioactivity"}],
    enhanced_query="Show me videos where Marie Curie discusses radioactivity in physics lectures",
    chosen_agent="video_search_agent",
    routing_confidence=0.85,
    search_quality=0.92,  # Quality of search results (0-1)
    agent_success=True,   # Agent completed successfully
    user_satisfaction=0.95,  # Explicit user feedback
    processing_time=1.2  # seconds
)

print(f"Computed reward: {reward:.3f}")  # 0.923

# Get routing recommendations (uses optimized policy)
recommendations = await optimizer.get_routing_recommendations(
    query="Find lecture videos about quantum mechanics",
    entities=[{"text": "quantum mechanics", "label": "topic"}],
    relationships=[]
)

print(f"Recommended agent: {recommendations['recommended_agent']}")  # video_search_agent
print(f"Confidence: {recommendations['confidence']:.2f}")  # 0.88
print(f"Reasoning: {recommendations['reasoning']}")
print(f"Optimization ready: {recommendations['optimization_ready']}")  # True
print(f"Training step: {recommendations['training_step']}")  # 15

# Get optimization status
status = optimizer.get_optimization_status()
print(f"Total experiences: {status['total_experiences']}")  # 523
print(f"Avg reward: {status['metrics']['avg_reward']}")  # 0.831
print(f"Success rate: {status['metrics']['success_rate']}")  # 0.89
print(f"Confidence accuracy: {status['metrics']['confidence_accuracy']}")  # 0.76
```

---

### Example 3: Modality-Specific Optimization with XGBoost

```python
from src.app.routing.modality_optimizer import ModalityOptimizer
from src.app.search.multi_modal_reranker import QueryModality

# Initialize modality optimizer
optimizer = ModalityOptimizer(
    tenant_id="production",
    model_dir=Path("outputs/models/modality"),
    vespa_client=vespa_client  # For synthetic data generation
)

# Optimize all modalities automatically
results = await optimizer.optimize_all_modalities(
    lookback_hours=24,      # Look back 24 hours for training data
    min_confidence=0.7      # Filter spans with confidence >= 0.7
)

for modality, result in results.items():
    print(f"\n{modality.value}:")
    if result["trained"]:
        print(f"  Strategy: {result['strategy']}")  # HYBRID, PURE_REAL, SYNTHETIC
        print(f"  Examples: {result['examples_count']}")  # 85
        print(f"  Expected improvement: {result['expected_improvement']:.3f}")  # 0.123
        print(f"  Training result: {result['training_result']['status']}")  # success
        print(f"  Validation accuracy: {result['training_result']['validation_accuracy']:.2f}")  # 0.92
    else:
        print(f"  Reason: {result['reason']}")  # insufficient_benefit

# Optimize specific modality with force training
video_result = await optimizer.optimize_modality(
    modality=QueryModality.VIDEO,
    lookback_hours=72,  # More data
    min_confidence=0.6,
    force_training=True  # Force training regardless of XGBoost decision
)

# Use trained model for predictions
prediction = optimizer.predict_agent(
    query="Find videos about deep learning tutorials",
    modality=QueryModality.VIDEO,
    query_features={
        "query_length": 35,
        "has_technical_terms": True,
        "routing_confidence": 0.78
    }
)

if prediction:
    print(f"Recommended agent: {prediction['recommended_agent']}")  # video_search_agent
    print(f"Confidence: {prediction['confidence']:.2f}")  # 0.91
    print(f"Reasoning: {prediction['reasoning']}")

# Get optimization summary
summary = optimizer.get_optimization_summary()
print(f"Total modalities trained: {summary['total_modalities']}")
print(f"Meta-model status: {summary['meta_models']}")
for modality, details in summary['modalities'].items():
    print(f"{modality}: {details['training_count']} trainings, last: {details['last_training']}")
```

---

### Example 4: Unified Optimizer (Bidirectional Learning)

```python
from src.app.routing.unified_optimizer import UnifiedOptimizer
from src.app.routing.advanced_optimizer import AdvancedRoutingOptimizer
from src.app.agents.workflow_intelligence import WorkflowIntelligence

# Initialize components
routing_optimizer = AdvancedRoutingOptimizer()
workflow_intelligence = WorkflowIntelligence()

# Create unified optimizer
unified_optimizer = UnifiedOptimizer(
    routing_optimizer=routing_optimizer,
    workflow_intelligence=workflow_intelligence
)

# Get successful workflows from orchestration
successful_workflows = workflow_intelligence.get_successful_workflows(
    min_quality=0.7,  # User satisfaction >= 0.7
    limit=100
)

# Integrate orchestration outcomes into routing optimization
integration_results = await unified_optimizer.integrate_orchestration_outcomes(
    successful_workflows
)

print(f"Workflows processed: {integration_results['workflows_processed']}")  # 100
print(f"Routing experiences created: {integration_results['routing_experiences_created']}")  # 87
print(f"Patterns learned: {integration_results['patterns_learned']}")
# {'parallel': 42, 'sequential': 28, 'conditional': 17}

# Run unified optimization cycle
optimization_results = await unified_optimizer.optimize_unified_policy()

print(f"Workflow optimization: {optimization_results['workflow_optimization']}")
print(f"Routing optimization: {optimization_results['routing_optimization']}")
print(f"Integration: {optimization_results['integration']}")
```

---

### Example 5: Optimizer Coordinator (Facade Pattern)

```python
from src.app.routing.optimizer_coordinator import (
    OptimizerCoordinator,
    OptimizationType
)

# Initialize coordinator
coordinator = OptimizerCoordinator(
    optimization_dir="optimization_results",
    tenant_id="production"
)

# Prepare training data
training_data = [
    {
        "query": "Find quantum physics lectures",
        "correct_agent": "video_search_agent",
        "entities": [{"text": "quantum physics", "label": "topic"}],
        "success": True,
        "user_satisfaction": 0.9
    },
    # ... more examples
]

# Route to appropriate optimizer automatically
routing_result = coordinator.optimize(
    type=OptimizationType.ROUTING,
    training_data=training_data
)

modality_result = coordinator.optimize(
    type=OptimizationType.MODALITY,
    training_data=training_data,
    modality="video"  # Required for modality optimization
)

cross_modal_result = coordinator.optimize(
    type=OptimizationType.CROSS_MODAL,
    training_data=training_data
)

# Get direct access to optimizer for advanced usage
routing_optimizer = coordinator.get_optimizer(OptimizationType.ROUTING)
recommendations = await routing_optimizer.get_routing_recommendations(
    query="Find lecture videos",
    entities=[],
    relationships=[]
)

# Get status of all optimizers
status = coordinator.get_optimization_status()
print(f"Loaded optimizers: {status['loaded_optimizers']}")
```

---

### Example 6: Single Optimization Cycle (Testing)

```python
from src.app.routing.optimization_orchestrator import OptimizationOrchestrator

# Initialize orchestrator
orchestrator = OptimizationOrchestrator(
    tenant_id="test",
    span_eval_interval_minutes=15,
    annotation_interval_minutes=30,
    feedback_interval_minutes=15
)

# Run single optimization cycle (non-blocking)
results = await orchestrator.run_once()

print(f"Span evaluation: {results['span_evaluation']}")
# {
#     "spans_processed": 45,
#     "experiences_created": 38,
#     "avg_confidence": 0.78
# }

print(f"Annotation requests: {results['annotation_requests']}")  # 12
print(f"Annotations generated: {results['annotations_generated']}")  # 10
print(f"Feedback loop: {results['feedback_loop']}")
# {
#     "annotations_processed": 10,
#     "experiences_updated": 10
# }
```

---

## Production Considerations

### 1. **Performance Optimization**

**Experience Replay Buffer:**
```python
# Configure for memory efficiency
config = AdvancedOptimizerConfig(
    experience_replay_size=1000,  # Limit memory usage
    batch_size=32,                # Balance training speed vs memory
    update_frequency=10           # Optimize every 10 experiences
)

# For high-volume systems, use sampling
batch_experiences = np.random.choice(
    experience_replay,
    size=min(batch_size, len(experience_replay)),
    replace=False
).tolist()
```

**Lazy Loading:**
```python
# OptimizerCoordinator lazy-loads optimizers to minimize memory
coordinator = OptimizerCoordinator()  # No optimizers loaded yet

# Optimizers loaded on first use
routing_optimizer = coordinator.get_optimizer(OptimizationType.ROUTING)  # Now loaded
```

**Asynchronous Optimization:**
```python
# Run optimization in background without blocking
if self._should_trigger_optimization():
    asyncio.create_task(self._run_optimization_step())  # Non-blocking
```

---

### 2. **Data Quality and Safety**

**Confidence Thresholds:**
```python
# Only use high-confidence spans for training
min_confidence = 0.7  # Adjust based on model calibration

# Filter low-quality experiences
high_quality_experiences = [
    exp for exp in experiences
    if exp.routing_confidence >= min_confidence
    and exp.search_quality >= 0.6
]
```

**Synthetic Data Control:**
```python
# Progressive strategies based on data availability
strategy = training_strategy_model.select_strategy(context)

# SYNTHETIC: Use only when real data < 20 examples
# HYBRID: Mix real + synthetic (1:1 ratio) for 20-50 examples
# PURE_REAL: Use only real data when >= 50 examples
```

**Experience Replay Diversity:**
```python
# Ensure diverse training batches (avoid overfitting to recent patterns)
batch_experiences = np.random.choice(
    experience_replay,  # Sample from historical buffer, not just recent
    size=batch_size,
    replace=False
)
```

---

### 3. **Multi-Tenant Isolation**

**Tenant-Specific Optimization:**
```python
# Each tenant has isolated optimization state
optimizer_tenant_a = AdvancedRoutingOptimizer(
    storage_dir="data/optimization/tenant_a"
)

optimizer_tenant_b = AdvancedRoutingOptimizer(
    storage_dir="data/optimization/tenant_b"
)

# Separate experience storage per tenant
orchestrator_a = OptimizationOrchestrator(tenant_id="tenant_a")
orchestrator_b = OptimizationOrchestrator(tenant_id="tenant_b")
```

**Shared vs Tenant-Specific Models:**
```python
# Option 1: Tenant-specific models (better personalization)
modality_optimizer = ModalityOptimizer(
    tenant_id="tenant_a",
    model_dir=Path(f"outputs/models/modality/tenant_a")
)

# Option 2: Shared models (faster cold start, less personalization)
shared_modality_optimizer = ModalityOptimizer(
    tenant_id="shared",
    model_dir=Path("outputs/models/modality/shared")
)
```

---

### 4. **Monitoring and Observability**

**Optimization Metrics Tracking:**
```python
# OptimizationOrchestrator provides comprehensive metrics
metrics = orchestrator.get_metrics()

# Key metrics to monitor:
- spans_evaluated: Total spans processed
- experiences_created: Routing experiences generated
- annotations_requested: Low-quality spans identified
- annotations_completed: Annotations generated
- optimizations_triggered: Number of optimization runs
- total_improvement: Cumulative performance improvement

# Alert on anomalies:
if metrics["optimizations_triggered"] == 0 and uptime > 24h:
    logger.warning("No optimizations triggered in 24h - check thresholds")

if metrics["annotations_completed"] / metrics["annotations_requested"] < 0.3:
    logger.warning("Low annotation success rate - check LLM availability")
```

**Performance Degradation Detection:**
```python
# Automatic triggering on performance degradation
config = AdvancedOptimizerConfig(
    performance_degradation_threshold=0.1  # 10% accuracy drop triggers optimization
)

# Monitor baseline vs current performance
current_metrics = optimizer._calculate_current_metrics()
if optimizer.baseline_metrics:
    degradation = optimizer.baseline_metrics.accuracy - current_metrics.accuracy
    if degradation > 0.1:
        logger.warning(f"Performance degradation: {degradation:.2%}")
        await optimizer.optimize()  # Auto-trigger optimization
```

**Logging Best Practices:**
```python
# Structured logging for observability
logger.info(
    f"GRPO optimization step {self.training_step} complete",
    extra={
        "optimizer": "AdvancedRoutingOptimizer",
        "training_step": self.training_step,
        "epsilon": self.current_epsilon,
        "experiences_count": len(self.experiences),
        "avg_reward": self.metrics.avg_reward
    }
)
```

---

### 5. **Production Deployment**

**Continuous Optimization Service:**
```python
# Deploy as long-running service
async def main():
    orchestrator = OptimizationOrchestrator(
        tenant_id="production",
        span_eval_interval_minutes=15,
        annotation_interval_minutes=30,
        feedback_interval_minutes=15,
        min_annotations_for_optimization=50
    )

    try:
        await orchestrator.start()  # Runs indefinitely
    except KeyboardInterrupt:
        logger.info("Shutting down optimization orchestrator")
    except Exception as e:
        logger.error(f"Orchestrator failed: {e}")
        # Restart with exponential backoff
        await asyncio.sleep(60)
        await main()

if __name__ == "__main__":
    asyncio.run(main())
```

**Graceful Shutdown:**
```python
# Save state before shutdown
async def shutdown():
    logger.info("Saving optimization state before shutdown...")
    await optimizer._persist_data()
    optimizer.save_checkpoint()
    logger.info("Optimization state saved")
```

**Health Checks:**
```python
# Expose health check endpoint
def get_health() -> Dict[str, Any]:
    status = optimizer.get_optimization_status()

    is_healthy = (
        status["optimizer_ready"]
        and status["total_experiences"] > 50
        and status["metrics"]["avg_reward"] > 0.5
    )

    return {
        "healthy": is_healthy,
        "optimizer_ready": status["optimizer_ready"],
        "total_experiences": status["total_experiences"],
        "avg_reward": status["metrics"]["avg_reward"],
        "last_updated": status["metrics"]["last_updated"]
    }
```

---

### 6. **Error Handling and Recovery**

**Fallback Strategies:**
```python
# Graceful degradation when optimization fails
try:
    optimized_prediction = await optimizer.optimize_routing_decision(
        query, entities, relationships, enhanced_query, baseline_prediction
    )
except Exception as e:
    logger.error(f"Optimization failed: {e}, using baseline")
    optimized_prediction = baseline_prediction  # Fallback to baseline
```

**Checkpoint and Recovery:**
```python
# Save checkpoints periodically
optimizer.save_checkpoint(filepath=Path("checkpoints/optimizer_20250107.json"))

# Restore from checkpoint after crash
optimizer.load_checkpoint(filepath=Path("checkpoints/optimizer_20250107.json"))
```

**Experience Persistence:**
```python
# Auto-persist every 10 experiences
if len(self.experiences) % 10 == 0:
    await self._persist_data()

# Load on startup
def _load_stored_data(self):
    experience_file = self.storage_dir / self.config.experience_file
    if experience_file.exists():
        with open(experience_file, "rb") as f:
            self.experiences = pickle.load(f)
        logger.info(f"Loaded {len(self.experiences)} routing experiences")
```

---

## Testing

### Test Files

**Optimization Orchestrator:**
- Location: `tests/routing/integration/test_optimization_orchestrator_integration.py`
- Focus: End-to-end optimization pipeline, span evaluation, annotation workflow
- Key Tests:
  - `test_optimization_orchestrator_initialization`
  - `test_run_once_optimization_cycle`
  - `test_continuous_optimization_loop`

**Advanced Routing Optimizer:**
- Location: `tests/routing/unit/test_advanced_routing_optimizer.py`
- Focus: GRPO optimization, experience replay, multi-stage optimizer selection
- Key Tests:
  - `test_record_routing_experience`
  - `test_multi_stage_optimizer_selection`
  - `test_grpo_optimization_step`
  - `test_confidence_calibration`

**Modality Optimizer:**
- Location: `tests/routing/unit/test_modality_optimizer.py`
- Focus: Per-modality optimization, XGBoost meta-learning, synthetic data
- Key Tests:
  - `test_optimize_modality_with_synthetic_data`
  - `test_xgboost_training_decision_model`
  - `test_modality_model_training`
  - `test_predict_agent_with_trained_model`

**Unified Optimizer:**
- Location: `tests/routing/integration/test_unified_optimizer_integration.py`
- Focus: Bidirectional learning between routing and orchestration
- Key Tests:
  - `test_integrate_orchestration_outcomes`
  - `test_unified_policy_optimization`
  - `test_workflow_to_routing_experience_conversion`

**Optimizer Coordinator:**
- Location: `tests/routing/unit/test_optimizer_coordinator.py`
- Focus: Facade pattern, lazy loading, optimizer routing
- Key Tests:
  - `test_optimizer_coordination_routing`
  - `test_lazy_loading_optimizers`
  - `test_get_optimization_status`

---

### Test Scenarios

**1. Multi-Stage Optimizer Selection:**
```python
def test_multi_stage_optimizer_selection():
    """Test that correct optimizer is selected based on dataset size"""
    config = AdvancedOptimizerConfig(optimizer_strategy="adaptive")
    optimizer = AdvancedRoutingOptimizer(config=config)

    # Small dataset → BootstrapFewShot
    small_trainset = [create_example() for _ in range(15)]
    info = optimizer.advanced_optimizer.get_optimization_info(len(small_trainset))
    assert info["primary_optimizer"] == "bootstrap"

    # Medium dataset → SIMBA
    medium_trainset = [create_example() for _ in range(60)]
    info = optimizer.advanced_optimizer.get_optimization_info(len(medium_trainset))
    assert info["primary_optimizer"] == "simba"

    # Large dataset → GEPA
    large_trainset = [create_example() for _ in range(250)]
    info = optimizer.advanced_optimizer.get_optimization_info(len(large_trainset))
    assert info["primary_optimizer"] == "gepa"
```

**2. Reward Signal Computation:**
```python
@pytest.mark.asyncio
async def test_reward_signal_computation():
    """Test reward computation from routing outcomes"""
    optimizer = AdvancedRoutingOptimizer()

    # High-quality routing
    reward_high = await optimizer.record_routing_experience(
        query="test query",
        entities=[],
        relationships=[],
        enhanced_query="test query",
        chosen_agent="video_search_agent",
        routing_confidence=0.9,
        search_quality=0.95,
        agent_success=True,
        user_satisfaction=0.92,
        processing_time=0.5
    )

    assert reward_high > 0.85  # High reward for good performance

    # Low-quality routing
    reward_low = await optimizer.record_routing_experience(
        query="test query",
        entities=[],
        relationships=[],
        enhanced_query="test query",
        chosen_agent="video_search_agent",
        routing_confidence=0.3,
        search_quality=0.4,
        agent_success=False,
        user_satisfaction=0.3,
        processing_time=5.0
    )

    assert reward_low < 0.4  # Low reward for poor performance
```

**3. XGBoost Meta-Model Training:**
```python
@pytest.mark.asyncio
async def test_xgboost_meta_model_training():
    """Test XGBoost meta-models for training decisions"""
    optimizer = ModalityOptimizer(tenant_id="test")

    # Create modeling contexts
    contexts = [
        ModelingContext(
            modality=QueryModality.VIDEO,
            real_sample_count=100,
            success_rate=0.85,
            days_since_last_training=30
        ),
        ModelingContext(
            modality=QueryModality.VIDEO,
            real_sample_count=10,
            success_rate=0.6,
            days_since_last_training=5
        )
    ]

    # Train decision model
    optimizer.training_decision_model.train(contexts, targets=[True, False])

    # Test predictions
    should_train_1, improvement_1 = optimizer.training_decision_model.should_train(contexts[0])
    assert should_train_1 == True  # High sample count, long time since training

    should_train_2, improvement_2 = optimizer.training_decision_model.should_train(contexts[1])
    assert should_train_2 == False  # Low sample count, recent training
```

**4. Integration Test - Complete Optimization Cycle:**
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_optimization_cycle():
    """Test end-to-end optimization orchestration"""
    orchestrator = OptimizationOrchestrator(
        tenant_id="test",
        span_eval_interval_minutes=1,
        annotation_interval_minutes=1,
        feedback_interval_minutes=1,
        min_annotations_for_optimization=5
    )

    # Run single cycle
    results = await orchestrator.run_once()

    # Verify span evaluation
    assert results["span_evaluation"]["spans_processed"] >= 0
    assert results["span_evaluation"]["experiences_created"] >= 0

    # Verify annotation workflow
    assert results["annotation_requests"] >= 0

    # Verify feedback loop
    assert results["feedback_loop"]["annotations_processed"] >= 0

    # Verify metrics updated
    metrics = orchestrator.get_metrics()
    assert metrics["spans_evaluated"] > 0
```

---

**Coverage:**
- **Unit tests**: 95%+ coverage of optimizer logic, reward computation, meta-models
- **Integration tests**: Complete optimization cycles, span evaluation, modality optimization
- **Performance tests**: Optimization speed, memory usage, convergence rates
- **Error handling tests**: Graceful degradation, checkpoint recovery, fallback strategies

---

## DSPy Training Data Requirements

### Overview

DSPy optimizers require properly formatted training examples with **all expected output fields** defined. Missing fields will cause `AttributeError: 'Example' object has no attribute 'field_name'` during metric evaluation.

### Training Data Format

Each DSPy `Example` must include:
1. **Input fields** (marked with `.with_inputs()`)
2. **All output fields** that metrics will access

**Example Structure:**
```python
import dspy

example = dspy.Example(
    # Input fields
    query="user query here",
    context="optional context",

    # Output fields (ALL fields that metrics check must be present)
    primary_intent="search",
    confidence=0.9,
    recommended_agent="video_search"
).with_inputs("query", "context")  # Specify which fields are inputs
```

### Query Analysis Training Data

**Required Output Fields:**
- `primary_intent`: Main intent category
- `complexity_level`: "simple" | "complex"
- `needs_video_search`: "true" | "false"
- `needs_text_search`: "true" | "false"
- `multimodal_query`: "true" | "false"
- `temporal_pattern`: Temporal info or "none"

**Example:**
```python
training_data = [
    dspy.Example(
        query="Show me videos of robots from yesterday",
        context="",
        # All output fields required for metrics
        primary_intent="video_search",
        complexity_level="simple",
        needs_video_search="true",
        needs_text_search="false",
        multimodal_query="false",
        temporal_pattern="yesterday",
    ).with_inputs("query", "context"),

    dspy.Example(
        query="Compare research papers on deep learning",
        context="academic",
        primary_intent="analysis",
        complexity_level="complex",
        needs_video_search="false",
        needs_text_search="true",
        multimodal_query="false",
        temporal_pattern="none",
    ).with_inputs("query", "context"),
]
```

### Agent Routing Training Data

**Required Output Fields:**
- `recommended_workflow`: Workflow type
- `primary_agent`: Main agent to use
- `routing_confidence`: Confidence score (0.0-1.0 as string)

**Example:**
```python
training_data = [
    dspy.Example(
        query="Show me videos",
        analysis_result="simple search",
        available_agents="video_search",
        # All output fields
        recommended_workflow="direct_search",
        primary_agent="video_search",
        routing_confidence="0.9",
    ).with_inputs("query", "analysis_result", "available_agents"),

    dspy.Example(
        query="Analyze data trends",
        analysis_result="complex analysis",
        available_agents="detailed_report",
        recommended_workflow="detailed_analysis",
        primary_agent="detailed_report",
        routing_confidence="0.85",
    ).with_inputs("query", "analysis_result", "available_agents"),
]
```

### Common Errors

#### Missing Output Fields

**Error:**
```python
AttributeError: 'Example' object has no attribute 'primary_intent'
```

**Cause:** Metric function accesses `example.primary_intent` but Example doesn't have that field.

**Fix:** Add the missing field to all training examples:
```python
example = dspy.Example(
    query="...",
    primary_intent="search",  # ← Add missing field
    # ... other fields
).with_inputs("query")
```

#### Incorrect Field Types

**Error:**
```python
TypeError: expected str, got bool
```

**Cause:** DSPy Examples store all fields as strings internally.

**Fix:** Convert to strings:
```python
# ❌ Bad
needs_video_search=True

# ✅ Good
needs_video_search="true"
```

### Validation

Before running optimization, validate your training data:

```python
def validate_training_data(examples, required_fields):
    """Validate all examples have required output fields."""
    for i, ex in enumerate(examples):
        for field in required_fields:
            if not hasattr(ex, field):
                raise ValueError(
                    f"Example {i} missing required field '{field}'"
                )
    print(f"✅ All {len(examples)} examples valid")

# Usage
required = ["primary_intent", "complexity_level", "needs_video_search"]
validate_training_data(training_data, required)
```

### Best Practices

1. **Define all output fields upfront** - Check what your metrics access
2. **Use consistent field names** - Match your DSPy signature output fields
3. **Validate before optimization** - Catch missing fields early
4. **Use string values** - DSPy converts everything to strings
5. **Document required fields** - Keep a reference list for your team

### File References

- `src/app/agents/dspy_agent_optimizer.py:327-385` - Training data loading
- `tests/agents/integration/test_dspy_optimization_integration.py` - Example tests with proper format

---

## Related Documentation

- **Routing Module Study Guide**: `02_ROUTING_MODULE.md` - Tiered routing strategies
- **Agents Module Study Guide**: `01_AGENTS_MODULE.md` - RoutingAgent integration
- **Telemetry Module Study Guide**: `05_TELEMETRY_MODULE.md` - Phoenix span collection
- **Evaluation Module Study Guide**: `06_EVALUATION_MODULE.md` - RoutingEvaluator

---

**Next Steps:**
1. Review DSPy 3.0 documentation for GEPA, MIPROv2, SIMBA optimizers
2. Experiment with different optimizer strategies (adaptive vs forced)
3. Monitor optimization metrics in production (avg_reward, success_rate, improvement_rate)
4. Tune reward weights for your use case (search_quality_weight, agent_success_weight)
5. Test synthetic data generation for modality optimization cold start

---

**File References:**
- `src/app/routing/optimization_orchestrator.py` - Complete optimization pipeline
- `src/app/routing/advanced_optimizer.py:136-1273` - GRPO optimization with multi-stage DSPy
- `src/app/routing/unified_optimizer.py:23-260` - Bidirectional routing + orchestration learning
- `src/app/routing/modality_optimizer.py:71-769` - Per-modality optimization with XGBoost
- `src/app/routing/optimizer_coordinator.py:25-207` - Facade for optimizer routing
- `src/app/routing/optimizer.py:87-771` - Base optimizer with auto-tuning
