# Optimization & Learning System

## Overview

Cogniverse uses **DSPy-powered optimization** to continuously improve routing decisions through experience replay and multi-stage optimization. The system automatically selects the best optimizer (Bootstrap → SIMBA → MIPRO → GEPA) based on dataset size.

## GEPA Optimizer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Experience Collection                      │
│  ┌────────────────────────────────────────────────┐         │
│  │ Routing Decision + Outcome                     │         │
│  │ - Query analysis results                       │         │
│  │ - Chosen agent + confidence                    │         │
│  │ - Search quality score                         │         │
│  │ - Processing time                              │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                   Experience Replay Buffer                   │
│  ┌────────────────────────────────────────────────┐         │
│  │ Circular buffer with max 1000 experiences      │         │
│  │ - Temporal ordering maintained                 │         │
│  │ - Reward computation on storage                │         │
│  │ - Sampling strategies for training             │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                   Optimizer Selection                        │
│  ┌────────────────────────────────────────────────┐         │
│  │ if examples < 10:   Bootstrap                  │         │
│  │ if 10 ≤ examples < 50:   SIMBA                │         │
│  │ if 50 ≤ examples < 200:  MIPRO                │         │
│  │ if examples ≥ 200:       GEPA                 │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                   Policy Optimization                        │
│  ┌────────────────────────────────────────────────┐         │
│  │ DSPy compile() with selected optimizer         │         │
│  │ - Update routing policy parameters             │         │
│  │ - Calibrate confidence scores                  │         │
│  │ - Validate on held-out set                     │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                   Deployment                                 │
│  ┌────────────────────────────────────────────────┐         │
│  │ - Save optimized policy                        │         │
│  │ - Update routing agent                         │         │
│  │ - Monitor performance metrics                  │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Optimization Pipeline

### 1. Experience Collection

Every routing decision creates an experience:

```python
from src.app.routing.advanced_optimizer import RoutingExperience

experience = RoutingExperience(
    query="Find videos about machine learning",
    query_analysis={
        "intent": "search",
        "complexity": "moderate",
        "entities": [{"type": "TOPIC", "text": "machine learning"}],
        "requirements": {"video_search": True}
    },
    chosen_agent="video_search_agent",
    confidence=0.92,
    search_quality_score=0.85,  # Computed from result relevance
    processing_time_ms=120,
    timestamp=datetime.now()
)

# Add to buffer
optimizer.add_experience(experience)
```

### 2. Reward Computation

Reward signal combines multiple factors:

```python
def compute_reward(experience: RoutingExperience) -> float:
    """
    Reward = weighted sum of:
    - Search quality score (0.5 weight)
    - Confidence calibration (0.3 weight)
    - Efficiency (0.2 weight)
    """
    quality_reward = experience.search_quality_score * 0.5

    # Calibration: penalize overconfident wrong decisions
    calibration_reward = (
        experience.confidence
        if experience.search_quality_score > 0.7
        else -experience.confidence
    ) * 0.3

    # Efficiency: faster is better (normalized)
    efficiency_reward = (1 - min(experience.processing_time_ms / 1000, 1)) * 0.2

    return quality_reward + calibration_reward + efficiency_reward
```

### 3. Optimizer Selection Logic

```python
def select_optimizer(num_experiences: int) -> str:
    """Select best optimizer based on dataset size"""
    if num_experiences < 10:
        return "bootstrap"  # Few-shot learning
    elif num_experiences < 50:
        return "simba"      # Similarity-based
    elif num_experiences < 200:
        return "mipro"      # Information-theoretic
    else:
        return "gepa"       # Gradient-based (best)
```

**Optimizer Characteristics:**

| Optimizer | Examples Needed | Training Time | Quality | Use Case |
|-----------|----------------|---------------|---------|----------|
| Bootstrap | < 10 | Instant | Low | Cold start |
| SIMBA | 10-50 | ~1 min | Medium | Early learning |
| MIPRO | 50-200 | ~5 min | Good | Active learning |
| GEPA | 200+ | ~15 min | Best | Production |

### 4. Policy Compilation

```python
from src.app.routing.advanced_optimizer import AdvancedRoutingOptimizer
import dspy

# Initialize optimizer
optimizer = AdvancedRoutingOptimizer(
    routing_policy=routing_policy,
    config=config
)

# Compile with selected optimizer
optimized_policy = optimizer.compile(
    routing_policy=routing_policy,
    trainset=training_examples,
    max_bootstrapped_demos=4,
    max_labeled_demos=8
)

# Update routing agent
routing_agent.update_policy(optimized_policy)
```

### 5. Confidence Calibration

After optimization, calibrate confidence scores:

```python
from sklearn.calibration import calibration_curve

# Get predictions on validation set
confidences = []
accuracies = []

for example in validation_set:
    prediction = optimized_policy(example.query_analysis)
    confidences.append(prediction.confidence)
    accuracies.append(1 if prediction.chosen_agent == example.ground_truth else 0)

# Calibrate
calibrated_model = CalibratedClassifierCV(
    optimized_policy,
    method='isotonic'
).fit(confidences, accuracies)
```

## Workflow Intelligence

### Bidirectional Learning

```
┌─────────────────────────────────────────────────────┐
│               Routing Decision                      │
│  "Choose video_search_agent for query"             │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│          Agent Execution & Outcome                  │
│  Success: High relevance results                    │
│  Failure: Poor results, timeout, error              │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│            Workflow Pattern Extraction              │
│  - Agent sequence that led to success               │
│  - Query patterns that work well                    │
│  - Failure patterns to avoid                        │
└─────────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│         Feedback to Routing Optimizer               │
│  Update routing policy with workflow insights       │
└─────────────────────────────────────────────────────┘
```

### Pattern Learning

```python
from src.app.routing.workflow_intelligence import WorkflowIntelligence

workflow_intel = WorkflowIntelligence()

# Learn from successful workflows
workflow_intel.record_workflow(
    query_pattern="video search for [TOPIC]",
    agent_sequence=["routing_agent", "video_search_agent"],
    outcome_quality=0.92,
    context={
        "complexity": "moderate",
        "entities": ["TOPIC"]
    }
)

# Extract patterns
patterns = workflow_intel.get_learned_patterns(
    min_occurrences=5,
    min_quality=0.8
)
# Returns: [
#   {
#     "pattern": "video search for TOPIC",
#     "recommended_sequence": ["routing_agent", "video_search_agent"],
#     "confidence": 0.95
#   }
# ]
```

## Monitoring & Metrics

### Optimization Metrics

Track optimizer performance over time:

```python
from src.app.routing.advanced_optimizer import get_optimizer_metrics

metrics = get_optimizer_metrics()
# {
#   "total_experiences": 245,
#   "current_optimizer": "gepa",
#   "optimization_runs": 12,
#   "avg_reward": 0.78,
#   "routing_accuracy": 0.89,
#   "confidence_calibration_error": 0.05,
#   "last_optimization": "2025-10-04T10:30:00Z"
# }
```

### Performance Dashboard

View optimization progress in Phoenix:

1. Navigate to experiment: `routing-optimization-{date}`
2. View metrics:
   - Routing accuracy over time
   - Confidence calibration curve
   - Reward distribution
   - Agent selection frequencies

### A/B Testing

Compare optimized vs baseline policies:

```python
from src.evaluation.routing_evaluator import RoutingEvaluator

evaluator = RoutingEvaluator()

# Run A/B test
results = evaluator.compare_policies(
    policy_a=baseline_policy,
    policy_b=optimized_policy,
    test_set=test_queries,
    metrics=["accuracy", "confidence_calibration", "latency"]
)

# Results:
# {
#   "policy_a": {"accuracy": 0.75, "calibration": 0.12, "latency_ms": 95},
#   "policy_b": {"accuracy": 0.89, "calibration": 0.05, "latency_ms": 105},
#   "winner": "policy_b",
#   "improvement": "+18.7% accuracy"
# }
```

## Running Optimization

### Manual Optimization

```bash
# Run optimization cycle
uv run python scripts/optimize_routing.py \
  --min-experiences 200 \
  --optimizer gepa \
  --validation-split 0.2 \
  --save-policy outputs/optimized_policy.json
```

### Automatic Optimization

Configure automatic optimization triggers:

```python
# In config
{
  "routing": {
    "optimization": {
      "enabled": true,
      "trigger_every_n_experiences": 10,
      "min_experiences_for_gepa": 200,
      "auto_deploy": true,
      "validation_threshold": 0.85
    }
  }
}
```

System will automatically:
1. Collect experiences
2. Trigger optimization at threshold
3. Validate optimized policy
4. Deploy if validation accuracy > threshold

### Optimization Schedule

```python
# Cron-based optimization (production)
# Run every night at 2 AM
0 2 * * * cd /app && uv run python scripts/optimize_routing.py --auto
```

## Troubleshooting

### GEPA Not Executing

**Symptom**: Optimizer stays on Bootstrap/SIMBA

**Cause**: Not enough experiences collected

**Solution**:
```python
# Check experience count
from src.app.routing.advanced_optimizer import get_optimizer

optimizer = get_optimizer()
print(f"Experiences: {optimizer.get_experience_count()}")
print(f"Current optimizer: {optimizer.get_current_optimizer()}")

# Need 200+ for GEPA
if optimizer.get_experience_count() < 200:
    print("Collect more experiences to trigger GEPA")
```

### Low Optimization Quality

**Symptom**: Optimized policy performs worse than baseline

**Causes**:
1. Noisy training data
2. Insufficient validation
3. Overfitting

**Solutions**:
```python
# 1. Filter low-quality experiences
optimizer.filter_experiences(min_quality=0.6)

# 2. Increase validation set size
optimizer.set_validation_split(0.3)

# 3. Add regularization
optimizer.compile(
    routing_policy=policy,
    trainset=examples,
    regularization=0.01  # L2 regularization
)
```

### Confidence Miscalibration

**Symptom**: High confidence on wrong predictions

**Solution**: Recalibrate confidence scores
```python
from src.app.routing.calibration import calibrate_confidence

calibrated_policy = calibrate_confidence(
    policy=optimized_policy,
    calibration_set=validation_examples,
    method='isotonic'  # or 'sigmoid'
)
```

## Best Practices

### Experience Collection

1. **Record all decisions**: Even if confidence is low
2. **Accurate reward signals**: Use real search quality metrics
3. **Diverse queries**: Ensure training data covers query space
4. **Temporal ordering**: Maintain chronological order in buffer

### Optimization Frequency

- **Development**: Optimize after every 10 experiences
- **Staging**: Optimize daily with 50+ new experiences
- **Production**: Optimize weekly with 200+ new experiences

### Validation Strategy

Always validate optimized policies before deployment:

```python
# Holdout validation
train_set = experiences[:800]
val_set = experiences[800:]

optimized_policy = optimizer.compile(routing_policy, train_set)

# Validate
accuracy = evaluate_policy(optimized_policy, val_set)

# Deploy only if better
if accuracy > current_policy_accuracy:
    deploy_policy(optimized_policy)
```

### Monitoring

Set up alerts for:
- Routing accuracy drops below 0.80
- Confidence calibration error > 0.10
- Optimization failures
- Experience buffer overflow

## Advanced Features

### Multi-Objective Optimization

Optimize for multiple objectives:

```python
optimizer.compile(
    routing_policy=policy,
    trainset=examples,
    objectives={
        "accuracy": 0.6,      # 60% weight
        "latency": 0.2,       # 20% weight
        "cost": 0.2           # 20% weight
    }
)
```

### Transfer Learning

Transfer knowledge between tenants:

```python
# Train on tenant A data
policy_a = optimizer.compile(policy, tenant_a_experiences)

# Fine-tune on tenant B (transfer learning)
policy_b = optimizer.compile(
    policy_a,  # Start from policy_a
    tenant_b_experiences,
    learning_rate=0.01  # Lower LR for fine-tuning
)
```

### Online Learning

Continuous learning in production:

```python
# Enable online learning
optimizer.enable_online_learning(
    update_every_n_experiences=5,
    learning_rate=0.001,
    momentum=0.9
)

# Policy updates automatically as new experiences arrive
```

## Related Documentation

- [Architecture Overview](architecture.md) - System architecture
- [Agent Orchestration](agent-orchestration.md) - Multi-agent coordination
- [Phoenix Integration](phoenix-integration.md) - Experiment tracking

**Last Updated**: 2025-10-04
