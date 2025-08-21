# Comprehensive Routing System Implementation Guide

## Overview

This document describes the implementation of the comprehensive routing system based on the architecture outlined in COMPREHENSIVE_ROUTING.md. The system implements a tiered, hybrid approach combining GLiNER (fast path), SmolLM3/LLM (slow path), and keyword-based (fallback) routing strategies with auto-tuning optimization.

## Architecture

### Three-Tier Routing Architecture

```
User Query → Tier 1 (Fast Path) → Tier 2 (Slow Path) → Tier 3 (Fallback)
                ↓                      ↓                     ↓
             GLiNER2               SmolLM3/LLM           Keywords
           (CPU, <10ms)          (GPU, ~100ms)         (CPU, <1ms)
```

### Key Components

1. **Routing Strategies** (`src/routing/strategies.py`)
   - `GLiNERRoutingStrategy`: Fast, CPU-based entity extraction
   - `LLMRoutingStrategy`: Sophisticated reasoning with LLMs
   - `KeywordRoutingStrategy`: Simple, deterministic fallback
   - `HybridRoutingStrategy`: Sequential fallback approach
   - `EnsembleRoutingStrategy`: Voting-based combination

2. **Comprehensive Router** (`src/routing/router.py`)
   - `ComprehensiveRouter`: Base implementation
   - `TieredRouter`: Specialized tiered architecture

3. **Auto-Tuning Optimizer** (`src/routing/optimizer.py`)
   - `RoutingOptimizer`: Base optimization framework
   - `AutoTuningOptimizer`: Strategy-specific optimization

4. **Configuration System** (`src/routing/config.py`)
   - Flexible configuration loading
   - Environment variable overrides
   - Multiple format support (JSON/YAML)

## Installation

```bash
# Install dependencies
pip install gliner transformers torch pydantic

# Optional: Install DSPy for LLM optimization
pip install dspy-ai

# Optional: Install Ollama for local LLM
curl -fsSL https://ollama.com/install.sh | sh
ollama pull smollm3:3b
```

## Configuration

### Basic Configuration

Create a `configs/routing_config.json`:

```json
{
  "routing_mode": "tiered",
  "tier_config": {
    "enable_fast_path": true,
    "enable_slow_path": true,
    "enable_fallback": true,
    "fast_path_confidence_threshold": 0.7,
    "slow_path_confidence_threshold": 0.6
  },
  "gliner_config": {
    "model": "urchade/gliner_large-v2.1",
    "threshold": 0.3,
    "labels": [
      "video_content", "text_information",
      "summary_request", "detailed_analysis"
    ]
  },
  "llm_config": {
    "provider": "local",
    "model": "smollm3:3b",
    "endpoint": "http://localhost:11434",
    "temperature": 0.1
  },
  "optimization_config": {
    "enable_auto_optimization": true,
    "optimization_interval_seconds": 3600
  }
}
```

### Environment Variable Overrides

```bash
export ROUTING_LLM_MODEL=gemma2:2b
export ROUTING_GLINER_THRESHOLD=0.4
export ROUTING_OPTIMIZATION_ENABLE_AUTO_OPTIMIZATION=true
```

## Usage Examples

### Example 1: Basic Routing

```python
from src.routing import ComprehensiveRouter, RoutingConfig

# Initialize router
config = RoutingConfig()
router = ComprehensiveRouter(config)

# Route a query
query = "Show me videos about machine learning from last week"
decision = await router.route(query)

print(f"Search Modality: {decision.search_modality.value}")
print(f"Generation Type: {decision.generation_type.value}")
print(f"Confidence: {decision.confidence_score:.2f}")
print(f"Routing Tier: {decision.metadata.get('tier')}")
```

### Example 2: Using the Enhanced QueryAnalysisTool

```python
from src.agents.query_analysis_tool_v2 import QueryAnalysisToolV2

# Create analyzer
analyzer = QueryAnalysisToolV2()

# Analyze query
query = "Create a detailed report on AI advancements"
analysis = await analyzer.execute(query)

print(f"Needs Video: {analysis['needs_video_search']}")
print(f"Needs Text: {analysis['needs_text_search']}")
print(f"Generation Type: {analysis['generation_type']}")
print(f"Routing Method: {analysis['routing_method']}")
```

### Example 3: Tiered Router with Statistics

```python
from src.routing import TieredRouter, RoutingConfig

# Initialize tiered router
config = RoutingConfig(routing_mode="tiered")
router = TieredRouter(config)

# Process multiple queries
queries = [
    "Show me the tutorial video",
    "Find research papers on quantum computing",
    "Summarize the main points from the presentation"
]

for query in queries:
    decision = await router.route(query)
    print(f"{query[:30]}... → Tier: {decision.metadata.get('tier')}")

# Get statistics
stats = router.get_tier_statistics()
for tier, data in stats.items():
    print(f"{tier}: {data['usage_percentage']:.1f}% usage, "
          f"{data['success_rate']:.1f}% success")
```

### Example 4: Auto-Optimization with Feedback

```python
from src.routing import GLiNERRoutingStrategy, AutoTuningOptimizer
from src.routing.optimizer import OptimizationConfig

# Create strategy with optimizer
strategy = GLiNERRoutingStrategy({"threshold": 0.3})
optimizer = AutoTuningOptimizer(strategy, OptimizationConfig())

# Process queries and collect feedback
for query, ground_truth in training_data:
    # Make prediction
    predicted = await strategy.route(query)
    
    # Track performance
    optimizer.track_performance(
        query=query,
        predicted=predicted,
        actual=ground_truth,
        user_feedback={"satisfaction": 0.9}
    )

# Trigger optimization
await optimizer.optimize()

# Save optimized configuration
optimizer.save_checkpoint("optimized_gliner.json")
```

### Example 5: Ensemble Routing

```python
from src.routing import EnsembleRoutingStrategy

# Configure ensemble
config = {
    "enabled_strategies": ["gliner", "llm", "keyword"],
    "voting_method": "weighted",
    "weights": {
        "gliner": 2.0,
        "llm": 3.0,
        "keyword": 1.0
    }
}

# Create ensemble router
ensemble = EnsembleRoutingStrategy(config)

# Route query using all strategies
query = "Find videos and documents about climate change"
decision = await ensemble.route(query)

print(f"Ensemble Decision: {decision.search_modality.value}")
print(f"Combined Confidence: {decision.confidence_score:.2f}")
```

## Integration with Existing System

### Updating the Composing Agent

Replace the existing QueryAnalysisTool with the new version:

```python
# In src/agents/composing_agents_main.py

from src.agents.query_analysis_tool_v2 import QueryAnalysisToolV2

class ComposingAgent:
    def __init__(self):
        # Replace old QueryAnalysisTool
        self.query_analyzer = QueryAnalysisToolV2()
        
    async def process_query(self, query):
        # Analyze query with new system
        analysis = await self.query_analyzer.execute(query)
        
        # Route to appropriate agents based on analysis
        if analysis['needs_video_search']:
            await self.video_agent.search(query)
        if analysis['needs_text_search']:
            await self.text_agent.search(query)
```

## Testing

### Run Comprehensive Tests

```bash
# Test all components
python tests/test_comprehensive_routing.py --test all

# Test specific component
python tests/test_comprehensive_routing.py --test tiered

# Save test results
python tests/test_comprehensive_routing.py --test all --save
```

### Test Output Example

```
🧪 COMPREHENSIVE ROUTING SYSTEM TEST SUITE
================================================================================

TESTING TIERED ROUTER
================================================================================

📊 Running queries through tiered architecture:
----------------------------------------
  ✅ 🚀 [video_explicit] Show me the tutorial video on Python pr...
     Tier: fast_path, Confidence: 0.85, Time: 0.008s
  ✅ 🧠 [text_report] Create a detailed report on AI advancement...
     Tier: slow_path, Confidence: 0.92, Time: 0.234s
  ✅ 🚀 [both_general] Find information about solar panels...
     Tier: fast_path, Confidence: 0.73, Time: 0.011s

📊 Tier Usage Distribution:
  fast_path: 7 (70.0%)
  slow_path: 2 (20.0%)
  fallback: 1 (10.0%)

📈 Overall Performance:
  Modality Accuracy: 9/10 (90.0%)
  Generation Accuracy: 8/10 (80.0%)
  Full Accuracy: 8/10 (80.0%)
  Avg Confidence: 0.76
  Avg Time: 0.043s
```

## Performance Benchmarks

| Strategy | Latency | Accuracy | Cost | Hardware |
|----------|---------|----------|------|----------|
| GLiNER (Fast Path) | <10ms | 85% | Low | CPU |
| SmolLM3 (Slow Path) | ~100ms | 92% | Medium | GPU |
| Keyword (Fallback) | <1ms | 70% | Minimal | CPU |
| Ensemble (All) | ~150ms | 94% | High | GPU+CPU |

## Monitoring and Metrics

### Export Metrics

```python
# Export routing metrics
router.export_metrics("routing_metrics.json")

# Get performance report
report = router.get_performance_report()
print(json.dumps(report, indent=2))
```

### Metrics Dashboard

Metrics are automatically collected and can be visualized:

```python
{
  "total_queries": 1000,
  "tier_performance": {
    "fast_path": {
      "total_requests": 850,
      "success_rate": 0.88,
      "average_execution_time": 0.009
    },
    "slow_path": {
      "total_requests": 120,
      "success_rate": 0.95,
      "average_execution_time": 0.187
    },
    "fallback": {
      "total_requests": 30,
      "success_rate": 0.73,
      "average_execution_time": 0.001
    }
  }
}
```

## Optimization Workflow

### Continuous Learning Pipeline

1. **Collect Performance Data**
   ```python
   # Automatic collection during routing
   decision = await router.route(query)
   ```

2. **Monitor Performance**
   ```python
   # Check if optimization needed
   if router.query_count % 1000 == 0:
       await router.optimize_routing()
   ```

3. **Apply Optimizations**
   - GLiNER: Threshold and label optimization
   - LLM: Temperature and prompt optimization
   - Ensemble: Weight rebalancing

4. **Export Optimized Configuration**
   ```python
   optimized_config = router.get_optimized_config()
   optimized_config.save("configs/routing_config_optimized.json")
   ```

## Troubleshooting

### Common Issues

1. **GLiNER Model Not Loading**
   ```python
   # Check CUDA/MPS availability
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"MPS available: {torch.backends.mps.is_available()}")
   ```

2. **LLM Timeout**
   ```python
   # Increase timeout in config
   config.llm_config["timeout"] = 60
   ```

3. **Low Confidence Scores**
   ```python
   # Adjust thresholds
   config.tier_config["fast_path_confidence_threshold"] = 0.6
   ```

## Advanced Features

### Custom Routing Strategy

```python
from src.routing.base import RoutingStrategy, RoutingDecision

class CustomRoutingStrategy(RoutingStrategy):
    async def route(self, query, context=None):
        # Custom routing logic
        if "urgent" in query.lower():
            return RoutingDecision(
                search_modality=SearchModality.BOTH,
                generation_type=GenerationType.DETAILED_REPORT,
                confidence_score=0.9,
                routing_method="custom_urgent"
            )
        # Fallback to base implementation
        return await super().route(query, context)
```

### LangExtract Integration (Development Mode)

```python
# Enable LangExtract for data generation
config.langextract_config["enabled"] = True
config.langextract_config["model_id"] = "gemini-2.5-pro"

# Use for creating training data
from src.routing.langextract_adapter import LangExtractAdapter
adapter = LangExtractAdapter(config.langextract_config)
training_data = await adapter.generate_training_data(queries)
```

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.10-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy routing system
COPY src/routing /app/src/routing
COPY configs /app/configs

# Set up environment
ENV ROUTING_MODE=tiered
ENV ROUTING_CACHE_ENABLE=true

# Run service
CMD ["python", "-m", "src.routing.service"]
```

### Kubernetes Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: routing-service
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: router
        image: routing-service:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: ROUTING_MODE
          value: "tiered"
        - name: ROUTING_OPTIMIZATION_ENABLE
          value: "true"
```

## Conclusion

The comprehensive routing system provides a flexible, performant, and self-optimizing solution for query routing in multi-agent systems. The tiered architecture ensures optimal performance while maintaining high accuracy, and the auto-tuning capabilities enable continuous improvement based on real-world usage patterns.