# Agentic Router Optimizer

Advanced query routing optimization using DSPy MIPROv2 with provider-agnostic architecture. This system automatically optimizes routing decisions through teacher-student distillation, supporting multiple model hosting providers.

## 🏗️ Architecture

```
src/optimizer/
├── schemas.py              # Pydantic models (RoutingDecision, AgenticRouter)
├── router_optimizer.py     # DSPy MIPROv2 optimization logic
├── orchestrator.py         # Main orchestration with provider abstractions
└── providers/              # Provider abstraction layer
    ├── base_provider.py    # Abstract interfaces
    ├── modal_provider.py   # Modal implementation
    ├── local_provider.py   # Ollama/local implementation
    └── __init__.py         # Provider registration
```

## 🧠 Key Components

### **1. Schemas (`schemas.py`)**
Defines the unified routing schema used throughout the system:

```python
class RoutingDecision(BaseModel):
    search_modality: str = Field(pattern="^(video|text)$")
    generation_type: str = Field(pattern="^(detailed_report|summary|raw_results)$")

class AgenticRouter(BaseModel):
    conversation_history: str
    user_query: str
    routing_decision: RoutingDecision
```

### **2. Router Optimizer (`router_optimizer.py`)**
Implements DSPy MIPROv2 optimization:

- **RouterModule**: DSPy module for routing decisions
- **Training Data Generation**: Creates diverse query examples
- **Teacher-Student Pipeline**: Uses large models to teach smaller ones
- **Evaluation Metrics**: Measures routing accuracy
- **OptimizedRouter**: Production-ready optimized router class

### **3. Orchestrator (`orchestrator.py`)**
Main coordination system with provider abstractions:

- **ModelClient**: Unified interface for all model providers
- **OptimizationOrchestrator**: Complete optimization workflow
- **Provider Management**: Automatic provider initialization and deployment
- **Artifact Handling**: Seamless artifact upload/download

### **4. Provider System (`providers/`)**
Extensible provider architecture:

**Base Provider (`base_provider.py`):**
```python
class ModelProvider(ABC):
    @abstractmethod
    def call_model(self, model_id: str, prompt: str, ...) -> str:
        pass

class ArtifactProvider(ABC):
    @abstractmethod
    def upload_artifact(self, local_path: str, remote_path: str) -> bool:
        pass
```

**Implementations:**
- **Modal Provider**: GPU-accelerated model hosting with VLLM
- **Local Provider**: Ollama integration for local development
- **Future**: AWS Bedrock, Google Vertex AI, Azure OpenAI

## ⚡ Quick Start

### **1. Configuration**
Create or update `config.json`:
```json
{
  "optimization": {
    "enabled": true,
    "type": "dspy",
    "teacher": {
      "model": "claude-3-5-sonnet-20241022",
      "provider": "anthropic"
    },
    "student": {
      "model": "google/gemma-3-1b-it",
      "provider": "modal"
    },
    "num_examples": 50,
    "num_candidates": 10,
    "num_trials": 20
  }
}
```

### **2. Run Optimization**
```bash
# Complete optimization workflow
python scripts/run_orchestrator.py --config config.json

# Test model connections only
python scripts/run_orchestrator.py --test-models

# Setup services only
python scripts/run_orchestrator.py --setup-only
```

### **3. Use Optimized Router**
```python
from src.optimizer.router_optimizer import OptimizedRouter

# Load optimized router
router = OptimizedRouter("optimization_results/router_prompt_artifact.json")

# Make routing decisions
decision = router.route(
    user_query="Show me how to cook pasta",
    conversation_history=""
)

print(decision.search_modality)  # "video"
print(decision.generation_type)  # "raw_results"
```

## 🔧 Provider Support

### **Modal Provider**
- **Features**: GPU acceleration, VLLM inference, auto-scaling
- **Models**: Any HuggingFace model
- **Use Cases**: Production deployment, high throughput

### **Local Provider (Ollama)**
- **Features**: Local inference, development testing
- **Models**: Ollama-supported models
- **Use Cases**: Development, offline inference

### **API Providers**
- **Anthropic**: Claude models via API
- **OpenAI**: GPT models via API
- **Use Cases**: Teacher models, high-quality generation

## 📊 Optimization Process

### **1. Training Data Generation**
```python
# Manual examples
examples = generate_training_examples()

# Teacher-generated examples
teacher_examples = generate_teacher_examples(teacher_lm, num_examples=50)
```

### **2. MIPROv2 Optimization**
```python
optimizer = MIPROv2(
    metric=routing_metric,
    num_candidates=10,
    init_temperature=0.7
)

optimized_router = optimizer.compile(
    router,
    trainset=train_set,
    valset=val_set
)
```

### **3. Evaluation**
```python
metrics = evaluate_routing_accuracy(router, test_set)
# {
#   "modality_accuracy": 0.85,
#   "generation_accuracy": 0.82,
#   "overall_accuracy": 0.78
# }
```

## 🚀 Production Integration

### **1. Export Artifacts**
Optimization produces portable artifacts:
```json
{
  "system_prompt": "optimized instructions",
  "few_shot_examples": [...],
  "model_config": {...}
}
```

### **2. Production API**
Artifacts are automatically uploaded and used by the production API:
```python
# src/inference/modal_inference_service.py loads artifacts automatically
```

### **3. Integration Examples**
```python
# Direct integration
from src.optimizer.orchestrator import OptimizationOrchestrator

orchestrator = OptimizationOrchestrator("config.json")
results = orchestrator.run_optimization()

# Production routing
import requests
response = requests.post(
    "https://your-api.modal.run/route",
    json={"user_query": "Show me cooking videos"}
)
```

## 🧪 Testing

Run comprehensive tests:
```bash
# Test structure and imports
python tests/test_optimizer_structure.py

# Test optimization workflow
python tests/routing/test_combined_routing.py
```

## 🔮 Performance

### **Expected Results**
- **Baseline Accuracy**: ~60-70%
- **Optimized Accuracy**: ~80-90%
- **Optimization Time**: 10-30 minutes
- **Production Latency**: <100ms

### **Scaling**
- **Throughput**: 100+ QPS per GPU
- **Cost**: ~$0.01 per 1000 queries
- **Auto-scaling**: Built-in with Modal

## 🛠️ Development

### **Adding New Providers**
1. Create provider class inheriting from base interfaces
2. Register in `providers/__init__.py`
3. Add configuration support
4. Test with orchestrator

### **Extending Schema**
1. Update `schemas.py` with new fields
2. Update validation patterns
3. Update evaluation metrics
4. Test with existing data

### **Custom Optimization**
```python
from src.optimizer.router_optimizer import optimize_router

results = optimize_router(
    student_model="your-model",
    teacher_model="teacher-model", 
    num_teacher_examples=100,
    output_dir="custom_results"
)
```

This optimizer system provides a solid foundation for building production-ready query routing with automatic optimization capabilities while maintaining flexibility for different deployment scenarios.