# Modal Deployment Guide - New Architecture

Complete guide for deploying the provider-agnostic Agentic Router system using the new unified architecture.

## 🎯 Architecture Overview

The new system uses a **provider-agnostic architecture** with a single orchestrator and unified inference service:

```
New Architecture:
├── src/optimizer/orchestrator.py     # Handles all optimization (teacher + student)
├── src/inference/modal_inference_service.py  # General-purpose Modal service
├── scripts/run_optimization.py       # Complete automated workflow
└── config.json                       # Unified configuration
```

**Key Benefits:**
- ✅ **Single deployment** instead of multiple services
- ✅ **Provider abstractions** - easy to switch between Modal/Ollama/APIs
- ✅ **Unified configuration** in config.json
- ✅ **Automatic artifact management** via Modal volumes
- ✅ **Complete automation** with one script
- ✅ **Cost optimization** - Serverless, pay per request
- ✅ **Auto-scaling** - Handle traffic spikes automatically

## 🚀 Quick Deployment (Recommended)

### 1. Setup Configuration
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
    "providers": {
      "modal": {
        "gpu_config": "A10G",
        "memory_mb": 16000
      },
      "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY"
      }
    }
  }
}
```

### 2. Set Modal Secrets
```bash
modal secret create anthropic-secret ANTHROPIC_API_KEY=your_key_here
modal secret create openai-secret OPENAI_API_KEY=your_key_here
```

### 3. Run Optimization

#### Option 1: Just Optimization
```bash
# Run optimization only (no deployment)
python scripts/run_orchestrator.py
```

#### Option 2: Full Workflow
```bash
# One command does everything:
# - Runs orchestrator optimization (teacher-student training)
# - Deploys student model to Modal
# - Uploads artifacts to Modal volume
# - Deploys production inference API
# - Runs comprehensive tests
python scripts/run_optimization.py
```

That's it! The system handles:
- ✅ Teacher model calls via Anthropic API
- ✅ Student model deployment on Modal
- ✅ MIPROv2 optimization
- ✅ Artifact storage and retrieval
- ✅ Production API deployment

## ⚙️ Component Details

### Orchestrator (`src/optimizer/orchestrator.py`)
**What it does:**
- Calls teacher model (Claude) via API to generate training examples
- Deploys student model (Gemma) on Modal automatically
- Runs MIPROv2 optimization using DSPy
- Uploads results to Modal volume

**Resource Usage:**
- Teacher: API calls only (no GPU needed)
- Student: Modal GPU for optimization (~A10G, 15-30 minutes)
- Cost: ~$5-15 one-time per optimization

### Inference Service (`src/inference/modal_inference_service.py`)
**What it does:**
- Loads optimization artifacts automatically from Modal volume
- Provides production routing API with sub-100ms latency
- Supports both Modal GPU and local Ollama fallback
- Auto-scales based on traffic

**Resource Usage:**
- GPU: T4 or A10G (always-on with auto-scaling)
- Cost: ~$0.60-1.10/hour, scales to zero when idle

## 🔧 Manual Deployment (Advanced)

If you need more control:

### 1. Run Orchestrator Only
```bash
# Run optimization without deployment
python scripts/run_orchestrator.py --config config.json

# Test model connections
python scripts/run_orchestrator.py --test-models

# Setup services only
python scripts/run_orchestrator.py --setup-only
```

### 2. Deploy Inference Service Only
```bash
# Deploy the Modal inference service
modal deploy src/inference/modal_inference_service.py

# The service will be available at endpoints like:
# https://your-app-generate.modal.run
# https://your-app-chat-completions.modal.run
```

### 3. Manual Artifact Management
```bash
# Upload artifacts manually
modal volume put optimization-artifacts \
  optimization_results/unified_router_prompt_artifact.json \
  /artifacts/unified_router_prompt_artifact.json

# Download artifacts
modal volume get optimization-artifacts \
  /artifacts/unified_router_prompt_artifact.json \
  ./downloaded_artifacts.json
```

## 📊 Resource Requirements

### Development/Optimization
```
Teacher Model (API-based):
- Provider: Anthropic Claude or OpenAI GPT-4
- Cost: ~$0.01-0.10 per example
- Total: ~$0.50-5.00 for 50 examples

Student Model (Modal):
- GPU: A10G (24GB) - sufficient for Gemma-3-1b
- Duration: 15-30 minutes for optimization
- Cost: ~$1.10 × 0.5 hours = ~$0.55

Total optimization cost: ~$1-6 per run
```

### Production
```
Inference Service:
- GPU: T4 (16GB) or A10G (24GB)
- Memory: 16GB sufficient
- Scaling: 1-10 instances based on traffic
- Cost: $0.60-1.10/hour × actual usage
```

## 🔄 Provider Flexibility

The system supports easy switching between providers:

### Modal + API (Recommended)
```yaml
teacher:
  provider: "anthropic"    # Claude API
student:
  provider: "modal"        # Gemma on Modal GPU
```

### Full Local Development
```yaml
teacher:
  provider: "local"        # Ollama locally
student:
  provider: "local"        # Ollama locally
```

### Hybrid Setup
```yaml
teacher:
  provider: "openai"       # GPT-4 API
student:
  provider: "local"        # Local Ollama for testing
```

## 🧪 Testing and Validation

### Automated Testing
```bash
# Test optimizer structure and imports
python tests/test_optimizer_structure.py

# Test routing optimization
python tests/routing/test_combined_routing.py
```

### Manual Testing
```bash
# Test API endpoint
curl -X POST https://your-api.modal.run/route \
  -H "Content-Type: application/json" \
  -d '{"user_query": "Show me cooking videos"}'

# Health check
curl https://your-api.modal.run/health

# Model info
curl https://your-api.modal.run/get-model-info
```

## 🚀 Modal Deployment Benefits

### Performance Gains vs Local
- **GPU Acceleration**: T4/A10G/A100 GPUs vs local CPU/MPS
- **Parallel Processing**: Run multiple optimizations simultaneously
- **Inference Speed**: 50ms on Modal vs 200ms+ locally
- **Always Available**: No need to keep local servers running

### Cost Optimization
- **Serverless Model**: Pay only for actual inference time
- **GPU Sharing**: Multiple models on shared infrastructure
- **Efficient Scaling**: Automatic up/down based on traffic
- **Example Costs**: 
  - T4 GPU: ~$0.59/hour (only when running)
  - A10G GPU: ~$1.10/hour (better for larger models)
  - Typical inference: ~$0.0001 per request

### Development Benefits
- **Reproducible Environment**: Same setup for all team members
- **Version Control**: Model versioning and rollback support
- **Built-in Monitoring**: Logging, metrics, and alerts
- **CI/CD Ready**: Automated deployment pipeline

## 📈 Monitoring and Scaling

### Built-in Monitoring
```python
# Production API includes metrics
{
  "search_modality": "video",
  "generation_type": "raw_results", 
  "latency_ms": 45,
  "confidence": 0.95,
  "provider": "modal",
  "status": "success"
}
```

### Modal Monitoring
```bash
# Check app status
modal stats

# View logs
modal logs agentic-router-production

# Monitor costs
modal billing
```

### Auto-scaling Configuration
The inference service automatically scales:
- **Min containers**: 2 (always-on for production)
- **Max containers**: 10 (handles traffic spikes)
- **Scale-down**: 5 minutes idle timeout
- **Concurrent requests**: 50 per container

## 🚨 Troubleshooting

### Common Issues

**"Provider not found" errors:**
```bash
# Check provider registration
python -c "from src.optimizer.providers import *; print('Providers loaded')"
```

**"Artifacts not found" errors:**
```bash
# Check Modal volume
modal volume ls optimization-artifacts

# Verify artifact path
modal volume get optimization-artifacts /artifacts/ ./temp/
```

**High latency:**
```bash
# Check GPU utilization
modal stats --gpu

# Monitor scaling behavior
modal logs agentic-router-production | grep scale
```

### Recovery Procedures

**Rerun optimization:**
```bash
python scripts/run_orchestrator.py --config config.json
```

**Redeploy Service:**
```bash
modal deploy src/inference/modal_inference_service.py
```

**Reset artifacts:**
```bash
# Clear and regenerate artifacts
modal volume rm optimization-artifacts /artifacts/unified_router_prompt_artifact.json
python scripts/run_optimization.py
```

## 🎯 Migration from Old Architecture

If you have old teacher/student services:

1. **Stop old services:**
   ```bash
   modal app stop teacher-service
   modal app stop student-service
   modal app stop miprov2-optimizer
   ```

2. **Deploy new architecture:**
   ```bash
   python scripts/run_optimization.py
   ```

3. **Update integrations:**
   - Use new API endpoint URLs
   - Update to new response schema
   - Remove separate service calls

The new architecture is much simpler and more maintainable!