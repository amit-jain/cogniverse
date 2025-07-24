# Inference Services

General-purpose Modal inference service for the Multi-Agent RAG System.

## 🚀 Service Overview

```
src/inference/
└── modal_inference_service.py  # General-purpose Modal inference service
```

## 🎯 Modal Inference Service

A simple, flexible text generation service that can be deployed on Modal for any inference needs.

### **Key Features:**
- **Dynamic Model Loading**: Can load any HuggingFace model
- **Multiple Endpoints**: 
  - `/generate` - Basic text generation
  - `/chat-completions` - OpenAI-compatible API
  - `/health` - Health check
  - `/models` - List loaded models
- **GPU Acceleration**: Configurable GPU (T4/A10G/A100)
- **Auto-scaling**: Handles traffic spikes automatically
- **Model Caching**: Efficient model reuse across requests

### **Configuration:**
```python
# Environment variables
DEFAULT_MODEL = "google/gemma-3-1b-it"
DEFAULT_GPU = "A10G"
DEFAULT_MEMORY = 16000
DEFAULT_TIMEOUT = 300
```

### **Deployment:**
```bash
# Deploy to Modal
modal deploy src/inference/modal_inference_service.py

# The service will be available at:
# https://general-inference-service-generate.modal.run
```

### **Usage Examples:**

**Text Generation:**
```bash
curl -X POST https://your-app-generate.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the capital of France?",
    "temperature": 0.7,
    "max_tokens": 50
  }'
```

**Chat Completions (OpenAI-compatible):**
```bash
curl -X POST https://your-app-chat-completions.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "temperature": 0.7,
    "max_tokens": 50
  }'
```

**Health Check:**
```bash
curl https://your-app-health.modal.run
```

## 🔧 Integration with Main System

The Modal inference service can be used by the composing agent when configured in `config.json`:

```json
{
  "inference": {
    "provider": "modal",
    "modal_endpoint": "https://general-inference-service-generate.modal.run",
    "model_config": {
      "temperature": 0.1,
      "max_tokens": 100
    }
  }
}
```

When `query_inference_engine.mode` is set to `"llm"`, the QueryAnalysisTool will use this service for routing decisions.

## 💰 Cost Optimization

- **T4 GPU**: ~$0.59/hour - Good for models up to 7B
- **A10G GPU**: ~$1.10/hour - Better for larger models
- **A100 GPU**: ~$3.20/hour - For very large models

The service uses:
- `keep_warm=1` to maintain one warm instance
- `allow_concurrent_inputs=10` for efficient batching
- `container_idle_timeout=300` to scale down after 5 minutes

## 🚨 Troubleshooting

**Model loading issues:**
```bash
# Check Modal logs
modal logs general-inference-service
```

**High latency:**
- Ensure model is cached (first request loads the model)
- Check GPU utilization with `modal stats --gpu`
- Consider using a faster GPU or smaller model

**Out of memory:**
- Use a smaller model or increase memory allocation
- Enable quantization in model_config