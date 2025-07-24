# GPU Recommendations for Modal Deployment

This guide provides GPU recommendations for deploying the Agentic Router system on Modal, including teacher models, student models, and optimization workloads.

## 🎯 Quick Recommendations

### **Production Setup (Recommended)**
- **Teacher Model (Training)**: Qwen 32B on A100-80GB (~$3.20/hour, 1 hour usage)
- **Student Model (Production)**: Gemma-3-1b on T4 (~$0.60/hour, always-on)
- **Optimization**: A100-80GB (~$0.80, 15 minutes)
- **Total Cost**: ~$10 one-time setup + ~$200-400/month production

### **Budget Setup**
- **Teacher Model**: Qwen 7B on T4 (~$0.60/hour)
- **Student Model**: Gemma-3-1b on T4 (~$0.60/hour)
- **Total Cost**: ~$5 one-time setup + ~$200-400/month production

### **Performance Setup**
- **Teacher Model**: Qwen 32B on H100-80GB (~$4.50/hour)
- **Student Model**: Gemma-3-1b on L4 (~$1.00/hour)
- **Total Cost**: ~$15 one-time setup + ~$300-600/month production

## 📊 Detailed Analysis

### Teacher Model Requirements (Training Phase)

#### **Large Teacher Models (32B Parameters)**
**Memory Requirements:**
- Model weights: 32B params × 2 bytes (FP16) = 64GB
- KV cache: ~8GB (for context length 4K)
- VLLM overhead: ~8GB (tensor caching, etc.)
- **Total needed: ~80GB**

**GPU Options:**
| GPU | Memory | Price/Hour | Best For |
|-----|--------|------------|----------|
| **A100-80GB** ✅ | 80GB | $3.20 | **Recommended** - Perfect fit |
| H100-80GB | 80GB | $4.50 | Fastest, but more expensive |
| A100-40GB × 2 | 80GB | $6.40 | Tensor parallel (overkill) |
| A6000 × 2 | 96GB | $4.40 | Alternative but slower |

**Performance Expectations:**
- Throughput: ~2-3 queries/second
- Latency: ~500ms per query
- Batch size: 8-16 queries efficiently
- Use case: Training data generation only

#### **Small Teacher Models (7B Parameters)**
**Memory Requirements:**
- Model weights: 7B params × 2 bytes = 14GB
- KV cache: ~4GB
- VLLM overhead: ~4GB
- **Total needed: ~22GB**

**GPU Options:**
| GPU | Memory | Price/Hour | Best For |
|-----|--------|------------|----------|
| **T4** ✅ | 16GB | $0.60 | Budget option (tight fit) |
| **A10G** ✅ | 24GB | $1.10 | Recommended for 7B |
| L4 | 24GB | $1.00 | Good alternative |

### Student Model Requirements (Production)

#### **Small Models (1-3B Parameters)**
**Memory Requirements:**
- Model weights: 1.5B params × 2 bytes (FP16) = 3GB
- KV cache: ~1GB (for context length 2K)
- VLLM overhead: ~2GB
- **Total needed: ~6GB**

**GPU Options:**
| GPU | Memory | Price/Hour | Best For |
|-----|--------|------------|----------|
| **T4** ✅ | 16GB | $0.60 | **Production** - Cost effective |
| L4 | 24GB | $1.00 | Faster inference |
| A10G | 24GB | $1.10 | Good alternative |
| RTX A6000 | 48GB | $2.20 | Overkill |

**Performance Expectations:**
- Throughput: ~20-30 queries/second
- Latency: ~50ms per query
- Batch size: 32+ queries efficiently
- Use case: Production routing

## 💰 Cost Analysis

### Development Phase
```
Teacher Training (1 hour):
- A100-80GB: $3.20 × 1 hour = $3.20
- Alternative T4: $0.60 × 1 hour = $0.60

Optimization (15 minutes):
- A100-80GB: $3.20 × 0.25 hour = $0.80
- Alternative T4: $0.60 × 0.25 hour = $0.15

Total Development: $4.00 (premium) or $0.75 (budget)
```

### Production Deployment
```
Student Service (Always-on):
- T4: $0.60/hour
- Keep-warm: 2 instances = $1.20/hour
- Monthly maximum: $1.20 × 24 × 30 = $864/month
- Actual cost: Much lower due to auto-scaling and idle timeout

Realistic production cost: $200-400/month for moderate usage
```

## ⚙️ Configuration Examples

### Teacher Service Configuration
```python
# For large teacher models (32B)
@app.function(
    image=teacher_image,
    gpu="A100-80GB",
    memory=80000,
    timeout=3600
)
def teacher_service():
    model = LLM(
        model="Qwen/Qwen2.5-32B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        dtype="float16"
    )
    return model

# For smaller teacher models (7B)
@app.function(
    image=teacher_image,
    gpu="A10G",
    memory=24000,
    timeout=1800
)
def teacher_service_budget():
    model = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        dtype="float16"
    )
    return model
```

### Student Service Configuration
```python
# Production student model
@app.function(
    image=student_image,
    gpu="T4",
    memory=16000,
    min_containers=2,  # Always-on
    timeout=30,
    scaledown_window=300
)
def student_service():
    model = LLM(
        model="google/gemma-3-1b-it",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        enforce_eager=True,  # Faster inference
        dtype="float16"
    )
    return model
```

### Optimization Service Configuration
```python
# MIPROv2 optimization
@app.function(
    image=optimizer_image,
    gpu="A100-80GB",
    memory=40000,
    timeout=3600  # 1 hour for optimization
)
def optimization_service():
    # Run MIPROv2 with high-end resources
    optimizer = MIPROv2(
        metric=routing_metric,
        num_candidates=25,  # More candidates on powerful hardware
        num_trials=50,      # More trials for better results
        verbose=True
    )
    return optimizer
```

## 🚀 Performance Optimization Tips

### Memory Optimization
```python
# Maximize memory efficiency
model = LLM(
    model=model_name,
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
    swap_space=4,  # 4GB swap space for larger models
    cpu_offloading=True,  # Offload to CPU when needed
    quantization="fp16"  # Use FP16 for efficiency
)
```

### Inference Optimization
```python
# Optimize for speed
sampling_params = SamplingParams(
    temperature=0.1,
    max_tokens=100,
    skip_special_tokens=True,
    use_beam_search=False,  # Faster than beam search
    early_stopping=True
)
```

### Batching Configuration
```python
# Efficient batching
@app.function(
    gpu="T4",
    allow_concurrent_inputs=50,  # High concurrency
    timeout=30
)
def batch_inference(queries: List[str]):
    # Process multiple queries together
    outputs = model.generate(queries, sampling_params)
    return [output.outputs[0].text for output in outputs]
```

## 📈 Scaling Strategies

### Auto-scaling Configuration
```python
@app.function(
    gpu="T4",
    min_containers=2,    # Always-on minimum
    max_containers=10,   # Scale up to 10 instances
    scaledown_window=300,  # 5 minutes before scaling down
    allow_concurrent_inputs=20
)
```

### Load Balancing
- Use multiple small instances rather than few large ones
- T4 instances scale faster than A100 instances
- Consider geographic distribution for global users

### Cost Monitoring
```python
# Track costs in your application
@app.function()
def log_usage():
    # Log inference counts, latency, costs
    # Set up alerts for unexpected usage spikes
    pass
```

## 🔧 Troubleshooting

### Common Issues

**Out of Memory Errors:**
- Reduce `gpu_memory_utilization` to 0.8 or lower
- Enable CPU offloading
- Use smaller batch sizes
- Consider model quantization

**Slow Inference:**
- Use `enforce_eager=True` for smaller models
- Optimize sampling parameters
- Use appropriate GPU for model size
- Enable model caching

**High Costs:**
- Set appropriate `scaledown_window`
- Use smaller GPUs when possible
- Monitor and optimize batch sizes
- Consider spot instances for development

### Monitoring Commands
```bash
# Check Modal usage
modal stats

# Monitor specific app
modal logs your-app-name

# Check costs
modal billing
```

This configuration will give you optimal performance and cost efficiency for the Agentic Router system on Modal.