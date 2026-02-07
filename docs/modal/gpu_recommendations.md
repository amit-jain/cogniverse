# GPU Recommendations for Cogniverse Multi-Modal Processing

**Package:** `cogniverse-runtime` (ingestion pipeline) + `cogniverse-agents` (search agents)

---

## Overview

Cogniverse processes **multi-modal content** across six content types:

- **VIDEO**: Frame-based and chunk-based processing

- **AUDIO**: Speech and audio analysis

- **IMAGE**: Visual similarity and search

- **DOCUMENT**: PDF, DOCX, text extraction

- **TEXT**: Natural language processing

- **DATAFRAME**: Tabular data analysis

This guide covers GPU requirements for embedding models used in ingestion and search pipelines.

---

## Quick Recommendations

### Production Setup (Recommended)
- **ColPali (Frame-Based)**: A100-40GB (~$3.00/hour for ingestion, minimal for search)
- **VideoPrism (Chunk-Based)**: L4 (~$1.00/hour for ingestion)
- **ColQwen2 (Image/Document)**: L4 or T4 (~$1.00/hour for ingestion)
- **Total**: ~$50-100 one-time ingestion per 1000 videos + minimal search costs

### Budget Setup
- **ColPali**: L4 GPU (~$1.00/hour, tight fit)
- **VideoPrism**: T4 (~$0.60/hour, base model only)
- **ColQwen2**: T4 (~$0.60/hour)
- **Total**: ~$30-60 per 1000 videos ingestion

### Performance Setup
- **ColPali**: A100-80GB (~$3.20/hour, optimal throughput)
- **VideoPrism**: A100-40GB (~$3.00/hour, large batches)
- **ColQwen2**: A10G (~$1.10/hour)
- **Total**: ~$100-150 per 1000 videos ingestion

---

## Model Requirements

### 1. ColPali (Frame-Based Video Embeddings)

**Model:** `vidore/colpali-v1.2` or `vidore/colsmol-500m`
**Content Types:** VIDEO (frames), IMAGE, DOCUMENT

#### Memory Requirements

**ColPali v1.2 (Full Model)**
- Model weights: ~3GB (base model)
- Image preprocessing: ~2GB per batch
- Patch embeddings: ~4GB (128 patches × 128 dims × batch size)
- CUDA overhead: ~2GB
- **Total: ~11GB minimum, 16GB recommended**

**ColSmol 500M (Smaller)**
- Model weights: ~2GB
- Preprocessing: ~1.5GB
- Embeddings: ~3GB
- **Total: ~7GB minimum, 12GB recommended**

#### GPU Options

| GPU | Memory | Price/Hour | Performance | Best For |
|-----|--------|------------|-------------|----------|
| **A100-40GB** ✅ | 40GB | $3.00 | Excellent | **Production ingestion** |
| **L4** | 24GB | $1.00 | Good | Budget production |
| **A10G** | 24GB | $1.10 | Good | Alternative to L4 |
| **T4** | 16GB | $0.60 | Fair | Tight fit, small batches |
| **A100-80GB** | 80GB | $3.20 | Excellent | Large batches |

#### Performance Expectations

**Ingestion (Frame Encoding):**

- **A100-40GB**: ~30-40 frames/second, batch size 32

- **L4**: ~15-20 frames/second, batch size 16

- **T4**: ~8-10 frames/second, batch size 4

**Search (Query Encoding):**

- Single query: ~50-100ms (any GPU)

- Batch queries: ~200ms for 10 queries

- Minimal GPU needed (can use CPU for light workloads)

#### Recommended Configuration

```python
# Production ingestion on Modal
@app.function(
    image=colpali_image,
    gpu="A100-40GB",
    memory=32000,
    timeout=3600
)
def encode_video_frames_colpali(frames: list):
    """Generate ColPali embeddings using colpali-engine API."""
    from colpali_engine.models import ColIdefics3, ColIdefics3Processor
    import torch

    model = ColIdefics3.from_pretrained(
        "vidore/colpali-v1.2",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    ).eval()
    processor = ColIdefics3Processor.from_pretrained("vidore/colpali-v1.2")

    batch_inputs = processor.process_images(frames).to(model.device)
    with torch.no_grad():
        embeddings = model(**batch_inputs)
    return embeddings.cpu().to(torch.float32).numpy()

# Budget ingestion on Modal
@app.function(
    image=colpali_image,
    gpu="L4",
    memory=24000,
    timeout=3600
)
def encode_video_frames_budget(frames: list):
    """Generate ColPali embeddings with smaller model for budget GPUs."""
    from colpali_engine.models import ColIdefics3, ColIdefics3Processor
    import torch

    model = ColIdefics3.from_pretrained(
        "vidore/colsmol-500m",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    ).eval()
    processor = ColIdefics3Processor.from_pretrained("vidore/colsmol-500m")

    batch_inputs = processor.process_images(frames).to(model.device)
    with torch.no_grad():
        embeddings = model(**batch_inputs)
    return embeddings.cpu().to(torch.float32).numpy()
```

---

### 2. VideoPrism (Chunk-Based Video Embeddings)

**Model:** `videoprism_public_v1_base_hf`
**Content Types:** VIDEO (chunks)

#### Memory Requirements

**VideoPrism Base (114M params, 458MB weights)**
- Model weights (float32): ~0.5GB
- JAX XLA compilation cache: ~2-5GB
- Video preprocessing (16 frames at 288x288): ~0.1GB
- Intermediate activations: ~2-4GB
- **Total: ~5-10GB minimum, 16GB recommended**

**VideoPrism Large (354M params, 1.42GB weights)**
- Model weights (float32): ~1.4GB
- JAX XLA compilation cache: ~3-6GB
- Intermediate activations: ~3-5GB
- **Total: ~8-12GB minimum, 24GB recommended**

#### GPU Options

| GPU | Memory | Price/Hour | Performance | Best For |
|-----|--------|------------|-------------|----------|
| **L4** ✅ | 24GB | $1.00 | Good | **Production (base and large)** |
| **A10G** | 24GB | $1.10 | Good | Alternative to L4 |
| **T4** | 16GB | $0.60 | Fair | Base model, small batches |
| **A100-40GB** | 40GB | $3.00 | Excellent | Large batches, high throughput |

#### Performance Expectations

**Ingestion (Chunk Encoding):**

- **L4**: ~3-5 chunks/second (30s chunks), batch size 4

- **A100-40GB**: ~8-12 chunks/second, batch size 8

- Processing time: ~2-5 minutes per 1-hour video

**Search (Query Encoding):**

- Single query: ~200-300ms

- Not typically used for search (ColPali preferred for frame-level)

#### Recommended Configuration

VideoPrism uses JAX/Flax with a custom loader (not the standard HuggingFace pattern). The actual loading is in `cogniverse_core.common.models.model_loaders`:

```python
# Production ingestion on Modal
@app.function(
    image=videoprism_image,
    gpu="L4",
    memory=24000,
    timeout=7200
)
def encode_video_chunks_videoprism():
    """Generate VideoPrism embeddings using the custom JAX loader."""
    import jax
    from cogniverse_core.common.models.model_loaders import get_or_load_model

    jax.config.update('jax_platform_name', 'gpu')

    # get_or_load_model returns (videoprism_loader, None) for VideoPrism models
    config = {"model_name": "videoprism_public_v1_base_hf"}
    loader, _ = get_or_load_model("videoprism_public_v1_base_hf", config, logger)

    # The loader provides extract_embeddings(frames) for pre-extracted frames,
    # or process_entire_video(video_path, sampling_fps) for full video processing
    result = loader.process_entire_video(video_path, sampling_fps=1.0)
    return result["embeddings"]

# High-performance ingestion on Modal
@app.function(
    image=videoprism_image,
    gpu="A100-40GB",
    memory=40000,
    timeout=3600
)
def encode_video_chunks_performance():
    """Same pipeline on A100 for higher throughput."""
    import jax
    from cogniverse_core.common.models.model_loaders import get_or_load_model

    jax.config.update('jax_platform_name', 'gpu')

    config = {"model_name": "videoprism_public_v1_base_hf"}
    loader, _ = get_or_load_model("videoprism_public_v1_base_hf", config, logger)

    result = loader.process_entire_video(video_path, sampling_fps=1.0)
    return result["embeddings"]
```

---

### 3. ColQwen2 (Multi-Modal Image/Document Embeddings)

**Model:** `vidore/colqwen-omni-v0.1` or `vidore/colqwen2-v1.0`
**Content Types:** IMAGE, DOCUMENT, TEXT

#### Memory Requirements

**ColQwen2 7B**
- Model weights: 7B params × 2 bytes (FP16) = 14GB
- Image preprocessing: ~4GB
- KV cache: ~4GB (for context length 4K)
- CUDA overhead: ~2GB
- **Total: ~24GB minimum, 32GB recommended**

**ColQwen2 with 4-bit Quantization:**

- Model weights: ~7GB (quantized)

- Preprocessing: ~4GB

- **Total: ~12GB minimum, 16GB recommended**

#### GPU Options

| GPU | Memory | Price/Hour | Performance | Best For |
|-----|--------|------------|-------------|----------|
| **L4** ✅ | 24GB | $1.00 | Good | **Production (full precision)** |
| **A10G** | 24GB | $1.10 | Good | Alternative to L4 |
| **T4** | 16GB | $0.60 | Fair | Quantized model only |
| **A100-40GB** | 40GB | $3.00 | Excellent | Large batches |

#### Performance Expectations

**Ingestion (Image/Document Encoding):**

- **L4**: ~10-15 images/second, batch size 8

- **A10G**: ~12-18 images/second, batch size 8

- **T4** (quantized): ~6-8 images/second, batch size 4

**Search (Query Encoding):**

- Single query: ~100-150ms

- Batch queries: ~300ms for 10 queries

#### Recommended Configuration

```python
# Production ingestion on Modal
@app.function(
    image=colqwen_image,
    gpu="L4",
    memory=24000,
    timeout=3600
)
def encode_images_colqwen(images: list):
    """Generate ColQwen embeddings using colpali-engine API."""
    from colpali_engine.models import ColQwen2_5Omni, ColQwen2_5OmniProcessor
    import torch

    model = ColQwen2_5Omni.from_pretrained(
        "vidore/colqwen-omni-v0.1",
        torch_dtype=torch.bfloat16,
        device_map="cuda"
    ).eval()
    processor = ColQwen2_5OmniProcessor.from_pretrained("vidore/colqwen-omni-v0.1")

    # Process images (optimal batch_size=8 for L4)
    batch_inputs = processor.process_images(images).to(model.device)

    with torch.no_grad():
        embeddings = model(**batch_inputs)

    return embeddings.cpu().to(torch.float32).numpy()

# Budget ingestion (quantized)
@app.function(
    image=colqwen_image,
    gpu="T4",
    memory=16000,
    timeout=3600
)
def encode_images_budget(images: list):
    """Generate ColQwen embeddings with quantization for budget GPUs."""
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    from transformers import BitsAndBytesConfig
    import torch

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v1.0",
        quantization_config=quantization_config,
        device_map="cuda"
    ).eval()
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")

    # Smaller batch for T4 (batch_size=4)
    batch_inputs = processor.process_images(images).to(model.device)

    with torch.no_grad():
        embeddings = model(**batch_inputs)

    return embeddings.cpu().to(torch.float32).numpy()
```

---

## Content Type Processing Requirements

### VIDEO Content

**Two Processing Approaches:**

1. **Frame-Based (ColPali)**
   - Extract frames at 1 FPS
   - Encode each frame independently
   - Best for: Fine-grained visual search, temporal reasoning
   - GPU: A100-40GB or L4
   - Time: ~2-5 minutes per 1-hour video

2. **Chunk-Based (VideoPrism)**
   - Split video into 30-second chunks
   - Encode temporal context
   - Best for: Coarse-grained search, video-level understanding
   - GPU: L4 (recommended) or T4 (base model)
   - Time: ~3-8 minutes per 1-hour video

**Recommended:** Use both approaches for comprehensive coverage.

### AUDIO Content

- Transcribed with Whisper for speech-to-text
- Text transcripts indexed via BM25 text search
- GPU: T4 sufficient for Whisper inference
- Alternative: CPU-based Whisper for lower cost

### IMAGE Content

- Single images or extracted frames
- Processed with ColQwen2 or ColPali
- GPU: L4 or A10G
- Time: ~100-200ms per image

### DOCUMENT Content

- PDF, DOCX, text extraction
- Convert to images for vision models (ColQwen2)
- Or use text embeddings (faster, CPU-based)
- GPU: L4 for vision-based, CPU for text-based

### TEXT Content

- Natural language text
- Typically CPU-based (nomic-embed-text, sentence-transformers)
- GPU: Optional, T4 sufficient for large batches

### DATAFRAME Content

- Tabular data processing
- Convert to text representation
- CPU-based processing
- GPU: Not required

---

## Cost Analysis

### Ingestion Costs (One-Time per Content)

**Per 1000 Videos (1 hour average length):**

| Model | GPU | Hours | Cost/Hour | Total Cost |
|-------|-----|-------|-----------|------------|
| ColPali (frames) | A100-40GB | 50 | $3.00 | $150 |
| VideoPrism (chunks) | L4 | 60 | $1.00 | $60 |
| **Total** | | | | **$210** |

**Budget Alternative:**
| Model | GPU | Hours | Cost/Hour | Total Cost |
|-------|-----|-------|-----------|------------|
| ColSmol (frames) | L4 | 80 | $1.00 | $80 |
| VideoPrism (chunks) | T4 | 80 | $0.60 | $48 |
| **Total** | | | | **$128** |

**Per 10,000 Images:**
| Model | GPU | Hours | Cost/Hour | Total Cost |
|-------|-----|-------|-----------|------------|
| ColQwen2 | L4 | 3 | $1.00 | $3 |

### Search Costs (Ongoing)

**Query Encoding (Minimal):**

- ColPali: ~$0.60/hour on T4 (handles 100+ queries/second)

- With autoscaling: ~$50-100/month for moderate traffic

- Consider CPU-based search for lower costs

**Recommendation:** Run search on cheaper GPUs (T4, L4) or CPU with cached embeddings.

---

## Multi-Modal Pipeline Example

For video content, the actual implementation uses `VideoIngestionPipeline.process_video_async()` which handles all embedding generation based on configured profiles. The example below shows how individual model calls map to Modal GPU functions:

```python
@app.function(
    image=ingestion_image,
    gpu="A100-40GB",
    memory=40000,
    timeout=7200
)
def ingest_multimodal_content(content_path: str, content_type: str):
    """
    Ingest multi-modal content and generate embeddings.

    Content Types: VIDEO, AUDIO, IMAGE, DOCUMENT, TEXT, DATAFRAME
    """
    from pathlib import Path
    from cogniverse_runtime.ingestion.pipeline import VideoIngestionPipeline
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_sdk.document import Document, ContentType

    config_manager = create_default_config_manager()

    if content_type == "VIDEO":
        # VideoIngestionPipeline handles all embedding generation per profile
        pipeline = VideoIngestionPipeline(
            tenant_id="default",
            config_manager=config_manager,
        )
        # process_video_async() runs keyframe extraction, transcription,
        # description generation, and embedding generation automatically
        import asyncio
        result = asyncio.run(pipeline.process_video_async(Path(content_path)))
        return result

    elif content_type == "IMAGE":
        colqwen_embeddings = encode_images_colqwen([content_path])

        doc = Document(
            content_type=ContentType.IMAGE,
            content_path=content_path,
        )
        doc.add_embedding("colqwen", colqwen_embeddings)
        return doc.id

    elif content_type == "DOCUMENT":
        doc_images = convert_pdf_to_images(content_path)
        colqwen_embeddings = encode_images_colqwen(doc_images)

        doc = Document(
            content_type=ContentType.DOCUMENT,
            content_path=content_path,
        )
        doc.add_embedding("colqwen", colqwen_embeddings)
        return doc.id

    elif content_type == "TEXT":
        text_embeddings = encode_text_cpu(content_path)

        doc = Document(
            content_type=ContentType.TEXT,
            text_content=read_text(content_path),
        )
        doc.add_embedding("text", text_embeddings)
        return doc.id
```

---

## Optimization Strategies

### 1. Batch Processing

```python
@app.function(
    gpu="A100-40GB",
    allow_concurrent_inputs=10
)
def batch_encode_frames(frame_batches: list[list]):
    """Process multiple videos in parallel batches."""
    from colpali_engine.models import ColIdefics3, ColIdefics3Processor
    import torch

    model = ColIdefics3.from_pretrained(
        "vidore/colsmol-500m", torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    processor = ColIdefics3Processor.from_pretrained("vidore/colsmol-500m")

    results = []
    for batch in frame_batches:
        batch_inputs = processor.process_images(batch).to(model.device)
        with torch.no_grad():
            embeddings = model(**batch_inputs)
        results.append(embeddings.cpu().to(torch.float32).numpy())
    return results
```

### 2. Mixed Precision

```python
# Use FP16 for memory efficiency
model = model.half()  # Convert to FP16
torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner
```

### 3. Gradient Checkpointing

```python
# For HuggingFace transformers models (ColPali, ColQwen) during training/fine-tuning
model.gradient_checkpointing_enable()
```

### 4. Model Quantization

```python
# 4-bit quantization for budget GPUs
from colpali_engine.models import ColQwen2
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v1.0",
    quantization_config=quantization_config
)
```

---

## Troubleshooting

### Out of Memory (OOM) Errors

**ColPali OOM:**

- Reduce batch size: 32 → 16 → 8 → 4

- Use ColSmol 500M instead of ColPali v1.2

- Enable gradient checkpointing

- Switch to larger GPU (L4 → A100-40GB)

**VideoPrism OOM:**

- Reduce chunk length: 30s → 15s → 10s

- Reduce batch size: 4 → 2 → 1

- Switch to larger GPU (T4 → L4 → A100-40GB)

- Use base model (114M params) instead of large model (354M params)

**ColQwen OOM:**

- Reduce batch size: 8 → 4 → 2

- Enable 4-bit quantization

- Reduce max sequence length

- Switch to L4 or A10G

### Slow Processing

**Check GPU utilization:**
```bash
nvidia-smi dmon -s u -d 1
```

**Common fixes:**

- Enable mixed precision (FP16)

- Increase batch size (if memory allows)

- Use faster GPUs (H100 > A100 > L4 > T4)

- Optimize data loading (prefetch, multi-threading)

### High Costs

**Ingestion optimization:**

- Use budget GPUs (L4 instead of A100-40GB)

- Use smaller models (ColSmol instead of ColPali)

- Batch processing (process multiple videos per GPU hour)

- Auto-scaling (scale down during idle periods)

**Search optimization:**

- Cache embeddings (avoid re-encoding)

- Use CPU for search when possible

- Use T4 or L4 for minimal costs

- Implement query batching

---

## Summary

**Key Takeaways:**

1. **ColPali (Frame-Based Video)**: A100-40GB recommended, L4 budget option
2. **VideoPrism (Chunk-Based Video)**: L4 recommended (114M params base), T4 budget option
3. **ColQwen2 (Image/Document)**: L4 recommended, T4 budget option
4. **Multi-Modal Support**: All six content types supported (VIDEO, AUDIO, IMAGE, DOCUMENT, TEXT, DATAFRAME)
5. **Cost Efficiency**: ~$200-250 per 1000 videos one-time ingestion, ~$50-100/month search

**Recommended Production Setup:**

- **Ingestion**: A100-40GB (ColPali) + L4 (VideoPrism) + L4 (ColQwen2)

- **Search**: T4 or L4 (or CPU for text-based)

- **Total**: ~$200-250 per 1000 videos ingestion, minimal ongoing costs

---

**For more information:**

- Installation Guide: [docs/operations/setup-installation.md](../operations/setup-installation.md)

- Ingestion Pipeline: [docs/modules/ingestion.md](../modules/ingestion.md)

- Model Documentation: [docs/modules/agents.md](../modules/agents.md)
