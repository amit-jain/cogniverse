# GPU Recommendations for Cogniverse Multi-Modal Processing

**Last Updated:** 2026-01-25
**Package:** `cogniverse-runtime` (ingestion pipeline) + `cogniverse-agents` (search agents)
**Purpose:** GPU requirements for multi-modal content processing (video, audio, images, documents, text, dataframes) with ColPali, VideoPrism, and ColQwen models

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
- **VideoPrism (Chunk-Based)**: A100-80GB (~$3.20/hour for ingestion)
- **ColQwen2 (Image/Document)**: L4 or T4 (~$1.00/hour for ingestion)
- **Total**: ~$50-100 one-time ingestion per 1000 videos + minimal search costs

### Budget Setup
- **ColPali**: L4 GPU (~$1.00/hour, tight fit)
- **VideoPrism**: A100-40GB (~$3.00/hour, minimum requirement)
- **ColQwen2**: T4 (~$0.60/hour)
- **Total**: ~$30-60 per 1000 videos ingestion

### Performance Setup
- **ColPali**: A100-80GB (~$3.20/hour, optimal throughput)
- **VideoPrism**: H100-80GB (~$4.50/hour, fastest)
- **ColQwen2**: A10G (~$1.10/hour)
- **Total**: ~$100-150 per 1000 videos ingestion

---

## Model Requirements

### 1. ColPali (Frame-Based Video Embeddings)

**Model:** `vidore/colpali-v1.2` or `vidore/colsmol-500m`
**Content Types:** VIDEO (frames), IMAGE, DOCUMENT
**Architecture:** Vision Transformer (ViT) with late interaction

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
# Production ingestion
@app.function(
    image=colpali_image,
    gpu="A100-40GB",
    memory=32000,
    timeout=3600
)
def encode_video_frames_colpali():
    from colpali_engine.models import ColPali

    model = ColPali.from_pretrained(
        "vidore/colpali-v1.2",
        torch_dtype=torch.float16,  # Use FP16 for memory efficiency
        device_map="cuda:0"
    )

    # Process frames in batches
    batch_size = 32  # Optimal for A100-40GB
    embeddings = model.encode_images(frame_batch)
    return embeddings

# Budget ingestion
@app.function(
    image=colpali_image,
    gpu="L4",
    memory=24000,
    timeout=3600
)
def encode_video_frames_budget():
    model = ColPali.from_pretrained(
        "vidore/colsmol-500m",  # Smaller model
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )
    batch_size = 16  # Smaller batch for L4
    embeddings = model.encode_images(frame_batch)
    return embeddings
```

---

### 2. VideoPrism (Chunk-Based Video Embeddings)

**Model:** `google/videoprism-base`
**Content Types:** VIDEO (chunks), AUDIO (from video)
**Architecture:** Video Transformer with temporal modeling

#### Memory Requirements

**VideoPrism Base**
- Model weights: ~20GB (large vision-language model)
- Video preprocessing: ~15GB (30-second chunks)
- Temporal embeddings: ~10GB
- CUDA overhead: ~5GB
- **Total: ~50GB minimum, 64GB recommended**

**VideoPrism with JAX/Flax:**
- JAX overhead: ~10GB (compilation caching)
- **Total: ~60GB minimum, 80GB recommended**

#### GPU Options

| GPU | Memory | Price/Hour | Performance | Best For |
|-----|--------|------------|-------------|----------|
| **A100-80GB** ✅ | 80GB | $3.20 | Excellent | **Production requirement** |
| **H100-80GB** | 80GB | $4.50 | Fastest | High-throughput ingestion |
| **A100-40GB × 2** | 80GB | $6.00 | Good | Tensor parallel (complex) |

**Note:** VideoPrism **requires 80GB minimum** due to model size and JAX compilation overhead. A100-40GB is **not sufficient**.

#### Performance Expectations

**Ingestion (Chunk Encoding):**
- **A100-80GB**: ~5-8 chunks/second (30s chunks), batch size 4
- **H100-80GB**: ~10-12 chunks/second, batch size 8
- Processing time: ~2-5 minutes per 1-hour video

**Search (Query Encoding):**
- Single query: ~200-300ms
- Not typically used for search (ColPali preferred for frame-level)

#### Recommended Configuration

```python
# Production ingestion
@app.function(
    image=videoprism_image,
    gpu="A100-80GB",
    memory=80000,
    timeout=7200  # 2 hours for batch processing
)
def encode_video_chunks_videoprism():
    import jax
    from transformers import FlaxVideoPrismModel

    # Configure JAX for A100
    jax.config.update('jax_platform_name', 'gpu')

    model = FlaxVideoPrismModel.from_pretrained(
        "google/videoprism-base",
        dtype=jax.numpy.float16  # Use FP16
    )

    # Process 30-second chunks
    batch_size = 4  # Optimal for A100-80GB
    embeddings = model.encode_video(video_chunks)
    return embeddings

# High-performance ingestion
@app.function(
    image=videoprism_image,
    gpu="H100-80GB",
    memory=80000,
    timeout=3600
)
def encode_video_chunks_performance():
    # Same as above but with larger batch size
    batch_size = 8  # H100 can handle larger batches
    embeddings = model.encode_video(video_chunks)
    return embeddings
```

---

### 3. ColQwen2 (Multi-Modal Image/Document Embeddings)

**Model:** `Qwen/Qwen2-VL-7B-Instruct`
**Content Types:** IMAGE, DOCUMENT, TEXT
**Architecture:** Vision-Language Model (VLM)

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
# Production ingestion
@app.function(
    image=colqwen_image,
    gpu="L4",
    memory=24000,
    timeout=3600
)
def encode_images_colqwen():
    from transformers import Qwen2VLForConditionalGeneration

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="cuda:0"
    )

    batch_size = 8  # Optimal for L4
    embeddings = model.encode_images(image_batch)
    return embeddings

# Budget ingestion (quantized)
@app.function(
    image=colqwen_image,
    gpu="T4",
    memory=16000,
    timeout=3600
)
def encode_images_budget():
    from transformers import BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        quantization_config=quantization_config,
        device_map="cuda:0"
    )

    batch_size = 4  # Smaller batch for T4
    embeddings = model.encode_images(image_batch)
    return embeddings
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
   - GPU: A100-80GB (required)
   - Time: ~3-8 minutes per 1-hour video

**Recommended:** Use both approaches for comprehensive coverage.

### AUDIO Content

- Extracted from video chunks
- Processed with VideoPrism (audio modality)
- Same GPU requirements as VideoPrism
- Alternative: Use Whisper for speech-to-text + text embeddings

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
| VideoPrism (chunks) | A100-80GB | 60 | $3.20 | $192 |
| **Total** | | | | **$342** |

**Budget Alternative:**
| Model | GPU | Hours | Cost/Hour | Total Cost |
|-------|-----|-------|-----------|------------|
| ColSmol (frames) | L4 | 80 | $1.00 | $80 |
| VideoPrism (chunks) | A100-80GB | 60 | $3.20 | $192 |
| **Total** | | | | **$272** |

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

Complete ingestion pipeline for multi-modal content:

```python
# Full ingestion pipeline
@app.function(
    image=ingestion_image,
    gpu="A100-80GB",  # Highest requirement (VideoPrism)
    memory=80000,
    timeout=7200
)
def ingest_multimodal_content(content_path: str, content_type: str):
    """
    Ingest multi-modal content and generate embeddings.

    Content Types: VIDEO, AUDIO, IMAGE, DOCUMENT, TEXT, DATAFRAME
    """
    from cogniverse_runtime.ingestion.pipeline import IngestionPipeline
    from cogniverse_sdk.document import Document, ContentType

    # Initialize pipeline
    pipeline = IngestionPipeline()

    if content_type == "VIDEO":
        # Frame-based embeddings
        colpali_embeddings = encode_frames_colpali(content_path)

        # Chunk-based embeddings
        videoprism_embeddings = encode_chunks_videoprism(content_path)

        # Create document with both embedding types
        doc = Document(
            content_type=ContentType.VIDEO,
            content_path=content_path,
            embeddings={
                "colpali_frame": colpali_embeddings,
                "videoprism_chunk": videoprism_embeddings
            }
        )

    elif content_type == "IMAGE":
        # Image embeddings
        colqwen_embeddings = encode_image_colqwen(content_path)

        doc = Document(
            content_type=ContentType.IMAGE,
            content_path=content_path,
            embeddings={
                "colqwen": colqwen_embeddings
            }
        )

    elif content_type == "DOCUMENT":
        # Document embeddings (vision-based)
        doc_images = convert_pdf_to_images(content_path)
        colqwen_embeddings = encode_images_colqwen(doc_images)

        doc = Document(
            content_type=ContentType.DOCUMENT,
            content_path=content_path,
            embeddings={
                "colqwen": colqwen_embeddings
            }
        )

    elif content_type == "TEXT":
        # Text embeddings (CPU-based)
        text_embeddings = encode_text_cpu(content_path)

        doc = Document(
            content_type=ContentType.TEXT,
            content_data=read_text(content_path),
            embeddings={
                "text": text_embeddings
            }
        )

    # Ingest to Vespa
    pipeline.ingest_document(doc)
    return doc.id
```

---

## Optimization Strategies

### 1. Batch Processing

```python
# Optimize throughput with batching
@app.function(
    gpu="A100-40GB",
    allow_concurrent_inputs=100
)
def batch_encode_frames(frame_batches: List[List[Image]]):
    """Process multiple videos in parallel batches."""
    results = []
    for batch in frame_batches:
        embeddings = model.encode_images(batch)
        results.append(embeddings)
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
# For very large models (VideoPrism)
model.gradient_checkpointing_enable()
```

### 4. Model Quantization

```python
# 4-bit quantization for budget GPUs
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModel.from_pretrained(
    model_name,
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
- **Upgrade to A100-80GB** (required for production)
- Use H100-80GB for better performance

**ColQwen OOM:**
- Reduce batch size: 8 → 4 → 2
- Enable 4-bit quantization
- Reduce max sequence length
- Switch to L4 or A10G

### Slow Processing

**Check GPU utilization:**
```python
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
2. **VideoPrism (Chunk-Based Video)**: A100-80GB **required**, no smaller alternative
3. **ColQwen2 (Image/Document)**: L4 recommended, T4 budget option
4. **Multi-Modal Support**: All six content types supported (VIDEO, AUDIO, IMAGE, DOCUMENT, TEXT, DATAFRAME)
5. **Cost Efficiency**: ~$300-350 per 1000 videos one-time ingestion, ~$50-100/month search

**Recommended Production Setup:**
- **Ingestion**: A100-40GB (ColPali) + A100-80GB (VideoPrism) + L4 (ColQwen2)
- **Search**: T4 or L4 (or CPU for text-based)
- **Total**: ~$300-400 per 1000 videos ingestion, minimal ongoing costs

---

**For more information:**
- Installation Guide: [docs/operations/setup-installation.md](../operations/setup-installation.md)
- Ingestion Pipeline: [docs/architecture/ingestion-pipeline.md](../architecture/ingestion-pipeline.md)
- Model Documentation: [docs/modules/agents.md](../modules/agents.md)
