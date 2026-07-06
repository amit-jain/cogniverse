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

### Important: ColPali, ColQwen3, image and document-visual profiles share one sidecar

The `video_colpali_smol500_mv_frame` (frame video), `video_colqwen_omni_mv_chunk_30s`
(chunk video, legacy `ColQwen2` schema label), `image_colpali_mv`, and
`document_visual_colpali` backend profiles all set
`embedding_model: TomoroAI/tomoro-colqwen3-embed-4b` — the **same** model,
served remotely by **one** vLLM `ColPaliForRetrieval` sidecar
(`RemoteColPaliLoader`). Local in-process loading of this model is
unsupported (`_is_colqwen3()` raises before any `from_pretrained` call); the
GPU sizing below is for that shared sidecar, not per-profile. Only size a
second GPU if you deploy the smaller local-loading fallback checkpoints
(`vidore/colpali-v1.2`, `vidore/colpali-v1.3-hf`, `vidore/colqwen2-v0.1`) for
a given profile instead of the shared sidecar.

VideoPrism (chunk and LVT single-vector profiles), CLAP (audio semantic
embeddings), and LateOn/LateOn-Code-edge (document and code text
embeddings) are separate model families with their own sidecars — see
[Model Requirements](#model-requirements) below.

---

## Quick Recommendations

### Production Setup (Recommended)
- **ColPali / ColQwen3 / Image / Document-visual (shared Tomoro sidecar)**: A100-40GB (~$3.00/hour for ingestion, minimal for search)
- **VideoPrism (Chunk-Based)**: L4 (~$1.00/hour for ingestion)
- **CLAP (Audio Semantic)**: CPU only — no GPU required (see [CLAP](#4-clap-audio-semantic-embeddings))
- **LateOn / LateOn-Code-edge (Document & Code Text)**: L4 or T4 (~$1.00/hour for ingestion)
- **Total**: ~$200-250 one-time ingestion per 1000 videos + minimal search costs

### Budget Setup
- **ColPali / ColQwen3 (shared sidecar)**: L4 GPU (~$1.00/hour, tight fit)
- **VideoPrism**: T4 (~$0.60/hour, base model only)
- **LateOn**: T4 (~$0.60/hour)
- **Total**: ~$120-130 per 1000 videos ingestion

### Performance Setup
- **ColPali / ColQwen3 (shared sidecar)**: A100-80GB (~$3.20/hour, optimal throughput)
- **VideoPrism**: A100-40GB (~$3.00/hour, large batches)
- **LateOn**: A10G (~$1.10/hour)
- **Total**: ~$300-350 per 1000 videos ingestion

---

## Model Requirements

### 1. ColPali (Frame-Based Video Embeddings)

**Model:** `TomoroAI/tomoro-colqwen3-embed-4b` (production default, served remotely
via a vLLM `ColPaliForRetrieval` sidecar). Local in-process loading of this
model is **unsupported** — `model_loaders._is_colqwen3()` raises
`RuntimeError` before any `from_pretrained` call ("ColQwen3/Tomoro models
are remote-only"), because the architecture (`qwen3_vl`) needs
`transformers>=4.57`, which the pinned `transformers==4.56.2` (capped for
pylate) cannot build. `vidore/colpali-v1.2` / `vidore/colpali-v1.3-hf`
remain supported for local, in-process loading (`ColPaliModelLoader`) as a
smaller fallback when no remote inference URL is configured — this GPU
guide covers both.
**Content Types:** VIDEO (frames), IMAGE, DOCUMENT

#### Memory Requirements

**TomoroAI/tomoro-colqwen3-embed-4b (production, remote vLLM sidecar, 4B params)**
- Model weights: ~8GB (4B params × 2 bytes, bfloat16)
- Image preprocessing: ~2GB per batch
- Patch embeddings: ~4GB (1024 patches × 320 dims × batch size)
- CUDA overhead: ~2GB
- **Total: ~16GB minimum, 24GB recommended** — sized for the shared vLLM sidecar, not the calling ingestion function

**vidore/colpali-v1.2 / vidore/colpali-v1.3-hf (local fallback, ~3B params)**
- Model weights: ~6GB (3B params × 2 bytes, bfloat16)
- Preprocessing: ~2GB per batch
- Patch embeddings: ~3GB
- **Total: ~12GB minimum, 16GB recommended**

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
    """Generate ColPali embeddings using colpali-engine API (local fallback
    checkpoint; production traffic hits the remote vLLM sidecar instead —
    see RemoteColPaliLoader)."""
    from colpali_engine.models import ColIdefics3, ColIdefics3Processor
    import torch

    # Avoid device_map= here: it routes through accelerate's meta-tensor
    # dispatch, which raises NotImplementedError on repeated loads in the
    # same process (see ColPaliModelLoader.load_model).
    model = ColIdefics3.from_pretrained(
        "vidore/colpali-v1.2",
        torch_dtype=torch.bfloat16,
    ).eval().to("cuda")
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
    """Generate ColPali embeddings with the v1.3-hf checkpoint for budget GPUs."""
    from colpali_engine.models import ColIdefics3, ColIdefics3Processor
    import torch

    model = ColIdefics3.from_pretrained(
        "vidore/colpali-v1.3-hf",
        torch_dtype=torch.bfloat16,
    ).eval().to("cuda")
    processor = ColIdefics3Processor.from_pretrained("vidore/colpali-v1.3-hf")

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
def encode_video_chunks_videoprism(video_path: str):
    """Generate VideoPrism embeddings using the custom JAX loader."""
    import jax
    from pathlib import Path
    from cogniverse_core.common.models.model_loaders import get_or_load_model

    jax.config.update('jax_platform_name', 'gpu')

    # get_or_load_model returns (videoprism_loader, None) for VideoPrism models
    config = {"model_name": "videoprism_public_v1_base_hf"}
    loader, _ = get_or_load_model("videoprism_public_v1_base_hf", config, logger)

    # The loader provides extract_embeddings(frames) for pre-extracted frames,
    # or process_entire_video(video_path, sampling_fps) for full video processing
    result = loader.process_entire_video(Path(video_path), sampling_fps=1.0)
    return result["embeddings"]

# High-performance ingestion on Modal
@app.function(
    image=videoprism_image,
    gpu="A100-40GB",
    memory=40000,
    timeout=3600
)
def encode_video_chunks_performance(video_path: str):
    """Same pipeline on A100 for higher throughput."""
    import jax
    from pathlib import Path
    from cogniverse_core.common.models.model_loaders import get_or_load_model

    jax.config.update('jax_platform_name', 'gpu')

    config = {"model_name": "videoprism_public_v1_base_hf"}
    loader, _ = get_or_load_model("videoprism_public_v1_base_hf", config, logger)

    result = loader.process_entire_video(Path(video_path), sampling_fps=1.0)
    return result["embeddings"]
```

#### LVT (single-vector) variants

The `video_videoprism_lvt_base_sv_chunk_6s` and
`video_videoprism_lvt_large_sv_chunk_6s` profiles use
`videoprism_lvt_public_v1_base` / `videoprism_lvt_public_v1_large` — global
(single-vector) embeddings over 6-second chunks instead of 30-second
multi-vector chunks. `VideoPrismModelLoader` detects `_lvt_` in the model
name and additionally loads a text encoder (`loader.load_text_encoder()`)
so the same embedding space can be queried with text, which adds to the
memory footprint above (allow the same GPU tier as the "Large" row, since
the LVT large text encoder is loaded alongside the vision tower).

---

### 3. ColQwen3 (Multi-Modal Image/Document Embeddings)

**Model:** `TomoroAI/tomoro-colqwen3-embed-4b` (production default — the same
model backing the ColPali profile above, served remotely via the same vLLM
sidecar; the `video_colqwen_omni_mv_chunk_30s` chunk-based video profile
keeps the legacy `ColQwen2` schema/model label but also resolves to this
model). `ColQwenQueryEncoder`'s local-loading default is
`vidore/colqwen-omni-v0.1`, but the `"omni"` branch of
`ColQwenModelLoader.load_model()` imports `ColQwen2_5Omni` /
`ColQwen2_5OmniProcessor`, which the pinned `colpali-engine==0.3.13` does
not export — that local path currently fails with `ImportError`. For a
working local fallback use the non-omni `vidore/colqwen2-v0.1` checkpoint
with `ColQwen2` / `ColQwen2Processor` (both present in 0.3.13), shown below.
**Content Types:** IMAGE, DOCUMENT, TEXT, VIDEO (chunks, 30s)

#### Memory Requirements

**TomoroAI/tomoro-colqwen3-embed-4b (4B params)**
- Model weights: 4B params × 2 bytes (FP16) = 8GB
- Image preprocessing: ~4GB
- KV cache: ~4GB (for context length 4K)
- CUDA overhead: ~2GB
- **Total: ~18GB minimum, 24GB recommended**

**With 4-bit Quantization:**

- Model weights: ~2GB (quantized)

- Preprocessing: ~4GB

- **Total: ~8GB minimum, 12GB recommended**

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
    """Generate ColQwen embeddings using colpali-engine API (local fallback
    checkpoint; production traffic hits the shared vLLM sidecar instead)."""
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    import torch

    model = ColQwen2.from_pretrained(
        "vidore/colqwen2-v0.1",
        torch_dtype=torch.bfloat16,
    ).eval().to("cuda")
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

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
        "vidore/colqwen2-v0.1",
        quantization_config=quantization_config,
    ).eval().to("cuda")
    processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v0.1")

    # Smaller batch for T4 (batch_size=4)
    batch_inputs = processor.process_images(images).to(model.device)

    with torch.no_grad():
        embeddings = model(**batch_inputs)

    return embeddings.cpu().to(torch.float32).numpy()
```

---

### 4. CLAP (Audio Semantic Embeddings)

**Model:** `laion/clap-htsat-unfused`, served by the `clap_embed` FastAPI
sidecar (`libs/runtime/cogniverse_runtime/sidecars/clap_embed.py`), used by
the `audio_clap_semantic` backend profile.
**Content Types:** AUDIO

#### GPU Requirements: none

The `clap_embed` sidecar image (`deploy/clap_embed/Dockerfile`) installs
CPU-only `torch` from the PyTorch CPU wheel index specifically to keep this
service GPU-free — CLAP (~150M params) runs fast enough on CPU for the
ingestion and query-time volumes this profile sees. Deploy it as a CPU-only
Modal function/container; no GPU line item is needed for this model.

#### API Shape

- `POST /embed/audio` — body `{"audio_b64": "..."}`, returns `{"vec": [512 floats]}` (`ClapModel.get_audio_features`)
- `POST /embed/text` — body `{"text": "..."}`, returns the matching 512-dim text vector (`get_text_features`), used by the audio-analysis agent to encode acoustic-mode queries
- `GET /health` — liveness probe

---

### 5. LateOn / LateOn-Code-edge (Document & Code Text Embeddings)

**Model:** `lightonai/LateOn` (backing the `document_text_semantic` and
`lateon_mv` profiles) and `lightonai/LateOn-Code-edge` (backing
`code_lateon_mv`) — ColBERT-family multi-vector text encoders, served
remotely (pylate sidecar or vLLM with `--hf-overrides
ColBERTModernBertModel`; see `cogniverse_core.query.encoders`).
**Content Types:** DOCUMENT, TEXT

#### GPU Options

| GPU | Memory | Price/Hour | Performance | Best For |
|-----|--------|------------|-------------|----------|
| **L4** ✅ | 24GB | $1.00 | Good | **Production** |
| **A10G** | 24GB | $1.10 | Good | Alternative to L4 |
| **T4** | 16GB | $0.60 | Fair | Small batches |

`inference_health_check.py` probes the deployed service at boot and refuses
to start if it serves a different model than the profile's
`embedding_model` expects — a mismatched sidecar would otherwise silently
produce wrong-but-valid-shaped 128-dim token vectors.

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
   - Split video into 30-second chunks (base/large) or 6-second chunks (LVT single-vector)
   - Encode temporal context
   - Best for: Coarse-grained search, video-level understanding
   - GPU: L4 (recommended) or T4 (base model)
   - Time: ~3-8 minutes per 1-hour video

**Recommended:** Use both approaches for comprehensive coverage.

### AUDIO Content

- Transcribed with `openai/whisper-large-v3-turbo` via the remote `vllm_asr` inference service (`AudioTranscriptionStrategy`); text transcripts indexed via BM25 text search
- Separately, semantically embedded with CLAP (`laion/clap-htsat-unfused`) for acoustic similarity search (`audio_clap_semantic` profile)
- GPU: whisper-large-v3-turbo needs a GPU-backed `vllm_asr` sidecar (L4/T4-class); CLAP embedding is CPU-only (see [CLAP](#4-clap-audio-semantic-embeddings))

### IMAGE Content

- Single images or extracted frames
- Processed with ColPali/ColQwen3 (shared `TomoroAI/tomoro-colqwen3-embed-4b` sidecar)
- GPU: L4 or A10G (on the shared sidecar)
- Time: ~100-200ms per image

### DOCUMENT Content

- PDF, DOCX, text extraction
- Convert to images for vision models (ColPali/ColQwen3, `document_visual_colpali` profile) — GPU-backed sidecar (L4 or A10G)
- Or use text embeddings via LateOn (`lightonai/LateOn`, `document_text_semantic` profile) — also GPU-backed (L4/T4), not CPU-based
- GPU: L4 for either the vision-based or LateOn text-based path

### TEXT Content

- Natural language text
- Embedded with DenseOn (`lightonai/DenseOn`) via a remote OpenAI-compatible endpoint when configured, or locally with `sentence-transformers/all-mpnet-base-v2` (`get_semantic_embedder()`) — used for agent memory, semantic caching, and wiki embeddings
- GPU: Optional — the local `all-mpnet-base-v2` fallback (~420MB) runs fine on CPU for typical batch sizes; DenseOn's remote endpoint runs GPU-backed if served via vLLM

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
| ColPali/ColQwen3 (shared Tomoro sidecar: frames, chunks, image, document-visual) | A100-40GB | 50 | $3.00 | $150 |
| VideoPrism (chunks) | L4 | 60 | $1.00 | $60 |
| LateOn (document/code text) | L4 | 20 | $1.00 | $20 |
| CLAP (audio semantic) | CPU only | — | $0.00 | $0 |
| **Total** | | | | **$230** |

**Budget Alternative** (local fallback checkpoint instead of the shared sidecar):
| Model | GPU | Hours | Cost/Hour | Total Cost |
|-------|-----|-------|-----------|------------|
| vidore/colpali-v1.3-hf (local fallback, frames) | L4 | 80 | $1.00 | $80 |
| VideoPrism (chunks) | T4 | 80 | $0.60 | $48 |
| **Total** | | | | **$128** |

**Per 10,000 Images:**
| Model | GPU | Hours | Cost/Hour | Total Cost |
|-------|-----|-------|-----------|------------|
| ColPali/ColQwen3 (shared sidecar) | L4 | 3 | $1.00 | $3 |

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
            tenant_id="your_org:production",
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
        # convert_pdf_to_images is a user-supplied utility (e.g. pdf2image);
        # not part of cogniverse_core — shown here for illustration only.
        doc_images = convert_pdf_to_images(content_path)
        colqwen_embeddings = encode_images_colqwen(doc_images)

        doc = Document(
            content_type=ContentType.DOCUMENT,
            content_path=content_path,
        )
        doc.add_embedding("colqwen", colqwen_embeddings)
        return doc.id

    elif content_type == "TEXT":
        # encode_text_cpu/read_text are user-supplied helpers illustrating
        # the DenseOn/all-mpnet-base-v2 path (get_semantic_embedder()) —
        # not literal cogniverse_core function names.
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
        "vidore/colpali-v1.3-hf", torch_dtype=torch.bfloat16
    ).eval().to("cuda")
    processor = ColIdefics3Processor.from_pretrained("vidore/colpali-v1.3-hf")

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
    "vidore/colqwen2-v0.1",
    quantization_config=quantization_config
)
```

---

## Troubleshooting

### Out of Memory (OOM) Errors

**ColPali OOM:**

- Reduce batch size: 32 → 16 → 8 → 4

- Use the smaller local-fallback checkpoint (`vidore/colpali-v1.3-hf`, ~3B) instead of the production `TomoroAI/tomoro-colqwen3-embed-4b` (~4B) — see [Memory Requirements](#memory-requirements)

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

- Use the smaller local-fallback checkpoint (`vidore/colpali-v1.3-hf`) instead of the shared production sidecar model

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

1. **ColPali / ColQwen3 (Frame Video, Chunk Video, Image, Document-Visual)**: one shared `TomoroAI/tomoro-colqwen3-embed-4b` vLLM sidecar — A100-40GB recommended, L4 budget option, local `vidore/colpali-v1.2`/`v1.3-hf`/`colqwen2-v0.1` fallback for smaller deployments
2. **VideoPrism (Chunk-Based Video)**: L4 recommended (114M params base, also 354M large and LVT single-vector variants), T4 budget option
3. **CLAP (Audio Semantic)**: CPU only — no GPU needed
4. **LateOn / LateOn-Code-edge (Document & Code Text)**: L4 recommended, T4 budget option
5. **Multi-Modal Support**: All six content types supported (VIDEO, AUDIO, IMAGE, DOCUMENT, TEXT, DATAFRAME)
6. **Cost Efficiency**: ~$200-250 per 1000 videos one-time ingestion, ~$50-100/month search

**Recommended Production Setup:**

- **Ingestion**: A100-40GB (shared ColPali/ColQwen3 sidecar) + L4 (VideoPrism) + L4 (LateOn) + CPU (CLAP)

- **Search**: T4 or L4 (or CPU for text-based)

- **Total**: ~$200-250 per 1000 videos ingestion, minimal ongoing costs

---

**For more information:**

- Installation Guide: [docs/operations/setup-installation.md](../operations/setup-installation.md)

- Ingestion Pipeline: [docs/modules/ingestion.md](../modules/ingestion.md)

- Model Documentation: [docs/modules/agents.md](../modules/agents.md)
