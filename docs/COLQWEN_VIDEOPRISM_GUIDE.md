# Comprehensive Guide to ColPali, ColQwen, and VideoPrism in Cogniverse

## Table of Contents
1. [Model Architectures and Internals](#model-architectures)
2. [Key Architectural Differences](#architectural-differences)
3. [How They Process Different Modalities](#modality-processing)
4. [Implementation in Cogniverse](#implementation)
5. [Configuration and Tuning](#configuration-tuning)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

---

## Model Architectures and Internals

### ColPali Architecture

ColPali is a **vision-language model** specifically designed for document retrieval using visual embeddings:

**Architecture Details:**
- **Base Model**: PaliGemma-3b-mix-448 (3B parameters)
- **Vision Backbone**: Vision Transformer (ViT) with 448×448 input resolution
- **Language Model**: Gemma-2B with visual token integration
- **Output**: Multi-vector embeddings (one per visual patch)

**Key Components:**
1. **Vision Transformer (ViT)**:
   - Input resolution: 448×448 pixels (dynamic resolution supported)
   - Patch size: 14×14 pixels
   - Number of patches: Up to 1024 (32×32 grid)
   - Patch embedding dimension: 128 (in your config)

2. **Late Interaction Mechanism**:
   - Each patch produces an independent embedding
   - Query-document similarity computed via MaxSim
   - No pooling - preserves spatial information

3. **Training Approach**:
   - ColBERT-style contrastive learning
   - Trained on document-query pairs
   - Optimizes for retrieval performance

**Internal Processing Flow:**
```
Image → ViT Patches → Transformer Layers → Linear Projection → Multi-Vector Output
         (32×32)        (Self-Attention)     (to 128 dim)      (1024 vectors)
```

### ColQwen-Omni Architecture

ColQwen-Omni is a **multimodal retrieval model** built on the Qwen2.5-Omni series that processes video, audio, and text jointly:

**Architecture Details:**
- **Base Model**: Qwen2.5-Omni (3B parameters)
- **Vision Encoder**: Modified ViT with temporal awareness
- **Audio Encoder**: Whisper-style encoder for 16kHz audio
- **Text Decoder**: Qwen language model with multimodal tokens
- **Output**: Multi-vector embeddings for all modalities

**Key Components:**
1. **Vision Processing**:
   - Dynamic grid-based tokenization
   - Configurable patch size and pixel limits
   - Temporal positional embeddings for video
   - Max patches: 1024 (configurable)

2. **Audio Processing**:
   - 16kHz sampling rate, mono channel
   - Mel-spectrogram feature extraction
   - ~800 tokens per 30-second segment
   - Cross-attention with visual features

3. **Multimodal Fusion**:
   - Token replacement mechanism for modalities
   - Unified attention across all inputs
   - Preserves temporal alignment

**Internal Processing Flow:**
```
Video → Frame Sampling → Visual Tokens ─┐
                                        ├→ Multimodal Transformer → Multi-Vector Output
Audio → Mel-Spectrogram → Audio Tokens ─┘   (Cross-Attention)      (128 dim × N tokens)
```

**Technical Specifications:**
- Model size: 3B parameters
- Vision resolution: Dynamic (max_pixels: 16777216)
- Audio sampling: 16kHz mono
- Embedding dimension: 128
- Segment duration: 15s (configurable)
- Output: Multi-vector embeddings

### VideoPrism Architecture

VideoPrism is a **general-purpose video encoder** from Google DeepMind designed for comprehensive video understanding:

**Architecture Details:**
- **Base Architecture**: Vision Transformer (ViT) with temporal extensions
- **Variants**: Base (ViT-B) and Large (ViT-L)
- **Framework**: JAX/Flax implementation
- **Pre-training**: Massive scale (1B images + 618M video clips)
- **Output**: Dense frame-level embeddings

**Key Components:**
1. **Spatial Encoding (ViT)**:
   - Input resolution: 288×288 pixels (fixed)
   - Patch size: 18×18 pixels
   - Spatial patches per frame: 16×16 = 256
   - Total patches: 256 × num_frames

2. **Temporal Encoding**:
   - Learnable temporal positional embeddings
   - Interpolation for arbitrary frame counts
   - 3D attention (space + time) in later layers
   - Maintains temporal coherence

3. **Multi-Stage Architecture**:
   ```
   Stage 1: Spatial ViT encoding per frame
   Stage 2: Temporal attention across frames
   Stage 3: Joint spatiotemporal representation
   ```

**Model Specifications:**

| Model | Parameters | Embedding Dim | Spatial Tokens | Total Tokens (16 frames) |
|-------|------------|---------------|----------------|-------------------------|
| Base  | 114M       | 768          | 16×16×16=4096  | 65,536                  |
| Large | 354M       | 1024         | 8×16×16=2048   | 32,768                  |

**Internal Processing Flow:**
```
Video Input → Frame Extraction → Spatial Patches → Temporal Encoding → Dense Embeddings
(T×288×288×3)   (@ sampling_fps)   (256 per frame)   (Interpolated PE)    (T×256×D)
```

**Unique Features:**
- No audio processing (pure visual)
- Handles variable frame rates via interpolation
- Trained on diverse video tasks simultaneously
- Single frozen model adaptable to many tasks

---

## Key Architectural Differences

### Comparison Table

| Feature | ColPali | ColQwen-Omni | VideoPrism |
|---------|---------|--------------|------------|
| **Primary Use** | Document retrieval | Multimodal retrieval | Video understanding |
| **Modalities** | Images/Documents | Video + Audio + Text | Video only |
| **Base Model** | PaliGemma-3B | Qwen2.5-Omni-3B | Custom ViT |
| **Parameters** | 3B | 3B | 114M-354M |
| **Input Resolution** | 448×448 (dynamic) | Dynamic (max pixels) | 288×288 (fixed) |
| **Audio Support** | ❌ | ✅ (16kHz) | ❌ |
| **Temporal Modeling** | ❌ | ✅ (cross-modal) | ✅ (native) |
| **Output Type** | Multi-vector | Multi-vector | Dense embeddings |
| **Embedding Dim** | 128 | 128 | 768/1024 |
| **Max Tokens/Patches** | 1024 | 1024 | 4096/2048 |
| **Framework** | PyTorch | PyTorch | JAX/Flax |

### Architectural Design Choices

#### ColPali: Optimized for Visual Document Understanding
- **Why Multi-Vector?** Preserves spatial layout crucial for documents
- **Why No Pooling?** Text location matters in documents
- **Why PaliGemma?** Pre-trained on OCR and document tasks
- **Trade-offs**: Large number of vectors, no temporal understanding

#### ColQwen-Omni: True Multimodal Integration
- **Why Audio?** Many videos have crucial audio information
- **Why Cross-Attention?** Aligns audio events with visual content
- **Why Dynamic Resolution?** Handles diverse video content efficiently
- **Trade-offs**: Complex architecture, higher computational cost

#### VideoPrism: Pure Visual Excellence
- **Why Fixed Resolution?** Optimizes for consistent quality
- **Why JAX?** Better performance on TPUs, cleaner implementation
- **Why No Audio?** Focuses on visual understanding tasks
- **Trade-offs**: No audio understanding, requires frame extraction

### Key Internal Mechanisms

#### 1. **Attention Mechanisms**
- **ColPali**: 2D spatial attention only
- **ColQwen**: Cross-modal attention (audio↔video)
- **VideoPrism**: 3D spatiotemporal attention

#### 2. **Positional Encodings**
- **ColPali**: 2D learned positional embeddings
- **ColQwen**: Separate positional encodings per modality
- **VideoPrism**: Interpolatable temporal positions

#### 3. **Token Generation**
- **ColPali**: Fixed grid (32×32 patches)
- **ColQwen**: Dynamic based on pixel budget
- **VideoPrism**: Fixed 16×16 per frame

---

## How They Process Different Modalities

### ColPali Processing Pipeline

#### Image/Document Processing
```python
# ColPali processes single images/documents:
1. Image Loading: Load at native resolution
2. Dynamic Padding: Pad to maintain aspect ratio
3. Patch Extraction: Divide into 14×14 pixel patches
4. Vision Encoding: Process through ViT layers
5. Linear Projection: Project to 128-dim embeddings
6. Output: 1024 vectors (32×32 grid)
```

**Key Processing Parameters:**
- No temporal processing (single frame only)
- Preserves spatial structure completely
- Each patch is independent (no pooling)
- Optimized for text-heavy images

### ColQwen-Omni Modality Processing

#### Video Processing
```python
# ColQwen processes video with these steps:
1. Frame Sampling: Extract frames at specified FPS (default: 1 fps)
2. Audio Extraction: Extract audio track at 16kHz mono
3. Multimodal Fusion: Process video+audio jointly
4. Patch Generation: Create visual patches (up to 1024)
5. Token Generation: ~800 audio tokens per 30s
6. Multi-vector Output: One embedding per patch/token
```

**In Your Implementation:**
```python
# From configs/config.json
"direct_video_colqwen": {
  "segment_duration": 15.0,      # 15-second segments
  "adaptive_segmentation": true,  # Adjust based on content
  "max_pixels": 16777216,        # Max pixels per segment
  "sampling_fps": 1.0            # 1 frame per second
}
```

#### Audio Processing
- **Chunking**: 30-second segments (configurable to 15s in your setup)
- **Preprocessing**: Resample to 16kHz, convert stereo→mono
- **Tokenization**: Converts audio waveform to tokens
- **Direct Retrieval**: No transcription needed!

### VideoPrism Modality Processing

#### Video Processing Pipeline
```python
# VideoPrism processing flow:
1. Frame Extraction: Sample frames from video
2. Spatial Resizing: Resize to 288×288
3. Normalization: Scale pixels to [0, 1]
4. Temporal Encoding: Interpolate positional embeddings
5. Vision Transformer: Process through ViT layers
6. Patch Embeddings: Generate spatial tokens
```

**In Your Implementation:**
```python
# From videoprism_loader.py
def preprocess_frames(self, frames):
    # Resize to 288x288
    frame_resized = cv2.resize(frame, (288, 288))
    # Normalize to [0, 1]
    frame_normalized = frame_resized.astype(np.float32) / 255.0
```

---

## Implementation in Cogniverse

### Current Architecture Overview

Your system implements a **configurable pipeline** with three main profiles:

1. **Frame-based ColPali** (Traditional approach)
2. **Direct Video ColQwen** (Audio-visual understanding)
3. **Direct Video VideoPrism** (Pure visual understanding)

### Key Implementation Details

#### 1. Video Processing Pipeline (`embedding_generator_v2/`)

**Document Structure** - Universal representation for all media:
```python
@dataclass
class Document:
    media_type: MediaType  # VIDEO, VIDEO_FRAME, etc.
    document_id: str
    source_id: str
    raw_embeddings: Union[np.ndarray, Dict[str, np.ndarray]]
    temporal_info: Optional[TemporalInfo]
    segment_info: Optional[SegmentInfo]
    metadata: Dict[str, Any]
```

#### 2. ColQwen Implementation

**Audio-Enabled Processing** (`colqwen_audio_processor.py`):
```python
class ColQwenAudioEnabledProcessor(ColQwen2_5OmniProcessor):
    def process_videos_with_audio(self, videos):
        # Enables audio extraction from videos
        # Processes both visual and audio content
        # Returns multi-modal embeddings
```

**Key Features:**
- Processes 15-second segments (configurable)
- Adaptive segmentation based on pixel budget
- Audio transcription included
- Multi-vector output (up to 1024 patches)

#### 3. VideoPrism Implementation

**JAX-based Processing** (`videoprism_loader.py`):
```python
class VideoPrismLoader:
    # Uses JAX/Flax for inference
    # Handles arbitrary frame counts
    # Generates 4096 (base) or 2048 (large) spatial tokens
    # CPU backend to avoid Metal issues
```

**Key Features:**
- 30-second segments by default
- No audio processing
- Pure visual embeddings
- Larger embedding dimensions (768/1024)

### Backend Integration

**Vespa Storage**:
- Handles multi-vector embeddings
- Supports bfloat16 tensor storage
- Hex encoding for efficient storage
- Multiple ranking strategies

**Remote Inference Support** (New!):
```python
# Configure remote endpoints for model offloading
"remote_inference_url": "https://your-endpoint.com",
"remote_inference_api_key": "your-key",
"remote_inference_provider": "infinity"  # or "modal"
```

---

## Configuration and Tuning

### Key Configuration Knobs

#### 1. Segment Duration
```json
"segment_duration": 15.0  // ColQwen: 15s, VideoPrism: 30s
```
- **Shorter segments** (5-15s): Better temporal precision, more documents
- **Longer segments** (30-60s): Better context, fewer documents

#### 2. Frame Sampling Rate
```json
"sampling_fps": 1.0  // Frames per second to extract
```
- **Higher FPS** (2-5): More visual detail, larger embeddings
- **Lower FPS** (0.5-1): Faster processing, smaller storage

#### 3. Max Patches/Pixels
```json
"max_patches": 1024,      // ColQwen
"max_pixels": 16777216    // ColQwen adaptive mode
```
- Controls visual detail vs. processing speed
- Adaptive mode adjusts dynamically

#### 4. Audio Processing
```json
"transcribe_audio": true,  // Enable for ColQwen
"use_audio": true         // In processor config
```
- Only relevant for ColQwen
- Adds ~800 tokens per 30s audio

### Model-Specific Configuration Knobs

#### ColPali Configuration
```json
{
  "embedding_model": "vidore/colsmol-500m",  // or larger models
  "batch_size": 32,                          // Memory vs speed
  "embedding_dim": 128,                      // Fixed by model
  "num_patches": 1024                        // Max spatial resolution
}
```

**Tuning Knobs:**
- **Model Size**: `colsmol-500m` (fast) vs `colpali-v1.2` (accurate)
- **Batch Size**: Higher = faster but more memory
- No temporal parameters (frame-based only)

#### ColQwen Configuration
```json
{
  "segment_duration": 15.0,        // Length of video segments
  "max_patches": 1024,            // Visual detail level
  "max_pixels": 16777216,         // Pixel budget per segment
  "adaptive_segmentation": true,   // Dynamic adjustment
  "sampling_fps": 1.0,            // Frame extraction rate
  "use_audio": true               // Enable audio processing
}
```

**Tuning Knobs:**
- **segment_duration**: 5-30s (shorter = more precise, longer = more context)
- **max_pixels**: Controls quality vs memory trade-off
- **sampling_fps**: 0.5-5.0 (depends on content dynamics)
- **adaptive_segmentation**: Let model optimize segment boundaries

#### VideoPrism Configuration
```json
{
  "segment_duration": 30.0,        // Fixed segment length
  "sampling_fps": 1.0,            // Frame rate
  "max_frames": 30,               // Frames per segment
  "min_frames": 8,                // Minimum frames
  "use_cpu": true,                // Avoid Metal issues
  "num_patches": 4096             // Base: 4096, Large: 2048
}
```

**Tuning Knobs:**
- **segment_duration**: Usually 30s (model trained on this)
- **max_frames**: 8-64 (memory vs temporal detail)
- **Model variant**: Base (faster) vs Large (better quality)

### Quality Optimization Strategies

#### For Better Retrieval Quality:

1. **Increase Segment Overlap**:
```python
# Add overlap between segments (not yet implemented)
overlap_duration = 5.0  # 5-second overlap
```

2. **Use Adaptive Segmentation**:
```json
"adaptive_segmentation": true,
"max_pixels": 16777216  // Adjust based on GPU memory
```

3. **Optimize Frame Sampling**:
```json
// For fast-moving content
"sampling_fps": 2.0

// For static content (presentations)
"sampling_fps": 0.5
```

4. **Choose Right Model for Content**:
- **ColPali**: Best for documents, slides, text-heavy content
- **ColQwen**: Best for content with important audio (lectures, interviews)
- **VideoPrism**: Best for pure visual content (action, surveillance)

#### Model Selection Guide:

| Content Type | Best Model | Why |
|--------------|------------|-----|
| Lectures/Talks | ColQwen | Audio + slides |
| Documents/PDFs | ColPali | Optimized for text |
| Music Videos | ColQwen | Audio-visual sync |
| Sports/Action | VideoPrism | Temporal dynamics |
| Surveillance | VideoPrism | Pure visual analysis |
| Presentations | ColPali | Static slides |
| Interviews | ColQwen | Speech important |

#### For Better Performance:

1. **Batch Processing**:
```json
"batch_size": 32  // Adjust based on GPU memory
```

2. **Remote Inference**:
```python
# Offload to dedicated GPU servers
"remote_inference_url": "https://modal.run/your-endpoint"
```

3. **Reduce Visual Resolution**:
```python
# For ColQwen, reduce max patches
"max_patches": 512  // Instead of 1024
```

---

## Performance Optimization

### Memory Management

**Current Issues & Solutions:**

1. **High Memory Usage with VideoPrism**:
   - Solution: Use CPU backend (`"use_cpu": true`)
   - Trade-off: Slower but stable

2. **Multi-Vector Storage**:
   - Each document can have 1000s of vectors
   - Solution: Use binary quantization for secondary index

### Processing Speed

**Optimization Techniques:**

1. **Parallel Processing**:
```python
"max_workers": 4  // Adjust based on CPU cores
```

2. **Caching**:
```python
# Models are cached after first load
_model_cache[model_name] = (model, processor)
```

3. **Immediate Feeding**:
```python
# Process and feed segments immediately
# Avoids memory buildup
self._feed_single_document(document)
```

### Storage Optimization

**Vespa Schema Optimizations:**

1. **Use bfloat16**:
```json
"type": "tensor<bfloat16>(patch{}, v[768])"
```

2. **Binary Embeddings**:
```json
// Secondary index for fast filtering
"embedding_binary": "tensor<int8>(patch{}, v[96])"
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. "Expected 768 values, but got 384" Error
**Cause**: Wrong tensor format (hex vs float)
**Solution**: Ensure bfloat16 schema and hex encoding
**Model-Specific**: Only affects VideoPrism (768/1024 dims)

#### 2. High Memory Usage

**By Model:**
- **ColPali**: Reduce batch_size (default 32→16)
- **ColQwen**: Reduce max_pixels or segment_duration
- **VideoPrism**: Reduce max_frames or use Base model

**Solutions:**
```python
# ColPali
"batch_size": 16  # Instead of 32

# ColQwen
"max_pixels": 8388608  # Half of default
"segment_duration": 10.0  # Shorter segments

# VideoPrism
"max_frames": 16  # Instead of 30
```

#### 3. Poor Retrieval Quality

**Model-Specific Issues:**

| Issue | ColPali | ColQwen | VideoPrism |
|-------|---------|---------|------------|
| Missing text | Use higher resolution model | Enable frame sampling | N/A - no OCR |
| Missing audio | N/A | Check audio enabled | N/A |
| Temporal confusion | Extract more frames | Reduce segment duration | Increase frame rate |
| Low precision | Use larger model | Adjust max_patches | Use Large model |

#### 4. Slow Processing

**Model-Specific Optimizations:**

```python
# ColPali (CPU-bound)
- Use smaller model (colsmol-500m)
- Enable GPU if available
- Batch processing crucial

# ColQwen (Memory-bound)
- Reduce max_pixels
- Use adaptive segmentation
- Enable remote inference

# VideoPrism (Compute-bound)
- Use Base model instead of Large
- Reduce frame count
- Use CPU backend for stability
```

### Debugging Tools

1. **Check Embeddings Shape**:
```python
logger.info(f"Embeddings shape: {embeddings.shape}")
# ColPali: (1024, 128) - fixed grid
# ColQwen: (variable, 128) - depends on content
# VideoPrism Base: (4096, 768)
# VideoPrism Large: (2048, 1024)
```

2. **Monitor Memory Usage**:
```python
import psutil
process = psutil.Process()
logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

3. **Profile Processing Time**:
```python
import time
start = time.time()
# ... processing ...
logger.info(f"Processing took: {time.time() - start:.2f}s")
```

---

## Model-Specific Optimization Tips

### ColPali Optimizations
1. **For Document-Heavy Content**:
   - Use larger models for better OCR
   - Process at highest resolution possible
   - Consider pre-processing to enhance text

2. **For Mixed Content**:
   - Extract frames at scene changes
   - Use with OCR post-processing

### ColQwen Optimizations
1. **For Audio-Critical Content**:
   - Increase segment overlap for continuity
   - Use shorter segments (10-15s)
   - Ensure audio track is clean

2. **For Visual-Audio Sync**:
   - Enable adaptive segmentation
   - Use higher sampling_fps
   - Monitor cross-attention weights

### VideoPrism Optimizations
1. **For Action/Motion**:
   - Increase frame rate (2-5 fps)
   - Use longer segments (30s)
   - Enable temporal attention layers

2. **For Static Content**:
   - Reduce frame rate (0.5 fps)
   - Use Base model for efficiency
   - Consider ColPali instead

## Future Improvements

Based on the architectures and your TODOs:

1. **Temporal-Aware Queries**: 
   - Test ColQwen's audio-visual alignment
   - Evaluate VideoPrism's temporal coherence
   - Compare with ColPali's static performance

2. **VideoPrism Text Encoder**: 
   - Implement using `vp.load_text_tokenizer`
   - Enable text-guided video retrieval
   - Compare with ColQwen's text understanding

3. **Hybrid Approaches**:
   - Use ColPali for keyframe analysis
   - Combine with ColQwen for audio
   - Leverage VideoPrism for motion

4. **Dynamic Model Selection**:
   - Analyze content characteristics
   - Auto-route to optimal model
   - Implement ensemble strategies

---

## References

1. [ColQwen-Omni Blog Post](https://huggingface.co/blog/manu/colqwen-omni-omnimodal-retrieval)
2. [VideoPrism GitHub](https://github.com/google-deepmind/videoprism)
3. [Qwen2-VL Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/qwen2_vl)
4. [Audio RAG Tutorial](https://github.com/ManuelFay/Tutorials/blob/main/Practical_3_AudioRAG.ipynb)
5. [VideoPrism Colab Demo](https://colab.research.google.com/github/google-deepmind/videoprism/blob/main/videoprism/colabs/videoprism_video_text_demo.ipynb)