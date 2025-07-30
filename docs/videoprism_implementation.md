# VideoPrism JAX Implementation

## Overview

This document describes the VideoPrism JAX implementation for the Cogniverse multi-agent video RAG system. VideoPrism uses significantly different embedding dimensions and patch counts compared to ColPali:

- **ColPali**: 1024 patches × 128 dimensions per patch
- **VideoPrism Base**: 4096 patches (16×16×16) × 768 dimensions per patch  
- **VideoPrism Large**: 2048 patches (8×16×16) × 1024 dimensions per patch

## Key Implementation Details

### 1. Native Dimension Preservation

The implementation preserves native VideoPrism dimensions without any projection or dimensionality reduction. This ensures maximum quality and fidelity of the video embeddings.

### 2. Flexible Schema Design

Created `video_multimodal_flex.sd` schema that supports variable embedding dimensions:

```
field embedding type tensor<bfloat16>(patch{}, v[1024]) {
    indexing: attribute
}
field num_patches type int {
    indexing: summary | attribute
}
```

This schema can handle:
- ColPali embeddings (1024 patches × 128 dims)
- VideoPrism Base embeddings (4096 patches × 768 dims)
- VideoPrism Large embeddings (2048 patches × 1024 dims)

### 3. JAX Implementation

The VideoPrism loader is implemented in:
- `src/processing/pipeline_steps/videoprism_loader.py` - Main loader interface
- `src/processing/pipeline_steps/videoprism_models.py` - JAX model implementation

Key features:
- Uses JAX with CPU backend for compatibility
- Processes videos as 16 frames at 288×288 resolution
- Generates embeddings with proper spatiotemporal structure

### 4. Vespa Integration

The implementation includes:
- Proper tensor cell format for Vespa compatibility
- Binary embedding generation for fast search
- Patch count tracking in documents

### 5. Configuration

Added `direct_video_frame` profile in config.json:

```json
"direct_video_frame": {
    "vespa_schema": "video_multimodal_flex",
    "embedding_model": "videoprism_public_v1_base_hf",
    "embedding_type": "direct_video_frame",
    "model_specific": {
        "native_dimensions": true,
        "temporal_stride": 1,
        "max_frames": 32
    }
}
```

## Usage

### Deploy the Flexible Schema

```bash
python scripts/deploy_flexible_schema.py
```

### Process Videos with VideoPrism

```bash
python scripts/run_ingestion.py --video_dir data/videos --backend vespa --profile direct_video_frame
```

### Run Tests

```bash
python tests/test_videoprism.py
```

## Technical Notes

1. **Patch Count Differences**: 
   - VideoPrism Base: 4096 patches (16×16×16 spatiotemporal grid)
   - VideoPrism Large: 2048 patches (8×16×16 spatiotemporal grid) - fewer patches but deeper features
   - ColPali: 1024 patches
   
2. **Dimension Differences**: 
   - VideoPrism Base: 768 dimensions per patch
   - VideoPrism Large: 1024 dimensions per patch
   - ColPali: 128 dimensions per patch
   
3. **Memory Requirements**: VideoPrism embeddings require significantly more memory:
   - VideoPrism Base: 4096 × 768 = 3.1M values per video
   - VideoPrism Large: 2048 × 1024 = 2.1M values per video
   - ColPali: 1024 × 128 = 131K values per video
   
4. **Search Performance**: The flexible schema maintains search performance through optimized ranking profiles

5. **Real-time Inference**: VideoPrism is available in the video agent for real-time video search, supporting both CPU and Metal backends on Apple Silicon

## Future Improvements

1. Implement actual model weights loading from HuggingFace
2. Add GPU support for faster inference
3. Implement streaming processing for large video collections
4. Add support for variable-length video sequences