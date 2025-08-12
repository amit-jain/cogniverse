# ColPali/SmolVLM Architecture and Token Handling

## Overview

This document explains the architecture of ColPali models (particularly ColSmol/SmolVLM) and how they handle image tokenization, patch generation, and embedding creation. This is crucial for understanding how to properly visualize and work with ColPali embeddings.

## Key Concepts

### 1. Image Tokenization in ColPali

ColPali uses a Vision Language Model (VLM) approach where images and text share the same embedding space. This is the fundamental innovation that allows text queries to be compared directly with image patches using cosine similarity.

### 2. Token Structure

For a typical image processed by ColPali, the token structure is:

```
Total Tokens = System Tokens + Visual Patches
```

- **System Tokens**: ~42-51 tokens including:
  - `<|im_start|>`, `User:` - conversation markers
  - `<fake_token_around_image>` - image boundary marker
  - `<row_X_col_Y>` - position markers for image tiles
  
- **Visual Patches**: The actual image content tokens (`<image>` tokens)

## Image Processing Pipeline

### 1. Image Splitting (Tiling)

For high-resolution images, Idefics3 (the base model) splits images into tiles:

- **Max tile size**: 512×512 pixels (configurable via `max_image_size`)
- **Tiling strategy**: Images are split into overlapping tiles
- **Example**: A 1280×720 image generates 13 tiles (not the expected 6, due to overlapping)

### 2. Pixel Shuffle Compression

SmolVLM uses an aggressive compression technique called **pixel shuffle** (space-to-depth):

- **2×2 pixel shuffle**: Reduces tokens by 4×
- **3×3 pixel shuffle**: Reduces tokens by 9×
- **Mechanism**: Trades spatial resolution for channel depth
  - Takes a 2×2 block of pixels and converts to 4 channels
  - Halves both height and width of feature map

### 3. Token Generation per Image Size

The number of tokens varies based on image dimensions:

| Image Size | Total Tokens | Visual Patches | System Tokens |
|------------|--------------|----------------|---------------|
| 224×224    | 1139         | 1088          | 51            |
| 336×336    | 1139         | 1088          | 51            |
| 512×512    | 1139         | 1088          | 51            |
| 1280×720   | 874          | 832           | 42            |
| 1920×1080  | 874          | 832           | 42            |

**Key insight**: Square images get more visual patches than wide/HD images.

### 4. Patch Reduction Mathematics

For a 1280×720 image:
- **Tiles created**: 13 tiles of 512×512
- **Potential patches per tile**: 1024 (with 16×16 patch size)
- **Total potential patches**: 13,312
- **Actual visual patches**: 832
- **Reduction factor**: ~16×

This reduction is achieved through pixel shuffle compression, not by discarding information.

## Embedding Dimensions

### Document Embeddings
- **Shape**: (num_patches, 128)
- **For 1280×720**: (874, 128) including system tokens
- **Flattened storage**: 874 × 128 = 111,872 dimensions

### Query Embeddings
- **Shape**: (num_tokens, 128) 
- **Typical**: (14, 128) for short queries
- **Dimension**: 14 × 128 = 1,792 dimensions

## MaxSim Operation

ColPali uses late interaction with MaxSim scoring:

```python
# For each query token, find max similarity to any document patch
similarities = cosine_similarity(query_tokens, doc_patches)
max_sims = similarities.max(axis=1)  # Max for each query token
score = max_sims.mean()  # Average across query tokens
```

## Visualization Approach

### Correct Token Extraction

When visualizing ColPali embeddings:

1. **Extract only visual patches** (skip system tokens):
   - For 874 total tokens: Use tokens 42-873 (832 visual patches)
   - For 1139 total tokens: Use tokens 51-1138 (1088 visual patches)

2. **Project all patches together**:
   - Include ALL visual patches from documents (no sampling needed)
   - Include ALL query tokens
   - Use cosine metric for UMAP (matches ColPali's similarity metric)

3. **Understanding the compression**:
   - The 832 patches are NOT a subset - they're compressed representations
   - Each patch encodes information from multiple original pixels
   - Similar to JPEG compression - efficient encoding, not information loss

### Joint Embedding Space

The key innovation of ColPali is that text and image tokens share the same 128-dimensional embedding space:

- Text tokens and image patches can be directly compared
- Semantic similarity is meaningful across modalities
- This enables direct projection of queries alongside documents

## Storage Format in Vespa

Embeddings are stored as mixed tensors with patch indices:

```json
{
  "embedding": {
    "0": [128-dim vector],  // First patch
    "1": [128-dim vector],  // Second patch
    ...
    "873": [128-dim vector] // Last patch
  }
}
```

For binary quantization:
- Each 128-dim float vector → 16 bytes (128 bits packed)
- Stored as hex strings in Vespa

## Performance Implications

### Memory Efficiency

Thanks to pixel shuffle compression:
- SmolVLM 256M model: <1GB GPU RAM
- SmolVLM 2.2B model: ~5GB GPU RAM
- Compare to Qwen2-VL 2B: 13-16GB GPU RAM

### Trade-offs

With aggressive compression (e.g., pool factor of 3):
- 66.7% reduction in vectors
- 97.8% of original performance retained

## Common Misconceptions

1. **"We're losing visual information"**: No, pixel shuffle compresses but preserves information
2. **"874 tokens seems too few"**: It's by design for efficiency
3. **"Query and image tokens can't cluster together"**: They can - they share the same learned embedding space
4. **"We should sample patches"**: No need - use all patches for best results

## Implementation Notes

### For Developers

1. Always check token counts to determine image format
2. Skip appropriate number of system tokens based on total count
3. Use cosine similarity for all comparisons (as ColPali does)
4. When projecting, include all patches without sampling
5. Remember that reduction happens at encoding time, not visualization time

### For Researchers

1. The pixel shuffle factor can be adjusted for different trade-offs
2. Token pooling can further reduce sequence length if needed
3. The architecture supports variable resolution inputs
4. Overlapping tiles provide better context preservation than simple grid splitting

## References

- [ColPali Paper](https://arxiv.org/html/2407.01449v2)
- [SmolVLM Architecture](https://huggingface.co/blog/smolvlm)
- [Idefics3 Documentation](https://huggingface.co/docs/transformers/model_doc/idefics3)
- [Pixel Shuffle in Vision Models](https://paperswithcode.com/method/pixelshuffle)