# Profile Naming Convention

## Formula: `{embedding_type}__{modality}_{model}_{variant}`

Where variant is optional and describes the processing approach (e.g., `6s`, `30s`, `global`, `frames`)

### Embedding Types:
- **`colpali`**: Multi-vector patch-based embeddings (1024 patches × 128 dims)
- **`colvision`**: Multi-vector vision embeddings with adaptive patches
- **`single`**: Single vector embeddings (one embedding per segment/chunk)

### Modalities:
- **`video`**: Video content (all current work)
- **`image`**: Static images (future)
- **`text`**: Pure text content (future)

### Model Identifiers:
- **`smol500`**: ColPali vidore/colsmol-500m
- **`qwen`**: ColQwen-Omni vision model
- **`videoprism_base`**: VideoPrism base model (768 dims)
- **`videoprism_large`**: VideoPrism large model (1024 dims)
- **`videoprism_lvt_base`**: VideoPrism LVT base (global video, 768 dims)
- **`videoprism_lvt_large`**: VideoPrism LVT large (global video, 1024 dims)

## Current Profiles → New Names:

| Current Name | New Name | Description |
|-------------|----------|-------------|
| `frame_based_colpali` | `colpali__video_smol500` | Frame extraction + ColPali embeddings |
| `direct_video_colqwen` | `colvision__video_qwen` | Direct video processing with ColQwen |
| `direct_video_frame` | `single__video_videoprism_base_30s` | 30s segments with VideoPrism base |
| `direct_video_frame_large` | `single__video_videoprism_large_30s` | 30s segments with VideoPrism large |
| `direct_video_global` | `single__video_videoprism_lvt_base_global` | Global video embedding |
| `direct_video_global_large` | `single__video_videoprism_lvt_large_global` | Global video embedding large |
| `video_chunks_videoprism` | `single__video_videoprism_large_6s` | 6s chunks (TwelveLabs-style) |

## Benefits:
1. **Model Agnostic**: The chunking strategy (6s, 30s, frames) is independent of the model
2. **Clear Categories**: Immediately know the embedding type and modality
3. **Extensible**: Easy to add new models or modalities
4. **Consistent**: All profiles follow the same pattern

## Schema Naming:
Schemas should follow a similar pattern but focusing on the data structure:
- `video_multipatch`: For ColPali/ColVision (multiple patches)
- `video_single`: For single embeddings (chunks or global)
- `video_chunks`: Special case for TwelveLabs-style (multiple chunks in one doc)

## Examples:
- Want to use a different model for 6s chunks? Just change the model part:
  - `single__video_videoprism_large` → `single__video_custommodel`
- Want to apply ColPali to images?
  - `colpali__image_smol500`
- Want text embeddings?
  - `single__text_bert`