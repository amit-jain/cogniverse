# Vespa Tensor Storage Format Documentation

## Overview
This document describes how multi-vector embeddings are stored in Vespa for the Cogniverse system.

## Storage Structure

### 1. Schema Definition
Tensors are defined in the schema as:
```json
{
  "name": "embedding",
  "type": "tensor<bfloat16>(patch{}, v[128])"
}
```

### 2. Document Storage Format
When embeddings are sent to Vespa as a dictionary:
```python
# What we send
{
  "embedding": {
    0: "hex_encoded_vector",
    1: "hex_encoded_vector",
    2: "hex_encoded_vector",
    ...
  }
}
```

### 3. Internal Vespa Storage
Vespa internally stores tensors with additional metadata:
```json
{
  "embedding": {
    "type": "tensor<bfloat16>(patch{}, v[128])",
    "blocks": {
      "0": "hex_encoded_vector",
      "1": "hex_encoded_vector",
      "2": "hex_encoded_vector",
      ...
    }
  }
}
```

## Important Notes

### Patch Count Confusion
When querying the document API and checking the embedding structure:
- `jq '.fields.embedding | keys'` returns `["blocks", "type"]` - **these are NOT patches!**
- The actual patches are inside `blocks`
- To count patches: `jq '.fields.embedding.blocks | length'`

### Example: ColQwen Storage
- ColQwen generates 5474 patches for a 15-second video segment
- All 5474 patches ARE stored in Vespa
- Common mistake: seeing 2 keys ("blocks", "type") and thinking only 2 patches are stored

### Querying Embeddings
```bash
# Wrong way - counts structure keys, not patches
curl -s 'http://localhost:8080/document/v1/video/video_colqwen/docid/test_video_segment_0' | \
  jq '.fields.embedding | keys | length'  # Returns 2

# Right way - counts actual patches
curl -s 'http://localhost:8080/document/v1/video/video_colqwen/docid/test_video_segment_0' | \
  jq '.fields.embedding.blocks | length'  # Returns 5474
```

## Model-Specific Patch Counts

### ColPali (Frame-based)
- Generates ~32 patches per frame (varies by image content)
- Each patch: 128 dimensions

### ColQwen-Omni (Segment-based)
- Generates variable patches per segment (e.g., 5474 for 15 seconds)
- Each patch: 128 dimensions
- Processes video+audio together

### VideoPrism Base (Frame-based)
- Fixed 4096 patches per frame
- Each patch: 768 dimensions

### VideoPrism Large (Frame-based)
- Fixed 2048 patches per frame
- Each patch: 1024 dimensions

## Binary Embeddings
Binary embeddings follow the same structure but with int8 values:
```json
{
  "embedding_binary": {
    "type": "tensor<int8>(patch{}, v[16])",
    "blocks": {
      "0": "hex_encoded_binary",
      "1": "hex_encoded_binary",
      ...
    }
  }
}
```