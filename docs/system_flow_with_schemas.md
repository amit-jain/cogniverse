# Complete System Flow with Schema Generation

## Overview

The Cogniverse system is a multi-agent video RAG system that supports multiple embedding models with different dimensions. Here's how everything works together:

## 1. Configuration Structure

```
config.json
├── active_video_profile: "frame_based_colpali"  # Current active profile
├── video_processing_profiles:
│   ├── frame_based_colpali:
│   │   ├── vespa_schema: "video_colpali"
│   │   ├── embedding_model: "vidore/colsmol-500m"
│   │   └── schema_config:
│   │       ├── embedding_dim: 128
│   │       └── num_patches: 1024
│   ├── direct_video_colqwen:
│   │   ├── vespa_schema: "video_colqwen"
│   │   └── embedding_dim: 128
│   ├── direct_video_frame: (VideoPrism Base)
│   │   ├── vespa_schema: "video_videoprism_base"
│   │   └── embedding_dim: 768
│   └── direct_video_frame_large: (VideoPrism Large)
│       ├── vespa_schema: "video_videoprism_large"
│       └── embedding_dim: 1024
```

## 2. Schema Generation Flow

### Step 1: Generate Schemas from Template
```bash
python scripts/generate_schema_from_template.py
```

This reads `config.json` and generates:
- `schemas/video_colpali.sd` (128 dims)
- `schemas/video_colqwen.sd` (128 dims)
- `schemas/video_videoprism_base.sd` (768 dims)
- `schemas/video_videoprism_large.sd` (1024 dims)

### Step 2: Deploy Schema to Vespa
```bash
# Deploy the schema for your chosen profile
vespa deploy schemas/video_colpali.sd  # or video_videoprism_base.sd, etc.
```

## 3. Multi-Agent System Startup

### When you run `./scripts/run_servers.sh vespa`:

1. **Checks Vespa is running** (starts it if needed)
2. **Starts Video Agent** on port 8001
   - Loads embedding model based on active profile
   - For VideoPrism: Initializes JAX inference engine
   - For ColPali/ColQwen: Loads PyTorch model
3. **Starts Text Agent** on port 8002 (if enabled)
4. **Starts Composing Agent** on port 8000
   - Routes queries to appropriate agents
   - Handles user interface

## 4. Video Processing/Ingestion Flow

### When you run `python scripts/run_ingestion.py --video_dir data/videos --backend vespa --profile direct_video_frame`:

1. **Profile Activation**:
   ```python
   os.environ["VIDEO_PROFILE"] = "direct_video_frame"
   # This selects VideoPrism Base configuration
   ```

2. **Schema Selection**:
   - Reads profile config → `vespa_schema: "video_videoprism_base"`
   - Uses schema with 768-dimensional embeddings

3. **Model Loading**:
   - For VideoPrism: Loads JAX model with native 768 dims
   - For ColPali: Loads PyTorch model with 128 dims

4. **Processing Pipeline**:
   ```
   Video Files → Keyframe Extraction → Embedding Generation → Vespa Storage
                                           ↓
                                    Uses model & schema from profile
   ```

5. **Vespa Document Creation**:
   - Embeddings stored with exact dimensions (no padding)
   - Binary embeddings: dimension / 8
   - Metadata includes `embedding_type`, `num_patches`

## 5. Query/Search Flow

### When a user queries through the Composing Agent:

1. **Query Analysis**: GLiNER determines if it's a video query
2. **Agent Routing**: Forwards to Video Agent (port 8001)
3. **Model Selection**: Video Agent uses active profile's model
4. **Embedding Generation**: 
   - Text query → embeddings (using appropriate model)
   - Dimensions match the schema (128, 768, or 1024)
5. **Vespa Search**: Searches in the profile's schema/index
6. **Results**: Returns ranked video segments

## 6. Key Points

### Multi-Model Support:
- Each model has its own schema with exact dimensions
- No padding needed - dimensions match model output
- Binary embeddings properly sized (dim/8)

### Profile Switching:
```bash
# Option 1: Set in config.json
"active_video_profile": "direct_video_frame"

# Option 2: Command line
python scripts/run_ingestion.py --profile direct_video_frame

# Option 3: Environment variable
export VIDEO_PROFILE=direct_video_frame
```

### Agent Communication:
```
User → Composing Agent → Video Agent → Vespa
                      ↘ Text Agent → Elasticsearch
```

### Schema Management:
1. Template in `schemas/video_multimodal_template.sd`
2. Generated schemas in `schemas/` directory
3. Each profile uses its designated schema
4. Schemas are NOT automatically deployed - manual deployment needed

## 7. Complete Workflow Example

```bash
# 1. Generate schemas for all profiles
python scripts/generate_schema_from_template.py

# 2. Deploy schema for VideoPrism Base
vespa deploy schemas/video_videoprism_base.sd

# 3. Start the multi-agent system
./scripts/run_servers.sh vespa

# 4. Process videos with VideoPrism Base
python scripts/run_ingestion.py \
  --video_dir data/videos \
  --backend vespa \
  --profile direct_video_frame

# 5. Query through web UI at http://localhost:8000
# The system will use VideoPrism Base for search
```

## 8. Architecture Benefits

1. **Flexibility**: Support multiple models without schema migrations
2. **Performance**: Each model uses optimal dimensions
3. **Modularity**: Agents can be updated independently
4. **Scalability**: Can run multiple agents with different models

The system maintains the original multi-agent architecture while adding support for multiple embedding models through the profile system.