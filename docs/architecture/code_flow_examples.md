# Code Flow Examples

## Processing Flow Example

### 1. Deploy Schemas (One-time or after schema changes)
```bash
$ python scripts/deploy_all_schemas.py
```

**What happens:**
```python
# deploy_all_schemas.py
schemas_dir = Path("configs/schemas")
for schema_file in schemas_dir.glob("*.json"):
    # Load and deploy each schema
    schema = parser.load_schema_from_json_file(schema_file)
    app_package.add_schema(schema)

# After deployment, automatically extract strategies
strategies = extract_all_ranking_strategies(schemas_dir)
save_ranking_strategies(strategies, schemas_dir / "ranking_strategies.json")
```

### 2. Process Videos
```bash
$ python scripts/run_ingestion.py --video_dir data/videos --backend vespa
```

**What happens:**
```python
# For each profile (e.g., "frame_based_colpali")
pipeline = VideoIngestionPipeline(config)

# Process videos → generates embeddings
embeddings = encoder.encode(frame)

# Get schema name from profile
schema_name = get_schema_for_profile("frame_based_colpali")  # → "video_frame"

# Load ranking strategies to understand tensor requirements
strategies = load_ranking_strategies()[schema_name]

# Determine what embeddings to generate based on strategies
needs_float = any(s.needs_float_embeddings for s in strategies.values())
needs_binary = any(s.needs_binary_embeddings for s in strategies.values())

# Get tensor dimensions from strategies
# e.g., strategies["float_float"].inputs → "tensor<float>(querytoken{}, v[128])"
tensor_dims = extract_tensor_dimensions(strategies)

# Format document with required fields
document = {
    "video_id": "abc123",
}

if needs_float:
    document["embedding"] = format_float_embeddings(embeddings, tensor_dims)

if needs_binary:
    binary_embeddings = generate_binary_embeddings(embeddings)
    document["embedding_binary"] = format_binary_embeddings(binary_embeddings, tensor_dims)

vespa_client.feed(schema_name, document)
```

## Query Flow Example

### 1. Initialize Search Backend
```python
from src.search import VespaSearchBackend

# Just specify schema - profile is auto-determined
backend = VespaSearchBackend(
    vespa_url="http://localhost",
    vespa_port=8080,
    schema_name="video_frame"
)
```

**What happens internally:**
```python
# Determine profile from schema
self.profile = get_profile_for_schema("video_frame")  # → "frame_based_colpali"

# Load ranking strategies (auto-generates if missing)
if not Path("configs/schemas/ranking_strategies.json").exists():
    strategies = extract_all_ranking_strategies(Path("configs/schemas"))
    save_ranking_strategies(strategies, ...)

# Load strategies for this schema
self.ranking_strategies = {
    "binary_binary": RankingConfig(...),
    "hybrid_binary_bm25": RankingConfig(...),
    # ... all strategies from schema
}
```

### 2. Execute Search
```python
# Encode query
encoder = QueryEncoderFactory.create_encoder("frame_based_colpali", "vidore/colsmol-500m")
embeddings = encoder.encode("person wearing winter clothes")

# Search with specific strategy
results = backend.search(
    query_embeddings=embeddings,
    query_text="person wearing winter clothes",
    ranking_strategy="hybrid_binary_bm25"
)
```

**What happens internally:**
```python
# Get strategy configuration
strategy = self.ranking_strategies["hybrid_binary_bm25"]
# strategy.needs_binary_embeddings = True
# strategy.needs_text_query = True

# Validate inputs
if strategy.needs_text_query and not query_text:
    raise ValueError("Strategy 'hybrid_binary_bm25' requires a text query")

# Build YQL based on strategy type
if strategy.strategy_type == SearchStrategyType.HYBRID:
    yql = "select ... from video_frame where userInput(@userQuery)"

# Format query body
query_body = {
    "yql": yql,
    "ranking.profile": "hybrid_binary_bm25",
    "userQuery": "person wearing winter clothes",
    "input.query(qtb)": format_binary_tensor(embeddings)  # For patch-based
}

# Send to Vespa
response = vespa.query(query_body)
```

## Key Integration Points

### 1. Schema → Profile Mapping
```python
# src/processing/vespa/schema_profile_mapping.py
SCHEMA_TO_PROFILE = {
    "video_frame": "frame_based_colpali",
    "video_videoprism_global": "direct_video_global",
}
```

### 2. Ranking Strategy Extraction
```python
# Happens automatically during:
# - Schema deployment (deploy_all_schemas.py)
# - Backend initialization (if missing)

extractor = RankingStrategyExtractor()
for schema_file in schemas_dir.glob("*.json"):
    strategies = extractor.extract_from_schema(schema_file)
    # Analyzes rank_profiles in schema JSON
    # Determines strategy type, requirements, etc.
```

### 3. Profile-Based Tensor Formatting
```python
# Global models (VideoPrism)
if profile_config.is_global:
    return embeddings.tolist()  # [1, 2, 3, ...]

# Patch-based models (ColPali)
else:
    return {"cells": [
        {"address": {"querytoken": "0", "v": "0"}, "value": 1.23},
        # ...
    ]}
```

## Error Examples

### Missing Embeddings
```python
backend.search(
    query_text="test",
    ranking_strategy="float_float"  # Requires embeddings
)
# ValueError: Strategy 'float_float' requires embeddings
```

### Unknown Strategy
```python
backend.search(
    query_embeddings=embeddings,
    ranking_strategy="unknown_strategy"
)
# ValueError: Unknown ranking strategy 'unknown_strategy' for schema 'video_frame'.
# Available strategies: ['binary_binary', 'hybrid_binary_bm25', ...]
```

### Unknown Schema
```python
backend = VespaSearchBackend(
    schema_name="non_existent_schema"
)
# ValueError: Cannot determine profile for schema non_existent_schema: Unknown schema
```

## The Complete Flow

1. **Schemas define everything** → `configs/schemas/*.json`
2. **Deployment extracts strategies** → `ranking_strategies.json` 
3. **Processing uses schema mapping** → profile → schema
4. **Query loads strategies** → validates → executes
5. **No manual steps** → Everything is automatic!