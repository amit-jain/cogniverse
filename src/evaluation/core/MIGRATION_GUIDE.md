# Migration Guide: From Video-Specific to Generic Evaluation

## Overview
The evaluation system has been refactored to be schema-driven and domain-agnostic. Video-specific functionality is now provided through plugins.

## Key Changes

### 1. Ground Truth Extraction

**Before (Video-specific):**
```python
from src.evaluation.core.ground_truth import BackendMetadataGroundTruthStrategy

strategy = BackendMetadataGroundTruthStrategy()
result = await strategy.extract_ground_truth(trace_data, backend)
# Returns: {"expected_videos": [...]}
```

**After (Generic):**
```python
from src.evaluation.core.ground_truth_generic import SchemaAwareGroundTruthStrategy

strategy = SchemaAwareGroundTruthStrategy()
result = await strategy.extract_ground_truth(trace_data, backend)
# Returns: {"expected_items": [...], "expected_videos": [...]}  # Both for compatibility
```

### 2. Scorers

**Before:**
```python
from src.evaluation.core.scorers import (
    ragas_context_relevancy_scorer,
    custom_diversity_scorer
)

# Hardcoded to look for video_id
scorers = [
    ragas_context_relevancy_scorer(),
    custom_diversity_scorer()
]
```

**After:**
```python
from src.evaluation.core.scorers_generic import (
    generic_relevance_scorer,
    generic_diversity_scorer,
    get_configured_scorers
)

# Automatically adapts to schema
config = {
    "use_relevance": True,
    "use_diversity": True,
    "scorer_plugins": ["src.evaluation.plugins.video"]  # Optional video-specific
}
scorers = get_configured_scorers(config)
```

### 3. Reranking

**Before:**
```python
from src.evaluation.core.reranking import DiversityRerankingStrategy

# Hardcoded video_id comparison
strategy = DiversityRerankingStrategy()
```

**After:**
```python
from src.evaluation.core.reranking_generic import get_reranking_strategy

# Schema-aware item extraction
config = {
    "lambda": 0.5,
    "schema_name": schema_name,
    "schema_fields": schema_fields
}
strategy = get_reranking_strategy("diversity", config)
```

### 4. Registering Video Plugin

To get video-specific behavior:

```python
from src.evaluation.plugins import register_video_plugin

# Register at startup
register_video_plugin()

# Or auto-register from config
from src.evaluation.plugins import auto_register_plugins

config = {
    "evaluation": {
        "plugins": ["video"]
    }
}
auto_register_plugins(config)
```

## Migration Steps

### Step 1: Update Imports

Replace old imports:
```python
# Old
from src.evaluation.core.ground_truth import BackendMetadataGroundTruthStrategy
from src.evaluation.core.scorers import custom_diversity_scorer
from src.evaluation.core.reranking import DiversityRerankingStrategy

# New
from src.evaluation.core.ground_truth_generic import SchemaAwareGroundTruthStrategy
from src.evaluation.core.scorers_generic import generic_diversity_scorer
from src.evaluation.core.reranking_generic import DiversityRerankingStrategy
```

### Step 2: Update Configuration

Add schema information to config:
```python
config = {
    "ground_truth_strategy": "backend",
    # No more hardcoded fields needed
    
    "scorers": {
        "use_relevance": True,
        "use_diversity": True,
        "scorer_plugins": ["src.evaluation.plugins.video"]  # If video-specific needed
    },
    
    "reranking": {
        "strategy": "diversity",
        "lambda": 0.5
    }
}
```

### Step 3: Update Result Handling

Handle both generic and domain-specific field names:
```python
# Old
expected_videos = result["expected_videos"]

# New - check both
expected = result.get("expected_items") or result.get("expected_videos", [])
```

### Step 4: Register Plugins (Optional)

If you need video-specific behavior:
```python
# In your initialization code
from src.evaluation.plugins import register_video_plugin

def initialize_evaluation():
    register_video_plugin()  # Adds video-specific analyzers
    # ... rest of initialization
```

## Benefits of Migration

1. **Schema Agnostic**: Works with any search domain (documents, images, products)
2. **Extensible**: Easy to add new domains through plugins
3. **Discoverable**: Automatically discovers schema fields
4. **Backward Compatible**: Still supports video-specific fields when video plugin is loaded
5. **Cleaner**: No hardcoded assumptions about data structure

## Testing Migration

Run tests to ensure compatibility:
```bash
# Test generic functionality
pytest src/evaluation/tests/test_generic_evaluation.py

# Test with video plugin
pytest src/evaluation/tests/test_video_plugin.py

# Test backward compatibility
pytest src/evaluation/tests/test_migration_compat.py
```

## Troubleshooting

### Issue: "expected_videos" not found
**Solution**: Use `expected_items` or register video plugin

### Issue: Item IDs not extracted correctly
**Solution**: Check schema analyzer is appropriate for your schema

### Issue: Temporal scoring not working
**Solution**: Ensure schema has temporal_fields defined

### Issue: Video-specific queries not analyzed properly
**Solution**: Register video plugin at startup

## Example: Complete Migration

**Before:**
```python
async def evaluate_video_search():
    from src.evaluation.core.ground_truth import BackendMetadataGroundTruthStrategy
    from src.evaluation.core.scorers import custom_diversity_scorer
    
    strategy = BackendMetadataGroundTruthStrategy(
        metadata_fields=["tags", "category"]  # Hardcoded
    )
    
    ground_truth = await strategy.extract_ground_truth(trace, backend)
    videos = ground_truth["expected_videos"]
    
    scorer = custom_diversity_scorer()  # Assumes video_id
    score = await scorer(state)
```

**After:**
```python
async def evaluate_search():
    from src.evaluation.core.ground_truth_generic import SchemaAwareGroundTruthStrategy
    from src.evaluation.core.scorers_generic import generic_diversity_scorer
    from src.evaluation.plugins import register_video_plugin
    
    # Optional: Register video plugin for video-specific features
    register_video_plugin()
    
    strategy = SchemaAwareGroundTruthStrategy()  # No hardcoding
    
    ground_truth = await strategy.extract_ground_truth(trace, backend)
    items = ground_truth["expected_items"]  # Generic field
    
    scorer = generic_diversity_scorer()  # Adapts to schema
    score = await scorer(state)
```

## Next Steps

1. Gradually migrate existing code using this guide
2. Remove old video-specific modules after migration
3. Create plugins for other domains (documents, images, etc.)
4. Update documentation to reflect generic approach