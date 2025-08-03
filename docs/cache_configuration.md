# Cache Configuration Guide

## Overview

The caching system stores expensive video processing artifacts (keyframes, transcripts, descriptions) to avoid recomputation. This guide explains the configuration options.

## Configuration

Add this to your `config.json`:

```json
"pipeline_cache": {
  "enabled": true,
  "backends": [
    {
      "backend_type": "filesystem",
      "base_path": "~/.cache/cogniverse/pipeline",
      "max_size_mb": 500000,
      "eviction_policy": "none",
      "enable_eviction": false,
      "priority": 0
    }
  ],
  "default_ttl": 0,
  "enable_compression": true,
  "serialization_format": "pickle"
}
```

## Configuration Options

### Global Options

- **`enabled`**: Enable/disable caching (default: `true`)
- **`default_ttl`**: Time to live in seconds. Use `0` for no expiration (default: `0`)
- **`enable_compression`**: Compress cached data (default: `true`)
- **`serialization_format`**: Format for serialization - `"pickle"`, `"json"`, `"msgpack"` (default: `"pickle"`)

### Backend Options

#### Filesystem Backend

- **`backend_type`**: Must be `"filesystem"`
- **`base_path`**: Cache directory path (default: `"~/.cache/cogniverse/pipeline"`)
- **`max_size_mb`**: Maximum cache size in MB (default: `500000` = 500GB)
- **`eviction_policy`**: Policy when full - `"lru"`, `"fifo"`, `"lfu"`, `"none"` (default: `"none"`)
- **`enable_eviction`**: Enable automatic eviction (default: `false`)
- **`priority`**: Backend priority, lower = higher priority (default: `0`)

## No-Eviction Setup

For unlimited caching without evictions:

```json
{
  "eviction_policy": "none",
  "enable_eviction": false,
  "default_ttl": 0,
  "max_size_mb": 500000
}
```

This configuration:
- Never deletes cached data automatically
- No expiration time
- 500GB soft limit (only for monitoring)

## Storage Estimates

Per video:
- Keyframes: ~5-10MB
- Transcript: ~50KB  
- Descriptions: ~100KB
- **Total**: ~5-11MB per video

For different video counts:
- 1,000 videos: ~5-11GB
- 10,000 videos: ~50-110GB
- 100,000 videos: ~500GB-1.1TB

## Migration from Existing Data

To migrate existing processed data to the cache:

```bash
# Run the migration script
python scripts/migrate_to_cache.py
```

This will:
1. Find all existing keyframes, transcripts, and descriptions
2. Copy them to the cache with proper metadata
3. Preserve the original files

## Cache Location

By default, cached data is stored in:
```
~/.cache/cogniverse/pipeline/
├── 00/
│   ├── 00a1b2c3.cache      # Cache file
│   └── 00a1b2c3.meta       # Metadata
├── 01/
├── ...
└── .metadata/              # Cache metadata
```

## Using the Cached Pipeline

```python
from src.processing.cached_video_pipeline import create_cached_pipeline

# Create pipeline with caching
pipeline = create_cached_pipeline(
    pipeline_config,
    profile="direct_video_colqwen"
)

# Process video - will use cache if available
results = pipeline.process_single_video(video_path, output_dir)

# Invalidate cache for a video if needed
await pipeline.invalidate_video_cache(video_path)

# Get cache statistics
stats = await pipeline.get_cache_stats()
```

## Benefits

1. **Performance**: 100-1000x faster for cached operations
2. **Resilience**: Resume processing after interruption
3. **Development**: Iterate quickly without reprocessing
4. **Cost**: Save on GPU compute for VLM and transcription

## Monitoring

Check cache usage:
```python
stats = await cache_manager.get_stats()
print(f"Cache size: {stats['manager']['size_bytes'] / 1e9:.2f} GB")
print(f"Total files: {stats['manager']['total_files']}")
```