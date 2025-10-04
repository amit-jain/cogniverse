# Production-Ready Pipeline Caching System

## Overview

This caching system is designed to cache expensive video processing artifacts:
- **Keyframes**: Extracted frames and metadata
- **Audio Transcripts**: Whisper transcriptions
- **Frame Descriptions**: VLM-generated descriptions
- **Embeddings**: Model-generated embeddings (optional)

## Architecture

### Core Components

1. **CacheBackend** - Abstract interface for storage backends
2. **CacheManager** - Manages multiple cache tiers
3. **PipelineArtifactCache** - Specialized cache for pipeline artifacts
4. **Storage Backends**:
   - FilesystemCacheBackend (implemented)
   - S3CacheBackend (stub for cloud storage)
   - RedisBackend (future)

### Key Features

- **Multi-tier Caching**: Local filesystem + cloud storage
- **Artifact-aware**: Understands video processing artifacts
- **TTL Support**: Automatic expiration
- **Size Management**: Automatic eviction when full
- **Async Operations**: Non-blocking I/O
- **Compression**: Optional compression for artifacts

## Usage Examples

### 1. Basic Pipeline Integration

```python
from src.cache import CacheConfig, FilesystemCacheConfig, S3CacheConfig
from src.cache.pipeline_cache import PipelineArtifactCache

# Configure multi-tier cache
cache_config = CacheConfig(
    backends=[
        # Fast local cache
        FilesystemCacheConfig(
            base_path="~/.cache/cogniverse/pipeline",
            max_size_mb=50000,  # 50GB
            eviction_policy="lru",
            priority=0  # Highest priority
        ),
        # Durable cloud storage
        S3CacheConfig(
            bucket="cogniverse-pipeline-cache",
            prefix="artifacts",
            priority=1  # Lower priority
        )
    ],
    default_ttl=604800,  # 7 days
    enable_compression=True
)

# Create cache manager and pipeline cache
cache_manager = CacheManager(cache_config)
pipeline_cache = PipelineArtifactCache(cache_manager)
```

### 2. Caching Keyframes

```python
# Check cache before extraction
cached_keyframes = await pipeline_cache.get_keyframes(
    video_path="path/to/video.mp4",
    threshold=0.999,
    max_frames=3000
)

if cached_keyframes:
    print("Using cached keyframes")
    return cached_keyframes

# Extract and cache
keyframes_metadata, keyframe_images = extract_keyframes(video_path)
await pipeline_cache.set_keyframes(
    video_path,
    keyframes_metadata,
    keyframe_images,
    threshold=0.999,
    max_frames=3000
)
```

### 3. Caching Audio Transcripts

```python
# Check cache
cached_transcript = await pipeline_cache.get_transcript(
    video_path="path/to/video.mp4",
    model_size="base",
    language="en"
)

if cached_transcript:
    return cached_transcript

# Transcribe and cache
transcript = transcribe_audio(video_path)
await pipeline_cache.set_transcript(
    video_path,
    transcript,
    model_size="base",
    language="en"
)
```

### 4. Complete Pipeline with Caching

```python
async def process_video_with_cache(video_path: str, pipeline_config: dict):
    # Get all cached artifacts
    artifacts = await pipeline_cache.get_all_artifacts(
        video_path, 
        pipeline_config
    )
    
    # Check what's missing
    if not artifacts.is_complete(pipeline_config):
        # Process only what's missing
        if pipeline_config["extract_keyframes"] and not artifacts.keyframes:
            artifacts.keyframes = await extract_and_cache_keyframes(video_path)
        
        if pipeline_config["transcribe_audio"] and not artifacts.audio_transcript:
            artifacts.audio_transcript = await transcribe_and_cache_audio(video_path)
        
        if pipeline_config["generate_descriptions"] and not artifacts.frame_descriptions:
            artifacts.frame_descriptions = await generate_and_cache_descriptions(video_path)
    
    return artifacts
```

## Cache Key Strategy

Cache keys are generated deterministically based on:
- Video file path + size + modification time
- Processing parameters (threshold, model, etc.)

Example keys:
```
video:a3f4b2c1:keyframes:threshold=0.999:max_frames=3000
video:a3f4b2c1:transcript:model=base:lang=en
video:a3f4b2c1:descriptions:model=Qwen2-VL:batch_size=500
```

## Performance Benefits

1. **Keyframe Extraction**: 
   - Original: 30-60s per video
   - Cached: <100ms

2. **Audio Transcription**:
   - Original: 10-30s per video (depending on length)
   - Cached: <50ms

3. **VLM Descriptions**:
   - Original: 60-300s per video (GPU dependent)
   - Cached: <50ms

## Storage Estimates

For a typical video:
- Keyframes: ~5-10MB (images + metadata)
- Transcript: ~50KB
- Descriptions: ~100KB
- Total: ~5-11MB per video

For 10,000 videos:
- Storage: ~50-110GB
- With compression: ~30-70GB

## Future Enhancements

1. **Redis Backend**: For distributed caching
2. **Cache Warming**: Pre-populate cache during off-peak hours
3. **Smart Invalidation**: Detect when source video changes
4. **Partial Caching**: Cache segments for long videos
5. **Compression Optimization**: Different algorithms for different artifacts
6. **Metrics Dashboard**: Visualize cache performance