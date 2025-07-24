# Video Processing Pipeline Documentation

This directory contains documentation for the unified video processing pipeline that powers the Cogniverse video search system.

## Pipeline Overview

The unified video processing pipeline (`src/processing/unified_video_pipeline.py`) is a configurable system that processes video files through multiple stages to create searchable indexes. It replaces the previous collection of separate scripts with a single, cohesive pipeline.

## Quick Start

```bash
# Process videos and create search index
python scripts/run_ingestion.py --video_dir data/videos --backend byaldi

# Process specific video with custom configuration
python scripts/run_ingestion.py --video_dir sample_videos --backend byaldi --config config.json
```

## Pipeline Architecture

### Core Components

1. **VideoIngestionPipeline** - Main pipeline orchestrator
2. **KeyframeExtractor** - Extracts representative frames from videos
3. **AudioTranscriber** - Transcribes audio with timestamps
4. **VLMDescriptor** - Generates visual descriptions using Modal VLM service
5. **EmbeddingGenerator** - Creates vector embeddings for search

### Pipeline Configuration

The pipeline behavior is controlled via `config.json`:

```json
{
  "pipeline_config": {
    "extract_keyframes": true,     // Extract representative frames
    "transcribe_audio": true,      // Generate audio transcripts
    "generate_descriptions": true, // Generate VLM descriptions
    "generate_embeddings": true    // Generate vector embeddings
  },
  "vlm_endpoint_url": "https://your-modal-vlm-endpoint.modal.run",
  "batch_size": 500,
  "timeout": 10800
}
```

## Pipeline Steps

### 1. Keyframe Extraction
- **Purpose**: Extract representative frames from videos
- **Method**: Histogram comparison to identify scene changes
- **Output**: Frame images and metadata with timestamps
- **Configuration**: `extract_keyframes: true`

### 2. Audio Transcription
- **Purpose**: Generate searchable text from audio
- **Method**: Faster-Whisper with word-level timestamps
- **Output**: Transcript segments with timing information
- **Configuration**: `transcribe_audio: true`

### 3. Visual Description Generation
- **Purpose**: Generate textual descriptions of visual content
- **Method**: Modal VLM service (Qwen2-VL-7B-Instruct)
- **Output**: Detailed descriptions for each keyframe
- **Configuration**: `generate_descriptions: true`

### 4. Vector Embedding Generation
- **Purpose**: Create semantic embeddings for search
- **Method**: ColPali model for multi-modal understanding
- **Output**: Vector embeddings for similarity search
- **Configuration**: `generate_embeddings: true`

## Data Structure

### Input
- Video files in supported formats (MP4, MOV, AVI, etc.)
- Placed in configured video directory (default: `data/videos/`)

### Output Structure
```
output_dir/
├── keyframes/
│   └── video_id/
│       ├── frame_001.jpg
│       ├── frame_002.jpg
│       └── ...
├── audio/
│   └── video_id.json
├── descriptions/
│   └── video_id.json
├── embeddings/
│   └── video_id.json
└── metadata/
    └── video_id.json
```

### Metadata Format
```json
{
  "video_id": "unique_video_identifier",
  "video_path": "/path/to/video.mp4",
  "duration": 120.5,
  "fps": 24.0,
  "keyframes": [
    {
      "frame_id": 1,
      "timestamp": 0.0,
      "path": "/path/to/keyframes/video_id/frame_001.jpg"
    }
  ],
  "processing_status": {
    "keyframes_extracted": true,
    "audio_transcribed": true,
    "descriptions_generated": true,
    "embeddings_generated": true
  }
}
```

## Caching and Resume Capability

The pipeline automatically detects and skips already-processed data:

- **Keyframes**: Skips if frame directory exists with expected files
- **Audio**: Skips if transcript file exists with valid segments
- **Descriptions**: Skips if descriptions file exists with content
- **Embeddings**: Skips if embedding file exists with vectors

This allows for:
- **Resuming interrupted processing**
- **Processing new videos without reprocessing existing ones**
- **Selective reprocessing by deleting specific output files**

## Commands Reference

### Basic Usage
```bash
# Process all videos in directory
python scripts/run_ingestion.py --video_dir data/videos --backend byaldi

# Process with custom output directory
python scripts/run_ingestion.py --video_dir videos --backend byaldi --output_dir processed_data

# Process single video
python scripts/run_ingestion.py --video_dir single_video --backend byaldi
```

### Advanced Options
```bash
# Use specific configuration file
python scripts/run_ingestion.py --video_dir data/videos --backend byaldi --config custom_config.json

# Skip specific steps (configure in config.json)
# Set extract_keyframes: false to skip keyframe extraction
# Set transcribe_audio: false to skip audio transcription
# etc.
```

## Integration with Search Backends

### Byaldi (Development)
- Lightweight vector database for rapid prototyping
- Automatic index creation and management
- Suitable for small to medium datasets

### Vespa (Production)
- Scalable vector database for production use
- Hybrid search capabilities (vector + text)
- Advanced filtering and ranking

## Performance Considerations

### Processing Speed
- **Keyframe extraction**: ~1-2 minutes per hour of video
- **Audio transcription**: ~2-5 minutes per hour of video
- **VLM descriptions**: ~10-30 minutes per hour of video (depends on Modal GPU)
- **Embedding generation**: ~5-10 minutes per hour of video

### Resource Usage
- **CPU**: Moderate usage during keyframe extraction and audio transcription
- **GPU**: Used by Modal VLM service (cloud-based)
- **Memory**: ~4-8GB RAM during processing
- **Storage**: ~50-100MB per hour of video for processed data

## Troubleshooting

### Common Issues

1. **Modal VLM Service Not Running**
   - The pipeline automatically starts the Modal service if needed
   - Check `config.json` for correct `vlm_endpoint_url`
   - Verify Modal CLI is authenticated: `modal setup`

2. **Empty Descriptions Generated**
   - Check Modal service logs for errors
   - Verify endpoint URL is correct and accessible
   - Ensure sufficient Modal credits

3. **Audio Transcription Failures**
   - Verify Faster-Whisper installation
   - Check video has audio track
   - Ensure sufficient disk space

4. **Pipeline Hangs or Crashes**
   - Check timeout settings in config
   - Verify sufficient RAM (16GB+ recommended)
   - Monitor disk space usage

### Debug Commands
```bash
# Test VLM service connectivity
curl -X POST your-modal-endpoint -d '{"frame_base64":"..."}'

# Check pipeline configuration
python -c "from src.tools.config import get_config; print(get_config().dict())"

# Validate processed data
python -c "import json; print(json.load(open('output_dir/metadata/video_id.json')))"
```

## See Also

- **[Setup Guide](../setup/detailed_setup.md)** - Complete system setup instructions
- **[Modal VLM Guide](../modal/deploy_modal_vlm.md)** - VLM service deployment
- **[CLAUDE.md](../../CLAUDE.md)** - System architecture and development guidelines
- **[Archive Documentation](../../archive/README.md)** - Information about replaced scripts