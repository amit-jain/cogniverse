# Visual Evaluator Configuration

## Overview
The visual evaluator uses multimodal LLMs to evaluate video search results by analyzing actual video frames.

## Current Configuration

### Frame Extraction
The system now extracts multiple frames from videos for evaluation:
- **frames_per_video**: 4 frames (configurable in config.json)
- **max_videos**: 3 top videos (configurable in config.json)
- Total frames sent: 12 frames to the multimodal model

### Supported Providers

1. **Ollama** (Local)
   - Model: llava:7b
   - Endpoint: http://localhost:11434
   - Supports multiple frames in single request

2. **Modal** (Remote)
   - Model: qwen2-vl
   - Custom endpoint deployment required

3. **OpenAI** 
   - Model: gpt-4-vision-preview
   - Requires API key

## Configuration in config.json

```json
"evaluators": {
  "visual_judge": {
    "provider": "ollama",
    "model": "llava:7b",
    "base_url": "http://localhost:11434",
    "api_key": null,
    "frames_per_video": 4,
    "max_videos": 3
  }
}
```

## Usage

```bash
# Run with visual evaluator
uv run python scripts/run_experiments_with_visualization.py \
  --profiles frame_based_colpali \
  --strategies binary_binary \
  --llm-evaluators \
  --evaluator visual_judge \
  --dataset-name test_visual_eval_v2
```

## How It Works

1. **Frame Extraction**: For each search result, the evaluator:
   - Locates the video file in `data/testset/evaluation/sample_videos/`
   - Extracts evenly-spaced frames (4 per video by default)
   - Converts frames to base64-encoded images

2. **Evaluation**: The multimodal LLM:
   - Receives the search query and extracted frames
   - Analyzes visual content for relevance
   - Provides a score (0-10) and reasoning
   - Results are stored in Phoenix for analysis

## Limitations

- Most multimodal LLMs work with images, not full videos
- Sending entire videos would require:
  - Video-native models (rare)
  - Much larger context windows
  - Significantly more processing time
- Current frame extraction provides good coverage while being practical

## Performance

With current settings (4 frames × 3 videos = 12 frames):
- Processing time: ~10 seconds per query with LLaVA
- Good balance between coverage and speed
- Captures temporal variations within videos