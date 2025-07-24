# Frame Boundary Algorithm

This document describes how keyframe temporal boundaries are calculated in the video processing pipeline.

## Overview

The system combines visual scene change detection with audio transcription to create semantically meaningful frame boundaries. Each keyframe represents a visual scene that may span multiple audio segments or partial segments.

## Algorithm Components

### 1. Visual Keyframe Extraction

**Location**: `archive/src/processing/video_ingestion_pipeline.py` (lines 148-196)

The keyframe extraction uses histogram comparison to detect visual scene changes:

```python
def extract_keyframes(video_path: str, threshold: float = 0.98) -> List[Tuple[Image.Image, float, float]]:
    """
    Extracts keyframes and their temporal boundaries from a video.
    
    Returns: List of (frame_image, start_time, end_time) tuples
    """
    keyframes = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    prev_hist = None
    start_time = 0.0
    
    while cap.isOpened():
        frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = frame_pos / fps
        current_hist = cv2.calcHist([frame_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        
        if prev_hist is not None:
            similarity = cv2.compareHist(prev_hist, current_hist, cv2.HISTCMP_CORREL)
            if similarity < threshold:  # Scene change detected
                end_time = current_time
                keyframes.append((frame_image, start_time, end_time))
                start_time = current_time  # Start of next scene
```

**Key Points**:
- Uses histogram correlation to detect scene changes (threshold typically 0.98)
- Each keyframe gets `start_time` and `end_time` based on when visual scenes change
- A keyframe's duration represents how long that visual scene persists
- Scene changes are detected when histogram correlation drops below threshold

### 2. Audio Transcription Mapping

**Location**: `archive/src/processing/video_ingestion_pipeline.py` (lines 398-400)

Audio transcripts are mapped TO the existing keyframe boundaries:

```python
# Step 1: Extract all words with timestamps from transcript segments
all_words = [word for seg in transcription_segments for word in seg.get('words', [])]

# Step 2: For each keyframe, find overlapping audio
for i, (frame_img, start_time, end_time) in enumerate(keyframes_data):
    # Find words spoken during this visual scene
    audio_segment_words = [
        word['word'] for word in all_words 
        if word.get('start', 0) >= start_time and word.get('end', 0) <= end_time
    ]
    audio_segment = " ".join(audio_segment_words).strip()
```

## Frame Boundary Examples

Using video `v_-IMXSEIabMM` as an example:

### Frame 41 Analysis

**Visual Scene**: 30.030s - 46.847s (16.8 second duration)

**Audio Content During This Scene**:
- Segment 6 (29.280s - 31.600s): "you're protecting your head, that's the most important."
- Segment 7 (32.540s - 38.720s): "Sacrificing a limb, hurting your hand, and saving your head, because having your head"
- Segment 8 (38.720s - 43.600s): "hit the ice, especially when it comes black ice, and getting a subdural hematoma, blood"
- Segment 9 (43.600s - 46.700s): "inside the brain, can be devastating for a lot of people."

**Interpretation**:
The 16.8-second keyframe represents a coherent visual scene where the speaker discusses head protection. The long duration indicates that the visual content (likely showing the same scene/angle) remains relatively stable while covering multiple related audio segments about the same topic.

## Design Rationale

### Why Not Align with Audio Segments?

1. **Visual vs Audio Boundaries**: Visual scene changes and speech segment boundaries often don't align
2. **Semantic Coherence**: A single visual scene may cover multiple related audio topics
3. **Search Efficiency**: Longer, semantically coherent frames are better for retrieval than artificially short segments

### Frame Duration Characteristics

- **Short durations (0.1-2s)**: Rapid visual changes (action sequences, cuts)
- **Medium durations (2-5s)**: Typical scene lengths
- **Long durations (5-20s)**: Stable visual scenes with extended dialogue/narration
- **Very long durations (>20s)**: May indicate processing errors or very static content

## Current Implementation vs Archive

### Archive Implementation (Sophisticated)
- Visual scene change detection with histogram comparison
- Precise temporal boundaries based on actual scene changes
- Audio mapped to visual boundaries
- Supports variable frame durations based on content

### Current Implementation (Simplified)
Location: `src/processing/pipeline_steps/embedding_generator.py` (lines 245-246)

```python
"start_time": float(timestamp),
"end_time": float(timestamp + 1.0),  # Fixed 1-second duration
```

The current implementation uses a simplified 1-second duration estimate rather than the sophisticated boundary detection from the archive.

## Metadata Structure

Each frame in the metadata contains:

```json
{
  "frame_id": 41,
  "filename": "frame_0041.jpg", 
  "start_time": 30.03,
  "end_time": 46.8468,
  "duration": 16.8168,
  "path": "data/videos/video_chatgpt_eval/processed/keyframes/v_-IMXSEIabMM/frame_0041.jpg"
}
```

Where:
- `start_time`: When this visual scene begins
- `end_time`: When this visual scene ends (next scene change)
- `duration`: Length of this visual scene (`end_time - start_time`)
- Audio content can be retrieved separately and mapped to this time range

## Implications for Search

1. **Long Frame Durations Are Normal**: A 16-second frame is semantically valid if it represents a stable visual scene
2. **Audio-Visual Coherence**: Frames contain both visual and audio content from the same time period
3. **Context Preservation**: Longer frames preserve more context for better search relevance
4. **Temporal Accuracy**: Frame boundaries reflect actual content transitions, not artificial segmentation