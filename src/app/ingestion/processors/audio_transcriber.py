#!/usr/bin/env python3
"""
Audio Transcription Step

Extracts and transcribes audio from videos using Faster-Whisper.
"""

import json
import time
import torch
from pathlib import Path
from typing import Dict, Any, Optional


class AudioTranscriber:
    """Handles audio transcription from videos"""
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
    
    def _load_model(self):
        """Lazy load the Whisper model"""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(self.model_size, device=self.device)
                print(f"  📝 Loaded Whisper model: {self.model_size} on {self.device}")
            except ImportError as e:
                raise ImportError(f"faster-whisper import failed: {e}")
    
    def transcribe_audio(self, video_path: Path, output_dir: Path = None) -> Dict[str, Any]:
        """Extract and transcribe audio from video"""
        print(f"🎵 Transcribing audio from: {video_path.name}")
        
        # Use OutputManager for consistent directory structure
        if output_dir is None:
            from src.common.utils.output_manager import get_output_manager
            output_manager = get_output_manager()
            transcript_file = output_manager.get_processing_dir("transcripts") / f"{video_path.stem}.json"
        else:
            # Legacy path support
            video_id = video_path.stem
            transcript_file = output_dir / "transcripts" / f"{video_id}.json"
        
        # Remove caching - always retranscribe audio
        
        video_id = video_path.stem
        try:
            self._load_model()
            
            start_time = time.time()
            segments, info = self._model.transcribe(str(video_path))
            
            transcript_data = {
                "video_id": video_id,
                "video_path": str(video_path),
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "segments": [],
                "full_text": "",
                "processing_time_seconds": 0,
                "created_at": time.time()
            }
            
            full_text_parts = []
            for segment in segments:
                segment_data = {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob
                }
                transcript_data["segments"].append(segment_data)
                full_text_parts.append(segment.text.strip())
            
            transcript_data["full_text"] = " ".join(full_text_parts)
            transcript_data["processing_time_seconds"] = time.time() - start_time
            
            # Save transcript
            transcript_file.parent.mkdir(parents=True, exist_ok=True)
            with open(transcript_file, 'w') as f:
                json.dump(transcript_data, f, indent=2)
                
            print(f"  ✅ Transcribed {len(transcript_data['segments'])} segments in {transcript_data['processing_time_seconds']:.1f}s")
            return transcript_data
            
        except Exception as e:
            print(f"  ❌ Audio transcription failed: {e}")
            return {
                "video_id": video_id,
                "video_path": str(video_path),
                "error": str(e),
                "created_at": time.time()
            }