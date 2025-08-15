"""
Document builder for ColQwen chunks schema with audio transcript support.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import logging
from pathlib import Path
import whisper
import subprocess
import tempfile
import os

logger = logging.getLogger(__name__)

class ColQwenChunksBuilder:
    """Builds documents for video_colqwen_chunks schema with audio transcripts."""
    
    def __init__(self, audio_model: str = "base"):
        """
        Initialize the builder.
        
        Args:
            audio_model: Whisper model size (tiny, base, small, medium, large)
        """
        self.audio_model = None
        self.audio_model_name = audio_model
        
    def _load_whisper_model(self):
        """Lazy load Whisper model."""
        if self.audio_model is None:
            logger.info(f"Loading Whisper model: {self.audio_model_name}")
            self.audio_model = whisper.load_model(self.audio_model_name)
            logger.info("Whisper model loaded")
    
    def extract_audio_segment(
        self, 
        video_path: Path, 
        start_time: float, 
        end_time: float,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Extract audio segment from video.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Optional output path for audio file
            
        Returns:
            Path to extracted audio file
        """
        if output_path is None:
            temp_dir = tempfile.gettempdir()
            output_path = Path(temp_dir) / f"audio_segment_{start_time}_{end_time}.wav"
        
        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-ss', str(start_time),
            '-to', str(end_time),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # WAV format
            '-ar', '16000',  # 16kHz sample rate for Whisper
            '-ac', '1',  # Mono
            str(output_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract audio: {e.stderr}")
            return None
    
    def transcribe_audio(self, audio_path: Path) -> str:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            self._load_whisper_model()
            result = self.audio_model.transcribe(str(audio_path))
            return result.get("text", "").strip()
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            return ""
    
    def process_video_segments(
        self,
        video_path: Path,
        segments: List[Dict[str, Any]],
        embeddings_list: List[np.ndarray],
        extract_audio: bool = True
    ) -> Dict[str, Any]:
        """
        Process video segments and build document for chunks schema.
        
        Args:
            video_path: Path to video file
            segments: List of segment info dicts with start_time, end_time
            embeddings_list: List of embeddings arrays for each segment
            extract_audio: Whether to extract and transcribe audio
            
        Returns:
            Document for Vespa
        """
        segment_transcripts = []
        start_times = []
        end_times = []
        
        # Process each segment
        for i, segment in enumerate(segments):
            start_time = segment['start_time']
            end_time = segment['end_time']
            
            start_times.append(float(start_time))
            end_times.append(float(end_time))
            
            # Extract and transcribe audio if requested
            if extract_audio:
                logger.info(f"Processing audio for segment {i+1}/{len(segments)}: {start_time:.1f}s - {end_time:.1f}s")
                
                audio_path = self.extract_audio_segment(video_path, start_time, end_time)
                if audio_path and audio_path.exists():
                    transcript = self.transcribe_audio(audio_path)
                    segment_transcripts.append(transcript)
                    
                    # Clean up temp audio file
                    try:
                        os.unlink(audio_path)
                    except:
                        pass
                    
                    logger.info(f"Segment {i+1} transcript: {transcript[:100]}...")
                else:
                    segment_transcripts.append("")
                    logger.warning(f"No audio extracted for segment {i+1}")
            else:
                segment_transcripts.append("")
        
        # Combine embeddings into 3D tensor structure
        # Shape: (num_segments, num_patches, embedding_dim)
        combined_embeddings = self._combine_embeddings(embeddings_list)
        
        return {
            "segment_transcripts": segment_transcripts,
            "start_times": start_times,
            "end_times": end_times,
            "embeddings": combined_embeddings,
            "num_segments": len(segments)
        }
    
    def _combine_embeddings(self, embeddings_list: List[np.ndarray]) -> Dict[str, Any]:
        """
        Combine segment embeddings into tensor format for Vespa.
        
        Args:
            embeddings_list: List of embeddings arrays
            
        Returns:
            Combined embeddings in Vespa tensor format
        """
        # For ColQwen with patches, we need to handle the patch dimension
        # Each embedding is shape (num_patches, embedding_dim)
        
        # Create tensor cells for Vespa
        cells = []
        
        for seg_idx, embeddings in enumerate(embeddings_list):
            if embeddings is None:
                continue
                
            # Handle different embedding shapes
            if len(embeddings.shape) == 2:
                # Shape: (num_patches, embedding_dim)
                num_patches, embedding_dim = embeddings.shape
                
                for patch_idx in range(num_patches):
                    for dim_idx in range(embedding_dim):
                        cells.append({
                            "address": {
                                "p": str(seg_idx),
                                "patch": str(patch_idx),
                                "v": str(dim_idx)
                            },
                            "value": float(embeddings[patch_idx, dim_idx])
                        })
            elif len(embeddings.shape) == 1:
                # Single vector per segment
                embedding_dim = embeddings.shape[0]
                
                for dim_idx in range(embedding_dim):
                    cells.append({
                        "address": {
                            "p": str(seg_idx),
                            "patch": "0",  # Single patch
                            "v": str(dim_idx)
                        },
                        "value": float(embeddings[dim_idx])
                    })
        
        return {"cells": cells}
    
    def build_document(
        self,
        video_id: str,
        video_title: str,
        video_path: Path,
        segments: List[Dict[str, Any]],
        embeddings_list: List[np.ndarray],
        extract_audio: bool = True,
        video_description: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Build complete document for video_colqwen_chunks schema.
        
        Args:
            video_id: Video identifier
            video_title: Video title
            video_path: Path to video file
            segments: List of segment info
            embeddings_list: List of embeddings for each segment
            extract_audio: Whether to extract audio transcripts
            video_description: Optional video description
            **kwargs: Additional fields
            
        Returns:
            Complete document for Vespa
        """
        # Process segments and get arrays
        segment_data = self.process_video_segments(
            video_path, 
            segments, 
            embeddings_list,
            extract_audio
        )
        
        # Calculate video duration
        duration = max([s['end_time'] for s in segments])
        
        # Build document
        document = {
            "put": f"id:video_colqwen_chunks:video_colqwen_chunks::{video_id}",
            "fields": {
                "video_id": video_id,
                "video_title": video_title,
                "video_description": video_description,
                "duration": float(duration),
                "num_segments": segment_data["num_segments"],
                "segment_transcripts": segment_data["segment_transcripts"],
                "start_times": segment_data["start_times"],
                "end_times": segment_data["end_times"],
                "embedding": segment_data["embeddings"],
                "creation_timestamp": kwargs.get("creation_timestamp", 0)
            }
        }
        
        # Add binary embeddings if provided
        if "embeddings_binary" in segment_data:
            document["fields"]["embedding_binary"] = segment_data["embeddings_binary"]
        
        return document