#!/usr/bin/env python3
"""
Embedding Processors - Handles the actual embedding generation and format conversion
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image
from pathlib import Path
import logging


class EmbeddingProcessor:
    """Handles embedding generation and format conversion"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def generate_embeddings_from_image(
        self, image_path: Path, model: Any, processor: Any
    ) -> Optional[np.ndarray]:
        """Generate embeddings from an image file"""
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")

            # Process image
            batch_images = processor.process_images([image]).to(model.device)

            # Generate embeddings
            with torch.no_grad():
                embeddings = model(**batch_images)

            # Convert to numpy
            embeddings_np = embeddings.cpu().to(torch.float32).numpy().squeeze(0)

            return embeddings_np

        except Exception as e:
            self.logger.error(f"Failed to generate embeddings for {image_path}: {e}")
            return None

    def generate_embeddings_from_video_segment(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        model: Any,
        processor: Any,
    ) -> Optional[np.ndarray]:
        """Generate embeddings from a video segment"""
        try:
            import tempfile
            import subprocess
            import os

            # Extract segment
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                tmp_path = tmp_file.name

            try:
                # Extract video segment
                segment_duration = end_time - start_time

                # Use appropriate ffmpeg command
                if segment_duration < 5.0:
                    # Re-encode for short segments
                    cmd = [
                        "ffmpeg",
                        "-i",
                        str(video_path),
                        "-ss",
                        str(start_time),
                        "-t",
                        str(segment_duration),
                        "-c:v",
                        "libx264",
                        "-preset",
                        "ultrafast",
                        "-c:a",
                        "copy",
                        "-y",
                        tmp_path,
                    ]
                else:
                    # Copy for longer segments
                    cmd = [
                        "ffmpeg",
                        "-i",
                        str(video_path),
                        "-ss",
                        str(start_time),
                        "-t",
                        str(segment_duration),
                        "-c",
                        "copy",
                        "-y",
                        tmp_path,
                    ]

                subprocess.run(cmd, check=True, capture_output=True)

                # Process video segment with audio if available
                if hasattr(processor, "process_videos_with_audio"):
                    batch_inputs = processor.process_videos_with_audio([tmp_path]).to(
                        model.device
                    )
                else:
                    batch_inputs = processor.process_videos([tmp_path]).to(model.device)

                # Generate embeddings
                with torch.no_grad():
                    embeddings = model(**batch_inputs)

                # Convert to numpy
                embeddings_np = embeddings.cpu().to(torch.float32).numpy()

                # Handle different output shapes
                if len(embeddings_np.shape) == 3:
                    embeddings_np = embeddings_np.squeeze(0)
                elif len(embeddings_np.shape) > 3:
                    embeddings_np = embeddings_np.reshape(-1, embeddings_np.shape[-1])

                return embeddings_np

            finally:
                # Clean up
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            self.logger.error(f"Failed to generate embeddings for video segment: {e}")
            return None

    def process_videoprism_segment(
        self,
        video_path: Path,
        start_time: float,
        end_time: float,
        videoprism_loader: Any,
    ) -> Optional[Dict[str, Any]]:
        """Process video segment with VideoPrism"""
        try:
            result = videoprism_loader.process_video_segment(
                video_path, start_time, end_time
            )
            return result
        except Exception as e:
            self.logger.error(f"Failed to process VideoPrism segment: {e}")
            return None

    def prepare_float_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Prepare float embeddings (returns raw numpy)"""
        # Just return raw embeddings - backend handles format conversion
        return embeddings

    def prepare_binary_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Prepare binary embeddings (returns binarized numpy)"""
        # Binarize but keep as numpy - backend handles hex conversion
        return (embeddings > 0).astype(np.uint8)
