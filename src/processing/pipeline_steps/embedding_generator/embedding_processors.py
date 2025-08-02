#!/usr/bin/env python3
"""
Embedding Processors - Handles the actual embedding generation and format conversion
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from pathlib import Path
import struct
from binascii import hexlify
import logging


class EmbeddingProcessor:
    """Handles embedding generation and format conversion"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    def generate_embeddings_from_image(
        self,
        image_path: Path,
        model: Any,
        processor: Any
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
        processor: Any
    ) -> Optional[np.ndarray]:
        """Generate embeddings from a video segment"""
        try:
            import tempfile
            import subprocess
            import os
            
            # Extract segment
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Extract video segment
                segment_duration = end_time - start_time
                
                # Use appropriate ffmpeg command
                if segment_duration < 5.0:
                    # Re-encode for short segments
                    cmd = [
                        'ffmpeg', '-i', str(video_path),
                        '-ss', str(start_time),
                        '-t', str(segment_duration),
                        '-c:v', 'libx264', '-preset', 'ultrafast',
                        '-c:a', 'copy',
                        '-y', tmp_path
                    ]
                else:
                    # Copy for longer segments
                    cmd = [
                        'ffmpeg', '-i', str(video_path),
                        '-ss', str(start_time),
                        '-t', str(segment_duration),
                        '-c', 'copy',
                        '-y', tmp_path
                    ]
                
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Process video segment
                if hasattr(processor, 'process_videos_with_audio'):
                    batch_inputs = processor.process_videos_with_audio([tmp_path]).to(model.device)
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
        videoprism_loader: Any
    ) -> Optional[Dict[str, Any]]:
        """Process video segment with VideoPrism"""
        try:
            result = videoprism_loader.process_video_segment(
                video_path,
                start_time,
                end_time
            )
            return result
        except Exception as e:
            self.logger.error(f"Failed to process VideoPrism segment: {e}")
            return None
    
    def convert_to_float_embeddings(
        self,
        embeddings: np.ndarray
    ) -> Dict[int, str]:
        """Convert embeddings to Vespa float format (hex-encoded bfloat16)"""
        try:
            embedding_dict = {}
            
            for patch_idx in range(len(embeddings)):
                # Convert to tensor and then to hex
                tensor = torch.tensor(embeddings[patch_idx], dtype=torch.float32)
                hex_string = self._tensor_to_hex_bfloat16(tensor)
                embedding_dict[patch_idx] = hex_string
            
            return embedding_dict
            
        except Exception as e:
            self.logger.error(f"Failed to convert to float embeddings: {e}")
            return {}
    
    def convert_to_binary_embeddings(
        self,
        embeddings: np.ndarray
    ) -> Dict[int, str]:
        """Convert embeddings to binary format"""
        try:
            # Binarize: positive values -> 1, negative/zero -> 0
            binarized = np.packbits(
                np.where(embeddings > 0, 1, 0),
                axis=1
            ).astype(np.int8)
            
            # Convert to hex strings
            embedding_dict = {}
            for idx in range(len(binarized)):
                hex_string = hexlify(binarized[idx].tobytes()).decode('utf-8')
                embedding_dict[idx] = hex_string
            
            return embedding_dict
            
        except Exception as e:
            self.logger.error(f"Failed to convert to binary embeddings: {e}")
            return {}
    
    def _tensor_to_hex_bfloat16(self, tensor: torch.Tensor) -> str:
        """Convert tensor to hex-encoded bfloat16 format"""
        if not tensor.is_floating_point():
            raise ValueError("Input tensor must be of float type")
        
        def float_to_bfloat16_hex(f: float) -> str:
            packed_float = struct.pack("=f", f)
            bfloat16_bits = struct.unpack("=H", packed_float[2:])[0]
            return format(bfloat16_bits, "04X")
        
        hex_list = [float_to_bfloat16_hex(float(val)) for val in tensor.flatten()]
        return "".join(hex_list)