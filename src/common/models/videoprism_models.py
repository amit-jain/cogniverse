"""
Minimal VideoPrism model implementation for JAX
Based on the VideoPrism repository but implemented locally
"""

import os
# Force CPU backend to avoid Metal issues
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple
import logging
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


class SimpleVideoPrismModel:
    """Simplified VideoPrism model wrapper for embeddings extraction"""
    
    def __init__(self, model_name: str = "videoprism_public_v1_base_hf"):
        self.model_name = model_name
        self.model = None
        self.state = None
        self.forward_fn = None
        
        # Model configuration
        # Use the regular VideoPrism models for video encoding
        if "base" in model_name:
            self.model_name = "videoprism_public_v1_base"
            self.embedding_dim = 768
            self.num_patches = 4096  # 16x16x16
        else:  # large
            self.model_name = "videoprism_public_v1_large"
            self.embedding_dim = 1024
            self.num_patches = 2048  # 8x16x16
        
        self.input_shape = (16, 288, 288, 3)  # frames, height, width, channels
        
    def load_model(self):
        """Load model weights from VideoPrism reference implementation"""
        logger.info(f"Loading VideoPrism model: {self.model_name}")
        
        try:
            # Import the actual VideoPrism implementation
            import sys
            from pathlib import Path
            
            # Get videoprism path from config or environment
            from src.common.config import get_config
            config = get_config()
            videoprism_path = config.get("videoprism_repo_path")
            
            if not videoprism_path:
                # Fallback to environment variable
                import os
                videoprism_path = os.environ.get("VIDEOPRISM_REPO_PATH")
            
            if not videoprism_path:
                raise ImportError("VideoPrism repository path not configured. Set videoprism_repo_path in config.json or VIDEOPRISM_REPO_PATH environment variable")
            
            videoprism_path = Path(videoprism_path)
            if not videoprism_path.exists():
                raise ImportError(f"VideoPrism not found at {videoprism_path}")
                
            sys.path.insert(0, str(videoprism_path))
            from videoprism import models as vp
            
            # Load the actual model
            logger.info(f"Building VideoPrism model: {self.model_name}")
            model = vp.get_model(self.model_name)
            
            # Load pretrained weights
            logger.info(f"Loading pretrained weights for {self.model_name}...")
            state = vp.load_pretrained_weights(self.model_name)
            
            # Create forward function for video-only inference
            def forward_fn_impl(frames):
                # Forward pass - returns (embeddings, extra_outputs)
                embeddings, _ = model.apply(
                    state,
                    frames,
                    train=False
                )
                return embeddings
            
            # Try to JIT compile, but fall back to non-JIT if it fails
            try:
                self.forward_fn = jax.jit(forward_fn_impl)
                logger.info("JIT compilation successful")
            except Exception as e:
                logger.warning(f"JIT compilation failed: {e}, using non-JIT version")
                self.forward_fn = forward_fn_impl
            self.model = model
            self.state = state
            logger.info(f"VideoPrism model loaded successfully from {videoprism_path}")
            
        except Exception as e:
            logger.error(f"Failed to load actual VideoPrism model: {e}")
            raise RuntimeError(f"VideoPrism model loading failed - cannot proceed with random embeddings: {e}")
        
    def preprocess_video(self, frames: np.ndarray) -> np.ndarray:
        """Preprocess video frames for model input"""
        # VideoPrism can handle arbitrary num_frames by interpolating temporal positional embeddings
        # No need to limit to 16 frames!
        
        # Normalize to [0, 1]
        frames = frames.astype(np.float32) / 255.0 if frames.dtype == np.uint8 else frames
        
        # Add batch dimension
        return np.expand_dims(frames, axis=0)
    
    def extract_embeddings(self, video_input: np.ndarray) -> Dict[str, Any]:
        """Extract embeddings from video input"""
        if self.forward_fn is None:
            self.load_model()
        
        # Convert to JAX array for forward pass
        # Ensure we're using float32
        video_input = video_input.astype(np.float32)
        
        # Create JAX array and run inference
        video_input_jax = jnp.array(video_input)
        embeddings = self.forward_fn(video_input_jax)
        
        return {
            "embeddings": np.array(embeddings[0]),  # Remove batch dimension and convert back to numpy
            "embedding_dim": self.embedding_dim,
            "num_patches": self.num_patches,
            "model_name": self.model_name
        }


def get_videoprism_model(model_name: str = "videoprism_public_v1_base_hf") -> SimpleVideoPrismModel:
    """Factory function to create VideoPrism model"""
    return SimpleVideoPrismModel(model_name)