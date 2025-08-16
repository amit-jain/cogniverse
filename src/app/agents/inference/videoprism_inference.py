"""
VideoPrism JAX inference engine for real-time video search
Supports both CPU and Metal backends
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Try Metal backend first, fallback to CPU
try:
    os.environ["JAX_PLATFORMS"] = "metal,cpu"
    import jax
    # Test if Metal is available
    try:
        _ = jax.devices("metal")[0]
        BACKEND = "metal"
        logger = logging.getLogger(__name__)
        logger.info("Using JAX Metal backend")
    except:
        BACKEND = "cpu"
        logger = logging.getLogger(__name__)
        logger.info("Using JAX CPU backend")
except ImportError:
    raise ImportError("JAX not available. Please install jax and jaxlib")

import jax.numpy as jnp
from huggingface_hub import hf_hub_download
import cv2
from PIL import Image


class VideoPrismInference:
    """Real-time VideoPrism inference engine for video search"""
    
    def __init__(self, model_name: str = "videoprism_public_v1_base_hf"):
        """
        Initialize VideoPrism inference engine
        
        Args:
            model_name: Model variant to use (base or large)
        """
        self.model_name = model_name
        self.model = None
        self.forward_fn = None
        
        # Model configuration
        if "base" in model_name:
            self.embedding_dim = 768
            self.num_patches = 4096  # 16x16x16 spatiotemporal patches
            self.spatial_patches = 16
            self.temporal_patches = 16
        else:  # large
            self.embedding_dim = 1024
            self.num_patches = 2048  # 8x16x16 spatiotemporal patches
            self.spatial_patches = 8
            self.temporal_patches = 16
        
        self.input_shape = (16, 288, 288, 3)  # T, H, W, C
        self.logger = logger
        
    def load_model(self):
        """Load VideoPrism model for inference"""
        if self.model is not None:
            return  # Already loaded
            
        self.logger.info(f"Loading VideoPrism model: {self.model_name} on {BACKEND}")
        
        # Import VideoPrism components
        try:
            # Try to import from the reference implementation
            import sys
            from src.common.config import get_config
            config = get_config()
            videoprism_path = config.get("videoprism_repo_path")
            
            if not videoprism_path:
                # Fallback to environment variable
                import os
                videoprism_path = os.environ.get("VIDEOPRISM_REPO_PATH")
            
            if videoprism_path:
                videoprism_path = Path(videoprism_path)
                if videoprism_path.exists():
                    sys.path.insert(0, str(videoprism_path))
                from videoprism import models as vp
                
                # Build and load model
                self.model = vp.MODELS[self.model_name]()
                state = vp.load_pretrained_weights(self.model_name)
                
                # Create JIT-compiled forward function
                @jax.jit
                def forward_fn(inputs):
                    return self.model.apply(state, inputs, train=False)
                
                self.forward_fn = forward_fn
                self.logger.info(f"VideoPrism model loaded successfully using {BACKEND} backend")
                
            else:
                # Fallback to placeholder implementation
                self.logger.warning("VideoPrism reference not found, using placeholder model")
                self._load_placeholder_model()
                
        except Exception as e:
            self.logger.warning(f"Failed to load actual model: {e}, using placeholder")
            self._load_placeholder_model()
    
    def _load_placeholder_model(self):
        """Load a placeholder model for testing"""
        @jax.jit
        def placeholder_forward(inputs):
            batch_size = inputs.shape[0]
            key = jax.random.PRNGKey(42)
            # Generate embeddings with correct shape
            embeddings = jax.random.normal(key, (batch_size, self.num_patches, self.embedding_dim))
            return embeddings
        
        self.forward_fn = placeholder_forward
        self.logger.info("Loaded placeholder VideoPrism model")
    
    def preprocess_frames(self, frames: List[np.ndarray]) -> jnp.ndarray:
        """
        Preprocess frames for VideoPrism
        
        Args:
            frames: List of frames as numpy arrays (H, W, C) in RGB format
            
        Returns:
            JAX array of shape (1, 16, 288, 288, 3) normalized to [0, 1]
        """
        processed_frames = []
        
        # Sample or pad to exactly 16 frames
        if len(frames) > 16:
            # Sample uniformly
            indices = np.linspace(0, len(frames) - 1, 16, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < 16:
            # Pad by repeating last frame
            while len(frames) < 16:
                frames.append(frames[-1])
        
        # Process each frame
        for frame in frames:
            # Resize to 288x288
            if frame.shape[:2] != (288, 288):
                frame = cv2.resize(frame, (288, 288))
            
            # Normalize to [0, 1]
            if frame.dtype == np.uint8:
                frame = frame.astype(np.float32) / 255.0
            
            processed_frames.append(frame)
        
        # Stack into video tensor
        video = np.stack(processed_frames, axis=0)  # (16, 288, 288, 3)
        video = np.expand_dims(video, axis=0)  # (1, 16, 288, 288, 3)
        
        return jnp.array(video)
    
    def extract_embeddings(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Extract embeddings from video frames
        
        Args:
            frames: List of frames as numpy arrays in RGB format
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        if self.forward_fn is None:
            self.load_model()
        
        # Preprocess frames
        video_input = self.preprocess_frames(frames)
        
        # Run inference
        self.logger.debug(f"Running inference on {len(frames)} frames")
        embeddings = self.forward_fn(video_input)
        
        # Convert to numpy
        embeddings_np = np.array(embeddings[0])  # Remove batch dimension
        
        return {
            "embeddings": embeddings_np,  # (num_patches, embedding_dim)
            "embedding_dim": self.embedding_dim,
            "num_patches": self.num_patches,
            "spatial_patches": self.spatial_patches,
            "temporal_patches": self.temporal_patches,
            "model_name": self.model_name,
            "backend": BACKEND
        }
    
    def embeddings_to_search_format(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Convert embeddings to format suitable for search
        
        Args:
            embeddings: Numpy array of shape (num_patches, embedding_dim)
            
        Returns:
            Dictionary with float and binary embeddings
        """
        # Generate float embeddings (keep native dimensions)
        float_embeddings = embeddings.astype(np.float32)
        
        # Generate binary embeddings for fast search
        binary_embeddings = np.packbits(np.where(embeddings > 0, 1, 0), axis=1).astype(np.int8)
        
        return {
            "float_embeddings": float_embeddings,
            "binary_embeddings": binary_embeddings,
            "num_patches": embeddings.shape[0],
            "embedding_dim": embeddings.shape[1]
        }
    
    def compute_similarity(self, query_embeddings: np.ndarray, doc_embeddings: np.ndarray) -> float:
        """
        Compute similarity between query and document embeddings
        
        Args:
            query_embeddings: Query embeddings (num_query_patches, embedding_dim)
            doc_embeddings: Document embeddings (num_doc_patches, embedding_dim)
            
        Returns:
            Similarity score
        """
        # Compute cosine similarity matrix
        query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        doc_norm = doc_embeddings / (np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute all pairwise similarities
        similarities = np.dot(query_norm, doc_norm.T)
        
        # Max pooling across document patches for each query patch
        max_similarities = np.max(similarities, axis=1)
        
        # Average across query patches
        return float(np.mean(max_similarities))


# Singleton instance management
_inference_engine = None


def get_videoprism_inference(model_name: str = "videoprism_public_v1_base_hf") -> VideoPrismInference:
    """Get or create VideoPrism inference engine"""
    global _inference_engine
    if _inference_engine is None or _inference_engine.model_name != model_name:
        _inference_engine = VideoPrismInference(model_name)
    return _inference_engine