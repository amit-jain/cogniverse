"""
VideoPrism model loader and inference utilities
"""

import logging
import numpy as np
import jax
import jax.numpy as jnp
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import cv2
from PIL import Image
from src.common.utils.retry import retry_with_backoff, RetryConfig

logger = logging.getLogger(__name__)

# Defer import to avoid JAX initialization issues
VIDEOPRISM_AVAILABLE = None
_videoprism_models = None

def _check_videoprism_available():
    """Check if VideoPrism is available, importing on first use"""
    global VIDEOPRISM_AVAILABLE, _videoprism_models
    if VIDEOPRISM_AVAILABLE is None:
        try:
            # Import from models directory - handle both relative and absolute imports
            try:
                from . import videoprism_models
                _videoprism_models = videoprism_models
            except ImportError:
                # Fallback to absolute import
                import src.common.models.videoprism_models as videoprism_models
                _videoprism_models = videoprism_models
            VIDEOPRISM_AVAILABLE = True
        except ImportError as e:
            logger.warning(f"VideoPrism import failed: {e}")
            VIDEOPRISM_AVAILABLE = False
    return VIDEOPRISM_AVAILABLE


class VideoPrismLoader:
    """Handles VideoPrism model loading and inference"""
    
    def __init__(self, model_name: str = "videoprism_public_v1_base_hf", config: Optional[Dict[str, Any]] = None):
        """
        Initialize VideoPrism loader
        
        Args:
            model_name: Model variant to load (base or large)
            config: Configuration dictionary for model-specific settings
        """
        if not _check_videoprism_available():
            raise ImportError("VideoPrism not available. Check videoprism_models.py")
            
        self.model_name = model_name
        self.model = None
        self.state = None
        self.forward_fn = None
        self.config = config or {}
        self.is_global = False  # Will be set for LVT models
        
        # Model can handle arbitrary num_frames by interpolating temporal positional embeddings
        # Training used 16 frames for base, 8 for large, but inference can use any number
        self.input_height = 288
        self.input_width = 288
        
        # Output embedding dimensions
        if "base" in model_name:
            self.embedding_dim = 768
            self.spatial_tokens = 16  # 16x16x16 = 4096 tokens
        else:  # large
            self.embedding_dim = 1024
            self.spatial_tokens = 8  # 8x16x16 = 2048 tokens
            
        logger.info(f"Initialized VideoPrism loader for {model_name}")
    
    @retry_with_backoff(
        config=RetryConfig(
            max_attempts=3,
            initial_delay=2.0,
            exceptions=(Exception,)  # Retry on any exception during model loading
        )
    )
    def load_model(self):
        """Load VideoPrism model and weights with retry logic"""
        if self.model is not None:
            return  # Already loaded
            
        logger.info(f"Loading VideoPrism model: {self.model_name}")
        
        # Build model using our local implementation
        self.model = _videoprism_models.get_videoprism_model(self.model_name)
        self.model.load_model()
        self.forward_fn = self.model.forward_fn
        
        logger.info(f"VideoPrism model loaded successfully")
    
    def preprocess_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess frames for VideoPrism input
        
        Args:
            frames: List of frames as numpy arrays (H, W, C) in RGB format
            
        Returns:
            JAX array of shape (1, num_frames, 288, 288, 3) normalized to [0, 1]
        """
        processed_frames = []
        
        # Process each frame - VideoPrism can handle arbitrary num_frames
        for frame in frames:
            # Resize to 288x288
            frame_resized = cv2.resize(frame, (self.input_width, self.input_height))
            
            # Normalize to [0, 1]
            frame_normalized = frame_resized.astype(np.float32) / 255.0
            
            processed_frames.append(frame_normalized)
        
        # Stack into batch
        video_input = np.stack(processed_frames, axis=0)  # (num_frames, 288, 288, 3)
        video_input = np.expand_dims(video_input, axis=0)  # (1, num_frames, 288, 288, 3)
        
        logger.info(f"Preprocessed {len(frames)} frames to shape: {video_input.shape}")
        
        return video_input  # Return numpy array
    
    def extract_embeddings(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Extract VideoPrism embeddings from frames
        
        Args:
            frames: List of frames as numpy arrays in RGB format
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        if self.model is None:
            self.load_model()
        
        # Preprocess frames
        video_input = self.preprocess_frames(frames)
        
        # Run inference using our model
        logger.info(f"Running VideoPrism inference on {len(frames)} frames")
        result = self.model.extract_embeddings(video_input)
        embeddings_np = result['embeddings']
        
        logger.info(f"Generated embeddings shape: {embeddings_np.shape}")
        
        return {
            "embeddings": embeddings_np,  # (num_tokens, embedding_dim)
            "embedding_dim": self.embedding_dim,
            "num_tokens": embeddings_np.shape[0],
            "model_name": self.model_name,
            "temporal_frames": len(frames)
        }
    
    def embeddings_to_vespa_format(self, embeddings: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Convert VideoPrism embeddings to Vespa format with native dimensions
        
        Args:
            embeddings: Numpy array of shape (num_tokens, embedding_dim)
            
        Returns:
            Tuple of (float_embeddings_dict, binary_embeddings) for Vespa
        """
        # Use native dimensions - no projection!
        # float_embeddings will be a dict with the full tensor specification
        embedding_dim = embeddings.shape[1]  # 768 or 1024
        num_patches = embeddings.shape[0]
        
        logger.info(f"Converting {num_patches} patches with {embedding_dim} dimensions to Vespa format")
        
        # Create float embeddings in Vespa tensor cell format
        float_cells = []
        for patch_idx in range(num_patches):
            for v_idx in range(embedding_dim):
                float_cells.append({
                    "address": {"patch": str(patch_idx), "v": str(v_idx)},
                    "value": float(embeddings[patch_idx, v_idx])
                })
        
        float_embeddings_dict = {
            "cells": float_cells
        }
        
        # Generate binary embeddings
        # Binary size should be embedding_dim / 8 (8 bits per byte)
        binary_embeddings = {}
        vectors = embeddings.astype(np.float32)
        binarized_vectors = np.packbits(np.where(vectors > 0, 1, 0), axis=1).astype(np.int8)
        
        from binascii import hexlify
        for idx in range(len(binarized_vectors)):
            binary_embeddings[f"patch{idx}"] = str(
                hexlify(binarized_vectors[idx].tobytes()), "utf-8"
            )
        
        return float_embeddings_dict, binary_embeddings
    
    def process_entire_video(self, video_path: Path, sampling_fps: float = 1.0) -> Dict[str, Any]:
        """
        Process entire video and extract embeddings
        
        Args:
            video_path: Path to video file
            sampling_fps: Sampling rate in FPS (default 1.0)
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        import cv2
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate frames to sample based on duration and sampling FPS
        num_frames_to_sample = int(duration * sampling_fps)
        
        # Apply max_frames limit from config
        max_frames = self.config.get("model_specific", {}).get("max_frames", 40)
        if num_frames_to_sample > max_frames:
            logger.warning(f"Limiting frames from {num_frames_to_sample} to {max_frames} (max_frames config)")
            num_frames_to_sample = max_frames
        
        logger.info(f"Processing entire video: {video_path.name}")
        logger.info(f"  Duration: {duration:.1f}s, Total frames: {total_frames}")
        logger.info(f"  Sampling at {sampling_fps} FPS = {num_frames_to_sample} frames")
        
        # Sample frames uniformly across the video
        if num_frames_to_sample > total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int).tolist()
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        
        # Extract embeddings
        embedding_result = self.extract_embeddings(frames)
        
        # Convert to Vespa format
        float_embeddings, binary_embeddings = self.embeddings_to_vespa_format(
            embedding_result["embeddings"]
        )
        
        return {
            "embeddings": embedding_result["embeddings"],
            "float_embeddings": float_embeddings,
            "binary_embeddings": binary_embeddings,
            "num_frames_sampled": len(frames),
            "duration": duration,
            "sampling_fps": sampling_fps,
            "embedding_dim": self.embedding_dim,
            "num_patches": embedding_result["num_tokens"]
        }
    
    def process_video_segment(self, video_path: Path, start_time: float, end_time: float) -> Dict[str, Any]:
        """
        Process a video segment and extract embeddings
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Dictionary containing embeddings and metadata
        """
        import cv2
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame range
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Get sampling FPS from config
        sampling_fps = self.config.get("model_specific", {}).get("sampling_fps", 1.0)
        max_frames = self.config.get("model_specific", {}).get("max_frames", 30)
        
        # Calculate frames to sample based on duration and sampling FPS
        segment_duration = end_time - start_time
        num_frames_to_sample = int(segment_duration * sampling_fps)
        num_frames_to_sample = min(num_frames_to_sample, max_frames)
        
        # Sample frames uniformly across the segment
        total_frames_in_segment = end_frame - start_frame
        if num_frames_to_sample >= total_frames_in_segment:
            frame_indices = list(range(start_frame, end_frame))
        else:
            frame_indices = np.linspace(start_frame, end_frame - 1, num_frames_to_sample, dtype=int).tolist()
        
        # Extract frames
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        
        if not frames:
            logger.warning(f"No frames extracted from segment {start_time}-{end_time}")
            return None
        
        logger.info(f"Extracted {len(frames)} frames from segment {start_time}-{end_time}s (sampling at {sampling_fps} FPS)")
        
        # Extract embeddings
        embedding_result = self.extract_embeddings(frames)
        
        # Convert to Vespa format
        float_embeddings, binary_embeddings = self.embeddings_to_vespa_format(
            embedding_result["embeddings"]
        )
        
        return {
            "float_embeddings": float_embeddings,
            "binary_embeddings": binary_embeddings,
            "num_frames_processed": len(frames),
            "start_time": start_time,
            "end_time": end_time,
            **embedding_result
        }


# Model instances cache
_videoprism_loaders = {}


class VideoPrismGlobalLoader(VideoPrismLoader):
    """Handles VideoPrism LVT model loading for global embeddings"""
    
    def __init__(self, model_name: str = "videoprism_lvt_public_v1_base", config: Optional[Dict[str, Any]] = None):
        """
        Initialize VideoPrism LVT loader for global embeddings
        
        Args:
            model_name: LVT model variant (base or large)
            config: Configuration dictionary
        """
        # Map model names to LVT variants
        if "global" in model_name:
            if "large" in model_name:
                actual_model = "videoprism_lvt_public_v1_large"
            else:
                actual_model = "videoprism_lvt_public_v1_base"
        else:
            actual_model = model_name
            
        super().__init__(actual_model, config)
        self.is_global = True
        
        # LVT models produce global embeddings
        logger.info(f"Initialized VideoPrism LVT loader for global embeddings: {actual_model}")
        
        # Text encoder components (loaded on demand)
        self.text_tokenizer = None
        self.text_encoder = None
    
    def embeddings_to_vespa_format(self, embeddings: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Convert VideoPrism global embeddings to Vespa format
        
        Args:
            embeddings: Numpy array of shape (embedding_dim,) for global embeddings
            
        Returns:
            Tuple of (float_embeddings_list, binary_embeddings_array) for Vespa
        """
        # For global embeddings, we have a single vector
        if len(embeddings.shape) == 1:
            # Single global embedding
            embeddings = embeddings.reshape(1, -1)
        
        embedding_dim = embeddings.shape[1]
        logger.info(f"Converting global embedding with {embedding_dim} dimensions to Vespa format")
        
        # For global embeddings, return as simple list
        float_embeddings_list = embeddings[0].tolist()
        
        # Generate binary embeddings (96 bytes for 768 dims)
        binary_vec = embeddings[0].astype(np.float32)
        binarized = np.packbits(np.where(binary_vec > 0, 1, 0)).astype(np.int8)
        
        return float_embeddings_list, binarized
    
    def extract_embeddings(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Extract global VideoPrism embeddings from frames
        
        Args:
            frames: List of frames as numpy arrays in RGB format
            
        Returns:
            Dictionary containing global embeddings and metadata
        """
        if self.model is None:
            self.load_model()
        
        # Preprocess frames
        video_input = self.preprocess_frames(frames)
        
        # Run inference using our model
        logger.info(f"Running VideoPrism LVT inference on {len(frames)} frames for global embedding")
        result = self.model.extract_embeddings(video_input)
        embeddings_np = result['embeddings']
        
        # For LVT models, we should get global embeddings
        # If we get multi-token output, average pool to create global
        if len(embeddings_np.shape) > 1 and embeddings_np.shape[0] > 1:
            logger.info(f"Averaging {embeddings_np.shape[0]} tokens to create global embedding")
            embeddings_np = np.mean(embeddings_np, axis=0)
        
        logger.info(f"Generated global embedding shape: {embeddings_np.shape}")
        
        return {
            "embeddings": embeddings_np,  # (embedding_dim,)
            "embedding_dim": self.embedding_dim,
            "num_tokens": 1,  # Global embedding
            "model_name": self.model_name,
            "temporal_frames": len(frames),
            "is_global": True
        }
    
    def process_entire_video(self, video_path: Path, sampling_fps: float = 1.0) -> Dict[str, Any]:
        """
        Process entire video and extract global embeddings
        
        Args:
            video_path: Path to video file
            sampling_fps: Sampling rate in FPS (default 1.0)
            
        Returns:
            Dictionary containing global embeddings and metadata
        """
        import cv2
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate frames to sample based on duration and sampling FPS
        num_frames_to_sample = int(duration * sampling_fps)
        
        # Apply max_frames limit from config
        max_frames = self.config.get("model_specific", {}).get("max_frames", 40)
        if num_frames_to_sample > max_frames:
            logger.info(f"Limiting frames from {num_frames_to_sample} to {max_frames} (max_frames config)")
            num_frames_to_sample = max_frames
        
        logger.info(f"Processing entire video: {video_path.name}")
        logger.info(f"  Duration: {duration:.1f}s, Total frames: {total_frames}")
        logger.info(f"  Sampling at {sampling_fps} FPS = {num_frames_to_sample} frames")
        
        # Sample frames uniformly across the video
        if num_frames_to_sample > total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int).tolist()
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        
        # Extract global embeddings
        embedding_result = self.extract_embeddings(frames)
        
        # Convert to Vespa format for global embeddings
        float_embeddings, binary_embeddings = self.embeddings_to_vespa_format(
            embedding_result["embeddings"]
        )
        
        return {
            "embeddings": embedding_result["embeddings"],
            "float_embeddings": float_embeddings,
            "binary_embeddings": binary_embeddings,
            "num_frames_sampled": len(frames),
            "duration": duration,
            "sampling_fps": sampling_fps,
            "embedding_dim": self.embedding_dim,
            "num_patches": 1,  # Global embedding is a single vector
            "is_global": True
        }
    
    def load_text_encoder(self):
        """Load text encoder components for LVT models"""
        if self.text_encoder is not None:
            return  # Already loaded
        
        logger.info(f"Loading text encoder for {self.model_name}")
        
        try:
            # Use our text encoder implementation - handle both relative and absolute imports
            try:
                from .videoprism_text_encoder import VideoPrismTextEncoder
            except ImportError:
                # Fallback to absolute import
                from src.common.models.videoprism_text_encoder import VideoPrismTextEncoder
            
            self.text_encoder = VideoPrismTextEncoder(
                self.model_name,
                self.embedding_dim
            )
            self.text_encoder.load()
            logger.info("Text encoder loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load text encoder: {e}")
            self.text_encoder = None
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text query to embeddings
        
        Args:
            text: Text query to encode
            
        Returns:
            Numpy array of shape (embedding_dim,) for global embeddings
        """
        if self.model is None:
            self.load_model()
        
        if self.text_encoder is None:
            self.load_text_encoder()
        
        if self.text_encoder is None:
            raise RuntimeError("Text encoder not available for this model")
        
        logger.info(f"Encoding text query: '{text}'")
        
        # Use our text encoder implementation
        embeddings = self.text_encoder.encode(text)
        logger.info(f"Generated text embeddings shape: {embeddings.shape}")
        
        return embeddings


def get_videoprism_loader(model_name: str = "videoprism_public_v1_base_hf", config: Optional[Dict[str, Any]] = None) -> VideoPrismLoader:
    """Get or create VideoPrism loader instance for specific model"""
    global _videoprism_loaders
    
    # Check if this is a global model request
    if "global" in model_name or "lvt" in model_name:
        if model_name not in _videoprism_loaders:
            _videoprism_loaders[model_name] = VideoPrismGlobalLoader(model_name, config)
    else:
        if model_name not in _videoprism_loaders:
            _videoprism_loaders[model_name] = VideoPrismLoader(model_name, config)
    
    return _videoprism_loaders[model_name]