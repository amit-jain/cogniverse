"""Query encoders for different video processing profiles."""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.common.models import get_or_load_model
from src.common.config import get_config

logger = logging.getLogger(__name__)


class QueryEncoder(ABC):
    """Abstract base class for query encoders"""
    
    @abstractmethod
    def encode(self, query: str) -> np.ndarray:
        """Encode a text query into embeddings"""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension"""
        pass


class ColPaliQueryEncoder(QueryEncoder):
    """Query encoder for ColPali models"""
    
    def __init__(self, model_name: str = "vidore/colsmol-500m"):
        # Use v2 model loader for consistency
        config = {"colpali_model": model_name}
        self.model, self.processor = get_or_load_model(model_name, config, logger)
        
        # Get device from model
        self.device = next(self.model.parameters()).device
        self.embedding_dim = 128  # ColPali uses 128-dim embeddings
        logger.info(f"Loaded ColPali query encoder: {model_name} on {self.device}")
    
    def encode(self, query: str) -> np.ndarray:
        """Encode query to multi-vector embeddings"""
        batch_queries = self.processor.process_queries([query]).to(self.device)
        with torch.no_grad():
            query_embeddings = self.model(**batch_queries)
        # Return as numpy array [num_patches, embedding_dim]
        return query_embeddings.cpu().numpy().squeeze(0)
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class ColQwenQueryEncoder(QueryEncoder):
    """Query encoder for ColQwen models"""
    
    def __init__(self, model_name: str = "vidore/colqwen-omni-v0.1"):
        # Use v2 model loader for consistency
        config = {"colpali_model": model_name}
        self.model, self.processor = get_or_load_model(model_name, config, logger)
        
        # Get device from model
        self.device = next(self.model.parameters()).device
        self.embedding_dim = 128  # ColQwen uses 128-dim embeddings
        logger.info(f"Loaded ColQwen query encoder: {model_name} on {self.device}")
    
    def encode(self, query: str) -> np.ndarray:
        """Encode query to multi-vector embeddings"""
        batch_queries = self.processor.process_queries([query]).to(self.device)
        with torch.no_grad():
            query_embeddings = self.model(**batch_queries)
        # Return as numpy array [num_patches, embedding_dim]
        return query_embeddings.cpu().numpy().squeeze(0)
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class VideoPrismQueryEncoder(QueryEncoder):
    """Query encoder for VideoPrism models"""
    
    def __init__(self, model_name: str = "videoprism_public_v1_base_hf"):
        self.model_name = model_name
        
        # VideoPrism dimensions
        self.is_global = "lvt" in model_name.lower() or "global" in model_name.lower()
        if "large" in model_name:
            self.embedding_dim = 1024
            self.num_patches = 2048  # For patch-based models
        else:
            self.embedding_dim = 768
            self.num_patches = 4096  # For patch-based models
        
        # Use v2 model loader - it returns the videoprism loader instance
        config = {"colpali_model": model_name, "model_name": model_name}
        logger.info(f"Creating VideoPrism loader for model: {model_name}")
        self.videoprism_loader, _ = get_or_load_model(model_name, config, logger)
        logger.info(f"Got loader type: {type(self.videoprism_loader).__name__}")
        logger.info(f"Loader has encode_text: {hasattr(self.videoprism_loader, 'encode_text')}")
        logger.info(f"Loader has text_tokenizer: {hasattr(self.videoprism_loader, 'text_tokenizer')}")
        
        # Get text encoding components from the loader
        if hasattr(self.videoprism_loader, 'text_tokenizer'):
            self.text_tokenizer = self.videoprism_loader.text_tokenizer
            self.forward_fn = self.videoprism_loader.text_forward_fn if hasattr(self.videoprism_loader, 'text_forward_fn') else None
            logger.info(f"Loaded VideoPrism query encoder: {model_name}")
        else:
            logger.warning(f"VideoPrism loader doesn't have text encoder support")
            self.text_tokenizer = None
            self.forward_fn = None
    
    def encode(self, query: str) -> np.ndarray:
        """Encode text query to embeddings matching VideoPrism format"""
        if not hasattr(self.videoprism_loader, 'encode_text'):
            raise RuntimeError(
                "VideoPrism loader doesn't have text encoding support. "
                "Ensure the model supports text encoding (LVT models)."
            )
        
        try:
            # Use the videoprism loader's text encoding method
            embeddings_np = self.videoprism_loader.encode_text(query)
            
            logger.info(f"Generated text embeddings shape: {embeddings_np.shape}")
            
            # For global embedding models, return as-is (single vector)
            if self.is_global:
                logger.info("Using global embedding format")
                # Ensure we return a 1D array for global models
                if len(embeddings_np.shape) > 1:
                    embeddings_np = embeddings_np.flatten()
                return embeddings_np
            
            # For patch-based models, the loader should handle the format
            return embeddings_np
            
        except Exception as e:
            logger.error(f"Failed to encode text query: {e}")
            raise
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class QueryEncoderFactory:
    """Factory to create appropriate query encoder based on profile"""
    
    _encoder_cache = {}  # Cache for config-based encoder mappings
    
    @staticmethod
    def create_encoder(profile: str, model_name: Optional[str] = None) -> QueryEncoder:
        """Create query encoder for the given profile
        
        Dynamically determines the encoder based on config.json video_processing_profiles
        """
        config = get_config()
        video_profiles = config.get("video_processing_profiles", {})
        
        # Check if profile exists in config
        if profile not in video_profiles:
            raise ValueError(f"Unknown profile: {profile}. Available profiles: {list(video_profiles.keys())}")
        
        profile_config = video_profiles[profile]
        
        # Use provided model_name or get from config
        if not model_name:
            model_name = profile_config.get("embedding_model")
            if not model_name:
                raise ValueError(f"No embedding_model specified for profile: {profile}")
        
        # Determine encoder type based on model name
        if "colpali" in model_name.lower() or "colsmol" in model_name.lower():
            return ColPaliQueryEncoder(model_name)
        elif "colqwen" in model_name.lower():
            return ColQwenQueryEncoder(model_name)
        elif "videoprism" in model_name.lower():
            return VideoPrismQueryEncoder(model_name)
        else:
            # Fallback: try to infer from profile name
            if "colpali" in profile.lower():
                return ColPaliQueryEncoder(model_name)
            elif "colqwen" in profile.lower():
                return ColQwenQueryEncoder(model_name)
            elif "videoprism" in profile.lower():
                return VideoPrismQueryEncoder(model_name)
            else:
                raise ValueError(f"Cannot determine encoder type for model: {model_name} in profile: {profile}")
    
    @staticmethod
    def get_supported_profiles() -> list:
        """Return list of supported profiles from config.json"""
        config = get_config()
        video_profiles = config.get("video_processing_profiles", {})
        return list(video_profiles.keys())