"""
Query encoders for different video processing profiles
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import logging

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
        from colpali_engine.models import ColIdefics3, ColIdefics3Processor
        
        # Detect device
        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            dtype = torch.float32
        else:
            self.device = "cpu"
            dtype = torch.float32
            
        self.model = ColIdefics3.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device
        ).eval()
        self.processor = ColIdefics3Processor.from_pretrained(model_name)
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
        # Load appropriate ColQwen model and processor
        if "omni" in model_name.lower():
            from colpali_engine.models import ColQwen2_5Omni, ColQwen2_5OmniProcessor
            model_class = ColQwen2_5Omni
            processor_class = ColQwen2_5OmniProcessor
        else:
            from colpali_engine.models import ColQwen2, ColQwen2Processor
            model_class = ColQwen2
            processor_class = ColQwen2Processor
        
        # Detect device
        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            dtype = torch.float32
        else:
            self.device = "cpu"
            dtype = torch.float32
            
        self.model = model_class.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device
        ).eval()
        self.processor = processor_class.from_pretrained(model_name)
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
        self.model = None
        self.text_tokenizer = None
        self.forward_fn = None
        
        # VideoPrism dimensions
        self.is_global = "lvt" in model_name.lower()  # Check if this is a global embedding model (LVT models)
        if "large" in model_name:
            self.embedding_dim = 1024
            self.num_patches = 2048  # For patch-based models
        else:
            self.embedding_dim = 768
            self.num_patches = 4096  # For patch-based models
        
        # Try to load VideoPrism with text encoding
        self._load_model()
    
    def _load_model(self):
        """Load VideoPrism model with text encoding capabilities"""
        try:
            import sys
            import os
            from pathlib import Path
            
            # Force CPU backend to avoid Metal issues
            os.environ["JAX_PLATFORM_NAME"] = "cpu"
            os.environ["JAX_PLATFORMS"] = "cpu"
            
            # Add VideoPrism to path
            videoprism_path = Path("/Users/amjain/source/hobby/videoprism")
            if not videoprism_path.exists():
                raise ImportError(f"VideoPrism not found at {videoprism_path}")
                
            sys.path.insert(0, str(videoprism_path))
            
            # Import VideoPrism modules
            from videoprism import models as vp
            
            # Load text tokenizer
            self.text_tokenizer = vp.load_text_tokenizer('c4_en')
            logger.info("Loaded VideoPrism text tokenizer")
            
            # Load model using the same approach as video encoding
            # Map our model name to VideoPrism's naming
            # We need the LVT (Language-Vision-Text) version for text encoding
            if "base" in self.model_name:
                vp_model_name = "videoprism_lvt_public_v1_base"
            else:
                vp_model_name = "videoprism_lvt_public_v1_large"
            
            # Load model and weights
            self.model = vp.MODELS[vp_model_name]()
            state = vp.load_pretrained_weights(vp_model_name)
            self.state = state
            
            # Create forward function for text encoding
            import jax
            @jax.jit
            def text_forward_fn(text_ids, text_paddings):
                # Forward pass with only text (no video frames)
                # FactorizedVideoCLIP expects: inputs, text_token_ids, text_paddings
                _, text_embeddings, _ = self.model.apply(
                    state,
                    inputs=None,  # No video frames
                    text_token_ids=text_ids,
                    text_paddings=text_paddings,
                    train=False,
                    normalize=True
                )
                return text_embeddings
            
            self.forward_fn = text_forward_fn
            self.vp = vp  # Store module reference
            logger.info(f"Loaded VideoPrism text encoder for {self.model_name}")
                
        except Exception as e:
            logger.error(f"Failed to load VideoPrism text encoder: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.text_tokenizer = None
    
    def encode(self, query: str) -> np.ndarray:
        """Encode text query to multi-patch embeddings matching VideoPrism format"""
        if self.text_tokenizer is None or self.forward_fn is None:
            raise RuntimeError(
                "VideoPrism text encoder not loaded. Ensure VideoPrism is installed "
                "and the model weights are available."
            )
        
        # Add prompt template to make it more video-like
        query_with_prompt = f"A video of {query}"
        
        # Tokenize text using VideoPrism's tokenizer
        text_ids, text_paddings = self.vp.tokenize_texts(
            self.text_tokenizer, 
            [query_with_prompt]
        )
        
        # Generate embeddings
        try:
            text_embeddings = self.forward_fn(text_ids, text_paddings)
            # Convert JAX array to numpy and remove batch dimension
            embeddings_np = np.array(text_embeddings[0])
            
            logger.info(f"Generated text embeddings shape: {embeddings_np.shape}")
            
            # For global embedding models (LVT), return as-is (single vector)
            if self.is_global:
                logger.info("Using global embedding format for LVT model")
                # Ensure we return a 1D array for global models
                if len(embeddings_np.shape) > 1:
                    embeddings_np = embeddings_np.flatten()
                return embeddings_np
            
            # For patch-based models, we need to tile the embedding
            # VideoPrism text encoder returns a single embedding vector.
            # For Vespa search which expects multiple query tokens, we need to
            # tile it to match the expected format. This is different from the
            # standard VideoPrism similarity computation shown in their Colab.
            else:
                if len(embeddings_np.shape) == 1:
                    # Single vector - tile it to create multi-patch format for Vespa
                    # This allows the text embedding to be compared against each video patch
                    embeddings_np = np.tile(embeddings_np, (self.num_patches, 1))
                    logger.info(f"Tiled text embedding for Vespa format: {embeddings_np.shape}")
                
                return embeddings_np
            
        except Exception as e:
            logger.error(f"Failed to encode text query: {e}")
            raise
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim


class QueryEncoderFactory:
    """Factory to create appropriate query encoder based on profile"""
    
    @staticmethod
    def create_encoder(profile: str, model_name: Optional[str] = None) -> QueryEncoder:
        """Create query encoder for the given profile"""
        
        if profile == "frame_based_colpali":
            return ColPaliQueryEncoder(model_name or "vidore/colsmol-500m")
        
        elif profile == "direct_video_colqwen":
            return ColQwenQueryEncoder(model_name or "vidore/colqwen-omni-v0.1")
        
        elif profile == "direct_video_frame":
            # VideoPrism Base
            return VideoPrismQueryEncoder(model_name or "videoprism_public_v1_base_hf")
        
        elif profile == "direct_video_frame_large":
            # VideoPrism Large
            return VideoPrismQueryEncoder(model_name or "videoprism_public_v1_large_hf")
        
        elif profile == "direct_video_global":
            # VideoPrism Global (LVT) Base
            return VideoPrismQueryEncoder(model_name or "videoprism_lvt_public_v1_base")
        
        elif profile == "direct_video_global_large":
            # VideoPrism Global (LVT) Large
            return VideoPrismQueryEncoder(model_name or "videoprism_lvt_public_v1_large")
        
        else:
            raise ValueError(f"Unknown profile: {profile}")
    
    @staticmethod
    def get_supported_profiles() -> list:
        """Return list of supported profiles"""
        return [
            "frame_based_colpali",
            "direct_video_colqwen",
            "direct_video_frame",
            "direct_video_frame_large",
            "direct_video_global",
            "direct_video_global_large"
        ]