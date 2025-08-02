"""
VideoPrism text encoder implementation for LVT models.

This provides text encoding support for VideoPrism global models (LVT variants)
which support cross-modal retrieval between text and video.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class VideoPrismTextEncoder:
    """Text encoder for VideoPrism LVT models"""
    
    def __init__(self, model_name: str, embedding_dim: int):
        """
        Initialize text encoder for VideoPrism
        
        Args:
            model_name: VideoPrism model variant
            embedding_dim: Output embedding dimension (768 or 1024)
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.text_encoder = None
        
    def load(self):
        """Load text encoder components"""
        logger.info(f"Loading text encoder for {self.model_name}")
        
        try:
            # Import VideoPrism models
            import sys
            from pathlib import Path
            from src.tools.config import get_config
            
            config = get_config()
            videoprism_path = config.get("videoprism_repo_path")
            
            if videoprism_path:
                sys.path.insert(0, str(videoprism_path))
            
            from videoprism import models as vp
            
            # Load the text tokenizer
            self.tokenizer = vp.load_text_tokenizer('c4_en')
            
            # Load the model to get text encoding capabilities
            model = vp.get_model(self.model_name)
            state = vp.load_pretrained_weights(self.model_name)
            
            # Store model and state for text encoding
            self.model = model
            self.state = state
            self._vp = vp
            
            logger.info(f"VideoPrism text encoder loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load text encoder: {e}")
            raise
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to embeddings
        
        Args:
            text: Input text query
            
        Returns:
            Numpy array of shape (embedding_dim,)
        """
        if not hasattr(self, 'model') or self.model is None:
            self.load()
        
        try:
            # Format text with prompt template as shown in the notebook
            PROMPT_TEMPLATE = 'a video of {}.'
            formatted_text = PROMPT_TEMPLATE.format(text)
            
            # Tokenize text - returns both ids and paddings as numpy arrays
            text_ids, text_paddings = self._vp.tokenize_texts(self.tokenizer, [formatted_text])
            
            # Generate embeddings using model.apply() directly like in the notebook
            # For text-only encoding, pass None for video inputs
            video_embeddings, text_embeddings, _ = self.model.apply(
                self.state,
                None,  # No video input
                text_ids,
                text_paddings,
                train=False
            )
            
            # Extract text embeddings and convert to numpy
            embeddings_np = np.array(text_embeddings[0])  # Remove batch dimension
            
            # Ensure correct dimension
            if embeddings_np.shape[0] != self.embedding_dim:
                # In production, this would use the actual VideoPrism projection layer
                current_dim = embeddings_np.shape[0]
                if current_dim < self.embedding_dim:
                    # Pad with zeros
                    padding = np.zeros(self.embedding_dim - current_dim)
                    embeddings_np = np.concatenate([embeddings_np, padding])
                else:
                    # Truncate
                    embeddings_np = embeddings_np[:self.embedding_dim]
            
            # NO normalization - the notebook doesn't normalize
            return embeddings_np
            
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise