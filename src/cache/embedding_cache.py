"""
Specialized cache for embeddings
"""

import hashlib
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .base import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Track cache performance statistics"""
    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    
    def record_hit(self):
        """Record a cache hit"""
        self.hits += 1
        self.total_requests += 1
    
    def record_miss(self):
        """Record a cache miss"""
        self.misses += 1
        self.total_requests += 1
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': self.total_requests,
            'hit_rate': self.hit_rate
        }


class EmbeddingCache:
    """Specialized cache for embeddings with efficient storage"""
    
    def __init__(self, cache_manager: CacheManager, ttl: int = 86400):
        """
        Initialize embedding cache
        
        Args:
            cache_manager: The cache manager instance
            ttl: Time to live in seconds (default 24 hours)
        """
        self.cache = cache_manager
        self.ttl = ttl
        self.stats = CacheStats()
    
    def _generate_key(self, text: str, model: str, prefix: str = "embedding") -> str:
        """Generate cache key from text and model"""
        # Create a deterministic key
        content = f"{model}:{text}"
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"{prefix}:{model}:{content_hash}"
    
    async def get_embedding(
        self, 
        text: str, 
        model: str
    ) -> Optional[np.ndarray]:
        """
        Get embedding from cache
        
        Args:
            text: The text that was embedded
            model: The model used for embedding
            
        Returns:
            Numpy array if found, None otherwise
        """
        key = self._generate_key(text, model)
        
        cached_data = await self.cache.get(key)
        if cached_data is not None:
            self.stats.record_hit()
            
            # Handle different storage formats
            if isinstance(cached_data, dict):
                # Reconstruct numpy array from dict
                dtype = cached_data.get('dtype', 'float32')
                shape = cached_data.get('shape', [])
                data = cached_data.get('data', [])
                
                if isinstance(data, bytes):
                    # Binary format
                    embedding = np.frombuffer(data, dtype=dtype).reshape(shape)
                else:
                    # List format
                    embedding = np.array(data, dtype=dtype).reshape(shape)
            else:
                # Legacy format - assume it's already a numpy array
                embedding = cached_data
            
            logger.debug(f"Cache hit for embedding: {key}")
            return embedding
        
        self.stats.record_miss()
        logger.debug(f"Cache miss for embedding: {key}")
        return None
    
    async def set_embedding(
        self, 
        text: str, 
        model: str, 
        embedding: np.ndarray
    ) -> bool:
        """
        Store embedding in cache
        
        Args:
            text: The text that was embedded
            model: The model used for embedding
            embedding: The embedding numpy array
            
        Returns:
            True if stored successfully
        """
        key = self._generate_key(text, model)
        
        # Store in an efficient format
        cache_data = {
            'dtype': str(embedding.dtype),
            'shape': list(embedding.shape),
            'data': embedding.tobytes(),  # Binary format for efficiency
            'text_preview': text[:100],  # Store preview for debugging
            'model': model,
            'timestamp': datetime.now().isoformat()
        }
        
        success = await self.cache.set(key, cache_data, self.ttl)
        
        if success:
            logger.debug(f"Stored embedding in cache: {key}")
        else:
            logger.warning(f"Failed to store embedding in cache: {key}")
        
        return success
    
    async def get_batch_embeddings(
        self,
        texts: list[str],
        model: str
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Get multiple embeddings from cache
        
        Args:
            texts: List of texts
            model: The model used for embedding
            
        Returns:
            Dictionary mapping text to embedding (or None if not cached)
        """
        results = {}
        
        for text in texts:
            embedding = await self.get_embedding(text, model)
            results[text] = embedding
        
        return results
    
    async def set_batch_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        model: str
    ) -> Dict[str, bool]:
        """
        Store multiple embeddings in cache
        
        Args:
            embeddings: Dictionary mapping text to embedding
            model: The model used for embedding
            
        Returns:
            Dictionary mapping text to success status
        """
        results = {}
        
        for text, embedding in embeddings.items():
            success = await self.set_embedding(text, model, embedding)
            results[text] = success
        
        return results
    
    async def clear_model_cache(self, model: str) -> int:
        """Clear all cached embeddings for a specific model"""
        # This would require pattern-based clearing
        # For now, return 0 as it's not implemented
        logger.warning("Model-specific cache clearing not yet implemented")
        return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self.stats.to_dict()