"""
Per-Modality Caching

Cache results by modality with intelligent TTL and LRU eviction.
Part of Phase 12: Production Readiness.
"""

import hashlib
import logging
import time
from collections import OrderedDict, defaultdict
from typing import Any, Dict, Optional

from src.app.search.multi_modal_reranker import QueryModality

logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU cache implementation"""

    def __init__(self, maxsize: int = 1000):
        """
        Initialize LRU cache

        Args:
            maxsize: Maximum number of items to cache
        """
        self.maxsize = maxsize
        self.cache: OrderedDict = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, updating LRU order"""
        if key not in self.cache:
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: str, value: Any):
        """Put item in cache, evicting oldest if full"""
        if key in self.cache:
            # Update existing, move to end
            self.cache.move_to_end(key)
        else:
            # Add new item
            if len(self.cache) >= self.maxsize:
                # Evict oldest (first item)
                self.cache.popitem(last=False)

        self.cache[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache"""
        return key in self.cache

    def __len__(self) -> int:
        """Get cache size"""
        return len(self.cache)

    def clear(self):
        """Clear all cached items"""
        self.cache.clear()


class ModalityCacheManager:
    """
    Cache results per modality with intelligent invalidation

    Features:
    - Separate LRU cache per modality
    - TTL-based expiration
    - Hit/miss tracking
    - Cache statistics per modality

    Example:
        cache = ModalityCacheManager()

        # Cache a result
        cache.cache_result(
            query="machine learning",
            modality=QueryModality.VIDEO,
            result={"videos": [...]}
        )

        # Retrieve cached result
        cached = cache.get_cached_result(
            query="machine learning",
            modality=QueryModality.VIDEO,
            ttl_seconds=3600
        )
    """

    def __init__(self, cache_size_per_modality: int = 1000):
        """
        Initialize cache manager

        Args:
            cache_size_per_modality: Max cache size per modality
        """
        # Create separate cache for each modality
        self.caches = {
            modality: LRUCache(maxsize=cache_size_per_modality)
            for modality in QueryModality
        }

        # Track cache statistics
        self.cache_stats = defaultdict(lambda: {"hits": 0, "misses": 0, "evictions": 0})

        logger.info(
            f"💾 Initialized ModalityCacheManager "
            f"(size_per_modality: {cache_size_per_modality})"
        )

    def get_cached_result(
        self,
        query: str,
        modality: QueryModality,
        ttl_seconds: int = 3600,
    ) -> Optional[Any]:
        """
        Get cached result if available and fresh

        Args:
            query: Query string
            modality: Query modality
            ttl_seconds: Time-to-live in seconds

        Returns:
            Cached result if available and fresh, None otherwise
        """
        cache_key = self._generate_cache_key(query, modality)

        if cache_key in self.caches[modality]:
            cached_entry = self.caches[modality].get(cache_key)

            # Check TTL
            if time.time() - cached_entry["timestamp"] < ttl_seconds:
                self.cache_stats[modality]["hits"] += 1

                logger.debug(
                    f"✅ Cache HIT: {modality.value} - {query[:50]}..."
                )

                return cached_entry["result"]
            else:
                # Expired
                logger.debug(
                    f"⏰ Cache EXPIRED: {modality.value} - {query[:50]}..."
                )

        self.cache_stats[modality]["misses"] += 1

        logger.debug(
            f"❌ Cache MISS: {modality.value} - {query[:50]}..."
        )

        return None

    def cache_result(
        self,
        query: str,
        modality: QueryModality,
        result: Any,
    ):
        """
        Store result in modality-specific cache

        Args:
            query: Query string
            modality: Query modality
            result: Result to cache
        """
        cache_key = self._generate_cache_key(query, modality)

        # Check if we're about to evict
        if len(self.caches[modality]) >= self.caches[modality].maxsize:
            if cache_key not in self.caches[modality]:
                self.cache_stats[modality]["evictions"] += 1

        self.caches[modality].put(
            cache_key,
            {
                "result": result,
                "timestamp": time.time(),
            },
        )

        logger.debug(
            f"💾 Cached: {modality.value} - {query[:50]}..."
        )

    def _generate_cache_key(self, query: str, modality: QueryModality) -> str:
        """
        Generate cache key from query and modality

        Args:
            query: Query string
            modality: Query modality

        Returns:
            Cache key (hash)
        """
        # Create deterministic key from query + modality
        key_string = f"{modality.value}:{query.lower().strip()}"
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get_cache_stats(self, modality: Optional[QueryModality] = None) -> Dict[str, Any]:
        """
        Get cache statistics

        Args:
            modality: Specific modality, or None for all

        Returns:
            Cache statistics
        """
        if modality:
            stats = self.cache_stats[modality]
            total = stats["hits"] + stats["misses"]
            hit_rate = stats["hits"] / total if total > 0 else 0.0

            return {
                "modality": modality.value,
                "hits": stats["hits"],
                "misses": stats["misses"],
                "evictions": stats["evictions"],
                "hit_rate": hit_rate,
                "cache_size": len(self.caches[modality]),
                "cache_capacity": self.caches[modality].maxsize,
            }
        else:
            # Aggregate stats for all modalities
            all_stats = {}
            for mod in QueryModality:
                all_stats[mod.value] = self.get_cache_stats(mod)
            return all_stats

    def invalidate_modality(self, modality: QueryModality):
        """
        Invalidate all cache entries for a modality

        Args:
            modality: Modality to invalidate
        """
        self.caches[modality].clear()
        logger.info(f"🗑️ Invalidated cache for {modality.value}")

    def invalidate_all(self):
        """Invalidate all caches"""
        for modality in QueryModality:
            self.caches[modality].clear()

        logger.info("🗑️ Invalidated all caches")

    def reset_stats(self):
        """Reset cache statistics"""
        self.cache_stats.clear()
        logger.info("📊 Reset cache statistics")
