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

from cogniverse_agents.search.multi_modal_reranker import QueryModality

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
            cache_size_per_modality: Max cache size (note: single cache now, not per-modality)
        """
        # Single cache for all queries
        self.cache = LRUCache(maxsize=cache_size_per_modality * len(QueryModality))

        # Track cache statistics per modality
        self.cache_stats = defaultdict(lambda: {"hits": 0, "misses": 0, "evictions": 0})

        logger.info(
            f"💾 Initialized ModalityCacheManager "
            f"(total_size: {cache_size_per_modality * len(QueryModality)})"
        )

    def get_cached_result_any_modality(
        self,
        query: str,
        ttl_seconds: int = 3600,
    ) -> Optional[Any]:
        """
        Get cached result with direct O(1) lookup

        Args:
            query: Query string
            ttl_seconds: Time-to-live in seconds

        Returns:
            Cached result if found and fresh, None otherwise
        """
        cache_key = self._generate_cache_key(query)

        if cache_key in self.cache:
            cached_entry = self.cache.get(cache_key)

            # Check TTL
            if time.time() - cached_entry["timestamp"] < ttl_seconds:
                modality = cached_entry["modality"]
                self.cache_stats[modality]["hits"] += 1
                logger.debug(f"✅ Cache HIT: {modality.value} - {query[:50]}...")
                return cached_entry["result"]
            else:
                # Expired
                logger.debug(f"⏰ Cache EXPIRED: {query[:50]}...")

        # Not found or expired
        logger.debug(f"❌ Cache MISS: {query[:50]}...")
        return None

    def get_cached_result(
        self,
        query: str,
        modality: QueryModality,
        ttl_seconds: int = 3600,
    ) -> Optional[Any]:
        """
        Get cached result if available and fresh

        Note: This method is for compatibility with code that knows the modality.
        Most code should use get_cached_result_any_modality() instead.

        Args:
            query: Query string
            modality: Expected query modality (for stats tracking)
            ttl_seconds: Time-to-live in seconds

        Returns:
            Cached result if available and fresh, None otherwise
        """
        cache_key = self._generate_cache_key(query)

        if cache_key in self.cache:
            cached_entry = self.cache.get(cache_key)

            # Check TTL
            if time.time() - cached_entry["timestamp"] < ttl_seconds:
                actual_modality = cached_entry["modality"]
                self.cache_stats[actual_modality]["hits"] += 1

                logger.debug(f"✅ Cache HIT: {actual_modality.value} - {query[:50]}...")

                return cached_entry["result"]
            else:
                # Expired
                logger.debug(f"⏰ Cache EXPIRED: {modality.value} - {query[:50]}...")

        self.cache_stats[modality]["misses"] += 1

        logger.debug(f"❌ Cache MISS: {modality.value} - {query[:50]}...")

        return None

    def cache_result(
        self,
        query: str,
        modality: QueryModality,
        result: Any,
    ):
        """
        Store result in cache with modality

        Args:
            query: Query string
            modality: Query modality
            result: Result to cache
        """
        cache_key = self._generate_cache_key(query)

        # Check if we're about to evict
        if len(self.cache) >= self.cache.maxsize:
            if cache_key not in self.cache:
                self.cache_stats[modality]["evictions"] += 1

        self.cache.put(
            cache_key,
            {
                "result": result,
                "modality": modality,
                "timestamp": time.time(),
            },
        )

        logger.debug(f"💾 Cached: {modality.value} - {query[:50]}...")

    def _generate_cache_key(self, query: str) -> str:
        """
        Generate cache key from query only

        Args:
            query: Query string

        Returns:
            Cache key (hash)

        Note: Modality is determined by which bucket the result is stored in,
        not part of the cache key. This allows searching across all buckets.
        """
        # Create deterministic key from query only
        key_string = query.lower().strip()
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get_cache_stats(
        self, modality: Optional[QueryModality] = None
    ) -> Dict[str, Any]:
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
                "cache_size": len(self.cache),  # Total cache size
                "cache_capacity": self.cache.maxsize,
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
        # Remove entries for this modality
        keys_to_remove = []
        for key in list(self.cache.cache.keys()):
            entry = self.cache.cache[key]
            if entry["modality"] == modality:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.cache.cache[key]

        logger.info(f"🗑️ Invalidated {len(keys_to_remove)} cache entries for {modality.value}")

    def invalidate_all(self):
        """Invalidate all caches"""
        self.cache.clear()
        logger.info("🗑️ Invalidated all caches")

    def reset_stats(self):
        """Reset cache statistics"""
        self.cache_stats.clear()
        logger.info("📊 Reset cache statistics")
