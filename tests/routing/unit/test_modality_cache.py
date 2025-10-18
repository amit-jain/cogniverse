"""
Unit tests for ModalityCacheManager
"""


import pytest
from cogniverse_agents.routing.modality_cache import LRUCache, ModalityCacheManager
from cogniverse_agents.search.multi_modal_reranker import QueryModality

from tests.utils.async_polling import wait_for_cache_expiration


class TestLRUCache:
    """Test LRUCache implementation"""

    def test_initialization(self):
        """Test cache initialization"""
        cache = LRUCache(maxsize=10)
        assert len(cache) == 0
        assert cache.maxsize == 10

    def test_put_and_get(self):
        """Test basic put and get"""
        cache = LRUCache(maxsize=10)

        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_nonexistent_returns_none(self):
        """Test getting non-existent key returns None"""
        cache = LRUCache(maxsize=10)
        assert cache.get("nonexistent") is None

    def test_contains(self):
        """Test __contains__ method"""
        cache = LRUCache(maxsize=10)
        cache.put("key1", "value1")

        assert "key1" in cache
        assert "key2" not in cache

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full"""
        cache = LRUCache(maxsize=3)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Cache is full, adding new item should evict oldest (key1)
        cache.put("key4", "value4")

        assert "key1" not in cache
        assert "key2" in cache
        assert "key3" in cache
        assert "key4" in cache

    def test_lru_order_updated_on_get(self):
        """Test that getting an item updates LRU order"""
        cache = LRUCache(maxsize=3)

        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")

        # Access key1 to make it most recently used
        cache.get("key1")

        # Add new item, should evict key2 (oldest)
        cache.put("key4", "value4")

        assert "key1" in cache  # Should still be here
        assert "key2" not in cache  # Should be evicted
        assert "key3" in cache
        assert "key4" in cache

    def test_update_existing_key(self):
        """Test updating existing key"""
        cache = LRUCache(maxsize=3)

        cache.put("key1", "value1")
        cache.put("key1", "value1_updated")

        assert cache.get("key1") == "value1_updated"
        assert len(cache) == 1

    def test_clear(self):
        """Test clearing cache"""
        cache = LRUCache(maxsize=10)

        cache.put("key1", "value1")
        cache.put("key2", "value2")

        cache.clear()

        assert len(cache) == 0
        assert "key1" not in cache


class TestModalityCacheManager:
    """Test ModalityCacheManager functionality"""

    @pytest.fixture
    def cache_manager(self):
        """Create cache manager instance"""
        return ModalityCacheManager(cache_size_per_modality=10)

    def test_initialization(self, cache_manager):
        """Test cache manager initialization"""
        # Should have cache for each modality
        for modality in QueryModality:
            assert modality in cache_manager.caches
            assert isinstance(cache_manager.caches[modality], LRUCache)

    def test_cache_and_retrieve_result(self, cache_manager):
        """Test caching and retrieving result"""
        query = "machine learning tutorials"
        result = {"videos": ["video1", "video2"]}

        # Cache result
        cache_manager.cache_result(query, QueryModality.VIDEO, result)

        # Retrieve cached result
        cached = cache_manager.get_cached_result(
            query, QueryModality.VIDEO, ttl_seconds=3600
        )

        assert cached == result

    def test_cache_miss_returns_none(self, cache_manager):
        """Test cache miss returns None"""
        cached = cache_manager.get_cached_result(
            "nonexistent query", QueryModality.VIDEO, ttl_seconds=3600
        )

        assert cached is None

    def test_different_modalities_separate_caches(self, cache_manager):
        """Test different modalities use separate caches"""
        query = "test query"
        video_result = {"videos": ["v1"]}
        doc_result = {"documents": ["d1"]}

        cache_manager.cache_result(query, QueryModality.VIDEO, video_result)
        cache_manager.cache_result(query, QueryModality.DOCUMENT, doc_result)

        # Each modality should have its own cached result
        assert (
            cache_manager.get_cached_result(query, QueryModality.VIDEO, 3600)
            == video_result
        )
        assert (
            cache_manager.get_cached_result(query, QueryModality.DOCUMENT, 3600)
            == doc_result
        )

    def test_ttl_expiration(self, cache_manager):
        """Test TTL expiration"""
        query = "test query"
        result = {"data": "test"}

        cache_manager.cache_result(query, QueryModality.VIDEO, result)

        # Should be cached
        assert (
            cache_manager.get_cached_result(query, QueryModality.VIDEO, ttl_seconds=1)
            is not None
        )

        # Wait for expiration
        wait_for_cache_expiration(ttl=1.0, buffer=0.1)

        # Should be expired
        assert (
            cache_manager.get_cached_result(query, QueryModality.VIDEO, ttl_seconds=1)
            is None
        )

    def test_cache_key_normalization(self, cache_manager):
        """Test cache key normalization (case-insensitive, whitespace)"""
        result = {"data": "test"}

        cache_manager.cache_result("  Test Query  ", QueryModality.VIDEO, result)

        # Different formatting should hit same cache
        assert (
            cache_manager.get_cached_result("test query", QueryModality.VIDEO, 3600)
            == result
        )
        assert (
            cache_manager.get_cached_result("TEST QUERY", QueryModality.VIDEO, 3600)
            == result
        )

    def test_cache_stats_hits_and_misses(self, cache_manager):
        """Test cache statistics tracking"""
        query = "test"
        result = {"data": "test"}

        # Cache a result
        cache_manager.cache_result(query, QueryModality.VIDEO, result)

        # Hit
        cache_manager.get_cached_result(query, QueryModality.VIDEO, 3600)

        # Miss
        cache_manager.get_cached_result("other query", QueryModality.VIDEO, 3600)

        stats = cache_manager.get_cache_stats(QueryModality.VIDEO)

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_stats_all_modalities(self, cache_manager):
        """Test getting stats for all modalities"""
        cache_manager.cache_result("q1", QueryModality.VIDEO, {"v": 1})
        cache_manager.cache_result("q2", QueryModality.DOCUMENT, {"d": 1})

        cache_manager.get_cached_result("q1", QueryModality.VIDEO, 3600)

        all_stats = cache_manager.get_cache_stats()

        assert "video" in all_stats
        assert "document" in all_stats
        assert all_stats["video"]["hits"] == 1
        assert all_stats["document"]["hits"] == 0

    def test_invalidate_modality(self, cache_manager):
        """Test invalidating specific modality cache"""
        cache_manager.cache_result("q1", QueryModality.VIDEO, {"v": 1})
        cache_manager.cache_result("q2", QueryModality.DOCUMENT, {"d": 1})

        cache_manager.invalidate_modality(QueryModality.VIDEO)

        # Video cache should be cleared
        assert cache_manager.get_cached_result("q1", QueryModality.VIDEO, 3600) is None

        # Document cache should still have data
        assert (
            cache_manager.get_cached_result("q2", QueryModality.DOCUMENT, 3600)
            is not None
        )

    def test_invalidate_all(self, cache_manager):
        """Test invalidating all caches"""
        cache_manager.cache_result("q1", QueryModality.VIDEO, {"v": 1})
        cache_manager.cache_result("q2", QueryModality.DOCUMENT, {"d": 1})

        cache_manager.invalidate_all()

        # All caches should be cleared
        assert cache_manager.get_cached_result("q1", QueryModality.VIDEO, 3600) is None
        assert (
            cache_manager.get_cached_result("q2", QueryModality.DOCUMENT, 3600) is None
        )

    def test_lru_eviction_per_modality(self, cache_manager):
        """Test LRU eviction works per modality"""
        # Create cache with small size
        small_cache = ModalityCacheManager(cache_size_per_modality=3)

        # Fill cache for VIDEO modality
        small_cache.cache_result("q1", QueryModality.VIDEO, {"v": 1})
        small_cache.cache_result("q2", QueryModality.VIDEO, {"v": 2})
        small_cache.cache_result("q3", QueryModality.VIDEO, {"v": 3})

        # Add 4th item, should evict oldest
        small_cache.cache_result("q4", QueryModality.VIDEO, {"v": 4})

        # q1 should be evicted
        assert small_cache.get_cached_result("q1", QueryModality.VIDEO, 3600) is None
        assert (
            small_cache.get_cached_result("q4", QueryModality.VIDEO, 3600) is not None
        )

        # Eviction count should be tracked
        stats = small_cache.get_cache_stats(QueryModality.VIDEO)
        assert stats["evictions"] == 1

    def test_reset_stats(self, cache_manager):
        """Test resetting statistics"""
        cache_manager.cache_result("q1", QueryModality.VIDEO, {"v": 1})
        cache_manager.get_cached_result("q1", QueryModality.VIDEO, 3600)

        cache_manager.reset_stats()

        stats = cache_manager.get_cache_stats(QueryModality.VIDEO)
        assert stats["hits"] == 0
        assert stats["misses"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
