"""
Schema-aware reranking strategies for evaluation results.

This module provides reranking strategies that adapt to any schema type
and can properly calculate similarity between results.
"""

import logging
import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any

from cogniverse_core.evaluation.core.schema_analyzer import get_schema_analyzer

logger = logging.getLogger(__name__)


class RerankingError(Exception):
    """Base exception for reranking errors."""

    pass


class RerankingStrategy(ABC):
    """Abstract base class for reranking strategies."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rerank search results.

        Args:
            query: The search query
            results: List of search results to rerank
            config: Optional configuration including schema info

        Returns:
            Reranked list of results
        """
        pass


class DiversityRerankingStrategy(RerankingStrategy):
    """Rerank to maximize diversity while maintaining relevance."""

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank for diversity using MMR (Maximal Marginal Relevance)."""
        if not results:
            return results

        config = config or {}
        lambda_param = config.get(
            "diversity_lambda", 0.5
        )  # Balance between relevance and diversity

        # Get schema analyzer for ID extraction
        schema_name = config.get("schema_name", "unknown")
        schema_fields = config.get("schema_fields", {})

        try:
            analyzer = get_schema_analyzer(schema_name, schema_fields)
        except Exception as e:
            logger.warning(f"Using default analyzer: {e}")
            from cogniverse_core.evaluation.core.schema_analyzer import (
                DefaultSchemaAnalyzer,
            )

            analyzer = DefaultSchemaAnalyzer()

        # Calculate MMR scores
        reranked = []
        remaining = results.copy()

        while remaining:
            if not reranked:
                # First item: pick the most relevant
                best_idx = 0
                best_score = remaining[0].get("score", 0.0)
                for i, r in enumerate(remaining):
                    if r.get("score", 0.0) > best_score:
                        best_idx = i
                        best_score = r["score"]
            else:
                # Subsequent items: balance relevance and diversity
                best_idx = -1
                best_mmr = float("-inf")

                for i, candidate in enumerate(remaining):
                    relevance = candidate.get("score", 0.0)

                    # Calculate maximum similarity to already selected items
                    max_sim = 0.0
                    for selected in reranked:
                        sim = self._calculate_similarity(candidate, selected, analyzer)
                        max_sim = max(max_sim, sim)

                    # MMR score
                    mmr = lambda_param * relevance - (1 - lambda_param) * max_sim

                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_idx = i

            if best_idx >= 0:
                selected = remaining.pop(best_idx)
                selected["diversity_rank"] = len(reranked) + 1
                selected["mmr_score"] = (
                    best_mmr if reranked else selected.get("score", 0.0)
                )
                reranked.append(selected)

        logger.info(f"Reranked {len(results)} results for diversity")
        return reranked

    def _calculate_similarity(
        self, result1: dict, result2: dict, analyzer: Any
    ) -> float:
        """Calculate similarity between two results."""
        # First check if they're the same item
        try:
            id1 = analyzer.extract_item_id(result1)
            id2 = analyzer.extract_item_id(result2)

            if id1 and id2 and id1 == id2:
                return 1.0  # Same item
        except Exception as e:
            logger.debug(f"Failed to extract IDs for similarity: {e}")

        # Calculate content similarity
        content1 = self._extract_content(result1)
        content2 = self._extract_content(result2)

        if content1 and content2:
            return self._calculate_text_similarity(content1, content2)

        # If we can't calculate similarity, assume they're different
        return 0.0

    def _extract_content(self, result: dict) -> str:
        """Extract text content from result."""
        content_fields = [
            "content",
            "text",
            "description",
            "summary",
            "title",
            "caption",
        ]
        contents = []

        for field in content_fields:
            if field in result and result[field]:
                contents.append(str(result[field]))
            elif "metadata" in result and field in result["metadata"]:
                contents.append(str(result["metadata"][field]))

        return " ".join(contents)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using TF-IDF cosine similarity."""
        if not text1 or not text2:
            return 0.0

        # Tokenize
        words1 = text1.lower().split()
        words2 = text2.lower().split()

        if not words1 or not words2:
            return 0.0

        # Create vocabulary
        vocab = list(set(words1 + words2))

        # Calculate TF vectors
        tf1 = self._calculate_tf(words1, vocab)
        tf2 = self._calculate_tf(words2, vocab)

        # Calculate IDF
        idf = self._calculate_idf([words1, words2], vocab)

        # Calculate TF-IDF vectors
        tfidf1 = [tf1[i] * idf[i] for i in range(len(vocab))]
        tfidf2 = [tf2[i] * idf[i] for i in range(len(vocab))]

        # Calculate cosine similarity
        return self._cosine_similarity(tfidf1, tfidf2)

    def _calculate_tf(self, words: list[str], vocab: list[str]) -> list[float]:
        """Calculate term frequency vector."""
        word_count = Counter(words)
        total = len(words)
        return [word_count.get(term, 0) / total for term in vocab]

    def _calculate_idf(
        self, documents: list[list[str]], vocab: list[str]
    ) -> list[float]:
        """Calculate inverse document frequency."""
        n_docs = len(documents)
        idf_scores = []

        for term in vocab:
            containing_docs = sum(1 for doc in documents if term in doc)
            if containing_docs > 0:
                idf = math.log(n_docs / containing_docs)
            else:
                idf = 0
            idf_scores.append(idf)

        return idf_scores

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


class ContentSimilarityRerankingStrategy(RerankingStrategy):
    """Rerank based on content similarity to query using multiple signals."""

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank by content similarity using TF-IDF and semantic matching."""
        if not results:
            return results

        config = config or {}

        # Get content fields from schema
        schema_fields = config.get("schema_fields", {})
        content_fields = schema_fields.get("content_fields", [])
        text_fields = schema_fields.get("text_fields", [])

        # Combine content and text fields
        searchable_fields = content_fields + text_fields

        if not searchable_fields:
            # Use common field names
            searchable_fields = [
                "content",
                "text",
                "description",
                "summary",
                "title",
                "caption",
            ]

        # Calculate similarity scores using multiple methods
        scored_results = []
        for result in results:
            # Extract content from available fields
            content = self._extract_content(result, searchable_fields)

            # Calculate different similarity scores
            keyword_score = self._calculate_keyword_similarity(query, content)
            tfidf_score = self._calculate_tfidf_similarity(query, content)
            semantic_score = self._calculate_semantic_similarity(query, content, result)

            # Combine scores with weights
            weights = config.get(
                "similarity_weights", {"keyword": 0.2, "tfidf": 0.5, "semantic": 0.3}
            )

            combined_score = (
                weights.get("keyword", 0.2) * keyword_score
                + weights.get("tfidf", 0.5) * tfidf_score
                + weights.get("semantic", 0.3) * semantic_score
            )

            result_copy = result.copy()
            result_copy["content_similarity_score"] = combined_score
            result_copy["similarity_components"] = {
                "keyword": keyword_score,
                "tfidf": tfidf_score,
                "semantic": semantic_score,
            }
            scored_results.append(result_copy)

        # Sort by combined similarity score
        reranked = sorted(
            scored_results, key=lambda x: x["content_similarity_score"], reverse=True
        )

        logger.info(f"Reranked {len(results)} results by content similarity")
        return reranked

    def _extract_content(self, result: dict, fields: list[str]) -> str:
        """Extract content from result using available fields."""
        contents = []

        for field in fields:
            # Try direct field
            if field in result and result[field]:
                contents.append(str(result[field]))

            # Try nested in metadata
            if "metadata" in result and field in result["metadata"]:
                contents.append(str(result["metadata"][field]))

        return " ".join(contents)

    def _calculate_keyword_similarity(self, query: str, content: str) -> float:
        """Calculate simple keyword overlap similarity."""
        if not content:
            return 0.0

        query_terms = set(query.lower().split())
        content_terms = set(content.lower().split())

        if not query_terms or not content_terms:
            return 0.0

        # Weighted Jaccard similarity
        intersection = query_terms & content_terms
        union = query_terms | content_terms

        # Give more weight to query terms that appear
        query_coverage = len(intersection) / len(query_terms)
        jaccard = len(intersection) / len(union) if union else 0

        # Combine with higher weight for query coverage
        return 0.7 * query_coverage + 0.3 * jaccard

    def _calculate_tfidf_similarity(self, query: str, content: str) -> float:
        """Calculate TF-IDF cosine similarity."""
        if not content:
            return 0.0

        # Tokenize
        query_words = query.lower().split()
        content_words = content.lower().split()

        if not query_words or not content_words:
            return 0.0

        # Create vocabulary
        vocab = list(set(query_words + content_words))

        # Calculate TF vectors
        query_tf = Counter(query_words)
        content_tf = Counter(content_words)

        # Simple IDF (using document frequency of 2)
        query_vec = []
        content_vec = []

        for term in vocab:
            # TF
            q_tf = query_tf.get(term, 0) / len(query_words)
            c_tf = content_tf.get(term, 0) / len(content_words)

            # IDF (simplified)
            docs_with_term = (1 if term in query_words else 0) + (
                1 if term in content_words else 0
            )
            idf = math.log(2 / docs_with_term) if docs_with_term > 0 else 0

            query_vec.append(q_tf * idf)
            content_vec.append(c_tf * idf)

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(query_vec, content_vec, strict=False))
        q_magnitude = math.sqrt(sum(a * a for a in query_vec))
        c_magnitude = math.sqrt(sum(b * b for b in content_vec))

        if q_magnitude == 0 or c_magnitude == 0:
            return 0.0

        return dot_product / (q_magnitude * c_magnitude)

    def _calculate_semantic_similarity(
        self, query: str, content: str, result: dict
    ) -> float:
        """Calculate semantic similarity using embeddings if available."""
        # Check if result has embeddings
        if "embedding" in result or "embeddings" in result:
            # If we had query embeddings, we could calculate cosine similarity
            # For now, use the original search score as a proxy for semantic similarity
            return min(1.0, result.get("score", 0.0))

        # Fallback to enhanced keyword matching with synonyms and related terms
        return self._calculate_enhanced_keyword_similarity(query, content)

    def _calculate_enhanced_keyword_similarity(self, query: str, content: str) -> float:
        """Enhanced keyword similarity with related terms."""
        if not content:
            return 0.0

        query_lower = query.lower()
        content_lower = content.lower()

        # Check for exact phrase match
        if query_lower in content_lower:
            return 1.0

        # Check for partial phrase matches
        query_words = query_lower.split()
        if len(query_words) > 1:
            # Check for bigrams
            for i in range(len(query_words) - 1):
                bigram = f"{query_words[i]} {query_words[i+1]}"
                if bigram in content_lower:
                    return 0.8

        # Word stem matching (simple version)
        query_stems = {self._simple_stem(word) for word in query_words}
        content_stems = {self._simple_stem(word) for word in content_lower.split()}

        if query_stems and content_stems:
            overlap = len(query_stems & content_stems)
            return overlap / len(query_stems)

        return 0.0

    def _simple_stem(self, word: str) -> str:
        """Simple stemming by removing common suffixes."""
        suffixes = ["ing", "ed", "s", "er", "est", "ly", "tion", "ment"]
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[: -len(suffix)]
        return word


class TemporalRerankingStrategy(RerankingStrategy):
    """Rerank based on temporal relevance and coherence."""

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank by temporal relevance."""
        if not results:
            return results

        config = config or {}
        schema_fields = config.get("schema_fields", {})
        temporal_fields = schema_fields.get("temporal_fields", [])

        if not temporal_fields:
            # No temporal fields, return as is
            logger.debug("No temporal fields found, skipping temporal reranking")
            return results

        # Detect temporal intent in query
        temporal_intent = self._detect_temporal_intent(query)

        if not temporal_intent:
            # No temporal intent, return as is
            return results

        # Extract temporal values and rerank
        scored_results = []
        for result in results:
            temporal_value = self._extract_temporal_value(result, temporal_fields)

            if temporal_value is not None:
                # Score based on temporal intent
                temporal_score = self._score_temporal_relevance(
                    temporal_value, temporal_intent, config
                )
            else:
                temporal_score = 0.0

            result_copy = result.copy()
            result_copy["temporal_score"] = temporal_score
            result_copy["temporal_value"] = temporal_value
            scored_results.append(result_copy)

        # Sort based on temporal intent
        if temporal_intent["type"] in ["latest", "recent", "newest"]:
            # Sort by most recent first
            reranked = sorted(
                scored_results,
                key=lambda x: (x["temporal_score"], x.get("temporal_value", 0)),
                reverse=True,
            )
        elif temporal_intent["type"] in ["oldest", "earliest", "first"]:
            # Sort by oldest first
            reranked = sorted(
                scored_results,
                key=lambda x: (
                    x["temporal_score"],
                    -x.get("temporal_value", float("inf")),
                ),
            )
        else:
            # Sort by temporal relevance score
            reranked = sorted(
                scored_results, key=lambda x: x["temporal_score"], reverse=True
            )

        logger.info(f"Reranked {len(results)} results by temporal relevance")
        return reranked

    def _detect_temporal_intent(self, query: str) -> dict[str, Any] | None:
        """Detect temporal intent in query."""
        query_lower = query.lower()

        # Patterns for temporal intent
        patterns = {
            "latest": ["latest", "most recent", "newest", "last"],
            "oldest": ["oldest", "earliest", "first"],
            "after": ["after", "since", "from"],
            "before": ["before", "until", "prior to"],
            "during": ["during", "in", "within"],
            "between": ["between", "from.*to"],
        }

        for intent_type, keywords in patterns.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return {"type": intent_type, "keyword": keyword, "query": query}

        # Check for specific time references
        import re

        # Year pattern
        year_match = re.search(r"\b(19|20)\d{2}\b", query_lower)
        if year_match:
            return {
                "type": "specific_year",
                "year": int(year_match.group()),
                "query": query,
            }

        # Month pattern
        months = [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]
        for month in months:
            if month in query_lower:
                return {"type": "specific_month", "month": month, "query": query}

        return None

    def _extract_temporal_value(
        self, result: dict, temporal_fields: list[str]
    ) -> float | None:
        """Extract temporal value from result."""
        for field in temporal_fields:
            value = None

            # Try direct field
            if field in result:
                value = result[field]
            elif "metadata" in result and field in result["metadata"]:
                value = result["metadata"][field]

            if value is not None:
                # Convert to numeric timestamp if possible
                if isinstance(value, int | float):
                    return float(value)
                elif isinstance(value, str):
                    # Try to parse as timestamp
                    try:
                        # Simple numeric parsing
                        return float(value)
                    except Exception:
                        # Could add date parsing here
                        pass

        return None

    def _score_temporal_relevance(
        self,
        temporal_value: float,
        temporal_intent: dict[str, Any],
        config: dict[str, Any],
    ) -> float:
        """Score temporal relevance based on intent."""
        intent_type = temporal_intent["type"]

        if intent_type in ["latest", "recent", "newest"]:
            # Higher score for more recent values
            # Normalize to 0-1 range (assuming timestamp-like values)
            max_time = config.get("max_timestamp", temporal_value + 1)
            min_time = config.get("min_timestamp", 0)

            if max_time > min_time:
                return (temporal_value - min_time) / (max_time - min_time)
            # If no range, score based on absolute recency (assuming higher values are more recent)
            # Normalize using sigmoid-like function
            import math

            normalized = 1.0 / (
                1.0 + math.exp(-temporal_value / 1000000)
            )  # Scale by typical timestamp range
            return normalized

        elif intent_type in ["oldest", "earliest", "first"]:
            # Higher score for older values
            max_time = config.get("max_timestamp", temporal_value + 1)
            min_time = config.get("min_timestamp", 0)

            if max_time > min_time:
                return 1.0 - (temporal_value - min_time) / (max_time - min_time)
            # If no range, score based on absolute age (lower values = older = higher score)
            # Use inverse sigmoid for scoring
            import math

            normalized = 1.0 / (
                1.0 + math.exp(temporal_value / 1000000)
            )  # Inverse of recency
            return normalized

        elif intent_type == "specific_year":
            # Score based on proximity to specified year
            target_year = temporal_intent["year"]

            # Convert temporal value to year if possible
            # Assuming temporal_value might be a timestamp or year
            try:
                if temporal_value > 10000:  # Likely a timestamp
                    from datetime import datetime

                    value_year = datetime.fromtimestamp(temporal_value).year
                else:
                    value_year = int(temporal_value)

                # Calculate proximity score (closer years get higher scores)
                year_diff = abs(value_year - target_year)
                if year_diff == 0:
                    return 1.0
                elif year_diff <= 1:
                    return 0.8
                elif year_diff <= 5:
                    # Gradual decay based on year difference
                    return (
                        0.6 - (year_diff - 2) * 0.1
                    )  # 0.6 for 2 years, 0.5 for 3, 0.4 for 4, 0.3 for 5
                else:
                    # Exponential decay for larger differences
                    return max(0.1, 1.0 / (1 + year_diff * 0.1))
            except (ValueError, OverflowError):
                return 0.3  # Can't parse year, give low score

        else:
            # Default scoring based on general temporal relevance
            # Use relative position in time range if available
            if "temporal_range" in config:
                time_range = config["temporal_range"]
                if len(time_range) == 2:
                    min_t, max_t = time_range
                    if max_t > min_t:
                        # Return normalized position in range
                        position = (temporal_value - min_t) / (max_t - min_t)
                        return max(0.0, min(1.0, position))

            # For unknown intent types, score based on recency with moderate weighting
            # More recent items get slightly higher scores
            import math

            # Use sigmoid function centered around median expected timestamp
            median_timestamp = config.get(
                "median_timestamp", 1700000000
            )  # Unix timestamp around 2023
            time_diff = abs(temporal_value - median_timestamp)
            # Decay function: closer to median = higher score
            score = 1.0 / (1.0 + time_diff / 86400000)  # Normalize by ~1000 days
            return max(0.1, min(0.9, score))  # Clamp between 0.1 and 0.9


class HybridRerankingStrategy(RerankingStrategy):
    """Combine multiple reranking strategies."""

    def __init__(self, strategies: list[RerankingStrategy] | None = None):
        """Initialize with list of strategies to combine."""
        self.strategies = strategies or [
            ContentSimilarityRerankingStrategy(),
            DiversityRerankingStrategy(),
            TemporalRerankingStrategy(),
        ]

    async def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Apply multiple reranking strategies and combine scores."""
        if not results:
            return results

        config = config or {}
        strategy_weights = config.get(
            "strategy_weights", {"content": 0.5, "diversity": 0.3, "temporal": 0.2}
        )

        # Apply each strategy
        strategy_results = {}
        for strategy in self.strategies:
            try:
                reranked = await strategy.rerank(query, results, config)
                strategy_name = strategy.__class__.__name__.replace(
                    "RerankingStrategy", ""
                ).lower()
                strategy_results[strategy_name] = reranked
            except Exception as e:
                logger.warning(f"Strategy {strategy.__class__.__name__} failed: {e}")

        if not strategy_results:
            return results

        # Combine scores from different strategies
        combined_results = []
        for _i, original_result in enumerate(results):
            combined = original_result.copy()
            combined_score = 0.0
            score_components = {}

            for strategy_name, reranked_list in strategy_results.items():
                # Find this result in the reranked list
                for j, reranked_result in enumerate(reranked_list):
                    # Match by original position or ID
                    if self._results_match(original_result, reranked_result):
                        # Calculate position-based score (higher position = higher score)
                        position_score = 1.0 - (j / len(reranked_list))
                        weight = strategy_weights.get(strategy_name, 0.33)
                        combined_score += weight * position_score
                        score_components[strategy_name] = position_score
                        break

            combined["hybrid_score"] = combined_score
            combined["score_components"] = score_components
            combined_results.append(combined)

        # Sort by combined score
        reranked = sorted(
            combined_results, key=lambda x: x["hybrid_score"], reverse=True
        )

        logger.info(
            f"Hybrid reranked {len(results)} results using {len(strategy_results)} strategies"
        )
        return reranked

    def _results_match(self, result1: dict, result2: dict) -> bool:
        """Check if two results are the same."""
        # Try to match by ID fields
        id_fields = ["id", "doc_id", "document_id", "item_id", "_id"]

        for field in id_fields:
            if field in result1 and field in result2:
                if result1[field] == result2[field]:
                    return True

        # Fallback to content comparison
        content_fields = ["content", "text", "title"]
        for field in content_fields:
            if field in result1 and field in result2:
                if result1[field] == result2[field]:
                    return True

        # Last resort - compare entire dictionaries
        return result1 == result2
