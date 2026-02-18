"""
SIMBA (Similarity-Based Memory Augmentation) for Query Enhancement Learning

This module implements SIMBA to improve query enhancement by maintaining
a memory of successful query transformations and using similarity-based
retrieval to enhance new queries based on historical patterns.

Key Features:
- Memory bank of successful query enhancement patterns
- Semantic similarity matching for query enhancement retrieval
- Learning from enhancement outcomes to improve future transformations
- Integration with DSPy 3.0 optimization features
- Adaptive memory management with relevance-based pruning
- Multi-modal similarity (text, entities, relationships)
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# DSPy 3.0 imports
import dspy
import numpy as np
from dspy.teleprompt import SIMBA

# Embedding and similarity imports
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


@dataclass
class QueryEnhancementPattern:
    """Pattern of successful query enhancement"""

    original_query: str
    enhanced_query: str
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    enhancement_strategy: str

    # Outcome metrics
    search_quality_improvement: float  # Improvement in search quality (0-1)
    routing_confidence_improvement: float  # Improvement in routing confidence
    user_satisfaction: Optional[float] = None  # User feedback (0-1)
    success_rate: float = 0.0  # Success rate of this pattern

    # Pattern metadata
    usage_count: int = 1
    avg_improvement: float = 0.0
    pattern_confidence: float = 0.5

    # Embeddings for similarity matching
    query_embedding: Optional[np.ndarray] = None
    entity_embedding: Optional[np.ndarray] = None
    relationship_embedding: Optional[np.ndarray] = None

    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)


@dataclass
class EnhancementMemoryMetrics:
    """Metrics for tracking SIMBA enhancement performance"""

    total_patterns: int
    avg_pattern_quality: float
    successful_enhancements: int
    failed_enhancements: int
    memory_hit_rate: float  # How often patterns are found
    similarity_threshold: float
    improvement_rate: float  # Rate of improvement over time
    pattern_diversity: float  # Diversity of patterns in memory
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class SIMBAConfig:
    """Configuration for SIMBA query enhancement"""

    # Memory parameters
    max_memory_size: int = 5000
    similarity_threshold: float = 0.7
    min_pattern_confidence: float = 0.5
    memory_cleanup_frequency: int = 100  # Cleanup every N operations

    # Enhancement parameters
    enhancement_weight_text: float = 0.6
    enhancement_weight_entities: float = 0.2
    enhancement_weight_relationships: float = 0.2
    min_improvement_threshold: float = 0.1

    # Learning parameters
    learning_rate: float = 0.01
    pattern_decay_factor: float = 0.95  # Decay unused patterns
    diversity_weight: float = 0.3  # Weight for maintaining diversity

    # Embedding model
    embedding_model_name: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    # Storage
    memory_file: str = "simba_memory.pkl"
    metrics_file: str = "simba_metrics.json"
    embedding_cache_file: str = "simba_embeddings.pkl"

    # Minimum patterns before starting SIMBA optimization
    min_patterns_for_optimization: int = 20

    # Trigger optimization every N patterns (after min_patterns_for_optimization is met)
    optimization_trigger_frequency: int = 50


class SIMBAQueryEnhancer:
    """
    SIMBA-based query enhancer that learns from successful enhancement patterns

    This class maintains a memory of successful query enhancements and uses
    similarity-based retrieval to improve new queries based on historical patterns.
    """

    def __init__(
        self,
        config: Optional[SIMBAConfig] = None,
        storage_dir: str = "data/enhancement",
    ):
        """Initialize SIMBA query enhancer"""
        self.config = config or SIMBAConfig()
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Memory storage
        self.enhancement_patterns: List[QueryEnhancementPattern] = []
        self.pattern_index = {}  # Fast lookup by similarity

        # Metrics tracking
        self.metrics = EnhancementMemoryMetrics(
            total_patterns=0,
            avg_pattern_quality=0.0,
            successful_enhancements=0,
            failed_enhancements=0,
            memory_hit_rate=0.0,
            similarity_threshold=self.config.similarity_threshold,
            improvement_rate=0.0,
            pattern_diversity=0.0,
        )

        # SIMBA components
        self.simba_optimizer = None
        self.enhancement_policy = None

        # Embedding model
        self.embedding_model = None
        self.embedding_cache = {}

        # State
        self.operation_count = 0
        self.last_cleanup = datetime.now()

        # Load existing data
        self._load_stored_data()

        # Initialize components
        self._initialize_embedding_model()
        self._initialize_simba_components()

        logger.info(
            f"SIMBA query enhancer initialized with {len(self.enhancement_patterns)} patterns"
        )

    def _initialize_embedding_model(self):
        """Initialize sentence transformer model for embeddings"""
        try:
            if SentenceTransformer is None:
                logger.warning(
                    "sentence-transformers not available, using mock embeddings"
                )
                return

            self.embedding_model = SentenceTransformer(self.config.embedding_model_name)
            logger.info(
                f"Initialized embedding model: {self.config.embedding_model_name}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None

    def _initialize_simba_components(self):
        """Initialize SIMBA optimization components"""
        try:
            # Create enhancement policy signatures
            class QueryEnhancementSignature(dspy.Signature):
                """Enhanced query generation based on learned patterns"""

                original_query = dspy.InputField(desc="Original user query")
                entities = dspy.InputField(desc="Extracted entities")
                relationships = dspy.InputField(desc="Extracted relationships")
                similar_patterns = dspy.InputField(
                    desc="Similar successful enhancement patterns"
                )

                enhanced_query = dspy.OutputField(
                    desc="Enhanced query based on patterns"
                )
                enhancement_strategy = dspy.OutputField(
                    desc="Strategy used for enhancement"
                )
                confidence = dspy.OutputField(desc="Confidence in enhancement (0-1)")

            # Create policy module
            class SIMBAEnhancementPolicy(dspy.Module):
                def __init__(self):
                    super().__init__()
                    self.enhance = dspy.ChainOfThought(QueryEnhancementSignature)

                def forward(
                    self,
                    original_query,
                    entities=None,
                    relationships=None,
                    similar_patterns=None,
                ):
                    entities_str = json.dumps(entities or [], default=str)
                    relationships_str = json.dumps(relationships or [], default=str)
                    patterns_str = json.dumps(similar_patterns or [], default=str)

                    return self.enhance(
                        original_query=original_query,
                        entities=entities_str,
                        relationships=relationships_str,
                        similar_patterns=patterns_str,
                    )

            self.enhancement_policy = SIMBAEnhancementPolicy()

            # Initialize SIMBA optimizer
            if (
                len(self.enhancement_patterns)
                >= self.config.min_patterns_for_optimization
            ):
                self.simba_optimizer = SIMBA()
                logger.info("SIMBA optimizer initialized with sufficient pattern data")
            else:
                logger.info(
                    f"Need {self.config.min_patterns_for_optimization - len(self.enhancement_patterns)} more patterns to start SIMBA optimization"
                )

        except Exception as e:
            logger.error(f"Failed to initialize SIMBA components: {e}")
            self.simba_optimizer = None

    async def enhance_query_with_patterns(
        self,
        original_query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Enhance query using SIMBA pattern matching

        Args:
            original_query: Original user query
            entities: Extracted entities from query
            relationships: Extracted relationships from query
            context: Optional context information

        Returns:
            Enhancement result with enhanced query and metadata
        """
        self.operation_count += 1

        try:
            # Find similar enhancement patterns
            similar_patterns = await self._find_similar_patterns(
                original_query, entities, relationships
            )

            # Apply SIMBA enhancement if patterns found
            if similar_patterns and self.enhancement_policy:
                enhanced_result = await self._apply_simba_enhancement(
                    original_query, entities, relationships, similar_patterns
                )
            else:
                # Fallback to basic enhancement
                enhanced_result = await self._apply_fallback_enhancement(
                    original_query, entities, relationships
                )

            # Update metrics
            if enhanced_result.get("enhanced", False):
                self.metrics.successful_enhancements += 1
            else:
                self.metrics.failed_enhancements += 1

            self._update_metrics()

            # Periodic cleanup
            if self.operation_count % self.config.memory_cleanup_frequency == 0:
                await self._cleanup_memory()

            return enhanced_result

        except Exception as e:
            logger.error(f"SIMBA query enhancement failed: {e}")
            return {
                "enhanced_query": original_query,
                "enhancement_strategy": "error_fallback",
                "confidence": 0.1,
                "enhanced": False,
                "error": str(e),
            }

    async def _find_similar_patterns(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> List[QueryEnhancementPattern]:
        """Find similar enhancement patterns from memory"""
        if not self.enhancement_patterns:
            return []

        try:
            # Get embeddings for current query
            query_embedding = await self._get_query_embedding(query)
            entity_embedding = await self._get_entity_embedding(entities)
            relationship_embedding = await self._get_relationship_embedding(
                relationships
            )

            # Calculate similarities with all patterns
            similarities = []
            for pattern in self.enhancement_patterns:
                if pattern.pattern_confidence < self.config.min_pattern_confidence:
                    continue

                similarity = await self._calculate_pattern_similarity(
                    query_embedding, entity_embedding, relationship_embedding, pattern
                )

                if similarity >= self.config.similarity_threshold:
                    similarities.append((similarity, pattern))

            # Sort by similarity and return top matches
            similarities.sort(key=lambda x: x[0], reverse=True)

            # Update hit rate metric
            hit_rate = len(similarities) / max(1, len(self.enhancement_patterns))
            self.metrics.memory_hit_rate = (
                0.9 * self.metrics.memory_hit_rate + 0.1 * hit_rate
            )

            # Return top similar patterns
            return [pattern for _, pattern in similarities[:5]]  # Top 5 matches

        except Exception as e:
            logger.error(f"Failed to find similar patterns: {e}")
            return []

    async def _calculate_pattern_similarity(
        self,
        query_embedding: np.ndarray,
        entity_embedding: np.ndarray,
        relationship_embedding: np.ndarray,
        pattern: QueryEnhancementPattern,
    ) -> float:
        """Calculate similarity between current query and a pattern"""
        try:
            similarities = []

            # Text similarity
            if query_embedding is not None and pattern.query_embedding is not None:
                text_sim = np.dot(query_embedding, pattern.query_embedding) / (
                    np.linalg.norm(query_embedding)
                    * np.linalg.norm(pattern.query_embedding)
                )
                similarities.append(text_sim * self.config.enhancement_weight_text)

            # Entity similarity
            if entity_embedding is not None and pattern.entity_embedding is not None:
                entity_sim = np.dot(entity_embedding, pattern.entity_embedding) / (
                    np.linalg.norm(entity_embedding)
                    * np.linalg.norm(pattern.entity_embedding)
                )
                similarities.append(
                    entity_sim * self.config.enhancement_weight_entities
                )

            # Relationship similarity
            if (
                relationship_embedding is not None
                and pattern.relationship_embedding is not None
            ):
                rel_sim = np.dot(
                    relationship_embedding, pattern.relationship_embedding
                ) / (
                    np.linalg.norm(relationship_embedding)
                    * np.linalg.norm(pattern.relationship_embedding)
                )
                similarities.append(
                    rel_sim * self.config.enhancement_weight_relationships
                )

            # Combine similarities
            if similarities:
                return sum(similarities) / sum(
                    [
                        self.config.enhancement_weight_text,
                        self.config.enhancement_weight_entities,
                        self.config.enhancement_weight_relationships,
                    ]
                )
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Error calculating pattern similarity: {e}")
            return 0.0

    async def _apply_simba_enhancement(
        self,
        original_query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        similar_patterns: List[QueryEnhancementPattern],
    ) -> Dict[str, Any]:
        """Apply SIMBA-based enhancement using similar patterns"""
        try:
            # Prepare pattern data for DSPy
            pattern_data = []
            for pattern in similar_patterns:
                pattern_info = {
                    "original": pattern.original_query,
                    "enhanced": pattern.enhanced_query,
                    "strategy": pattern.enhancement_strategy,
                    "improvement": pattern.avg_improvement,
                    "confidence": pattern.pattern_confidence,
                }
                pattern_data.append(pattern_info)

            # Apply enhancement policy
            enhancement_result = self.enhancement_policy(
                original_query=original_query,
                entities=entities,
                relationships=relationships,
                similar_patterns=pattern_data,
            )

            enhanced_query = enhancement_result.enhanced_query
            strategy = enhancement_result.enhancement_strategy
            confidence = float(enhancement_result.confidence)

            # Update pattern usage
            for pattern in similar_patterns:
                pattern.usage_count += 1
                pattern.last_used = datetime.now()

            return {
                "enhanced_query": enhanced_query,
                "enhancement_strategy": f"simba_{strategy}",
                "confidence": confidence,
                "enhanced": enhanced_query != original_query,
                "similar_patterns_used": len(similar_patterns),
                "pattern_avg_improvement": np.mean(
                    [p.avg_improvement for p in similar_patterns]
                ),
            }

        except Exception as e:
            logger.error(f"SIMBA enhancement failed: {e}")
            return await self._apply_fallback_enhancement(
                original_query, entities, relationships
            )

    async def _apply_fallback_enhancement(
        self,
        original_query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Apply basic fallback enhancement when no patterns match"""
        # Simple rule-based enhancement as fallback
        enhanced_query = original_query

        # Add entity expansions
        if entities:
            entity_texts = [e.get("text", "") for e in entities[:3]]
            if entity_texts:
                enhanced_query += f" related to {', '.join(entity_texts)}"

        # Add relationship context
        if relationships:
            rel_text = relationships[0]
            if all(key in rel_text for key in ["subject", "relation", "object"]):
                enhanced_query += f" involving {rel_text['subject']} {rel_text['relation']} {rel_text['object']}"

        return {
            "enhanced_query": enhanced_query,
            "enhancement_strategy": "fallback_rule_based",
            "confidence": 0.5,
            "enhanced": enhanced_query != original_query,
            "similar_patterns_used": 0,
        }

    async def record_enhancement_outcome(
        self,
        original_query: str,
        enhanced_query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        enhancement_strategy: str,
        search_quality_improvement: float,
        routing_confidence_improvement: float,
        user_satisfaction: Optional[float] = None,
    ) -> None:
        """
        Record outcome of query enhancement for pattern learning

        Args:
            original_query: Original query
            enhanced_query: Enhanced query
            entities: Entities from query
            relationships: Relationships from query
            enhancement_strategy: Strategy used for enhancement
            search_quality_improvement: Improvement in search quality (0-1)
            routing_confidence_improvement: Improvement in routing confidence
            user_satisfaction: Optional user feedback (0-1)
        """
        try:
            # Only record if there was meaningful improvement
            if search_quality_improvement < self.config.min_improvement_threshold:
                return

            # Generate embeddings
            query_embedding = await self._get_query_embedding(original_query)
            entity_embedding = await self._get_entity_embedding(entities)
            relationship_embedding = await self._get_relationship_embedding(
                relationships
            )

            # Calculate pattern metrics
            avg_improvement = (
                search_quality_improvement + routing_confidence_improvement
            ) / 2
            if user_satisfaction is not None:
                avg_improvement = (avg_improvement + user_satisfaction) / 2

            # Create or update pattern
            pattern = QueryEnhancementPattern(
                original_query=original_query,
                enhanced_query=enhanced_query,
                entities=entities,
                relationships=relationships,
                enhancement_strategy=enhancement_strategy,
                search_quality_improvement=search_quality_improvement,
                routing_confidence_improvement=routing_confidence_improvement,
                user_satisfaction=user_satisfaction,
                avg_improvement=avg_improvement,
                pattern_confidence=min(avg_improvement, 1.0),
                query_embedding=query_embedding,
                entity_embedding=entity_embedding,
                relationship_embedding=relationship_embedding,
            )

            # Check for similar existing patterns
            existing_pattern = await self._find_existing_pattern(pattern)

            if existing_pattern:
                # Update existing pattern
                existing_pattern.usage_count += 1
                existing_pattern.avg_improvement = (
                    0.8 * existing_pattern.avg_improvement + 0.2 * avg_improvement
                )
                existing_pattern.pattern_confidence = min(
                    existing_pattern.avg_improvement, 1.0
                )
                existing_pattern.last_used = datetime.now()
            else:
                # Add new pattern
                self.enhancement_patterns.append(pattern)

                # Maintain memory size limit
                if len(self.enhancement_patterns) > self.config.max_memory_size:
                    await self._prune_memory()

            # Update metrics
            self.metrics.total_patterns = len(self.enhancement_patterns)
            self._update_metrics()

            # Trigger SIMBA optimization if conditions met
            if self._should_trigger_optimization():
                await self._run_simba_optimization()

            logger.debug(
                f"Recorded enhancement pattern: improvement={avg_improvement:.3f}"
            )

        except Exception as e:
            logger.error(f"Failed to record enhancement outcome: {e}")

    async def _find_existing_pattern(
        self, new_pattern: QueryEnhancementPattern
    ) -> Optional[QueryEnhancementPattern]:
        """Find if similar pattern already exists"""
        if new_pattern.query_embedding is None:
            return None

        for existing in self.enhancement_patterns:
            if existing.query_embedding is None:
                continue

            similarity = await self._calculate_pattern_similarity(
                new_pattern.query_embedding,
                new_pattern.entity_embedding,
                new_pattern.relationship_embedding,
                existing,
            )

            if similarity >= 0.9:  # Very high similarity threshold for duplicates
                return existing

        return None

    def _should_trigger_optimization(self) -> bool:
        """Determine if SIMBA optimization should be triggered"""
        if len(self.enhancement_patterns) < self.config.min_patterns_for_optimization:
            return False

        # Trigger every N new patterns (configurable)
        if (
            len(self.enhancement_patterns) % self.config.optimization_trigger_frequency
            == 0
        ):
            return True

        # Trigger if pattern quality is declining
        recent_patterns = self.enhancement_patterns[-10:]
        if len(recent_patterns) >= 10:
            recent_quality = np.mean([p.avg_improvement for p in recent_patterns])
            if recent_quality < self.metrics.avg_pattern_quality - 0.1:
                return True

        return False

    @staticmethod
    def _enhancement_quality_metric(example, pred, trace=None):
        """Metric for SIMBA optimization: evaluates if the predicted enhancement is valid."""
        pred_query = getattr(pred, "enhanced_query", "")
        if not pred_query or not pred_query.strip():
            return 0.0
        # Non-trivial enhancement (not just echoing the original)
        original = getattr(example, "original_query", "")
        if pred_query.strip().lower() == original.strip().lower():
            return 0.5
        return 1.0

    async def _run_simba_optimization(self):
        """Run SIMBA optimization to improve enhancement policy"""
        # Lazy initialize SIMBA optimizer when enough patterns have accumulated
        if self.simba_optimizer is None:
            if (
                len(self.enhancement_patterns)
                >= self.config.min_patterns_for_optimization
            ):
                self.simba_optimizer = SIMBA(
                    metric=self._enhancement_quality_metric,
                )
                logger.info(
                    "Lazy-initialized SIMBA optimizer after reaching "
                    f"{len(self.enhancement_patterns)} patterns"
                )
            else:
                return

        if not self.enhancement_policy:
            return

        try:
            logger.info("Running SIMBA optimization...")

            # Prepare training examples from successful patterns
            training_examples = []
            for pattern in self.enhancement_patterns[-100:]:  # Use recent patterns
                if pattern.avg_improvement >= 0.6:  # Only successful patterns
                    example = dspy.Example(
                        original_query=pattern.original_query,
                        entities=json.dumps(pattern.entities, default=str),
                        relationships=json.dumps(pattern.relationships, default=str),
                        similar_patterns="",  # Will be filled by SIMBA
                        enhanced_query=pattern.enhanced_query,
                        enhancement_strategy=pattern.enhancement_strategy,
                        confidence=str(pattern.pattern_confidence),
                    ).with_inputs(
                        "original_query",
                        "entities",
                        "relationships",
                        "similar_patterns",
                    )

                    training_examples.append(example)

            # Run SIMBA optimization
            if training_examples:
                optimized_policy = self.simba_optimizer.compile(
                    self.enhancement_policy,
                    trainset=training_examples,
                )

                self.enhancement_policy = optimized_policy
                logger.info("SIMBA optimization completed")

        except Exception as e:
            logger.error(f"SIMBA optimization failed: {e}")

    async def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Get embedding for query text"""
        if not self.embedding_model:
            return None

        cache_key = f"query:{hash(query)}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        try:
            embedding = self.embedding_model.encode([query])[0]
            self.embedding_cache[cache_key] = embedding
            return embedding

        except Exception as e:
            logger.error(f"Failed to get query embedding: {e}")
            return None

    async def _get_entity_embedding(
        self, entities: List[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """Get combined embedding for entities"""
        if not self.embedding_model or not entities:
            return None

        entity_texts = [e.get("text", "") for e in entities if e.get("text")]
        if not entity_texts:
            return None

        cache_key = f"entities:{hash(','.join(entity_texts))}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        try:
            embeddings = self.embedding_model.encode(entity_texts)
            combined_embedding = np.mean(embeddings, axis=0)
            self.embedding_cache[cache_key] = combined_embedding
            return combined_embedding

        except Exception as e:
            logger.error(f"Failed to get entity embeddings: {e}")
            return None

    async def _get_relationship_embedding(
        self, relationships: List[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """Get combined embedding for relationships"""
        if not self.embedding_model or not relationships:
            return None

        rel_texts = []
        for rel in relationships:
            if all(key in rel for key in ["subject", "relation", "object"]):
                rel_text = f"{rel['subject']} {rel['relation']} {rel['object']}"
                rel_texts.append(rel_text)

        if not rel_texts:
            return None

        cache_key = f"relationships:{hash(','.join(rel_texts))}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        try:
            embeddings = self.embedding_model.encode(rel_texts)
            combined_embedding = np.mean(embeddings, axis=0)
            self.embedding_cache[cache_key] = combined_embedding
            return combined_embedding

        except Exception as e:
            logger.error(f"Failed to get relationship embeddings: {e}")
            return None

    def _update_metrics(self):
        """Update SIMBA metrics"""
        if self.enhancement_patterns:
            self.metrics.avg_pattern_quality = np.mean(
                [p.avg_improvement for p in self.enhancement_patterns]
            )

            # Calculate pattern diversity (how spread out the patterns are)
            if len(self.enhancement_patterns) > 1:
                strategies = [p.enhancement_strategy for p in self.enhancement_patterns]
                unique_strategies = set(strategies)
                self.metrics.pattern_diversity = len(unique_strategies) / len(
                    self.enhancement_patterns
                )

        self.metrics.last_updated = datetime.now()

    async def _cleanup_memory(self):
        """Clean up memory by removing low-quality patterns"""
        if len(self.enhancement_patterns) <= self.config.max_memory_size // 2:
            return

        try:
            # Remove patterns with low confidence or old unused patterns
            cutoff_date = datetime.now() - timedelta(days=30)

            filtered_patterns = []
            for pattern in self.enhancement_patterns:
                # Keep pattern if it's high quality or recently used
                if (
                    pattern.pattern_confidence >= self.config.min_pattern_confidence
                    or pattern.last_used > cutoff_date
                    or pattern.usage_count > 5
                ):
                    filtered_patterns.append(pattern)

            removed_count = len(self.enhancement_patterns) - len(filtered_patterns)
            self.enhancement_patterns = filtered_patterns

            if removed_count > 0:
                logger.info(
                    f"Cleaned up {removed_count} low-quality patterns from memory"
                )

        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")

    async def _prune_memory(self):
        """Prune memory to stay within size limits"""
        if len(self.enhancement_patterns) <= self.config.max_memory_size:
            return

        try:
            # Sort by pattern quality and recency
            def pattern_score(pattern):
                recency_bonus = (
                    1.0
                    if pattern.last_used > datetime.now() - timedelta(days=7)
                    else 0.5
                )
                usage_bonus = min(pattern.usage_count / 10.0, 0.5)
                return pattern.pattern_confidence + recency_bonus + usage_bonus

            self.enhancement_patterns.sort(key=pattern_score, reverse=True)

            # Keep only the best patterns
            removed_count = len(self.enhancement_patterns) - self.config.max_memory_size
            self.enhancement_patterns = self.enhancement_patterns[
                : self.config.max_memory_size
            ]

            logger.info(f"Pruned {removed_count} patterns from memory")

        except Exception as e:
            logger.error(f"Memory pruning failed: {e}")

    async def _persist_data(self):
        """Persist enhancement patterns and metrics"""
        try:
            # Save patterns (without embeddings to save space)
            patterns_for_storage = []
            for pattern in self.enhancement_patterns:
                storage_pattern = pattern.__dict__.copy()
                # Remove heavy embedding data
                storage_pattern.pop("query_embedding", None)
                storage_pattern.pop("entity_embedding", None)
                storage_pattern.pop("relationship_embedding", None)
                patterns_for_storage.append(storage_pattern)

            memory_file = self.storage_dir / self.config.memory_file
            with open(memory_file, "wb") as f:
                pickle.dump(patterns_for_storage, f)

            # Save embeddings separately
            embedding_cache_file = self.storage_dir / self.config.embedding_cache_file
            with open(embedding_cache_file, "wb") as f:
                pickle.dump(self.embedding_cache, f)

            # Save metrics
            metrics_file = self.storage_dir / self.config.metrics_file
            metrics_dict = {
                "total_patterns": self.metrics.total_patterns,
                "avg_pattern_quality": self.metrics.avg_pattern_quality,
                "successful_enhancements": self.metrics.successful_enhancements,
                "failed_enhancements": self.metrics.failed_enhancements,
                "memory_hit_rate": self.metrics.memory_hit_rate,
                "similarity_threshold": self.metrics.similarity_threshold,
                "improvement_rate": self.metrics.improvement_rate,
                "pattern_diversity": self.metrics.pattern_diversity,
                "last_updated": self.metrics.last_updated.isoformat(),
            }

            with open(metrics_file, "w") as f:
                json.dump(metrics_dict, f, indent=2)

            logger.debug(
                f"Persisted {len(self.enhancement_patterns)} patterns and metrics"
            )

        except Exception as e:
            logger.error(f"Failed to persist SIMBA data: {e}")

    def _load_stored_data(self):
        """Load previously stored patterns and metrics"""
        try:
            # Load patterns
            memory_file = self.storage_dir / self.config.memory_file
            if memory_file.exists():
                with open(memory_file, "rb") as f:
                    stored_patterns = pickle.load(f)

                # Convert stored patterns back to objects
                for stored in stored_patterns:
                    pattern = QueryEnhancementPattern(**stored)
                    self.enhancement_patterns.append(pattern)

                logger.info(
                    f"Loaded {len(self.enhancement_patterns)} enhancement patterns"
                )

            # Load embedding cache
            cache_file = self.storage_dir / self.config.embedding_cache_file
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")

            # Load metrics
            metrics_file = self.storage_dir / self.config.metrics_file
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics_dict = json.load(f)

                self.metrics.total_patterns = metrics_dict.get("total_patterns", 0)
                self.metrics.avg_pattern_quality = metrics_dict.get(
                    "avg_pattern_quality", 0.0
                )
                self.metrics.successful_enhancements = metrics_dict.get(
                    "successful_enhancements", 0
                )
                self.metrics.failed_enhancements = metrics_dict.get(
                    "failed_enhancements", 0
                )
                self.metrics.memory_hit_rate = metrics_dict.get("memory_hit_rate", 0.0)
                self.metrics.similarity_threshold = metrics_dict.get(
                    "similarity_threshold", self.config.similarity_threshold
                )
                self.metrics.improvement_rate = metrics_dict.get(
                    "improvement_rate", 0.0
                )
                self.metrics.pattern_diversity = metrics_dict.get(
                    "pattern_diversity", 0.0
                )

                if "last_updated" in metrics_dict:
                    self.metrics.last_updated = datetime.fromisoformat(
                        metrics_dict["last_updated"]
                    )

                logger.info("Loaded SIMBA metrics")

        except Exception as e:
            logger.error(f"Failed to load stored SIMBA data: {e}")

    def get_enhancement_status(self) -> Dict[str, Any]:
        """Get current SIMBA enhancement status and metrics"""
        return {
            "simba_enabled": self.simba_optimizer is not None,
            "total_patterns": len(self.enhancement_patterns),
            "embedding_model_available": self.embedding_model is not None,
            "metrics": {
                "total_patterns": self.metrics.total_patterns,
                "avg_pattern_quality": round(self.metrics.avg_pattern_quality, 3),
                "successful_enhancements": self.metrics.successful_enhancements,
                "failed_enhancements": self.metrics.failed_enhancements,
                "success_rate": round(
                    self.metrics.successful_enhancements
                    / max(
                        1,
                        self.metrics.successful_enhancements
                        + self.metrics.failed_enhancements,
                    ),
                    3,
                ),
                "memory_hit_rate": round(self.metrics.memory_hit_rate, 3),
                "pattern_diversity": round(self.metrics.pattern_diversity, 3),
                "improvement_rate": round(self.metrics.improvement_rate, 3),
                "last_updated": self.metrics.last_updated.isoformat(),
            },
            "config": {
                "max_memory_size": self.config.max_memory_size,
                "similarity_threshold": self.config.similarity_threshold,
                "min_pattern_confidence": self.config.min_pattern_confidence,
                "embedding_model": self.config.embedding_model_name,
            },
            "cache_size": len(self.embedding_cache),
        }

    async def reset_memory(self):
        """Reset SIMBA memory (useful for testing or fresh start)"""
        logger.warning("Resetting SIMBA enhancement memory...")

        self.enhancement_patterns.clear()
        self.embedding_cache.clear()
        self.operation_count = 0

        self.metrics = EnhancementMemoryMetrics(
            total_patterns=0,
            avg_pattern_quality=0.0,
            successful_enhancements=0,
            failed_enhancements=0,
            memory_hit_rate=0.0,
            similarity_threshold=self.config.similarity_threshold,
            improvement_rate=0.0,
            pattern_diversity=0.0,
        )

        # Clear stored files
        try:
            for filename in [
                self.config.memory_file,
                self.config.metrics_file,
                self.config.embedding_cache_file,
            ]:
                file_path = self.storage_dir / filename
                if file_path.exists():
                    file_path.unlink()

        except Exception as e:
            logger.error(f"Failed to clear stored files: {e}")

        # Re-initialize components
        self._initialize_simba_components()

        logger.info("SIMBA memory reset complete")
