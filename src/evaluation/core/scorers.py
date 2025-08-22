"""
Schema-driven evaluation scorers that work with any data type.

These scorers use the schema analyzer to understand the data structure
and extract relevant fields without hardcoded assumptions.
"""

import logging
from typing import Any

from inspect_ai.scorer import Score, mean, scorer

from src.evaluation.core.schema_analyzer import get_schema_analyzer

logger = logging.getLogger(__name__)


def get_configured_scorers(config: dict[str, Any]) -> list:
    """Get list of scorers based on configuration.

    Args:
        config: Configuration dictionary with scorer settings

    Returns:
        List of configured scorers

    Raises:
        ValueError: If config is None or invalid
    """
    if config is None:
        raise ValueError("config is required for scorer configuration")

    scorers = []

    # Add scorers based on config
    if config.get("use_relevance", True):
        scorers.append(relevance_scorer())

    if config.get("use_diversity", True):
        scorers.append(diversity_scorer())

    if config.get("use_temporal", False):
        scorers.append(schema_aware_temporal_scorer())

    if config.get("use_precision_recall", False):
        scorers.extend([precision_scorer(), recall_scorer()])

    # Add visual/LLM evaluators from plugin if configured
    if config.get("enable_llm_evaluators", False) or config.get(
        "enable_quality_evaluators", False
    ):
        try:
            from src.evaluation.plugins.visual_evaluator import get_visual_scorers

            visual_scorers = get_visual_scorers(config)
            scorers.extend(visual_scorers)
        except ImportError:
            logger.warning("Visual evaluator plugin not available")

    if not scorers:
        logger.warning("No scorers configured, using default relevance scorer")
        scorers.append(relevance_scorer())

    return scorers


@scorer(metrics=[mean()])
def relevance_scorer():
    """
    Schema-agnostic relevance scorer using keyword matching.
    Works with any schema by extracting text content from results.
    """

    async def score(state, target=None) -> Score:
        query = state.input.get("query", "")

        if not query:
            return Score(value=0.0, explanation="No query provided", metadata={})

        all_scores = {}

        for config_key, output in state.outputs.items():
            if not output.get("success", False):
                all_scores[config_key] = 0.0
                continue

            results = output.get("results", [])
            if not results:
                all_scores[config_key] = 0.0
                continue

            # Extract text content from results
            contexts = []
            for r in results:
                # Try multiple content fields
                content = (
                    r.get("content")
                    or r.get("text")
                    or r.get("description")
                    or r.get("transcript")
                    or r.get("caption")
                    or ""
                )
                if content:
                    contexts.append(str(content))

            # Calculate relevance
            if contexts:
                relevancy = _calculate_keyword_relevance(query, contexts)
                all_scores[config_key] = relevancy
            else:
                all_scores[config_key] = 0.0

        avg_score = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0

        explanation = "Relevance scores: " + ", ".join(
            f"{k}={v:.3f}" for k, v in all_scores.items()
        )

        return Score(
            value=avg_score,
            explanation=explanation,
            metadata={"individual_scores": all_scores},
        )

    return score


@scorer(metrics=[mean()])
def precision_scorer():
    """
    Precision scorer that requires ground truth and uses schema analyzer.

    Precision = |retrieved ∩ relevant| / |retrieved|
    """

    async def score(state, target=None) -> Score:
        # Check for ground truth in state.output
        if not hasattr(state, "output") or not state.output:
            return Score(
                value=0.0,
                explanation="No ground truth available for precision calculation",
                metadata={},
            )

        # Get expected items - must be explicit, no fallbacks
        expected_items = state.output.get("expected_items")
        if expected_items is None:
            return Score(
                value=0.0,
                explanation="No expected_items field in ground truth",
                metadata={},
            )

        expected_set = set(expected_items) if expected_items else set()

        # Get schema info from metadata
        metadata = getattr(state, "metadata", {})
        schema_name = metadata.get("schema_name")
        schema_fields = metadata.get("schema_fields", {})

        if not schema_name:
            return Score(
                value=0.0, explanation="No schema information available", metadata={}
            )

        # Get schema analyzer
        try:
            analyzer = get_schema_analyzer(schema_name, schema_fields)
        except Exception as e:
            return Score(
                value=0.0,
                explanation=f"Failed to get schema analyzer: {e}",
                metadata={},
            )

        all_scores = {}

        for config_key, output in state.outputs.items():
            if not output.get("success", False):
                all_scores[config_key] = 0.0
                continue

            results = output.get("results", [])
            if not results:
                all_scores[config_key] = (
                    0.0 if expected_set else 1.0
                )  # Perfect if both empty
                continue

            # Extract item IDs using schema analyzer
            retrieved_ids = set()
            for r in results:
                try:
                    item_id = analyzer.extract_item_id(r)
                    if item_id:
                        retrieved_ids.add(item_id)
                except Exception as e:
                    logger.warning(f"Failed to extract item ID: {e}")

            # Calculate precision
            if retrieved_ids:
                relevant_retrieved = retrieved_ids.intersection(expected_set)
                precision = len(relevant_retrieved) / len(retrieved_ids)
                all_scores[config_key] = precision
            else:
                all_scores[config_key] = 0.0

        avg_score = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0

        explanation = f"Precision scores (schema: {schema_name}): " + ", ".join(
            f"{k}={v:.3f}" for k, v in all_scores.items()
        )

        return Score(
            value=avg_score,
            explanation=explanation,
            metadata={
                "individual_scores": all_scores,
                "schema": schema_name,
                "expected_count": len(expected_set),
            },
        )

    return score


@scorer(metrics=[mean()])
def recall_scorer():
    """
    Recall scorer that requires ground truth and uses schema analyzer.

    Recall = |retrieved ∩ relevant| / |relevant|
    """

    async def score(state, target=None) -> Score:
        # Check for ground truth in state.output
        if not hasattr(state, "output") or not state.output:
            return Score(
                value=0.0,
                explanation="No ground truth available for recall calculation",
                metadata={},
            )

        # Get expected items - must be explicit, no fallbacks
        expected_items = state.output.get("expected_items")
        if expected_items is None:
            return Score(
                value=0.0,
                explanation="No expected_items field in ground truth",
                metadata={},
            )

        expected_set = set(expected_items) if expected_items else set()

        # Early return if no expected items
        if not expected_set:
            return Score(
                value=1.0,  # Perfect recall if nothing expected
                explanation="No items expected",
                metadata={},
            )

        # Get schema info from metadata
        metadata = getattr(state, "metadata", {})
        schema_name = metadata.get("schema_name")
        schema_fields = metadata.get("schema_fields", {})

        if not schema_name:
            return Score(
                value=0.0, explanation="No schema information available", metadata={}
            )

        # Get schema analyzer
        try:
            analyzer = get_schema_analyzer(schema_name, schema_fields)
        except Exception as e:
            return Score(
                value=0.0,
                explanation=f"Failed to get schema analyzer: {e}",
                metadata={},
            )

        all_scores = {}

        for config_key, output in state.outputs.items():
            if not output.get("success", False):
                all_scores[config_key] = 0.0
                continue

            results = output.get("results", [])

            # Extract item IDs using schema analyzer
            retrieved_ids = set()
            for r in results:
                try:
                    item_id = analyzer.extract_item_id(r)
                    if item_id:
                        retrieved_ids.add(item_id)
                except Exception as e:
                    logger.warning(f"Failed to extract item ID: {e}")

            # Calculate recall
            relevant_retrieved = retrieved_ids.intersection(expected_set)
            recall = len(relevant_retrieved) / len(expected_set)
            all_scores[config_key] = recall

        avg_score = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0

        explanation = f"Recall scores (schema: {schema_name}): " + ", ".join(
            f"{k}={v:.3f}" for k, v in all_scores.items()
        )

        return Score(
            value=avg_score,
            explanation=explanation,
            metadata={
                "individual_scores": all_scores,
                "schema": schema_name,
                "expected_count": len(expected_set),
            },
        )

    return score


@scorer(metrics=[mean()])
def diversity_scorer():
    """
    Diversity scorer using schema analyzer to extract item IDs.
    Measures uniqueness of results.
    """

    async def score(state, target=None) -> Score:
        # Get schema info from metadata
        metadata = getattr(state, "metadata", {})
        schema_name = metadata.get("schema_name", "unknown")
        schema_fields = metadata.get("schema_fields", {})

        # Get schema analyzer - use default if none specified
        try:
            analyzer = get_schema_analyzer(schema_name, schema_fields)
        except Exception as e:
            logger.warning(f"Using default analyzer due to: {e}")
            from src.evaluation.core.schema_analyzer import DefaultSchemaAnalyzer

            analyzer = DefaultSchemaAnalyzer()

        all_scores = {}

        for config_key, output in state.outputs.items():
            if not output.get("success", False):
                all_scores[config_key] = 0.0
                continue

            results = output.get("results", [])
            if not results:
                all_scores[config_key] = 0.0
                continue

            # Extract item IDs using schema analyzer
            item_ids = []
            for r in results:
                try:
                    item_id = analyzer.extract_item_id(r)
                    if item_id:
                        item_ids.append(item_id)
                except Exception as e:
                    logger.warning(f"Failed to extract item ID: {e}")

            # Calculate diversity
            if item_ids:
                unique_ids = set(item_ids)
                diversity = len(unique_ids) / len(item_ids)
                all_scores[config_key] = diversity
            else:
                all_scores[config_key] = 0.0

        avg_score = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0

        explanation = f"Result diversity (schema: {schema_name}): " + ", ".join(
            f"{k}={v:.3f}" for k, v in all_scores.items()
        )

        return Score(
            value=avg_score,
            explanation=explanation,
            metadata={"individual_scores": all_scores, "schema": schema_name},
        )

    return score


@scorer(metrics=[mean()])
def schema_aware_temporal_scorer():
    """
    Temporal coherence scorer that checks if the schema has temporal fields.
    Only applies to schemas with temporal data.
    """

    async def score(state, target=None) -> Score:
        query = state.input.get("query", "")

        # Check if query has temporal intent
        temporal_keywords = [
            "when",
            "after",
            "before",
            "during",
            "timeline",
            "sequence",
            "order",
            "first",
            "last",
            "then",
        ]

        is_temporal_query = any(kw in query.lower() for kw in temporal_keywords)

        if not is_temporal_query:
            return Score(
                value=1.0,  # N/A - perfect score
                explanation="Not a temporal query",
                metadata={"temporal_query": False},
            )

        # Get schema info
        metadata = getattr(state, "metadata", {})
        schema_fields = metadata.get("schema_fields", {})

        # Check if schema has temporal fields
        temporal_fields = schema_fields.get("temporal_fields", [])

        if not temporal_fields:
            return Score(
                value=1.0,  # N/A - perfect score
                explanation="Not a temporal schema",
                metadata={"temporal_schema": False},
            )

        all_scores = {}

        for config_key, output in state.outputs.items():
            if not output.get("success", False):
                all_scores[config_key] = 0.0
                continue

            results = output.get("results", [])
            if not results:
                all_scores[config_key] = 0.0
                continue

            # Extract temporal values
            temporal_values = []
            for r in results:
                for field in temporal_fields:
                    if field in r:
                        temporal_values.append(r[field])
                        break  # Use first temporal field found

            # Check if properly ordered
            if temporal_values and len(temporal_values) > 1:
                is_ordered = all(
                    temporal_values[i] <= temporal_values[i + 1]
                    for i in range(len(temporal_values) - 1)
                )
                all_scores[config_key] = 1.0 if is_ordered else 0.0
            else:
                all_scores[config_key] = 1.0  # Single or no temporal values

        avg_score = sum(all_scores.values()) / len(all_scores) if all_scores else 0.0

        explanation = "Temporal coherence: " + ", ".join(
            f"{k}={v:.1f}" for k, v in all_scores.items()
        )

        return Score(
            value=avg_score,
            explanation=explanation,
            metadata={
                "individual_scores": all_scores,
                "temporal_fields": temporal_fields,
            },
        )

    return score


def _calculate_keyword_relevance(query: str, contexts: list[str]) -> float:
    """Calculate keyword-based relevance score.

    Args:
        query: Search query
        contexts: List of text contexts from results

    Returns:
        Relevance score between 0 and 1
    """
    if not query or not contexts:
        return 0.0

    # Simple keyword matching
    query_words = set(query.lower().split())

    scores = []
    for context in contexts:
        context_words = set(context.lower().split())
        if context_words:
            overlap = len(query_words.intersection(context_words))
            score = overlap / len(query_words)
            scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0
