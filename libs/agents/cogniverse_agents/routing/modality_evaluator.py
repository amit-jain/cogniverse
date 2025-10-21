"""
Modality Evaluator

Converts routing spans into modality-specific training examples with specialized features.
Part of Phase 11: Multi-Modal Optimization.
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from cogniverse_agents.routing.modality_example import ModalityExample
from cogniverse_agents.routing.modality_span_collector import ModalitySpanCollector
from cogniverse_agents.search.multi_modal_reranker import QueryModality

logger = logging.getLogger(__name__)


class ModalityEvaluator:
    """
    Evaluates routing spans to create modality-specific training examples

    Key Features:
    - Converts spans into ModalityExample objects
    - Extracts modality-specific features from span attributes
    - Supports synthetic data augmentation
    - Filters examples based on quality thresholds

    Modality-Specific Features:
    - VIDEO: temporal constraints, tutorial indicators, visual content needs
    - DOCUMENT: citation requirements, technical depth, document type
    - IMAGE: diagram type, visual complexity, annotation needs
    - AUDIO: audio type, duration preference, transcript requirements
    """

    # Feature extractors for each modality
    VIDEO_FEATURES = {
        "keywords": [
            "video",
            "watch",
            "show",
            "tutorial",
            "demonstration",
            "walkthrough",
        ],
        "temporal": ["how to", "learn", "explain"],
        "visual": ["visualize", "see", "display", "animate"],
    }

    DOCUMENT_FEATURES = {
        "keywords": [
            "read",
            "paper",
            "article",
            "documentation",
            "whitepaper",
            "guide",
        ],
        "technical": [
            "specification",
            "technical",
            "detailed",
            "comprehensive",
            "analysis",
        ],
        "research": ["research", "study", "citation", "reference"],
    }

    IMAGE_FEATURES = {
        "keywords": [
            "image",
            "picture",
            "diagram",
            "chart",
            "visualization",
            "infographic",
        ],
        "visual": ["architecture", "flowchart", "screenshot", "illustration"],
        "complexity": ["detailed", "complex", "annotated", "labeled"],
    }

    AUDIO_FEATURES = {
        "keywords": ["audio", "listen", "podcast", "recording", "talk", "interview"],
        "temporal": ["lecture", "discussion", "presentation", "speech"],
        "transcript": ["transcript", "transcription", "spoken"],
    }

    def __init__(
        self,
        span_collector: Optional[ModalitySpanCollector] = None,
        tenant_id: str = "default",
    ):
        """
        Initialize modality evaluator

        Args:
            span_collector: Optional span collector (creates one if not provided)
            tenant_id: Tenant identifier for multi-tenancy
        """
        self.tenant_id = tenant_id
        self.span_collector = span_collector or ModalitySpanCollector(tenant_id)

        logger.info(f"🔧 Initialized ModalityEvaluator for tenant '{tenant_id}'")

    async def create_training_examples(
        self,
        lookback_hours: int = 24,
        min_confidence: float = 0.7,
        augment_with_synthetic: bool = False,
        synthetic_ratio: float = 0.3,
    ) -> Dict[QueryModality, List[ModalityExample]]:
        """
        Create modality-specific training examples from spans

        Args:
            lookback_hours: How far back to look for spans
            min_confidence: Minimum routing confidence to include
            augment_with_synthetic: Whether to add synthetic examples
            synthetic_ratio: Ratio of synthetic to real examples (if augment=True)

        Returns:
            Dictionary mapping QueryModality to list of ModalityExample objects:
            {
                QueryModality.VIDEO: [example1, example2, ...],
                QueryModality.DOCUMENT: [example3, example4, ...],
                ...
            }
        """
        logger.info(
            f"📊 Creating training examples from last {lookback_hours} hours "
            f"(min_confidence: {min_confidence})"
        )

        # Collect spans grouped by modality
        modality_spans = await self.span_collector.collect_spans_by_modality(
            lookback_hours=lookback_hours,
            min_confidence=min_confidence,
        )

        if not modality_spans:
            logger.warning("⚠️ No spans found - returning empty examples")
            return {}

        # Convert spans to training examples
        training_examples: Dict[QueryModality, List[ModalityExample]] = {}

        for modality, spans in modality_spans.items():
            examples = []

            for span_data in spans:
                try:
                    example = self._span_to_example(span_data, modality)
                    if example:
                        examples.append(example)
                except Exception as e:
                    logger.warning(f"⚠️ Error converting span to example: {e}")
                    continue

            training_examples[modality] = examples
            logger.info(
                f"  {modality.value}: {len(examples)} examples "
                f"(from {len(spans)} spans)"
            )

        # Augment with synthetic data if requested
        if augment_with_synthetic:
            training_examples = await self._augment_with_synthetic(
                training_examples, synthetic_ratio
            )

        total_examples = sum(len(exs) for exs in training_examples.values())
        logger.info(
            f"✅ Created {total_examples} training examples "
            f"across {len(training_examples)} modalities"
        )

        return training_examples

    def _span_to_example(
        self, span_data: Dict[str, Any], modality: QueryModality
    ) -> Optional[ModalityExample]:
        """
        Convert span data to ModalityExample with modality-specific features

        Args:
            span_data: Span data dictionary from collector
            modality: Detected modality

        Returns:
            ModalityExample or None if conversion fails
        """
        attributes = span_data.get("attributes", {})

        # Extract query text
        query = self._extract_query(attributes)
        if not query:
            return None

        # Extract agent that handled this query
        agent = self._extract_agent(attributes)
        if not agent:
            return None

        # Determine success
        success = span_data.get("status_code") == "OK"

        # Extract modality-specific features
        modality_features = self._extract_modality_features(query, modality, attributes)

        return ModalityExample(
            query=query,
            modality=modality,
            correct_agent=agent,
            success=success,
            modality_features=modality_features,
            is_synthetic=False,
            synthetic_source=None,
        )

    def _extract_query(self, attributes: Dict[str, Any]) -> Optional[str]:
        """Extract query text from span attributes"""
        # Try nested format
        if "query" in attributes:
            query_attrs = attributes["query"]
            if isinstance(query_attrs, dict):
                query_text = query_attrs.get("text") or query_attrs.get("query")
                if query_text:
                    return str(query_text)

        # Try dot notation
        query_text = attributes.get("query.text") or attributes.get("query.query")
        if query_text:
            return str(query_text)

        # Try input attributes
        if "input" in attributes:
            input_attrs = attributes["input"]
            if isinstance(input_attrs, dict):
                query_text = input_attrs.get("value")
                if query_text:
                    return str(query_text)

        return None

    def _extract_agent(self, attributes: Dict[str, Any]) -> Optional[str]:
        """Extract selected agent from span attributes"""
        # Try routing attributes
        if "routing" in attributes:
            routing_attrs = attributes["routing"]
            if isinstance(routing_attrs, dict):
                agent = routing_attrs.get("selected_agent") or routing_attrs.get(
                    "agent"
                )
                if agent:
                    return str(agent)

        # Try dot notation
        agent = attributes.get("routing.selected_agent") or attributes.get(
            "routing.agent"
        )
        if agent:
            return str(agent)

        # Try output attributes
        if "output" in attributes:
            output_attrs = attributes["output"]
            if isinstance(output_attrs, dict):
                agent = output_attrs.get("agent")
                if agent:
                    return str(agent)

        return None

    def _extract_modality_features(
        self, query: str, modality: QueryModality, attributes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract modality-specific features from query and attributes

        Args:
            query: Query text
            modality: Query modality
            attributes: Span attributes

        Returns:
            Dictionary of modality-specific features
        """
        features: Dict[str, Any] = {}
        query_lower = query.lower()

        # Get feature definitions for this modality
        feature_defs = self._get_feature_definitions(modality)

        # Extract keyword matches
        for feature_type, keywords in feature_defs.items():
            matches = [kw for kw in keywords if kw in query_lower]
            features[f"{feature_type}_keywords"] = matches
            features[f"has_{feature_type}"] = len(matches) > 0

        # Extract query characteristics
        features["query_length"] = len(query.split())
        features["has_question"] = "?" in query
        features["has_specific_entity"] = self._has_capitalized_entity(query)

        # Extract routing metadata
        routing_attrs = attributes.get("routing", {})
        if isinstance(routing_attrs, dict):
            features["routing_confidence"] = routing_attrs.get("confidence", 0.0)
            features["detected_modalities"] = routing_attrs.get(
                "detected_modalities", []
            )

        # Extract result metadata (if available)
        features["result_count"] = self._extract_result_count(attributes)

        return features

    def _get_feature_definitions(self, modality: QueryModality) -> Dict[str, List[str]]:
        """Get feature keyword definitions for modality"""
        feature_map = {
            QueryModality.VIDEO: self.VIDEO_FEATURES,
            QueryModality.DOCUMENT: self.DOCUMENT_FEATURES,
            QueryModality.IMAGE: self.IMAGE_FEATURES,
            QueryModality.AUDIO: self.AUDIO_FEATURES,
        }
        return feature_map.get(modality, {})

    def _has_capitalized_entity(self, query: str) -> bool:
        """Check if query contains capitalized entities (proper nouns)"""
        words = query.split()
        # Count words that start with capital letter (excluding first word)
        capitalized = sum(1 for w in words[1:] if w and w[0].isupper())
        return capitalized > 0

    def _extract_result_count(self, attributes: Dict[str, Any]) -> int:
        """Extract result count from span attributes"""
        # Try output attributes
        if "output" in attributes:
            output_attrs = attributes["output"]
            if isinstance(output_attrs, dict):
                count = output_attrs.get("result_count")
                if count is not None:
                    return int(count)

        # Try results array
        results = attributes.get("results", [])
        if isinstance(results, list):
            return len(results)

        return 0

    async def _augment_with_synthetic(
        self,
        training_examples: Dict[QueryModality, List[ModalityExample]],
        synthetic_ratio: float,
    ) -> Dict[QueryModality, List[ModalityExample]]:
        """
        Augment training examples with synthetic data using NEW system

        Args:
            training_examples: Existing real examples
            synthetic_ratio: Ratio of synthetic to real examples

        Returns:
            Augmented training examples
        """
        from cogniverse_synthetic import (
            ModalityExampleSchema,
            SyntheticDataRequest,
            SyntheticDataService,
        )

        service = SyntheticDataService(vespa_client=None, backend_config=None)
        augmented_examples = training_examples.copy()

        for modality, real_examples in training_examples.items():
            if not real_examples:
                continue

            # Calculate how many synthetic examples to add
            synthetic_count = int(len(real_examples) * synthetic_ratio)

            if synthetic_count > 0:
                logger.info(
                    f"🎲 Generating {synthetic_count} synthetic examples for {modality.value}"
                )

                # Generate using SyntheticDataService directly
                request = SyntheticDataRequest(
                    optimizer="modality",
                    count=synthetic_count
                )
                response = await service.generate(request)

                # Convert to ModalityExample objects
                synthetic_examples = [
                    ModalityExample.from_schema(ModalityExampleSchema(**ex))
                    for ex in response.data
                ]

                # Combine real and synthetic
                augmented_examples[modality] = real_examples + synthetic_examples

        total_real = sum(len(exs) for exs in training_examples.values())
        total_augmented = sum(len(exs) for exs in augmented_examples.values())

        logger.info(
            f"✅ Augmented {total_real} real examples to {total_augmented} total "
            f"(added {total_augmented - total_real} synthetic)"
        )

        return augmented_examples

    def filter_by_quality(
        self,
        examples: Dict[QueryModality, List[ModalityExample]],
        min_query_length: int = 3,
        min_confidence: float = 0.7,
        require_success: bool = True,
    ) -> Dict[QueryModality, List[ModalityExample]]:
        """
        Filter training examples by quality thresholds

        Args:
            examples: Training examples to filter
            min_query_length: Minimum query word count
            min_confidence: Minimum routing confidence
            require_success: Only keep successful examples

        Returns:
            Filtered examples
        """
        filtered_examples: Dict[QueryModality, List[ModalityExample]] = {}

        for modality, example_list in examples.items():
            filtered = []

            for example in example_list:
                # Check query length
                if len(example.query.split()) < min_query_length:
                    continue

                # Check confidence (if features available)
                if example.modality_features:
                    confidence = example.modality_features.get(
                        "routing_confidence", 1.0
                    )
                    if confidence < min_confidence:
                        continue

                # Check success
                if require_success and not example.success:
                    continue

                filtered.append(example)

            filtered_examples[modality] = filtered

            if len(filtered) < len(example_list):
                logger.info(
                    f"  {modality.value}: Filtered {len(example_list)} → {len(filtered)} examples"
                )

        return filtered_examples

    def get_feature_statistics(
        self, examples: Dict[QueryModality, List[ModalityExample]]
    ) -> Dict[str, Any]:
        """
        Compute statistics about extracted features

        Args:
            examples: Training examples

        Returns:
            Statistics dictionary
        """
        stats: Dict[str, Any] = {
            "modality_counts": {},
            "feature_coverage": {},
            "success_rates": {},
            "query_characteristics": {},
        }

        for modality, example_list in examples.items():
            if not example_list:
                continue

            modality_key = modality.value

            # Count examples
            stats["modality_counts"][modality_key] = len(example_list)

            # Success rate
            success_count = sum(1 for ex in example_list if ex.success)
            stats["success_rates"][modality_key] = success_count / len(example_list)

            # Feature coverage
            feature_coverage = defaultdict(int)
            query_lengths = []

            for example in example_list:
                if example.modality_features:
                    for feature_key, value in example.modality_features.items():
                        if feature_key.startswith("has_") and value:
                            feature_coverage[feature_key] += 1

                    query_lengths.append(
                        example.modality_features.get("query_length", 0)
                    )

            stats["feature_coverage"][modality_key] = dict(feature_coverage)

            # Query characteristics
            if query_lengths:
                stats["query_characteristics"][modality_key] = {
                    "avg_length": sum(query_lengths) / len(query_lengths),
                    "min_length": min(query_lengths),
                    "max_length": max(query_lengths),
                }

        return stats
