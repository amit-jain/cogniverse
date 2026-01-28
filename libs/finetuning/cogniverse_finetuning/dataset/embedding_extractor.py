"""
Triplet extraction for embedding model fine-tuning.

Extracts (anchor, positive, negative) triplets from search logs
for contrastive learning with hard negative mining.
"""

import json
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set

import numpy as np
import pandas as pd

from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


@dataclass
class Triplet:
    """
    Training triplet for contrastive learning.

    Structure:
    - anchor: Query (text, image, or video)
    - positive: Relevant result (clicked or high-scored)
    - negative: Irrelevant result (hard negative from search)
    - modality: Type of content (video, image, text)
    - metadata: Additional context (query_id, timestamp, etc.)
    """

    anchor: str
    positive: str
    negative: str
    modality: Literal["video", "image", "text"]
    metadata: Dict


class TripletExtractor:
    """
    Extract training triplets from search telemetry.

    Uses TelemetryProvider to query search spans and annotations,
    then applies hard negative mining to create challenging examples.
    """

    def __init__(self, provider: TelemetryProvider):
        """
        Initialize with telemetry provider.

        Args:
            provider: TelemetryProvider instance for querying spans/annotations
        """
        self.provider = provider

    async def extract(
        self,
        project: str,
        modality: Literal["video", "image", "text"],
        strategy: Literal["top_k", "above_threshold", "random_sampling"] = "top_k",
        min_triplets: int = 100,
    ) -> List[Triplet]:
        """
        Extract triplets from search logs.

        Args:
            project: Project name (e.g., "cogniverse-tenant1")
            modality: Content type to extract
            strategy: Hard negative mining strategy
            min_triplets: Minimum triplets needed

        Returns:
            List of Triplet objects

        Workflow:
        1. Query search spans from TelemetryProvider
        2. Extract search results (candidates)
        3. Get click/relevance annotations (positive labels)
        4. Mine hard negatives (high-scoring but not clicked)
        5. Create triplets (anchor=query, positive=clicked, negative=hard)
        """
        logger.info(
            f"Extracting triplets: project={project}, modality={modality}, "
            f"strategy={strategy}, min={min_triplets}"
        )

        # 1. Query search spans
        spans_df = await self.provider._trace_store.get_spans(project=project)

        if spans_df.empty:
            logger.warning(f"No spans found in project {project}")
            return []

        # Filter for search spans
        search_spans = self._filter_search_spans(spans_df, modality)
        logger.info(f"Found {len(search_spans)} search spans")

        # 2. Query annotations for clicks/relevance
        annotations_df = await self.provider._annotation_store.get_annotations(
            spans_df=search_spans,
            project=project,
            annotation_names=["result_click", "result_relevance"],
        )

        logger.info(f"Found {len(annotations_df)} annotations")

        # 3. Extract triplets
        triplets = []
        for _, span in search_spans.iterrows():
            span_triplets = self._extract_from_span(
                span, annotations_df, strategy, modality
            )
            triplets.extend(span_triplets)

        logger.info(f"Extracted {len(triplets)} triplets")

        # 4. Check if sufficient
        if len(triplets) < min_triplets:
            logger.warning(
                f"Insufficient triplets: {len(triplets)} < {min_triplets}. "
                "Consider synthetic generation or lowering threshold."
            )

        return triplets

    def _filter_search_spans(
        self, spans_df: pd.DataFrame, modality: str
    ) -> pd.DataFrame:
        """Filter spans for search operations with specific modality."""
        # Search spans have name containing "search"
        search_mask = spans_df["name"].str.lower().str.contains("search", na=False)

        # Modality filter (check span attributes)
        modality_mask = spans_df["attributes"].apply(
            lambda attrs: (
                self._check_modality(attrs, modality)
                if isinstance(attrs, dict)
                else False
            )
        )

        return spans_df[search_mask & modality_mask].copy()

    def _check_modality(self, attributes: Dict, target_modality: str) -> bool:
        """Check if span attributes match target modality."""
        # Check common modality fields
        modality_fields = ["input.modality", "modality", "search_type"]

        for field in modality_fields:
            if field in attributes:
                value = attributes[field]
                if isinstance(value, str) and target_modality.lower() in value.lower():
                    return True

        return False

    def _extract_from_span(
        self,
        span: pd.Series,
        annotations_df: pd.DataFrame,
        strategy: str,
        modality: str,
    ) -> List[Triplet]:
        """Extract triplets from a single search span."""
        try:
            # Get query (anchor)
            attributes = span.get("attributes", {})
            if isinstance(attributes, str):
                attributes = json.loads(attributes)

            query = attributes.get("input.query", attributes.get("query", ""))
            if not query:
                return []

            # Get search results
            results_str = attributes.get(
                "output.results", attributes.get("results", "[]")
            )
            if isinstance(results_str, str):
                results = json.loads(results_str)
            else:
                results = results_str

            if not results:
                return []

            # Get annotations for this span
            span_id = span.get("span_id", span.get("context.span_id", ""))
            span_annotations = annotations_df[annotations_df["span_id"] == span_id]

            # Identify clicked/relevant results (positives)
            clicked_ids = self._get_clicked_results(span_annotations)

            if not clicked_ids:
                # No positive examples - skip
                return []

            # Mine hard negatives
            hard_negatives = self._mine_hard_negatives(results, clicked_ids, strategy)

            if not hard_negatives:
                # No negatives - skip
                return []

            # Create triplets
            triplets = []
            for positive_id in clicked_ids:
                for negative_id in hard_negatives:
                    # Get content for positive and negative
                    positive_content = self._get_result_content(results, positive_id)
                    negative_content = self._get_result_content(results, negative_id)

                    if positive_content and negative_content:
                        triplets.append(
                            Triplet(
                                anchor=query,
                                positive=positive_content,
                                negative=negative_content,
                                modality=modality,
                                metadata={
                                    "span_id": span_id,
                                    "timestamp": span.get("start_time"),
                                    "strategy": strategy,
                                },
                            )
                        )

            return triplets

        except Exception as e:
            logger.warning(f"Failed to extract triplets from span: {e}")
            return []

    def _get_clicked_results(self, annotations: pd.DataFrame) -> Set[str]:
        """Extract clicked/relevant result IDs from annotations."""
        clicked = set()

        for _, annotation in annotations.iterrows():
            # Check annotation type
            ann_name = annotation.get("name", "")

            if ann_name == "result_click":
                # Direct click annotation
                result_id = annotation.get("result.result_id")
                if result_id:
                    clicked.add(result_id)

            elif ann_name == "result_relevance":
                # Relevance score annotation
                score = annotation.get("result.score", 0.0)
                if score >= 0.7:  # High relevance threshold
                    result_id = annotation.get("result.result_id")
                    if result_id:
                        clicked.add(result_id)

        return clicked

    def _mine_hard_negatives(
        self,
        results: List[Dict],
        clicked_ids: Set[str],
        strategy: Literal["top_k", "above_threshold", "random_sampling"],
    ) -> List[str]:
        """
        Mine hard negatives from search results.

        Strategies:
        1. top_k: Top K non-clicked results (model thinks relevant, user disagrees)
        2. above_threshold: Non-clicked results above median score
        3. random_sampling: Random non-clicked results (easier negatives)
        """
        # Filter non-clicked results
        non_clicked = [r for r in results if r.get("id") not in clicked_ids]

        if not non_clicked:
            return []

        if strategy == "top_k":
            # Top 5 non-clicked results (hardest negatives)
            # These are high-scoring results that user didn't click
            return [r["id"] for r in non_clicked[:5]]

        elif strategy == "above_threshold":
            # Non-clicked results with score > median
            scores = [r.get("score", 0.0) for r in non_clicked]
            if not scores:
                return []

            threshold = np.median(scores)
            return [r["id"] for r in non_clicked if r.get("score", 0.0) > threshold]

        elif strategy == "random_sampling":
            # Random sample for easier training
            k = min(5, len(non_clicked))
            sampled = random.sample(non_clicked, k=k)
            return [r["id"] for r in sampled]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _get_result_content(self, results: List[Dict], result_id: str) -> Optional[str]:
        """Extract content from result by ID."""
        for result in results:
            if result.get("id") == result_id:
                # Try different content fields
                content_fields = ["content", "text", "description", "title"]
                for field in content_fields:
                    if field in result:
                        return result[field]

                # Fallback: return ID if no content found
                return result_id

        return None


class TripletDataset:
    """
    Dataset of triplets for contrastive learning.

    Wraps list of Triplet objects with metadata.
    """

    def __init__(self, triplets: List[Triplet], modality: str):
        self.triplets = triplets
        self.modality = modality

    def __len__(self) -> int:
        return len(self.triplets)

    def to_dict_list(self) -> List[Dict]:
        """
        Convert to list of dicts for sentence-transformers.

        Format:
        [
            {"anchor": "...", "positive": "...", "negative": "..."},
            ...
        ]
        """
        return [
            {
                "anchor": t.anchor,
                "positive": t.positive,
                "negative": t.negative,
                "modality": t.modality,
                "metadata": t.metadata,
            }
            for t in self.triplets
        ]

    def to_input_examples(self):
        """
        Convert to sentence-transformers InputExample format.

        Returns:
            List[InputExample] for training
        """
        from sentence_transformers import InputExample

        return [
            InputExample(texts=[t.anchor, t.positive, t.negative])
            for t in self.triplets
        ]
