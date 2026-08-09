"""
Triplet extraction for embedding model fine-tuning.

Extracts (anchor, positive, negative) triplets from search logs
for contrastive learning with hard negative mining.
"""

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Set

import numpy as np
import pandas as pd

from cogniverse_foundation.telemetry.providers.base import TelemetryProvider
from cogniverse_foundation.telemetry.span_contract import (
    RELEVANCE_POSITIVE_THRESHOLD,
    RESULT_CLICK,
    RESULT_ID_META_KEY,
    RESULT_RELEVANCE,
    read_span_io,
)

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

        try:
            spans_df = await self.provider.traces.get_all_spans(project=project)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to query search spans from project {project}"
            ) from exc

        if not isinstance(spans_df, pd.DataFrame):
            raise TypeError(
                f"Search span query for project {project} returned "
                f"{type(spans_df).__name__}, expected pandas.DataFrame"
            )

        if spans_df.empty:
            logger.warning(f"No spans found in project {project}")
            return []

        # Filter for search spans
        search_spans = self._filter_search_spans(spans_df, modality)
        logger.info(f"Found {len(search_spans)} search spans")
        if search_spans.empty:
            return []

        # 2. Query annotations for clicks/relevance
        try:
            annotations_df = await self.provider.annotations.get_annotations(
                spans_df=search_spans,
                project=project,
                annotation_names=[RESULT_CLICK, RESULT_RELEVANCE],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to query result annotations from project {project}"
            ) from exc

        logger.info(f"Found {len(annotations_df)} annotations")
        if annotations_df.empty:
            return []

        # Phoenix returns the annotations frame indexed by span_id, with the
        # annotation name in an 'annotation_name' column. Normalize to the
        # (span_id column, name column) shape the extraction below reads —
        # otherwise every span lookup raised KeyError('span_id') and the
        # extractor yielded zero triplets.
        if not isinstance(annotations_df, pd.DataFrame):
            raise TypeError(
                f"Annotation query for project {project} returned "
                f"{type(annotations_df).__name__}, expected pandas.DataFrame"
            )
        if (
            annotations_df.index.name == "span_id"
            and "span_id" not in annotations_df.columns
        ):
            annotations_df = annotations_df.reset_index()
        if "annotation_name" in annotations_df.columns:
            annotations_df = annotations_df.rename(columns={"annotation_name": "name"})
        required = {"span_id", "name", "result.score"}
        missing = sorted(required - set(annotations_df.columns))
        if missing:
            raise ValueError(
                f"Annotation frame for project {project} is missing columns: {missing}"
            )
        annotations_df = annotations_df.copy()
        annotations_df["span_id"] = annotations_df["span_id"].map(str)

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
        if spans_df.empty:
            return spans_df.copy()
        # Search spans have name containing "search"
        search_mask = spans_df["name"].str.lower().str.contains("search", na=False)

        # Modality filter — read the canonical modality slot every producer writes.
        def _modality_matches(row) -> bool:
            m = read_span_io(row)["modality"]
            return isinstance(m, str) and modality.lower() in m.lower()

        modality_mask = spans_df.apply(_modality_matches, axis=1)

        return spans_df[search_mask & modality_mask].copy()

    def _extract_from_span(
        self,
        span: pd.Series,
        annotations_df: pd.DataFrame,
        strategy: str,
        modality: str,
    ) -> List[Triplet]:
        """Extract triplets from a single search span."""
        span_id_value = span.get("span_id", span.get("context.span_id"))
        if span_id_value is None or not str(span_id_value).strip():
            raise ValueError("Search span is missing a non-empty span ID")
        span_id = str(span_id_value)

        span_io = read_span_io(span)
        query = span_io["input"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Search span {span_id} has no non-empty query")
        query = query.strip()

        results = span_io["output"]
        if results in (None, []):
            return []
        if not isinstance(results, list) or not all(
            isinstance(result, dict) for result in results
        ):
            raise ValueError(f"Search span {span_id} output must be a list of objects")

        result_ids = [self._result_id(result) for result in results]
        if any(result_id is None for result_id in result_ids):
            raise ValueError(f"Search span {span_id} contains a result without an ID")
        if len(set(result_ids)) != len(result_ids):
            raise ValueError(f"Search span {span_id} contains duplicate result IDs")

        span_annotations = annotations_df[annotations_df["span_id"] == span_id]
        clicked_ids = self._get_clicked_results(span_annotations)
        if not clicked_ids:
            return []

        unknown_clicked = clicked_ids - set(result_ids)
        if unknown_clicked:
            raise ValueError(
                f"Search span {span_id} annotations reference unknown result IDs: "
                f"{sorted(unknown_clicked)}"
            )

        hard_negatives = self._mine_hard_negatives(results, clicked_ids, strategy)
        if not hard_negatives:
            return []

        triplets = []
        for positive_id in sorted(clicked_ids):
            for negative_id in hard_negatives:
                triplets.append(
                    Triplet(
                        anchor=query,
                        positive=self._get_result_content(
                            results, positive_id, span_id=span_id
                        ),
                        negative=self._get_result_content(
                            results, negative_id, span_id=span_id
                        ),
                        modality=modality,
                        metadata={
                            "span_id": span_id,
                            "timestamp": span.get("start_time"),
                            "strategy": strategy,
                        },
                    )
                )

        return triplets

    @staticmethod
    def _annotation_result_id(annotation: pd.Series) -> Optional[str]:
        """The clicked result id rides on the annotation metadata (Phoenix's
        ``result`` only carries label/score, never a result_id). Read the
        flattened ``metadata.result_id`` column or the nested ``metadata`` dict.
        """
        val = annotation.get(f"metadata.{RESULT_ID_META_KEY}")
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            canonical = str(val).strip()
            return canonical or None
        meta = annotation.get("metadata")
        if isinstance(meta, dict) and RESULT_ID_META_KEY in meta:
            value = meta[RESULT_ID_META_KEY]
            if value is not None:
                canonical = str(value).strip()
                return canonical or None
        return None

    def _get_clicked_results(self, annotations: pd.DataFrame) -> Set[str]:
        """Extract clicked/relevant result IDs from annotations."""
        clicked = set()

        for _, annotation in annotations.iterrows():
            # Check annotation type
            ann_name = annotation.get("name", "")
            if ann_name not in {RESULT_CLICK, RESULT_RELEVANCE}:
                raise ValueError(f"Unsupported result annotation name: {ann_name!r}")

            result_id = self._annotation_result_id(annotation)
            if result_id is None:
                raise ValueError(
                    f"{ann_name} annotation is missing metadata.{RESULT_ID_META_KEY}"
                )

            if ann_name == RESULT_CLICK:
                # Direct click annotation
                clicked.add(result_id)

            elif ann_name == RESULT_RELEVANCE:
                # Relevance score annotation
                score = annotation.get("result.score")
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise ValueError(
                        f"{ann_name} annotation for result {result_id} has an "
                        "invalid numeric score"
                    )
                if score >= RELEVANCE_POSITIVE_THRESHOLD:
                    clicked.add(result_id)

        return clicked

    @staticmethod
    def _result_id(result: Dict) -> Optional[str]:
        """Result identifier, tolerant of the backend's id key.

        Search results carry the id under ``document_id`` / ``video_id`` /
        ``source_id`` (the same set the span filter recognises), not a bare
        ``id`` — matching only ``id`` produced zero triplets on real data.
        """
        for key in ("document_id", "video_id", "source_id", "id"):
            val = result.get(key)
            if val is not None:
                canonical = str(val).strip()
                return canonical or None
        return None

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
        non_clicked = [r for r in results if self._result_id(r) not in clicked_ids]

        if not non_clicked:
            return []

        if strategy == "top_k":
            # Top 5 non-clicked results (hardest negatives)
            # These are high-scoring results that user didn't click
            return [self._result_id(r) for r in non_clicked[:5]]

        elif strategy == "above_threshold":
            # Non-clicked results with score > median
            scores = [r.get("score", 0.0) for r in non_clicked]
            if not scores:
                return []

            threshold = np.median(scores)
            return [
                self._result_id(r)
                for r in non_clicked
                if r.get("score", 0.0) > threshold
            ]

        elif strategy == "random_sampling":
            # Random sample for easier training
            k = min(5, len(non_clicked))
            sampled = random.sample(non_clicked, k=k)
            return [self._result_id(r) for r in sampled]

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _get_result_content(
        self, results: List[Dict], result_id: str, *, span_id: str
    ) -> str:
        """Extract content from result by ID."""
        for result in results:
            if self._result_id(result) == result_id:
                # Try different content fields
                content_fields = ["content", "text", "description", "title"]
                for field in content_fields:
                    value = result.get(field)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                raise ValueError(
                    f"Search span {span_id} result {result_id} has no non-empty "
                    "textual content"
                )

        raise ValueError(f"Search span {span_id} has no result {result_id}")


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
