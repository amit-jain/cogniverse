"""
Extract preference pairs from telemetry annotations for DPO training.

Uses AnnotationStore to extract approved/rejected annotation pairs
for Direct Preference Optimization (DPO) training.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


@dataclass
class PreferencePair:
    """Single preference pair (chosen vs rejected) for DPO."""

    prompt: str
    chosen: str
    rejected: str
    metadata: Dict[str, Any]


@dataclass
class PreferenceDataset:
    """Dataset of preference pairs for DPO training."""

    pairs: List[PreferencePair]
    metadata: Dict[str, Any]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame([
            {
                "prompt": pair.prompt,
                "chosen": pair.chosen,
                "rejected": pair.rejected,
                **pair.metadata
            }
            for pair in self.pairs
        ])

    def save(self, path: str, format: Literal["jsonl", "parquet"] = "jsonl") -> None:
        """
        Save dataset to file.

        Args:
            path: Output file path
            format: Output format (jsonl or parquet)
        """
        df = self.to_dataframe()

        if format == "jsonl":
            df.to_json(path, orient="records", lines=True)
        elif format == "parquet":
            df.to_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved {len(self.pairs)} preference pairs to {path} ({format})")


class PreferencePairExtractor:
    """
    Extract preference pairs from telemetry annotations.

    Uses TelemetryProvider's TraceStore and AnnotationStore to find
    spans with both approved and rejected annotations, creating
    preference pairs for Direct Preference Optimization (DPO) training.
    """

    def __init__(self, provider: TelemetryProvider):
        """
        Initialize extractor with telemetry provider.

        Args:
            provider: Initialized TelemetryProvider (e.g., PhoenixProvider)
        """
        self.provider = provider

    async def extract(
        self,
        project: str,
        agent_type: Literal["routing", "profile_selection", "entity_extraction"],
        min_pairs: int = 10,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> PreferenceDataset:
        """
        Extract preference pairs from annotations.

        Args:
            project: Project name (e.g., "cogniverse-tenant1")
            agent_type: Type of agent to extract for
            min_pairs: Minimum number of preference pairs required
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            PreferenceDataset with chosen/rejected pairs

        Raises:
            ValueError: If insufficient preference pairs found
        """
        logger.info(
            f"Extracting preference pairs: project={project}, "
            f"agent={agent_type}, min_pairs={min_pairs}"
        )

        # Get stores from provider (using public properties)
        trace_store = self.provider.traces
        annotation_store = self.provider.annotations

        # 1. Query spans for the agent type
        logger.info(f"Querying spans from project={project}...")
        spans_df = await trace_store.get_spans(
            project=project,
            start_time=start_time,
            end_time=end_time,
            limit=10000,
        )

        if spans_df.empty:
            raise ValueError(f"No spans found in project {project}")

        # Filter spans by agent type
        agent_spans = self._filter_agent_spans(spans_df, agent_type)
        logger.info(f"Found {len(agent_spans)} {agent_type} spans")

        # 2. Get annotations for these spans
        logger.info(f"Querying annotations for {len(agent_spans)} spans...")
        annotations_df = await annotation_store.get_annotations(
            spans_df=agent_spans,
            project=project,
        )

        logger.info(f"Found {len(annotations_df)} total annotations")

        # 3. Group annotations by span_id and extract pairs
        pairs = self._create_preference_pairs(agent_spans, annotations_df, agent_type)

        logger.info(f"Created {len(pairs)} preference pairs")

        if len(pairs) < min_pairs:
            raise ValueError(
                f"Insufficient preference pairs: {len(pairs)} < {min_pairs}. "
                "Need spans with both approved and rejected annotations."
            )

        return PreferenceDataset(
            pairs=pairs,
            metadata={
                "project": project,
                "agent_type": agent_type,
                "total_spans": len(spans_df),
                "agent_spans": len(agent_spans),
                "total_annotations": len(annotations_df),
                "preference_pairs": len(pairs),
                "created_at": datetime.utcnow().isoformat(),
            }
        )

    def _filter_agent_spans(
        self, spans_df: pd.DataFrame, agent_type: str
    ) -> pd.DataFrame:
        """
        Filter spans for specific agent type.

        Args:
            spans_df: All spans from project
            agent_type: Agent type to filter for

        Returns:
            Filtered DataFrame with agent-specific spans
        """
        # Agent span naming convention
        agent_keywords = {
            "routing": ["routing", "route"],
            "profile_selection": ["profile", "selection"],
            "entity_extraction": ["entity", "extraction"],
        }

        keywords = agent_keywords.get(agent_type, [agent_type])

        # Filter spans by name containing keywords (case-insensitive)
        mask = spans_df["name"].str.lower().str.contains("|".join(keywords), na=False)
        return spans_df[mask].copy()

    def _create_preference_pairs(
        self,
        spans_df: pd.DataFrame,
        annotations_df: pd.DataFrame,
        agent_type: str,
    ) -> List[PreferencePair]:
        """
        Create preference pairs from spans with multiple annotations.

        For DPO, we need:
        - Same prompt (input)
        - Chosen response (approved annotation)
        - Rejected response (rejected annotation)

        Strategy:
        1. Group annotations by span_id
        2. Find spans with both approved and rejected annotations
        3. Create pairs from the different response variations

        Args:
            spans_df: Agent spans
            annotations_df: All annotations
            agent_type: Type of agent

        Returns:
            List of PreferencePair objects
        """
        pairs = []

        if annotations_df.empty:
            return pairs

        # Group annotations by span_id
        grouped_annotations = annotations_df.groupby("span_id")

        for span_id, annotation_group in grouped_annotations:
            # Check if we have both approved and rejected for this span
            approved_mask = (annotation_group["result.label"] == "approved") | (
                annotation_group["result.score"] >= 0.5
            )
            rejected_mask = (annotation_group["result.label"] == "rejected") | (
                annotation_group["result.score"] < 0.5
            )

            approved = annotation_group[approved_mask]
            rejected = annotation_group[rejected_mask]

            if approved.empty or rejected.empty:
                continue

            # Get the span data
            span_row = spans_df[spans_df["context.span_id"] == span_id]
            if span_row.empty:
                continue

            span_row = span_row.iloc[0]

            # Extract prompt (same for both chosen and rejected)
            prompt = self._extract_prompt(span_row, agent_type)
            if not prompt:
                continue

            # Extract chosen and rejected responses
            chosen_response = self._extract_response_from_annotation(
                approved.iloc[0], span_row
            )
            rejected_response = self._extract_response_from_annotation(
                rejected.iloc[0], span_row
            )

            if not chosen_response or not rejected_response:
                continue

            # Skip pairs where chosen and rejected are identical (no learning signal)
            if chosen_response == rejected_response:
                logger.warning(
                    f"Skipping pair with identical chosen/rejected responses for span {span_id}"
                )
                continue

            pairs.append(PreferencePair(
                prompt=prompt,
                chosen=chosen_response,
                rejected=rejected_response,
                metadata={
                    "span_id": span_id,
                    "agent_type": agent_type,
                    "chosen_score": float(approved.iloc[0].get("result.score", 1.0)),
                    "rejected_score": float(rejected.iloc[0].get("result.score", 0.0)),
                    "start_time": span_row.get("start_time"),
                }
            ))

        return pairs

    def _extract_prompt(self, span_row: pd.Series, agent_type: str) -> str:
        """Extract prompt from span attributes."""
        # Get attributes
        attributes = {
            k: v for k, v in span_row.items()
            if k.startswith("attributes.")
        }

        # Common input attribute names
        input_keys = [
            "attributes.input.query",
            "attributes.input.text",
            "attributes.input.request",
            "attributes.query",
            "attributes.text",
        ]

        for key in input_keys:
            if key in attributes and attributes[key]:
                return str(attributes[key])

        return ""

    def _extract_response_from_annotation(
        self, annotation_row: pd.Series, span_row: pd.Series
    ) -> str:
        """
        Extract response from annotation metadata or span output.

        Args:
            annotation_row: Annotation row
            span_row: Associated span row

        Returns:
            Response text
        """
        # First try to get response from annotation metadata
        # (if human edited the response)
        metadata_cols = [col for col in annotation_row.index if col.startswith("metadata.")]
        for col in metadata_cols:
            if "response" in col.lower() or "output" in col.lower():
                value = annotation_row[col]
                if value:
                    if isinstance(value, (dict, list)):
                        import json
                        return json.dumps(value)
                    return str(value)

        # Fallback to span output attributes
        attributes = {
            k: v for k, v in span_row.items()
            if k.startswith("attributes.")
        }

        output_keys = [
            "attributes.output.response",
            "attributes.output.result",
            "attributes.output.decision",
            "attributes.response",
            "attributes.result",
        ]

        for key in output_keys:
            if key in attributes and attributes[key]:
                value = attributes[key]
                if isinstance(value, (dict, list)):
                    import json
                    return json.dumps(value)
                return str(value)

        return ""
