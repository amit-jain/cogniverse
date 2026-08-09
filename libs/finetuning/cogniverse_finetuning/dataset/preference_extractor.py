"""
Extract preference pairs from telemetry annotations for DPO training.

Uses AnnotationStore to extract approved/rejected annotation pairs
for Direct Preference Optimization (DPO) training.
"""

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from cogniverse_finetuning.dataset.output_projection import project_training_output
from cogniverse_foundation.telemetry.providers.base import TelemetryProvider
from cogniverse_foundation.telemetry.span_contract import PREFERENCE_CHOSEN_THRESHOLD

logger = logging.getLogger(__name__)


def classify_preference_annotation(annotation_row: pd.Series) -> Optional[str]:
    """Classify one review annotation only when label and score agree."""
    label = annotation_row.get("result.label")
    label_classification = (
        label.lower()
        if isinstance(label, str) and label.lower() in {"approved", "rejected"}
        else None
    )

    score = annotation_row.get("result.score")
    score_classification = None
    if (
        not isinstance(score, bool)
        and isinstance(score, (int, float))
        and math.isfinite(float(score))
    ):
        score_classification = (
            "approved" if float(score) >= PREFERENCE_CHOSEN_THRESHOLD else "rejected"
        )

    if (
        label_classification is not None
        and score_classification is not None
        and label_classification != score_classification
    ):
        return None
    return label_classification or score_classification


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
        return pd.DataFrame(
            [
                {
                    "prompt": pair.prompt,
                    "chosen": pair.chosen,
                    "rejected": pair.rejected,
                    **pair.metadata,
                }
                for pair in self.pairs
            ]
        )

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
        spans_df = await trace_store.get_all_spans(
            project=project,
            start_time=start_time,
            end_time=end_time,
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
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
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
            "routing": ["routing", "route", "gateway"],
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
            classifications = annotation_group.apply(
                self._classify_annotation,
                axis=1,
            )
            approved = annotation_group[classifications == "approved"]
            rejected = annotation_group[classifications == "rejected"]

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

            projected_approved = [
                (
                    approved_row,
                    self._project_reviewed_response(
                        approved_row,
                        agent_type=agent_type,
                        span_id=str(span_id),
                        role="chosen",
                    ),
                )
                for _, approved_row in approved.iterrows()
            ]
            projected_rejected = [
                (
                    rejected_row,
                    self._project_reviewed_response(
                        rejected_row,
                        agent_type=agent_type,
                        span_id=str(span_id),
                        role="rejected",
                    ),
                )
                for _, rejected_row in rejected.iterrows()
            ]

            selected = None
            for approved_row, chosen_response in projected_approved:
                for rejected_row, rejected_response in projected_rejected:
                    if chosen_response != rejected_response:
                        selected = (
                            approved_row,
                            rejected_row,
                            chosen_response,
                            rejected_response,
                        )
                        break
                if selected is not None:
                    break

            if selected is None:
                logger.warning(
                    "Skipping preference span %s whose canonical chosen and "
                    "rejected responses are identical",
                    span_id,
                )
                continue

            approved_row, rejected_row, chosen_response, rejected_response = selected

            pairs.append(
                PreferencePair(
                    prompt=prompt,
                    chosen=chosen_response,
                    rejected=rejected_response,
                    metadata={
                        "span_id": span_id,
                        "agent_type": agent_type,
                        "chosen_score": float(approved_row["result.score"]),
                        "rejected_score": float(rejected_row["result.score"]),
                        "start_time": span_row.get("start_time"),
                    },
                )
            )

        return pairs

    def _project_reviewed_response(
        self,
        annotation_row: pd.Series,
        *,
        agent_type: str,
        span_id: str,
        role: str,
    ) -> str:
        context = f"{agent_type} preference span {span_id} {role} response"
        response = self._extract_annotation_response(annotation_row).strip()
        if not response:
            raise ValueError(f"{context} must be present in annotation metadata")
        try:
            response_values = json.loads(response)
        except json.JSONDecodeError as error:
            raise ValueError(f"{context} must be valid JSON: {error.msg}") from error
        return project_training_output(
            agent_type,
            response_values,
            context=context,
        )

    @staticmethod
    def _classify_annotation(annotation_row: pd.Series) -> Optional[str]:
        return classify_preference_annotation(annotation_row)

    def _extract_prompt(self, span_row: pd.Series, agent_type: str) -> str:
        """Extract prompt from span attributes."""
        # Get attributes
        attributes = {k: v for k, v in span_row.items() if k.startswith("attributes.")}

        # Common input attribute names
        input_keys = [
            "attributes.input.value",
            "attributes.input.query",
            "attributes.input.text",
            "attributes.input.request",
            "attributes.query",
            "attributes.text",
        ]

        for key in input_keys:
            if key in attributes and attributes[key]:
                return str(attributes[key])

        # Phoenix groups namespaced attributes into a nested dict column:
        # input.query -> attributes.input == {"query": ...}. Read those too.
        for ns_col, subkeys in (
            ("attributes.input", ("query", "text", "request", "value")),
            ("attributes.query", ("value",)),
        ):
            ns = attributes.get(ns_col)
            if isinstance(ns, dict):
                for sub in subkeys:
                    if ns.get(sub):
                        return str(ns[sub])

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
        annotation_response = self._extract_annotation_response(annotation_row)
        if annotation_response:
            return annotation_response

        # Fallback to span output attributes
        attributes = {k: v for k, v in span_row.items() if k.startswith("attributes.")}

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

        # Phoenix groups namespaced attributes into a nested dict column:
        # output.response -> attributes.output == {"response": ...}.
        ns = attributes.get("attributes.output")
        if isinstance(ns, dict):
            for sub in ("response", "result", "decision", "value"):
                if ns.get(sub):
                    value = ns[sub]
                    if isinstance(value, (dict, list)):
                        import json

                        return json.dumps(value)
                    return str(value)

        return ""

    @staticmethod
    def _extract_annotation_response(annotation_row: pd.Series) -> str:
        """Extract an explicitly reviewed response from annotation metadata."""
        meta = annotation_row.get("metadata")
        if isinstance(meta, dict):
            for key, value in meta.items():
                if ("response" in key.lower() or "output" in key.lower()) and value:
                    if isinstance(value, (dict, list)):
                        import json

                        return json.dumps(value)
                    return str(value)

        metadata_cols = [
            col for col in annotation_row.index if col.startswith("metadata.")
        ]
        for col in metadata_cols:
            if "response" in col.lower() or "output" in col.lower():
                value = annotation_row[col]
                if value:
                    if isinstance(value, (dict, list)):
                        import json

                        return json.dumps(value)
                    return str(value)

        return ""
