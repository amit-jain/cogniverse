"""
Modality Span Collector

Collects and groups Phoenix routing spans by modality for specialized optimization.
Part of Phase 11: Multi-Modal Optimization.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import phoenix as px
from cogniverse_core.telemetry.config import SPAN_NAME_ROUTING, TelemetryConfig

from cogniverse_agents.search.multi_modal_reranker import QueryModality

logger = logging.getLogger(__name__)


class ModalitySpanCollector:
    """
    Collects Phoenix spans and groups them by modality for per-modality optimization.

    Key Features:
    - Query Phoenix for routing spans
    - Group spans by query.modality_intent attribute (set in Phase 10)
    - Support filtering by success rate, confidence thresholds
    - Return Dict[QueryModality, List[SpanData]]
    """

    def __init__(self, tenant_id: str = "default"):
        """
        Initialize modality span collector

        Args:
            tenant_id: Tenant identifier for multi-tenancy
        """
        self.tenant_id = tenant_id
        self.telemetry_config = TelemetryConfig.from_env()
        self.phoenix_client = px.Client()

        # Get project name where routing spans are stored
        self.project_name = self.telemetry_config.get_project_name(
            tenant_id, service="cogniverse-orchestration"
        )

        logger.info(
            f"🔧 Initialized ModalitySpanCollector for tenant '{tenant_id}' "
            f"(project: {self.project_name})"
        )

    async def collect_spans_by_modality(
        self,
        lookback_hours: int = 24,
        min_confidence: float = 0.0,
        min_success_rate: Optional[float] = None,
        max_spans_per_modality: Optional[int] = None,
    ) -> Dict[QueryModality, List[Dict[str, Any]]]:
        """
        Collect routing spans and group by modality

        Args:
            lookback_hours: How far back to look for spans
            min_confidence: Minimum routing confidence to include
            min_success_rate: Minimum success rate filter (optional)
            max_spans_per_modality: Maximum spans to collect per modality

        Returns:
            Dictionary mapping QueryModality to list of span data:
            {
                QueryModality.VIDEO: [span1, span2, ...],
                QueryModality.DOCUMENT: [span3, span4, ...],
                ...
            }
        """
        logger.info(
            f"🔍 Collecting spans from last {lookback_hours} hours "
            f"(project: {self.project_name})"
        )

        # Query Phoenix for routing spans
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)

        try:
            spans_df = self.phoenix_client.get_spans_dataframe(
                project_name=self.project_name,
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as e:
            logger.error(f"❌ Error querying Phoenix spans: {e}")
            return {}

        if spans_df.empty:
            logger.info(f"📭 No spans found in project {self.project_name}")
            return {}

        # Filter for routing spans
        routing_spans_df = spans_df[spans_df["name"] == SPAN_NAME_ROUTING]

        if routing_spans_df.empty:
            logger.info(
                f"📭 No cogniverse.routing spans found "
                f"(total spans: {len(spans_df)})"
            )
            return {}

        logger.info(
            f"📊 Found {len(routing_spans_df)} routing spans "
            f"(from {len(spans_df)} total spans)"
        )

        # Group spans by modality
        modality_spans = defaultdict(list)

        for _, span_row in routing_spans_df.iterrows():
            try:
                # Apply confidence filter
                confidence = self._extract_confidence(span_row)
                if confidence < min_confidence:
                    continue

                # Apply success rate filter if specified
                if min_success_rate is not None:
                    success = self._extract_success(span_row)
                    if not success:
                        continue

                # Extract modality intent
                modality_intent = self._extract_modality_intent(span_row)

                if not modality_intent:
                    # No modality detected - skip or categorize as TEXT
                    continue

                # Determine primary modality
                primary_modality = self._determine_primary_modality(modality_intent)

                if primary_modality:
                    # Convert span row to dictionary for easier handling
                    span_data = self._span_row_to_dict(span_row)
                    modality_spans[primary_modality].append(span_data)

            except Exception as e:
                logger.warning(f"⚠️ Error processing span: {e}")
                continue

        # Apply per-modality limits if specified
        if max_spans_per_modality:
            for modality in modality_spans:
                if len(modality_spans[modality]) > max_spans_per_modality:
                    # Keep most recent spans
                    modality_spans[modality] = sorted(
                        modality_spans[modality],
                        key=lambda s: s.get("start_time", datetime.min),
                        reverse=True,
                    )[:max_spans_per_modality]

        # Log distribution
        for modality, spans in modality_spans.items():
            logger.info(f"  {modality.value}: {len(spans)} spans")

        logger.info(
            f"✅ Grouped {sum(len(s) for s in modality_spans.values())} spans "
            f"into {len(modality_spans)} modalities"
        )

        return dict(modality_spans)

    def _extract_modality_intent(self, span_row: pd.Series) -> List[str]:
        """
        Extract query.modality_intent attribute set in Phase 10

        Args:
            span_row: Pandas Series containing span data

        Returns:
            List of modality intent strings (e.g., ["video", "document"])
        """
        # Try Phoenix flattened format first
        attributes = span_row.get("attributes", {})

        # Check for nested query attributes
        if isinstance(attributes, dict) and "query" in attributes:
            query_attrs = attributes["query"]
            if isinstance(query_attrs, dict):
                modality_intent = query_attrs.get("modality_intent")
                if modality_intent:
                    return (
                        modality_intent
                        if isinstance(modality_intent, list)
                        else [modality_intent]
                    )

        # Try direct attribute access (Phoenix dot notation)
        modality_intent_attr = attributes.get("query.modality_intent")
        if modality_intent_attr:
            return (
                modality_intent_attr
                if isinstance(modality_intent_attr, list)
                else [modality_intent_attr]
            )

        # Fallback: try to infer from routing attributes
        routing_attrs = attributes.get("routing", {})
        if isinstance(routing_attrs, dict):
            detected_modalities = routing_attrs.get("detected_modalities", [])
            if detected_modalities:
                return detected_modalities

        return []

    def _determine_primary_modality(
        self, modality_intent: List[str]
    ) -> Optional[QueryModality]:
        """
        Determine primary modality when multiple are detected

        Priority order: VIDEO > IMAGE > AUDIO > DOCUMENT > TEXT

        Args:
            modality_intent: List of detected modality strings

        Returns:
            Primary QueryModality or None
        """
        if not modality_intent:
            return None

        # Priority-based selection
        priority_order = [
            ("video", QueryModality.VIDEO),
            ("image", QueryModality.IMAGE),
            ("audio", QueryModality.AUDIO),
            ("document", QueryModality.DOCUMENT),
            ("text", QueryModality.TEXT),
        ]

        for keyword, modality in priority_order:
            if keyword in modality_intent:
                return modality

        # Default to TEXT if nothing matches
        return QueryModality.TEXT

    def _extract_confidence(self, span_row: pd.Series) -> float:
        """Extract routing confidence from span"""
        attributes = span_row.get("attributes", {})

        # Try routing.confidence
        if isinstance(attributes, dict):
            routing_attrs = attributes.get("routing", {})
            if isinstance(routing_attrs, dict):
                confidence = routing_attrs.get("confidence")
                if confidence is not None:
                    return float(confidence)

        return 0.0

    def _extract_success(self, span_row: pd.Series) -> bool:
        """
        Determine if routing was successful

        Success indicators:
        - status_code = OK
        - No errors in span
        - Result count > 0 (if available)
        """
        # Check status code
        status_code = span_row.get("status_code", "")
        if status_code != "OK":
            return False

        # Check for errors
        attributes = span_row.get("attributes", {})
        if isinstance(attributes, dict):
            routing_attrs = attributes.get("routing", {})
            if isinstance(routing_attrs, dict):
                if routing_attrs.get("error"):
                    return False

        return True

    def _span_row_to_dict(self, span_row: pd.Series) -> Dict[str, Any]:
        """
        Convert Pandas Series to dictionary for easier handling

        Args:
            span_row: Pandas Series from Phoenix dataframe

        Returns:
            Dictionary with span data
        """
        return {
            "span_id": span_row.get("context.span_id", ""),
            "start_time": span_row.get("start_time"),
            "end_time": span_row.get("end_time"),
            "status_code": span_row.get("status_code", ""),
            "attributes": span_row.get("attributes", {}),
            "name": span_row.get("name", ""),
        }

    async def get_modality_statistics(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Get statistics about modality distribution

        Args:
            lookback_hours: How far back to look

        Returns:
            Statistics dictionary with counts, success rates, etc.
        """
        modality_spans = await self.collect_spans_by_modality(lookback_hours)

        stats = {
            "total_spans": sum(len(spans) for spans in modality_spans.values()),
            "modality_distribution": {},
            "lookback_hours": lookback_hours,
            "timestamp": datetime.now().isoformat(),
        }

        for modality, spans in modality_spans.items():
            success_count = sum(1 for s in spans if self._is_successful_span(s))
            stats["modality_distribution"][modality.value] = {
                "count": len(spans),
                "success_count": success_count,
                "success_rate": success_count / len(spans) if spans else 0.0,
            }

        return stats

    def _is_successful_span(self, span_data: Dict[str, Any]) -> bool:
        """Check if span represents successful routing"""
        return span_data.get("status_code") == "OK"
