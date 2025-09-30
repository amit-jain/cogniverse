"""
Phoenix Span Evaluator for Routing Optimization

This module extracts routing experiences from Phoenix telemetry spans
and feeds them to the AdvancedRoutingOptimizer for continuous learning.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import phoenix as px
from opentelemetry.trace import Span as ReadableSpan

from src.app.telemetry.config import TelemetryConfig
from src.evaluation.span_evaluator import SpanEvaluator

from .advanced_optimizer import AdvancedRoutingOptimizer, RoutingExperience

logger = logging.getLogger(__name__)


class PhoenixSpanEvaluator:
    """
    Evaluates Phoenix spans to extract routing experiences for optimization

    This class leverages the existing SpanEvaluator infrastructure to:
    1. Extract routing decisions and outcomes from Phoenix spans
    2. Compute quality metrics from span data
    3. Feed real experiences to the routing optimizer
    4. Replace synthetic training data with actual routing telemetry
    """

    def __init__(self, optimizer: AdvancedRoutingOptimizer, tenant_id: str = "default"):
        """Initialize span evaluator with routing optimizer"""
        self.optimizer = optimizer
        self.tenant_id = tenant_id
        self.telemetry_config = TelemetryConfig.from_env()

        # Initialize SpanEvaluator for reading existing spans
        self.span_evaluator = SpanEvaluator()
        self._last_evaluation_time = datetime.now()
        self._processed_span_ids = set()

        # Initialize Phoenix client for the routing optimization project
        self.routing_client = px.Client()

        logger.info(f"🔧 Initialized PhoenixSpanEvaluator for tenant '{tenant_id}'")

    async def evaluate_routing_spans(
        self, lookback_hours: int = 1, batch_size: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate routing spans from the last N hours

        Args:
            lookback_hours: How far back to look for spans
            batch_size: Maximum spans to process in one batch

        Returns:
            Evaluation results and metrics
        """
        logger.info(f"🔍 Evaluating routing spans from last {lookback_hours} hours")

        # Use existing SpanEvaluator to get recent spans
        # Look for routing-related spans instead of search spans
        spans_df = self.span_evaluator.get_recent_spans(
            hours=lookback_hours,
            operation_name="routing_agent.route_query",  # Look for routing operations
            limit=batch_size,
        )

        if spans_df.empty:
            logger.info("📭 No routing spans found, trying broader search")
            # Fallback to any spans that might contain routing data
            spans_df = self.span_evaluator.get_recent_spans(
                hours=lookback_hours,
                operation_name=None,  # Get all spans
                limit=batch_size,
            )

        if spans_df.empty:
            logger.info("📭 No spans found at all")
            return {"spans_processed": 0, "experiences_created": 0}

        experiences_created = 0
        evaluation_results = {
            "spans_processed": len(spans_df),
            "experiences_created": 0,
            "success_rate": 0.0,
            "avg_confidence": 0.0,
            "agent_distribution": {},
            "errors": [],
        }

        for _, span_row in spans_df.iterrows():
            try:
                # Skip if already processed
                span_id = span_row.get("span_id", "")
                if span_id in self._processed_span_ids:
                    continue

                # Extract routing experience from span
                experience = self._extract_routing_experience_from_df_row(span_row)
                if experience:
                    # Record experience with optimizer
                    reward = await self.optimizer.record_routing_experience(
                        query=experience.query,
                        entities=experience.entities,
                        relationships=experience.relationships,
                        enhanced_query=experience.enhanced_query,
                        chosen_agent=experience.chosen_agent,
                        routing_confidence=experience.routing_confidence,
                        search_quality=experience.search_quality,
                        agent_success=experience.agent_success,
                        processing_time=experience.processing_time,
                        user_satisfaction=experience.user_satisfaction,
                    )

                    experiences_created += 1
                    self._processed_span_ids.add(span_id)

                    logger.info(
                        f"✅ Created experience: {experience.chosen_agent} "
                        f"(confidence: {experience.routing_confidence:.3f}, "
                        f"reward: {reward:.3f})"
                    )

            except Exception as e:
                logger.error(f"❌ Error processing span: {e}")
                evaluation_results["errors"].append(str(e))

        # Update evaluation results
        evaluation_results["experiences_created"] = experiences_created
        self._last_evaluation_time = datetime.now()

        logger.info(
            f"🎯 Span evaluation complete: {experiences_created} experiences "
            f"from {len(spans_df)} spans"
        )

        return evaluation_results

    def _extract_routing_experience_from_df_row(
        self, span_row
    ) -> Optional[RoutingExperience]:
        """
        Extract routing experience from a DataFrame row

        Args:
            span_row: Pandas Series containing span data

        Returns:
            RoutingExperience if span contains routing data, None otherwise
        """
        try:
            # Extract span attributes
            attributes = span_row.get("attributes", {})

            # Check if this looks like a routing span
            if not self._is_routing_span_df(span_row):
                return None

            # Extract routing data from span attributes or try to infer from outputs
            query = attributes.get("query", "")
            if not query:
                # Try to get query from outputs if it's a search result
                outputs = span_row.get("outputs", {})
                if "results" in outputs and outputs["results"]:
                    # This might be search results, skip for now as we want routing spans
                    return None

            # Try to infer routing information from available data
            chosen_agent = attributes.get(
                "chosen_agent", "video_search"
            )  # Default fallback
            routing_confidence = float(
                attributes.get("confidence", 0.5)
            )  # Default confidence

            # Extract entities and relationships (if available)
            entities = self._parse_json_attribute(attributes.get("entities", "[]"))
            relationships = self._parse_json_attribute(
                attributes.get("relationships", "[]")
            )
            enhanced_query = attributes.get("enhanced_query", query)

            # Derive quality metrics from span data
            search_quality = self._compute_search_quality_df(span_row)
            agent_success = self._determine_agent_success_df(span_row)
            processing_time = 0.0  # Would need timing data from span

            # Create routing experience
            experience = RoutingExperience(
                query=query,
                entities=entities,
                relationships=relationships,
                enhanced_query=enhanced_query,
                chosen_agent=chosen_agent,
                routing_confidence=routing_confidence,
                search_quality=search_quality,
                agent_success=agent_success,
                processing_time=processing_time,
                user_satisfaction=None,  # Could be extracted from feedback spans
                timestamp=datetime.now(),
            )

            return experience

        except Exception as e:
            logger.error(f"❌ Error extracting experience from span: {e}")
            return None

    def _is_routing_span_df(self, span_row) -> bool:
        """Check if span represents a routing operation"""
        operation_name = span_row.get("operation_name", "")
        attributes = span_row.get("attributes", {})

        return (
            "routing" in operation_name.lower()
            or "route_query" in operation_name.lower()
            or "chosen_agent" in attributes
            or "confidence" in attributes
            or
            # For now, accept any span with a query as potentially routing-related
            "query" in attributes
        )

    def _compute_search_quality_df(self, span_row) -> float:
        """
        Compute search quality score from span data

        This analyzes various span metrics to derive a quality score:
        - Number of results returned
        - Response time
        - Error status
        - User interaction patterns (if available)
        """
        try:
            # Default quality score
            quality = 0.5
            attributes = span_row.get("attributes", {})

            # Check for errors in span status
            span_status = span_row.get("status", None)
            if span_status:
                if span_status == "ERROR":
                    return 0.1  # Low quality for errors
                elif span_status == "OK":
                    quality += 0.2  # Bonus for successful completion

            # Factor in response time (faster is better, up to a point)
            # For DataFrame, we don't have direct span timing, use default
            processing_time = 0.0
            if processing_time > 0:
                if processing_time < 1.0:  # Under 1 second is good
                    quality += 0.2
                elif processing_time < 5.0:  # Under 5 seconds is acceptable
                    quality += 0.1

            # Check for result count in attributes
            result_count = attributes.get("agent.result_count", 0)
            if isinstance(result_count, (int, float)):
                if result_count > 0:
                    quality += 0.1  # Bonus for having results
                if result_count >= 5:
                    quality += 0.1  # Extra bonus for multiple results

            # Ensure quality is in [0, 1] range
            return max(0.0, min(1.0, quality))

        except Exception as e:
            logger.warning(f"⚠️ Error computing search quality: {e}")
            return 0.5  # Default neutral quality

    def _determine_agent_success_df(self, span_row) -> bool:
        """Determine if the agent call was successful from DataFrame row"""
        try:
            attributes = span_row.get("attributes", {})

            # Check span status
            span_status = span_row.get("status", None)
            if span_status:
                return span_status == "OK"

            # Check for explicit success/failure in attributes
            agent_success = attributes.get("agent.success")
            if agent_success is not None:
                return bool(agent_success)

            # Check for HTTP status codes
            http_status = attributes.get("http.status_code")
            if http_status:
                return 200 <= int(http_status) < 300

            # Default to success if no clear indicators
            return True

        except Exception as e:
            logger.warning(f"⚠️ Error determining agent success: {e}")
            return False

    def _determine_agent_success(
        self, span: ReadableSpan, attributes: Dict[str, Any]
    ) -> bool:
        """Determine if the agent call was successful"""
        try:
            # Check span status
            span_status = getattr(span, "status", None)
            if span_status and hasattr(span_status, "status_code"):
                return span_status.status_code.name == "OK"

            # Check for explicit success/failure in attributes
            agent_success = attributes.get("agent.success")
            if agent_success is not None:
                return bool(agent_success)

            # Check for HTTP status codes
            http_status = attributes.get("http.status_code")
            if http_status:
                return 200 <= int(http_status) < 300

            # Default to success if no clear indicators
            return True

        except Exception as e:
            logger.warning(f"⚠️ Error determining agent success: {e}")
            return False

    def _get_processing_time(self, span: ReadableSpan) -> float:
        """Extract processing time from span duration"""
        try:
            if hasattr(span, "start_time") and hasattr(span, "end_time"):
                # Convert nanoseconds to seconds
                duration_ns = span.end_time - span.start_time
                return duration_ns / 1e9
            return 0.0
        except Exception:
            return 0.0

    def _parse_json_attribute(self, value: str) -> List[Dict[str, Any]]:
        """Parse JSON string attribute into Python object"""
        try:
            import json

            if isinstance(value, str):
                return json.loads(value)
            elif isinstance(value, list):
                return value
            else:
                return []
        except (json.JSONDecodeError, TypeError):
            return []

    async def start_continuous_evaluation(
        self, interval_minutes: int = 15, lookback_hours: int = 2
    ):
        """
        Start continuous evaluation of routing spans

        Args:
            interval_minutes: How often to run evaluation
            lookback_hours: How far back to look for spans
        """
        logger.info(
            f"🚀 Starting continuous span evaluation "
            f"(interval: {interval_minutes}m, lookback: {lookback_hours}h)"
        )

        while True:
            try:
                await self.evaluate_routing_spans(lookback_hours=lookback_hours)
                await asyncio.sleep(interval_minutes * 60)
            except Exception as e:
                logger.error(f"❌ Error in continuous evaluation: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
