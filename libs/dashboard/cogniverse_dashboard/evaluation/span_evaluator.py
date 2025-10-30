"""
Span evaluator for evaluating existing traces in Phoenix

This module evaluates existing spans using both reference-free and golden dataset evaluators
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd
from cogniverse_core.telemetry.manager import TelemetryManager
from cogniverse_core.telemetry.providers.base import TelemetryProvider

from .evaluators.golden_dataset import (
    GoldenDatasetEvaluator,
    create_low_scoring_golden_dataset,
)
from .evaluators.reference_free import create_reference_free_evaluators

logger = logging.getLogger(__name__)


class SpanEvaluator:
    """
    Evaluates existing spans using various evaluators.

    Uses telemetry provider abstraction for backend-agnostic span querying and evaluation logging.
    """

    def __init__(
        self,
        provider: Optional[TelemetryProvider] = None,
        tenant_id: Optional[str] = None,
        project_name: Optional[str] = None
    ):
        """
        Initialize span evaluator.

        Args:
            provider: Telemetry provider (if None, uses TelemetryManager's provider)
            tenant_id: Tenant identifier for querying spans
            project_name: Project name for span queries
        """
        # Store tenant_id first (needed for provider lookup)
        self.tenant_id = tenant_id or "default"

        # Get provider from TelemetryManager if not provided
        if provider is None:
            telemetry_manager = TelemetryManager()
            provider = telemetry_manager.get_provider(tenant_id=self.tenant_id)

        self.provider = provider
        self.project_name = project_name

        # Initialize evaluators
        self.reference_free_evaluators = create_reference_free_evaluators()
        self.golden_evaluator = GoldenDatasetEvaluator(
            create_low_scoring_golden_dataset()
        )

    async def get_recent_spans(
        self,
        hours: int = 6,
        operation_name: str | None = "search_service.search",
        limit: int = 1000,
        project: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Retrieve recent spans from telemetry backend.

        Args:
            hours: Number of hours to look back
            operation_name: Filter by operation name
            limit: Maximum number of spans to retrieve
            project: Project name (uses instance project_name if not provided)

        Returns:
            DataFrame with span information
        """
        try:
            # Use telemetry provider to get spans
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)

            # Determine project name
            project_to_query = project or self.project_name
            if not project_to_query:
                # Build default project name from tenant
                project_to_query = f"cogniverse-{self.tenant_id}"

            # Get spans dataframe from telemetry provider
            spans_df = await self.provider.traces.get_spans(
                project=project_to_query,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )

            if spans_df is None or spans_df.empty:
                logger.info("No spans found in Phoenix, using mock data")
                return self._create_mock_spans_df()

            # Filter by operation name if specified
            if operation_name:
                spans_df = spans_df[spans_df["name"] == operation_name]

            # Limit results
            if len(spans_df) > limit:
                spans_df = spans_df.head(limit)

            logger.info(f"Retrieved {len(spans_df)} spans from Phoenix")

            # Convert to expected format
            formatted_spans = []
            for _, span in spans_df.iterrows():
                # Extract attributes dict
                attributes = {}

                # Look for query in various places
                query = None
                if "attributes.input.value" in span:
                    query = span["attributes.input.value"]
                elif "input.value" in span:
                    query = span["input.value"]
                elif "attributes.query" in span:
                    query = span["attributes.query"]

                if query:
                    attributes["query"] = query

                # Parse output value - it might be a string representation
                output_value = span.get("attributes.output.value", "")
                results = []

                # Check if it's a string representation of a tuple (e.g., "(19, 128)")
                if (
                    isinstance(output_value, str)
                    and output_value.startswith("(")
                    and output_value.endswith(")")
                ):
                    # This looks like embedding dimensions, not search results
                    # Skip this span as it's not a search result
                    continue
                elif isinstance(output_value, list):
                    results = output_value
                elif isinstance(output_value, str):
                    # Try to parse as JSON if it's a string
                    try:
                        import json

                        results = json.loads(output_value)
                    except Exception:
                        # Try to evaluate as Python literal (for string representations of lists)
                        try:
                            import ast

                            results = ast.literal_eval(output_value)
                            if not isinstance(results, list):
                                continue
                        except Exception:
                            # If not parseable, skip this span
                            continue

                # Only include spans that look like search results
                # Search results should have dictionaries with document_id/video_id
                if not query:
                    continue

                # Check if results look like search results
                if isinstance(results, list) and len(results) > 0:
                    # Check if first result has expected structure
                    if not isinstance(results[0], dict) or not any(
                        k in results[0]
                        for k in ["document_id", "video_id", "source_id"]
                    ):
                        continue
                elif not isinstance(results, list):
                    continue

                formatted_span = {
                    "span_id": span.get("context.span_id", ""),
                    "trace_id": span.get("context.trace_id", ""),
                    "operation_name": span.get("name", ""),
                    "attributes": attributes,
                    "outputs": {"results": results},
                }

                formatted_spans.append(formatted_span)

            return pd.DataFrame(formatted_spans)

        except Exception as e:
            logger.error(f"Error retrieving spans from Phoenix: {e}")
            logger.info("Using mock span data for demonstration")
            return self._create_mock_spans_df()

    def _create_mock_spans_df(self) -> pd.DataFrame:
        """Create mock spans data for testing"""
        mock_spans = [
            {
                "span_id": "span_001",
                "trace_id": "trace_001",
                "operation_name": "search_service.search",
                "attributes": {
                    "query": "person wearing winter clothes outdoors in daylight",
                    "is_test_query": True,
                    "dataset_id": "golden_test_v1",
                },
                "outputs": {
                    "results": [
                        {"video_id": "v_-IMXSEIabMM", "score": 0.85},
                        {"video_id": "v_HWFrgou1LD2Q", "score": 0.72},
                    ]
                },
            },
            {
                "span_id": "span_002",
                "trace_id": "trace_002",
                "operation_name": "search_service.search",
                "attributes": {"query": "industrial machinery"},
                "outputs": {
                    "results": [
                        {"video_id": "v_7qOJRNOtTV4", "score": 0.91},
                        {"video_id": "v_J0nA4VgnoCo", "score": 0.88},
                    ]
                },
            },
        ]
        return pd.DataFrame(mock_spans)

    async def evaluate_spans(
        self, spans_df: pd.DataFrame, evaluator_names: list[str] | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        Evaluate spans using specified evaluators

        Args:
            spans_df: DataFrame containing spans to evaluate
            evaluator_names: List of evaluator names to use (None = all)

        Returns:
            Dictionary mapping evaluator name to evaluation results DataFrame
        """
        if evaluator_names is None:
            evaluator_names = list(self.reference_free_evaluators.keys()) + [
                "golden_dataset"
            ]

        evaluation_results = {}

        for eval_name in evaluator_names:
            logger.info(f"Running evaluator: {eval_name}")

            if eval_name == "golden_dataset":
                evaluator = self.golden_evaluator
            else:
                evaluator = self.reference_free_evaluators.get(eval_name)

            if not evaluator:
                logger.warning(f"Evaluator '{eval_name}' not found")
                continue

            # Prepare evaluation data
            eval_records = []

            for _, span in spans_df.iterrows():
                # Extract span data
                span_id = span.get("span_id")
                attributes = span.get("attributes", {})
                outputs = span.get("outputs", {})

                query = attributes.get("query", "")
                results = outputs.get("results", [])

                # Run evaluation
                try:
                    eval_result = await evaluator.evaluate(
                        input=query, output=results, metadata=attributes
                    )

                    eval_records.append(
                        {
                            "span_id": span_id,
                            "score": eval_result.score,
                            "label": eval_result.label,
                            "explanation": eval_result.explanation,
                        }
                    )

                except Exception as e:
                    logger.error(f"Error evaluating span {span_id}: {e}")
                    eval_records.append(
                        {
                            "span_id": span_id,
                            "score": -1.0,
                            "label": "error",
                            "explanation": str(e),
                        }
                    )

            evaluation_results[eval_name] = pd.DataFrame(eval_records)

        return evaluation_results

    async def upload_evaluations_to_backend(
        self,
        evaluations: dict[str, pd.DataFrame],
        project: Optional[str] = None
    ):
        """
        Upload evaluation results to telemetry backend.

        Args:
            evaluations: Dictionary mapping evaluator name to results DataFrame
            project: Project name (uses instance project_name if not provided)
        """
        # Determine project name
        project_to_use = project or self.project_name
        if not project_to_use:
            project_to_use = f"cogniverse-{self.tenant_id}"

        for eval_name, eval_df in evaluations.items():
            if eval_df.empty:
                logger.warning(f"No evaluations to upload for {eval_name}")
                continue

            try:
                # Upload to telemetry backend using provider abstraction
                await self.provider.annotations.log_evaluations(
                    eval_name=eval_name,
                    evaluations_df=eval_df,
                    project=project_to_use
                )
                logger.info(
                    f"Uploaded {len(eval_df)} evaluations for '{eval_name}' "
                    f"(project={project_to_use})"
                )

            except Exception as e:
                logger.error(f"Failed to upload evaluations for {eval_name}: {e}")

    async def run_evaluation_pipeline(
        self,
        hours: int = 6,
        operation_name: str | None = "search_service.search",
        evaluator_names: list[str] | None = None,
        upload_to_backend: bool = True,
        project: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Run complete evaluation pipeline on recent spans.

        Args:
            hours: Number of hours to look back for spans
            operation_name: Filter by operation name
            evaluator_names: List of evaluators to use
            upload_to_backend: Whether to upload results to telemetry backend
            project: Project name (uses instance project_name if not provided)

        Returns:
            Summary of evaluation results
        """
        logger.info(f"Starting span evaluation pipeline for last {hours} hours")

        # Get recent spans
        spans_df = await self.get_recent_spans(hours, operation_name, project=project)
        logger.info(f"Retrieved {len(spans_df)} spans")

        if spans_df.empty:
            return {"error": "No spans found"}

        # Run evaluations
        evaluations = await self.evaluate_spans(spans_df, evaluator_names)

        # Upload to telemetry backend if requested
        if upload_to_backend:
            await self.upload_evaluations_to_backend(evaluations, project=project)

        # Generate summary
        summary = {
            "num_spans_evaluated": len(spans_df),
            "evaluators_run": list(evaluations.keys()),
            "results": {},
        }

        for eval_name, eval_df in evaluations.items():
            if not eval_df.empty:
                summary["results"][eval_name] = {
                    "num_evaluated": len(eval_df),
                    "mean_score": eval_df["score"].mean(),
                    "score_distribution": eval_df["label"].value_counts().to_dict(),
                }

        return summary


async def evaluate_existing_spans():
    """
    Example function to evaluate existing spans
    """
    evaluator = SpanEvaluator()

    # Run evaluation on recent spans
    summary = await evaluator.run_evaluation_pipeline(
        hours=24,  # Last 24 hours
        evaluator_names=["relevance", "diversity", "golden_dataset"],
        upload_to_phoenix=True,
    )

    logger.info(f"Evaluation complete: {summary}")
    return summary
