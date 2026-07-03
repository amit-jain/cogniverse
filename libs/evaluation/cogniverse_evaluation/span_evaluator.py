"""
Span evaluator for evaluating existing traces using telemetry provider

This module evaluates existing spans using both reference-free and golden dataset evaluators
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

from .evaluators.golden_dataset import (
    GoldenDatasetEvaluator,
    create_low_scoring_golden_dataset,
)
from .evaluators.reference_free import create_reference_free_evaluators

if TYPE_CHECKING:
    from .providers.base import EvaluationProvider

logger = logging.getLogger(__name__)


class SpanEvaluator:
    """
    Evaluates existing spans using telemetry provider and various evaluators
    """

    def __init__(
        self,
        tenant_id: str,
        provider: Optional["EvaluationProvider"] = None,
        project_name: str = "cogniverse-default",
    ):
        """
        Initialize span evaluator.

        Args:
            provider: Optional evaluator provider (None = use default)
            tenant_id: Tenant identifier
            project_name: Project name for telemetry
        """
        # Get provider (lazy import to avoid circular deps)
        if provider is None:
            from .providers import get_evaluation_provider

            # Pass project_name via config, not as a direct argument
            provider = get_evaluation_provider(
                tenant_id=tenant_id,
                config={"project_name": project_name} if project_name else None,
            )

        self.provider = provider
        self.tenant_id = tenant_id
        self.project_name = project_name

        # Initialize evaluators
        self.reference_free_evaluators = create_reference_free_evaluators()
        self.golden_evaluator = GoldenDatasetEvaluator(
            create_low_scoring_golden_dataset()
        )

    @staticmethod
    def _extract_span_outputs(
        output_value: Any, require_search_shape: bool
    ) -> Optional[dict]:
        """Parse a span's ``output.value`` into an ``outputs`` dict.

        Returns ``None`` to skip the span. With ``require_search_shape`` (the
        search / golden consumers) only search-result lists survive — preserving
        the original behaviour. Without it (live per-agent judging) the raw
        parsed output is kept under ``value`` so summary/report strings and
        gateway routing-decision dicts are scored too, not just search lists.
        """
        import ast
        import json

        # Embedding-dimension spans like "(19, 128)" are encoder internals,
        # never an agent output — always skip.
        if (
            isinstance(output_value, str)
            and output_value.startswith("(")
            and output_value.endswith(")")
        ):
            return None

        parsed: Any = output_value
        if isinstance(output_value, str):
            parsed = None
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(output_value)
                    break
                except Exception:
                    parsed = None

        if require_search_shape:
            results = parsed if isinstance(parsed, list) else None
            if isinstance(results, list) and len(results) > 0:
                if not isinstance(results[0], dict) or not any(
                    k in results[0] for k in ("document_id", "video_id", "source_id")
                ):
                    return None
            elif not isinstance(results, list):
                return None
            return {"results": results}

        return {
            "results": parsed if isinstance(parsed, list) else [],
            "value": parsed if parsed is not None else output_value,
        }

    async def get_recent_spans(
        self,
        hours: int = 6,
        operation_name: str | None = "search_service.search",
        limit: int = 1000,
        require_search_shape: bool = True,
    ) -> pd.DataFrame:
        """
        Retrieve recent spans from telemetry provider

        Args:
            hours: Number of hours to look back
            operation_name: Filter by operation name
            limit: Maximum number of spans to retrieve
            require_search_shape: When True (default) only spans whose output is
                a list of search-result dicts survive — what the golden/search
                evaluators consume. Set False for live per-agent evaluation so
                summary/report/gateway spans (string / routing-dict outputs) are
                returned with their raw output under ``outputs["value"]`` instead
                of being silently dropped.

        Returns:
            DataFrame with span information
        """
        try:
            # Use provider to get spans
            # Timezone-aware UTC: the Phoenix trace store passes aware
            # datetimes through unchanged but mislabels naive ones as UTC,
            # which skews the window by the local offset and finds no spans
            # off-UTC (then falls back to mock data).
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)

            # Get spans dataframe from telemetry provider
            spans_df = await self.provider.telemetry.traces.get_spans(
                project=self.project_name,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )

            if spans_df is None or spans_df.empty:
                # No traffic -> empty, never fabricated spans (QualityMonitor
                # would persist them as live quality scores).
                logger.info("No spans found for the window")
                return pd.DataFrame()

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

                # Skip spans with no query input — every agent judge prompt
                # needs the query, and a search span without one is unusable.
                if not query:
                    continue

                output_value = span.get("attributes.output.value", "")
                outputs = self._extract_span_outputs(output_value, require_search_shape)
                if outputs is None:
                    continue

                formatted_span = {
                    "span_id": span.get("context.span_id", ""),
                    "trace_id": span.get("context.trace_id", ""),
                    "operation_name": span.get("name", ""),
                    "attributes": attributes,
                    "outputs": outputs,
                }

                formatted_spans.append(formatted_span)

            return pd.DataFrame(formatted_spans)

        except Exception as e:
            logger.error(f"Error retrieving spans from Phoenix: {e}")
            return pd.DataFrame()

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
        self,
        spans_df: pd.DataFrame,
        evaluator_names: list[str] | None = None,
        skip_span_ids: dict[str, set[str]] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Evaluate spans using specified evaluators

        Args:
            spans_df: DataFrame containing spans to evaluate
            evaluator_names: List of evaluator names to use (None = all)
            skip_span_ids: Optional ``{evaluator_name: {span_id, ...}}`` of
                spans already evaluated for that evaluator. Such (span,
                evaluator) pairs are skipped so a re-run doesn't duplicate
                annotations. See ``_already_evaluated_span_ids``.

        Returns:
            Dictionary mapping evaluator name to evaluation results DataFrame
        """
        if evaluator_names is None:
            evaluator_names = list(self.reference_free_evaluators.keys()) + [
                "golden_dataset"
            ]

        skip_span_ids = skip_span_ids or {}
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

            already_done = skip_span_ids.get(eval_name, set())

            # Prepare evaluation data
            eval_records = []

            for _, span in spans_df.iterrows():
                # Extract span data
                span_id = span.get("span_id")
                if span_id in already_done:
                    # Already carries this evaluator's annotation — skip so
                    # the re-run is incremental and doesn't double-annotate.
                    continue
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

    async def _already_evaluated_span_ids(
        self,
        hours: int,
        operation_name: str | None,
        evaluator_names: list[str],
        limit: int = 1000,
    ) -> dict[str, set[str]]:
        """Return ``{evaluator_name: {span_id, ...}}`` for spans that already
        carry that evaluator's annotation in the time window.

        Queries the telemetry provider's annotation store per evaluator name
        (``get_annotations`` takes the raw spans DataFrame and filters by
        annotation name). This is the incremental gate: spans already
        annotated for an evaluator are not re-evaluated. A query failure
        degrades to an empty set (the span is treated as un-evaluated and
        re-run) rather than dropping it silently.
        """
        result: dict[str, set[str]] = {name: set() for name in evaluator_names}

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        try:
            raw_spans_df = await self.provider.telemetry.traces.get_spans(
                project=self.project_name,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )
        except Exception as e:
            logger.warning("Incremental gate: failed to fetch spans (%s)", e)
            return result

        if raw_spans_df is None or raw_spans_df.empty:
            return result
        if operation_name and "name" in raw_spans_df.columns:
            raw_spans_df = raw_spans_df[raw_spans_df["name"] == operation_name]
        if raw_spans_df.empty:
            return result

        for name in evaluator_names:
            try:
                annotations_df = (
                    await self.provider.telemetry.annotations.get_annotations(
                        spans_df=raw_spans_df,
                        project=self.project_name,
                        annotation_names=[name],
                    )
                )
            except Exception as e:
                logger.warning(
                    "Incremental gate: annotation query failed for %s (%s)", name, e
                )
                continue
            if annotations_df is not None and not annotations_df.empty:
                # Phoenix returns the annotations frame indexed by span_id
                # (no ``span_id`` column); fall back to the index.
                if "span_id" in annotations_df.columns:
                    span_ids = annotations_df["span_id"].dropna().tolist()
                else:
                    span_ids = [s for s in annotations_df.index.tolist() if s]
                result[name] = set(span_ids)

        return result

    async def upload_evaluations(self, evaluations: dict[str, pd.DataFrame]):
        """
        Upload evaluation results as annotations

        Args:
            evaluations: Dictionary mapping evaluator name to results DataFrame
        """
        for eval_name, eval_df in evaluations.items():
            if eval_df.empty:
                logger.warning(f"No evaluations to upload for {eval_name}")
                continue

            try:
                # One bulk upload per evaluator — the dataframe already has
                # the span_id/score/label/explanation columns the batch API
                # expects. The previous per-row add_annotation loop paid one
                # HTTP round-trip per (span, evaluator) pair on every
                # quality-monitor cycle.
                await self.provider.telemetry.annotations.log_evaluations(
                    eval_name=eval_name,
                    evaluations_df=eval_df,
                    project=self.project_name,
                )

                logger.info(f"Uploaded {len(eval_df)} evaluations for {eval_name}")

            except Exception as e:
                logger.error(f"Failed to upload evaluations for {eval_name}: {e}")

    async def run_evaluation_pipeline(
        self,
        hours: int = 6,
        operation_name: str | None = "search_service.search",
        evaluator_names: list[str] | None = None,
        upload_evaluations: bool = True,
        incremental: bool = True,
    ) -> dict[str, Any]:
        """
        Run complete evaluation pipeline on recent spans

        Args:
            hours: Number of hours to look back for spans
            operation_name: Filter by operation name
            evaluator_names: List of evaluators to use
            upload_evaluations: Whether to upload results as annotations
            incremental: When True (default), skip (span, evaluator) pairs
                that already carry that evaluator's annotation, so a re-run
                over the same window only evaluates new spans / new
                evaluators instead of re-annotating everything.

        Returns:
            Summary of evaluation results. ``num_spans_retrieved`` is the
            window size; per-evaluator ``num_evaluated`` / ``num_skipped``
            report the incremental split; top-level ``num_skipped`` is the
            total (span, evaluator) pairs skipped.
        """
        logger.info(f"Starting span evaluation pipeline for last {hours} hours")

        # Get recent spans
        spans_df = await self.get_recent_spans(hours, operation_name)
        logger.info(f"Retrieved {len(spans_df)} spans")

        if spans_df.empty:
            return {"error": "No spans found"}

        if evaluator_names is None:
            evaluator_names = list(self.reference_free_evaluators.keys()) + [
                "golden_dataset"
            ]

        present_ids = (
            set(spans_df["span_id"].dropna())
            if "span_id" in spans_df.columns
            else set()
        )
        skip_span_ids: dict[str, set[str]] = {}
        if incremental:
            skip_span_ids = await self._already_evaluated_span_ids(
                hours, operation_name, evaluator_names
            )

        # Run evaluations (skipping already-annotated pairs)
        evaluations = await self.evaluate_spans(
            spans_df, evaluator_names, skip_span_ids=skip_span_ids
        )

        # Upload evaluations if requested
        if upload_evaluations:
            await self.upload_evaluations(evaluations)

        # Generate summary — per-evaluator evaluated/skipped split.
        per_eval_skipped = {
            name: len(present_ids & skip_span_ids.get(name, set()))
            for name in evaluator_names
        }
        summary: dict[str, Any] = {
            "num_spans_retrieved": len(spans_df),
            "num_skipped": sum(per_eval_skipped.values()),
            "incremental": incremental,
            "evaluators_run": list(evaluator_names),
            "results": {},
        }

        for eval_name in evaluator_names:
            eval_df = evaluations.get(eval_name, pd.DataFrame())
            summary["results"][eval_name] = {
                "num_evaluated": len(eval_df),
                "num_skipped": per_eval_skipped.get(eval_name, 0),
                "mean_score": (eval_df["score"].mean() if not eval_df.empty else None),
                "score_distribution": (
                    eval_df["label"].value_counts().to_dict()
                    if not eval_df.empty
                    else {}
                ),
            }

        return summary
