"""
Orchestration Evaluator for Workflow Optimization

Extracts orchestration workflow execution data from telemetry spans
and feeds them to WorkflowIntelligence for continuous learning.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

from cogniverse_foundation.telemetry.config import SPAN_NAME_ORCHESTRATION
from cogniverse_foundation.telemetry.manager import get_telemetry_manager

from cogniverse_agents.workflow_intelligence import (
    WorkflowExecution,
    WorkflowIntelligence,
)

if TYPE_CHECKING:
    from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


class OrchestrationEvaluator:
    """
    Evaluates orchestration spans to extract workflow execution data

    This class:
    1. Queries cogniverse.orchestration spans from telemetry
    2. Extracts workflow execution metrics (pattern, agents, timing, success)
    3. Computes quality metrics (parallel efficiency, agent performance)
    4. Feeds WorkflowExecution records to WorkflowIntelligence
    5. Enables learning from real orchestration outcomes
    """

    def __init__(
        self, workflow_intelligence: WorkflowIntelligence, tenant_id: str = "default"
    ):
        """
        Initialize orchestration span evaluator

        Args:
            workflow_intelligence: Workflow optimizer to feed experiences to
            tenant_id: Tenant identifier for multi-tenant projects
        """
        self.workflow_intelligence = workflow_intelligence
        self.tenant_id = tenant_id

        # Get telemetry manager and use its config (shared singleton config)
        telemetry_manager = get_telemetry_manager()
        self.telemetry_config = telemetry_manager.config
        self.provider: "TelemetryProvider" = telemetry_manager.get_provider(
            tenant_id=tenant_id
        )

        # Get project name for orchestration spans
        self.project_name = self.telemetry_config.get_project_name(
            tenant_id, service="cogniverse.orchestration"
        )

        # Track processed spans to avoid duplicates
        self._processed_span_ids = set()
        self._last_evaluation_time = datetime.now()

        logger.info(
            f"🔧 Initialized OrchestrationEvaluator for tenant '{tenant_id}' "
            f"(project: {self.project_name})"
        )

    async def evaluate_orchestration_spans(
        self, lookback_hours: int = 1, batch_size: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluate orchestration spans from the last N hours

        Args:
            lookback_hours: How far back to look for spans
            batch_size: Maximum spans to process in one batch

        Returns:
            Evaluation results with workflow executions processed
        """
        logger.info(
            f"🔍 Evaluating orchestration spans from last {lookback_hours} hours "
            f"(project: {self.project_name})"
        )

        # Query cogniverse.orchestration spans
        # Use UTC timezone-aware datetime to avoid timezone confusion
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=lookback_hours)

        try:
            spans_df = await self.provider.traces.get_spans(
                project=self.project_name,
                start_time=start_time,
                end_time=end_time,
                limit=10000,
            )
        except Exception as e:
            logger.error(f"❌ Error querying telemetry spans: {e}")
            return {
                "spans_processed": 0,
                "workflows_extracted": 0,
                "errors": [str(e)],
            }

        if spans_df.empty:
            logger.info("📭 No orchestration spans found in time range")
            return {
                "spans_processed": 0,
                "workflows_extracted": 0,
            }

        # Filter for orchestration spans only
        orchestration_spans = spans_df[spans_df["name"] == SPAN_NAME_ORCHESTRATION]

        logger.info(f"📊 Found {len(orchestration_spans)} orchestration spans")

        # Extract workflow executions
        workflows_extracted = 0
        errors = []

        for _, span_row in orchestration_spans.head(batch_size).iterrows():
            try:
                # Skip if already processed
                span_id = span_row.get("context.span_id")
                if span_id in self._processed_span_ids:
                    continue

                # Extract workflow execution data
                workflow_execution = self._extract_workflow_execution(span_row)
                if not workflow_execution:
                    continue

                # Feed to WorkflowIntelligence
                await self.workflow_intelligence.record_execution(workflow_execution)

                self._processed_span_ids.add(span_id)
                workflows_extracted += 1

            except Exception as e:
                logger.error(f"❌ Error processing span {span_id}: {e}")
                errors.append(str(e))

        self._last_evaluation_time = end_time

        result = {
            "spans_processed": len(orchestration_spans),
            "workflows_extracted": workflows_extracted,
            "errors": errors,
            "evaluation_time": end_time.isoformat(),
        }

        logger.info(
            f"✅ Orchestration evaluation complete: {workflows_extracted} workflows extracted"
        )

        return result

    def _extract_workflow_execution(self, span_row) -> Optional[WorkflowExecution]:
        """
        Extract WorkflowExecution from Phoenix span data

        Extracts:
        - workflow_id, query, query_type
        - orchestration_pattern, agent_sequence
        - execution_time, individual agent times
        - success, error details
        - parallel_efficiency, confidence_score
        """
        try:
            attributes = span_row.get("attributes", {})

            # Extract core workflow data
            workflow_id = attributes.get("orchestration.workflow_id")
            query = attributes.get("orchestration.query")

            if not workflow_id or not query:
                logger.warning("Span missing workflow_id or query, skipping")
                return None

            # Extract orchestration details
            orchestration_pattern = attributes.get("orchestration.pattern")
            agents_str = attributes.get("orchestration.agents_used", "")
            agent_sequence = agents_str.split(",") if agents_str else []

            execution_order_str = attributes.get("orchestration.execution_order", "")
            execution_order = (
                execution_order_str.split(",") if execution_order_str else []
            )

            # Extract timing
            execution_time = float(attributes.get("orchestration.execution_time", 0.0))

            # Extract success/failure
            status_code = span_row.get("status_code", "OK")
            success = status_code == "OK"
            error_details = span_row.get("status_message") if not success else None

            # Extract task metrics
            task_count = int(attributes.get("orchestration.tasks_completed", 0))

            # Compute parallel efficiency (if parallel pattern)
            parallel_efficiency = self._compute_parallel_efficiency(
                orchestration_pattern, attributes, execution_time
            )

            # Extract confidence (from routing decision that triggered orchestration)
            confidence_score = float(attributes.get("routing.confidence", 0.0))

            # Classify query type for pattern learning
            query_type = self._classify_query_type(query, orchestration_pattern)

            # Create WorkflowExecution
            workflow_execution = WorkflowExecution(
                workflow_id=workflow_id,
                query=query,
                query_type=query_type,
                execution_time=execution_time,
                success=success,
                agent_sequence=agent_sequence or execution_order,
                task_count=task_count,
                parallel_efficiency=parallel_efficiency,
                confidence_score=confidence_score,
                error_details=error_details,
                metadata={
                    "orchestration_pattern": orchestration_pattern,
                    "execution_order": execution_order,
                    "span_id": span_row.get("context.span_id"),
                    "tenant_id": self.tenant_id,
                },
            )

            return workflow_execution

        except Exception as e:
            logger.error(f"Error extracting workflow execution: {e}")
            return None

    def _compute_parallel_efficiency(
        self, pattern: str, attributes: Dict, total_time: float
    ) -> float:
        """
        Compute parallel efficiency metric

        Parallel efficiency = (sum of individual agent times) / total_time
        Ideal = number_of_agents (perfect parallelization)
        """
        if pattern != "parallel":
            return 0.0

        # Extract individual agent times if available
        # Format: "agent1:1.2,agent2:1.5,agent3:0.8"
        agent_times_str = attributes.get("orchestration.agent_times", "")
        if not agent_times_str:
            return 0.0

        try:
            agent_times = []
            for entry in agent_times_str.split(","):
                if ":" in entry:
                    _, time_str = entry.split(":")
                    agent_times.append(float(time_str))

            if not agent_times or total_time == 0:
                return 0.0

            # Efficiency = sum(individual_times) / total_time
            # If perfectly parallel: 3 agents @ 1s each = 3s work / 1s total = 3.0
            efficiency = sum(agent_times) / total_time
            return min(efficiency, len(agent_times))  # Cap at number of agents

        except Exception as e:
            logger.warning(f"Error computing parallel efficiency: {e}")
            return 0.0

    def _classify_query_type(self, query: str, pattern: str) -> str:
        """
        Classify query type for pattern learning

        Returns:
            Query type string (e.g., "multi_modal_analysis", "sequential_report")
        """
        query_lower = query.lower()

        # Multi-modal patterns
        if any(
            word in query_lower for word in ["videos and documents", "images and text"]
        ):
            return "multi_modal_search"

        # Analysis/report patterns
        if any(word in query_lower for word in ["detailed", "analysis", "report"]):
            if pattern == "sequential":
                return "sequential_report"
            return "detailed_analysis"

        # Summary patterns
        if any(word in query_lower for word in ["summarize", "summary", "overview"]):
            return "summarization"

        # Default based on pattern
        return f"{pattern}_query"
