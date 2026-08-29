"""
Orchestration Evaluator for Workflow Optimization

Extracts orchestration workflow execution data from telemetry spans
and feeds them to WorkflowIntelligence for continuous learning.
"""

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict

from cogniverse_agents.workflow.intelligence import WorkflowIntelligence
from cogniverse_foundation.telemetry.config import SPAN_NAME_ORCHESTRATION
from cogniverse_foundation.telemetry.manager import get_telemetry_manager
from cogniverse_sdk.interfaces.workflow_store import WorkflowExecution

if TYPE_CHECKING:
    from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)

_SPAN_QUERY_ATTEMPTS = 3
_SPAN_QUERY_TIMEOUT_S = 30.0
_SPAN_QUERY_RETRY_DELAY_S = 0.25


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
        self,
        workflow_intelligence: WorkflowIntelligence,
        tenant_id: str,
    ):
        """
        Initialize orchestration span evaluator

        Args:
            workflow_intelligence: Workflow optimizer to feed experiences to
            tenant_id: Tenant identifier for multi-tenant projects
        """
        from cogniverse_core.common.tenant_utils import canonical_tenant_id

        self.workflow_intelligence = workflow_intelligence
        # The runtime writes orchestration spans under the canonical tenant
        # project; a caller passing a raw id (e.g. a dashboard tab's
        # current_tenant) must resolve the SAME provider + project scope, or the
        # evaluator queries an empty project and reports no orchestration spans.
        self.tenant_id = canonical_tenant_id(tenant_id)

        # Get telemetry manager and use its config (shared singleton config)
        telemetry_manager = get_telemetry_manager()
        self.telemetry_config = telemetry_manager.config
        self.provider: "TelemetryProvider" = telemetry_manager.get_provider(
            tenant_id=self.tenant_id
        )

        self.project_name = self.telemetry_config.get_project_name(self.tenant_id)

        self._evaluation_cursor: tuple[datetime, str] | None = None
        self._evaluation_lock = asyncio.Lock()

        logger.info(
            f"🔧 Initialized OrchestrationEvaluator for tenant '{self.tenant_id}' "
            f"(project: {self.project_name})"
        )

    async def evaluate_orchestration_spans(
        self,
        lookback_hours: float = 1,
        batch_size: int = 50,
        evaluation_end_time: datetime | None = None,
    ) -> Dict[str, Any]:
        """Evaluate one incremental span window at a time."""
        async with self._evaluation_lock:
            return await self._evaluate_orchestration_spans(
                lookback_hours=lookback_hours,
                batch_size=batch_size,
                evaluation_end_time=evaluation_end_time,
            )

    async def _evaluate_orchestration_spans(
        self,
        lookback_hours: float = 1,
        batch_size: int = 50,
        evaluation_end_time: datetime | None = None,
    ) -> Dict[str, Any]:
        """
        Evaluate orchestration spans from the last N hours

        Args:
            lookback_hours: How far back to look for spans
            batch_size: Maximum spans to process in one batch
            evaluation_end_time: Fixed upper bound reused while draining batches

        Returns:
            Evaluation results with workflow executions processed
        """
        logger.info(
            f"🔍 Evaluating orchestration spans from last {lookback_hours} hours "
            f"(project: {self.project_name})"
        )

        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive int")
        end_time = evaluation_end_time or datetime.now(timezone.utc)
        if end_time.utcoffset() is None:
            raise ValueError("evaluation_end_time must be timezone-aware")
        end_time = end_time.astimezone(timezone.utc)
        if self._evaluation_cursor is not None:
            start_time = min(self._evaluation_cursor[0], end_time)
        else:
            start_time = end_time - timedelta(hours=lookback_hours)

        query_error: Exception | None = None
        for attempt in range(1, _SPAN_QUERY_ATTEMPTS + 1):
            try:
                spans_df = await asyncio.wait_for(
                    self.provider.traces.get_all_spans(
                        project=self.project_name,
                        start_time=start_time,
                        end_time=end_time,
                        filters={"name": SPAN_NAME_ORCHESTRATION},
                    ),
                    timeout=_SPAN_QUERY_TIMEOUT_S,
                )
                break
            except Exception as exc:
                query_error = exc
                logger.warning(
                    "Orchestration telemetry query attempt %d/%d failed: %s",
                    attempt,
                    _SPAN_QUERY_ATTEMPTS,
                    exc,
                )
                if attempt < _SPAN_QUERY_ATTEMPTS:
                    await asyncio.sleep(_SPAN_QUERY_RETRY_DELAY_S)
        else:
            logger.error("❌ Error querying telemetry spans: %s", query_error)
            raise RuntimeError(
                "Failed to query orchestration telemetry"
            ) from query_error

        if spans_df.empty:
            logger.info("📭 No orchestration spans found in time range")
            return {
                "spans_processed": 0,
                "workflows_extracted": 0,
                "errors": [],
                "evaluation_time": end_time.isoformat(),
                "has_more": False,
            }

        # Filter for orchestration spans only
        orchestration_spans = spans_df[spans_df["name"] == SPAN_NAME_ORCHESTRATION]

        logger.info(f"📊 Found {len(orchestration_spans)} orchestration spans")

        try:
            ordered_spans = sorted(
                (
                    (self._span_cursor(span_row), span_row)
                    for _, span_row in orchestration_spans.iterrows()
                ),
                key=lambda entry: entry[0],
            )
        except ValueError as exc:
            logger.error("❌ Invalid orchestration span ordering data: %s", exc)
            raise ValueError("Invalid orchestration span ordering data") from exc

        if self._evaluation_cursor is not None:
            ordered_spans = [
                entry for entry in ordered_spans if entry[0] > self._evaluation_cursor
            ]
        batch = ordered_spans[:batch_size]
        has_more = len(ordered_spans) > batch_size

        prepared = []
        for cursor, span_row in batch:
            prepared.append(
                (
                    cursor,
                    self._extract_workflow_execution(span_row, self.tenant_id),
                )
            )

        workflows_extracted = 0
        for cursor, workflow_execution in prepared:
            try:
                await self.workflow_intelligence.record_execution(workflow_execution)
            except Exception as e:
                span_id = cursor[1]
                logger.error(f"❌ Error recording span {span_id}: {e}")
                raise RuntimeError(
                    f"Failed to record orchestration span {span_id}"
                ) from e
            workflows_extracted += 1
            self._evaluation_cursor = cursor

        result = {
            "spans_processed": len(prepared),
            "workflows_extracted": workflows_extracted,
            "errors": [],
            "evaluation_time": end_time.isoformat(),
            "has_more": has_more,
        }

        logger.info(
            f"✅ Orchestration evaluation complete: {workflows_extracted} workflows extracted"
        )

        return result

    @staticmethod
    def _span_cursor(span_row) -> tuple[datetime, str]:
        start_time = span_row.get("start_time")
        if not isinstance(start_time, datetime) or start_time.utcoffset() is None:
            raise ValueError(
                "orchestration span start_time must be a timezone-aware datetime"
            )
        span_id = span_row.get("context.span_id")
        if not isinstance(span_id, str) or not span_id:
            raise ValueError("orchestration span_id must be a non-empty str")
        return start_time.astimezone(timezone.utc), span_id

    @staticmethod
    def _extract_workflow_execution(
        span_row,
        tenant_id: str,
    ) -> WorkflowExecution:
        """
        Extract WorkflowExecution from Phoenix span data

        Extracts:
        - workflow_id, query, query_type
        - orchestration_pattern, agent_sequence
        - execution_time, individual agent times
        - success, error details
        - parallel_efficiency, confidence_score
        """
        from cogniverse_foundation.telemetry.span_contract import read_span_io

        span_io = read_span_io(span_row)
        output = span_io["output"] if isinstance(span_io["output"], dict) else {}

        def require_output(field: str) -> Any:
            if field not in output:
                raise ValueError(f"orchestration span requires observed field {field}")
            return output[field]

        workflow_id = require_output("workflow_id")
        if not isinstance(workflow_id, str) or not workflow_id.strip():
            raise ValueError("orchestration span workflow_id must be a non-empty str")

        query = span_io["input"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("orchestration span requires observed field query")

        orchestration_pattern = require_output("pattern")
        if not isinstance(orchestration_pattern, str) or not orchestration_pattern:
            raise ValueError("orchestration span pattern must be a non-empty str")

        agent_sequence = require_output("agent_sequence")
        if not isinstance(agent_sequence, list) or any(
            not isinstance(agent, str) or not agent for agent in agent_sequence
        ):
            raise ValueError("orchestration span agent_sequence must be a list of str")
        if len(agent_sequence) != len(set(agent_sequence)):
            raise ValueError(
                "orchestration span agent_sequence must not contain duplicates"
            )

        execution_order = output.get("execution_order", [])
        if not isinstance(execution_order, list) or any(
            not isinstance(agent, str) or not agent for agent in execution_order
        ):
            raise ValueError("orchestration span execution_order must be a list of str")

        execution_time = require_output("execution_time")
        if (
            type(execution_time) is not float
            or not math.isfinite(execution_time)
            or execution_time < 0.0
        ):
            raise ValueError(
                "orchestration span execution_time must be a non-negative finite float"
            )

        success = require_output("success")
        if type(success) is not bool:
            raise ValueError("orchestration span success must be a bool")
        status_code = span_row.get("status_code")
        if success:
            if not agent_sequence:
                raise ValueError(
                    "successful orchestration span agent_sequence must be non-empty"
                )
            if status_code == "ERROR":
                raise ValueError(
                    "successful orchestration span cannot have ERROR status"
                )
            error_details = None
        else:
            if status_code != "ERROR":
                raise ValueError("failed orchestration span must have ERROR status")
            error_details = output.get("error_summary")
            if not isinstance(error_details, str) or not error_details.strip():
                raise ValueError(
                    "failed orchestration span error_summary must be a non-empty str"
                )
            if len(error_details) > 512:
                raise ValueError(
                    "failed orchestration span error_summary cannot exceed 512 characters"
                )

        tasks_completed = require_output("tasks_completed")
        if type(tasks_completed) is not int or tasks_completed < 0:
            raise ValueError(
                "orchestration span tasks_completed must be a non-negative int"
            )
        task_count = len(agent_sequence)
        if tasks_completed > task_count:
            raise ValueError(
                "orchestration span tasks_completed cannot exceed agent_sequence"
            )

        agent_observations = None
        if "agent_observations" in output:
            agent_observations = OrchestrationEvaluator._extract_agent_observations(
                output["agent_observations"],
                agent_sequence,
            )

        has_agent_times = bool(output.get("agent_times"))
        parallel_efficiency = OrchestrationEvaluator._compute_parallel_efficiency(
            orchestration_pattern, output, execution_time
        )
        parallel_semantics = (
            "observed_parallel_efficiency"
            if orchestration_pattern != "parallel" or has_agent_times
            else "unobserved_zero_sentinel"
        )

        if "confidence" in output:
            confidence_score = output["confidence"]
            if (
                type(confidence_score) is not float
                or not math.isfinite(confidence_score)
                or not 0.0 <= confidence_score <= 1.0
            ):
                raise ValueError(
                    "orchestration span confidence must be a finite float between 0 and 1"
                )
            confidence_semantics = "observed_confidence_score"
        else:
            confidence_score = 0.0
            confidence_semantics = "unobserved_zero_sentinel"

        query_type = OrchestrationEvaluator._classify_query_type(
            query, orchestration_pattern
        )

        metadata = {
            "orchestration_pattern": orchestration_pattern,
            "execution_order": execution_order,
            "tasks_completed": tasks_completed,
            "span_id": span_row.get("context.span_id"),
            "tenant_id": tenant_id,
            "_outcome_metadata": {
                "observed": True,
                "required_field_semantics": {
                    "execution_time": "observed_duration_seconds",
                    "success": "observed_execution_outcome",
                    "parallel_efficiency": parallel_semantics,
                    "confidence_score": confidence_semantics,
                },
            },
        }
        if agent_observations is not None:
            metadata["agent_observations"] = agent_observations

        return WorkflowExecution(
            workflow_id=workflow_id,
            query=query,
            query_type=query_type,
            execution_time=execution_time,
            success=success,
            agent_sequence=agent_sequence,
            task_count=task_count,
            parallel_efficiency=parallel_efficiency,
            confidence_score=confidence_score,
            user_satisfaction=None,
            error_details=error_details,
            metadata=metadata,
        )

    @staticmethod
    def _extract_agent_observations(
        raw_observations: Any,
        agent_sequence: list[str],
    ) -> list[Dict[str, Any]]:
        if not isinstance(raw_observations, list):
            raise ValueError("orchestration span agent_observations must be a list")
        allowed_agents = set(agent_sequence)
        observations = []
        for raw in raw_observations:
            if not isinstance(raw, dict):
                raise ValueError("orchestration span agent observation must be a dict")
            required = {"agent_name", "execution_time", "success"}
            allowed = required | {"confidence"}
            if set(raw) - allowed or not required <= set(raw):
                raise ValueError(
                    "orchestration span agent observation must contain agent_name, "
                    "execution_time, success, and optional confidence"
                )
            agent_name = raw["agent_name"]
            if not isinstance(agent_name, str) or agent_name not in allowed_agents:
                raise ValueError(
                    "orchestration span agent observation references an agent "
                    "outside agent_sequence"
                )
            execution_time = raw["execution_time"]
            if (
                type(execution_time) is not float
                or not math.isfinite(execution_time)
                or execution_time < 0.0
            ):
                raise ValueError(
                    "orchestration span agent observation execution_time must be "
                    "a non-negative finite float"
                )
            if type(raw["success"]) is not bool:
                raise ValueError(
                    "orchestration span agent observation success must be a bool"
                )
            if "confidence" in raw:
                confidence = raw["confidence"]
                if (
                    type(confidence) is not float
                    or not math.isfinite(confidence)
                    or not 0.0 <= confidence <= 1.0
                ):
                    raise ValueError(
                        "orchestration span agent observation confidence must be "
                        "a finite float between 0 and 1"
                    )
            observations.append(dict(raw))
        return observations

    @staticmethod
    def _compute_parallel_efficiency(
        pattern: str, attributes: Dict, total_time: float
    ) -> float:
        """
        Compute normalized parallel efficiency in the inclusive range 0..1.
        """
        if pattern != "parallel":
            return 0.0

        # Extract individual agent times if available
        # Format: "agent1:1.2,agent2:1.5,agent3:0.8"
        agent_times_str = attributes.get("agent_times")
        if not agent_times_str:
            return 0.0
        if not isinstance(agent_times_str, str):
            raise ValueError("orchestration span agent_times must be a str")
        if total_time <= 0.0:
            raise ValueError(
                "orchestration span execution_time must be positive when agent_times are observed"
            )

        agent_times = []
        for entry in agent_times_str.split(","):
            parts = entry.split(":", maxsplit=1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                raise ValueError("orchestration span agent_times is malformed")
            agent_time = float(parts[1])
            if not math.isfinite(agent_time) or agent_time < 0.0:
                raise ValueError(
                    "orchestration span agent_times must contain non-negative finite values"
                )
            agent_times.append(agent_time)

        efficiency = sum(agent_times) / (len(agent_times) * total_time)
        if efficiency > 1.0:
            raise ValueError(
                "orchestration span agent_times exceed the observed execution_time"
            )
        return efficiency

    @staticmethod
    def _classify_query_type(query: str, pattern: str) -> str:
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
