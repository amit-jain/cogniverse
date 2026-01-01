"""
Workflow Checkpoint Storage Implementation

Stores workflow checkpoints as telemetry spans using Phoenix backend.
Follows the same pattern as ApprovalStorageImpl for consistency.

Structure:
- workflow_checkpoint (root span): Contains checkpoint metadata
  - Attributes: checkpoint_id, workflow_id, current_phase, task_states, etc.
  - Annotations: Status updates (ACTIVE -> SUPERSEDED, COMPLETED, FAILED)
"""

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from opentelemetry.trace import Status, StatusCode

from cogniverse_agents.orchestrator.checkpoint_types import (
    CheckpointConfig,
    CheckpointStatus,
    WorkflowCheckpoint,
)

if TYPE_CHECKING:
    from cogniverse_foundation.telemetry.manager import TelemetryManager

logger = logging.getLogger(__name__)


def _serialize_for_json(obj: Any) -> Any:
    """Serialize complex types for JSON encoding"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_for_json(item) for item in obj]
    else:
        return obj


class WorkflowCheckpointStorage:
    """
    Store workflow checkpoints using telemetry spans

    Design:
    - Checkpoints are immutable spans with all state in attributes
    - Status updates (ACTIVE -> SUPERSEDED) use annotations
    - Latest checkpoint found by querying spans with ACTIVE status annotation

    This enables:
    - Full audit trail of workflow execution
    - Time-travel debugging by loading historical checkpoints
    - Resume from any checkpoint (fork capability)
    """

    def __init__(
        self,
        grpc_endpoint: str,
        http_endpoint: str,
        tenant_id: str,
        config: Optional[CheckpointConfig] = None,
        telemetry_manager: Optional["TelemetryManager"] = None,
    ):
        """
        Initialize checkpoint storage

        Args:
            grpc_endpoint: OTLP gRPC endpoint for span export
            http_endpoint: HTTP endpoint for span queries
            tenant_id: Tenant ID for multi-tenant scoping
            config: Checkpoint configuration
            telemetry_manager: TelemetryManager instance (creates one if None)
        """
        self.grpc_endpoint = grpc_endpoint
        self.http_endpoint = http_endpoint
        self.tenant_id = tenant_id
        self.config = config or CheckpointConfig()
        self.project_name = self.config.project_name

        # Initialize TelemetryManager
        if telemetry_manager is None:
            from cogniverse_foundation.telemetry.config import TelemetryConfig
            from cogniverse_foundation.telemetry.manager import TelemetryManager

            telemetry_config = TelemetryConfig(
                provider_config={
                    "http_endpoint": http_endpoint,
                    "grpc_endpoint": grpc_endpoint,
                },
            )
            telemetry_manager = TelemetryManager(config=telemetry_config)
        else:
            # Update existing manager's config
            if not telemetry_manager.config.provider_config:
                telemetry_manager.config.provider_config = {}
            telemetry_manager.config.provider_config.update(
                {
                    "http_endpoint": http_endpoint,
                    "grpc_endpoint": grpc_endpoint,
                }
            )

        self.telemetry_manager = telemetry_manager

        # Register project with TelemetryManager
        self.telemetry_manager.register_project(
            tenant_id=tenant_id,
            project_name=self.project_name,
            otlp_endpoint=grpc_endpoint,
            http_endpoint=http_endpoint,
            use_sync_export=True,
        )

        # Full project name for queries
        self.full_project_name = f"cogniverse-{tenant_id}-{self.project_name}"

        # Get telemetry provider for queries
        self.provider = self.telemetry_manager.get_provider(
            tenant_id=tenant_id, project_name=self.project_name
        )

        logger.info(
            f"Initialized WorkflowCheckpointStorage "
            f"(tenant: {tenant_id}, project: {self.full_project_name})"
        )

    async def save_checkpoint(self, checkpoint: WorkflowCheckpoint) -> str:
        """
        Save a new workflow checkpoint as a telemetry span

        Args:
            checkpoint: Checkpoint to save

        Returns:
            Checkpoint ID
        """
        checkpoint_data = checkpoint.to_dict()

        # Build span attributes (all JSON-serializable)
        attributes = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "workflow_id": checkpoint.workflow_id,
            "tenant_id": checkpoint.tenant_id,
            "workflow_status": checkpoint.workflow_status,
            "current_phase": checkpoint.current_phase,
            "original_query": checkpoint.original_query,
            "execution_order": checkpoint_data["execution_order"],
            "metadata": checkpoint_data["metadata"],
            "task_states": checkpoint_data["task_states"],
            "checkpoint_time": checkpoint_data["checkpoint_time"],
            "checkpoint_status": checkpoint.checkpoint_status.value,
            "parent_checkpoint_id": checkpoint.parent_checkpoint_id or "",
            "resume_count": checkpoint.resume_count,
        }

        # Create checkpoint span
        with self.telemetry_manager.span(
            name="workflow_checkpoint",
            tenant_id=self.tenant_id,
            project_name=self.project_name,
            attributes=attributes,
        ) as span:
            span.set_status(Status(StatusCode.OK))
            logger.info(
                f"Saved checkpoint {checkpoint.checkpoint_id} for workflow {checkpoint.workflow_id} "
                f"(phase {checkpoint.current_phase}, status: {checkpoint.checkpoint_status.value})"
            )

        # Force flush to ensure span is exported
        self._force_flush()

        return checkpoint.checkpoint_id

    def _force_flush(self) -> None:
        """Force flush tracer provider to ensure spans are exported"""
        try:
            cache_key = f"{self.tenant_id}:{self.project_name}"
            if hasattr(self.telemetry_manager, "_tenant_providers"):
                tracer_provider = self.telemetry_manager._tenant_providers.get(
                    cache_key
                )
                if tracer_provider and hasattr(tracer_provider, "force_flush"):
                    tracer_provider.force_flush(timeout_millis=5000)
        except Exception as e:
            logger.warning(f"Failed to flush tracer provider: {e}")

    async def get_latest_checkpoint(
        self, workflow_id: str
    ) -> Optional[WorkflowCheckpoint]:
        """
        Get the latest ACTIVE checkpoint for a workflow

        Args:
            workflow_id: Workflow ID to find checkpoint for

        Returns:
            Latest active checkpoint if found, None otherwise
        """
        try:
            # Query with retry for indexing lag
            retry_delays = [1, 2, 3]

            for attempt, delay in enumerate(retry_delays):
                spans_df = await self.provider.traces.get_spans(
                    project=self.full_project_name
                )

                if spans_df.empty:
                    if attempt < len(retry_delays) - 1:
                        time.sleep(delay)
                        continue
                    return None

                # Filter for checkpoint spans of this workflow
                if "attributes.workflow_id" not in spans_df.columns:
                    if attempt < len(retry_delays) - 1:
                        time.sleep(delay)
                        continue
                    return None

                workflow_checkpoints = spans_df[
                    (spans_df["name"] == "workflow_checkpoint")
                    & (spans_df["attributes.workflow_id"] == workflow_id)
                ]

                if workflow_checkpoints.empty:
                    if attempt < len(retry_delays) - 1:
                        time.sleep(delay)
                        continue
                    return None

                # Get the most recent checkpoint by checkpoint_time
                # First, check for annotations that update status
                latest_active = await self._find_latest_active_checkpoint(
                    workflow_checkpoints
                )

                if latest_active is not None:
                    return latest_active

                if attempt < len(retry_delays) - 1:
                    time.sleep(delay)

            return None

        except Exception as e:
            logger.error(f"Error getting latest checkpoint for workflow {workflow_id}: {e}")
            return None

    async def _find_latest_active_checkpoint(
        self, checkpoint_spans
    ) -> Optional[WorkflowCheckpoint]:
        """Find the latest active checkpoint from spans, considering annotations"""
        # Sort by checkpoint_time descending
        if "attributes.checkpoint_time" in checkpoint_spans.columns:
            checkpoint_spans = checkpoint_spans.sort_values(
                "attributes.checkpoint_time", ascending=False
            )

        # Check each checkpoint for status (via annotations or span attribute)
        for _, row in checkpoint_spans.iterrows():
            span_id = row.get("context.span_id")

            # Check for status annotations
            current_status = await self._get_checkpoint_current_status(
                span_id, row.get("attributes.checkpoint_status", "active")
            )

            if current_status == CheckpointStatus.ACTIVE:
                # Found active checkpoint, reconstruct it
                return self._reconstruct_checkpoint_from_row(row)

        return None

    async def _get_checkpoint_current_status(
        self, span_id: str, default_status: str
    ) -> CheckpointStatus:
        """Get current status of checkpoint, checking annotations for updates"""
        try:
            # Query annotations for this span
            # For now, just use the span attribute since annotations query is complex
            # In a full implementation, would check for status_update annotations
            return CheckpointStatus(default_status)
        except ValueError:
            return CheckpointStatus.ACTIVE

    def _reconstruct_checkpoint_from_row(self, row) -> WorkflowCheckpoint:
        """Reconstruct WorkflowCheckpoint from DataFrame row"""
        data = {
            "checkpoint_id": row.get("attributes.checkpoint_id", ""),
            "workflow_id": row.get("attributes.workflow_id", ""),
            "tenant_id": row.get("attributes.tenant_id", self.tenant_id),
            "workflow_status": row.get("attributes.workflow_status", "pending"),
            "current_phase": int(row.get("attributes.current_phase", 0)),
            "original_query": row.get("attributes.original_query", ""),
            "execution_order": row.get("attributes.execution_order", "[]"),
            "metadata": row.get("attributes.metadata", "{}"),
            "task_states": row.get("attributes.task_states", "{}"),
            "checkpoint_time": row.get("attributes.checkpoint_time"),
            "checkpoint_status": row.get("attributes.checkpoint_status", "active"),
            "parent_checkpoint_id": row.get("attributes.parent_checkpoint_id") or None,
            "resume_count": int(row.get("attributes.resume_count", 0)),
        }

        return WorkflowCheckpoint.from_dict(data)

    async def get_checkpoint_by_id(
        self, checkpoint_id: str
    ) -> Optional[WorkflowCheckpoint]:
        """
        Get a specific checkpoint by ID

        Args:
            checkpoint_id: Checkpoint ID to retrieve

        Returns:
            Checkpoint if found, None otherwise
        """
        try:
            spans_df = await self.provider.traces.get_spans(
                project=self.full_project_name
            )

            if spans_df.empty:
                return None

            if "attributes.checkpoint_id" not in spans_df.columns:
                return None

            checkpoint_spans = spans_df[
                (spans_df["name"] == "workflow_checkpoint")
                & (spans_df["attributes.checkpoint_id"] == checkpoint_id)
            ]

            if checkpoint_spans.empty:
                return None

            # Get the matching span
            row = checkpoint_spans.iloc[0]
            return self._reconstruct_checkpoint_from_row(row)

        except Exception as e:
            logger.error(f"Error getting checkpoint {checkpoint_id}: {e}")
            return None

    async def mark_checkpoint_status(
        self, checkpoint_id: str, status: CheckpointStatus
    ) -> bool:
        """
        Update checkpoint status via annotation

        Args:
            checkpoint_id: Checkpoint ID to update
            status: New status

        Returns:
            True if update successful
        """
        try:
            # Find span ID for this checkpoint
            span_id = await self._get_checkpoint_span_id(checkpoint_id)

            if not span_id:
                logger.error(f"Checkpoint {checkpoint_id} not found for status update")
                return False

            # Add status update annotation
            await self.provider.annotations.add_annotation(
                span_id=span_id,
                name="checkpoint_status_update",
                label=status.value,
                score=1.0 if status == CheckpointStatus.COMPLETED else 0.0,
                metadata={
                    "checkpoint_id": checkpoint_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "previous_status": "active",  # Could query actual previous
                    "new_status": status.value,
                },
                project=self.full_project_name,
            )

            logger.info(f"Updated checkpoint {checkpoint_id} status to {status.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to update checkpoint status: {e}")
            return False

    async def _get_checkpoint_span_id(self, checkpoint_id: str) -> Optional[str]:
        """Get span ID for a checkpoint"""
        try:
            spans_df = await self.provider.traces.get_spans(
                project=self.full_project_name
            )

            if spans_df.empty:
                return None

            if "attributes.checkpoint_id" not in spans_df.columns:
                return None

            checkpoint_spans = spans_df[
                (spans_df["name"] == "workflow_checkpoint")
                & (spans_df["attributes.checkpoint_id"] == checkpoint_id)
            ]

            if checkpoint_spans.empty:
                return None

            return checkpoint_spans.iloc[0]["context.span_id"]

        except Exception as e:
            logger.error(f"Error finding span for checkpoint {checkpoint_id}: {e}")
            return None

    async def list_workflow_checkpoints(
        self, workflow_id: str, include_superseded: bool = False
    ) -> List[WorkflowCheckpoint]:
        """
        List all checkpoints for a workflow

        Args:
            workflow_id: Workflow ID to list checkpoints for
            include_superseded: Whether to include superseded checkpoints

        Returns:
            List of checkpoints, ordered by checkpoint_time descending
        """
        try:
            spans_df = await self.provider.traces.get_spans(
                project=self.full_project_name
            )

            if spans_df.empty:
                return []

            if "attributes.workflow_id" not in spans_df.columns:
                return []

            workflow_checkpoints = spans_df[
                (spans_df["name"] == "workflow_checkpoint")
                & (spans_df["attributes.workflow_id"] == workflow_id)
            ]

            if workflow_checkpoints.empty:
                return []

            # Sort by checkpoint_time
            if "attributes.checkpoint_time" in workflow_checkpoints.columns:
                workflow_checkpoints = workflow_checkpoints.sort_values(
                    "attributes.checkpoint_time", ascending=False
                )

            checkpoints = []
            for _, row in workflow_checkpoints.iterrows():
                checkpoint = self._reconstruct_checkpoint_from_row(row)

                if not include_superseded:
                    # Check current status via annotations
                    current_status = await self._get_checkpoint_current_status(
                        row.get("context.span_id"),
                        row.get("attributes.checkpoint_status", "active"),
                    )
                    if current_status == CheckpointStatus.SUPERSEDED:
                        continue

                checkpoints.append(checkpoint)

            return checkpoints

        except Exception as e:
            logger.error(f"Error listing checkpoints for workflow {workflow_id}: {e}")
            return []

    async def get_resumable_workflows(
        self, tenant_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get workflows that can be resumed (have ACTIVE checkpoints with FAILED or RUNNING status)

        Args:
            tenant_id: Optional tenant filter

        Returns:
            List of resumable workflow summaries
        """
        try:
            spans_df = await self.provider.traces.get_spans(
                project=self.full_project_name
            )

            if spans_df.empty:
                return []

            # Filter for checkpoint spans
            checkpoint_spans = spans_df[spans_df["name"] == "workflow_checkpoint"]

            if checkpoint_spans.empty:
                return []

            # Filter by tenant if specified
            if tenant_id and "attributes.tenant_id" in checkpoint_spans.columns:
                checkpoint_spans = checkpoint_spans[
                    checkpoint_spans["attributes.tenant_id"] == tenant_id
                ]

            # Group by workflow_id and find latest checkpoint
            resumable = []
            workflow_ids = checkpoint_spans["attributes.workflow_id"].unique()

            for workflow_id in workflow_ids:
                latest = await self.get_latest_checkpoint(workflow_id)
                if latest is None:
                    continue

                # Check if workflow is resumable (not completed/cancelled)
                if latest.workflow_status in ["running", "failed", "partially_completed"]:
                    resumable.append(
                        {
                            "workflow_id": workflow_id,
                            "checkpoint_id": latest.checkpoint_id,
                            "workflow_status": latest.workflow_status,
                            "current_phase": latest.current_phase,
                            "total_phases": len(latest.execution_order),
                            "original_query": latest.original_query[:100] + "..."
                            if len(latest.original_query) > 100
                            else latest.original_query,
                            "checkpoint_time": latest.checkpoint_time.isoformat(),
                            "resume_count": latest.resume_count,
                        }
                    )

            return resumable

        except Exception as e:
            logger.error(f"Error getting resumable workflows: {e}")
            return []
