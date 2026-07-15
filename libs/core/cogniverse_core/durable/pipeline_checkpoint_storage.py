"""Telemetry-span persistence for stage-pipeline checkpoints.

Reuses the same Phoenix-span mechanism as the orchestrator's checkpoint
store — save each checkpoint as an immutable ``pipeline_checkpoint`` span,
find the latest by ``workflow_id`` (with a short retry for span-indexing
lag), and record status transitions as span annotations. Reads are hardened:
a backend outage raises rather than reading as "no checkpoint" (which would
silently restart a long-running workflow from scratch).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from opentelemetry.trace import Status, StatusCode

from cogniverse_core.durable.pipeline_checkpoint import (
    PipelineCheckpoint,
    PipelineCheckpointConfig,
    PipelineCheckpointStatus,
)

if TYPE_CHECKING:
    from cogniverse_foundation.telemetry.manager import TelemetryManager

logger = logging.getLogger(__name__)

_SPAN_NAME = "pipeline_checkpoint"
_STATUS_ANNOTATION = "pipeline_checkpoint_status"


class PipelineCheckpointStorage:
    """Persist ``PipelineCheckpoint`` records as telemetry spans."""

    def __init__(
        self,
        grpc_endpoint: str,
        http_endpoint: str,
        tenant_id: str,
        config: Optional[PipelineCheckpointConfig] = None,
        telemetry_manager: Optional["TelemetryManager"] = None,
    ):
        self.grpc_endpoint = grpc_endpoint
        self.http_endpoint = http_endpoint
        self.tenant_id = tenant_id
        self.config = config or PipelineCheckpointConfig()
        self.project_name = self.config.project_name

        if telemetry_manager is None:
            from cogniverse_foundation.telemetry.config import TelemetryConfig
            from cogniverse_foundation.telemetry.manager import TelemetryManager

            telemetry_manager = TelemetryManager(
                config=TelemetryConfig(
                    provider_config={
                        "http_endpoint": http_endpoint,
                        "grpc_endpoint": grpc_endpoint,
                    },
                )
            )
        else:
            if not telemetry_manager.config.provider_config:
                telemetry_manager.config.provider_config = {}
            telemetry_manager.config.provider_config.update(
                {"http_endpoint": http_endpoint, "grpc_endpoint": grpc_endpoint}
            )

        self.telemetry_manager = telemetry_manager
        self.telemetry_manager.register_project(
            tenant_id=tenant_id,
            project_name=self.project_name,
            otlp_endpoint=grpc_endpoint,
            http_endpoint=http_endpoint,
            use_sync_export=True,
        )
        self.full_project_name = f"cogniverse-{tenant_id}-{self.project_name}"
        self.provider = self.telemetry_manager.get_provider(
            tenant_id=tenant_id, project_name=self.project_name
        )
        logger.info(
            "Initialized PipelineCheckpointStorage (tenant: %s, project: %s)",
            tenant_id,
            self.full_project_name,
        )

    async def save_checkpoint(self, checkpoint: PipelineCheckpoint) -> str:
        """Persist a checkpoint as a ``pipeline_checkpoint`` span."""
        with self.telemetry_manager.span(
            name=_SPAN_NAME,
            tenant_id=self.tenant_id,
            project_name=self.project_name,
            attributes=checkpoint.to_dict(),
        ) as span:
            span.set_status(Status(StatusCode.OK))

        # The exporter blocks up to 5s; run it off the event loop.
        await asyncio.to_thread(self._force_flush)
        logger.info(
            "Saved pipeline checkpoint %s (workflow %s, phase %d/%d)",
            checkpoint.checkpoint_id,
            checkpoint.workflow_id,
            checkpoint.phase_index,
            len(checkpoint.phases),
        )
        return checkpoint.checkpoint_id

    def _force_flush(self) -> None:
        try:
            cache_key = f"{self.tenant_id}:{self.project_name}"
            if hasattr(self.telemetry_manager, "_tenant_providers"):
                tracer_provider = self.telemetry_manager._tenant_providers.get(
                    cache_key
                )
                if tracer_provider and hasattr(tracer_provider, "force_flush"):
                    tracer_provider.force_flush(timeout_millis=5000)
        except Exception as exc:
            logger.warning("Failed to flush tracer provider: %r", exc)

    async def get_latest_checkpoint(
        self, workflow_id: str
    ) -> Optional[PipelineCheckpoint]:
        """The most recent checkpoint for ``workflow_id`` (status annotation wins)."""
        try:
            retry_delays = [1, 2, 3]
            for attempt, delay in enumerate(retry_delays):
                spans_df = await self.provider.traces.get_spans(
                    project=self.full_project_name,
                    filters={"name": _SPAN_NAME},
                )
                if spans_df.empty or "attributes.workflow_id" not in spans_df.columns:
                    if attempt < len(retry_delays) - 1:
                        await asyncio.sleep(delay)
                        continue
                    return None

                rows = spans_df[
                    (spans_df["name"] == _SPAN_NAME)
                    & (spans_df["attributes.workflow_id"] == workflow_id)
                ]
                if rows.empty:
                    if attempt < len(retry_delays) - 1:
                        await asyncio.sleep(delay)
                        continue
                    return None

                if "attributes.created_at" in rows.columns:
                    rows = rows.sort_values("attributes.created_at", ascending=False)
                row = rows.iloc[0]
                checkpoint = self._reconstruct_from_row(row)
                checkpoint.status = await self._get_current_status(
                    row.get("context.span_id"), checkpoint.status
                )
                return checkpoint

            return None
        except Exception as exc:
            # A backend failure is not "no checkpoint" — returning None here
            # would silently restart the workflow from scratch.
            logger.error(
                "Error getting latest pipeline checkpoint for %s: %r", workflow_id, exc
            )
            raise

    async def _get_current_status(self, span_id, default_status: str) -> str:
        """Latest status label from annotations, else the span's own status."""
        import pandas as pd

        try:
            spans_df = pd.DataFrame({"context.span_id": [span_id]})
            annotations = await self.provider.annotations.get_annotations(
                spans_df=spans_df,
                project=self.full_project_name,
                annotation_names=[_STATUS_ANNOTATION],
            )
        except Exception as exc:
            logger.warning(
                "pipeline checkpoint %s status-annotation read failed: %r",
                span_id,
                exc,
            )
            return default_status

        if annotations is not None and not annotations.empty:
            if "created_at" in annotations.columns:
                annotations = annotations.sort_values("created_at")
            label = annotations.iloc[-1].get("result.label")
            if label:
                return str(label)
        return default_status

    def _reconstruct_from_row(self, row) -> PipelineCheckpoint:
        data = {
            "checkpoint_id": row.get("attributes.checkpoint_id", ""),
            "workflow_id": row.get("attributes.workflow_id", ""),
            "tenant_id": row.get("attributes.tenant_id", self.tenant_id),
            "status": row.get("attributes.status", "active"),
            "phases": row.get("attributes.phases", "[]"),
            "phase_index": int(row.get("attributes.phase_index", 0)),
            "completed_units": row.get("attributes.completed_units", "{}"),
            "cursor": row.get("attributes.cursor", ""),
            "metadata": row.get("attributes.metadata", "{}"),
            "created_at": row.get("attributes.created_at"),
            "resume_count": int(row.get("attributes.resume_count", 0)),
        }
        return PipelineCheckpoint.from_dict(data)

    async def mark_status(
        self, checkpoint_id: str, status: PipelineCheckpointStatus
    ) -> bool:
        """Record a status transition as a span annotation."""
        try:
            span_id = await self._get_span_id(checkpoint_id)
            if not span_id:
                logger.error(
                    "Pipeline checkpoint %s not found for status update", checkpoint_id
                )
                return False
            await self.provider.annotations.add_annotation(
                span_id=span_id,
                name=_STATUS_ANNOTATION,
                label=status.value,
                score=1.0 if status == PipelineCheckpointStatus.COMPLETED else 0.0,
                metadata={
                    "checkpoint_id": checkpoint_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "new_status": status.value,
                },
                project=self.full_project_name,
            )
            logger.info(
                "Marked pipeline checkpoint %s status=%s", checkpoint_id, status.value
            )
            return True
        except Exception as exc:
            logger.error("Failed to update pipeline checkpoint status: %r", exc)
            raise

    async def _get_span_id(self, checkpoint_id: str) -> Optional[str]:
        try:
            spans_df = await self.provider.traces.get_spans(
                project=self.full_project_name,
                filters={"name": _SPAN_NAME},
            )
            if spans_df.empty or "attributes.checkpoint_id" not in spans_df.columns:
                return None
            rows = spans_df[
                (spans_df["name"] == _SPAN_NAME)
                & (spans_df["attributes.checkpoint_id"] == checkpoint_id)
            ]
            if rows.empty:
                return None
            return rows.iloc[0]["context.span_id"]
        except Exception as exc:
            logger.error(
                "Error finding span for pipeline checkpoint %s: %r", checkpoint_id, exc
            )
            raise
