"""
Approval Storage Implementation

Stores approval data as telemetry spans with annotations using pluggable provider.
Enables approval workflow tracing and analysis.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd
from opentelemetry.trace import Status, StatusCode

from cogniverse_core.approval.interfaces import (
    ApprovalBatch,
    ApprovalStatus,
    ApprovalStorage,
    ReviewDecision,
    ReviewItem,
)

if TYPE_CHECKING:
    from cogniverse_foundation.telemetry.manager import TelemetryManager

logger = logging.getLogger(__name__)


def _serialize_for_json(obj: Any) -> Any:
    """
    Serialize complex types for JSON encoding

    Handles datetime objects by converting to ISO format strings.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_for_json(item) for item in obj]
    else:
        return obj


class ApprovalStorageImpl(ApprovalStorage):
    """
    Store approval data using telemetry provider as spans

    Structure:
    - approval_batch (root span): Contains batch metadata
      - approval_item (child span): One per review item
        - Attributes: item_id, confidence, status, data
        - Annotations: Human decisions with feedback

    Benefits:
    - Integrated with telemetry infrastructure (Phoenix, LangSmith, etc.)
    - Trace approval workflows alongside optimization
    - Query and analyze approval patterns
    - No additional database needed
    - Provider-agnostic implementation
    """

    def __init__(
        self,
        grpc_endpoint: str,
        http_endpoint: str,
        tenant_id: str,
        telemetry_manager: Optional["TelemetryManager"] = None,
    ):
        """
        Initialize approval storage for synthetic data workflow

        Args:
            grpc_endpoint: OTLP gRPC endpoint for span export (e.g., "http://localhost:4317")
            http_endpoint: HTTP endpoint for span queries (e.g., "http://localhost:6006")
            tenant_id: Tenant ID for multi-tenant project scoping
            telemetry_manager: TelemetryManager instance (if None, creates one)
        """
        from cogniverse_core.common.tenant_utils import canonical_tenant_id

        # Runtime writers register approval spans under the canonical tenant;
        # register, name, and query the same scope regardless of spelling.
        tenant_id = canonical_tenant_id(tenant_id)

        self.tenant_id = tenant_id
        self.project_name = "synthetic_data"

        # Use TelemetryManager for creating spans with proper tenant scoping
        if telemetry_manager is None:
            from cogniverse_foundation.telemetry.config import TelemetryConfig
            from cogniverse_foundation.telemetry.manager import TelemetryManager

            # Configure telemetry (provider auto-discovered via registry)
            config = TelemetryConfig(
                provider_config={
                    "http_endpoint": http_endpoint,
                    "grpc_endpoint": grpc_endpoint,
                },
            )
            telemetry_manager = TelemetryManager(config=config)
        else:
            # Update existing manager's config with endpoint settings
            if not telemetry_manager.config.provider_config:
                telemetry_manager.config.provider_config = {}
            telemetry_manager.config.provider_config.update(
                {
                    "http_endpoint": http_endpoint,
                    "grpc_endpoint": grpc_endpoint,
                }
            )

        self.telemetry_manager = telemetry_manager

        # Register project with TelemetryManager using both endpoints
        self.telemetry_manager.register_project(
            tenant_id=tenant_id,
            project_name=self.project_name,
            otlp_endpoint=grpc_endpoint,
            http_endpoint=http_endpoint,
            use_sync_export=True,  # Use sync export for tests
        )

        # Compute full project name using TelemetryManager logic
        # Format: cogniverse-{tenant_id}-{project_name}
        self.full_project_name = f"cogniverse-{tenant_id}-{self.project_name}"

        # Get telemetry provider for querying spans/annotations/datasets
        # Pass project_name so provider can get http_endpoint from registered project config
        self.provider = self.telemetry_manager.get_provider(
            tenant_id=tenant_id, project_name=self.project_name
        )

        logger.info(
            f"Initialized ApprovalStorageImpl "
            f"(tenant: {tenant_id}, project: {self.full_project_name}, "
            f"grpc: {grpc_endpoint}, http: {http_endpoint}, "
            f"provider: {self.provider.name})"
        )

    async def save_batch(self, batch: ApprovalBatch) -> str:
        """
        Save approval batch as telemetry span tree

        Creates:
        - Root span for batch with context attributes
        - Child span for each item with confidence and status

        Args:
            batch: Batch to save

        Returns:
            Batch ID
        """
        attributes = {
            "batch_id": batch.batch_id,
            "total_items": len(batch.items),
            "auto_approved": len(batch.auto_approved),
            "pending_review": len(batch.pending_review),
            "context": json.dumps(_serialize_for_json(batch.context)),
        }

        # Only add timestamp if it's not None (OpenTelemetry requirement)
        if batch.created_at:
            attributes["created_at"] = batch.created_at.isoformat()

        with self.telemetry_manager.span(
            name="approval_batch",
            tenant_id=self.tenant_id,
            project_name=self.project_name,
            attributes=attributes,
        ) as batch_span:
            # Create child span for each item
            for item in batch.items:
                self._create_item_span(item)

            batch_span.set_status(Status(StatusCode.OK))
            logger.info(f"Saved batch {batch.batch_id} to telemetry backend")

        # Force flush to ensure spans are exported immediately
        # With sync export (SimpleSpanProcessor), spans should be exported immediately
        # but we force flush to be extra sure
        try:
            # Access the cached tracer provider for this tenant/project
            cache_key = f"{self.tenant_id}:{self.project_name}"
            if hasattr(self.telemetry_manager, "_tenant_providers"):
                tracer_provider = self.telemetry_manager._tenant_providers.get(
                    cache_key
                )
                if tracer_provider and hasattr(tracer_provider, "force_flush"):
                    success = tracer_provider.force_flush(timeout_millis=5000)
                    logger.info(
                        f"Force flush completed for tenant {self.tenant_id}: success={success}"
                    )
                else:
                    logger.warning(
                        f"Tracer provider for {cache_key} does not support force_flush"
                    )
            else:
                logger.warning(
                    "TelemetryManager does not have _tenant_providers attribute"
                )
        except Exception as e:
            logger.error(f"Failed to flush tracer provider: {e}", exc_info=True)

        return batch.batch_id

    def _create_item_span(self, item: ReviewItem) -> None:
        """Create telemetry span for a review item"""
        attributes = {
            "item_id": item.item_id,
            "confidence": item.confidence,
            "status": item.status.value,
            "data": json.dumps(_serialize_for_json(item.data)),
            "metadata": json.dumps(_serialize_for_json(item.metadata)),
        }

        # Only add timestamp attributes if they're not None (OpenTelemetry requirement)
        if item.created_at:
            attributes["created_at"] = item.created_at.isoformat()
        if item.reviewed_at:
            attributes["reviewed_at"] = item.reviewed_at.isoformat()

        with self.telemetry_manager.span(
            name="approval_item",
            tenant_id=self.tenant_id,
            project_name=self.project_name,
            attributes=attributes,
        ) as item_span:
            item_span.set_status(Status(StatusCode.OK))

    async def _fetch_project_spans_with_retry(self, batch_id: str):
        """Query the project's spans, retrying with backoff for indexing lag.

        Returns the spans DataFrame (possibly empty) once the batch appears or
        the retries are exhausted.
        """
        retry_delays = [2, 5, 10, 15, 20]  # seconds (total: 52s)

        project_spans = None
        for attempt, delay in enumerate(retry_delays):
            logger.debug(
                f"Attempt {attempt + 1}/{len(retry_delays)}: Querying telemetry "
                f"backend for batch {batch_id}"
            )
            # Both span types: get_batch reconstructs the batch from the
            # approval_batch root AND its approval_item children in this same
            # frame — filtering to approval_batch alone empties every batch.
            project_spans = await self.provider.traces.get_spans(
                project=self.full_project_name,
                filters={"name": ["approval_batch", "approval_item"]},
            )
            if (
                not project_spans.empty
                and "attributes.batch_id" in project_spans.columns
            ):
                batch_check = project_spans[
                    (project_spans["name"] == "approval_batch")
                    & (project_spans["attributes.batch_id"] == batch_id)
                ]
                if not batch_check.empty:
                    logger.info(f"Found batch {batch_id} on attempt {attempt + 1}")
                    break

            if attempt < len(retry_delays) - 1:
                logger.debug(f"Batch {batch_id} not found yet, retrying in {delay}s")
                await asyncio.sleep(delay)

        return project_spans

    def _reconstruct_item(self, item_row, annotations_df) -> ReviewItem:
        """Rebuild one ReviewItem from its flattened span row (+ latest
        annotation). Raises on any malformed field — the caller isolates
        the failure to this item."""
        # In Phoenix 11.18.0, attributes are flattened as columns
        item_id = item_row.get("attributes.item_id", "")

        # Get initial status from span (default: pending_review)
        status_value = item_row.get("attributes.status", "pending_review")
        status = ApprovalStatus(status_value)

        # Parse timestamps from span attributes first
        created_at = item_row.get("attributes.created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        reviewed_at = item_row.get("attributes.reviewed_at")
        if isinstance(reviewed_at, str):
            reviewed_at = datetime.fromisoformat(reviewed_at)

        # Check for annotations (status updates) for this span
        # Annotations take precedence over span attributes
        # Match annotations by item_id in metadata (since annotations_df has no span_id column)
        if not annotations_df.empty:
            # Filter annotations for this specific item using metadata.item_id
            item_annotations = annotations_df[
                annotations_df["metadata"].apply(
                    lambda x: isinstance(x, dict) and x.get("item_id") == item_id
                )
            ]

            logger.debug(f"Item {item_id}: found {len(item_annotations)} annotations")

            if not item_annotations.empty:
                # Get latest annotation (most recent status)
                # Sort by created_at if available
                if "created_at" in item_annotations.columns:
                    latest_annotation = item_annotations.sort_values(
                        "created_at", ascending=False
                    ).iloc[0]
                else:
                    latest_annotation = item_annotations.iloc[-1]

                # Update status from annotation
                # telemetry annotations API returns label in 'result.label' column
                annotation_label = latest_annotation.get("result.label", "")
                if annotation_label:
                    try:
                        status = ApprovalStatus(annotation_label)
                        logger.debug(
                            f"Item {item_id} status from annotation: {status.value}"
                        )

                        # Also extract reviewed_at from annotation metadata if available
                        annotation_metadata = latest_annotation.get("metadata", {})
                        if isinstance(annotation_metadata, dict):
                            reviewed_at_str = annotation_metadata.get("reviewed_at")
                            if reviewed_at_str:
                                reviewed_at = datetime.fromisoformat(reviewed_at_str)

                    except ValueError:
                        logger.warning(
                            f"Invalid status label in annotation: {annotation_label}"
                        )
            else:
                logger.debug(
                    f"Item {item_id}: no annotations matched, keeping span status {status.value}"
                )

        # Parse data and metadata from flattened attributes
        data_raw = item_row.get("attributes.data", "{}")
        data = json.loads(data_raw) if isinstance(data_raw, str) else data_raw

        metadata_raw = item_row.get("attributes.metadata", "{}")
        metadata = (
            json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
        )

        confidence = float(item_row.get("attributes.confidence", 0.0))

        item = ReviewItem(
            item_id=item_id,
            data=data,
            confidence=confidence,
            status=status,
            metadata=metadata,
            created_at=created_at,
            reviewed_at=reviewed_at,
        )
        return item

    async def get_batch(
        self, batch_id: str, spans_df: Optional["pd.DataFrame"] = None
    ) -> Optional[ApprovalBatch]:
        """
        Retrieve approval batch from telemetry backend using SDK APIs with retry

        Queries spans (immutable item creation) and annotations (status updates)
        to reconstruct current batch state. Uses exponential backoff for telemetry backend indexing lag.

        Args:
            batch_id: Batch ID to retrieve
            spans_df: Optional pre-fetched project spans. When provided (e.g. by
                get_pending_batches, which already queried the whole project),
                the per-batch span re-fetch is skipped.

        Returns:
            ApprovalBatch if found, None otherwise
        """
        try:
            if spans_df is not None:
                project_spans = spans_df
            else:
                project_spans = await self._fetch_project_spans_with_retry(batch_id)

            if project_spans is None or project_spans.empty:
                logger.warning(
                    f"No spans found for project {self.full_project_name} after retries"
                )
                return None

            # Filter for this batch
            # Attributes are flattened: attributes.batch_id, attributes.total_items, etc.
            if "attributes.batch_id" not in project_spans.columns:
                logger.warning(
                    "No attributes.batch_id column in DataFrame - attributes not available"
                )
                return None

            batch_spans = project_spans[
                (project_spans["name"] == "approval_batch")
                & (project_spans["attributes.batch_id"] == batch_id)
            ]

            if batch_spans.empty:
                logger.warning(
                    f"Batch {batch_id} not found in telemetry backend after retries"
                )
                return None

            # Get batch span row
            batch_row = batch_spans.iloc[0]
            batch_span_id = batch_row["context.span_id"]

            # Get child item spans
            item_spans = project_spans[
                (project_spans["name"] == "approval_item")
                & (project_spans["parent_id"] == batch_span_id)
            ]

            # Query annotations for the latest status of each item. Item
            # approve/reject status lives ONLY in annotations, so a telemetry
            # outage here MUST propagate to the outer handler (which raises):
            # swallowing it left annotations_df empty and rebuilt every item at
            # its span-time pending_review, silently reverting all decisions —
            # the workflow then re-prompts resolved items / sits in
            # awaiting_approval forever. A genuine absence returns an empty
            # frame (not an exception), so the propagate-on-error contract
            # matches get_pending_batches, the sibling querying the same spans.
            span_ids = item_spans["context.span_id"].tolist()
            logger.debug(f"Querying annotations for {len(span_ids)} spans: {span_ids}")
            annotations_df = await self.provider.annotations.get_annotations(
                spans_df=item_spans,
                project=self.full_project_name,
                annotation_names=["item_status_update", "human_approval"],
            )
            logger.info(f"Found {len(annotations_df)} annotations for batch items")
            if not annotations_df.empty:
                logger.debug(f"Annotation columns: {list(annotations_df.columns)}")

            # Reconstruct items — one malformed span (truncated data
            # blob, junk confidence, unknown status) costs that item,
            # not the whole pending-approvals view. The outage contract
            # (raise) stays with the outer handler: backend failures
            # surface, bad rows don't.
            items = []
            skipped_malformed = 0
            for _, item_row in item_spans.iterrows():
                try:
                    items.append(self._reconstruct_item(item_row, annotations_df))
                except Exception as item_exc:
                    skipped_malformed += 1
                    logger.error(
                        "Skipping malformed approval item %r in batch %s: %r",
                        item_row.get("attributes.item_id", "<unknown>"),
                        batch_id,
                        item_exc,
                    )
            if skipped_malformed:
                logger.error(
                    "Batch %s: %d malformed item span(s) skipped — the batch "
                    "view is partial",
                    batch_id,
                    skipped_malformed,
                )

            # Parse context from batch attributes
            context_raw = batch_row.get("attributes.context", "{}")
            context = (
                json.loads(context_raw) if isinstance(context_raw, str) else context_raw
            )

            batch = ApprovalBatch(
                batch_id=batch_id,
                items=items,
                context=context,
            )

            logger.info(
                f"Retrieved batch {batch_id} from telemetry backend with {len(items)} items (status from annotations)"
            )
            # Debug: log item statuses
            status_counts = {}
            for item in items:
                status_counts[item.status.value] = (
                    status_counts.get(item.status.value, 0) + 1
                )
            logger.info(f"Batch {batch_id} status breakdown: {status_counts}")
            return batch

        except Exception as e:
            # A telemetry outage is NOT "batch not found". get_pending_batches
            # (the sibling querying the same spans) lets get_spans raise, and so
            # must this: flattening to None made an outage read as a missing
            # batch, so apply_approvals kept the stale pre-decision batch and the
            # workflow sat in awaiting_approval forever. A genuine not-found
            # returns None above (empty spans); this branch is a real failure.
            logger.error(
                f"Error retrieving batch {batch_id} from telemetry backend: {e}",
                exc_info=True,
            )
            raise

    async def update_item(
        self, item: ReviewItem, batch_id: Optional[str] = None
    ) -> None:
        """
        Update review item status using telemetry annotations

        Logs status change as annotation on the original item span.
        Uses telemetry backend's annotations API for human/system feedback.

        Args:
            item: Item with updated status
            batch_id: Optional batch ID to help find the span
        """
        # Find the span ID for this item
        span_id = await self.get_item_span_id(item.item_id, batch_id=batch_id)

        if not span_id:
            logger.error(f"Cannot update item {item.item_id}: span not found")
            raise ValueError(f"Span not found for item {item.item_id}")

        # Log the status update as annotation using telemetry annotations API
        try:
            # Prepare metadata
            metadata = {
                "item_id": item.item_id,
                "confidence": item.confidence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if item.reviewed_at:
                metadata["reviewed_at"] = item.reviewed_at.isoformat()

            # Add annotation using telemetry provider
            logger.info(
                f"Creating annotation for item {item.item_id} (status={item.status.value}) on span {span_id}"
            )
            await self.provider.annotations.add_annotation(
                span_id=span_id,
                name="item_status_update",
                label=item.status.value,  # "approved", "rejected", etc.
                score=1.0 if item.status == ApprovalStatus.APPROVED else 0.0,
                metadata=metadata,
                project=self.full_project_name,
            )

            logger.info(
                f"Successfully created annotation for item {item.item_id}: status={item.status.value}, span_id={span_id}"
            )

        except Exception as e:
            logger.error(
                f"Failed to add annotation for item {item.item_id}: {e}", exc_info=True
            )
            raise

    async def get_pending_batches(
        self, context_filter: Optional[Dict[str, Any]] = None
    ) -> List[ApprovalBatch]:
        """
        Get batches with pending reviews by querying telemetry backend

        Args:
            context_filter: Optional filter by batch context

        Returns:
            List of batches with pending items
        """
        try:
            await asyncio.sleep(0.5)  # Give telemetry backend time to process spans

            # Both span types: each pending batch is reconstructed via
            # get_batch(spans_df=...) below, which reads the approval_item
            # children out of this same frame — approval_batch alone would
            # give every batch empty items.
            spans_df = await self.provider.traces.get_spans(
                project=self.full_project_name,
                filters={"name": ["approval_batch", "approval_item"]},
            )

            if spans_df.empty:
                return []

            # Filter for approval_batch spans with pending_review > 0
            # In Phoenix 11.18.0, attributes are flattened
            if (
                "attributes.batch_id" not in spans_df.columns
                or "attributes.pending_review" not in spans_df.columns
            ):
                logger.warning("Required attributes columns not found in DataFrame")
                return []

            # Filter batch spans with pending_review > 0, handling NA values
            pending_counts = pd.to_numeric(
                spans_df["attributes.pending_review"], errors="coerce"
            ).fillna(0)
            batch_spans = spans_df[
                (spans_df["name"] == "approval_batch") & (pending_counts > 0)
            ]

            pending_batches = []
            for _, row in batch_spans.iterrows():
                batch_id = row.get("attributes.batch_id")

                if not batch_id:
                    continue

                # Apply context filter if provided
                if context_filter:
                    context_raw = row.get("attributes.context", "{}")
                    context = (
                        json.loads(context_raw)
                        if isinstance(context_raw, str)
                        else context_raw
                    )
                    match = all(context.get(k) == v for k, v in context_filter.items())
                    if not match:
                        continue

                # Retrieve full batch, reusing the spans already fetched above
                # instead of re-querying the whole project per batch (N+1).
                batch = await self.get_batch(batch_id, spans_df=spans_df)
                if batch:
                    pending_batches.append(batch)

            logger.debug(f"Found {len(pending_batches)} pending batches")
            return pending_batches

        except Exception as e:
            # Propagate: an empty approval queue must mean "nothing pending",
            # never "the telemetry backend was unreachable".
            logger.error(
                f"Error retrieving pending batches from telemetry backend: {e!r}"
            )
            raise

    async def record_decision(self, decision: ReviewDecision, item: ReviewItem) -> None:
        """
        Record human decision as telemetry annotation

        Args:
            decision: Human decision
            item: Review item being decided on
        """
        attributes = {
            "item_id": decision.item_id,
            "approved": decision.approved,
            "reviewer": decision.reviewer or "unknown",
            "timestamp": (
                decision.timestamp.isoformat()
                if decision.timestamp
                else datetime.now(timezone.utc).isoformat()
            ),
            "feedback": decision.feedback or "",
            "corrections": json.dumps(decision.corrections),
        }

        with self.telemetry_manager.span(
            name="approval_decision",
            tenant_id=self.tenant_id,
            project_name=self.project_name,
            attributes=attributes,
        ) as decision_span:
            # Add event for the decision
            decision_span.add_event(
                "human_decision",
                attributes={
                    "item_id": decision.item_id,
                    "approved": decision.approved,
                    "has_feedback": bool(decision.feedback),
                    "has_corrections": len(decision.corrections) > 0,
                },
            )

            decision_span.set_status(Status(StatusCode.OK))
            logger.info(
                f"Recorded decision for {decision.item_id}: "
                f"{'APPROVED' if decision.approved else 'REJECTED'}"
            )

    async def get_item_span_id(
        self, item_id: str, batch_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Get span ID for an approval_item by item_id using telemetry SDK with retry

        Uses exponential backoff to handle telemetry backend indexing lag.

        Args:
            item_id: Item ID to find span for
            batch_id: Optional batch ID to narrow search

        Returns:
            Span ID if found, None otherwise
        """
        try:
            # Retry with exponential backoff
            max_retries = 3
            retry_delays = [0.5, 1, 2]  # seconds

            for attempt, delay in enumerate(retry_delays):
                # Query spans using telemetry provider
                project_spans = await self.provider.traces.get_spans(
                    project=self.full_project_name,
                    filters={"name": "approval_item"},
                )

                if not project_spans.empty:
                    # Filter for approval_item spans with matching item_id
                    # In Phoenix 11.18.0, attributes are flattened
                    if "attributes.item_id" in project_spans.columns:
                        item_spans = project_spans[
                            (project_spans["name"] == "approval_item")
                            & (project_spans["attributes.item_id"] == item_id)
                        ]

                        if not item_spans.empty:
                            # Get the most recent span (by start_time)
                            latest_span = item_spans.sort_values(
                                "start_time", ascending=False
                            ).iloc[0]
                            span_id = latest_span["context.span_id"]
                            logger.debug(
                                f"Found span {span_id} for item {item_id} on attempt {attempt + 1}"
                            )
                            return span_id

                # Retry with backoff
                if attempt < len(retry_delays) - 1:
                    logger.debug(
                        f"Span for item {item_id} not found, retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)

            logger.warning(
                f"No span found for item {item_id} after {max_retries} retries"
            )
            return None

        except Exception as e:
            # Outage is NOT "span not indexed yet" (that returns None above). A
            # backend failure flattened to None silently skipped the approval
            # decision annotation in apply_decision (the item persisted APPROVED
            # but the decision log was lost). Raise so the caller surfaces it.
            logger.error(f"Error finding span for item {item_id}: {e}")
            raise

    async def log_approval_decision(
        self,
        span_id: str,
        item_id: str,
        approved: bool,
        feedback: Optional[str] = None,
        reviewer: Optional[str] = None,
    ) -> bool:
        """
        Log approval decision as annotation using telemetry annotations API

        Records human approval/rejection decisions as annotations on item spans.
        Uses telemetry backend's proper annotations API for semantic feedback.

        Args:
            span_id: Span ID of the approval_item span to annotate
            item_id: Item ID being approved/rejected
            approved: True if approved, False if rejected
            feedback: Optional human feedback text
            reviewer: Optional reviewer identifier

        Returns:
            True if annotation logged successfully
        """
        try:
            # Prepare metadata
            metadata = {
                "item_id": item_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if reviewer:
                metadata["reviewer"] = reviewer
            if feedback:
                metadata["feedback"] = feedback

            # Add annotation using telemetry provider
            await self.provider.annotations.add_annotation(
                span_id=span_id,
                name="human_approval",
                label="approved" if approved else "rejected",
                score=1.0 if approved else 0.0,
                metadata=metadata,
                project=self.full_project_name,
            )

            logger.info(
                f"Added approval annotation for item {item_id} on span {span_id}: "
                f"{'APPROVED' if approved else 'REJECTED'}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to log approval decision annotation: {e}", exc_info=True
            )
            return False

    async def append_to_training_dataset(
        self,
        dataset_name: str,
        items: List[ReviewItem],
        project_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Append approved items to telemetry backend dataset for training

        Organizes approved items into a telemetry dataset that can be used for
        DSPy optimization or model training.

        Args:
            dataset_name: Name of the telemetry dataset (will be created if doesn't exist)
            items: List of approved ReviewItems to add to dataset
            project_context: Optional context about the project/task

        Returns:
            True if items appended successfully
        """
        try:
            # Convert approved items to dataset records
            dataset_records = []
            for item in items:
                # Extract query/input from item data
                data = item.data

                # Build dataset record with input/output structure
                record = {
                    "item_id": item.item_id,
                    "confidence": item.confidence,
                    "status": item.status.value,
                    "created_at": (
                        item.created_at.isoformat() if item.created_at else None
                    ),
                    "reviewed_at": (
                        item.reviewed_at.isoformat() if item.reviewed_at else None
                    ),
                }

                # Add item data fields
                record.update(data)

                # Add metadata
                if item.metadata:
                    record.update(
                        {f"metadata.{k}": v for k, v in item.metadata.items()}
                    )

                # Add project context
                if project_context:
                    record.update(
                        {f"context.{k}": v for k, v in project_context.items()}
                    )

                dataset_records.append(record)

            if not dataset_records:
                logger.warning("No items to append to dataset")
                return False

            # Create DataFrame
            df = pd.DataFrame(dataset_records)

            # Absent dataset (KeyError/ValueError sentinel) -> create it;
            # anything else -- including an outage mid-append -- propagates
            # to the outer raise instead of masquerading as first-run
            # creation against a live dataset.
            try:
                await self.provider.datasets.get_dataset(name=dataset_name)
                dataset_exists = True
            except (KeyError, ValueError):
                dataset_exists = False
            if dataset_exists:
                logger.info(
                    f"Appending {len(dataset_records)} items to existing dataset '{dataset_name}'"
                )
                await self.provider.datasets.append_to_dataset(
                    name=dataset_name, data=df
                )
                logger.info(
                    f"Appended to dataset '{dataset_name}' with {len(dataset_records)} items"
                )
            else:
                logger.info(
                    f"Creating new dataset '{dataset_name}' with {len(dataset_records)} items"
                )
                await self.provider.datasets.create_dataset(name=dataset_name, data=df)

            logger.info(
                f"Successfully added {len(dataset_records)} approved items to dataset '{dataset_name}'"
            )
            return True

        except Exception as e:
            # Outage while appending: raise instead of returning False. A
            # swallowed False left apply_decision reporting success with the item
            # marked APPROVED but never written to the training dataset — a torn
            # state. (An empty item list returns False above; that's a no-op.)
            logger.error(
                f"Failed to append items to training dataset: {e}", exc_info=True
            )
            raise
