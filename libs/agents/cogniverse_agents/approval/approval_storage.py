"""
Approval Storage Implementation

Stores approval data as telemetry spans with annotations using pluggable provider.
Enables approval workflow tracing and analysis.
"""

import ast
import asyncio
import copy
import hashlib
import json
import logging
import math
import re
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import pandas as pd
import redis.asyncio as aioredis
from opentelemetry.trace import Status, StatusCode

from cogniverse_agents.approval.replacement_store import RedisReplacementRecordStore
from cogniverse_core.approval.interfaces import (
    ApprovalBatch,
    ApprovalStatus,
    ApprovalStorage,
    ReviewDecision,
    ReviewItem,
    approved_synthetic_dataset_name,
)
from cogniverse_core.approval.training_schema import (
    APPROVED_SYNTHETIC_AGENT_TYPES,
    APPROVED_SYNTHETIC_OUTPUT_FIELDS,
    validate_approved_training_values,
)
from cogniverse_foundation.telemetry.providers.base import DatasetStore

if TYPE_CHECKING:
    from cogniverse_foundation.telemetry.manager import TelemetryManager

logger = logging.getLogger(__name__)


class _ApprovalDatasetLockOwnershipLost(RuntimeError):
    pass


_TRAINING_AGENT_OUTPUT_FIELDS = APPROVED_SYNTHETIC_OUTPUT_FIELDS
_TRAINING_AGENT_TYPES = APPROVED_SYNTHETIC_AGENT_TYPES


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


def _require_training_string(
    item: ReviewItem,
    field: str,
    *,
    values: Dict[str, Any],
) -> str:
    value = values.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Training dataset item {item.item_id!r} requires a non-empty "
            f"{field} string"
        )
    return value


def _validate_training_entity_values(item: ReviewItem) -> None:
    entities = item.data.get("entities")
    if not isinstance(entities, list):
        raise ValueError(
            f"Training dataset item {item.item_id!r} entities must be a list"
        )
    for position, entity in enumerate(entities):
        if (
            not isinstance(entity, dict)
            or set(entity) != {"text", "type"}
            or not all(
                isinstance(entity[key], str) and entity[key].strip()
                for key in ("text", "type")
            )
        ):
            raise ValueError(
                f"Training dataset item {item.item_id!r} entity at position "
                f"{position} requires exactly non-empty text and type strings"
            )

    relationships = item.data.get("relationships")
    if not isinstance(relationships, list):
        raise ValueError(
            f"Training dataset item {item.item_id!r} relationships must be a list"
        )
    for position, relationship in enumerate(relationships):
        if (
            not isinstance(relationship, dict)
            or set(relationship) != {"source", "target", "type"}
            or not all(
                isinstance(relationship[key], str) and relationship[key].strip()
                for key in ("source", "target", "type")
            )
        ):
            raise ValueError(
                f"Training dataset item {item.item_id!r} relationship at position "
                f"{position} requires exactly non-empty source, target, and type strings"
            )


def _validate_training_item_schema(item: ReviewItem) -> None:
    if not isinstance(item.data, dict):
        raise ValueError(
            f"Training dataset item {item.item_id!r} data must be a dictionary"
        )
    if not isinstance(item.metadata, dict):
        raise ValueError(
            f"Training dataset item {item.item_id!r} metadata must be a dictionary"
        )

    agent_type = item.metadata.get("agent_type")
    if not isinstance(agent_type, str) or not agent_type.strip():
        raise ValueError(
            f"Training dataset item {item.item_id!r} requires a non-empty "
            "metadata.agent_type string"
        )
    if agent_type not in _TRAINING_AGENT_TYPES:
        raise ValueError(
            f"Training dataset item {item.item_id!r} has unsupported "
            f"metadata.agent_type {agent_type!r}"
        )

    validate_approved_training_values(
        item.data,
        agent_type,
        context=f"Training dataset item {item.item_id!r}",
    )

    _require_training_string(item, "query", values=item.data)
    output_field = _TRAINING_AGENT_OUTPUT_FIELDS.get(agent_type)
    if output_field is not None:
        _require_training_string(item, output_field, values=item.data)
    else:
        _validate_training_entity_values(item)


def _replacement_payload(item: ReviewItem) -> Dict[str, Any]:
    payload = {
        "item_id": item.item_id,
        "data": _serialize_for_json(item.data),
        "confidence": item.confidence,
        "status": item.status.value,
        "metadata": _serialize_for_json(item.metadata),
        "created_at": item.created_at.isoformat(),
        "reviewed_at": item.reviewed_at.isoformat() if item.reviewed_at else None,
    }
    _validate_replacement_payload(payload)
    return payload


def _review_decision_payload(decision: ReviewDecision) -> Dict[str, Any]:
    if not isinstance(decision.timestamp, datetime):
        raise ValueError("Review decision timestamp is required")
    if decision.timestamp.tzinfo is None or decision.timestamp.utcoffset() is None:
        raise ValueError("Review decision timestamp must include timezone information")
    return {
        "item_id": decision.item_id,
        "approved": decision.approved,
        "reviewer": decision.reviewer,
        "feedback": decision.feedback,
        "corrections": copy.deepcopy(decision.corrections),
        "timestamp": decision.timestamp.isoformat(),
    }


def _review_decision_from_payload(payload: Dict[str, Any]) -> ReviewDecision:
    try:
        decision = ReviewDecision(
            item_id=payload["item_id"],
            approved=payload["approved"],
            reviewer=payload["reviewer"],
            feedback=payload["feedback"],
            corrections=copy.deepcopy(payload["corrections"]),
            timestamp=datetime.fromisoformat(payload["timestamp"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Canonical review decision record is invalid") from exc
    if _review_decision_payload(decision) != payload:
        raise RuntimeError("Canonical review decision record is invalid")
    return decision


def _approval_batch_payload(batch: ApprovalBatch) -> Dict[str, Any]:
    if not isinstance(batch.created_at, datetime):
        raise ValueError("Approval batch created_at is required")
    if batch.created_at.tzinfo is None or batch.created_at.utcoffset() is None:
        raise ValueError("Approval batch created_at must include timezone information")
    return {
        "batch_id": batch.batch_id,
        "context": _serialize_for_json(batch.context),
        "created_at": batch.created_at.isoformat(),
        "items": [
            {
                "item_id": item.item_id,
                "data": _serialize_for_json(item.data),
                "confidence": item.confidence,
                "status": item.status.value,
                "metadata": _serialize_for_json(item.metadata),
                "created_at": item.created_at.isoformat(),
                "reviewed_at": (
                    item.reviewed_at.isoformat() if item.reviewed_at else None
                ),
            }
            for item in batch.items
        ],
    }


def _apply_canonical_batch_timestamps(
    batch: ApprovalBatch,
    payload: Dict[str, Any],
) -> None:
    if payload.get("batch_id") != batch.batch_id:
        raise RuntimeError("Canonical approval batch ID does not match submission")
    selected_items = payload.get("items")
    if not isinstance(selected_items, list) or [
        selected.get("item_id") if isinstance(selected, dict) else None
        for selected in selected_items
    ] != [item.item_id for item in batch.items]:
        raise RuntimeError(
            "Canonical approval batch item order does not match submission"
        )
    batch.created_at = _canonical_aware_timestamp(
        payload.get("created_at"), "batch created_at"
    )
    for item, selected in zip(batch.items, selected_items, strict=True):
        item.created_at = _canonical_aware_timestamp(
            selected.get("created_at"),
            f"item {item.item_id!r} created_at",
        )


def _replacement_from_payload(payload: Dict[str, Any]) -> ReviewItem:
    _validate_replacement_payload(payload)
    reviewed_at = payload["reviewed_at"]
    return ReviewItem(
        item_id=payload["item_id"],
        data=payload["data"],
        confidence=float(payload["confidence"]),
        status=ApprovalStatus(payload["status"]),
        metadata=payload["metadata"],
        created_at=datetime.fromisoformat(payload["created_at"]),
        reviewed_at=datetime.fromisoformat(reviewed_at) if reviewed_at else None,
    )


def _canonical_aware_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Replacement {field} must be a non-empty string")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Replacement {field} must include timezone information")
    if parsed.isoformat() != value:
        raise ValueError(f"Replacement {field} must use canonical ISO format")
    return parsed


def _validate_replacement_payload(payload: Dict[str, Any]) -> None:
    required_fields = {
        "item_id",
        "data",
        "confidence",
        "status",
        "metadata",
        "created_at",
        "reviewed_at",
    }
    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise ValueError(
            "Replacement payload must contain exactly "
            + ", ".join(sorted(required_fields))
        )
    if not isinstance(payload["item_id"], str) or not payload["item_id"].strip():
        raise ValueError("Replacement item_id must be a non-empty string")
    if not isinstance(payload["data"], dict):
        raise ValueError("Replacement data must be an object")
    confidence = payload["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
    ):
        raise ValueError("Replacement confidence must be a finite number from 0 to 1")
    if payload["status"] != ApprovalStatus.REGENERATED.value:
        raise ValueError("Replacement status must be regenerated")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Replacement metadata must be an object")
    original_item_id = metadata.get("original_item_id")
    if not isinstance(original_item_id, str) or not original_item_id.strip():
        raise ValueError(
            "Replacement metadata.original_item_id must be a non-empty string"
        )
    decision = metadata.get("decision")
    required_decision_fields = {
        "reviewer",
        "feedback",
        "corrections",
        "timestamp",
    }
    if not isinstance(decision, dict) or set(decision) != required_decision_fields:
        raise ValueError(
            "Replacement metadata.decision must contain exactly "
            "corrections, feedback, reviewer, timestamp"
        )
    if not isinstance(decision["reviewer"], str) or not decision["reviewer"].strip():
        raise ValueError("Replacement decision reviewer must be a non-empty string")
    if not isinstance(decision["feedback"], str):
        raise ValueError("Replacement decision feedback must be a string")
    if not isinstance(decision["corrections"], dict):
        raise ValueError("Replacement decision corrections must be an object")
    _canonical_aware_timestamp(decision["timestamp"], "decision timestamp")
    _canonical_aware_timestamp(payload["created_at"], "created_at")
    if payload["reviewed_at"] is not None:
        _canonical_aware_timestamp(payload["reviewed_at"], "reviewed_at")


def _strict_json_object_pairs(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _replacement_payload_from_record(
    record_json: Any,
    record_sha256: Any,
) -> Dict[str, Any]:
    if not isinstance(record_json, str) or not record_json:
        raise ValueError("Replacement record JSON must be a non-empty string")
    if not _is_sha256(record_sha256):
        raise ValueError("Replacement record digest must be a SHA-256 hex string")
    if hashlib.sha256(record_json.encode("utf-8")).hexdigest() != record_sha256:
        raise ValueError("Replacement record digest does not match its JSON")
    payload = json.loads(
        record_json,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_strict_json_object_pairs,
    )
    if not isinstance(payload, dict):
        raise ValueError("Replacement record JSON must contain an object")
    if _canonical_approval_json(payload) != record_json:
        raise ValueError("Replacement record JSON is not canonical")
    _validate_replacement_payload(payload)
    return payload


def _canonical_approval_json(value: Dict[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _decision_digest(record: Dict[str, Any]) -> str:
    decision_intent = copy.deepcopy(record.get("metadata.decision"))
    if isinstance(decision_intent, dict):
        decision_intent.pop("timestamp", None)
    identity = {
        "item_id": record.get("item_id"),
        "status": record.get("status"),
        "decision": decision_intent,
    }
    return hashlib.sha256(
        _canonical_approval_json(identity).encode("utf-8")
    ).hexdigest()


def _provider_value_matches_canonical(observed: Any, canonical: Any) -> bool:
    """Compare a Phoenix round-trip value with its signed canonical value."""
    if canonical is None:
        return (
            observed is None
            or (isinstance(observed, str) and observed == "")
            or (
                isinstance(observed, float)
                and not isinstance(observed, bool)
                and math.isnan(observed)
            )
        )
    if isinstance(canonical, (dict, list)) and isinstance(observed, str):
        try:
            observed = ast.literal_eval(observed)
        except (SyntaxError, ValueError):
            return False
        try:
            _canonical_approval_json({"value": observed})
        except (TypeError, ValueError):
            return False
        return observed == canonical
    if isinstance(canonical, bool):
        return (isinstance(observed, bool) and observed is canonical) or (
            isinstance(observed, str) and observed == str(canonical)
        )
    if isinstance(canonical, int):
        if isinstance(observed, bool):
            return False
        if isinstance(observed, int):
            return observed == canonical
        return (
            isinstance(observed, str)
            and re.fullmatch(r"-?(?:0|[1-9][0-9]*)", observed) is not None
            and int(observed) == canonical
        )
    if isinstance(canonical, float):
        if isinstance(observed, bool):
            return False
        if isinstance(observed, (int, float)):
            return math.isfinite(observed) and observed == canonical
        if not isinstance(observed, str) or observed != observed.strip():
            return False
        try:
            parsed = float(observed)
        except ValueError:
            return False
        return math.isfinite(parsed) and parsed == canonical
    return observed == canonical


def _is_provider_missing_placeholder(value: Any) -> bool:
    """Identify scalar nulls added while heterogeneous records share a frame."""
    if value is None or (isinstance(value, str) and value == ""):
        return True
    if isinstance(value, (str, bytes, dict, list, tuple, set)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def validate_approved_dataset_record(
    record: Dict[str, Any],
    *,
    tenant_id: str,
    dataset_name: str,
    position: int,
) -> Dict[str, Any]:
    context = f"tenant={tenant_id} dataset={dataset_name} row={position}"
    item_id = record.get("item_id")
    if not isinstance(item_id, str) or not item_id:
        raise RuntimeError(f"Approved dataset row has no item_id: {context}")
    item_context = f"{context} item={item_id}"

    canonical_json = record.get("metadata.approval_record_json")
    if not isinstance(canonical_json, str) or not canonical_json:
        raise RuntimeError(
            "Approved dataset item has invalid metadata.approval_record_json: "
            f"{item_context}"
        )
    try:
        canonical_record = json.loads(
            canonical_json,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Approved dataset item has invalid canonical JSON: {item_context}"
        ) from exc
    if not isinstance(canonical_record, dict):
        raise RuntimeError(
            f"Approved dataset item canonical content is not an object: {item_context}"
        )
    try:
        reserialized = _canonical_approval_json(canonical_record)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Approved dataset item canonical content is not JSON-safe: {item_context}"
        ) from exc
    if reserialized != canonical_json:
        raise RuntimeError(
            f"Approved dataset item content is not canonical JSON: {item_context}"
        )
    if "metadata.approval_record_json" in canonical_record or (
        "metadata.approval_record_sha256" in canonical_record
    ):
        raise RuntimeError(
            f"Approved dataset item canonical content is recursive: {item_context}"
        )
    if canonical_record.get("item_id") != item_id:
        raise RuntimeError(
            f"Approved dataset item_id differs from canonical content: {item_context}"
        )

    record_digest = record.get("metadata.approval_record_sha256")
    if not _is_sha256(record_digest):
        raise RuntimeError(
            "Approved dataset item has invalid metadata.approval_record_sha256: "
            f"{item_context}"
        )
    expected_record_digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    if record_digest != expected_record_digest:
        raise RuntimeError(
            f"Approved dataset item canonical content hash does not match: {item_context}"
        )

    decision_digest = record.get("metadata.approval_decision_sha256")
    if not _is_sha256(decision_digest):
        raise RuntimeError(
            "Approved dataset item has invalid metadata.approval_decision_sha256: "
            f"{item_context}"
        )
    if canonical_record.get("metadata.approval_decision_sha256") != decision_digest:
        raise RuntimeError(
            f"Approved dataset item decision hash differs from canonical content: {item_context}"
        )
    if _decision_digest(canonical_record) != decision_digest:
        raise RuntimeError(
            f"Approved dataset item decision content hash does not match: {item_context}"
        )

    decision_timestamp = record.get("metadata.approval_decision_timestamp")
    if (
        canonical_record.get("metadata.approval_decision_timestamp")
        != decision_timestamp
    ):
        raise RuntimeError(
            "Approved dataset item decision timestamp differs from canonical content: "
            f"{item_context}"
        )
    if not isinstance(decision_timestamp, str):
        raise RuntimeError(
            "Approved dataset item has invalid metadata.approval_decision_timestamp: "
            f"{item_context}"
        )
    try:
        parsed_timestamp = datetime.fromisoformat(decision_timestamp)
    except ValueError as exc:
        raise RuntimeError(
            "Approved dataset item has invalid metadata.approval_decision_timestamp: "
            f"{item_context}"
        ) from exc
    if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
        raise RuntimeError(
            "Approved dataset item has naive metadata.approval_decision_timestamp: "
            f"{item_context}"
        )
    if canonical_record.get("reviewed_at") != decision_timestamp:
        raise RuntimeError(
            f"Approved dataset item reviewed_at differs from decision timestamp: {item_context}"
        )
    decision = canonical_record.get("metadata.decision")
    if isinstance(decision, dict) and decision.get("timestamp") != decision_timestamp:
        raise RuntimeError(
            f"Approved dataset item decision timestamp is inconsistent: {item_context}"
        )

    unsigned_record = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "metadata.approval_record_json",
            "metadata.approval_record_sha256",
        }
    }
    missing_fields = sorted(set(canonical_record) - set(unsigned_record))
    unexpected_fields = sorted(
        key
        for key in set(unsigned_record) - set(canonical_record)
        if not _is_provider_missing_placeholder(unsigned_record[key])
    )
    mismatched_fields = sorted(
        key
        for key in set(canonical_record) & set(unsigned_record)
        if not _provider_value_matches_canonical(
            unsigned_record[key], canonical_record[key]
        )
    )
    if missing_fields or unexpected_fields or mismatched_fields:
        raise RuntimeError(
            "Approved dataset item content differs from canonical content: "
            f"{item_context} missing={missing_fields} "
            f"unexpected={unexpected_fields} mismatched={mismatched_fields}"
        )

    return {
        **canonical_record,
        "metadata.approval_record_json": canonical_json,
        "metadata.approval_record_sha256": record_digest,
    }


def _validated_approved_dataset_snapshot(
    payload: Any,
    *,
    tenant_id: str,
    dataset_name: str,
) -> pd.DataFrame:
    context = f"tenant={tenant_id} dataset={dataset_name}"
    if not isinstance(payload, pd.DataFrame):
        raise RuntimeError(
            "Approved dataset payload is not a pandas DataFrame: "
            f"{context} got={type(payload).__name__}"
        )
    if "input" not in payload.columns:
        raise RuntimeError(f"Approved dataset payload has no input column: {context}")

    canonical_records = []
    positions_by_id: Dict[str, List[int]] = {}
    for position, record in enumerate(payload["input"].tolist()):
        if not isinstance(record, dict):
            raise RuntimeError(
                f"Approved dataset row has no input record: {context} row={position}"
            )
        canonical = validate_approved_dataset_record(
            record,
            tenant_id=tenant_id,
            dataset_name=dataset_name,
            position=position,
        )
        item_id = canonical["item_id"]
        positions_by_id.setdefault(item_id, []).append(position)
        canonical_records.append(canonical)

    for item_id, positions in positions_by_id.items():
        if len(positions) != 1:
            raise RuntimeError(
                "Approved dataset contains duplicate item records: "
                f"{context} item={item_id} count={len(positions)}"
            )

    snapshot = payload.copy(deep=True)
    snapshot["input"] = canonical_records
    return snapshot


class _ApprovedDatasetIntegrityStore(DatasetStore):
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    @staticmethod
    def _tenant_id(name: str) -> Optional[str]:
        prefix = "approved_synthetic_data-"
        if not name.startswith(prefix):
            return None
        tenant_id = name[len(prefix) :]
        return tenant_id or None

    @staticmethod
    def _validate_outgoing(name: str, data: Any, tenant_id: str) -> None:
        if not isinstance(data, pd.DataFrame):
            raise RuntimeError(
                "Approved dataset payload is not a pandas DataFrame: "
                f"tenant={tenant_id} dataset={name} got={type(data).__name__}"
            )
        records = [row.to_dict() for _, row in data.iterrows()]
        _validated_approved_dataset_snapshot(
            pd.DataFrame({"input": records}),
            tenant_id=tenant_id,
            dataset_name=name,
        )

    async def create_dataset(
        self,
        name: str,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        tenant_id = self._tenant_id(name)
        if tenant_id is not None:
            self._validate_outgoing(name, data, tenant_id)
        return await self._delegate.create_dataset(
            name=name,
            data=data,
            metadata=metadata,
        )

    async def get_dataset(self, name: str) -> pd.DataFrame:
        payload = await self._delegate.get_dataset(name=name)
        tenant_id = self._tenant_id(name)
        if tenant_id is None:
            return payload
        return _validated_approved_dataset_snapshot(
            payload,
            tenant_id=tenant_id,
            dataset_name=name,
        )

    async def append_to_dataset(
        self,
        name: str,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        tenant_id = self._tenant_id(name)
        if tenant_id is not None:
            self._validate_outgoing(name, data, tenant_id)
        await self._delegate.append_to_dataset(
            name=name,
            data=data,
            metadata=metadata,
        )

    async def delete_dataset(self, name: str) -> bool:
        return await self._delegate.delete_dataset(name=name)


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
    - Store approval spans in the telemetry provider and canonical replacement
      selections in Redis
    - Provider-agnostic implementation
    """

    def __init__(
        self,
        grpc_endpoint: str,
        http_endpoint: str,
        tenant_id: str,
        telemetry_manager: Optional["TelemetryManager"] = None,
        redis_url: Optional[str] = None,
    ):
        """
        Initialize approval storage for synthetic data workflow

        Args:
            grpc_endpoint: OTLP gRPC endpoint for span export (e.g., "http://localhost:4317")
            http_endpoint: HTTP endpoint for span queries (e.g., "http://localhost:6006")
            tenant_id: Tenant ID for multi-tenant project scoping
            telemetry_manager: TelemetryManager instance (if None, creates one)
            redis_url: Redis endpoint required for approved dataset writes and
                replacement selection
        """
        from cogniverse_core.common.tenant_utils import canonical_tenant_id

        # Runtime writers register approval spans under the canonical tenant;
        # register, name, and query the same scope regardless of spelling.
        tenant_id = canonical_tenant_id(tenant_id)

        self.tenant_id = tenant_id
        self.project_name = "synthetic_data"
        self.redis_url = redis_url
        self._replacement_records = (
            RedisReplacementRecordStore(redis_url) if redis_url else None
        )

        if telemetry_manager is None:
            from cogniverse_foundation.telemetry.config import TelemetryConfig
            from cogniverse_foundation.telemetry.manager import TelemetryManager

            config = TelemetryConfig(
                provider_config={
                    "http_endpoint": http_endpoint,
                    "grpc_endpoint": grpc_endpoint,
                },
            )
            telemetry_manager = TelemetryManager(config=config)
        else:
            if not telemetry_manager.config.provider_config:
                telemetry_manager.config.provider_config = {}
            telemetry_manager.config.provider_config.update(
                {
                    "http_endpoint": http_endpoint,
                    "grpc_endpoint": grpc_endpoint,
                }
            )

        self.telemetry_manager = telemetry_manager

        self.telemetry_manager.register_project(
            tenant_id=tenant_id,
            project_name=self.project_name,
            otlp_endpoint=grpc_endpoint,
            http_endpoint=http_endpoint,
            use_sync_export=False,
        )

        self.full_project_name = f"cogniverse-{tenant_id}-{self.project_name}"

        self.provider = self.telemetry_manager.get_provider(
            tenant_id=tenant_id, project_name=self.project_name
        )
        dataset_store = self.provider.datasets
        if not isinstance(dataset_store, _ApprovedDatasetIntegrityStore):
            self.provider._dataset_store = _ApprovedDatasetIntegrityStore(dataset_store)

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
        from cogniverse_core.common.tenant_utils import require_tenant_id

        batch_tenant = require_tenant_id(
            batch.context.get("tenant_id"),
            source=f"approval batch {batch.batch_id} context",
        )
        if batch_tenant != self.tenant_id:
            raise ValueError(
                "Approval batch tenant does not match its storage: "
                f"batch={batch.batch_id} context_tenant={batch_tenant} "
                f"storage_tenant={self.tenant_id}"
            )
        batch.context["tenant_id"] = batch_tenant

        if self._replacement_records is not None:
            selected_batch = await self._replacement_records.select_approval_batch(
                tenant_id=self.tenant_id,
                batch_id=batch.batch_id,
                candidate=_approval_batch_payload(batch),
            )
            _apply_canonical_batch_timestamps(batch, selected_batch.payload)

        attributes = {
            "batch_id": batch.batch_id,
            "total_items": len(batch.items),
            "auto_approved": len(batch.auto_approved),
            "pending_review": len(batch.pending_review),
            "context": json.dumps(_serialize_for_json(batch.context)),
        }

        if batch.created_at:
            attributes["created_at"] = batch.created_at.isoformat()

        try:
            await asyncio.to_thread(
                self._write_batch_spans,
                batch,
                attributes,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Approval batch {batch.batch_id} spans failed to export"
            ) from exc
        return batch.batch_id

    def _write_batch_spans(
        self,
        batch: ApprovalBatch,
        attributes: Dict[str, Any],
    ) -> None:
        with self.telemetry_manager.span(
            name="approval_batch",
            tenant_id=self.tenant_id,
            project_name=self.project_name,
            attributes=attributes,
            require_export=True,
        ) as batch_span:
            for item in batch.items:
                self._create_item_span(item, batch_id=batch.batch_id)
            batch_span.set_status(Status(StatusCode.OK))
        logger.info("Saved batch %s to telemetry backend", batch.batch_id)

    def _create_item_span(self, item: ReviewItem, *, batch_id: str) -> None:
        """Create telemetry span for a review item"""
        metadata_batch_id = item.metadata.get("approval_batch_id")
        if metadata_batch_id is not None and metadata_batch_id != batch_id:
            raise ValueError(
                f"Approval item {item.item_id!r} belongs to batch "
                f"{metadata_batch_id!r}, not {batch_id!r}"
            )
        persisted_metadata = item.metadata | {"approval_batch_id": batch_id}
        attributes = {
            "item_id": item.item_id,
            "confidence": item.confidence,
            "status": item.status.value,
            "data": json.dumps(_serialize_for_json(item.data)),
            "metadata": json.dumps(_serialize_for_json(persisted_metadata)),
        }

        if item.created_at:
            attributes["created_at"] = item.created_at.isoformat()
        if item.reviewed_at:
            attributes["reviewed_at"] = item.reviewed_at.isoformat()

        with self.telemetry_manager.span(
            name="approval_item",
            tenant_id=self.tenant_id,
            project_name=self.project_name,
            attributes=attributes,
            require_export=True,
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
            project_spans = await self.provider.traces.get_all_spans(
                project=self.full_project_name,
                filters={
                    "name": [
                        "approval_batch",
                        "approval_item",
                        "approval_item_replacement",
                    ]
                },
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
        annotation). Raises on any malformed field."""

        def required_attribute(name: str):
            if name not in item_row:
                raise ValueError(
                    f"approval item {item_id or '<unknown>'} is missing {name}"
                )
            value = item_row[name]
            try:
                missing = bool(pd.isna(value))
            except (TypeError, ValueError):
                missing = False
            if value is None or missing:
                raise ValueError(
                    f"approval item {item_id or '<unknown>'} is missing {name}"
                )
            return value

        # In Phoenix 11.18.0, attributes are flattened as columns
        item_id = item_row.get("attributes.item_id")
        if not isinstance(item_id, str) or not item_id.strip():
            raise ValueError("approval item span has no non-empty attributes.item_id")
        item_id = item_id.strip()

        status_value = required_attribute("attributes.status")
        status = ApprovalStatus(status_value)

        created_at = required_attribute("attributes.created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        if not isinstance(created_at, datetime):
            raise ValueError(
                f"approval item {item_id} has invalid attributes.created_at"
            )

        reviewed_at = item_row.get("attributes.reviewed_at")
        if reviewed_at is not None and bool(pd.isna(reviewed_at)):
            reviewed_at = None
        if isinstance(reviewed_at, str):
            reviewed_at = datetime.fromisoformat(reviewed_at)
        if reviewed_at is not None and not isinstance(reviewed_at, datetime):
            raise ValueError(
                f"approval item {item_id} has invalid attributes.reviewed_at"
            )
        persisted_decision = None

        # Status annotations take precedence over immutable span attributes.
        if not annotations_df.empty:
            if "annotation_name" not in annotations_df.columns:
                raise ValueError(
                    f"approval item {item_id} annotations have no annotation_name"
                )
            item_annotations = annotations_df[
                (annotations_df["annotation_name"] == "item_status_update")
                & annotations_df["metadata"].apply(
                    lambda value: (
                        isinstance(value, dict) and value.get("item_id") == item_id
                    )
                )
            ]

            logger.debug(f"Item {item_id}: found {len(item_annotations)} annotations")

            if not item_annotations.empty:
                if "created_at" in item_annotations.columns:
                    latest_annotation = item_annotations.sort_values(
                        "created_at", ascending=False
                    ).iloc[0]
                else:
                    latest_annotation = item_annotations.iloc[-1]

                # telemetry annotations API returns label in 'result.label' column
                annotation_label = latest_annotation.get("result.label", "")
                if annotation_label:
                    status = ApprovalStatus(annotation_label)
                    logger.debug(
                        f"Item {item_id} status from annotation: {status.value}"
                    )

                    annotation_metadata = latest_annotation.get("metadata")
                    if not isinstance(annotation_metadata, dict):
                        raise ValueError(
                            f"approval item {item_id} annotation metadata is not an object"
                        )
                    reviewed_at_str = annotation_metadata.get("reviewed_at")
                    if reviewed_at_str:
                        reviewed_at = datetime.fromisoformat(reviewed_at_str)
                    if "decision" in annotation_metadata:
                        persisted_decision = annotation_metadata["decision"]
                        if not isinstance(persisted_decision, dict):
                            raise ValueError(
                                f"approval item {item_id} decision metadata is not an object"
                            )
                        persisted_decision = copy.deepcopy(persisted_decision)
            else:
                logger.debug(
                    f"Item {item_id}: no annotations matched, keeping span status {status.value}"
                )

        data_raw = required_attribute("attributes.data")
        data = json.loads(data_raw) if isinstance(data_raw, str) else data_raw
        if not isinstance(data, dict):
            raise ValueError(f"approval item {item_id} data is not an object")

        metadata_raw = required_attribute("attributes.metadata")
        metadata = (
            json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
        )
        if not isinstance(metadata, dict):
            raise ValueError(f"approval item {item_id} metadata is not an object")
        if persisted_decision is not None:
            metadata["decision"] = persisted_decision

        confidence_raw = required_attribute("attributes.confidence")
        if isinstance(confidence_raw, bool):
            raise ValueError(f"approval item {item_id} confidence is not numeric")
        confidence = float(confidence_raw)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError(
                f"approval item {item_id} confidence must be between 0 and 1"
            )

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

    def _reconstruct_replacement(
        self, replacement_row, annotations_df: pd.DataFrame
    ) -> ReviewItem:
        """Rebuild a regenerated item from its replacement event."""
        payload = _replacement_payload_from_record(
            replacement_row.get("attributes.replacement_record_json"),
            replacement_row.get("attributes.replacement_record_sha256"),
        )
        item_id = payload["item_id"]
        if replacement_row.get("attributes.replacement_item_id") != item_id:
            raise ValueError(
                "Replacement event item ID does not match its canonical record"
            )
        if (
            replacement_row.get("attributes.original_item_id")
            != payload["metadata"]["original_item_id"]
        ):
            raise ValueError(
                "Replacement event original item ID does not match its canonical record"
            )
        status = ApprovalStatus(payload["status"])
        created_at = datetime.fromisoformat(payload["created_at"])
        reviewed_at_value = payload["reviewed_at"]
        reviewed_at = (
            datetime.fromisoformat(reviewed_at_value) if reviewed_at_value else None
        )
        if not annotations_df.empty:
            item_annotations = annotations_df[
                annotations_df["metadata"].apply(
                    lambda value: (
                        isinstance(value, dict) and value.get("item_id") == item_id
                    )
                )
            ]
            if not item_annotations.empty:
                if "created_at" in item_annotations.columns:
                    latest = item_annotations.sort_values(
                        "created_at", ascending=False
                    ).iloc[0]
                else:
                    latest = item_annotations.iloc[-1]
                label = latest.get("result.label", "")
                if label:
                    status = ApprovalStatus(label)
                annotation_metadata = latest.get("metadata", {})
                reviewed_at_value = (
                    annotation_metadata.get("reviewed_at")
                    if isinstance(annotation_metadata, dict)
                    else None
                )
                if reviewed_at_value:
                    reviewed_at = datetime.fromisoformat(reviewed_at_value)

        return ReviewItem(
            item_id=item_id,
            data=payload["data"],
            confidence=float(payload["confidence"]),
            metadata=payload["metadata"],
            status=status,
            created_at=created_at,
            reviewed_at=reviewed_at,
        )

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

            if "attributes.batch_id" not in project_spans.columns:
                raise RuntimeError(
                    "Approval span response is missing attributes.batch_id "
                    f"for project {self.full_project_name}"
                )

            batch_spans = project_spans[
                (project_spans["name"] == "approval_batch")
                & (project_spans["attributes.batch_id"] == batch_id)
            ]

            if batch_spans.empty:
                logger.warning(
                    f"Batch {batch_id} not found in telemetry backend after retries"
                )
                return None

            root_record_columns = (
                "attributes.batch_id",
                "attributes.total_items",
                "attributes.auto_approved",
                "attributes.pending_review",
                "attributes.context",
                "attributes.created_at",
            )
            root_records = {
                tuple(row.get(column) for column in root_record_columns)
                for _, row in batch_spans.iterrows()
            }
            if len(root_records) != 1:
                raise RuntimeError(
                    f"Approval batch {batch_id!r} has conflicting root spans in "
                    f"project {self.full_project_name}"
                )

            batch_row = batch_spans.iloc[0]
            batch_span_ids = set(batch_spans["context.span_id"].tolist())

            item_spans = project_spans[
                (project_spans["name"] == "approval_item")
                & (project_spans["parent_id"].isin(batch_span_ids))
            ]
            if "start_time" in item_spans.columns:
                item_spans = item_spans.sort_values("start_time")

            replacement_spans = project_spans[
                (project_spans["name"] == "approval_item_replacement")
                & (project_spans["attributes.batch_id"] == batch_id)
            ]
            if "start_time" in replacement_spans.columns:
                replacement_spans = replacement_spans.sort_values("start_time")

            # Query annotations for the latest status of each item. Item
            # approve/reject status lives ONLY in annotations, so a telemetry
            # outage here MUST propagate to the outer handler (which raises):
            # swallowing it left annotations_df empty and rebuilt every item at
            # its span-time pending_review, silently reverting all decisions —
            # the workflow then re-prompts resolved items / sits in
            # awaiting_approval forever. A genuine absence returns an empty
            # frame (not an exception), so the propagate-on-error contract
            # matches get_pending_batches, the sibling querying the same spans.
            annotated_spans = pd.concat(
                [item_spans, replacement_spans], ignore_index=True
            )
            span_ids = annotated_spans["context.span_id"].tolist()
            logger.debug(f"Querying annotations for {len(span_ids)} spans: {span_ids}")
            annotations_df = await self.provider.annotations.get_annotations(
                spans_df=annotated_spans,
                project=self.full_project_name,
                annotation_names=["item_status_update"],
            )
            logger.info(f"Found {len(annotations_df)} annotations for batch items")
            if not annotations_df.empty:
                logger.debug(f"Annotation columns: {list(annotations_df.columns)}")

            items = []
            items_by_id: Dict[str, ReviewItem] = {}
            for _, item_row in item_spans.iterrows():
                try:
                    item = self._reconstruct_item(item_row, annotations_df)
                except Exception as item_exc:
                    item_id = item_row.get("attributes.item_id", "<unknown>")
                    raise RuntimeError(
                        f"Approval batch {batch_id!r} contains malformed item "
                        f"{item_id!r}"
                    ) from item_exc
                existing_item = items_by_id.get(item.item_id)
                if existing_item is not None:
                    if existing_item != item:
                        raise RuntimeError(
                            f"Approval batch {batch_id!r} contains conflicting retry "
                            f"records for item {item.item_id!r}"
                        )
                    continue
                items.append(item)
                items_by_id[item.item_id] = item

            reconstructed_replacements = []
            replacement_json_by_original: Dict[str, str] = {}
            for _, replacement_row in replacement_spans.iterrows():
                original_item_id = replacement_row.get("attributes.original_item_id")
                try:
                    replacement = self._reconstruct_replacement(
                        replacement_row, annotations_df
                    )
                except Exception as replacement_exc:
                    raise RuntimeError(
                        f"Approval batch {batch_id!r} contains malformed replacement "
                        f"for original item {original_item_id!r}"
                    ) from replacement_exc
                record_json = replacement_row["attributes.replacement_record_json"]
                previous_json = replacement_json_by_original.get(original_item_id)
                if previous_json is not None:
                    if previous_json != record_json:
                        raise RuntimeError(
                            f"Approval batch {batch_id!r} contains conflicting "
                            "replacement events for original item "
                            f"{original_item_id!r}"
                        )
                    continue
                replacement_json_by_original[original_item_id] = record_json
                reconstructed_replacements.append((original_item_id, replacement))

            for original_item_id, replacement in reconstructed_replacements:
                if original_item_id not in items_by_id:
                    raise ValueError(
                        "Replacement event references unknown original item "
                        f"{original_item_id!r} in batch {batch_id!r}"
                    )
                existing = items_by_id.get(replacement.item_id)
                if existing is not None and existing != replacement:
                    raise ValueError(
                        "Replacement item ID conflicts with existing item "
                        f"{replacement.item_id!r} in batch {batch_id!r}"
                    )

                items_by_id[original_item_id].status = ApprovalStatus.REJECTED
                if existing is None:
                    items.append(replacement)
                    items_by_id[replacement.item_id] = replacement

            if "attributes.context" not in batch_row:
                raise RuntimeError(
                    f"Approval batch {batch_id!r} is missing attributes.context"
                )
            context_raw = batch_row["attributes.context"]
            context = (
                json.loads(context_raw) if isinstance(context_raw, str) else context_raw
            )
            if not isinstance(context, dict):
                raise RuntimeError(
                    f"Approval batch {batch_id!r} context is not an object"
                )
            created_at_raw = batch_row.get("attributes.created_at")
            if isinstance(created_at_raw, str):
                created_at = datetime.fromisoformat(created_at_raw)
                if created_at.isoformat() != created_at_raw:
                    raise RuntimeError(
                        f"Approval batch {batch_id!r} created_at is not canonical"
                    )
            elif isinstance(created_at_raw, datetime):
                created_at = created_at_raw
            else:
                raise RuntimeError(
                    f"Approval batch {batch_id!r} has invalid created_at"
                )
            if created_at.tzinfo is None or created_at.utcoffset() is None:
                raise RuntimeError(f"Approval batch {batch_id!r} has naive created_at")

            batch = ApprovalBatch(
                batch_id=batch_id,
                items=items,
                context=context,
                created_at=created_at,
            )

            logger.info(
                f"Retrieved batch {batch_id} from telemetry backend with {len(items)} items (status from annotations)"
            )
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
        span_id = await self.get_item_span_id(item.item_id, batch_id=batch_id)

        if not span_id:
            logger.error(f"Cannot update item {item.item_id}: span not found")
            raise ValueError(f"Span not found for item {item.item_id}")

        try:
            metadata = {
                "item_id": item.item_id,
                "confidence": item.confidence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if item.reviewed_at:
                metadata["reviewed_at"] = item.reviewed_at.isoformat()
            if "decision" in item.metadata:
                decision = item.metadata["decision"]
                if not isinstance(decision, dict):
                    raise ValueError(
                        f"Approval item {item.item_id} decision metadata must be an object"
                    )
                metadata["decision"] = copy.deepcopy(decision)

            logger.info(
                f"Creating annotation for item {item.item_id} (status={item.status.value}) on span {span_id}"
            )
            await self.provider.annotations.add_annotation(
                span_id=span_id,
                name="item_status_update",
                label=item.status.value,  # "approved", "rejected", etc.
                score=(
                    1.0
                    if item.status
                    in {ApprovalStatus.APPROVED, ApprovalStatus.AUTO_APPROVED}
                    else 0.0
                ),
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

    async def _replacement_event_exists(
        self,
        *,
        batch_id: str,
        original_item_id: str,
        selected_json: str,
        selected_sha256: str,
    ) -> bool:
        spans = await self.provider.traces.get_all_spans(
            project=self.full_project_name,
            filters={"name": "approval_item_replacement"},
        )
        if spans.empty:
            return False
        required_columns = {
            "attributes.batch_id",
            "attributes.original_item_id",
            "attributes.replacement_item_id",
            "attributes.replacement_record_json",
            "attributes.replacement_record_sha256",
        }
        missing_columns = required_columns.difference(spans.columns)
        if missing_columns:
            raise RuntimeError(
                "Approval replacement span response is missing required columns "
                f"{sorted(missing_columns)}"
            )
        matches = spans[
            (spans["attributes.batch_id"] == batch_id)
            & (spans["attributes.original_item_id"] == original_item_id)
        ]
        if matches.empty:
            return False

        records = []
        for _, row in matches.iterrows():
            payload = _replacement_payload_from_record(
                row["attributes.replacement_record_json"],
                row["attributes.replacement_record_sha256"],
            )
            if row["attributes.replacement_item_id"] != payload["item_id"]:
                raise RuntimeError(
                    "Replacement event item ID does not match its canonical record"
                )
            if payload["metadata"]["original_item_id"] != original_item_id:
                raise RuntimeError(
                    "Replacement event original item ID does not match its "
                    "canonical record"
                )
            records.append(
                (
                    row["attributes.replacement_record_json"],
                    row["attributes.replacement_record_sha256"],
                )
            )

        if len(set(records)) != 1:
            raise RuntimeError(
                f"Found conflicting replacement events for batch={batch_id} "
                f"original={original_item_id}"
            )
        if records[0] != (selected_json, selected_sha256):
            raise RuntimeError(
                f"Found conflicting replacement event for batch={batch_id} "
                f"original={original_item_id}"
            )
        return True

    async def replace_item(
        self,
        batch_id: str,
        original: ReviewItem,
        replacement: ReviewItem,
    ) -> None:
        """Select one replacement in Redis, then persist it in Phoenix."""
        operation = (
            f"batch={batch_id} original={original.item_id} "
            f"replacement={replacement.item_id}"
        )
        if replacement.item_id == original.item_id:
            raise ValueError(f"Replacement must have a new item ID: {operation}")
        if replacement.status is not ApprovalStatus.REGENERATED:
            raise ValueError(f"Replacement must be regenerated: {operation}")
        if replacement.metadata.get("original_item_id") != original.item_id:
            raise ValueError(
                f"Replacement metadata must identify the original item: {operation}"
            )
        decision = replacement.metadata.get("decision")
        required_decision_fields = {
            "reviewer",
            "feedback",
            "corrections",
            "timestamp",
        }
        if not isinstance(decision, dict) or set(decision) != required_decision_fields:
            raise ValueError(
                f"Replacement metadata must contain the complete decision: {operation}"
            )

        canonical_decision = await self.select_review_decision(
            batch_id=batch_id,
            original_item_id=original.item_id,
            decision=ReviewDecision(
                item_id=original.item_id,
                approved=False,
                feedback=decision["feedback"],
                corrections=copy.deepcopy(decision["corrections"]),
                reviewer=decision["reviewer"],
                timestamp=datetime.fromisoformat(decision["timestamp"]),
            ),
        )
        replacement.metadata["decision"] = {
            "reviewer": canonical_decision.reviewer,
            "feedback": canonical_decision.feedback,
            "corrections": copy.deepcopy(canonical_decision.corrections),
            "timestamp": canonical_decision.timestamp.isoformat(),
        }

        if self._replacement_records is None:
            raise RuntimeError(
                f"redis_url is required for approval item replacement: {operation}"
            )
        selected_record = await self._replacement_records.select_canonical(
            tenant_id=self.tenant_id,
            batch_id=batch_id,
            original_item_id=original.item_id,
            candidate=_replacement_payload(replacement),
        )
        try:
            replacement = _replacement_from_payload(selected_record.payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Canonical replacement record is invalid: {operation}"
            ) from exc
        if replacement.status is not ApprovalStatus.REGENERATED:
            raise RuntimeError(f"Canonical replacement is not regenerated: {operation}")
        if replacement.metadata.get("original_item_id") != original.item_id:
            raise RuntimeError(
                f"Canonical replacement identifies a different original: {operation}"
            )

        operation = (
            f"batch={batch_id} original={original.item_id} "
            f"replacement={replacement.item_id}"
        )

        attributes = {
            "batch_id": batch_id,
            "original_item_id": original.item_id,
            "replacement_item_id": replacement.item_id,
            "replacement_record_json": selected_record.json,
            "replacement_record_sha256": selected_record.sha256,
        }

        try:
            async with self._replacement_records.replacement_event_lock(
                tenant_id=self.tenant_id,
                batch_id=batch_id,
                original_item_id=original.item_id,
            ):
                if await self._replacement_event_exists(
                    batch_id=batch_id,
                    original_item_id=original.item_id,
                    selected_json=selected_record.json,
                    selected_sha256=selected_record.sha256,
                ):
                    return

                await asyncio.to_thread(self._write_replacement_span, attributes)

                if await self._wait_for_replacement_event(
                    batch_id=batch_id,
                    original_item_id=original.item_id,
                    selected_json=selected_record.json,
                    selected_sha256=selected_record.sha256,
                ):
                    return
        except Exception as exc:
            raise RuntimeError(f"Failed to persist replacement: {operation}") from exc

        raise RuntimeError(f"Replacement was not visible after export: {operation}")

    def _write_replacement_span(self, attributes: Dict[str, Any]) -> None:
        with self.telemetry_manager.span(
            name="approval_item_replacement",
            tenant_id=self.tenant_id,
            project_name=self.project_name,
            attributes=attributes,
            require_export=True,
        ) as replacement_span:
            replacement_span.set_status(Status(StatusCode.OK))

    async def _wait_for_replacement_event(
        self,
        *,
        batch_id: str,
        original_item_id: str,
        selected_json: str,
        selected_sha256: str,
    ) -> bool:
        for delay in (0, 0.25, 0.5, 1, 2, 4):
            if delay:
                await asyncio.sleep(delay)
            if await self._replacement_event_exists(
                batch_id=batch_id,
                original_item_id=original_item_id,
                selected_json=selected_json,
                selected_sha256=selected_sha256,
            ):
                return True
        return False

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
            spans_df = await self.provider.traces.get_all_spans(
                project=self.full_project_name,
                filters={
                    "name": [
                        "approval_batch",
                        "approval_item",
                        "approval_item_replacement",
                    ]
                },
            )

            if spans_df.empty:
                return []

            if (
                "attributes.batch_id" not in spans_df.columns
                or "attributes.pending_review" not in spans_df.columns
            ):
                raise RuntimeError(
                    "Approval span query omitted required batch_id or pending_review "
                    "attributes"
                )

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
                if batch and batch.pending_review:
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
            max_retries = 3
            retry_delays = [0.5, 1, 2]  # seconds

            for attempt, delay in enumerate(retry_delays):
                span_names = (
                    ["approval_batch", "approval_item"]
                    if batch_id is not None
                    else "approval_item"
                )
                project_spans = await self.provider.traces.get_all_spans(
                    project=self.full_project_name,
                    filters={"name": span_names},
                )

                if not project_spans.empty:
                    required_columns = {
                        "attributes.item_id",
                        "context.span_id",
                        "name",
                        "start_time",
                    }
                    if batch_id is not None:
                        required_columns.update({"attributes.batch_id", "parent_id"})
                    missing_columns = required_columns.difference(project_spans.columns)
                    if missing_columns:
                        raise RuntimeError(
                            "Approval span response is missing required columns "
                            f"{sorted(missing_columns)}"
                        )

                    item_spans = project_spans[
                        (project_spans["name"] == "approval_item")
                        & (project_spans["attributes.item_id"] == item_id)
                    ]
                    if batch_id is not None:
                        batch_spans = project_spans[
                            (project_spans["name"] == "approval_batch")
                            & (project_spans["attributes.batch_id"] == batch_id)
                        ]
                        batch_span_ids = set(batch_spans["context.span_id"])
                        item_spans = item_spans[
                            item_spans["parent_id"].isin(batch_span_ids)
                        ]

                    if not item_spans.empty:
                        latest_span = item_spans.sort_values(
                            "start_time", ascending=False
                        ).iloc[0]
                        span_id = latest_span["context.span_id"]
                        logger.debug(
                            f"Found span {span_id} for item {item_id} on attempt {attempt + 1}"
                        )
                        return span_id

                replacement_spans = await self.provider.traces.get_all_spans(
                    project=self.full_project_name,
                    filters={"name": "approval_item_replacement"},
                )
                if not replacement_spans.empty:
                    required_replacement_columns = {
                        "attributes.replacement_item_id",
                        "attributes.batch_id",
                        "context.span_id",
                        "start_time",
                    }
                    missing_columns = required_replacement_columns.difference(
                        replacement_spans.columns
                    )
                    if missing_columns:
                        raise RuntimeError(
                            "Approval replacement span response is missing required "
                            f"columns {sorted(missing_columns)}"
                        )
                    matches = replacement_spans[
                        replacement_spans["attributes.replacement_item_id"] == item_id
                    ]
                    if batch_id is not None:
                        matches = matches[matches["attributes.batch_id"] == batch_id]
                    if not matches.empty:
                        latest_span = matches.sort_values(
                            "start_time", ascending=False
                        ).iloc[0]
                        span_id = latest_span["context.span_id"]
                        logger.debug(
                            "Found replacement span %s for item %s on attempt %d",
                            span_id,
                            item_id,
                            attempt + 1,
                        )
                        return span_id

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
        decision_timestamp: Optional[datetime] = None,
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
            decision_timestamp: Exact time at which the reviewer decided

        Returns:
            True when the annotation persisted. A write failure RAISES: the
            reviewer identity and feedback live only in this annotation (the
            item-status update carries neither), so swallowing the failure
            silently drops the reviewer history while the decision reads
            as applied.
        """
        try:
            recorded_at = decision_timestamp or datetime.now(timezone.utc)
            metadata = {
                "item_id": item_id,
                "timestamp": recorded_at.isoformat(),
                "reviewed_at": recorded_at.isoformat(),
            }
            if reviewer:
                metadata["reviewer"] = reviewer
            if feedback:
                metadata["feedback"] = feedback

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
            raise

    _DATASET_LOCK_LEASE_MS = 120_000
    _DATASET_LOCK_WAIT_SECONDS = 10.0
    _DATASET_LOCK_POLL_SECONDS = 0.05
    _DATASET_LOCK_SOCKET_TIMEOUT_SECONDS = 2.0
    _DATASET_LOCK_RENEW_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
end
return 0
""".strip()
    _DATASET_LOCK_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
""".strip()

    async def _renew_approval_dataset_lock(
        self,
        *,
        client: Any,
        lock_key: str,
        token: str,
        operation: str,
        owner_task: asyncio.Task,
        stop: asyncio.Event,
        failure: Dict[str, BaseException],
    ) -> None:
        renewal_interval = self._DATASET_LOCK_LEASE_MS / 3000
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=renewal_interval)
                return
            except TimeoutError:
                pass

            try:
                renewed = await client.eval(
                    self._DATASET_LOCK_RENEW_SCRIPT,
                    1,
                    lock_key,
                    token,
                    self._DATASET_LOCK_LEASE_MS,
                )
            except Exception as exc:
                failure["error"] = exc
                owner_task.cancel()
                return

            if renewed != 1:
                failure["error"] = _ApprovalDatasetLockOwnershipLost(
                    "Approved dataset lock ownership was lost during renewal: "
                    f"{operation}"
                )
                owner_task.cancel()
                return

    @staticmethod
    def _raise_dataset_lock_renewal_failure(
        failure: Dict[str, BaseException],
        operation: str,
    ) -> None:
        renewal_error = failure.get("error")
        if renewal_error is None:
            return
        if isinstance(renewal_error, _ApprovalDatasetLockOwnershipLost):
            raise renewal_error
        raise RuntimeError(
            f"Failed to renew approved dataset lock: {operation}"
        ) from renewal_error

    @asynccontextmanager
    async def _approval_dataset_lock(self, dataset_name: str):
        """Hold a renewable Redis lock for one tenant-qualified dataset write."""
        operation = f"tenant={self.tenant_id} dataset={dataset_name}"
        if not self.redis_url:
            raise RuntimeError(
                f"redis_url is required for approved dataset writes: {operation}"
            )

        lock_key = f"cogniverse:approval:dataset-lock:{self.tenant_id}:{dataset_name}"
        token = secrets.token_hex(16)
        client = aioredis.from_url(
            self.redis_url,
            decode_responses=True,
            socket_connect_timeout=self._DATASET_LOCK_SOCKET_TIMEOUT_SECONDS,
            socket_timeout=self._DATASET_LOCK_SOCKET_TIMEOUT_SECONDS,
            retry_on_timeout=False,
        )
        acquired = False
        deadline = asyncio.get_running_loop().time() + self._DATASET_LOCK_WAIT_SECONDS
        renewal_task = None
        renewal_stop = asyncio.Event()
        renewal_failure: Dict[str, BaseException] = {}
        try:
            while not acquired:
                try:
                    acquired = bool(
                        await client.set(
                            lock_key,
                            token,
                            nx=True,
                            px=self._DATASET_LOCK_LEASE_MS,
                        )
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to acquire approved dataset lock: {operation}"
                    ) from exc
                if acquired:
                    break
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError(
                        f"Timed out acquiring approved dataset lock: {operation}"
                    )
                await asyncio.sleep(self._DATASET_LOCK_POLL_SECONDS)

            owner_task = asyncio.current_task()
            if owner_task is None:
                raise RuntimeError(
                    f"Approved dataset lock has no owning task: {operation}"
                )
            renewal_task = asyncio.create_task(
                self._renew_approval_dataset_lock(
                    client=client,
                    lock_key=lock_key,
                    token=token,
                    operation=operation,
                    owner_task=owner_task,
                    stop=renewal_stop,
                    failure=renewal_failure,
                )
            )
            operation_failed = False
            try:
                try:
                    yield
                except asyncio.CancelledError:
                    self._raise_dataset_lock_renewal_failure(
                        renewal_failure,
                        operation,
                    )
                    raise
                self._raise_dataset_lock_renewal_failure(
                    renewal_failure,
                    operation,
                )
            except BaseException:
                operation_failed = True
                raise
            finally:
                renewal_stop.set()
                if renewal_task is not None:
                    if not renewal_task.done():
                        renewal_task.cancel()
                    try:
                        await renewal_task
                    except asyncio.CancelledError:
                        pass
                try:
                    released = await client.eval(
                        self._DATASET_LOCK_RELEASE_SCRIPT,
                        1,
                        lock_key,
                        token,
                    )
                    if released != 1:
                        raise RuntimeError(
                            f"Approved dataset lock ownership was lost: {operation}"
                        )
                except Exception as exc:
                    if operation_failed:
                        logger.exception(
                            "Failed to release approved dataset lock after an error: %s",
                            operation,
                        )
                    else:
                        raise RuntimeError(
                            f"Failed to release approved dataset lock: {operation}"
                        ) from exc
        finally:
            await client.aclose()

    @staticmethod
    def _approved_item_copy(item: ReviewItem, decision: ReviewDecision) -> ReviewItem:
        if not decision.approved:
            raise ValueError("persist_approved_item requires an approved decision")
        if decision.item_id != item.item_id:
            raise ValueError(
                "Approval decision item does not match review item: "
                f"decision={decision.item_id} item={item.item_id}"
            )
        if not isinstance(decision.timestamp, datetime):
            raise ValueError("Review decision timestamp is required")
        if decision.timestamp.tzinfo is None or decision.timestamp.utcoffset() is None:
            raise ValueError(
                "Review decision timestamp must include timezone information"
            )

        approved = copy.deepcopy(item)
        approved.status = (
            ApprovalStatus.AUTO_APPROVED
            if item.status is ApprovalStatus.AUTO_APPROVED
            else ApprovalStatus.APPROVED
        )
        approved.reviewed_at = decision.timestamp
        approved.metadata["decision"] = {
            "reviewer": decision.reviewer,
            "feedback": decision.feedback,
            "corrections": copy.deepcopy(decision.corrections),
            "timestamp": decision.timestamp.isoformat(),
        }
        return approved

    @staticmethod
    def _training_dataset_record(
        item: ReviewItem,
        project_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if item.status is not ApprovalStatus.APPROVED:
            raise ValueError(f"Training dataset item {item.item_id!r} must be approved")
        if item.reviewed_at is None:
            raise ValueError(
                f"Training dataset item {item.item_id!r} requires reviewed_at"
            )
        _validate_training_item_schema(item)

        if "confidence" in item.data:
            data_confidence = item.data["confidence"]
            if (
                isinstance(data_confidence, bool)
                or not isinstance(data_confidence, float)
                or not math.isfinite(data_confidence)
                or not math.isclose(
                    data_confidence,
                    item.confidence,
                    rel_tol=0.0,
                    abs_tol=math.ulp(data_confidence),
                )
            ):
                raise ValueError(
                    f"Training dataset item {item.item_id!r} data confidence "
                    "must exactly match ReviewItem.confidence: "
                    f"data={data_confidence!r} item={item.confidence!r}"
                )

        reserved = {"item_id", "status", "created_at", "reviewed_at"}
        invalid_data_keys = {
            key
            for key in item.data
            if key in reserved
            or key.startswith("metadata.")
            or key.startswith("context.")
        }
        if invalid_data_keys:
            raise ValueError(
                f"Training dataset item {item.item_id!r} uses reserved fields "
                f"{sorted(invalid_data_keys)}"
            )

        record = {
            "item_id": item.item_id,
            "confidence": item.confidence,
            "status": item.status.value,
            "created_at": item.created_at.isoformat(),
            "reviewed_at": item.reviewed_at.isoformat(),
            **_serialize_for_json(item.data),
            **{
                f"metadata.{key}": _serialize_for_json(value)
                for key, value in item.metadata.items()
            },
            **{
                f"context.{key}": _serialize_for_json(value)
                for key, value in (project_context or {}).items()
            },
        }
        decision = record.get("metadata.decision")
        decision_intent = copy.deepcopy(decision)
        if isinstance(decision_intent, dict):
            decision_timestamp = decision_intent.pop("timestamp", None)
            if (
                decision_timestamp is not None
                and decision_timestamp != record["reviewed_at"]
            ):
                raise ValueError(
                    f"Training dataset item {item.item_id!r} decision timestamp "
                    "must match reviewed_at"
                )
        identity = {
            "item_id": item.item_id,
            "status": item.status.value,
            "decision": decision_intent,
        }
        identity_json = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        record["metadata.approval_decision_sha256"] = hashlib.sha256(
            identity_json.encode("utf-8")
        ).hexdigest()
        record["metadata.approval_decision_timestamp"] = record["reviewed_at"]
        ApprovalStorageImpl._set_approval_record_digest(record)
        return record

    @staticmethod
    def _set_approval_record_digest(record: Dict[str, Any]) -> None:
        digest_input = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "metadata.approval_record_json",
                "metadata.approval_record_sha256",
            }
        }
        canonical = _canonical_approval_json(digest_input)
        record["metadata.approval_record_json"] = canonical
        record["metadata.approval_record_sha256"] = hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _approval_decision_timestamp(record: Dict[str, Any], item_id: str) -> datetime:
        value = record.get("metadata.approval_decision_timestamp")
        if not isinstance(value, str):
            raise RuntimeError(
                "Approved dataset item has no canonical decision timestamp: "
                f"item={item_id}"
            )
        try:
            timestamp = datetime.fromisoformat(value)
        except ValueError as exc:
            raise RuntimeError(
                "Approved dataset item has invalid canonical decision timestamp: "
                f"item={item_id}"
            ) from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise RuntimeError(
                "Approved dataset item has naive canonical decision timestamp: "
                f"item={item_id}"
            )
        return timestamp

    async def _append_training_records_locked(
        self,
        dataset_name: str,
        items: List[ReviewItem],
        project_context: Optional[Dict[str, Any]],
    ) -> Dict[str, datetime]:
        from cogniverse_foundation.telemetry.providers.base import DatasetNotFoundError

        if not items:
            return {}

        records = [
            self._training_dataset_record(item, project_context) for item in items
        ]
        records_by_id = {record["item_id"]: record for record in records}
        if len(records_by_id) != len(records):
            raise ValueError(
                f"Approved dataset write contains duplicate item IDs: "
                f"tenant={self.tenant_id} dataset={dataset_name}"
            )

        dataset_exists = True
        try:
            existing_frame = await self.provider.datasets.get_dataset(name=dataset_name)
        except DatasetNotFoundError:
            dataset_exists = False
            existing_frame = None
        if dataset_exists:
            existing_frame = _validated_approved_dataset_snapshot(
                existing_frame,
                tenant_id=self.tenant_id,
                dataset_name=dataset_name,
            )

        missing_records = []
        canonical_timestamps: Dict[str, datetime] = {}
        if not dataset_exists:
            missing_records = records
            canonical_timestamps = {
                item_id: self._approval_decision_timestamp(record, item_id)
                for item_id, record in records_by_id.items()
            }
        else:
            existing_by_id: Dict[str, List[Dict[str, Any]]] = {}
            for position, (_, row) in enumerate(existing_frame.iterrows()):
                existing = row.get("input")
                if not isinstance(existing, dict):
                    raise RuntimeError(
                        "Approved dataset row is not an input record: "
                        f"tenant={self.tenant_id} dataset={dataset_name} "
                        f"row={position}"
                    )
                item_id = existing.get("item_id")
                if not isinstance(item_id, str) or not item_id:
                    raise RuntimeError(
                        "Approved dataset row has no item_id: "
                        f"tenant={self.tenant_id} dataset={dataset_name} "
                        f"row={position}"
                    )
                existing_by_id.setdefault(item_id, []).append(existing)

            for item_id, record in records_by_id.items():
                matches = existing_by_id.get(item_id, [])
                if not matches:
                    missing_records.append(record)
                    canonical_timestamps[item_id] = self._approval_decision_timestamp(
                        record, item_id
                    )
                    continue
                if len(matches) != 1:
                    raise RuntimeError(
                        "Approved dataset contains duplicate item records: "
                        f"tenant={self.tenant_id} dataset={dataset_name} "
                        f"item={item_id} count={len(matches)}"
                    )
                existing = matches[0]
                existing_digest = existing.get("metadata.approval_record_sha256")
                if existing_digest == record["metadata.approval_record_sha256"]:
                    canonical_timestamps[item_id] = self._approval_decision_timestamp(
                        existing, item_id
                    )
                    continue

                raise RuntimeError(
                    "Approved dataset item conflicts with immutable record: "
                    f"tenant={self.tenant_id} dataset={dataset_name} "
                    f"item={item_id}"
                )

        if not missing_records:
            return canonical_timestamps

        frame = pd.DataFrame(missing_records)
        if not dataset_exists:
            await self.provider.datasets.create_dataset(name=dataset_name, data=frame)
        else:
            await self.provider.datasets.append_to_dataset(
                name=dataset_name,
                data=frame,
            )
        return canonical_timestamps

    async def persist_approved_item(
        self,
        *,
        batch_id: str,
        dataset_name: str,
        item: ReviewItem,
        decision: ReviewDecision,
        project_context: Optional[Dict[str, Any]] = None,
    ) -> ReviewItem:
        """Persist one approval in retry-safe dataset-first order.

        The returned item preserves ``AUTO_APPROVED`` for threshold decisions
        and uses ``APPROVED`` for human decisions; the caller's item is never
        mutated. The training dataset always stores the canonical ``APPROVED``
        status consumed by finetuning. Its write is idempotent by immutable item
        ID and record digest. Reviewer and status annotations are emitted only
        after the dataset contains that exact record.
        """
        operation = (
            f"tenant={self.tenant_id} dataset={dataset_name} "
            f"batch={batch_id} item={item.item_id}"
        )
        try:
            canonical_decision = await self.select_review_decision(
                batch_id=batch_id,
                original_item_id=item.item_id,
                decision=decision,
            )
            approved = self._approved_item_copy(item, canonical_decision)
            training_item = copy.deepcopy(approved)
            training_item.status = ApprovalStatus.APPROVED
            async with self._approval_dataset_lock(dataset_name):
                canonical_timestamps = await self._append_training_records_locked(
                    dataset_name,
                    [training_item],
                    project_context,
                )
                canonical_timestamp = canonical_timestamps[approved.item_id]
                if canonical_timestamp != canonical_decision.timestamp:
                    raise RuntimeError(
                        "Approved dataset decision timestamp differs from Redis "
                        f"selection: {operation}"
                    )
                span_id = await self.get_item_span_id(
                    approved.item_id,
                    batch_id=batch_id,
                )
                if not span_id:
                    raise RuntimeError(
                        f"Approval item span is not visible: {operation}"
                    )
                await self.log_approval_decision(
                    span_id=span_id,
                    item_id=approved.item_id,
                    approved=True,
                    feedback=canonical_decision.feedback,
                    reviewer=canonical_decision.reviewer,
                    decision_timestamp=canonical_decision.timestamp,
                )
                await self.update_item(approved, batch_id=batch_id)
            return approved
        except Exception as exc:
            raise RuntimeError(f"Failed to persist approved item: {operation}") from exc

    async def select_review_decision(
        self,
        *,
        batch_id: str,
        original_item_id: str,
        decision: ReviewDecision,
    ) -> ReviewDecision:
        if decision.item_id != original_item_id:
            raise ValueError(
                "Review decision item does not match original item: "
                f"decision={decision.item_id} original={original_item_id}"
            )
        if self._replacement_records is None:
            raise RuntimeError(
                "redis_url is required to select a canonical review decision: "
                f"tenant={self.tenant_id} batch={batch_id} item={original_item_id}"
            )
        selected_record = await self._replacement_records.select_review_decision(
            tenant_id=self.tenant_id,
            batch_id=batch_id,
            original_item_id=original_item_id,
            candidate=_review_decision_payload(decision),
        )
        canonical = _review_decision_from_payload(selected_record.payload)
        decision.item_id = canonical.item_id
        decision.approved = canonical.approved
        decision.reviewer = canonical.reviewer
        decision.feedback = canonical.feedback
        decision.corrections = copy.deepcopy(canonical.corrections)
        decision.timestamp = canonical.timestamp
        return decision

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
            True if all exact records exist after the call. An empty input
            returns False. Redis, dataset, and immutable-record conflicts raise.
        """
        expected_dataset_name = approved_synthetic_dataset_name(self.tenant_id)
        if dataset_name != expected_dataset_name:
            raise ValueError(
                f"Approval dataset name must be {expected_dataset_name!r}, "
                f"got {dataset_name!r}"
            )
        if not items:
            return False
        try:
            async with self._approval_dataset_lock(dataset_name):
                await self._append_training_records_locked(
                    dataset_name,
                    items,
                    project_context,
                )
                return True
        except Exception as exc:
            logger.error(
                "Failed to append items to training dataset: tenant=%s dataset=%s",
                self.tenant_id,
                dataset_name,
                exc_info=True,
            )
            raise RuntimeError(
                "Failed to append items to training dataset: "
                f"tenant={self.tenant_id} dataset={dataset_name}"
            ) from exc
