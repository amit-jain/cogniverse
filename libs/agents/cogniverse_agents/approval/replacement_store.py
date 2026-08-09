"""Redis-backed selection of one canonical regenerated approval item."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

import redis.asyncio as aioredis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True)
class CanonicalReplacementRecord:
    """Strict JSON bytes and digest for one selected replacement payload."""

    payload: dict[str, Any]
    json: str
    sha256: str


class RedisReplacementRecordStore:
    """Persist the first replacement payload selected across all replicas."""

    _EVENT_LOCK_LEASE_MS = 30_000
    _EVENT_LOCK_WAIT_SECONDS = 30.0
    _REVIEW_DECISION_FIELDS = {
        "item_id",
        "approved",
        "reviewer",
        "feedback",
        "corrections",
        "timestamp",
    }
    _APPROVAL_BATCH_FIELDS = {"batch_id", "context", "created_at", "items"}
    _APPROVAL_ITEM_FIELDS = {
        "item_id",
        "data",
        "confidence",
        "status",
        "metadata",
        "created_at",
        "reviewed_at",
    }

    def __init__(self, redis_url: str) -> None:
        if not redis_url.strip():
            raise ValueError("redis_url must be non-empty")
        self._redis_url = redis_url

    @staticmethod
    def _key(tenant_id: str, batch_id: str, original_item_id: str) -> str:
        identity = json.dumps(
            [tenant_id, batch_id, original_item_id],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return f"cogniverse:approval:replacement:{digest}"

    @staticmethod
    def _decision_key(tenant_id: str, batch_id: str, original_item_id: str) -> str:
        identity = json.dumps(
            [tenant_id, batch_id, original_item_id],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return f"cogniverse:approval:decision:{digest}"

    @staticmethod
    def _approval_batch_key(tenant_id: str, batch_id: str) -> str:
        identity = json.dumps(
            [tenant_id, batch_id],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return f"cogniverse:approval:batch:{digest}"

    @classmethod
    def _event_lock_key(
        cls, tenant_id: str, batch_id: str, original_item_id: str
    ) -> str:
        return f"{cls._key(tenant_id, batch_id, original_item_id)}:event-lock"

    @staticmethod
    def _operation(tenant_id: str, batch_id: str, original_item_id: str) -> str:
        identities = {
            "tenant_id": tenant_id,
            "batch_id": batch_id,
            "original_item_id": original_item_id,
        }
        for name, value in identities.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        return f"tenant={tenant_id} batch={batch_id} original={original_item_id}"

    @staticmethod
    def _batch_operation(tenant_id: str, batch_id: str) -> str:
        for name, value in {"tenant_id": tenant_id, "batch_id": batch_id}.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        return f"tenant={tenant_id} batch={batch_id}"

    @staticmethod
    def _decode_record(
        payload: str,
        *,
        operation: str,
    ) -> CanonicalReplacementRecord:
        try:
            selected = json.loads(
                payload,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_strict_object_pairs,
            )
            if not isinstance(selected, dict):
                raise ValueError("replacement payload must be an object")
            canonical = json.dumps(
                selected,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if canonical != payload:
                raise ValueError("replacement payload is not canonical JSON")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Stored canonical replacement is invalid for {operation}"
            ) from exc
        return CanonicalReplacementRecord(
            payload=selected,
            json=canonical,
            sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    @classmethod
    def _review_decision_intent(
        cls,
        payload: Mapping[str, Any],
        *,
        operation: str,
        original_item_id: str,
        stored: bool,
    ) -> dict[str, Any]:
        invalid = (
            "Stored canonical review decision is invalid"
            if stored
            else "Review decision candidate is invalid"
        )
        error_type = RuntimeError if stored else ValueError
        if set(payload) != cls._REVIEW_DECISION_FIELDS:
            raise error_type(f"{invalid} for {operation}: fields must be exact")
        if payload["item_id"] != original_item_id:
            raise error_type(f"{invalid} for {operation}: item_id does not match")
        if not isinstance(payload["approved"], bool):
            raise error_type(f"{invalid} for {operation}: approved must be boolean")
        if payload["reviewer"] is not None and not isinstance(payload["reviewer"], str):
            raise error_type(f"{invalid} for {operation}: reviewer must be a string")
        if payload["feedback"] is not None and not isinstance(payload["feedback"], str):
            raise error_type(f"{invalid} for {operation}: feedback must be a string")
        if not isinstance(payload["corrections"], dict):
            raise error_type(
                f"{invalid} for {operation}: corrections must be an object"
            )
        timestamp = payload["timestamp"]
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp)
        except (TypeError, ValueError) as exc:
            raise error_type(
                f"{invalid} for {operation}: timestamp must be canonical ISO format"
            ) from exc
        if (
            parsed_timestamp.tzinfo is None
            or parsed_timestamp.utcoffset() is None
            or parsed_timestamp.isoformat() != timestamp
        ):
            raise error_type(
                f"{invalid} for {operation}: timestamp must be canonical and aware"
            )
        return {
            key: payload[key] for key in cls._REVIEW_DECISION_FIELDS - {"timestamp"}
        }

    @staticmethod
    def _require_canonical_timestamp(
        value: Any,
        *,
        field: str,
        invalid: str,
        operation: str,
        error_type: type[ValueError] | type[RuntimeError],
    ) -> None:
        try:
            parsed = datetime.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise error_type(
                f"{invalid} for {operation}: {field} must be canonical ISO format"
            ) from exc
        if (
            parsed.tzinfo is None
            or parsed.utcoffset() is None
            or parsed.isoformat() != value
        ):
            raise error_type(
                f"{invalid} for {operation}: {field} must be canonical and aware"
            )

    @classmethod
    def _approval_batch_intent(
        cls,
        payload: Mapping[str, Any],
        *,
        batch_id: str,
        operation: str,
        stored: bool,
    ) -> dict[str, Any]:
        invalid = (
            "Stored canonical approval batch is invalid"
            if stored
            else "Approval batch candidate is invalid"
        )
        error_type = RuntimeError if stored else ValueError
        if set(payload) != cls._APPROVAL_BATCH_FIELDS:
            raise error_type(f"{invalid} for {operation}: fields must be exact")
        if payload["batch_id"] != batch_id:
            raise error_type(f"{invalid} for {operation}: batch_id does not match")
        if not isinstance(payload["context"], dict):
            raise error_type(f"{invalid} for {operation}: context must be an object")
        cls._require_canonical_timestamp(
            payload["created_at"],
            field="created_at",
            invalid=invalid,
            operation=operation,
            error_type=error_type,
        )
        items = payload["items"]
        if not isinstance(items, list) or not items:
            raise error_type(f"{invalid} for {operation}: items must be non-empty")
        canonical_items = []
        seen_item_ids = set()
        for position, item in enumerate(items):
            if not isinstance(item, dict) or set(item) != cls._APPROVAL_ITEM_FIELDS:
                raise error_type(
                    f"{invalid} for {operation}: item {position} fields must be exact"
                )
            item_id = item["item_id"]
            if not isinstance(item_id, str) or not item_id.strip():
                raise error_type(
                    f"{invalid} for {operation}: item {position} has no item_id"
                )
            if item_id in seen_item_ids:
                raise error_type(
                    f"{invalid} for {operation}: duplicate item_id {item_id!r}"
                )
            seen_item_ids.add(item_id)
            if not isinstance(item["data"], dict) or not isinstance(
                item["metadata"], dict
            ):
                raise error_type(
                    f"{invalid} for {operation}: item {item_id!r} data and metadata "
                    "must be objects"
                )
            confidence = item["confidence"]
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= confidence <= 1
            ):
                raise error_type(
                    f"{invalid} for {operation}: item {item_id!r} confidence is invalid"
                )
            if not isinstance(item["status"], str) or not item["status"]:
                raise error_type(
                    f"{invalid} for {operation}: item {item_id!r} status is invalid"
                )
            cls._require_canonical_timestamp(
                item["created_at"],
                field=f"item {item_id!r} created_at",
                invalid=invalid,
                operation=operation,
                error_type=error_type,
            )
            reviewed_at = item["reviewed_at"]
            if reviewed_at is not None:
                cls._require_canonical_timestamp(
                    reviewed_at,
                    field=f"item {item_id!r} reviewed_at",
                    invalid=invalid,
                    operation=operation,
                    error_type=error_type,
                )
            canonical_items.append(
                {key: value for key, value in item.items() if key != "created_at"}
            )
        return {
            "batch_id": payload["batch_id"],
            "context": payload["context"],
            "items": canonical_items,
        }

    async def select_canonical(
        self,
        *,
        tenant_id: str,
        batch_id: str,
        original_item_id: str,
        candidate: Mapping[str, Any],
    ) -> CanonicalReplacementRecord:
        """Return the first payload stored for an approval replacement key."""
        operation = self._operation(tenant_id, batch_id, original_item_id)
        if not isinstance(candidate, Mapping):
            raise ValueError(f"Replacement candidate must be an object for {operation}")
        try:
            payload = json.dumps(
                candidate,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Replacement candidate is not strict JSON for {operation}"
            ) from exc
        key = self._key(tenant_id, batch_id, original_item_id)
        redis = aioredis.from_url(
            self._redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            retry_on_timeout=False,
        )
        try:
            created = await redis.set(key, payload, nx=True)
            selected_payload = payload if created else await redis.get(key)
        except RedisError as exc:
            raise RuntimeError(
                f"Failed to select canonical replacement for {operation}"
            ) from exc
        finally:
            await redis.aclose()

        if selected_payload is None:
            raise RuntimeError(
                f"Canonical replacement disappeared after selection for {operation}"
            )

        return self._decode_record(selected_payload, operation=operation)

    async def select_review_decision(
        self,
        *,
        tenant_id: str,
        batch_id: str,
        original_item_id: str,
        candidate: Mapping[str, Any],
    ) -> CanonicalReplacementRecord:
        """Return the first timestamp for one otherwise exact review decision."""
        operation = self._operation(tenant_id, batch_id, original_item_id)
        if not isinstance(candidate, Mapping):
            raise ValueError(
                f"Review decision candidate must be an object for {operation}"
            )
        try:
            payload = json.dumps(
                candidate,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Review decision candidate is not strict JSON for {operation}"
            ) from exc
        candidate_record = self._decode_record(payload, operation=operation)
        candidate_intent = self._review_decision_intent(
            candidate_record.payload,
            operation=operation,
            original_item_id=original_item_id,
            stored=False,
        )

        redis = aioredis.from_url(
            self._redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            retry_on_timeout=False,
        )
        key = self._decision_key(tenant_id, batch_id, original_item_id)
        try:
            created = await redis.set(key, payload, nx=True)
            selected_payload = payload if created else await redis.get(key)
        except RedisError as exc:
            raise RuntimeError(
                f"Failed to select canonical review decision for {operation}"
            ) from exc
        finally:
            await redis.aclose()

        if selected_payload is None:
            raise RuntimeError(
                f"Canonical review decision disappeared after selection for {operation}"
            )
        selected = self._decode_record(selected_payload, operation=operation)
        selected_intent = self._review_decision_intent(
            selected.payload,
            operation=operation,
            original_item_id=original_item_id,
            stored=True,
        )
        if selected_intent != candidate_intent:
            raise RuntimeError(
                f"Review decision conflicts with canonical review decision for {operation}"
            )
        return selected

    async def select_approval_batch(
        self,
        *,
        tenant_id: str,
        batch_id: str,
        candidate: Mapping[str, Any],
    ) -> CanonicalReplacementRecord:
        """Select first timestamps for an otherwise exact approval batch."""
        operation = self._batch_operation(tenant_id, batch_id)
        if not isinstance(candidate, Mapping):
            raise ValueError(
                f"Approval batch candidate must be an object for {operation}"
            )
        try:
            payload = json.dumps(
                candidate,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Approval batch candidate is not strict JSON for {operation}"
            ) from exc
        candidate_record = self._decode_record(payload, operation=operation)
        candidate_intent = self._approval_batch_intent(
            candidate_record.payload,
            batch_id=batch_id,
            operation=operation,
            stored=False,
        )

        redis = aioredis.from_url(
            self._redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            retry_on_timeout=False,
        )
        key = self._approval_batch_key(tenant_id, batch_id)
        try:
            created = await redis.set(key, payload, nx=True)
            selected_payload = payload if created else await redis.get(key)
        except RedisError as exc:
            raise RuntimeError(
                f"Failed to select canonical approval batch for {operation}"
            ) from exc
        finally:
            await redis.aclose()

        if selected_payload is None:
            raise RuntimeError(
                f"Canonical approval batch disappeared after selection for {operation}"
            )
        selected = self._decode_record(selected_payload, operation=operation)
        selected_intent = self._approval_batch_intent(
            selected.payload,
            batch_id=batch_id,
            operation=operation,
            stored=True,
        )
        if selected_intent != candidate_intent:
            raise RuntimeError(
                f"Approval batch conflicts with canonical approval batch for {operation}"
            )
        return selected

    @asynccontextmanager
    async def replacement_event_lock(
        self,
        *,
        tenant_id: str,
        batch_id: str,
        original_item_id: str,
    ):
        """Serialize Phoenix replacement export for one immutable Redis record."""
        operation = self._operation(tenant_id, batch_id, original_item_id)
        owner_task = asyncio.current_task()
        if owner_task is None:
            raise RuntimeError(
                f"Replacement event lock has no owning task for {operation}"
            )
        key = self._event_lock_key(tenant_id, batch_id, original_item_id)
        owner = secrets.token_hex(16)
        redis = aioredis.from_url(
            self._redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            retry_on_timeout=False,
        )
        deadline = time.monotonic() + self._EVENT_LOCK_WAIT_SECONDS
        try:
            while not await redis.set(
                key,
                owner,
                nx=True,
                px=self._EVENT_LOCK_LEASE_MS,
            ):
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Timed out acquiring replacement event lock for {operation}"
                    )
                await asyncio.sleep(0.05)
        except RedisError as exc:
            await redis.aclose()
            raise RuntimeError(
                f"Failed to acquire replacement event lock for {operation}"
            ) from exc
        except BaseException:
            await redis.aclose()
            raise

        stop = asyncio.Event()
        renewal_failure: list[BaseException] = []

        def raise_renewal_failure() -> None:
            if renewal_failure:
                raise RuntimeError(
                    f"Failed to renew replacement event lock for {operation}"
                ) from renewal_failure[0]

        async def renew() -> None:
            interval = self._EVENT_LOCK_LEASE_MS / 3000
            while True:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=interval)
                    return
                except TimeoutError:
                    pass
                try:
                    renewed = await redis.eval(
                        "if redis.call('get', KEYS[1]) == ARGV[1] then "
                        "return redis.call('pexpire', KEYS[1], ARGV[2]) else "
                        "return 0 end",
                        1,
                        key,
                        owner,
                        self._EVENT_LOCK_LEASE_MS,
                    )
                except RedisError as exc:
                    renewal_failure.append(exc)
                    owner_task.cancel()
                    return
                if renewed != 1:
                    renewal_failure.append(RuntimeError("lock ownership was lost"))
                    owner_task.cancel()
                    return

        renewal_task = asyncio.create_task(renew())
        body_failed = False
        try:
            try:
                yield
            except asyncio.CancelledError:
                raise_renewal_failure()
                raise
            raise_renewal_failure()
        except BaseException:
            body_failed = True
            raise
        finally:
            stop.set()
            await renewal_task
            try:
                released = await redis.eval(
                    "if redis.call('get', KEYS[1]) == ARGV[1] then "
                    "return redis.call('del', KEYS[1]) else return 0 end",
                    1,
                    key,
                    owner,
                )
                if released != 1 and not body_failed:
                    raise RuntimeError(
                        f"Replacement event lock ownership was lost for {operation}"
                    )
            except RedisError as exc:
                if body_failed:
                    logger.error(
                        "Failed to release replacement event lock after error for %s",
                        operation,
                    )
                else:
                    raise RuntimeError(
                        f"Failed to release replacement event lock for {operation}"
                    ) from exc
            finally:
                await redis.aclose()
