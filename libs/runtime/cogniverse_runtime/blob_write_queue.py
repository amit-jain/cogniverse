"""Write-behind queue for admin config-blob persistence.

Admin PUTs (pin quotas, signature variants) persist small per-tenant blobs
through the artifact store. Applied inline, each PUT pays several store
round-trips and stalls behind telemetry load; through this queue the PUT is
accepted immediately and applied in the background.

Contracts:
  * accepted != applied — ``status()`` and ``pending_content`` report what is
    still in flight.
  * read-your-write — ``pending_content`` returns the accepted content until
    it is durably applied, so readers overlay it on the stale store value.
  * no silent data loss — a write the queue cannot persist after
    ``max_attempts`` is retained as a typed :class:`BlobWriteFailed` that
    ``raise_if_failed`` surfaces on every subsequent read until a newer write
    supersedes it.

Writes to one (tenant, kind, key) coalesce last-write-wins on the accepted
content, keeping the merge base of the earliest coalesced write: applying a
write re-reads the durable blob and replays the fields that write changed, so
a peer's field persisted in between is not overwritten by this one's stale
snapshot.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BlobKey = Tuple[str, str, str]
# (tenant_id, kind, key, accepted_content, merge_base_content)
Applier = Callable[[str, str, str, str, str], Awaitable[None]]


class BlobWriteFailed(Exception):
    """An accepted blob write could not be persisted."""

    def __init__(
        self,
        tenant_id: str,
        kind: str,
        key: str,
        cause: BaseException,
        content: str,
        base: str,
    ):
        super().__init__(
            f"blob write {kind}/{key} for tenant {tenant_id} failed: {cause}"
        )
        self.tenant_id = tenant_id
        self.kind = kind
        self.key = key
        self.content = content
        self.base = base
        self.__cause__ = cause


class BlobWriteQueue:
    def __init__(
        self,
        apply: Applier,
        *,
        max_attempts: int = 3,
        backoff_s: float = 0.5,
    ):
        self._apply = apply
        self._max_attempts = max_attempts
        self._backoff_s = backoff_s
        # Insertion-ordered; overwriting a key keeps its original position,
        # so first-enqueue order across keys is preserved while content
        # coalesces to the newest value.
        self._pending: Dict[BlobKey, Tuple[str, str]] = {}
        self._failed: Dict[BlobKey, BlobWriteFailed] = {}
        self._inflight: Optional[BlobKey] = None
        self._settled = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    def enqueue(
        self, tenant_id: str, kind: str, key: str, content: str, *, base: str
    ) -> None:
        """Accept a write. Returns before any store round-trip.

        ``base`` is the content ``content`` was merged onto; the difference
        between them is the set of fields this write changed, which is what is
        replayed onto the durable blob when the write is applied. A write that
        supersedes an accepted or failed one keeps that one's base, so the
        replayed set still covers every field the superseded write changed and
        never persisted.
        """
        blob_key = (tenant_id, kind, key)
        failed = self._failed.pop(blob_key, None)
        existing = self._pending.get(blob_key)
        if existing is not None:
            base = existing[0]
        elif failed is not None:
            base = failed.base
        self._pending[blob_key] = (base, content)
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(
                self._drain(), name="blob-write-queue-drain"
            )

    def pending_content(self, tenant_id: str, kind: str, key: str) -> Optional[str]:
        """The accepted-but-not-yet-applied content for a key, if any."""
        entry = self._pending.get((tenant_id, kind, key))
        return None if entry is None else entry[1]

    def raise_if_failed(self, tenant_id: str, kind: str, key: str) -> None:
        error = self._failed.get((tenant_id, kind, key))
        if error is not None:
            raise error

    def failed_error(
        self, tenant_id: str, kind: str, key: str
    ) -> Optional[BlobWriteFailed]:
        return self._failed.get((tenant_id, kind, key))

    def status(self) -> Dict[str, object]:
        failed: List[BlobKey] = list(self._failed)
        return {"pending": len(self._pending), "failed": failed}

    async def flush(self) -> None:
        """Wait until every accepted write is applied or failed."""
        while self._pending or self._inflight is not None:
            self._settled.clear()
            await self._settled.wait()

    async def _drain(self) -> None:
        while self._pending:
            blob_key = next(iter(self._pending))
            entry = self._pending[blob_key]
            base, content = entry
            self._inflight = blob_key
            try:
                await self._apply_with_retries(blob_key, content, base)
            except BlobWriteFailed as error:
                if self._pending.get(blob_key) == entry:
                    # Terminal for the content that failed; a newer enqueue
                    # supersedes both the entry and the error.
                    del self._pending[blob_key]
                    self._failed[blob_key] = error
                    logger.exception(
                        "Blob write %s/%s for tenant %s failed terminally; "
                        "the accepted value was NOT persisted",
                        blob_key[1],
                        blob_key[2],
                        blob_key[0],
                    )
            else:
                logger.info(
                    "Blob write %s/%s for tenant %s applied",
                    blob_key[1],
                    blob_key[2],
                    blob_key[0],
                )
                if self._pending.get(blob_key) == entry:
                    del self._pending[blob_key]
                # else: superseded mid-apply; reprocess with the newer content.
            finally:
                self._inflight = None
                self._settled.set()

    async def _apply_with_retries(
        self, blob_key: BlobKey, content: str, base: str
    ) -> None:
        tenant_id, kind, key = blob_key
        for attempt in range(1, self._max_attempts + 1):
            try:
                await self._apply(tenant_id, kind, key, content, base)
                return
            except Exception as exc:
                if attempt == self._max_attempts:
                    raise BlobWriteFailed(
                        tenant_id, kind, key, exc, content, base
                    ) from exc
                logger.warning(
                    "Blob write %s/%s for tenant %s failed on attempt %d/%d, "
                    "retrying: %s",
                    kind,
                    key,
                    tenant_id,
                    attempt,
                    self._max_attempts,
                    exc,
                )
                await asyncio.sleep(self._backoff_s * attempt)
