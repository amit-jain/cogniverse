"""Cross-process lease serialising Vespa application-package replacement."""

import logging
import os
import socket
import time
import uuid
from typing import Any, Optional

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_sdk.interfaces.config_store import ConfigScope, ConfigStore

logger = logging.getLogger(__name__)

_SERVICE = "schema_deploy_lease"
_KEY = "application"
_POLL_SECONDS = 0.25

# Longer than one prepare-and-activate attempt (the deploy POST's read
# timeout), so a holder that renews before each attempt cannot expire while
# that attempt is in flight, and short enough that a holder killed mid-deploy
# blocks its peers for a bounded time.
DEFAULT_LEASE_SECONDS = 600.0

# How long a deployer waits for a live holder before failing. Request-facing
# deploys run this wait on a worker thread — a tenant's first wiki access
# deploys its schema — so it is bounded well under a request timeout rather
# than at the lease expiry; a waiter that gives up raises and the next attempt
# re-queues.
DEFAULT_WAIT_SECONDS = 120.0


class DeploymentLeaseLost(RuntimeError):
    """The holder no longer owns the lease and must not activate a package."""


class SchemaDeployLease:
    """One writer at a time for the whole Vespa application package.

    Every replacement posts the complete package, so two processes that each
    build from their own snapshot activate packages missing the other's
    schemas. The lease record lives in the config store under the system
    tenant, holds the current holder and the hold time it was taken with, and
    moves only through the store's compare-and-set, so processes on different
    pods contend for it correctly.

    Expiry is measured as elapsed time on the observer's own monotonic clock,
    never as a timestamp one node writes and another compares against its own
    wall clock: a waiter takes over only after it has itself watched the
    record's version stand still for the holder's hold time, and a holder
    treats its own lease as lost once its own monotonic clock passes that
    hold time since its last successful claim. Clock skew between nodes
    therefore cannot break mutual exclusion.
    """

    def __init__(
        self,
        store: ConfigStore,
        *,
        lease_seconds: Optional[float] = None,
        wait_seconds: Optional[float] = None,
    ) -> None:
        self._store = store
        self._lease_seconds = (
            DEFAULT_LEASE_SECONDS if lease_seconds is None else lease_seconds
        )
        self._wait_seconds = (
            DEFAULT_WAIT_SECONDS if wait_seconds is None else wait_seconds
        )
        self.holder = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
        self._held_since: Optional[float] = None

    def _read(self) -> tuple[Optional[dict[str, Any]], int]:
        entry = self._store.get_config(
            tenant_id=SYSTEM_TENANT_ID,
            scope=ConfigScope.SCHEMA,
            service=_SERVICE,
            config_key=_KEY,
        )
        if entry is None:
            return None, 0
        return entry.config_value, entry.version

    def _claim(self, expected_version: int, holder: Optional[str]) -> bool:
        entry = self._store.compare_and_set_config(
            tenant_id=SYSTEM_TENANT_ID,
            scope=ConfigScope.SCHEMA,
            service=_SERVICE,
            config_key=_KEY,
            config_value={"holder": holder, "lease_seconds": self._lease_seconds},
            expected_version=expected_version,
        )
        return entry is not None

    def _hold_seconds(self, record: Optional[dict[str, Any]]) -> float:
        if record is None:
            return self._lease_seconds
        return float(record.get("lease_seconds", self._lease_seconds))

    def acquire(self) -> "SchemaDeployLease":
        """Take the lease, waiting out a live holder up to ``wait_seconds``."""
        deadline = time.monotonic() + self._wait_seconds
        watched_version: Optional[int] = None
        watched_since = time.monotonic()
        while True:
            record, version = self._read()
            current = None if record is None else record.get("holder")
            if version != watched_version:
                # The holder renewed, released, or changed: its hold time
                # starts again from this observation.
                watched_version = version
                watched_since = time.monotonic()
            stalled_for = time.monotonic() - watched_since
            if (
                current in (None, self.holder)
                or stalled_for >= self._hold_seconds(record)
            ) and self._claim(version, self.holder):
                self._held_since = time.monotonic()
                logger.info("Vespa deployment lease acquired by %s", self.holder)
                return self
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Vespa deployment lease still held by {current!r} after "
                    f"{self._wait_seconds}s; refusing to replace the application "
                    f"package concurrently with another deployer"
                )
            time.sleep(_POLL_SECONDS)

    def renew(self) -> None:
        """Extend the lease, or raise if this holder no longer owns it."""
        if (
            self._held_since is None
            or time.monotonic() - self._held_since >= self._lease_seconds
        ):
            raise DeploymentLeaseLost("Vespa deployment lease expired or was replaced")
        record, version = self._read()
        current = None if record is None else record.get("holder")
        if current != self.holder or not self._claim(version, self.holder):
            raise DeploymentLeaseLost("Vespa deployment lease expired or was replaced")
        self._held_since = time.monotonic()

    def release(self) -> None:
        """Hand the lease back; a lease already taken over is left alone.

        A store failure here is not the deploy's failure: the package is
        already activated. The record is left to be taken over once peers
        have watched it stand still for the hold time.
        """
        try:
            record, version = self._read()
            if record is not None and record.get("holder") == self.holder:
                self._claim(version, None)
        except Exception as exc:
            logger.warning(
                "Vespa deployment lease held by %s could not be released "
                "(%s: %s); peers take it over after %.0fs",
                self.holder,
                type(exc).__name__,
                exc,
                self._lease_seconds,
            )
        finally:
            self._held_since = None
