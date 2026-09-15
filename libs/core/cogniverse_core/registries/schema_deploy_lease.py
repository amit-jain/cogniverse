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
DEFAULT_WAIT_SECONDS = 900.0


class DeploymentLeaseLost(RuntimeError):
    """The holder no longer owns the lease and must not activate a package."""


class SchemaDeployLease:
    """One writer at a time for the whole Vespa application package.

    Every replacement posts the complete package, so two processes that each
    build from their own snapshot activate packages missing the other's
    schemas. The lease record lives in the config store under the system
    tenant, holds the current holder and its expiry, and moves only through
    the store's compare-and-set, so processes on different pods contend for
    it correctly. Expiry is wall-clock, shared through the store.
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
        expires_at = 0.0 if holder is None else time.time() + self._lease_seconds
        entry = self._store.compare_and_set_config(
            tenant_id=SYSTEM_TENANT_ID,
            scope=ConfigScope.SCHEMA,
            service=_SERVICE,
            config_key=_KEY,
            config_value={"holder": holder, "expires_at": expires_at},
            expected_version=expected_version,
        )
        return entry is not None

    def acquire(self) -> "SchemaDeployLease":
        """Take the lease, waiting out a live holder up to ``wait_seconds``."""
        deadline = time.monotonic() + self._wait_seconds
        while True:
            record, version = self._read()
            current = None if record is None else record.get("holder")
            expires_at = 0.0 if record is None else float(record.get("expires_at", 0.0))
            if current in (None, self.holder) or time.time() >= expires_at:
                if self._claim(version, self.holder):
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
        record, version = self._read()
        current = None if record is None else record.get("holder")
        expires_at = 0.0 if record is None else float(record.get("expires_at", 0.0))
        if current != self.holder or time.time() >= expires_at:
            raise DeploymentLeaseLost("Vespa deployment lease expired or was replaced")
        if not self._claim(version, self.holder):
            raise DeploymentLeaseLost("Vespa deployment lease expired or was replaced")

    def release(self) -> None:
        """Hand the lease back; a lease already taken over is left alone."""
        record, version = self._read()
        if record is None or record.get("holder") != self.holder:
            return
        self._claim(version, None)
