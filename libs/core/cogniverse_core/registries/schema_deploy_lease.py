"""Cross-process lease serialising Vespa application-package replacement."""

import logging
import os
import socket
import threading
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

_process_state = threading.Lock()
# holder -> (record version, monotonic time this process first saw it there).
# Kept across acquire() calls, so a record that stands still is taken over
# once its hold time has passed even though each wait is shorter than that.
_stalled_records: dict[str, tuple[int, float]] = {}
# Holders this process has released without the store confirming the record
# was cleared. Nothing in this process activates under them any more.
_released_holders: set[str] = set()


def _stalled_for(holder: str, version: int) -> float:
    now = time.monotonic()
    with _process_state:
        seen = _stalled_records.get(holder)
        if seen is None or seen[0] != version:
            _stalled_records[holder] = (version, now)
            return 0.0
        return now - seen[1]


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
    wall clock: a process takes over only after it has itself watched the
    record's version stand still for the holder's hold time, across as many
    waits as that takes, and a holder treats its own lease as lost once its
    own monotonic clock passes that hold time since its last successful
    claim. Clock skew between nodes therefore cannot break mutual exclusion.
    """

    def __init__(
        self,
        store: ConfigStore,
        *,
        lease_seconds: Optional[float] = None,
        wait_seconds: Optional[float] = None,
        tenant_id: str = SYSTEM_TENANT_ID,
        service: str = _SERVICE,
        config_key: str = _KEY,
        purpose: str = "Vespa deployment",
    ) -> None:
        self._store = store
        self._tenant_id = tenant_id
        self._service = service
        self._config_key = config_key
        self._purpose = purpose
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
            tenant_id=self._tenant_id,
            scope=ConfigScope.SCHEMA,
            service=self._service,
            config_key=self._config_key,
        )
        if entry is None:
            return None, 0
        return entry.config_value, entry.version

    def _claim(self, expected_version: int, holder: Optional[str]) -> bool:
        entry = self._store.compare_and_set_config(
            tenant_id=self._tenant_id,
            scope=ConfigScope.SCHEMA,
            service=self._service,
            config_key=self._config_key,
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
        watched: Optional[str] = None
        while True:
            record, version = self._read()
            current = None if record is None else record.get("holder")
            if watched is not None and watched != current:
                with _process_state:
                    _stalled_records.pop(watched, None)
            watched = current
            if current in (None, self.holder):
                claimable = True
            else:
                with _process_state:
                    released_here = current in _released_holders
                claimable = released_here or _stalled_for(
                    current, version
                ) >= self._hold_seconds(record)
            if claimable and self._claim(version, self.holder):
                with _process_state:
                    _stalled_records.pop(current, None)
                    _released_holders.discard(current)
                    _released_holders.discard(self.holder)
                self._held_since = time.monotonic()
                logger.info("%s lease acquired by %s", self._purpose, self.holder)
                return self
            if time.monotonic() >= deadline:
                if self._purpose == "Vespa deployment":
                    raise TimeoutError(
                        f"Vespa deployment lease still held by {current!r} after "
                        f"{self._wait_seconds}s; refusing to replace the application "
                        f"package concurrently with another deployer"
                    )
                raise TimeoutError(
                    f"{self._purpose} lease still held by {current!r} after "
                    f"{self._wait_seconds}s; refusing concurrent ownership"
                )
            time.sleep(_POLL_SECONDS)

    def renew(self) -> None:
        """Extend the lease, or raise if this holder no longer owns it."""
        if (
            self._held_since is None
            or time.monotonic() - self._held_since >= self._lease_seconds
        ):
            raise DeploymentLeaseLost(f"{self._purpose} lease expired or was replaced")
        record, version = self._read()
        current = None if record is None else record.get("holder")
        if current != self.holder or not self._claim(version, self.holder):
            raise DeploymentLeaseLost(f"{self._purpose} lease expired or was replaced")
        self._held_since = time.monotonic()

    def release(self) -> None:
        """Hand the lease back; a lease already taken over is left alone.

        A store failure here is not the deploy's failure: the package is
        already activated. A record the store did not confirm cleared is
        taken over at once by this process, and by peers once they have
        watched it stand still for the hold time.
        """
        cleared = False
        try:
            record, version = self._read()
            if record is not None and record.get("holder") == self.holder:
                cleared = self._claim(version, None)
        except Exception as exc:
            logger.warning(
                "%s lease held by %s could not be released "
                "(%s: %s); peers take it over after %.0fs",
                self._purpose,
                self.holder,
                type(exc).__name__,
                exc,
                self._lease_seconds,
            )
        finally:
            self._held_since = None
            if not cleared:
                with _process_state:
                    _released_holders.add(self.holder)
