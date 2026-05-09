"""OpenShell mTLS cert rotation.

The OpenShell SDK loads its mTLS material from the active gateway's
metadata directory at boot (``~/.config/openshell/gateways/<name>``).
That directory contains:

  * ``metadata.json``  — gateway endpoint + name
  * ``mtls/ca.crt``    — gateway CA bundle
  * ``mtls/tls.crt``   — client cert
  * ``mtls/tls.key``   — client key

When the operator rotates these (e.g. via ``openshell auth refresh`` or
an external cert-manager controller), the long-lived ``SandboxClient``
inside :class:`SandboxManager` keeps using the previous TLS material
because it cached the file paths — and the next exec eventually fails
with an auth error from the gateway.

This module ships a :class:`CertRotator` that:

  1. Watches the cert directory's file mtimes on a schedule (default
     ``300s`` — slow enough to be free, fast enough to catch a rotation
     within typical cert grace windows).
  2. On detected change, calls ``SandboxManager.reconnect()`` so the
     client re-reads the metadata + cert files.
  3. Exposes ``trigger_on_auth_failure()`` so the existing exec error
     paths in :class:`SandboxManager` can hook a "reconnect now"
     attempt the moment the gateway returns a TLS or auth error,
     instead of waiting for the next polling tick.

Lifecycle mirrors :class:`GatewayHealthProbe`: ``start()`` schedules
the loop, ``stop()`` awaits clean shutdown.

The rotator does not itself rotate certs — that is the operator's
responsibility (cert-manager, manual ``openshell auth``, etc.). It only
ensures cogniverse picks up the new material without a process restart.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from opentelemetry import trace

if TYPE_CHECKING:
    from cogniverse_runtime.sandbox_manager import SandboxManager

logger = logging.getLogger(__name__)


_DEFAULT_INTERVAL_SECONDS = 300.0
_AUTH_RETRY_BACKOFF_SECONDS = 5.0


def _gateway_dir(cluster: Optional[str] = None) -> Optional[Path]:
    """Return the active gateway's config directory, or None when unknown."""
    base = Path(
        os.environ.get(
            "OPENSHELL_CONFIG_DIR",
            str(Path.home() / ".config" / "openshell"),
        )
    )
    name = cluster
    if name is None:
        active = base / "active_gateway"
        if not active.exists():
            return None
        try:
            name = active.read_text(encoding="utf-8").strip()
        except OSError:
            return None
    if not name:
        return None
    candidate = base / "gateways" / name
    return candidate if candidate.exists() else None


def _watched_files(gw_dir: Path) -> list[Path]:
    """Files whose mtime change implies a rotation has occurred."""
    return [
        gw_dir / "metadata.json",
        gw_dir / "mtls" / "ca.crt",
        gw_dir / "mtls" / "tls.crt",
        gw_dir / "mtls" / "tls.key",
    ]


def snapshot_mtimes(gw_dir: Path) -> Dict[str, float]:
    """Return a {filename: mtime_ns_float} snapshot, missing files → 0.0."""
    out: Dict[str, float] = {}
    for f in _watched_files(gw_dir):
        try:
            out[str(f)] = f.stat().st_mtime
        except OSError:
            out[str(f)] = 0.0
    return out


def diff_mtimes(before: Dict[str, float], after: Dict[str, float]) -> list[str]:
    """Return the list of files whose mtime changed (or appeared/disappeared)."""
    changed: list[str] = []
    for path in set(before) | set(after):
        if before.get(path, 0.0) != after.get(path, 0.0):
            changed.append(path)
    return sorted(changed)


class CertRotator:
    """Background watcher that re-reads OpenShell mTLS material on rotation.

    Args:
        sandbox_manager: SandboxManager whose client gets reconnected on
            cert change.
        interval_seconds: Polling cadence. Default 300 s — picked so the
            cost is negligible while still catching rotations inside
            typical cert grace windows. Tunable for tests.
        cluster: When set, watch a non-active cluster's directory. Useful
            for staging environments that pin a non-default gateway.
        tracer: Optional opentelemetry tracer; when None, the global
            tracer is used. Tests can inject a recording tracer.
    """

    def __init__(
        self,
        sandbox_manager: "SandboxManager",
        interval_seconds: float = _DEFAULT_INTERVAL_SECONDS,
        cluster: Optional[str] = None,
        tracer: Optional[trace.Tracer] = None,
    ) -> None:
        if interval_seconds < 1.0:
            raise ValueError("interval_seconds must be >= 1")
        self._mgr = sandbox_manager
        self._interval = interval_seconds
        self._cluster = cluster
        self._tracer = tracer or trace.get_tracer(__name__)
        self._task: Optional[asyncio.Task] = None
        self._stop_evt: Optional[asyncio.Event] = None
        self._last_snapshot: Optional[Dict[str, float]] = None
        self._last_rotation_at: Optional[float] = None
        self._last_auth_trigger_at: float = 0.0

    @property
    def last_rotation_at(self) -> Optional[float]:
        """Wall-clock seconds when the last detected rotation occurred."""
        return self._last_rotation_at

    @property
    def last_snapshot(self) -> Optional[Dict[str, float]]:
        """Latest mtime snapshot (None before first probe)."""
        return self._last_snapshot

    def _resolve_dir(self) -> Optional[Path]:
        """Return the gateway dir or None (logged once)."""
        gw_dir = _gateway_dir(self._cluster)
        if gw_dir is None:
            logger.debug(
                "CertRotator: no OpenShell gateway config dir for cluster=%r; "
                "skipping rotation watch",
                self._cluster,
            )
        return gw_dir

    def probe_once(self) -> Tuple[bool, list[str]]:
        """Run one mtime check; reconnect on change. Returns (changed, paths).

        Always emits a span with the changed-paths list (empty list when
        no rotation was detected) so operators can correlate cert-change
        events to subsequent probe latency / availability flips.
        """
        gw_dir = self._resolve_dir()
        if gw_dir is None:
            self._record_span(False, [], reason="no_gateway_dir")
            return False, []

        snapshot = snapshot_mtimes(gw_dir)
        if self._last_snapshot is None:
            # First call: capture baseline only — first probe never reconnects.
            self._last_snapshot = snapshot
            self._record_span(False, [], reason="baseline_capture")
            return False, []

        changed = diff_mtimes(self._last_snapshot, snapshot)
        if not changed:
            self._record_span(False, [], reason="unchanged")
            return False, []

        self._last_snapshot = snapshot
        self._last_rotation_at = time.time()
        self._reconnect_for_rotation(changed)
        self._record_span(True, changed, reason="rotation_detected")
        return True, changed

    def trigger_on_auth_failure(self, error_repr: str) -> bool:
        """Force a reconnect attempt now in response to an auth/TLS error.

        Rate-limited to one trigger per `_AUTH_RETRY_BACKOFF_SECONDS` so a
        burst of failing requests doesn't thrash the gateway with
        repeated handshake attempts. Returns True if reconnect was
        attempted, False if rate-limited.
        """
        now = time.time()
        if now - self._last_auth_trigger_at < _AUTH_RETRY_BACKOFF_SECONDS:
            logger.debug(
                "CertRotator.trigger_on_auth_failure: rate-limited "
                "(last trigger %.1fs ago)",
                now - self._last_auth_trigger_at,
            )
            self._record_span(False, [], reason="auth_trigger_rate_limited")
            return False
        self._last_auth_trigger_at = now
        # Refresh the snapshot so the next polling tick doesn't double-fire.
        gw_dir = self._resolve_dir()
        if gw_dir is not None:
            self._last_snapshot = snapshot_mtimes(gw_dir)
        self._reconnect_for_rotation([f"auth_failure:{error_repr[:60]}"])
        self._record_span(True, [], reason="auth_failure_reconnect")
        return True

    def _reconnect_for_rotation(self, changed: list[str]) -> None:
        """Reconnect the SandboxManager client; never raise."""
        try:
            ok = self._mgr.reconnect()
            logger.info(
                "OpenShell cert rotation detected (%d files changed); "
                "reconnect %s. Files: %s",
                len(changed),
                "succeeded" if ok else "failed",
                changed,
            )
        except Exception:  # never let a reconnect blow up the rotator
            logger.exception("CertRotator: reconnect raised; staying with old client")

    def _record_span(
        self, changed: bool, changed_paths: list[str], *, reason: str
    ) -> None:
        with self._tracer.start_as_current_span("openshell.cert_rotation") as span:
            span.set_attribute("openshell.cert_rotation_detected", 1 if changed else 0)
            span.set_attribute("openshell.cert_rotation_reason", reason)
            if changed_paths:
                span.set_attribute(
                    "openshell.cert_rotation_changed_paths",
                    ",".join(changed_paths),
                )

    def start(self) -> None:
        """Start the background polling loop on the running event loop."""
        if self._task is not None and not self._task.done():
            logger.debug("CertRotator already running; start() is a no-op")
            return
        loop = asyncio.get_running_loop()
        self._stop_evt = asyncio.Event()
        self._task = loop.create_task(self._run_loop(), name="openshell_cert_rotator")
        logger.info(
            "OpenShell cert rotator started (interval=%.1fs cluster=%s)",
            self._interval,
            self._cluster or "active",
        )

    async def stop(self) -> None:
        """Stop the loop and await clean shutdown."""
        if self._stop_evt is not None:
            self._stop_evt.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=self._interval + 1)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None
        logger.info("OpenShell cert rotator stopped")

    async def _run_loop(self) -> None:
        assert self._stop_evt is not None
        while not self._stop_evt.is_set():
            try:
                # ``probe_once`` is sync (cheap stat calls); run on a thread
                # to keep the event loop free.
                await asyncio.to_thread(self.probe_once)
            except Exception:
                logger.exception("Unhandled error during OpenShell cert rotation probe")
            try:
                await asyncio.wait_for(self._stop_evt.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass
