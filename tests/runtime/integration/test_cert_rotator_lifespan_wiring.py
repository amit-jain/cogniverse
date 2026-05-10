"""CertRotator wiring — CertRotator is started by main.py lifespan and reacts to changes.

Without this test, ``CertRotator`` was a class nobody instantiated. The
test boots the real ``main.py`` lifespan (skipping the sandbox connect
via ``policy=disabled`` for half the cases, exercising it end-to-end
for the other half), then:

  * asserts the rotator was constructed and attached to the SandboxManager
  * touches a real cert file in the gateway dir
  * asserts the SandboxManager's ``reconnect`` was called on the next probe

The point is to verify the *wiring*, not the rotator's internals
(those have their own unit tests).
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest
from fastapi import FastAPI

from cogniverse_runtime.openshell_cert_rotator import CertRotator
from cogniverse_runtime.sandbox_manager import SandboxManager, SandboxPolicy

pytestmark = pytest.mark.integration


@pytest.fixture
def gateway_dir(tmp_path: Path, monkeypatch) -> Path:
    """A real temp dir laid out like an OpenShell gateway config."""
    base = tmp_path / "openshell"
    gw_name = "lifespan-test-gw"
    gw_dir = base / "gateways" / gw_name
    (gw_dir / "mtls").mkdir(parents=True)
    (gw_dir / "metadata.json").write_text(
        '{"name":"lifespan-test-gw"}', encoding="utf-8"
    )
    (gw_dir / "mtls" / "ca.crt").write_text("CA_BASELINE", encoding="utf-8")
    (gw_dir / "mtls" / "tls.crt").write_text("TLS_CRT_BASELINE", encoding="utf-8")
    (gw_dir / "mtls" / "tls.key").write_text("TLS_KEY_BASELINE", encoding="utf-8")
    (base / "active_gateway").write_text(gw_name, encoding="utf-8")
    monkeypatch.setenv("OPENSHELL_CONFIG_DIR", str(base))
    return gw_dir


class TestLifespanWiresRotator:
    """Run the real lifespan; assert app.state.cert_rotator + attach hookup."""

    @pytest.mark.asyncio
    async def test_lifespan_starts_and_attaches_rotator(
        self, gateway_dir: Path, monkeypatch
    ):
        """End-to-end: lifespan startup wires + lifespan shutdown stops the rotator."""
        # Use OPTIONAL so the lifespan doesn't refuse-to-start when the gateway
        # is unreachable; we don't need a real gateway for the wiring assertion.
        monkeypatch.setenv("COGNIVERSE_SANDBOX_POLICY", "optional")
        # Fast probe so the test can observe a tick within seconds.
        monkeypatch.setenv("COGNIVERSE_SANDBOX_CERT_ROTATION_INTERVAL", "1")
        # Disable the lifecycle scheduler — it tries to find Mem0 instances and
        # prolongs the boot; this test cares only about cert wiring.
        monkeypatch.setenv("COGNIVERSE_MEMORY_LIFECYCLE_DISABLED", "1")
        # Probe interval kept small for fast test shutdown.
        monkeypatch.setenv("COGNIVERSE_SANDBOX_PROBE_INTERVAL", "1")

        from cogniverse_runtime.main import lifespan

        app = FastAPI()
        async with lifespan(app):
            rotator = getattr(app.state, "cert_rotator", None)
            assert rotator is not None, (
                "lifespan did not store cert_rotator on app.state — "
                "the wiring branch is not being entered"
            )
            assert isinstance(rotator, CertRotator)
            # The rotator must be attached to the SandboxManager so an
            # auth-failure on exec triggers a fresh handshake.
            assert rotator._mgr is not None
            # Give the loop one tick so probe_once captures the baseline.
            await asyncio.sleep(2.0)
            # Baseline snapshot should have been captured.
            assert rotator.last_snapshot is not None

        # Post-shutdown: the task must have been awaited cleanly.
        assert rotator._task is None

    @pytest.mark.asyncio
    async def test_disabled_rotator_via_env(self, gateway_dir: Path, monkeypatch):
        """COGNIVERSE_SANDBOX_CERT_ROTATION_DISABLED=1 → no rotator started."""
        monkeypatch.setenv("COGNIVERSE_SANDBOX_POLICY", "optional")
        monkeypatch.setenv("COGNIVERSE_SANDBOX_CERT_ROTATION_DISABLED", "1")
        monkeypatch.setenv("COGNIVERSE_MEMORY_LIFECYCLE_DISABLED", "1")
        monkeypatch.setenv("COGNIVERSE_SANDBOX_PROBE_INTERVAL", "1")
        # dspy.configure can only be called once per asyncio task; the
        # first lifespan-using test in this module already configured it.
        # Stub it so the lifespan can re-run for subsequent tests.
        import dspy

        monkeypatch.setattr(dspy, "configure", lambda *a, **kw: None)

        from cogniverse_runtime.main import lifespan

        app = FastAPI()
        async with lifespan(app):
            assert getattr(app.state, "cert_rotator", None) is None

    @pytest.mark.asyncio
    async def test_disabled_when_sandbox_disabled(self, gateway_dir: Path, monkeypatch):
        """sandbox.policy=disabled → cert rotator not started either."""
        monkeypatch.setenv("COGNIVERSE_SANDBOX_POLICY", "disabled")
        monkeypatch.setenv("COGNIVERSE_MEMORY_LIFECYCLE_DISABLED", "1")
        # Even if the rotator-disable env is unset, the sandbox-disabled
        # branch should still skip the rotator.
        monkeypatch.delenv("COGNIVERSE_SANDBOX_CERT_ROTATION_DISABLED", raising=False)
        import dspy

        monkeypatch.setattr(dspy, "configure", lambda *a, **kw: None)

        from cogniverse_runtime.main import lifespan

        app = FastAPI()
        async with lifespan(app):
            assert getattr(app.state, "cert_rotator", None) is None


class TestRotatorReactsThroughSandboxManager:
    """The other half of the wire: real cert change → SandboxManager.reconnect()."""

    def test_real_cert_change_reaches_attached_sandbox_manager(self, gateway_dir: Path):
        """Construct SandboxManager + CertRotator the same way main.py does;
        modify a cert file; assert the manager's reconnect path was hit.

        We use ``policy=DISABLED`` so SandboxManager.__init__ doesn't try
        to talk to a real OpenShell gateway — the boundary we care about
        here is the rotator → manager wire, not the manager → gateway wire.
        """
        # Spy on reconnect by replacing it with a tracker; SandboxManager
        # has the real attach_cert_rotator/method, so this exercises the
        # actual hook contract.
        mgr = SandboxManager(policy=SandboxPolicy.DISABLED)
        reconnect_calls: list[float] = []

        def _tracking_reconnect():
            reconnect_calls.append(time.time())
            # Don't actually reconnect (gateway is disabled), but emulate
            # "True" so the rotator's success-log path runs.
            return True

        mgr.reconnect = _tracking_reconnect  # type: ignore[method-assign]

        rotator = CertRotator(sandbox_manager=mgr, interval_seconds=60)
        mgr.attach_cert_rotator(rotator)

        # Baseline.
        rotator.probe_once()
        assert reconnect_calls == [], "baseline probe must not reconnect"

        # Real fs change: bump the mtime to a deterministic later value.
        os.utime(
            gateway_dir / "mtls" / "tls.crt",
            (3_000_000_000, 3_000_000_000),
        )

        changed, paths = rotator.probe_once()
        assert changed is True
        assert any(p.endswith("tls.crt") for p in paths)
        assert len(reconnect_calls) == 1, (
            "rotator must call SandboxManager.reconnect() on detected rotation"
        )

        # And the wire goes the other way too: an auth-failure on exec
        # eagerly triggers reconnect via the attached rotator.
        # Restore reconnect and re-spy for clarity.
        reconnect_calls.clear()
        mgr.reconnect = _tracking_reconnect  # type: ignore[method-assign]

        # _maybe_trigger_cert_rotator only fires for auth-shaped errors.
        mgr._maybe_trigger_cert_rotator(
            PermissionError("x509: certificate has expired")
        )
        # The rotator's trigger is rate-limited; one call should have fired.
        assert len(reconnect_calls) == 1, (
            "auth-failure path must drive a reconnect through the rotator"
        )
