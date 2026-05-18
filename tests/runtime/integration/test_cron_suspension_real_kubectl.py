"""Integration test for the e2e cronworkflow suspend/restore fixture.

Uses real kubectl against the live k3d cluster. Skips if either
``kubectl`` is unavailable or the Argo CronWorkflow CRD isn't
installed — same skip pattern as other infrastructure-dependent
integration tests in this repo (``_openshell_cli_available`` etc).

The test creates a throwaway CronWorkflow in the cogniverse namespace,
runs the suspend helper against the live cluster, verifies the live
object's ``spec.suspend`` flipped to ``true``, then runs the restore
helper and verifies the flag flipped back to ``false``. Real
kubectl, real CRD, real spec — no subprocess mocks.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid

import pytest

from tests.e2e.conftest import (
    _restore_cronworkflows,
    _suspend_cronworkflows_for_session,
)

NAMESPACE = "cogniverse"
pytestmark = pytest.mark.integration


def _kubectl_available() -> bool:
    if shutil.which("kubectl") is None:
        return False
    result = subprocess.run(
        ["kubectl", "get", "cronworkflows", "-n", NAMESPACE],
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.returncode == 0


skip_if_no_argo = pytest.mark.skipif(
    not _kubectl_available(),
    reason="kubectl + Argo CronWorkflow CRD not available",
)


def _apply_cronworkflow(name: str, suspend: bool) -> None:
    """Create a no-op CronWorkflow in the live cluster."""
    manifest = {
        "apiVersion": "argoproj.io/v1alpha1",
        "kind": "CronWorkflow",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "labels": {"cogniverse.test/source": "integration-cron-fixture"},
        },
        "spec": {
            "schedule": "0 0 31 2 *",  # never (Feb 31) — keeps it idle
            "suspend": suspend,
            "concurrencyPolicy": "Forbid",
            "workflowSpec": {
                "entrypoint": "noop",
                "templates": [
                    {
                        "name": "noop",
                        "container": {
                            "image": "busybox:1.36",
                            "command": ["sh", "-c", "exit 0"],
                        },
                    }
                ],
            },
        },
    }
    result = subprocess.run(
        ["kubectl", "apply", "-n", NAMESPACE, "-f", "-"],
        input=json.dumps(manifest),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"kubectl apply failed: {result.stderr}"


def _read_suspend(name: str) -> bool:
    out = subprocess.run(
        ["kubectl", "get", "cronworkflow", name, "-n", NAMESPACE, "-o", "json"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, f"kubectl get failed: {out.stderr}"
    return bool((json.loads(out.stdout).get("spec") or {}).get("suspend"))


def _delete_cronworkflow(name: str) -> None:
    subprocess.run(
        ["kubectl", "delete", "cronworkflow", name, "-n", NAMESPACE, "--wait=false"],
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.fixture
def fresh_cronworkflow():
    """Create + tear down a uniquely-named test CronWorkflow."""
    name = f"cv-integration-cron-{uuid.uuid4().hex[:10]}"
    created: list[str] = []
    yield_value = {"name": name, "created": created}
    yield yield_value
    for n in {name, *created}:
        _delete_cronworkflow(n)


@skip_if_no_argo
class TestSuspendRestoreAgainstLiveCluster:
    def test_unsuspended_cron_is_flipped_to_suspended_then_restored(
        self, fresh_cronworkflow
    ):
        name = fresh_cronworkflow["name"]
        _apply_cronworkflow(name, suspend=False)
        fresh_cronworkflow["created"].append(name)

        assert _read_suspend(name) is False, (
            "precondition: cronworkflow must start un-suspended on the live cluster"
        )

        # Run the real suspend helper — patches every un-suspended cron
        # across the namespace, not just ours, so filter the result to
        # our test name when asserting we were toggled.
        toggled = _suspend_cronworkflows_for_session()
        try:
            assert name in toggled, (
                f"suspend helper must report our test cron as toggled; "
                f"got toggled={sorted(toggled)}"
            )
            assert _read_suspend(name) is True, (
                "suspend helper did not flip spec.suspend on the live "
                "object — the cluster still sees suspend=false"
            )
        finally:
            _restore_cronworkflows(toggled)

        assert _read_suspend(name) is False, (
            "restore helper did not flip spec.suspend back — leaving the "
            "live cluster in a suspended state across runs is the exact "
            "leak this fixture was meant to fix"
        )

    def test_already_suspended_cron_is_not_re_enabled_on_restore(
        self, fresh_cronworkflow
    ):
        """A cron that was suspended BEFORE the fixture must stay suspended.

        Pin the contract that the fixture's restore only touches what
        suspend explicitly toggled — a pre-existing user-suspended cron
        must survive the round-trip with suspend still true.
        """
        name = fresh_cronworkflow["name"]
        _apply_cronworkflow(name, suspend=True)
        fresh_cronworkflow["created"].append(name)

        assert _read_suspend(name) is True, "precondition: starts suspended"

        toggled = _suspend_cronworkflows_for_session()
        assert name not in toggled, (
            f"suspend helper must not re-patch an already-suspended cron; "
            f"got toggled={sorted(toggled)}"
        )

        _restore_cronworkflows(toggled)

        assert _read_suspend(name) is True, (
            "restore helper re-enabled a cron it had not suspended — the "
            "fixture leaked state into the user's pre-existing config"
        )
