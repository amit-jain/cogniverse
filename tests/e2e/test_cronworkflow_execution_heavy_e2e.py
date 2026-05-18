"""E2E coverage for CronWorkflow execution paths — heavy tier.

Heavy = the workflow takes 5-10+ minutes because it drives DSPy
training, distillation, or large Phoenix dataset generation against
the live LM endpoint. Opt-in only via ``-m e2e_heavy`` so the standard
sweep stays bounded.

Same contract as the light tier (``test_cronworkflow_execution_e2e.py``):
submit a one-off Workflow derived from the CronWorkflow's
``workflowSpec``, wait for terminal phase, assert ``Succeeded`` AND
that the real functional side effect landed on the live backend. The
"Succeeded" assertion alone is too weak for these workflows — each one
exists to produce a specific artifact / dataset / strategy update, and
the test must prove that landed.
"""

from __future__ import annotations

import shutil
import subprocess
import time

import httpx
import pytest

NAMESPACE = "cogniverse"
RUNTIME = (
    "http://localhost:28000"  # runtime.service.nodePort — matches tests/e2e/conftest.py
)
HEAVY_TIMEOUT_S = 1500.0  # 25 min — covers DSPy training + Phoenix work
POLL_INTERVAL_S = 10.0


# Re-use the light-tier helpers verbatim by importing them. Keeps the
# kubectl plumbing in one place; the heavy tier only differs in
# duration and per-cron post-state assertion.
from tests.e2e.test_cronworkflow_execution_e2e import (  # noqa: E402
    _cronworkflow_exists,
    _delete_workflow,
    _submit_workflow_from_cron,
    _wait_for_workflow_terminal,
    _workflow_pod_logs,
)


def _kubectl_available() -> bool:
    return shutil.which("kubectl") is not None


def _submit_and_wait_succeeded_heavy(cron_name: str) -> str:
    wf = _submit_workflow_from_cron(cron_name)
    try:
        status = _wait_for_workflow_terminal(wf, timeout_s=HEAVY_TIMEOUT_S)
        phase = status.get("phase") or "Unknown"
        if phase != "Succeeded":
            logs = _workflow_pod_logs(wf)
            pytest.fail(
                f"Workflow {wf} (from {cron_name}) phase={phase!r}, expected "
                f"'Succeeded'.\nstatus.message={status.get('message')!r}\n"
                f"--- pod logs (tail 500) ---\n{logs[-4000:]}\n--- end logs ---"
            )
        return wf
    finally:
        _delete_workflow(wf)


# ---------------------------------------------------------------------------
# Phoenix dataset helpers (synthetic-generation)
# ---------------------------------------------------------------------------


def _phoenix_url() -> str:
    # Phoenix container port 6006 → k3d-serverlb NodePort 26006.
    # Without 26006 the GET silently 0-results and the test
    # mis-reports "no new datasets appeared" when the workflow
    # actually created them.
    return "http://localhost:26006"


def _phoenix_dataset_names() -> set[str]:
    """Full set of dataset names via Phoenix HTTP API, paginating cursor.

    The endpoint returns ``data`` (10 per page) + ``next_cursor`` for
    the next page. Without paging, any dataset older than the most
    recent 10 is invisible — a newly-created dataset can be present
    yet the diff against ``names_before`` looks empty if both pages
    overlap into the same 10 head rows. The synthetic-generation
    workflow adds ~3 new datasets per run, which would never fit
    inside the head-10 window once the cluster has any real history.
    """
    names: set[str] = set()
    cursor: str | None = None
    try:
        with httpx.Client(timeout=30.0) as client:
            while True:
                params = {"cursor": cursor} if cursor else None
                r = client.get(f"{_phoenix_url()}/v1/datasets", params=params)
                if r.status_code != 200:
                    return names
                payload = r.json()
                names.update(d.get("name", "") for d in payload.get("data") or [])
                cursor = payload.get("next_cursor")
                if not cursor:
                    break
    except (httpx.HTTPError, OSError):
        pass
    return names


# ---------------------------------------------------------------------------
# Runtime-backed memory helpers (scheduled-distillation)
# ---------------------------------------------------------------------------


def _count_learned_strategies(tenant_full_id: str) -> int:
    """Count learned_strategy memories for a tenant via the admin route."""
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.get(
                f"{RUNTIME}/admin/tenants/{tenant_full_id}/memories",
                params={"kind": "learned_strategy", "limit": 1000},
            )
            if r.status_code != 200:
                return -1
            return len(r.json().get("memories", []))
    except (httpx.HTTPError, OSError):
        return -1


# ---------------------------------------------------------------------------
# Artifact-version helpers (agent-optimization)
# ---------------------------------------------------------------------------


def _artifact_version(tenant_full_id: str, agent_type: str) -> int:
    """Read the active artifact version for an agent. -1 on unavailable."""
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.get(
                f"{RUNTIME}/admin/tenants/{tenant_full_id}/artifacts/{agent_type}/active"
            )
            if r.status_code == 404:
                return 0
            if r.status_code != 200:
                return -1
            payload = r.json()
            for key in ("prompts_version", "version", "active_version"):
                if isinstance(payload.get(key), int):
                    return payload[key]
            return 0
    except (httpx.HTTPError, OSError):
        return -1


def _runtime_deployment_generation() -> int:
    """observedGeneration of the runtime deployment — bumps on rollout restart."""
    out = subprocess.run(
        [
            "kubectl",
            "get",
            "deployment",
            "-n",
            NAMESPACE,
            "-l",
            "app.kubernetes.io/component=runtime",
            "-o",
            "jsonpath={.items[*].status.observedGeneration}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return int(out.stdout.strip() or "0")


# ---------------------------------------------------------------------------
# Heavy-tier tests
# ---------------------------------------------------------------------------


@pytest.mark.e2e_heavy
@pytest.mark.skipif(
    not _kubectl_available(), reason="kubectl not available in test environment"
)
class TestAgentOptimizationWorkflow:
    """Weekly agent-optimization trains every DSPy module in parallel and
    then bounces the runtime so the new artifacts load. Functional intent:
    artifact version bumped + rollout observedGeneration advanced."""

    def test_workflow_advances_some_artifact_version_and_restarts_runtime(self):
        if not _cronworkflow_exists("cogniverse-agent-optimization"):
            pytest.skip("cogniverse-agent-optimization CronWorkflow not deployed")

        # Capture pre-state per agent (default tenant — matches the cron's
        # workflow.parameters.tenant-id="default" arg).
        agents = ("query_enhancement_agent", "profile_selection_agent", "summary_agent")
        versions_before = {a: _artifact_version("default", a) for a in agents}
        gen_before = _runtime_deployment_generation()

        _submit_and_wait_succeeded_heavy("cogniverse-agent-optimization")

        # Functional outcome 1: runtime deployment was rolled.
        gen_after = _runtime_deployment_generation()
        assert gen_after > gen_before, (
            f"agent-optimization workflow Succeeded but the runtime "
            f"deployment observedGeneration did not advance "
            f"({gen_before} → {gen_after}); the restart-runtime step "
            f"must have fired for new DSPy artifacts to load"
        )

        # Functional outcome 2: at least one agent's active artifact
        # version bumped. Not every agent always trains (depends on
        # trace volume); requiring all of them would be brittle. But
        # ZERO advances means optimization produced no usable artifacts.
        versions_after = {a: _artifact_version("default", a) for a in agents}
        advanced = [
            a
            for a in agents
            if versions_after[a] > 0 and versions_after[a] > versions_before[a]
        ]
        assert advanced, (
            f"agent-optimization workflow Succeeded and bounced the runtime, "
            f"but no agent artifact version advanced.\n"
            f"  before: {versions_before}\n  after:  {versions_after}"
        )


@pytest.mark.e2e_heavy
@pytest.mark.skipif(
    not _kubectl_available(), reason="kubectl not available in test environment"
)
class TestScheduledDistillationWorkflow:
    """Scheduled-distillation runs quality_monitor_cli --once which audits
    quality + distills strategies. Functional intent: at least one
    learned_strategy memory was written OR the existing pool has a fresh
    confirmation_count bump."""

    def test_workflow_writes_or_updates_learned_strategy_memory(self):
        if not _cronworkflow_exists("cogniverse-scheduled-distillation"):
            pytest.skip("cogniverse-scheduled-distillation CronWorkflow not deployed")

        tenant_full_id = "default"
        count_before = _count_learned_strategies(tenant_full_id)

        wf = _submit_and_wait_succeeded_heavy("cogniverse-scheduled-distillation")
        logs = _workflow_pod_logs(wf)

        count_after = _count_learned_strategies(tenant_full_id)
        if count_before == -1 or count_after == -1:
            pytest.fail(
                f"learned_strategy count probe failed against runtime; "
                f"before={count_before}, after={count_after}.\n"
                f"--- pod logs (tail 500) ---\n{logs[-3000:]}"
            )

        # Functional outcome: either a new strategy was distilled OR
        # the run recorded that distillation ran but produced no new
        # rows (acceptable when no quality drop fired). Both are
        # visible in the pod logs — bare 'Succeeded' is not enough.
        advanced = count_after > count_before
        recorded = any(
            marker in logs
            for marker in (
                "distillation complete",
                "no strategies to distill",
                '"distilled"',
                "distilled_count",
            )
        )
        assert advanced or recorded, (
            f"scheduled-distillation Succeeded but neither the "
            f"learned_strategy count advanced ({count_before} → "
            f"{count_after}) nor the pod logs recorded a distillation "
            f"outcome marker. Logs tail:\n{logs[-3000:]}"
        )


@pytest.mark.e2e_heavy
@pytest.mark.skipif(
    not _kubectl_available(), reason="kubectl not available in test environment"
)
class TestSyntheticGenerationWorkflow:
    """Weekly synthetic-generation produces training datasets in Phoenix
    for every optimizer type. Functional intent: at least one new
    Phoenix dataset matching ``synthetic-*`` exists after the workflow."""

    def test_workflow_creates_new_phoenix_datasets_for_each_optimizer(self):
        if not _cronworkflow_exists("cogniverse-synthetic-generation"):
            pytest.skip("cogniverse-synthetic-generation CronWorkflow not deployed")

        names_before = _phoenix_dataset_names()

        wf = _submit_and_wait_succeeded_heavy("cogniverse-synthetic-generation")
        logs = _workflow_pod_logs(wf)

        # Wait briefly for the dataset to be visible via the Phoenix API
        # (the workflow writes via OTLP; the HTTP list-datasets endpoint
        # is read-after-write within a few seconds).
        names_after = names_before
        for _ in range(6):
            time.sleep(POLL_INTERVAL_S)
            names_after = _phoenix_dataset_names()
            if names_after - names_before:
                break

        new_datasets = names_after - names_before
        if not new_datasets:
            pytest.fail(
                "synthetic-generation workflow Succeeded but no new "
                "Phoenix datasets appeared.\n"
                f"  before: {sorted(names_before)}\n"
                f"  after:  {sorted(names_after)}\n"
                f"--- pod logs (tail 500) ---\n{logs[-3000:]}"
            )

        # Functional outcome: every requested optimizer type produced
        # at least one dataset (workflow args pass --agents
        # simba,profile,workflow). The chart pins those three so the
        # test pins them too.
        for optimizer in ("simba", "profile", "workflow"):
            matched = [n for n in new_datasets if optimizer in n.lower()]
            assert matched, (
                f"synthetic-generation must produce a dataset for "
                f"optimizer={optimizer!r}; new datasets after run: "
                f"{sorted(new_datasets)}.\n"
                f"--- pod logs (tail 500) ---\n{logs[-3000:]}"
            )
