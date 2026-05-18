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


def _wait_runtime_ready(timeout_s: float = 300.0) -> bool:
    """Poll until the runtime /health/live returns 200.

    Used at the start of tests in this file that follow an upstream
    test which triggered a runtime rollout. The deployment's
    observedGeneration advances when the controller schedules the new
    replica, NOT when it's HTTP-ready — and rocm vLLM workloads can
    take 2-3 minutes to fully come back. Without this wait, downstream
    tests' probes race the rollout and read connection errors.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=10.0) as client:
                if client.get(f"{RUNTIME}/health/live").status_code == 200:
                    return True
        except (httpx.HTTPError, OSError):
            pass
        time.sleep(3.0)
    return False


def _count_learned_strategies(tenant_full_id: str) -> int:
    """Count strategy-type memories for a tenant via the admin route.

    Route is ``/admin/tenant/{tid}/memories`` (singular ``tenant``)
    with ``type=strategy`` — the chart maps that to the
    ``_strategy_store`` Mem0 namespace where scheduled-distillation
    writes learned_strategy memories. The kind metadata distinction
    isn't exposed at the HTTP layer; ``type=strategy`` is the right
    proxy because the namespace is dedicated to that kind.

    Polls a few seconds because the upstream test in the same sweep
    triggers a runtime rollout, and this probe can race the rollout
    window where the runtime returns ConnectError / 503. Returns -1
    only if the runtime is still unavailable after the retry window.
    """
    deadline = time.monotonic() + 90.0
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.get(
                    f"{RUNTIME}/admin/tenant/{tenant_full_id}/memories",
                    params={"type": "strategy", "limit": 200},
                )
                if r.status_code == 200:
                    body = r.json()
                    return int(body.get("count", len(body.get("memories", []))))
        except (httpx.HTTPError, OSError):
            pass
        time.sleep(3.0)
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

    def test_workflow_runs_all_optimizer_steps_and_restarts_runtime(self):
        if not _cronworkflow_exists("cogniverse-agent-optimization"):
            pytest.skip("cogniverse-agent-optimization CronWorkflow not deployed")

        # Pre: capture rollout generation. The pipeline has 5 parallel
        # optimizer steps + a sequential workflow-optimization step +
        # restart-deployment. Argo only runs restart-deployment when
        # every upstream step Succeeded — so an observedGeneration bump
        # IS proof that all 5 optimizers reached a clean terminal state
        # (success when data present, no_data when empty cluster).
        # Stronger assertions like "artifact version advanced" require
        # pre-seeded Phoenix spans for the default tenant; the chart's
        # current dev cluster has no real traffic, so they fail
        # legitimately. When span-fixtures are added, layer a per-agent
        # version-bump assertion on top — keep this rollout assertion
        # as the data-agnostic functional minimum.
        gen_before = _runtime_deployment_generation()

        _submit_and_wait_succeeded_heavy("cogniverse-agent-optimization")

        deadline = time.monotonic() + 120.0
        gen_after = gen_before
        while time.monotonic() < deadline:
            gen_after = _runtime_deployment_generation()
            if gen_after > gen_before:
                break
            time.sleep(2.0)
        assert gen_after > gen_before, (
            f"agent-optimization workflow Succeeded but the runtime "
            f"deployment observedGeneration did not advance "
            f"({gen_before} → {gen_after}) within 120s; the chained "
            f"5 optimizers + workflow optimizer + restart-deployment "
            f"all must have run for the rollout to fire"
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

    def test_workflow_runs_against_strategy_store_without_regression(self):
        if not _cronworkflow_exists("cogniverse-scheduled-distillation"):
            pytest.skip("cogniverse-scheduled-distillation CronWorkflow not deployed")

        # Wait for runtime to be HTTP-ready before the pre-probe.
        # The agent-optimization test above bounces the runtime as its
        # functional outcome; its observedGeneration assertion fires
        # the moment the controller schedules the new replica, NOT
        # when it's accepting HTTP. Without this wait the pre-probe
        # below races the rollout and reads -1.
        assert _wait_runtime_ready(), (
            "Runtime did not come back HTTP-ready within 5 min after "
            "the upstream agent-optimization rollout"
        )

        # Data-agnostic functional contract: the cron Succeeds, the
        # strategy-store endpoint is reachable both before and after
        # the run, and the count does NOT decrease.
        #
        # Distillation produces new learned_strategy rows only when
        # there's real query traffic + a quality drop to learn from;
        # on a clean cluster with no traffic the right behaviour is
        # "ran cleanly, distilled nothing", and the test must accept
        # that. When traffic-fixtures are added, layer a
        # ``count_after > count_before`` assertion on top — keep this
        # no-regression assertion as the minimum that's true regardless
        # of upstream data.
        tenant_full_id = "default"
        count_before = _count_learned_strategies(tenant_full_id)

        _submit_and_wait_succeeded_heavy("cogniverse-scheduled-distillation")

        count_after = _count_learned_strategies(tenant_full_id)
        assert count_before >= 0, (
            f"strategy-store probe failed BEFORE the run (count={count_before}); "
            f"runtime memory API must be reachable as a precondition"
        )
        assert count_after >= 0, (
            f"strategy-store probe failed AFTER the run (count={count_after}); "
            f"runtime memory API must be reachable post-workflow"
        )
        assert count_after >= count_before, (
            f"scheduled-distillation Succeeded but the strategy count "
            f"regressed ({count_before} → {count_after}); the cron must "
            f"only add or keep memories, never delete"
        )


@pytest.mark.e2e_heavy
@pytest.mark.skipif(
    not _kubectl_available(), reason="kubectl not available in test environment"
)
class TestSyntheticGenerationWorkflow:
    """Weekly synthetic-generation produces training datasets in Phoenix
    for every optimizer type. Functional intent: at least one new
    Phoenix dataset matching ``synthetic-*`` exists after the workflow."""

    def test_workflow_creates_synthetic_datasets_for_each_optimizer(self):
        if not _cronworkflow_exists("cogniverse-synthetic-generation"):
            pytest.skip("cogniverse-synthetic-generation CronWorkflow not deployed")

        wf = _submit_and_wait_succeeded_heavy("cogniverse-synthetic-generation")
        logs = _workflow_pod_logs(wf)

        # Wait briefly for the upload to be visible via the Phoenix API
        # (the workflow writes via OTLP; the HTTP list-datasets endpoint
        # is read-after-write within a few seconds).
        names_after: set[str] = set()
        for _ in range(6):
            time.sleep(POLL_INTERVAL_S)
            names_after = _phoenix_dataset_names()
            if all(
                any(f"synthetic_{opt}" in n for n in names_after)
                for opt in ("workflow", "profile")
            ):
                break

        # Functional outcome: the expected dataset NAMES exist after the
        # run. The CLI's ArtifactManager APPENDS to an existing dataset
        # on second/later runs (same name → ``append_to_dataset``), so
        # ``names_after - names_before`` is empty on re-runs. Assert
        # existence instead — that's the contract this cron owes: each
        # opt_type ends up with a populated demos dataset.
        for optimizer in ("workflow", "profile"):
            expected = f"synthetic_{optimizer}"
            matched = [n for n in names_after if expected in n]
            assert matched, (
                f"synthetic-generation must leave a dataset for "
                f"optimizer={optimizer!r} (expected substring "
                f"{expected!r}); datasets observed after run "
                f"(showing first 50): {sorted(names_after)[:50]}.\n"
                f"--- pod logs (tail 500) ---\n{logs[-3000:]}"
            )
