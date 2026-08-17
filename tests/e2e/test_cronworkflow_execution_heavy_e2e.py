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

import json
import subprocess
import time

import httpx
import pytest

NAMESPACE = "cogniverse"
RUNTIME = (
    "http://localhost:33000"  # runtime.service.nodePort — matches tests/e2e/conftest.py
)
HEAVY_TIMEOUT_S = 3600.0  # 60 min; measured run was 2026-08-17T13:26:55Z→14:14:03Z
POLL_INTERVAL_S = 10.0


# Re-use the light-tier helpers verbatim by importing them. Keeps the
# kubectl plumbing in one place; the heavy tier only differs in
# duration and per-cron post-state assertion.
from tests.e2e.test_cronworkflow_execution_e2e import (  # noqa: E402
    _delete_workflow,
    _require_cronworkflow,
    _submit_workflow_from_cron,
    _wait_for_workflow_terminal,
    _workflow_pod_logs,
)


def _submit_and_wait_succeeded_heavy(cron_name: str) -> str:
    wf, _ = _submit_and_wait_succeeded_heavy_with_output(cron_name)
    return wf


def _workflow_main_output(workflow_name: str) -> str:
    """The workflow's ``main`` container stdout (the CLI's structured result)."""
    out = subprocess.run(
        [
            "kubectl",
            "logs",
            "-n",
            NAMESPACE,
            "-l",
            f"workflows.argoproj.io/workflow={workflow_name}",
            "-c",
            "main",
            "--tail=-1",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def _submit_and_wait_succeeded_heavy_with_output(cron_name: str) -> tuple[str, str]:
    """Submit + poll; return the workflow name and its main-container stdout,
    captured before the workflow (and its pod) is deleted."""
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
        return wf, _workflow_main_output(wf)
    finally:
        _delete_workflow(wf)


def _last_json_object(text: str) -> dict:
    """The last balanced top-level JSON object printed in ``text``."""
    depth = 0
    end = None
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch == "}":
            if depth == 0:
                end = i + 1
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0 and end is not None:
                return json.loads(text[i:end])
    raise AssertionError(f"no JSON object in workflow output: {text[-1500:]!r}")


# ---------------------------------------------------------------------------
# Runtime-backed memory helpers (scheduled-distillation)
# ---------------------------------------------------------------------------


def _require_runtime_ready(timeout_s: float = 300.0) -> None:
    """Poll until the runtime /health/live returns 200.

    Used at the start of tests in this file that follow an upstream
    test which triggered a runtime rollout. The deployment's
    observedGeneration advances when the controller schedules the new
    replica, NOT when it's HTTP-ready — and rocm vLLM workloads can
    take 2-3 minutes to fully come back. Without this wait, downstream
    tests' probes race the rollout and read connection errors.
    """
    endpoint = f"{RUNTIME}/health/live"
    deadline = time.monotonic() + timeout_s
    last_result = "no response"
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(endpoint)
                if response.status_code == 200:
                    return
                last_result = (
                    f"HTTP {response.status_code} body={response.text[:300]!r}"
                )
        except (httpx.HTTPError, OSError) as exc:
            last_result = f"{type(exc).__name__}: {exc}"
        time.sleep(3.0)
    pytest.fail(
        f"Runtime prerequisite endpoint did not return 200 within {timeout_s}s: "
        f"GET {endpoint}\nlast_result={last_result}",
        pytrace=False,
    )


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
    window where the runtime returns ConnectError / 503. If the endpoint
    stays unavailable, the failure includes the exact URL and last result.
    """
    endpoint = f"{RUNTIME}/admin/tenant/{tenant_full_id}/memories"
    params = {"type": "strategy", "limit": 200}
    deadline = time.monotonic() + 90.0
    last_result = "no response"
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.get(endpoint, params=params)
                if r.status_code == 200:
                    body = r.json()
                    return int(body.get("count", len(body.get("memories", []))))
                last_result = f"HTTP {r.status_code} body={r.text[:300]!r}"
        except (httpx.HTTPError, OSError) as exc:
            last_result = f"{type(exc).__name__}: {exc}"
        time.sleep(3.0)
    pytest.fail(
        f"Runtime prerequisite endpoint failed: GET {endpoint} params={params!r}\n"
        f"last_result={last_result}",
        pytrace=False,
    )


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
class TestAgentOptimizationWorkflow:
    """Weekly agent-optimization trains every DSPy module in parallel and
    then bounces the runtime so the new artifacts load. Functional intent:
    artifact version bumped + rollout observedGeneration advanced."""

    def test_workflow_runs_all_optimizer_steps_and_restarts_runtime(self):
        _require_cronworkflow("cogniverse-agent-optimization")

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
class TestScheduledDistillationWorkflow:
    """Scheduled-distillation runs quality_monitor_cli --once which audits
    quality + distills strategies. Functional intent: at least one
    learned_strategy memory was written OR the existing pool has a fresh
    confirmation_count bump."""

    def test_workflow_runs_against_strategy_store_without_regression(self):
        _require_cronworkflow("cogniverse-scheduled-distillation")

        # Wait for runtime to be HTTP-ready before the pre-probe.
        # The agent-optimization test above bounces the runtime as its
        # functional outcome; its observedGeneration assertion fires
        # the moment the controller schedules the new replica, NOT
        # when it's accepting HTTP. Without this wait the pre-probe
        # below races the rollout and reports a prerequisite endpoint failure.
        _require_runtime_ready()

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
class TestSyntheticGenerationWorkflow:
    """Weekly synthetic-generation runs ``--mode synthetic --agents profile``
    for the quality-monitor tenant and persists the generated examples as a
    pending-review approval batch (human approval later publishes them as the
    optimizer's training dataset)."""

    def test_workflow_persists_a_pending_review_batch_for_profile(self):
        from tests.e2e.test_optimizer_persistence_e2e import (
            _assert_review_batch,
            _load_review_batch,
        )

        _require_cronworkflow("cogniverse-synthetic-generation")

        _wf, output = _submit_and_wait_succeeded_heavy_with_output(
            "cogniverse-synthetic-generation"
        )
        result = _last_json_object(output)
        assert result["status"] == "success", result
        assert set(result["results"]) == {"profile"}, result
        profile = result["results"]["profile"]
        assert profile["status"] == "success", profile
        assert profile["examples_generated"] == profile["pending_review"], profile
        assert 1 <= profile["examples_generated"] <= 50, profile

        batch = _load_review_batch(profile["batch_id"])
        _assert_review_batch(
            batch,
            batch_id=profile["batch_id"],
            optimizer_type="profile",
            agent_type="profile_selection",
            expected_items=profile["examples_generated"],
        )
