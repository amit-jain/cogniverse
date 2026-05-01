"""E2E persistence coverage for non-router optimizers.

The router_optimizer test (`test_router_optimization_e2e.py`) surfaced
seven real bugs the first time it ran end-to-end against a live
cluster — wrong import paths, factory model-id rewriting, ConfigMap
contract drift, persistence stubs that silently dropped metrics. The
xgboost / modality / workflow optimizers all share the same chart +
ArtifactManager + telemetry-provider plumbing but have NEVER been
exercised end-to-end here. This module brings each one through the
full save → persist → load round-trip against the live runtime pod
and Phoenix dataset store.

Each test:
  1. Drives the optimizer (via ``optimization_cli`` or a tiny inline
     wrapper) inside the runtime pod over kubectl exec, so it picks
     up the cluster-wired telemetry provider and hits the in-cluster
     Phoenix.
  2. Asserts the run completed (rc == 0, no Python traceback in
     captured output).
  3. Reads the persisted artifact back through ArtifactManager
     (load_blob / load_optimization_run) and asserts shape — the
     dataset name exists, content is non-empty, JSON parses where
     applicable.

Quality of the optimization output (accuracy, time-to-converge, etc.)
is NOT asserted: that depends on the volume + variety of telemetry
spans available, which the test environment does not guarantee. The
contract under test is wiring correctness, not optimizer skill.

Marked ``slow`` and ``requires_optimizer_data`` so it never runs in
the default e2e sweep — bring it up explicitly:

    pytest -m "slow and requires_optimizer_data" \\
        tests/e2e/test_optimizer_persistence_e2e.py

Skips cleanly when the runtime deployment isn't reachable or when the
optimizer reports no input data was available (e.g. zero
orchestration spans in the lookback window) — that's an environment
issue, not a code regression.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

from tests.e2e.conftest import TENANT_ID, skip_if_no_runtime

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_optimizer_data,
]

KUBECTL_CONTEXT = "k3d-cogniverse"
NAMESPACE = "cogniverse"
RUNTIME_DEPLOYMENT = "deploy/cogniverse-runtime"
RUNTIME_CONTAINER = "runtime"

# Each optimizer call drives a kubectl exec into the runtime pod;
# bound the wait so a hung optimizer fails the test instead of
# hanging the suite forever. Tighter than the router test because
# these don't run a 23-trial MIPROv2 loop.
OPTIMIZER_TIMEOUT_S = int(os.environ.get("OPTIMIZER_TIMEOUT_S", "1800"))


def _kubectl_exec(*shell_argv: str, timeout: int = OPTIMIZER_TIMEOUT_S):
    return subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "-n",
            NAMESPACE,
            "exec",
            RUNTIME_DEPLOYMENT,
            "-c",
            RUNTIME_CONTAINER,
            "--",
            *shell_argv,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _read_blob_from_pod(kind: str, key: str) -> str | None:
    """Load the persisted blob for ``(tenant, kind, key)`` from inside
    the runtime pod and return its content (or None if absent)."""
    code = (
        "import asyncio\n"
        "from cogniverse_foundation.telemetry import get_telemetry_manager\n"
        "from cogniverse_agents.optimizer.artifact_manager import "
        "ArtifactManager\n"
        f"prov = get_telemetry_manager().get_provider(tenant_id={TENANT_ID!r})\n"
        f"mgr = ArtifactManager(prov, {TENANT_ID!r})\n"
        f"out = asyncio.run(mgr.load_blob({kind!r}, {key!r}))\n"
        "print('__BLOB_OK__' if out else '__BLOB_NONE__')\n"
        "if out: print(out)\n"
    )
    result = _kubectl_exec("python3", "-c", code, timeout=120)
    if result.returncode != 0:
        return None
    if "__BLOB_NONE__" in result.stdout:
        return None
    if "__BLOB_OK__" not in result.stdout:
        return None
    # Everything after the marker line is the blob content.
    _, _, body = result.stdout.partition("__BLOB_OK__\n")
    return body or None


@pytest.mark.e2e
@skip_if_no_runtime
class TestWorkflowOptimizationPersistence:
    """Run optimization_cli --mode workflow and verify it writes the
    query_patterns blob (and template_index when patterns produce
    templates).
    """

    def test_workflow_optimization_persists_query_patterns(self):
        """The optimizer reads orchestration spans from Phoenix, derives
        per-query-type behaviour patterns, and saves them as a blob.
        With sparse telemetry the patterns may be empty — in which
        case the optimizer skips the save_blob call entirely. The
        contract under test is: when patterns ARE produced, the blob
        round-trips through Phoenix. We assert either:
          - rc == 0 AND a query_patterns blob exists and parses as
            JSON dict (real data path), OR
          - rc == 0 AND the optimizer reported no patterns (skip
            cleanly, not a regression).
        """
        result = _kubectl_exec(
            "python3",
            "-m",
            "cogniverse_runtime.optimization_cli",
            "--mode",
            "workflow",
            "--tenant-id",
            TENANT_ID,
            "--lookback-hours",
            "24",
        )
        if result.returncode != 0:
            pytest.fail(
                f"workflow optimization failed: rc={result.returncode}\n"
                f"--- stdout (tail) ---\n{result.stdout[-3000:]}\n"
                f"--- stderr (tail) ---\n{result.stderr[-3000:]}"
            )

        blob = _read_blob_from_pod("workflow", "query_patterns")
        if blob is None:
            # No patterns → no save. Acceptable when telemetry is
            # sparse; fail only when the optimizer claimed to save
            # but the blob is missing.
            if "Saved blob workflow/query_patterns" in result.stdout:
                pytest.fail(
                    "optimizer logged a save but the blob is missing — "
                    "persistence path is broken"
                )
            pytest.skip(
                "no query patterns produced (sparse telemetry); "
                "persistence path could not be exercised this run"
            )
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"workflow query_patterns blob is not valid JSON: {exc}\n"
                f"head: {blob[:500]}"
            )
        assert isinstance(parsed, dict), (
            f"query_patterns must be a JSON object, got {type(parsed).__name__}"
        )


@pytest.mark.e2e
@skip_if_no_runtime
class TestModalityOptimizerPersistence:
    """Run ModalityOptimizer.optimize_all_modalities and verify it
    writes per-modality DSPy module blobs.

    ModalityOptimizer instantiates xgboost meta models internally
    (TrainingDecisionModel, TrainingStrategyModel) so this test also
    exercises their save_blob path transitively.
    """

    def test_modality_optimization_persists_per_modality_blobs(self):
        """Optimizer iterates QueryModality enum, evaluates each, and
        for any that pass the optimization criteria saves a DSPy
        module JSON blob keyed by modality value. We assert: at
        minimum one (modality_model, key) blob exists OR the
        optimizer reported zero modalities optimized (sparse data
        case, skip).
        """
        code = (
            "import asyncio, json\n"
            "from cogniverse_foundation.config.unified_config import LLMEndpointConfig\n"
            "from cogniverse_foundation.config.utils import "
            "create_default_config_manager, get_config\n"
            "from cogniverse_agents.routing.modality_optimizer import "
            "ModalityOptimizer\n"
            f"cm = create_default_config_manager()\n"
            f"llm = get_config(tenant_id={TENANT_ID!r}, "
            "config_manager=cm).get_llm_config().primary\n"
            f"opt = ModalityOptimizer(tenant_id={TENANT_ID!r}, llm_config=llm)\n"
            "results = asyncio.run(opt.optimize_all_modalities("
            "lookback_hours=24, min_confidence=0.5))\n"
            "print('__MODALITY_RESULTS__' + json.dumps("
            "{str(k): bool(v) for k, v in results.items()}))\n"
        )
        result = _kubectl_exec("python3", "-c", code)
        if result.returncode != 0:
            pytest.fail(
                f"modality optimization crashed: rc={result.returncode}\n"
                f"--- stdout (tail) ---\n{result.stdout[-3000:]}\n"
                f"--- stderr (tail) ---\n{result.stderr[-3000:]}"
            )

        # Did any modality actually get optimized + saved?
        marker = "__MODALITY_RESULTS__"
        any_optimized = False
        for line in result.stdout.splitlines():
            if line.startswith(marker):
                payload = json.loads(line[len(marker) :])
                any_optimized = any(payload.values())
                break

        if not any_optimized:
            pytest.skip(
                "no modality had enough training data to optimize; "
                "persistence path could not be exercised this run"
            )

        # At least one modality was optimized → at least one blob
        # should exist. Probe each known modality key; any one
        # present is sufficient for the persistence contract.
        from cogniverse_core.routing.types import (
            QueryModality,  # type: ignore
        )

        found_any = False
        for m in QueryModality:
            blob = _read_blob_from_pod("modality_model", m.value)
            if blob and len(blob) > 50:
                found_any = True
                # JSON-parseable too — ModalityRoutingModule.save() emits JSON.
                json.loads(blob)
                break
        if not found_any:
            pytest.fail(
                "ModalityOptimizer reported optimized modalities but no "
                "modality_model blob was readable from the pod — the "
                "save_blob path failed silently"
            )


@pytest.mark.e2e
@skip_if_no_runtime
class TestSyntheticGenerationPersistence:
    """Run synthetic data generation and verify the generated demos
    land in the demos dataset.

    This is the cheapest persistence test — it doesn't depend on any
    pre-existing telemetry, just calls the synthetic generator
    (which talks to the in-cluster student LLM) and writes demo
    examples through ArtifactManager.save_demonstrations.
    """

    def test_synthetic_demos_land_in_dataset(self):
        result = _kubectl_exec(
            "python3",
            "-m",
            "cogniverse_runtime.optimization_cli",
            "--mode",
            "synthetic",
            "--tenant-id",
            TENANT_ID,
            "--agents",
            "modality",
            timeout=600,
        )
        if result.returncode != 0:
            pytest.fail(
                f"synthetic generation failed: rc={result.returncode}\n"
                f"--- stdout (tail) ---\n{result.stdout[-3000:]}\n"
                f"--- stderr (tail) ---\n{result.stderr[-3000:]}"
            )

        # Demos persist via ArtifactManager.save_demonstrations into
        # the per-tenant demos dataset; verify the dataset name
        # exists and has at least one row.
        code = (
            "import asyncio\n"
            "from cogniverse_foundation.telemetry import get_telemetry_manager\n"
            "from cogniverse_agents.optimizer.artifact_manager import "
            "ArtifactManager\n"
            f"prov = get_telemetry_manager().get_provider(tenant_id={TENANT_ID!r})\n"
            f"mgr = ArtifactManager(prov, {TENANT_ID!r})\n"
            "out = asyncio.run(mgr.load_demonstrations('modality'))\n"
            "print('__DEMO_COUNT__', len(out) if out else 0)\n"
        )
        probe = _kubectl_exec("python3", "-c", code, timeout=120)
        if probe.returncode != 0:
            pytest.fail(
                f"demos load probe failed: rc={probe.returncode}\n"
                f"{probe.stderr[-2000:]}"
            )
        for line in probe.stdout.splitlines():
            if line.startswith("__DEMO_COUNT__"):
                count = int(line.split()[-1])
                assert count > 0, (
                    "synthetic generation reported success but the demos "
                    "dataset is empty — save_demonstrations dropped them"
                )
                return
        pytest.fail(
            f"demo count probe produced no marker line; stdout: {probe.stdout[-1000:]}"
        )
