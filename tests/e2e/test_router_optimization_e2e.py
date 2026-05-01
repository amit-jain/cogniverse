"""E2E test: real DSPy MIPROv2 router optimization with student + teacher LLMs.

The router_optimizer wires:
- ``llm_config.primary`` (student) — the runtime's main chat model;
  evaluated against both baseline and optimized prompts.
- ``llm_config.teacher`` — generates few-shot demos that MIPROv2
  considers; used only during compilation, not at runtime.

The teacher (`cyankiwi/Qwen3.6-27B-AWQ-INT4`, dense Qwen 3.6 27B,
AWQ-INT4 quantized) is scaled to zero in the chart by default.
This test scales it up before optimization, runs the real
``router_optimizer.optimize_router``
inside the runtime pod (so it reaches the in-cluster student/teacher
endpoints over their cluster DNS), parses the returned artifact
metrics, and asserts MIPROv2 produced a program that does not regress
relative to the baseline.

Marked ``slow`` and ``requires_teacher_model`` so it never runs in the
default e2e sweep — bring the teacher up explicitly to exercise this
path:

    pytest -m "slow and requires_teacher_model" \\
        tests/e2e/test_router_optimization_e2e.py

Skips cleanly when the teacher deployment doesn't exist (chart wasn't
rendered with the teacher block) or when the host can't bring it
ready inside the timeout.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from tests.e2e.conftest import TENANT_ID, skip_if_no_runtime

pytestmark = [
    pytest.mark.slow,
    pytest.mark.requires_models,
    pytest.mark.requires_teacher_model,
]

KUBECTL_CONTEXT = "k3d-cogniverse"
NAMESPACE = "cogniverse"
RUNTIME_DEPLOYMENT = "deploy/cogniverse-runtime"
RUNTIME_CONTAINER = "runtime"
TEACHER_DEPLOYMENT = "deployment/cogniverse-vllm-llm-teacher"

# Inference sidecars NOT used by router optimization (only the LLMs are
# called for query routing). Scaled to 0 while teacher runs so its
# memory request fits within node allocatable on a 123 GiB unified
# memory host. Restored on teardown.
SIDECARS_TO_PARK = (
    "deployment/cogniverse-vllm-asr",
    "deployment/cogniverse-vllm-colpali",
)

# Teacher cold-start budget. Qwen 3.6 27B AWQ-INT4 loads ~14 GiB
# of weights from HF cache on first run. Chart probes are
# initialDelaySeconds=900, failureThreshold=24, periodSeconds=30 → pod
# can stay unready for up to 900 + 24*30 = 1620s before liveness kills
# it. Test waits up to 1500s by default (cap to fail before the pod
# would self-restart and confuse rollout status).
TEACHER_READY_TIMEOUT_S = int(os.environ.get("TEACHER_READY_TIMEOUT_S", "1500"))

# MIPROv2 budget. ``optimize_router`` does num_trials=20, num_candidates=10,
# minibatch_full_eval_steps=10. Each trial calls the student for
# evaluation; with 30 teacher examples + 33 manual = 63 total, an 80/20
# split = ~13 val examples per trial. End-to-end on Strix Halo (gfx1151)
# with the AWQ-INT4 teacher generating at ~5 tok/s observed: teacher
# example generation alone takes ~50 min for 30 examples (~500 tokens
# each), then bootstrap + 23 trials adds another 30-60 min. Default
# 7200s (2h) covers the observed worst case with margin; CPU-only runs
# need to override OPTIMIZE_TIMEOUT_S higher.
OPTIMIZE_TIMEOUT_S = int(os.environ.get("OPTIMIZE_TIMEOUT_S", "7200"))


def _kubectl(*args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "-n",
            NAMESPACE,
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _teacher_deployment_exists() -> bool:
    """True iff the vllm-llm-teacher deployment is rendered in this chart.

    ``inference.vllm_llm_teacher`` is opt-in. When the chart was rendered
    without it the test must skip — there is nothing to scale up.
    """
    out = _kubectl(
        "get",
        TEACHER_DEPLOYMENT,
        "--ignore-not-found",
        "-o",
        "name",
        timeout=15,
    )
    return bool(out.stdout.strip())


@pytest.fixture(scope="module")
def teacher_pod_ready():
    """Scale the teacher deployment to 1 replica and wait for ready.

    Scales back to 0 on teardown. If the deployment doesn't exist or
    can't come ready inside the budget, skips the whole module rather
    than failing — these are environment problems, not regressions in
    the code under test.
    """
    if not _teacher_deployment_exists():
        pytest.skip(
            f"{TEACHER_DEPLOYMENT} not rendered (chart's "
            f"inference.vllm_llm_teacher block disabled)."
        )

    # Park sidecars not used by routing optimization to free memory.
    # Record their original replica counts so teardown restores them.
    parked_replicas: dict[str, int] = {}
    for dep in SIDECARS_TO_PARK:
        out = _kubectl("get", dep, "-o", "jsonpath={.spec.replicas}", timeout=15)
        try:
            parked_replicas[dep] = int(out.stdout.strip() or "0")
        except ValueError:
            parked_replicas[dep] = 1
        _kubectl("scale", dep, "--replicas=0", timeout=30)

    scale_up = _kubectl("scale", TEACHER_DEPLOYMENT, "--replicas=1", timeout=30)
    if scale_up.returncode != 0:
        for dep, n in parked_replicas.items():
            _kubectl("scale", dep, f"--replicas={n}", timeout=30)
        pytest.skip(
            f"failed to scale teacher up: rc={scale_up.returncode} "
            f"stderr={scale_up.stderr.strip()}"
        )

    rollout = _kubectl(
        "rollout",
        "status",
        TEACHER_DEPLOYMENT,
        f"--timeout={TEACHER_READY_TIMEOUT_S}s",
        timeout=TEACHER_READY_TIMEOUT_S + 60,
    )
    if rollout.returncode != 0:
        _kubectl("scale", TEACHER_DEPLOYMENT, "--replicas=0", timeout=30)
        for dep, n in parked_replicas.items():
            _kubectl("scale", dep, f"--replicas={n}", timeout=30)
        pytest.skip(
            f"teacher did not become ready within {TEACHER_READY_TIMEOUT_S}s: "
            f"{rollout.stderr.strip()}"
        )

    try:
        yield
    finally:
        _kubectl("scale", TEACHER_DEPLOYMENT, "--replicas=0", timeout=30)
        for dep, n in parked_replicas.items():
            _kubectl("scale", dep, f"--replicas={n}", timeout=30)


def _run_optimization_in_pod(num_examples: int = 30) -> dict:
    """Invoke router_optimizer inside the runtime pod and return the
    parsed artifacts dict.

    Running inside the pod lets the optimizer reach the cluster-internal
    student/teacher endpoints over service DNS, matches what the chart's
    optimization CronWorkflow does, and inherits the runtime's wired
    config (LLMConfig.primary / .teacher, telemetry provider).
    """
    # The optimizer prints human-readable progress and saves artifacts
    # via ArtifactManager. We re-fetch the persisted metrics from the
    # ArtifactManager rather than trying to parse stdout — the test
    # cares about whether the run completed and produced sensible
    # baseline+optimized metrics, not about the print stream.
    cmd = [
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
        "python3",
        "-m",
        "cogniverse_agents.optimizer.router_optimizer",
        "--tenant-id",
        TENANT_ID,
        "--use-teacher",
        "--num-examples",
        str(num_examples),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=OPTIMIZE_TIMEOUT_S,
    )

    # Save combined output for post-hoc inspection during long runs.
    log_path = "/tmp/router_optimization_e2e_stdout.log"
    with open(log_path, "w") as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n\n--- stderr ---\n")
            f.write(result.stderr)

    if result.returncode != 0:
        pytest.fail(
            f"router_optimizer failed: rc={result.returncode}\n"
            f"--- stdout (tail) ---\n{result.stdout[-4000:]}\n"
            f"--- stderr (tail) ---\n{result.stderr[-4000:]}"
        )

    # Accuracy lines come from logger.info which writes to stderr in the pod.
    # Combine both streams for the parse so the test works regardless of where
    # the runtime's logging handler directs output.
    combined_output = result.stdout + "\n" + result.stderr
    metrics = _parse_metrics_from_stdout(combined_output)

    # Verify the optimizer actually persisted its run record to Phoenix
    # (not just printed it). Earlier the persistence path was a no-op
    # stub and every run silently lost its metrics. We exec back into
    # the runtime pod (where the telemetry provider is wired) to call
    # ``load_optimization_run`` and confirm the saved record matches.
    persisted = _load_persisted_metrics_from_pod()
    if persisted is None:
        pytest.fail(
            "Optimizer ran successfully but no metrics record was "
            "persisted via ArtifactManager.log_optimization_run — the "
            "persistence path is broken (regressed back to a no-op?)."
        )
    assert persisted["agent_type"] == "router"
    assert persisted["tenant_id"] == TENANT_ID
    assert persisted["metrics"]["baseline"] == metrics["baseline"], (
        f"persisted baseline {persisted['metrics']['baseline']} differs "
        f"from in-process metrics {metrics['baseline']}"
    )
    assert persisted["metrics"]["optimized"] == metrics["optimized"], (
        f"persisted optimized {persisted['metrics']['optimized']} differs "
        f"from in-process metrics {metrics['optimized']}"
    )
    return metrics


def _load_persisted_metrics_from_pod() -> dict | None:
    """Read the ArtifactManager-persisted optimization run from inside
    the runtime pod, where the telemetry provider is configured.
    """
    code = (
        "import asyncio, json\n"
        "from cogniverse_foundation.telemetry import get_telemetry_manager\n"
        "from cogniverse_agents.optimizer.artifact_manager import "
        "ArtifactManager\n"
        f"prov = get_telemetry_manager().get_provider(tenant_id={TENANT_ID!r})\n"
        f"mgr = ArtifactManager(prov, {TENANT_ID!r})\n"
        "rec = asyncio.run(mgr.load_optimization_run('router'))\n"
        "print('__METRICS_JSON__' + json.dumps(rec, default=str))\n"
    )
    cmd = [
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
        "python3",
        "-c",
        code,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("__METRICS_JSON__"):
            payload = line[len("__METRICS_JSON__") :]
            try:
                import json

                return json.loads(payload)
            except json.JSONDecodeError:
                return None
    return None


def _parse_metrics_from_stdout(stdout: str) -> dict:
    """Extract baseline + optimized metrics from the optimizer's stdout.

    The optimizer emits via logger.info:
        Baseline accuracy: {'modality_accuracy': 0.4, ...}
        Optimized accuracy: {'modality_accuracy': 0.7, ...}

    These lines may appear on stdout or stderr depending on the pod's
    logging configuration. Pass the combined stream from the caller.
    """
    import ast

    baseline: dict | None = None
    optimized: dict | None = None
    for line in stdout.splitlines():
        # Match both bare lines and logger-prefixed lines like
        # "INFO:cogniverse_agents.optimizer...:Baseline accuracy: {...}"
        for marker, target in (
            ("Baseline accuracy: ", "baseline"),
            ("Optimized accuracy: ", "optimized"),
        ):
            if marker not in line:
                continue
            payload = line.split(marker, 1)[1]
            try:
                parsed = ast.literal_eval(payload)
            except (SyntaxError, ValueError):
                continue
            if target == "baseline":
                baseline = parsed
            else:
                optimized = parsed

    if baseline is None or optimized is None:
        pytest.fail(
            "Could not extract baseline+optimized accuracy from optimizer "
            f"stdout (last 3KB shown):\n{stdout[-3000:]}"
        )
    return {"baseline": baseline, "optimized": optimized}


@pytest.mark.e2e
@skip_if_no_runtime
class TestRouterOptimizationWithTeacher:
    """Real MIPROv2 student-teacher optimization round-trip.

    Bring the teacher pod up, call ``optimize_router`` against the live
    student + teacher endpoints, verify it returns shaped metrics and
    that the optimized program does not regress on routing accuracy.
    """

    def test_optimize_router_completes_and_does_not_regress(self, teacher_pod_ready):
        """End-to-end: optimization compiles a router and reports
        baseline + optimized metrics, with optimized ≥ baseline on
        overall accuracy.

        We deliberately do NOT assert ``optimized > baseline`` strictly:
        MIPROv2 can plateau on small bootstrapped sets, and the metric
        is two-class with grain at 1/N for small N. The contract is
        that the optimizer ran end-to-end without raising and produced
        an optimized program that's at least as good on the eval split.
        """
        metrics = _run_optimization_in_pod(num_examples=30)

        baseline = metrics["baseline"]
        optimized = metrics["optimized"]

        # Shape: every accuracy field present and a finite float in [0, 1].
        for key in ("modality_accuracy", "generation_accuracy", "overall_accuracy"):
            for label, m in (("baseline", baseline), ("optimized", optimized)):
                assert key in m, f"{label} missing {key!r}: {m}"
                v = m[key]
                assert isinstance(v, (int, float)), (
                    f"{label}.{key} must be numeric, got {type(v).__name__}"
                )
                assert 0.0 <= float(v) <= 1.0, f"{label}.{key}={v} outside [0, 1]"

        # Non-regression: MIPROv2 must not produce a program that is
        # strictly worse than the un-optimized baseline. Equality is
        # acceptable (small eval sets plateau), but a regression means
        # the optimization is broken.
        assert optimized["overall_accuracy"] >= baseline["overall_accuracy"], (
            f"optimized overall accuracy regressed: "
            f"baseline={baseline['overall_accuracy']:.3f}, "
            f"optimized={optimized['overall_accuracy']:.3f}"
        )
