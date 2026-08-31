"""Unified-pool GPU memory budget rendered by the Helm chart.

``--gpu-memory-utilization`` is a fraction of the GPU pool, and on a unified
host that pool is carved out of system RAM. Every fraction handed to a model
is memory the desktop, the cluster's CPU-side services and the test
containers no longer have, so the enabled fractions have to sum to less than
the whole pool with room left over.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"

# Measured on the ROCm host this overlay targets:
#   /sys/class/drm/card1/device/mem_info_gtt_total = 103079215104 B = 96 GiB
#   /proc/meminfo MemTotal                         = 129462244 kB  = 123.46 GiB
POOL_GIB = 96.0
SYSTEM_RAM_GIB = 123.46

# Everything that must fit in system RAM outside the vLLM reservations.
# Cluster figure is the summed memory requests of the non-inference workloads
# rendered by this same overlay; the rest is desktop + daemons, the e2e
# suite's own containers, the pylate pods that allocate from the pool without
# declaring a fraction, and page-cache slack.
RESERVED_GIB = {
    # Summed memory LIMITS of the non-inference workloads rendered by this
    # overlay, which since requests==limits is their whole commitment. It was
    # previously their summed requests, which understated the reservation by
    # every gigabyte the scheduler had handed out twice.
    # Pinned against the render by
    # test_memory_qos_budget.test_cluster_service_limits_stay_within_the_gpu_budget_reservation.
    "cluster services": 30.5,
    "desktop and daemons": 10.0,
    "test containers": 12.0,
    # Pool-allocating pods that declare no fraction, reserved at their
    # memory limits (2 x 4Gi). See test_pool_pods_declare_a_fraction_or_limit.
    "pylate pods": 8.0,
    # CPU-only, but reserved at its limit rather than its request because the
    # limit is what it may actually take.
    "gliner": 8.0,
    "page cache slack": 8.0,
}
# Bounded by what is left of system RAM after RESERVED_GIB, not by the nominal
# pool: 123.46 - 76.5 = 46.96 GiB, and 0.48 * 96 = 46.08 GiB fits inside it.
MAX_UTILIZATION_SUM = 0.48

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm CLI not installed — chart tests require helm",
)


def _render(*extra: str) -> list[dict]:
    command = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "-f",
        str(CHART_PATH / "values.rocm.yaml"),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
    ]
    command.extend(extra)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        f"helm template failed (exit {result.returncode}):\n{result.stderr}"
    )
    return [d for d in yaml.safe_load_all(result.stdout) if d]


def _rocm_utilizations() -> dict[str, float]:
    utilizations = {}
    declared = 0
    for document in _render():
        if document.get("kind") != "Deployment":
            continue
        component = document["metadata"]["labels"].get(
            "app.kubernetes.io/component", ""
        )
        if not component.startswith("inference-"):
            continue
        container = document["spec"]["template"]["spec"]["containers"][0]
        # Two rendered shapes: a plain arg list, and a shell-wrapped serve
        # script whose flags are single-quoted across line continuations.
        # Normalizing quotes and backslashes reads both.
        blob = (
            " ".join(container.get("args") or [])
            + " "
            + " ".join(container.get("command") or [])
        )
        blob = re.sub(r"[\\'\"]", " ", blob)
        declared += blob.count("--gpu-memory-utilization")
        match = re.search(r"--gpu-memory-utilization\s+([0-9.]+)", blob)
        if match:
            utilizations[component.removeprefix("inference-")] = float(match.group(1))

    # A service whose rendered form this parser cannot read would drop out of
    # the budget silently, which is the one way this guard could pass while
    # the pool is oversubscribed.
    assert len(utilizations) == declared, (
        f"parsed {len(utilizations)} fractions but the render declares "
        f"{declared}; a service is escaping the budget"
    )
    return utilizations


def test_reserved_ram_leaves_the_expected_pool_share_for_models():
    reserved = sum(RESERVED_GIB.values())

    assert reserved == 76.5
    assert round(SYSTEM_RAM_GIB - reserved, 2) == 46.96
    # The ceiling has to be buyable out of what is left after the reservations
    # above, not merely out of the nominal pool.
    assert round(MAX_UTILIZATION_SUM * POOL_GIB, 2) == 46.08
    assert MAX_UTILIZATION_SUM * POOL_GIB <= SYSTEM_RAM_GIB - reserved


def test_enabled_rocm_models_fit_the_unified_pool_budget():
    utilizations = _rocm_utilizations()

    assert utilizations == {
        "vllm_colpali": 0.18,
        "vllm_llm_student": 0.22,
        "vllm_asr": 0.04,
        "denseon": 0.05,
    }

    total = round(sum(utilizations.values()), 4)
    assert total == 0.49
    assert total <= MAX_UTILIZATION_SUM
    assert round(total * POOL_GIB, 2) == 47.04
    # Named headroom: what the budget still has after every model reserves.
    assert round((MAX_UTILIZATION_SUM - total) * POOL_GIB, 2) == 4.8


def _pool_pods(*extra: str) -> dict[str, dict]:
    """Inference pods that can allocate from the unified pool.

    A pod reaches the pool exactly when the chart gives it the ROCm device
    nodes, so the /dev/kfd + /dev/dri mounts are the structural marker. Any
    future service rendered with them is picked up here without the test
    having to name it.
    """
    pods = {}
    for document in _render(*extra):
        if document.get("kind") != "Deployment":
            continue
        component = document["metadata"]["labels"].get(
            "app.kubernetes.io/component", ""
        )
        if not component.startswith("inference-"):
            continue
        spec = document["spec"]["template"]["spec"]
        volumes = {v["name"] for v in spec.get("volumes", [])}
        if not {"kfd", "dri"} <= volumes:
            continue
        container = spec["containers"][0]
        blob = (
            " ".join(container.get("args") or [])
            + " "
            + " ".join(container.get("command") or [])
        )
        blob = re.sub(r"[\\'\"]", " ", blob)
        match = re.search(r"--gpu-memory-utilization\s+([0-9.]+)", blob)
        limit = container.get("resources", {}).get("limits", {}).get("memory")
        pods[component.removeprefix("inference-")] = {
            "fraction": float(match.group(1)) if match else None,
            "limit_gib": float(limit.removesuffix("Gi")) if limit else None,
        }
    return pods


def test_pool_pods_declare_a_fraction_or_limit():
    pods = _pool_pods()

    assert pods == {
        "vllm_colpali": {"fraction": 0.18, "limit_gib": 20.0},
        "vllm_llm_student": {"fraction": 0.22, "limit_gib": 24.0},
        "vllm_llm_teacher": {"fraction": 0.2, "limit_gib": 20.0},
        "vllm_asr": {"fraction": 0.04, "limit_gib": 6.0},
        "denseon": {"fraction": 0.05, "limit_gib": 4.0},
        # PyTorch services: the caching allocator has no fraction knob, so
        # their declared bound is the memory limit, reserved above.
        "colbert_pylate": {"fraction": None, "limit_gib": 4.0},
        "code_colbert_pylate": {"fraction": None, "limit_gib": 4.0},
    }

    unbounded = {
        name
        for name, pod in pods.items()
        if pod["fraction"] is None and pod["limit_gib"] is None
    }
    assert unbounded == set()


def test_fraction_free_pool_pods_are_reserved_at_their_limits():
    pods = _pool_pods()

    limit_bounded = sorted(
        name for name, pod in pods.items() if pod["fraction"] is None
    )

    assert limit_bounded == ["code_colbert_pylate", "colbert_pylate"]
    assert sum(pods[name]["limit_gib"] for name in limit_bounded) == 8.0
    # The reserve above must cover them, or they are consuming pool the
    # budget already promised to something else.
    assert RESERVED_GIB["pylate pods"] == 8.0


def test_budget_guard_rejects_an_unbounded_pool_pod():
    """A pool pod with neither a fraction nor a limit must be caught."""
    pods = _pool_pods("--set", "inference.colbert_pylate.resources.limits.memory=null")

    assert pods["colbert_pylate"] == {"fraction": None, "limit_gib": None}

    unbounded = {
        name
        for name, pod in pods.items()
        if pod["fraction"] is None and pod["limit_gib"] is None
    }
    assert unbounded == {"colbert_pylate"}


def test_distillation_teacher_is_not_resident_on_the_unified_host():
    utilizations = _rocm_utilizations()

    # 27B AWQ-INT4 weights alone claim ~14.5 GiB of the 96 GiB pool, which
    # does not fit alongside the serving set within the budget above.
    assert "vllm_llm_teacher" not in utilizations


def _pylate_request_limits() -> dict[str, dict[str, str]]:
    limits = {}
    for document in _render():
        if document.get("kind") != "Deployment":
            continue
        component = document["metadata"]["labels"].get(
            "app.kubernetes.io/component", ""
        )
        if not component.startswith("inference-"):
            continue
        container = document["spec"]["template"]["spec"]["containers"][0]
        env = {e["name"]: e.get("value") for e in container.get("env", [])}
        keys = {"MAX_INPUT_ITEMS", "MAX_INPUT_CHARS", "ENCODE_BATCH_SIZE"}
        if keys <= env.keys():
            limits[component.removeprefix("inference-")] = {
                k: env[k] for k in sorted(keys)
            }
    return limits


def test_pylate_pods_carry_request_bounds():
    limits = _pylate_request_limits()

    # colbert_pylate sets them in values; code_colbert_pylate omits the block
    # and takes the template defaults. Both must arrive identical.
    assert limits == {
        "colbert_pylate": {
            "ENCODE_BATCH_SIZE": "32",
            "MAX_INPUT_CHARS": "2000000",
            "MAX_INPUT_ITEMS": "256",
        },
        "code_colbert_pylate": {
            "ENCODE_BATCH_SIZE": "32",
            "MAX_INPUT_CHARS": "2000000",
            "MAX_INPUT_ITEMS": "256",
        },
    }


def test_request_bounds_render_as_plain_integers():
    """YAML reads a large plain integer as a float, which renders as 2e+06.

    The server parses these with int(), so a float-formatted value crashes
    the pod at startup rather than at deploy time.
    """
    for service, bounds in _pylate_request_limits().items():
        for name, value in bounds.items():
            assert str(int(value)) == value, f"{service}.{name} is not an integer"
