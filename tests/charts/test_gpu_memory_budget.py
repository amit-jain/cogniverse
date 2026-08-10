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
    "cluster services": 25.4,
    "desktop and daemons": 10.0,
    "test containers": 12.0,
    "pylate pods": 8.0,
    "gliner": 3.0,
    "page cache slack": 8.0,
}
MAX_UTILIZATION_SUM = 0.59

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm CLI not installed — chart tests require helm",
)


def _rocm_utilizations() -> dict[str, float]:
    result = subprocess.run(
        [
            "helm",
            "template",
            "cogniverse",
            str(CHART_PATH),
            "-f",
            str(CHART_PATH / "values.rocm.yaml"),
            "--set",
            "runtime.qualityMonitor.tenantId=test-tenant",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"helm template failed (exit {result.returncode}):\n{result.stderr}"
    )

    utilizations = {}
    declared = 0
    for document in yaml.safe_load_all(result.stdout):
        if document is None or document.get("kind") != "Deployment":
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

    assert reserved == 66.4
    assert round(SYSTEM_RAM_GIB - reserved, 2) == 57.06
    # The ceiling has to be buyable out of what is left after the reservations
    # above, not merely out of the nominal pool.
    assert MAX_UTILIZATION_SUM * POOL_GIB == 56.64
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
    assert round((MAX_UTILIZATION_SUM - total) * POOL_GIB, 2) == 9.6


def test_distillation_teacher_is_not_resident_on_the_unified_host():
    utilizations = _rocm_utilizations()

    # 27B AWQ-INT4 weights alone claim ~14.5 GiB of the 96 GiB pool, which
    # does not fit alongside the serving set within the budget above.
    assert "vllm_llm_teacher" not in utilizations
