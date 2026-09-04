"""Memory QoS and the RAM budget the rendered workloads may actually take.

Kubernetes schedules on ``requests`` and kills on ``limits``. A container
whose request is below its limit is Burstable: the scheduler places it as if
it needed the request, and the kernel then has to find the difference when it
takes the limit. Summed across a chart, that difference is memory the node
promised twice.

On this host the arithmetic is unforgiving, because the GPU pool is carved
out of the same system RAM as the pods: ``test_gpu_memory_budget`` reserves a
fixed share for cluster services and hands the rest to the model fractions.
That reservation is only true if the cluster services cannot exceed it, which
is what ``requests == limits`` buys.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.charts.test_gpu_memory_budget import RESERVED_GIB, SYSTEM_RAM_GIB

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"

# Workload kinds whose pods are resident for the life of the release. Jobs and
# CronJobs are transient and are budgeted by the slack term instead.
RESIDENT_KINDS = {"Deployment", "StatefulSet", "DaemonSet"}


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


def _quantity_gib(value: str | None) -> float:
    """Parse a Kubernetes memory quantity into GiB."""
    if value is None:
        return 0.0
    text = str(value)
    for suffix, scale in (
        ("Gi", 1.0),
        ("Mi", 1 / 1024),
        ("Ki", 1 / 1024**2),
        ("G", 1e9 / 1024**3),
        ("M", 1e6 / 1024**3),
    ):
        if text.endswith(suffix):
            return float(text[: -len(suffix)]) * scale
    return float(text) / 1024**3


def _resident_containers(documents: list[dict]) -> list[tuple[str, str, dict]]:
    """Every container of every resident workload, with its owning names."""
    containers = []
    for document in documents:
        if document.get("kind") not in RESIDENT_KINDS:
            continue
        name = document["metadata"]["name"]
        spec = document["spec"]["template"]["spec"]
        for container in spec.get("initContainers", []) + spec["containers"]:
            containers.append((name, container["name"], container))
    return containers


def _is_inference(workload_name: str) -> bool:
    return "-vllm-" in workload_name or workload_name.endswith(
        ("-denseon", "-gliner", "-clap-embed", "-colbert-pylate", "-face-embed")
    )


def test_every_resident_container_reserves_the_memory_it_may_take():
    """No resident container may request less memory than its limit.

    CPU is deliberately left burstable: exceeding a CPU limit throttles the
    container, and the limits sum to more cores than the node has, so
    reserving them would make the release unschedulable. Exceeding a memory
    limit kills the container, and memory the scheduler handed out twice is
    memory the kernel has to find under pressure.
    """
    documents = _render()
    containers = _resident_containers(documents)
    assert containers, "render produced no resident containers"

    burstable = []
    for workload, container_name, container in containers:
        resources = container.get("resources") or {}
        requests = resources.get("requests") or {}
        limits = resources.get("limits") or {}
        if requests.get("memory") != limits.get("memory"):
            burstable.append(
                f"{workload}/{container_name}: "
                f"requests={requests.get('memory')} "
                f"limits={limits.get('memory')}"
            )

    assert burstable == [], (
        "resident containers request less memory than they may take; the "
        "scheduler places them on requests and the kernel must find the "
        "limits:\n  " + "\n  ".join(burstable)
    )


def test_cluster_service_limits_stay_within_the_gpu_budget_reservation():
    """The GPU budget's cluster-service reservation must cover what they may take.

    ``test_gpu_memory_budget`` subtracts a fixed cluster-services figure from
    system RAM before handing the remainder to the model fractions. That
    figure was the summed memory *requests*; once requests equal limits it is
    the whole commitment. If the cluster services can exceed it, the pool
    arithmetic is wrong by the difference and the model fractions are
    oversubscribed against the same RAM.
    """
    documents = _render()
    reserved = RESERVED_GIB["cluster services"]

    total = 0.0
    breakdown = []
    for workload, container_name, container in _resident_containers(documents):
        if _is_inference(workload):
            continue
        limit = _quantity_gib(
            ((container.get("resources") or {}).get("limits") or {}).get("memory")
        )
        total += limit
        breakdown.append(f"{workload}/{container_name}={limit:.2f}Gi")

    assert total <= reserved, (
        f"cluster-service memory limits sum to {total:.2f}Gi but "
        f"test_gpu_memory_budget reserves {reserved}Gi for them; the GPU "
        f"pool arithmetic is oversubscribed by {total - reserved:.2f}Gi.\n  "
        + "\n  ".join(sorted(breakdown))
    )


def test_every_container_reading_the_config_can_authenticate_to_inference():
    """A container that mounts the config must be able to call what it names.

    ``cogniverse-config`` carries the inference endpoints, and under Modal
    serving those reject unauthenticated requests. The dashboard hit this: its
    memory tab constructs Mem0 in-process, and the two Vespa schema deploys
    inside ``initialize`` run *before* the line that raises on the missing key,
    so every Streamlit rerun re-ran both deploys on the render thread until the
    liveness probe killed the pod.

    Derived from the render, not a restated list, so a new container that
    mounts the config without the bearer fails here.
    """
    documents = _render()
    offenders = []
    for document in documents:
        if document.get("kind") not in RESIDENT_KINDS:
            continue
        workload = document["metadata"]["name"]
        spec = document["spec"]["template"]["spec"]
        mounts_config = any(
            (volume.get("configMap") or {}).get("name") == "cogniverse-config"
            for volume in spec.get("volumes", [])
        )
        if not mounts_config:
            continue
        for container in spec["containers"]:
            names = {entry["name"] for entry in container.get("env", [])}
            if "COGNIVERSE_INFERENCE_API_KEY" not in names:
                offenders.append(f"{workload}/{container['name']}")
    assert offenders == [], (
        "containers mount cogniverse-config (which names the inference "
        f"endpoints) but cannot authenticate to them: {sorted(offenders)}"
    )


def _render_as_deployed(*extra: str) -> list[dict]:
    """Render the overlay stack the k3d deploy path actually applies.

    ``_render`` above takes the device overlay alone, which is enough for the
    QoS shape but not for the budget: values.k3s.yaml is what sets the replica
    counts the scheduler multiplies by.
    """
    command = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "-f",
        str(CHART_PATH / "values.k3s.yaml"),
        "-f",
        str(CHART_PATH / "values.rocm.yaml"),
    ]
    for values in extra:
        command.extend(["-f", str(CHART_PATH / values)])
    command.extend(["--set", "runtime.qualityMonitor.tenantId=test-tenant"])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, (
        f"helm template failed (exit {result.returncode}):\n{result.stderr}"
    )
    return [d for d in yaml.safe_load_all(result.stdout) if d]


def _scheduled_memory_gib(documents: list[dict]) -> float:
    """Memory the scheduler must find for one copy of the release.

    A pod's request is ``max(largest init container, sum of the run
    containers)`` -- init containers run before the others, so their
    reservations do not add. Summing all containers flat overstates any
    workload with a heavy init container and understates nothing, so the
    distinction has to be kept or the budget is measuring the wrong number.
    """
    total = 0.0
    for document in documents:
        if document.get("kind") not in RESIDENT_KINDS:
            continue
        spec = document["spec"]["template"]["spec"]
        replicas = document["spec"].get("replicas", 1) or 1
        running = sum(
            _quantity_gib(
                ((c.get("resources") or {}).get("requests") or {}).get("memory")
            )
            for c in spec.get("containers", [])
        )
        init = max(
            [
                _quantity_gib(
                    ((c.get("resources") or {}).get("requests") or {}).get("memory")
                )
                for c in spec.get("initContainers", [])
            ]
            or [0.0]
        )
        total += max(running, init) * replicas
    return total


def test_modal_llm_serving_fits_the_host_the_overlay_targets():
    """The deployed release must fit the node, or nothing schedules.

    Requests are what the scheduler places on, so a release whose requests
    exceed the node does not degrade -- it wedges. The pods that do fit are
    bound and hold their reservations while the ones that do not sit Pending,
    and because the startup gate chain makes the later pods wait on an earlier
    one, a single unplaceable pod at the head strands every pod behind it.
    """
    total = _scheduled_memory_gib(_render_as_deployed("values.modal-llm.yaml"))

    assert round(total, 2) == 109.50, (
        f"resident memory requests under Modal chat serving are {total:.2f}Gi, "
        "not the pinned 109.50Gi"
    )
    assert total <= SYSTEM_RAM_GIB, (
        f"the release requests {total:.2f}Gi but the host has "
        f"{SYSTEM_RAM_GIB}Gi; it cannot schedule"
    )


def test_local_chat_serving_overcommits_this_host_by_the_overlay_delta():
    """Serving both chat models locally exceeds this host by the overlay's 44Gi.

    values.modal-llm.yaml states that figure in prose; this checks it against
    the render.
    """
    local = _scheduled_memory_gib(_render_as_deployed())
    modal = _scheduled_memory_gib(_render_as_deployed("values.modal-llm.yaml"))

    assert round(local - modal, 2) == 44.00, (
        f"the Modal overlay returns {local - modal:.2f}Gi, not the 44Gi its "
        "own header claims"
    )
    assert round(local, 2) == 153.50, (
        f"resident memory requests under local chat serving are {local:.2f}Gi, "
        "not the pinned 153.50Gi"
    )
    assert local > SYSTEM_RAM_GIB, (
        f"local chat serving now fits in {SYSTEM_RAM_GIB}Gi ({local:.2f}Gi); "
        "the Modal overlay is no longer required on this host and the deploy "
        "path's serving default should be revisited"
    )
