"""Memory of the long-lived processes the runtime image runs.

With one glibc malloc arena per thread, what a worker thread frees (decoded
frames, base64 images, parsed responses) stays resident in that thread's
arena. Fed the 22 jobs one e2e ingestor pod ran before its OOM kill, the
ingestion worker in the runtime image grew by about 140 MiB per 1280x720
frame-profile job and died at its 2 GiB limit on the last one; with
``MALLOC_ARENA_MAX=2`` the same sequence finished with a cgroup peak of
1212 MiB, and 990 MiB once the chunk embeddings are held as float32 and
ffmpeg runs on 4 threads.
"""

import pytest

from tests.charts.test_litellm_bundled_cost_map import (
    RUNTIME_DOCKERFILE,
    _final_stage_env,
    _rendered_cogniverse_containers,
)
from tests.charts.test_memory_qos_budget import _render

VARIABLE = "MALLOC_ARENA_MAX"


def _resources(workload: str) -> dict:
    (container,) = [
        container
        for document in _render()
        if document.get("kind") == "Deployment"
        and document["metadata"]["name"] == workload
        for container in document["spec"]["template"]["spec"]["containers"]
    ]
    return container["resources"]


def test_the_runtime_image_caps_malloc_arenas_at_two():
    assert _final_stage_env(RUNTIME_DOCKERFILE)[VARIABLE] == "2"


@pytest.mark.parametrize("backend", ["cpu", "cuda", "rocm"])
def test_no_workload_overrides_the_image_s_arena_cap(backend):
    containers = _rendered_cogniverse_containers(
        f"runtime.backend={backend}", f"dashboard.backend={backend}"
    )

    assert {
        "Deployment/cogniverse-runtime",
        "Deployment/cogniverse-ingestor",
        "Deployment/cogniverse-quality-monitor",
    } <= {owner for owner, _ in containers}
    assert [
        (owner, entry)
        for owner, container in containers
        for entry in container.get("env") or []
        if entry.get("name") == VARIABLE
    ] == []


def test_the_ingestor_limit_covers_its_measured_peak_twice():
    """990 MiB measured over the 22-job sequence; 2 GiB leaves the same again
    for longer videos and higher-resolution frames."""
    assert _resources("cogniverse-ingestor") == {
        "limits": {"cpu": "4", "memory": "2Gi"},
        "requests": {"cpu": "1", "memory": "2Gi"},
    }


def test_the_quality_monitor_limit_covers_its_measured_peak():
    """The monitor died at 3 GiB in its annotation cycle, pulling the
    flywheel tenant's 24 h window of 10,000 spans with every attribute: that
    pull raised a 118 MiB process to 1617 MiB. Pulling span ids, then only the
    annotated spans' rows, the cycle peaks at 355 MiB; the running pod sits at
    588 MiB between cycles."""
    assert _resources("cogniverse-quality-monitor") == {
        "limits": {"cpu": "1", "memory": "3Gi"},
        "requests": {"cpu": "250m", "memory": "3Gi"},
    }
