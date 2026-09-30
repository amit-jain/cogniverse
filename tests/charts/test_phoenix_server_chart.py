"""The Phoenix server container: pinned image, disabled surfaces, startup budget.

The image is pinned by tag and multi-arch index digest. The unauthenticated
MCP endpoint, the in-app agent assistant and the assistant's GitHub tools are
off. The first start after an upgrade runs Phoenix's schema migrations before
/health answers, so a startupProbe holds liveness off for longer than a
migration takes.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"

PHOENIX_IMAGE = (
    "arizephoenix/phoenix:20.16.0"
    "@sha256:d55a4ffac8c670e2d0bf72e44e81e32a73e832b7ce449e6e4567487adfa9d8d6"
)

OVERLAY_STACKS = {
    "base": (),
    "k3s": ("values.k3s.yaml",),
    "k3s+rocm+modal-llm": (
        "values.k3s.yaml",
        "values.rocm.yaml",
        "values.modal-llm.yaml",
    ),
}

pytestmark = [
    pytest.mark.unit,
    pytest.mark.ci_fast,
    pytest.mark.skipif(
        shutil.which("helm") is None,
        reason="helm CLI not installed — chart tests require helm",
    ),
]


def _phoenix_container(*set_args: str, overlays: tuple[str, ...] = ()) -> dict:
    args = ["helm", "template", "cogniverse", str(CHART_PATH)]
    for overlay in overlays:
        args += ["-f", str(CHART_PATH / overlay)]
    args += ["--set", "runtime.qualityMonitor.tenantId=test-tenant"]
    for value in set_args:
        args += ["--set", value]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    statefulsets = [
        doc
        for doc in yaml.safe_load_all(result.stdout)
        if doc
        and doc.get("kind") == "StatefulSet"
        and doc["metadata"]["name"] == "cogniverse-phoenix"
    ]
    assert len(statefulsets) == 1
    containers = statefulsets[0]["spec"]["template"]["spec"]["containers"]
    assert [c["name"] for c in containers] == ["phoenix"]
    return containers[0]


def _env(container: dict) -> dict:
    return {e["name"]: e.get("value") for e in container["env"]}


def _window_seconds(probe: dict) -> int:
    return (
        probe.get("initialDelaySeconds", 0)
        + probe["periodSeconds"] * probe["failureThreshold"]
    )


@pytest.mark.parametrize("stack", sorted(OVERLAY_STACKS))
def test_image_is_pinned_by_tag_and_digest(stack):
    container = _phoenix_container(overlays=OVERLAY_STACKS[stack])

    assert container["image"] == PHOENIX_IMAGE


def test_cleared_digest_falls_back_to_the_tag():
    container = _phoenix_container("phoenix.image.digest=")

    assert container["image"] == "arizephoenix/phoenix:20.16.0"


@pytest.mark.parametrize("stack", sorted(OVERLAY_STACKS))
def test_mcp_server_and_agent_assistant_are_off(stack):
    env = _env(_phoenix_container(overlays=OVERLAY_STACKS[stack]))

    assert (
        env["PHOENIX_ENABLE_MCP_SERVER"],
        env["PHOENIX_DISABLE_AGENT_ASSISTANT"],
        env["PHOENIX_AGENTS_DISABLE_GITHUB"],
    ) == (
        "false",
        "true",
        "true",
    )


@pytest.mark.parametrize("stack", sorted(OVERLAY_STACKS))
def test_startup_probe_holds_liveness_off_through_migrations(stack):
    container = _phoenix_container(overlays=OVERLAY_STACKS[stack])

    assert container["startupProbe"] == {
        "httpGet": {"path": "/health", "port": 6006},
        "periodSeconds": 10,
        "timeoutSeconds": 5,
        "failureThreshold": 60,
    }
    assert _window_seconds(container["startupProbe"]) == 600
    assert container["livenessProbe"] == {
        "httpGet": {"path": "/health", "port": 6006},
        "initialDelaySeconds": 30,
        "periodSeconds": 30,
        "timeoutSeconds": 10,
        "failureThreshold": 3,
    }
