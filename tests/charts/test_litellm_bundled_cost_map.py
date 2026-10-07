"""Every cogniverse process runs with litellm's bundled model cost map.

litellm's import fetches its cost map from raw.githubusercontent.com unless
``LITELLM_LOCAL_MODEL_COST_MAP`` is ``True``, so a process without it waits
on GitHub (up to 5 s) at its first LM call. The runtime and dashboard images
set it; every chart workload that runs cogniverse code runs one of those two
images and leaves the image's value alone.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from cogniverse_cli.images import IMAGE_DOCKERFILES

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"
RUNTIME_DOCKERFILE = REPO_ROOT / "libs" / "runtime" / "Dockerfile"
DASHBOARD_DOCKERFILE = REPO_ROOT / "libs" / "dashboard" / "Dockerfile"
VARIABLE = "LITELLM_LOCAL_MODEL_COST_MAP"

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm CLI not installed — chart tests require helm",
)


def _final_stage_env(dockerfile: Path) -> dict[str, str]:
    text = dockerfile.read_text()
    final = text[text.rindex("\nFROM ") :]
    env: dict[str, str] = {}
    for line in final.splitlines():
        if line.startswith("ENV ") and "=" in line:
            name, value = line[len("ENV ") :].split("=", 1)
            env[name.strip()] = value.strip()
    return env


def _image_repositories() -> dict[str, set[str]]:
    values = yaml.safe_load((CHART_PATH / "values.yaml").read_text())
    repositories: dict[str, set[str]] = {}
    for image in ("runtime", "dashboard"):
        block = values[image]
        repositories[image] = {block["image"]["repository"]} | {
            entry["repository"] for entry in block["imagesByBackend"].values()
        }
    repositories["runtime"].add(values["ingestor"]["image"]["repository"])
    return repositories


def _containers(node, owner):
    if isinstance(node, dict):
        if isinstance(node.get("image"), str):
            yield owner, node
        for value in node.values():
            yield from _containers(value, owner)
    elif isinstance(node, list):
        for value in node:
            yield from _containers(value, owner)


def _rendered_cogniverse_containers(*set_args: str) -> list[tuple[str, dict]]:
    args = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
    ]
    for value in set_args:
        args += ["--set", value]
    rendered = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    repositories = set().union(*_image_repositories().values())
    found = []
    for doc in yaml.safe_load_all(rendered):
        if not doc:
            continue
        owner = f"{doc['kind']}/{doc['metadata']['name']}"
        for _, container in _containers(doc, owner):
            if container["image"].rsplit(":", 1)[0] in repositories:
                found.append((owner, container))
    return found


def test_the_checked_dockerfiles_are_the_ones_the_images_build_from():
    assert {image: IMAGE_DOCKERFILES[image] for image in ("runtime", "dashboard")} == {
        "runtime": RUNTIME_DOCKERFILE.relative_to(REPO_ROOT).as_posix(),
        "dashboard": DASHBOARD_DOCKERFILE.relative_to(REPO_ROOT).as_posix(),
    }


def test_the_runtime_image_sets_the_bundled_cost_map():
    assert _final_stage_env(RUNTIME_DOCKERFILE)[VARIABLE] == "True"


def test_the_dashboard_image_sets_the_bundled_cost_map():
    assert _final_stage_env(DASHBOARD_DOCKERFILE)[VARIABLE] == "True"


@pytest.mark.parametrize("backend", ["cpu", "cuda", "rocm"])
def test_every_cogniverse_workload_runs_an_image_that_sets_it(backend):
    containers = _rendered_cogniverse_containers(
        f"runtime.backend={backend}", f"dashboard.backend={backend}"
    )
    owners = {owner for owner, _ in containers}

    assert {
        "Deployment/cogniverse-runtime",
        "Deployment/cogniverse-ingestor",
        "Deployment/cogniverse-quality-monitor",
        "Deployment/cogniverse-dashboard",
    } <= owners
    # The chart leaves the image's value in force: no container sets it.
    assert [
        (owner, entry)
        for owner, container in containers
        for entry in container.get("env") or []
        if entry.get("name") == VARIABLE
    ] == []
