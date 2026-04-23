"""Container image build and import into k3d."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

RUNTIME_TAG = "cogniverse/runtime:dev"
DASHBOARD_TAG = "cogniverse/dashboard:dev"
PYLATE_TAG = "cogniverse/pylate:dev"


def has_workspace_source(project_root: Path) -> bool:
    """Check if workspace source is available for building images."""
    return (project_root / "libs" / "runtime").is_dir()


def build_images(project_root: Path) -> list[str]:
    """Build all cogniverse-owned Docker images.

    Returns list of image tags that were built. The pylate inference image is
    always built so ``--set inference.<name>.engine=pylate`` against an
    existing k3d cluster works without a separate build step.
    """
    workspace_builds = [
        (RUNTIME_TAG, "libs/runtime/Dockerfile", "."),
        (DASHBOARD_TAG, "libs/dashboard/Dockerfile", "."),
        (PYLATE_TAG, "deploy/pylate/Dockerfile", "deploy/pylate"),
    ]
    built: list[str] = []
    for tag, dockerfile, context in workspace_builds:
        subprocess.run(
            ["docker", "build", "-f", dockerfile, "-t", tag, context],
            cwd=str(project_root),
            check=True,
            timeout=3600,
        )
        built.append(tag)
    return built


def import_images(cluster_name: str, tags: list[str]) -> None:
    """Import Docker images into a k3d cluster."""
    subprocess.run(
        ["k3d", "image", "import", *tags, "-c", cluster_name],
        check=True,
        timeout=300,
    )


def _read_third_party_images(values_file: Path, skip_llm: bool = False) -> list[str]:
    """Read third-party image references from Helm values file."""
    with open(values_file) as f:
        values = yaml.safe_load(f) or {}

    images = []

    # Vespa
    vespa = values.get("vespa", {}).get("image", {})
    if vespa.get("repository"):
        images.append(f"{vespa['repository']}:{vespa.get('tag', 'latest')}")

    # Phoenix
    phoenix = values.get("phoenix", {}).get("image", {})
    if phoenix.get("repository"):
        images.append(f"{phoenix['repository']}:{phoenix.get('tag', 'latest')}")

    # LLM builtin (only if not using external)
    if not skip_llm:
        llm = values.get("llm", {}).get("builtin", {}).get("image", {})
        if llm.get("repository"):
            images.append(f"{llm['repository']}:{llm.get('tag', 'latest')}")

    return images


def pull_and_import_third_party(
    cluster_name: str,
    values_file: Path,
    *,
    skip_llm: bool = False,
) -> None:
    """Pull third-party images locally and import into k3d.

    Reads image references from the Helm values file rather than
    hardcoding them. This avoids slow in-cluster pulls that cause
    pod startup timeouts.
    """
    images = _read_third_party_images(values_file, skip_llm=skip_llm)
    if not images:
        return

    for image in images:
        subprocess.run(
            ["docker", "pull", image],
            check=False,  # Don't fail if pull is slow/offline
            timeout=600,
        )

    subprocess.run(
        ["k3d", "image", "import", *images, "-c", cluster_name],
        check=False,  # Non-fatal if import fails
        timeout=600,
    )
