"""First-party image tags follow the release/dev scheme.

Base ``values.yaml`` ships the release tag ``<appVersion>`` with
``pullPolicy: IfNotPresent`` (a clean cluster pulls from the registry); the k3s
dev overlay swaps in the locally-built ``<appVersion>-dev`` tag with
``pullPolicy: Never`` (``cogniverse up`` builds + imports these). No first-party
image may carry the old floating ``:dev`` / ``:0.1.0`` tags, and the tag tracks
Chart.yaml ``appVersion`` so a version bump flows through everywhere.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"
APP_VERSION = str(yaml.safe_load((CHART_PATH / "Chart.yaml").read_text())["appVersion"])

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm CLI not installed — chart tests require helm",
)


def _render(*set_args: str, values: str | None = None) -> list[dict]:
    cmd = [
        "helm",
        "template",
        "cogniverse",
        str(CHART_PATH),
        # The chart fail-fasts if qualityMonitor.tenantId is empty (a value the
        # deploy overlay normally supplies); pin a placeholder so images are the
        # only variable under test.
        "--set",
        "runtime.qualityMonitor.tenantId=test-tenant",
    ]
    if values is not None:
        cmd.extend(["-f", str(CHART_PATH / values)])
    for arg in set_args:
        cmd.extend(["--set", arg])
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed (exit {result.returncode}):\n{result.stderr}"
        )
    return [d for d in yaml.safe_load_all(result.stdout) if d is not None]


def _first_party_images(docs: list[dict]) -> list[tuple[str, str | None]]:
    """Every (image, pullPolicy) for a first-party (``cogniverse/*``) container,
    walking the rendered manifests recursively — Deployments, StatefulSets,
    Jobs, and the Argo Workflow/CronWorkflow templates nest container specs at
    different depths."""
    found: list[tuple[str, str | None]] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            image = node.get("image")
            if isinstance(image, str) and image.startswith("cogniverse/"):
                found.append((image, node.get("imagePullPolicy")))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for doc in docs:
        walk(doc)
    return found


@pytest.mark.unit
class TestFirstPartyTagScheme:
    def test_base_ships_release_tag_and_pulls(self) -> None:
        images = _first_party_images(_render())
        assert images, "no first-party images rendered in the base chart"
        for image, policy in images:
            assert image.endswith(f":{APP_VERSION}"), (
                f"{image} is not the release tag :{APP_VERSION}"
            )
            assert policy == "IfNotPresent", (
                f"{image} pullPolicy {policy!r} != IfNotPresent (prod must pull)"
            )

    def test_k3s_overlay_uses_dev_tag_and_never_pulls(self) -> None:
        images = _first_party_images(_render(values="values.k3s.yaml"))
        assert images, "no first-party images rendered under values.k3s.yaml"
        for image, policy in images:
            assert image.endswith(f":{APP_VERSION}-dev"), (
                f"{image} is not the dev tag :{APP_VERSION}-dev"
            )
            assert policy == "Never", (
                f"{image} pullPolicy {policy!r} != Never (k3d uses local builds)"
            )

    def test_no_floating_dev_or_legacy_first_party_tags(self) -> None:
        for values in (None, "values.k3s.yaml"):
            images = _first_party_images(_render(values=values))
            stray = [i for i, _ in images if i.endswith((":dev", ":0.1.0"))]
            assert not stray, f"stray legacy first-party tags ({values}): {stray}"

    def test_face_embed_enable_path_matches_built_image(self) -> None:
        """Flipping the face_embed sidecar on renders the tag ``build_images``
        produces: the locally-built ``-dev`` image under k3s, the release image
        for a registry pull in base."""
        dev = _first_party_images(
            _render("inference.face_embed.enabled=true", values="values.k3s.yaml")
        )
        assert (f"cogniverse/face-embed:{APP_VERSION}-dev", "Never") in dev

        prod = _first_party_images(_render("inference.face_embed.enabled=true"))
        assert (f"cogniverse/face-embed:{APP_VERSION}", "IfNotPresent") in prod
