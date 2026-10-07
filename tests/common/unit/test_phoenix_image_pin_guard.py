"""Every local and CI Phoenix run uses the image the chart pins.

Test fixtures, CI service containers and the start scripts each name the
Phoenix image. One that stays behind on an upgrade tests a server the cluster
no longer runs, so each reference must be the chart's ``repository:tag@digest``.
"""

from __future__ import annotations

import pathlib
import re

import yaml

REPO = pathlib.Path(__file__).resolve().parents[3]
CHART_VALUES = REPO / "charts" / "cogniverse" / "values.yaml"
SCANNED = (
    (REPO / "tests", "*.py"),
    (REPO / "scripts", "*.py"),
    (REPO / "scripts", "*.sh"),
    (REPO / ".github" / "workflows", "*.yml"),
)
NOT_A_RUN = {
    # Unit tests of the image-pull helper name a placeholder tag.
    "tests/cli/unit/test_images.py",
    # Asserts the rendered chart, including the tag-only fallback.
    "tests/charts/test_phoenix_server_chart.py",
}

_PHOENIX_IMAGE_REF = re.compile(r"arizephoenix/phoenix:[\w.-]+(?:@sha256:[0-9a-f]+)?")


def _chart_image() -> str:
    image = yaml.safe_load(CHART_VALUES.read_text(encoding="utf-8"))["phoenix"]["image"]
    return f"{image['repository']}:{image['tag']}@{image['digest']}"


def test_every_local_and_ci_phoenix_run_uses_the_chart_image():
    refs: dict[str, list[str]] = {}
    for base, pattern in SCANNED:
        for path in base.rglob(pattern):
            relative = path.relative_to(REPO).as_posix()
            if relative in NOT_A_RUN or path == pathlib.Path(__file__):
                continue
            for ref in _PHOENIX_IMAGE_REF.findall(path.read_text(encoding="utf-8")):
                refs.setdefault(ref, []).append(relative)

    assert {ref: sorted(files) for ref, files in refs.items()} == {
        _chart_image(): [
            ".github/workflows/evaluation-tests.yml",
            ".github/workflows/evaluation-tests.yml",
            ".github/workflows/telemetry-tests.yml",
            ".github/workflows/telemetry-tests.yml",
            "scripts/start_phoenix.py",
            "scripts/start_phoenix.sh",
            "scripts/start_phoenix.sh",
            "tests/conftest.py",
            "tests/evaluation/conftest.py",
            "tests/runtime/integration/test_artefact_store_outage_contract.py",
        ]
    }


_NO_EXTERNAL = "PHOENIX_ALLOW_EXTERNAL_RESOURCES=false"


def _spawns_reaching_out(source: str) -> int:
    """How many Phoenix images ``source`` starts without
    ``PHOENIX_ALLOW_EXTERNAL_RESOURCES=false``. With external resources
    allowed, Phoenix connects to its docs host while starting, so a test
    server's startup depends on the internet."""
    return len(_PHOENIX_IMAGE_REF.findall(source)) - source.count(_NO_EXTERNAL)


def test_a_spawn_that_allows_external_resources_is_counted():
    image = f'"{_chart_image()}"'
    isolated = f'"-e", "{_NO_EXTERNAL}", {image}'
    assert [
        _spawns_reaching_out(source)
        for source in (isolated, image, f"{isolated}\n{image}", "")
    ] == [0, 1, 1, 0]


def test_every_test_phoenix_server_keeps_to_the_local_network():
    spawns = {
        path.relative_to(REPO).as_posix(): _spawns_reaching_out(
            path.read_text(encoding="utf-8")
        )
        for path in (REPO / "tests").rglob("*.py")
        if path.relative_to(REPO).as_posix() not in NOT_A_RUN
        and path != pathlib.Path(__file__)
        and _PHOENIX_IMAGE_REF.search(path.read_text(encoding="utf-8"))
    }
    assert spawns == {
        "tests/conftest.py": 0,
        "tests/evaluation/conftest.py": 0,
        "tests/runtime/integration/test_artefact_store_outage_contract.py": 0,
    }
