"""``ci_fast`` CI selections must not reach a fixture that starts a model server.

The fast CI runner has 7 GB of RAM; a vLLM sidecar's weights do not fit, so a
``ci_fast`` test that resolves one fails in CI while passing on a dev host.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

from tests.fixtures.ci_workflows import gating_selections, load_workflows

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# Fixtures that start a model server. Every fixture that provisions one does so
# through one of these, and pytest resolves the whole closure, so naming the
# roots catches a test that reaches them through any chain of intermediates.
MODEL_SPAWNING_FIXTURES = frozenset({"vllm_sidecar", "ensure_host_ollama"})

_PLUGIN = """
import json, pytest

@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items):
    roots = {roots!r}
    offenders = [
        [item.nodeid, sorted(set(getattr(item, "fixturenames", ())) & roots)]
        for item in items
        if set(getattr(item, "fixturenames", ())) & roots
    ]
    print("CI_FAST_OFFENDERS:" + json.dumps(offenders))
"""


def ci_fast_selections() -> dict[str, list[str]]:
    """Map each CI marker expression naming ``ci_fast`` to the paths it runs.

    Derived from the workflows so a new or renamed selection is covered without
    editing this test.
    """
    by_expr: dict[str, set[str]] = defaultdict(set)
    for selection in gating_selections(load_workflows(WORKFLOWS)):
        if selection.marker_expr and "ci_fast" in selection.marker_expr:
            by_expr[selection.marker_expr].update(selection.paths)
    return {expr: sorted(paths) for expr, paths in by_expr.items()}


def collect_offenders(
    paths: list[str], marker_expr: str, rootdir: Path = REPO_ROOT
) -> list[list]:
    """Return ``[nodeid, reached_fixtures]`` for a selection, post-deselection."""
    with tempfile.TemporaryDirectory(prefix="ci_fast_probe_") as plugin_dir:
        (Path(plugin_dir) / "_ci_fast_probe_plugin.py").write_text(
            _PLUGIN.format(roots=set(MODEL_SPAWNING_FIXTURES))
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (plugin_dir, env.get("PYTHONPATH", "")) if part
        )
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *paths,
                "-m",
                marker_expr,
                "--collect-only",
                "-q",
                "-p",
                "_ci_fast_probe_plugin",
                "-p",
                "no:cacheprovider",
            ],
            cwd=rootdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
    marker = "CI_FAST_OFFENDERS:"
    for line in result.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    raise AssertionError(
        f"collection reported nothing for {marker_expr!r} over {paths} "
        f"(rc={result.returncode})\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )


def test_ci_fast_selections_are_derived_from_the_workflows() -> None:
    """The guard covers every ``ci_fast`` selection CI actually runs."""
    selections = ci_fast_selections()
    assert set(selections) == {
        "ci_fast",
        "unit and ci_fast",
        "integration and ci_fast and not requires_lm",
    }
    assert selections["ci_fast"] == ["tests/agents/integration"]
    assert selections["unit and ci_fast"] == ["tests/memory/unit"]


def test_the_detector_reports_a_ci_fast_test_that_reaches_a_spawner(
    tmp_path: Path,
) -> None:
    """Synthetic offender: proves the detector fires rather than always passing.

    A repo-wide "no offenders remain" assertion cannot protect its own detector.
    """
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    ci_fast: fast\n    integration: integration\n"
    )
    (tmp_path / "conftest.py").write_text(
        "import pytest\n\n\n@pytest.fixture\ndef vllm_sidecar():\n    return object()\n"
        "\n\n@pytest.fixture\ndef indirect(vllm_sidecar):\n    return vllm_sidecar\n"
    )
    (tmp_path / "test_synthetic.py").write_text(
        "import pytest\n\n\n"
        "@pytest.mark.integration\n@pytest.mark.ci_fast\n"
        "def test_reaches_spawner_indirectly(indirect):\n    pass\n\n\n"
        "@pytest.mark.integration\n@pytest.mark.ci_fast\n"
        "def test_clean():\n    pass\n"
    )
    offenders = collect_offenders(["."], "integration and ci_fast", rootdir=tmp_path)
    assert offenders == [
        ["test_synthetic.py::test_reaches_spawner_indirectly", ["vllm_sidecar"]]
    ]


@pytest.mark.parametrize("marker_expr", sorted(ci_fast_selections()))
def test_no_ci_fast_selection_provisions_a_model_server(marker_expr: str) -> None:
    assert collect_offenders(ci_fast_selections()[marker_expr], marker_expr) == []


def test_the_probe_never_writes_into_the_tree_it_collects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe file inside the checkout makes the tree dirty for the seconds it
    exists, which the e2e fixture and the push gate refuse; the plugin must be
    importable from outside ``rootdir``."""
    rootdir = tmp_path / "repo"
    rootdir.mkdir()
    seen: dict[str, object] = {}

    def observe(argv, **kwargs):
        seen["entries"] = sorted(p.name for p in rootdir.iterdir())
        env = kwargs["env"]
        seen["importable"] = any(
            (Path(part) / "_ci_fast_probe_plugin.py").is_file()
            for part in env["PYTHONPATH"].split(os.pathsep)
        )
        return subprocess.CompletedProcess(argv, 0, "CI_FAST_OFFENDERS:[]\n", "")

    monkeypatch.setattr(subprocess, "run", observe)
    assert collect_offenders(["tests"], "ci_fast", rootdir=rootdir) == []
    assert seen == {"entries": [], "importable": True}
