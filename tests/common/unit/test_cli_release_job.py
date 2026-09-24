"""The CLI workflow runs the release, clean-install and image-provisioning
files in their own job, with a test-step budget sized from their measured
duration."""

from __future__ import annotations

import math
from pathlib import Path

import yaml

from tests.fixtures.ci_workflows import load_workflows

WORKFLOWS_DIR = Path(__file__).resolve().parents[3] / ".github" / "workflows"

INSTALLING_FILES = (
    "tests/cli/integration/test_release_scripts.py",
    "tests/cli/integration/test_release_clean_install.py",
    "tests/cli/integration/test_image_model_provisioning.py",
)

# pytest wall time of the three files on a 32-thread host (1349 s clean
# install, 450 s release scripts, 56 s image provisioning), scaled for a
# 4-vCPU hosted runner.
MEASURED_HOST_SECONDS = 1855
RUNNER_SLOWDOWN = 3


def _ignored(path: str, ignores: tuple[str, ...]) -> bool:
    return any(path == ignored or path.startswith(ignored + "/") for ignored in ignores)


def test_the_installing_files_run_only_in_the_release_job():
    workflows = load_workflows(WORKFLOWS_DIR)

    running = {
        path: [
            (selection.workflow, selection.job)
            for workflow in workflows
            if workflow.commit_gating
            for selection in workflow.selections
            if selection.names(path) and not _ignored(path, selection.ignores)
        ]
        for path in INSTALLING_FILES
    }

    assert running == {
        path: [("cli-tests.yml", "release-tests")] for path in INSTALLING_FILES
    }


def test_the_release_job_budgets_the_measured_run_and_frees_disk_first():
    job = yaml.safe_load((WORKFLOWS_DIR / "cli-tests.yml").read_text())["jobs"][
        "release-tests"
    ]
    steps = {step.get("name"): step for step in job["steps"]}
    names = list(steps)
    test_step = steps["Run release, clean-install and image-provisioning tests"]

    assert names.index("Free up disk space") < names.index("Install dependencies")
    assert test_step["timeout-minutes"] >= math.ceil(
        MEASURED_HOST_SECONDS * RUNNER_SLOWDOWN / 60
    )
