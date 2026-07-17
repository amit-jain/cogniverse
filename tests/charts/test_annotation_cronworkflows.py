"""Chart wiring for the annotation-identification and feedback CronWorkflows.

Renders the chart and pins:
1. both CronWorkflows exist and invoke ``quality_monitor_cli`` with the right
   one-shot flag;
2. the cron schedules mirror the ``IntervalConfig`` defaults
   (``annotation_interval_minutes`` / ``feedback_interval_minutes``) — the
   config knob and the chart value must not drift apart silently.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CHART_PATH = REPO_ROOT / "charts" / "cogniverse"

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None,
    reason="helm CLI not installed — chart tests require helm",
)


def _render_chart() -> list:
    result = subprocess.run(
        [
            "helm",
            "template",
            "cogniverse",
            str(CHART_PATH),
            "--set",
            "runtime.qualityMonitor.tenantId=test-tenant",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"helm template failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )
    return [doc for doc in yaml.safe_load_all(result.stdout) if doc]


def _cronworkflow(manifests, name):
    for doc in manifests:
        if (
            doc.get("kind") == "CronWorkflow"
            and doc.get("metadata", {}).get("name") == name
        ):
            return doc
    raise AssertionError(f"CronWorkflow {name} not rendered")


def _container_args(cron):
    return cron["spec"]["workflowSpec"]["templates"][0]["container"]["args"]


def test_annotation_cycle_cronworkflow_invokes_the_flag():
    cron = _cronworkflow(_render_chart(), "cogniverse-annotation-cycle")
    args = _container_args(cron)
    assert "--annotation-cycle" in args
    assert "--runtime-url" in args
    assert cron["spec"]["concurrencyPolicy"] == "Forbid"


def test_annotation_feedback_cronworkflow_invokes_the_flag():
    cron = _cronworkflow(_render_chart(), "cogniverse-annotation-feedback")
    args = _container_args(cron)
    assert "--annotation-feedback" in args
    assert "--argo-url" in args


def test_schedules_mirror_interval_config_defaults():
    """The cron cadence and the IntervalConfig knob are one contract: a change
    to either without the other silently de-syncs the loop's documented
    behavior from its actual schedule."""
    from cogniverse_agents.routing.config import IntervalConfig

    intervals = IntervalConfig()
    manifests = _render_chart()

    cycle = _cronworkflow(manifests, "cogniverse-annotation-cycle")
    assert (
        cycle["spec"]["schedule"]
        == f"*/{intervals.annotation_interval_minutes} * * * *"
    )

    feedback = _cronworkflow(manifests, "cogniverse-annotation-feedback")
    assert (
        feedback["spec"]["schedule"]
        == f"*/{intervals.feedback_interval_minutes} * * * *"
    )
