"""Lint coverage for the Argo CronWorkflow manifests in ``workflows/``.

Each YAML must (a) parse as valid YAML, (b) declare the expected Argo
Kind, (c) carry a non-empty ``schedule`` that matches the documented
cron form, and (d) contain no left-in placeholder bash markers like the
former ``# Add actual Vespa backup command`` line.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS_DIR = Path(__file__).resolve().parents[3] / "workflows"

CRON_FIELD_RE = re.compile(
    # 5-field standard cron: minute hour dom month dow
    r"^\s*\S+\s+\S+\s+\S+\s+\S+\s+\S+\s*$"
)


def _list_workflow_files() -> list[Path]:
    return sorted(WORKFLOWS_DIR.glob("*.yaml"))


def test_workflow_dir_contains_at_least_one_manifest() -> None:
    files = _list_workflow_files()
    assert files, f"no YAML manifests in {WORKFLOWS_DIR}"


@pytest.mark.parametrize(
    "manifest_path",
    _list_workflow_files(),
    ids=lambda p: p.name,
)
def test_manifest_parses_as_yaml(manifest_path: Path) -> None:
    body = manifest_path.read_text()
    # ``yaml.safe_load_all`` because the file may contain multiple docs.
    docs = list(yaml.safe_load_all(body))
    assert docs, f"{manifest_path.name} contains no documents"
    for doc in docs:
        assert isinstance(doc, dict), (
            f"{manifest_path.name} top-level doc is not a mapping: {type(doc).__name__}"
        )


@pytest.mark.parametrize(
    "manifest_path",
    _list_workflow_files(),
    ids=lambda p: p.name,
)
def test_manifest_declares_known_argo_kind(manifest_path: Path) -> None:
    """Every manifest must be one of ``Workflow``, ``CronWorkflow``,
    or ``WorkflowTemplate``."""
    body = manifest_path.read_text()
    for doc in yaml.safe_load_all(body):
        if not isinstance(doc, dict):
            continue
        kind = doc.get("kind", "")
        assert kind in {"Workflow", "CronWorkflow", "WorkflowTemplate"}, (
            f"{manifest_path.name} declares unexpected kind={kind!r}"
        )


@pytest.mark.parametrize(
    "manifest_path",
    _list_workflow_files(),
    ids=lambda p: p.name,
)
def test_cronworkflow_carries_valid_schedule(manifest_path: Path) -> None:
    """Any ``CronWorkflow`` must have a 5-field cron expression."""
    body = manifest_path.read_text()
    for doc in yaml.safe_load_all(body):
        if not isinstance(doc, dict) or doc.get("kind") != "CronWorkflow":
            continue
        spec = doc.get("spec", {})
        schedule = spec.get("schedule")
        assert schedule, f"{manifest_path.name} CronWorkflow missing spec.schedule"
        assert CRON_FIELD_RE.match(schedule), (
            f"{manifest_path.name} schedule={schedule!r} is not a 5-field cron"
        )


@pytest.mark.parametrize(
    "manifest_path",
    _list_workflow_files(),
    ids=lambda p: p.name,
)
def test_no_left_in_todo_placeholders_in_embedded_bash(manifest_path: Path) -> None:
    """Catches the literal ``# Add actual <X> command`` placeholders that
    used to ship in scheduled-maintenance.yaml."""
    body = manifest_path.read_text()
    forbidden_markers = (
        "# Add actual",
        "# TODO: replace",
        "# FIXME",
        "# XXX placeholder",
    )
    offending = [m for m in forbidden_markers if m in body]
    assert not offending, (
        f"{manifest_path.name} carries placeholder markers {offending} — "
        "ship a real command or remove the section."
    )
