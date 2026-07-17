"""The annotation-feedback cycle turns accumulated human labels into recompiles.

``run_annotation_feedback_cycle`` counts human annotations per agent type over
the lookback window and, when the volume gates pass and the per-agent cooldown
has elapsed, submits the agent's compile workflow to Argo:

- search/summary/report → a scored trigger dataset (labels → ``quality_map``)
  plus a ``--mode triggered`` workflow;
- gateway/routing → a ``--mode gateway-thresholds`` refresh at the cheaper
  ``min_annotations_for_update`` gate;
- query_enhancement/entity_extraction/profile_selection → their dedicated
  compile modes.

State (last poll, per-agent last optimization) lives in the config store so
the stateless cron honors ``poll_interval_minutes`` and
``min_days_between_optimizations`` across runs.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from cogniverse_agents.routing.config import (
    AutomationRulesConfig,
    FeedbackConfig,
    OptimizationTriggersConfig,
)
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.quality_monitor_cli import run_annotation_feedback_cycle
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

NOW = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)


def _rules(**triggers):
    return AutomationRulesConfig(
        optimization_triggers=OptimizationTriggersConfig(
            min_annotations_for_optimization=50,
            min_days_between_optimizations=1,
            annotation_lookback_hours=24,
            **triggers,
        ),
        feedback=FeedbackConfig(min_annotations_for_update=10),
    )


def _config_manager():
    store = InMemoryConfigStore()
    store.initialize()
    return ConfigManager(store=store)


def _annotated_rows(n, label="correct"):
    return [
        {
            "span_id": f"s{i}",
            "query": f"query {i}",
            "annotation_label": label if i % 2 == 0 else "wrong",
            "output": {"answer": f"a{i}"},
            "human_reviewed": True,
        }
        for i in range(n)
    ]


class _StorageStub:
    """Per-agent annotated-row counts for the cycle to read."""

    rows_by_type: dict = {}

    def __init__(self, tenant_id, agent_type="routing"):
        self.agent_type = agent_type

    async def query_annotated_spans(self, start_time, end_time, only_human_reviewed):
        return self.rows_by_type.get(self.agent_type, [])


class _ArgoCapture:
    def __init__(self):
        self.posts = []

    async def post(self, url, json=None):
        self.posts.append((url, json))

        class _Resp:
            status_code = 201

            @staticmethod
            def json():
                return {"metadata": {"name": "wf-1"}}

            text = ""

        return _Resp()


class _DatasetStoreStub:
    def __init__(self):
        self.created = []

    async def create_dataset(self, name, data, metadata=None):
        self.created.append((name, data, metadata))


async def _run(
    rules, rows_by_type, config_manager=None, now=NOW, force=False, pod_spec=None
):
    _StorageStub.rows_by_type = rows_by_type
    argo = _ArgoCapture()
    datasets = _DatasetStoreStub()
    with patch(
        "cogniverse_agents.routing.annotation_storage.AnnotationStorage",
        _StorageStub,
    ):
        result = await run_annotation_feedback_cycle(
            tenant_id="acme:acme",
            argo_url="http://argo:2746",
            argo_namespace="cogniverse",
            automation_rules=rules,
            config_manager=config_manager or _config_manager(),
            http_client=argo,
            dataset_store=datasets,
            now=now,
            force=force,
            pod_spec=pod_spec,
        )
    return result, argo, datasets


@pytest.mark.asyncio
async def test_search_volume_gate_builds_dataset_and_submits_triggered():
    result, argo, datasets = await _run(_rules(), {"search": _annotated_rows(50)})

    assert result["agents"]["search"]["action"] == "recompile"
    assert result["agents"]["search"]["annotations"] == 50

    # A scored trigger dataset was created with quality_map scores.
    assert len(datasets.created) == 1
    name, df, metadata = datasets.created[0]
    assert name.startswith("optimization-trigger-acme:acme-")
    records = df.to_dict("records")
    assert len(records) == 50
    by_cat = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)
        assert r["agent"] == "search"
    # correct → 0.9 (high_scoring), wrong → 0.3 (low_scoring)
    assert all(r["score"] == 0.9 for r in by_cat["high_scoring"])
    assert all(r["score"] == 0.3 for r in by_cat["low_scoring"])

    # One triggered-mode workflow, carrying the dataset name.
    assert len(argo.posts) == 1
    url, body = argo.posts[0]
    assert url == "http://argo:2746/api/v1/workflows/cogniverse"
    args = body["workflow"]["spec"]["templates"][0]["container"]["args"]
    assert "--mode" in args and "triggered" in args
    params = {
        p["name"]: p["value"]
        for p in body["workflow"]["spec"]["arguments"]["parameters"]
    }
    assert params["agents"] == "search"
    assert params["trigger-dataset"] == name


@pytest.mark.asyncio
async def test_routing_update_gate_submits_gateway_thresholds():
    result, argo, datasets = await _run(_rules(), {"routing": _annotated_rows(12)})

    assert result["agents"]["routing"]["action"] == "thresholds_refresh"
    assert datasets.created == []
    assert len(argo.posts) == 1
    args = argo.posts[0][1]["workflow"]["spec"]["templates"][0]["container"]["args"]
    assert "gateway-thresholds" in args


@pytest.mark.asyncio
async def test_dedicated_mode_for_profile_selection():
    result, argo, datasets = await _run(
        _rules(), {"profile_selection": _annotated_rows(50)}
    )

    assert result["agents"]["profile_selection"]["action"] == "recompile"
    assert datasets.created == []  # dedicated modes train from spans
    args = argo.posts[0][1]["workflow"]["spec"]["templates"][0]["container"]["args"]
    assert "profile" in args


@pytest.mark.asyncio
async def test_below_gates_takes_no_action():
    result, argo, datasets = await _run(_rules(), {"search": _annotated_rows(5)})

    assert result["agents"]["search"]["action"] == "none"
    assert argo.posts == []
    assert datasets.created == []


@pytest.mark.asyncio
async def test_cooldown_blocks_resubmission():
    cm = _config_manager()

    # First run submits and records last_optimization_at.
    result1, argo1, _ = await _run(
        _rules(), {"search": _annotated_rows(50)}, config_manager=cm
    )
    assert len(argo1.posts) == 1

    # Second run 2 hours later (cooldown = 1 day) must not resubmit.
    result2, argo2, _ = await _run(
        _rules(),
        {"search": _annotated_rows(50)},
        config_manager=cm,
        now=NOW + timedelta(hours=2),
        force=True,  # bypass the poll gate; the cooldown must hold on its own
    )
    assert result2["agents"]["search"]["action"] == "cooldown"
    assert argo2.posts == []

    # After the cooldown elapses, it submits again.
    result3, argo3, _ = await _run(
        _rules(),
        {"search": _annotated_rows(50)},
        config_manager=cm,
        now=NOW + timedelta(days=1, hours=1),
        force=True,
    )
    assert result3["agents"]["search"]["action"] == "recompile"
    assert len(argo3.posts) == 1


@pytest.mark.asyncio
async def test_poll_interval_self_gate():
    cm = _config_manager()
    await _run(_rules(), {}, config_manager=cm)

    # A second run 3 minutes later (poll interval 15m) skips entirely.
    result, argo, _ = await _run(
        _rules(),
        {"search": _annotated_rows(50)},
        config_manager=cm,
        now=NOW + timedelta(minutes=3),
    )
    assert result["status"] == "skipped_recent_poll"
    assert argo.posts == []


@pytest.mark.asyncio
async def test_bare_tenant_id_is_canonicalized():
    """A bare tenant id must resolve to the canonical form everywhere the
    cycle touches — Argo parameters and the persisted loop state — so the
    cron, the runtime, and the compile all agree on one tenant project."""
    from cogniverse_runtime.quality_monitor_cli import _load_loop_state

    cm = _config_manager()
    _StorageStub.rows_by_type = {"routing": _annotated_rows(12)}
    argo = _ArgoCapture()
    with patch(
        "cogniverse_agents.routing.annotation_storage.AnnotationStorage",
        _StorageStub,
    ):
        result = await run_annotation_feedback_cycle(
            tenant_id="acme",
            argo_url="http://argo:2746",
            argo_namespace="cogniverse",
            automation_rules=_rules(),
            config_manager=cm,
            http_client=argo,
            dataset_store=_DatasetStoreStub(),
            now=NOW,
        )

    assert result["agents"]["routing"]["action"] == "thresholds_refresh"
    params = {
        p["name"]: p["value"]
        for p in argo.posts[0][1]["workflow"]["spec"]["arguments"]["parameters"]
    }
    assert params["tenant-id"] == "acme:acme"
    state = _load_loop_state(cm, "acme:acme")
    assert "routing" in state.get("last_optimization_at", {}), state


@pytest.mark.asyncio
async def test_pod_spec_reaches_spawned_manifest():
    from cogniverse_evaluation.quality_monitor import OptimizationWorkflowPodSpec

    pod_spec = OptimizationWorkflowPodSpec(
        image="cogniverse/runtime-rocm:dev",
        env={"BACKEND_URL": "http://cogniverse-vespa"},
        config_map="cogniverse-config",
        dev_source_hostpath="/cogniverse-src",
    )
    _, argo, _ = await _run(
        _rules(), {"routing": _annotated_rows(12)}, pod_spec=pod_spec
    )

    assert len(argo.posts) == 1
    template = argo.posts[0][1]["workflow"]["spec"]["templates"][0]
    assert template["container"]["image"] == "cogniverse/runtime-rocm:dev"
    assert template["container"]["env"] == [
        {"name": "BACKEND_URL", "value": "http://cogniverse-vespa"}
    ]
    assert {v["name"] for v in template["volumes"]} == {
        "config",
        "src-libs",
        "src-scripts",
    }


def test_workflow_pod_spec_from_env(monkeypatch):
    from cogniverse_runtime.quality_monitor_cli import _workflow_pod_spec_from_env

    monkeypatch.delenv("OPTIMIZATION_WORKFLOW_IMAGE", raising=False)
    assert _workflow_pod_spec_from_env() is None

    monkeypatch.setenv("OPTIMIZATION_WORKFLOW_IMAGE", "cogniverse/runtime-rocm:dev")
    monkeypatch.setenv("OPTIMIZATION_CONFIG_MAP", "cogniverse-config")
    monkeypatch.setenv("OPTIMIZATION_DEV_HOSTPATH", "/cogniverse-src")
    monkeypatch.setenv("BACKEND_URL", "http://cogniverse-vespa")
    monkeypatch.setenv("BACKEND_PORT", "8080")
    monkeypatch.delenv("TELEMETRY_HTTP_ENDPOINT", raising=False)
    monkeypatch.delenv("TELEMETRY_OTLP_ENDPOINT", raising=False)

    spec = _workflow_pod_spec_from_env()
    assert spec.image == "cogniverse/runtime-rocm:dev"
    assert spec.config_map == "cogniverse-config"
    assert spec.dev_source_hostpath == "/cogniverse-src"
    assert spec.env == {
        "BACKEND_URL": "http://cogniverse-vespa",
        "BACKEND_PORT": "8080",
    }
