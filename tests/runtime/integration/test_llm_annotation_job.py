"""The ``llm-annotate`` optimization mode against real Phoenix, the real config
store and the real annotation LM.

Decisions are recorded with the output slot ``GatewayAgent._emit_routing_span``
writes. Phoenix is read and written through a forwarding proxy, so a test can
hold or fail its requests.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path
from uuid import uuid4

import httpx
import litellm
import pandas as pd
import pytest

import cogniverse_foundation.telemetry.manager as telemetry_manager_module
from cogniverse_agents.routing.annotation_storage import AnnotationStorage
from cogniverse_agents.routing.llm_auto_annotator import (
    AnnotationLabel,
)
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.config import BatchExportConfig, TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import get_telemetry_registry
from cogniverse_runtime import optimization_cli
from cogniverse_runtime.routers import tenant as tenant_router
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.telemetry_metric_spans import record_routing
from tests.utils.web_client import free_port

pytestmark = [pytest.mark.integration]

REPO_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "config.json"
# The labels the annotation prompt asks the LM for.
PROMPT_LABELS = {
    AnnotationLabel.CORRECT_ROUTING.value,
    AnnotationLabel.WRONG_ROUTING.value,
    AnnotationLabel.AMBIGUOUS.value,
    AnnotationLabel.INSUFFICIENT_INFO.value,
}


@pytest.fixture(scope="module")
def phoenix_proxy(phoenix_container):
    with InterceptFaultProxy(phoenix_container["http_endpoint"]) as proxy:
        yield proxy


@pytest.fixture(scope="module")
def telemetry(phoenix_container, phoenix_proxy):
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()
    manager = TelemetryManager(
        config=TelemetryConfig(
            otlp_endpoint=phoenix_container["otlp_endpoint"],
            provider_config={
                "http_endpoint": phoenix_proxy.url,
                "grpc_endpoint": phoenix_container["grpc_endpoint"],
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
    )
    telemetry_manager_module._telemetry_manager = manager
    yield manager
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()


@pytest.fixture(autouse=True)
def _forward_everything(phoenix_proxy):
    yield
    phoenix_proxy.intercept = None


def _use_config(tmp_path, monkeypatch, *, batch, annotator=None):
    """The session's config with a batch cap of ``batch`` and, when given,
    ``annotator`` as the ``llm_auto_annotator`` endpoint override."""
    config = json.loads(
        Path(os.environ.get("COGNIVERSE_CONFIG") or REPO_CONFIG).read_text()
    )
    config["automation_rules"]["annotation_thresholds"]["max_annotations_per_batch"] = (
        batch
    )
    if annotator is not None:
        config["llm_config"]["overrides"]["llm_auto_annotator"] = annotator
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(path))


def _tenant(prefix):
    return canonical_tenant_id(f"{prefix}{uuid4().hex[:8]}")


def _record(telemetry, tenant):
    """One decision of each kind, oldest first: confident (not flagged), very
    low confidence, failed, near the boundary, and one a reviewer labelled."""
    spans = {
        name: record_routing(
            telemetry, tenant, agent, confidence, 50, minutes_ago=age, failed=failed
        )
        for name, agent, confidence, failed, age in [
            ("confident", "search_agent", 0.9, False, 10),
            ("very_low", "search_agent", 0.2, False, 9),
            ("failed", "summarizer_agent", 0.5, True, 8),
            ("boundary", "search_agent", 0.65, False, 7),
            ("reviewed", "summarizer_agent", 0.4, False, 6),
        ]
    }
    telemetry.force_flush(timeout_millis=10000)
    return spans


async def _labels(tenant, spans, *, until=None, timeout=60.0):
    """The ``routing_annotation`` of each recorded decision, by name; waits
    until ``until`` holds when given."""
    storage = AnnotationStorage(tenant_id=tenant)
    by_id = {span_id: name for name, span_id in spans.items()}
    deadline = time.monotonic() + timeout
    while True:
        frame = await storage.provider.annotations.get_annotations(
            spans_df=pd.DataFrame({"context.span_id": list(spans.values())}),
            project=storage.project_name,
            annotation_names=[storage.annotation_name],
        )
        if "updated_at" in frame.columns:
            frame = frame.sort_values("updated_at")
        labels = {
            by_id[span_id]: {
                "label": row["result.label"],
                "annotator": row["metadata"].get("annotator"),
                "human_reviewed": row["metadata"].get("human_reviewed"),
            }
            for span_id, row in frame.iterrows()
        }
        if until is None or until(labels) or time.monotonic() > deadline:
            return labels
        await asyncio.sleep(1)


async def _review(tenant, spans):
    await AnnotationStorage(tenant_id=tenant).store_human_annotation(
        span_id=spans["reviewed"],
        label=AnnotationLabel.CORRECT,
        reasoning="Right agent.",
        annotator_id="dana",
    )
    return await _labels(tenant, spans, until=lambda labels: "reviewed" in labels)


async def _routed(tenant, expected, timeout=90.0):
    """Wait until Phoenix serves ``expected`` routing decisions of the tenant."""
    storage = AnnotationStorage(tenant_id=tenant)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        end = pd.Timestamp.now(tz="UTC")
        spans = await storage.provider.traces.get_all_spans(
            project=storage.project_name,
            start_time=end - pd.Timedelta(hours=1),
            end_time=end,
            filters={"name": "cogniverse.routing"},
        )
        if len(spans) >= expected:
            return
        await asyncio.sleep(2)
    raise AssertionError(f"Phoenix never served {expected} decisions of {tenant}")


def _run(tenant):
    return optimization_cli.run_llm_annotation(tenant_id=tenant, lookback_hours=1)


REVIEWED = {"label": "correct", "annotator": "dana", "human_reviewed": True}


def _llm_labelled(labels, names):
    return {name for name in names if labels.get(name, {}).get("annotator") == "llm"}


def test_llm_annotate_is_a_manual_optimization_mode():
    assert "llm-annotate" in tenant_router._MANUAL_OPTIMIZE_MODES
    parser = optimization_cli.build_parser()
    assert parser.parse_args(["--mode", "llm-annotate"]).mode == "llm-annotate"


@pytest.mark.requires_lm
async def test_flagged_unlabelled_decisions_are_labelled_by_the_llm_in_batches(
    telemetry, tmp_path, monkeypatch
):
    _use_config(tmp_path, monkeypatch, batch=2)
    tenant = _tenant("llmannotate")
    spans = _record(telemetry, tenant)
    await _routed(tenant, 5)
    await _review(tenant, spans)

    first = await _run(tenant)
    after_first = await _labels(
        tenant, spans, until=lambda labels: len(_llm_labelled(labels, spans)) == 2
    )
    second = await _run(tenant)
    after_second = await _labels(
        tenant, spans, until=lambda labels: len(_llm_labelled(labels, spans)) == 3
    )

    # The two oldest high-priority decisions go first; the boundary decision
    # waits for the next run, the reviewed one is never sent.
    assert {k: first[k] for k in first if k != "labels"} == {
        "status": "success",
        "needing_review": 4,
        "already_labelled": 1,
        "labelled": 2,
        "deferred": 1,
    }
    assert sum(first["labels"].values()) == 2
    assert set(first["labels"]) <= PROMPT_LABELS
    assert _llm_labelled(after_first, spans) == {"very_low", "failed"}
    assert {k: second[k] for k in second if k != "labels"} == {
        "status": "success",
        "needing_review": 4,
        "already_labelled": 3,
        "labelled": 1,
        "deferred": 0,
    }
    assert _llm_labelled(after_second, spans) == {"very_low", "failed", "boundary"}
    assert after_second["reviewed"] == REVIEWED
    assert "confident" not in after_second
    for name in ("very_low", "failed", "boundary"):
        assert after_second[name]["label"] in PROMPT_LABELS
        assert after_second[name]["human_reviewed"] is False
    assert after_second["very_low"] == after_first["very_low"]


async def test_an_unreachable_annotation_lm_fails_the_run_and_stores_nothing(
    telemetry, tmp_path, monkeypatch
):
    dead = f"http://127.0.0.1:{free_port()}/v1"
    _use_config(tmp_path, monkeypatch, batch=10, annotator={"api_base": dead})
    tenant = _tenant("llmannotatedown")
    spans = _record(telemetry, tenant)
    await _routed(tenant, 5)
    await _review(tenant, spans)

    with pytest.raises(litellm.exceptions.InternalServerError) as raised:
        await _run(tenant)

    assert "Connection error" in str(raised.value)
    assert await _labels(tenant, spans) == {"reviewed": REVIEWED}


async def test_an_unreadable_telemetry_backend_fails_the_run_not_as_no_data(
    telemetry, tmp_path, monkeypatch, phoenix_proxy
):
    dead = f"http://127.0.0.1:{free_port()}/v1"
    _use_config(tmp_path, monkeypatch, batch=10, annotator={"api_base": dead})
    tenant = _tenant("llmannotateunread")
    spans = _record(telemetry, tenant)
    await _routed(tenant, 5)

    def fail_span_reads(method, path, body):
        return (503, {"detail": "down"}) if "/spans" in path else None

    phoenix_proxy.intercept = fail_span_reads
    with pytest.raises(httpx.HTTPStatusError) as raised:
        await _run(tenant)
    phoenix_proxy.intercept = None

    assert raised.value.response.status_code == 503
    assert await _labels(tenant, spans) == {}


@pytest.mark.requires_lm
async def test_a_failed_label_write_fails_the_run_and_a_rerun_labels(
    telemetry, tmp_path, monkeypatch, phoenix_proxy
):
    _use_config(tmp_path, monkeypatch, batch=2)
    tenant = _tenant("llmannotatewrite")
    spans = _record(telemetry, tenant)
    await _routed(tenant, 5)
    await _review(tenant, spans)

    def fail_annotation_writes(method, path, body):
        if method == "POST" and "span_annotations" in path:
            return (503, {"detail": "down"})
        return None

    phoenix_proxy.intercept = fail_annotation_writes
    with pytest.raises(Exception) as raised:
        await _run(tenant)
    phoenix_proxy.intercept = None
    unlabelled = await _labels(tenant, spans)
    rerun = await _run(tenant)
    labelled = await _labels(
        tenant, spans, until=lambda labels: len(_llm_labelled(labels, spans)) == 2
    )

    assert "503" in str(raised.value)
    assert unlabelled == {"reviewed": REVIEWED}
    assert (rerun["already_labelled"], rerun["labelled"], rerun["deferred"]) == (
        1,
        2,
        1,
    )
    assert _llm_labelled(labelled, spans) == {"very_low", "failed"}


@pytest.mark.requires_lm
async def test_concurrent_runs_label_only_their_own_tenant(
    telemetry, tmp_path, monkeypatch, phoenix_proxy
):
    _use_config(tmp_path, monkeypatch, batch=10)
    tenants = [_tenant("llmannotateone"), _tenant("llmannotatetwo")]
    spans = {tenant: _record(telemetry, tenant) for tenant in tenants}
    for tenant in tenants:
        await _routed(tenant, 5)
    barrier = threading.Barrier(len(tenants), timeout=60)
    held = []

    def hold_first_reads(method, path, body):
        # Both runs have asked Phoenix for their decisions before either
        # is answered.
        if "/spans" in path and len(held) < len(tenants):
            held.append(path)
            barrier.wait()
        return None

    phoenix_proxy.intercept = hold_first_reads
    results = await asyncio.gather(*(_run(tenant) for tenant in tenants))
    phoenix_proxy.intercept = None
    labels = [
        await _labels(
            tenant,
            spans[tenant],
            until=lambda found, own=spans[tenant]: len(_llm_labelled(found, own)) == 4,
        )
        for tenant in tenants
    ]

    assert len(held) == len(tenants)
    assert [(r["needing_review"], r["labelled"]) for r in results] == [(4, 4), (4, 4)]
    for tenant, found in zip(tenants, labels):
        assert _llm_labelled(found, spans[tenant]) == {
            "very_low",
            "failed",
            "boundary",
            "reviewed",
        }
        assert "confident" not in found
