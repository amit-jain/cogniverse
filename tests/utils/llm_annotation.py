"""Recording routing decisions and running ``llm-annotate`` against them."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pandas as pd

import cogniverse_foundation.telemetry.manager as telemetry_manager_module
from cogniverse_agents.routing.annotation_storage import AnnotationStorage
from cogniverse_agents.routing.llm_auto_annotator import AnnotationLabel
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.config import BatchExportConfig, TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.registry import get_telemetry_registry
from cogniverse_runtime import optimization_cli
from tests.utils.telemetry_metric_spans import record_routing

REPO_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "config.json"
# The labels the annotation prompt asks the LM for.
PROMPT_LABELS = {
    AnnotationLabel.CORRECT_ROUTING.value,
    AnnotationLabel.WRONG_ROUTING.value,
    AnnotationLabel.AMBIGUOUS.value,
    AnnotationLabel.INSUFFICIENT_INFO.value,
}


def use_annotation_config(tmp_path, monkeypatch, *, batch, annotator=None):
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


def new_tenant(prefix):
    return canonical_tenant_id(f"{prefix}{uuid4().hex[:8]}")


def record_decisions(telemetry, tenant):
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


async def read_labels(tenant, spans, *, until=None, timeout=60.0):
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


async def review_one(tenant, spans):
    await AnnotationStorage(tenant_id=tenant).store_human_annotation(
        span_id=spans["reviewed"],
        label=AnnotationLabel.CORRECT,
        reasoning="Right agent.",
        annotator_id="dana",
    )
    return await read_labels(tenant, spans, until=lambda labels: "reviewed" in labels)


async def wait_for_decisions(tenant, expected, timeout=90.0):
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


def run_annotation(tenant):
    return optimization_cli.run_llm_annotation(tenant_id=tenant, lookback_hours=1)


REVIEWED = {"label": "correct", "annotator": "dana", "human_reviewed": True}


def llm_labelled(labels, names):
    return {name for name in names if labels.get(name, {}).get("annotator") == "llm"}


@contextmanager
def annotation_telemetry(
    phoenix_container, proxy_url: str
) -> Iterator[TelemetryManager]:
    """The process telemetry manager, exporting to ``phoenix_container`` and
    reading through ``proxy_url``."""
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()
    manager = TelemetryManager(
        config=TelemetryConfig(
            otlp_endpoint=phoenix_container["otlp_endpoint"],
            provider_config={
                "http_endpoint": proxy_url,
                "grpc_endpoint": phoenix_container["grpc_endpoint"],
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
    )
    telemetry_manager_module._telemetry_manager = manager
    try:
        yield manager
    finally:
        TelemetryManager.reset()
        get_telemetry_registry().clear_cache()
