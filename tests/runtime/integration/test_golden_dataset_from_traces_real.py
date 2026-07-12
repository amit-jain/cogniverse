"""Real-Phoenix round-trip for the golden-dataset miner's score fetch.

fetch_traces_with_evaluations() used ``self.provider.evaluations`` — an
attribute the real PhoenixProvider does not have — so every run AttributeError'd,
swallowed it, and mined a score-less (empty) dataset. Scores actually live as
span annotations; this proves the annotation score is merged into the ``score``
column against a real Phoenix container.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = pytest.mark.integration

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "create_golden_dataset_from_traces.py"
)


def _load_generator_cls():
    spec = importlib.util.spec_from_file_location("golden_ds", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GoldenDatasetGenerator


@pytest.fixture
def telemetry_manager(phoenix_container):
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    config = TelemetryConfig(
        otlp_endpoint=phoenix_container["otlp_endpoint"],
        provider_config={
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["grpc_endpoint"],
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    manager = TelemetryManager(config)
    yield manager
    try:
        manager.shutdown()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_fetch_merges_annotation_score(phoenix_container, telemetry_manager):
    tenant = f"golden{uuid4().hex[:8]}"
    project = f"cogniverse-{tenant}"

    with telemetry_manager.span(
        name="search_agent",
        tenant_id=tenant,
        attributes={
            "input.value": json.dumps({"query": "blurry robot"}),
            "output.value": json.dumps({"results": [{"video_id": "vX"}]}),
        },
    ) as span:
        span_id = format(span.get_span_context().span_id, "016x")
    telemetry_manager.force_flush(timeout_millis=10000)

    provider = telemetry_manager.get_provider(tenant_id=tenant)

    found = False
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        end = datetime.now(timezone.utc)
        spans = await provider.traces.get_spans(
            project=project,
            start_time=end - timedelta(hours=1),
            end_time=end,
            limit=100,
        )
        if (
            spans is not None
            and not spans.empty
            and "context.span_id" in spans.columns
            and (spans["context.span_id"] == span_id).any()
        ):
            found = True
            break
        await asyncio.sleep(2)
    assert found, f"search span not indexed in {project}"

    await provider.annotations.add_annotation(
        span_id=span_id,
        name="relevance",
        label="poor",
        score=0.2,
        metadata={},
        project=project,
    )

    generator_cls = _load_generator_cls()
    gen = generator_cls.__new__(generator_cls)
    gen.tenant_id = tenant
    gen.hours_back = 1
    gen.provider = provider

    merged = None
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        merged = await gen.fetch_traces_with_evaluations()
        if (
            merged is not None
            and not merged.empty
            and "score" in merged.columns
            and merged.loc[merged["context.span_id"] == span_id, "score"].notna().any()
        ):
            break
        await asyncio.sleep(2)

    assert merged is not None and "score" in merged.columns
    row_score = merged.loc[merged["context.span_id"] == span_id, "score"]
    assert not row_score.empty
    assert float(row_score.iloc[0]) == pytest.approx(0.2)
