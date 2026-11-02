"""Reviewed routing decisions become query classifier examples.

The span rows are the gateway's own ``cogniverse.routing`` spans, emitted by
a real GatewayAgent and laid out the way Phoenix returns them, so a change to
what the gateway records breaks these tests instead of the training set.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from cogniverse_agents.gateway_agent import (
    SIMPLE_ROUTE_MAP,
    GatewayAgent,
    GatewayDeps,
    GatewayInput,
)
from cogniverse_core.common.tenant_utils import TEST_TENANT_ID
from cogniverse_finetuning.dataset.gateway_labels import (
    QueryLabelExample,
    gateway_label_examples,
    suggested_agent_labels,
)
from tests.agents.unit._recording_telemetry import RecordingTelemetryManager

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


async def _routing_rows(queries: list[str]) -> pd.DataFrame:
    agent = GatewayAgent(deps=GatewayDeps())
    gliner = MagicMock()
    gliner.predict_entities.return_value = []
    agent._gliner_model = gliner
    telemetry = RecordingTelemetryManager()
    agent.telemetry_manager = telemetry
    for query in queries:
        await agent._process_impl(GatewayInput(query=query, tenant_id=TEST_TENANT_ID))
    rows = []
    for call, span in zip(telemetry.calls, telemetry.spans):
        if call["name"] != "cogniverse.routing":
            continue
        row = {f"attributes.{key}": value for key, value in span.attributes.items()}
        row["context.span_id"] = f"span-{len(rows)}"
        rows.append(row)
    return pd.DataFrame(rows)


QUERIES = [
    "show me cat videos",  # rules: simple video raw_results -> search_agent
    "compare the two reports",  # rules: complex (keyword)
    "find dog photos",
    "listen to podcasts",
    "summarize this",
    "find documents",
]


@pytest.mark.asyncio
async def test_each_review_settles_exactly_its_heads():
    spans = await _routing_rows(QUERIES)
    decisions = [
        json.loads(row["attributes.output.value"]) for _, row in spans.iterrows()
    ]
    assert [
        (d["complexity"], d["modality"], d["generation_type"]) for d in decisions[:2]
    ] == [("simple", "video", "raw_results"), ("complex", "video", "raw_results")]

    reviews = {
        "span-0": ("correct", {}),
        "span-1": ("wrong", {"suggested_agent": "image_search_agent"}),
        "span-2": ("wrong", {"suggested_agent": "orchestrator_agent"}),
        "span-3": ("wrong_routing", {"suggested_agent": "summarizer_agent"}),
        "span-4": ("ambiguous", {}),
        "span-5": ("wrong", {}),
    }

    assert gateway_label_examples(spans, reviews) == [
        QueryLabelExample(
            "span-0",
            "show me cat videos",
            {
                "complexity": "simple",
                "modality": "video",
                "generation_type": "raw_results",
            },
        ),
        QueryLabelExample(
            "span-1",
            "compare the two reports",
            {
                "complexity": "simple",
                "modality": "image",
                "generation_type": "raw_results",
            },
        ),
        QueryLabelExample("span-2", "find dog photos", {"complexity": "complex"}),
        QueryLabelExample(
            "span-3",
            "listen to podcasts",
            {"complexity": "simple", "generation_type": "summary"},
        ),
    ]


@pytest.mark.asyncio
async def test_unreviewed_spans_give_no_examples():
    spans = await _routing_rows(QUERIES[:2])
    assert gateway_label_examples(spans, {}) == []
    assert gateway_label_examples(spans, {"span-9": ("correct", {})}) == []


@pytest.mark.asyncio
async def test_a_correct_review_of_a_corrupt_decision_raises():
    spans = await _routing_rows(QUERIES[:1])
    decision = json.loads(spans.loc[0, "attributes.output.value"])
    spans.loc[0, "attributes.output.value"] = json.dumps(
        {**decision, "modality": "hologram"}
    )
    with pytest.raises(ValueError, match="records modality='hologram', not one of"):
        gateway_label_examples(spans, {"span-0": ("correct", {})})


def test_suggested_agent_labels_follow_the_route_map():
    assert {
        agent: suggested_agent_labels(agent)
        for agent in sorted({*SIMPLE_ROUTE_MAP.values(), "orchestrator_agent"})
    } == {
        "audio_analysis_agent": {
            "complexity": "simple",
            "modality": "audio",
            "generation_type": "raw_results",
        },
        "detailed_report_agent": {
            "complexity": "simple",
            "generation_type": "detailed_report",
        },
        "document_agent": {
            "complexity": "simple",
            "modality": "document",
            "generation_type": "raw_results",
        },
        "image_search_agent": {
            "complexity": "simple",
            "modality": "image",
            "generation_type": "raw_results",
        },
        "orchestrator_agent": {"complexity": "complex"},
        "search_agent": {"complexity": "simple", "generation_type": "raw_results"},
        "summarizer_agent": {"complexity": "simple", "generation_type": "summary"},
    }
    assert suggested_agent_labels("coding_agent") == {}
