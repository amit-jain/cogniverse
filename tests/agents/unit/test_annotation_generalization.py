"""Human-annotation capture generalized to every agent type.

Pins:
- ``AnnotationStorage`` is per-agent-type: annotations persist under
  ``{agent_type}_annotation`` (routing keeps ``routing_annotation`` so stored
  data stays readable);
- ``AnnotationRequest`` carries ``agent_type``;
- ``AnnotationLabel`` gains agent-generic ``correct``/``wrong`` labels and
  ``quality_map`` maps them;
- span analysis reads the canonical ``input.value``/``output.value`` slots the
  emitters actually write — the old read of ``attributes.routing`` (never set
  by the gateway) made the annotation agent identify zero spans in production.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _stub_telemetry_manager():
    tm = MagicMock()
    tm.config.get_project_name.return_value = "cogniverse-t-runtime"
    provider = MagicMock()
    provider.annotations.add_annotation = AsyncMock()
    tm.get_provider.return_value = provider
    return tm


class TestAnnotationStoragePerAgentType:
    def _storage(self, **kwargs):
        from cogniverse_agents.routing.annotation_storage import AnnotationStorage

        with patch(
            "cogniverse_agents.routing.annotation_storage.get_telemetry_manager",
            return_value=_stub_telemetry_manager(),
        ):
            return AnnotationStorage(tenant_id="t", **kwargs)

    def test_default_agent_type_keeps_routing_annotation_name(self):
        storage = self._storage()
        assert storage.agent_type == "routing"
        assert storage.annotation_name == "routing_annotation"

    def test_agent_type_selects_annotation_name(self):
        storage = self._storage(agent_type="summary")
        assert storage.annotation_name == "summary_annotation"

    @pytest.mark.asyncio
    async def test_store_human_annotation_persists_under_agent_name(self):
        from cogniverse_agents.routing.llm_auto_annotator import AnnotationLabel

        storage = self._storage(agent_type="entity_extraction")
        await storage.store_human_annotation(
            span_id="s1", label=AnnotationLabel.CORRECT, reasoning="good extraction"
        )

        call = storage.provider.annotations.add_annotation.await_args
        assert call.kwargs["name"] == "entity_extraction_annotation"
        assert call.kwargs["label"] == "correct"
        assert call.kwargs["score"] == 1.0
        assert call.kwargs["metadata"]["human_reviewed"] is True

    def test_legacy_class_name_still_importable(self):
        from cogniverse_agents.routing.annotation_storage import (
            AnnotationStorage,
            RoutingAnnotationStorage,
        )

        assert RoutingAnnotationStorage is AnnotationStorage


class TestGenericLabels:
    def test_annotation_label_has_generic_members(self):
        from cogniverse_agents.routing.llm_auto_annotator import AnnotationLabel

        assert AnnotationLabel.CORRECT.value == "correct"
        assert AnnotationLabel.WRONG.value == "wrong"
        # Legacy routing labels stay for stored-data back-compat.
        assert AnnotationLabel.CORRECT_ROUTING.value == "correct_routing"
        assert AnnotationLabel.WRONG_ROUTING.value == "wrong_routing"

    def test_quality_map_scores_generic_labels(self):
        from cogniverse_agents.routing.config import FeedbackConfig

        quality_map = FeedbackConfig().quality_map
        assert quality_map["correct"] == 0.9
        assert quality_map["wrong"] == 0.3
        assert quality_map["correct_routing"] == 0.9
        assert quality_map["wrong_routing"] == 0.3


class TestAnnotationRequestAgentType:
    def test_agent_type_defaults_to_routing(self):
        from datetime import datetime

        from cogniverse_agents.routing.annotation_agent import (
            AnnotationPriority,
            AnnotationRequest,
        )
        from cogniverse_evaluation.evaluators.routing_evaluator import RoutingOutcome

        request = AnnotationRequest(
            span_id="s1",
            timestamp=datetime.now(),
            query="q",
            chosen_agent="video_search",
            routing_confidence=0.5,
            outcome=RoutingOutcome.AMBIGUOUS,
            priority=AnnotationPriority.HIGH,
            reason="r",
            context={},
        )
        assert request.agent_type == "routing"
        assert request.to_dict()["agent_type"] == "routing"


class TestSpanAnalysisReadsCanonicalSlots:
    def _agent(self, **kwargs):
        from cogniverse_agents.routing.annotation_agent import AnnotationAgent

        with patch(
            "cogniverse_agents.routing.annotation_agent.get_telemetry_manager",
            return_value=_stub_telemetry_manager(),
        ):
            return AnnotationAgent(tenant_id="t", **kwargs)

    def _canonical_routing_row(self, confidence=0.2):
        """A routing span the way the gateway actually emits it: canonical
        input.value/output.value only — no attributes.routing."""
        return pd.Series(
            {
                "context.span_id": "span-1",
                "name": "cogniverse.routing",
                "start_time": "2026-07-17T10:00:00Z",
                "status_code": "OK",
                "parent_id": "parent-1",
                "attributes.input.value": "find robot videos",
                "attributes.output.value": json.dumps(
                    {"chosen_agent": "video_search", "confidence": confidence}
                ),
            }
        )

    def test_canonical_low_confidence_routing_span_is_flagged(self):
        agent = self._agent()
        request = agent._analyze_span_for_annotation(self._canonical_routing_row())

        assert request is not None
        assert request.query == "find robot videos"
        assert request.chosen_agent == "video_search"
        assert request.routing_confidence == pytest.approx(0.2)
        assert request.agent_type == "routing"

    def test_legacy_routing_attributes_still_analyzed(self):
        agent = self._agent()
        row = pd.Series(
            {
                "context.span_id": "span-2",
                "name": "cogniverse.routing",
                "start_time": "2026-07-17T10:00:00Z",
                "status_code": "OK",
                "parent_id": "parent-1",
                "attributes.routing": {
                    "query": "old span",
                    "chosen_agent": "video_search",
                    "confidence": 0.2,
                },
            }
        )
        request = agent._analyze_span_for_annotation(row)

        assert request is not None
        assert request.query == "old span"

    def test_confident_successful_span_not_flagged(self):
        agent = self._agent()
        request = agent._analyze_span_for_annotation(
            self._canonical_routing_row(confidence=0.95)
        )
        assert request is None

    @pytest.mark.asyncio
    async def test_identify_queries_the_agent_types_span_name(self):
        agent = self._agent()
        agent.provider.traces.get_spans = AsyncMock(return_value=pd.DataFrame())

        await agent.identify_spans_needing_annotation(
            lookback_hours=1, agent_type="query_enhancement"
        )

        call = agent.provider.traces.get_spans.await_args
        assert call.kwargs["filters"] == {"name": "cogniverse.query_enhancement"}

    def test_non_routing_span_analysis_uses_agent_type(self):
        agent = self._agent()
        row = pd.Series(
            {
                "context.span_id": "qe-1",
                "name": "cogniverse.query_enhancement",
                "start_time": "2026-07-17T10:00:00Z",
                "status_code": "OK",
                "attributes.input.value": "find cats",
                "attributes.output.value": json.dumps(
                    {"enhanced_query": "find cats", "confidence": 0.2}
                ),
            }
        )
        request = agent._analyze_span_for_annotation(
            row, agent_type="query_enhancement"
        )

        assert request is not None
        assert request.agent_type == "query_enhancement"
        assert request.query == "find cats"
        assert request.routing_confidence == pytest.approx(0.2)
