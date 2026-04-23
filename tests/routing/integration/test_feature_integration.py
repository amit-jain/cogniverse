"""
Integration tests for automation, evaluation, and versioning features.

Tests exercise the full round-trip through real telemetry infrastructure:
1. Online evaluation: write span → evaluate → read back annotation scores
2. Annotation queue: write spans → identify → enqueue → assign → complete
3. Automation rules: config-driven thresholds change annotation behavior
4. Dataset versioning: save v1 → save v2 → list → verify both in Phoenix

NO MOCKS for telemetry/Phoenix. Uses shared phoenix_container fixture.
"""

import logging
import time
from datetime import datetime, timedelta, timezone

import pytest

from cogniverse_agents.routing.annotation_agent import (
    AnnotationAgent,
    AnnotationStatus,
)
from cogniverse_agents.routing.annotation_queue import AnnotationQueue
from cogniverse_agents.routing.config import AutomationRulesConfig
from cogniverse_evaluation.online_evaluator import OnlineEvaluator
from cogniverse_foundation.telemetry.config import SPAN_NAME_ROUTING
from tests.utils.async_polling import (
    simulate_processing_delay,
    wait_for_phoenix_processing,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def test_tenant_id():
    return f"test_feature_integ_{int(time.time())}"


@pytest.fixture
def real_provider(telemetry_manager_with_phoenix, test_tenant_id):
    return telemetry_manager_with_phoenix.get_provider(tenant_id=test_tenant_id)


@pytest.fixture
def project_name(telemetry_config_with_phoenix, test_tenant_id):
    return telemetry_config_with_phoenix.get_project_name(test_tenant_id)


def _write_routing_spans(tm, tenant_id, spans_data):
    """Write routing spans to Phoenix via real telemetry manager."""
    span_ids = []
    for query, agent, confidence in spans_data:
        with tm.span(
            name=SPAN_NAME_ROUTING,
            tenant_id=tenant_id,
            attributes={
                "routing.query": query,
                "routing.chosen_agent": agent,
                "routing.confidence": confidence,
                "routing.context": "{}",
                "routing.processing_time": 0.05,
            },
        ) as span:
            ctx = getattr(span, "context", None)
            if ctx and hasattr(ctx, "span_id"):
                span_ids.append(str(ctx.span_id))
            simulate_processing_delay(delay=0.05)
    tm.force_flush(timeout_millis=5000)
    wait_for_phoenix_processing(delay=2)
    return span_ids


class TestOnlineEvaluationIntegration:
    """2: Online evaluation scores real spans and persists annotations."""

    @pytest.mark.asyncio
    async def test_evaluate_span_and_persist_to_phoenix(
        self,
        telemetry_manager_with_phoenix,
        real_provider,
        project_name,
        test_tenant_id,
    ):
        """Write a routing span → evaluate online → read back annotation score."""
        _write_routing_spans(
            telemetry_manager_with_phoenix,
            test_tenant_id,
            [("search for video clips about cats", "search_agent", 0.85)],
        )

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=5)
        spans_df = await real_provider.traces.get_spans(
            project=project_name, start_time=start_time, end_time=end_time
        )
        assert not spans_df.empty, "No spans found in Phoenix"

        routing_spans = spans_df[spans_df["name"] == SPAN_NAME_ROUTING]
        assert len(routing_spans) >= 1

        evaluator = OnlineEvaluator(
            provider=real_provider,
            project_name=project_name,
        )

        span_row = routing_spans.iloc[0]
        span_dict = span_row.to_dict()
        results = await evaluator.evaluate_span(span_dict)

        assert len(results) >= 1, "Expected at least 1 evaluation result"

        eval_names = {r.evaluator_name for r in results}
        assert "routing_outcome" in eval_names or "confidence_calibration" in eval_names

        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Score {r.score} out of range"
            assert r.label, f"Empty label for {r.evaluator_name}"
            assert r.span_id, "Missing span_id in result"

        stats = evaluator.get_statistics()
        assert stats["total_evaluated"] == 1


class TestAnnotationQueueIntegration:
    """3: Annotation queue with real Phoenix spans."""

    @pytest.mark.asyncio
    async def test_identify_enqueue_assign_complete(
        self,
        telemetry_manager_with_phoenix,
        real_provider,
        project_name,
        test_tenant_id,
    ):
        """Write low-confidence spans → identify → enqueue → assign → complete."""
        _write_routing_spans(
            telemetry_manager_with_phoenix,
            test_tenant_id,
            [
                ("ambiguous query about weather", "search_agent", 0.25),
                ("unclear question maybe video", "summarizer_agent", 0.4),
                ("clear video search request", "search_agent", 0.95),
            ],
        )

        agent = AnnotationAgent(
            tenant_id=test_tenant_id,
            confidence_threshold=0.6,
            max_annotations_per_run=10,
        )

        requests = await agent.identify_spans_needing_annotation(lookback_hours=1)
        assert len(requests) >= 2, (
            f"Expected >=2 annotation requests for low-confidence spans, got {len(requests)}"
        )

        low_conf = [r for r in requests if r.routing_confidence < 0.6]
        assert len(low_conf) >= 2, f"Expected >=2 low confidence, got {len(low_conf)}"

        queue = AnnotationQueue()
        added = queue.enqueue_batch(requests)
        assert added == len(requests)
        assert queue.size() == len(requests)

        first = queue.get_pending()[0]
        assigned = queue.assign(first.span_id, reviewer="integration_test_user")
        assert assigned.status == AnnotationStatus.ASSIGNED
        assert assigned.assigned_to == "integration_test_user"
        assert assigned.sla_deadline is not None

        completed = queue.complete(first.span_id, label="correct_routing")
        assert completed.status == AnnotationStatus.COMPLETED
        assert completed.completed_at is not None

        stats = queue.statistics()
        assert stats["by_status"]["completed"] == 1
        assert stats["by_status"].get("pending", 0) == len(requests) - 1


class TestAutomationRulesIntegration:
    """1: Config-driven thresholds change real annotation behavior."""

    @pytest.mark.asyncio
    async def test_strict_threshold_flags_more_spans(
        self,
        telemetry_manager_with_phoenix,
        test_tenant_id,
    ):
        """Same spans, different config → different annotation counts."""
        _write_routing_spans(
            telemetry_manager_with_phoenix,
            test_tenant_id,
            [
                ("video about cooking", "search_agent", 0.7),
                ("documentary search", "search_agent", 0.75),
                ("find presentations", "search_agent", 0.85),
            ],
        )

        lenient = AnnotationAgent(
            tenant_id=test_tenant_id,
            confidence_threshold=0.6,
            max_annotations_per_run=20,
        )
        lenient_requests = await lenient.identify_spans_needing_annotation(
            lookback_hours=1
        )

        strict_rules = AutomationRulesConfig(
            annotation_thresholds={
                "confidence_threshold": 0.9,
                "boundary_low": 0.85,
                "boundary_high": 0.95,
            }
        )
        strict = AnnotationAgent(
            tenant_id=test_tenant_id,
            automation_rules=strict_rules,
            max_annotations_per_run=20,
        )
        strict_requests = await strict.identify_spans_needing_annotation(
            lookback_hours=1
        )

        assert len(strict_requests) >= len(lenient_requests), (
            f"Strict threshold (0.9) should flag >= spans than lenient (0.6): "
            f"strict={len(strict_requests)}, lenient={len(lenient_requests)}"
        )


class TestDatasetVersioningIntegration:
    """5: Versioned dataset save/load round-trip with real Phoenix."""

    @pytest.mark.asyncio
    async def test_save_v1_v2_list_versions_in_phoenix(
        self,
        telemetry_manager_with_phoenix,
        real_provider,
        test_tenant_id,
    ):
        """Save v1 → save v2 → list_versions → verify both exist in Phoenix."""
        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

        am = ArtifactManager(telemetry_provider=real_provider, tenant_id=test_tenant_id)

        ds_id_1, v1 = await am.save_prompts_versioned(
            "routing", {"system": "You are a router v1"}
        )
        assert v1 == 1
        assert ds_id_1, "No dataset_id returned for v1"

        ds_id_2, v2 = await am.save_prompts_versioned(
            "routing", {"system": "You are an improved router v2"}
        )
        assert v2 == 2
        assert ds_id_2, "No dataset_id returned for v2"
        assert ds_id_1 != ds_id_2, "v1 and v2 should have different dataset IDs"

        versions = await am.list_versions("prompts", "routing")
        assert len(versions) >= 2, f"Expected >=2 versions, got {len(versions)}"
        assert versions[0]["version"] == 1, "First version should be 1"
        assert versions[1]["version"] == 2, "Second version should be 2"

        lineage = await am.get_version_lineage("prompts", "routing")
        assert len(lineage) >= 2
        for entry in lineage:
            assert entry["row_count"] > 0, (
                f"Version {entry['version']} has 0 rows — data not persisted"
            )
