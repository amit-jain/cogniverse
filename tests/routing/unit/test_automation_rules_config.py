"""
Unit tests for AutomationRulesConfig — declarative automation rules.

Tests:
1. Default values are correct
2. Pydantic validation catches invalid input
3. Round-trip serialization (dict → config → dict, JSON file → config → JSON file)
4. Config propagation to OptimizationOrchestrator, AnnotationAgent, AnnotationFeedbackLoop
5. Behavior change: overriding thresholds changes component behavior
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from cogniverse_agents.routing.annotation_agent import (
    AnnotationAgent,
    AnnotationPriority,
)
from cogniverse_agents.routing.annotation_feedback_loop import AnnotationFeedbackLoop
from cogniverse_agents.routing.config import (
    AnnotationThresholdsConfig,
    AutomationRulesConfig,
)


class TestAutomationRulesConfigDefaults:
    """Verify default values match the previously hardcoded constants."""

    def test_default_annotation_thresholds(self):
        config = AutomationRulesConfig()
        t = config.annotation_thresholds
        assert t.confidence_threshold == 0.6
        assert t.very_low_confidence == 0.3
        assert t.boundary_low == 0.6
        assert t.boundary_high == 0.75
        assert t.failure_lookback_hours == 24
        assert t.max_annotations_per_run == 50
        assert t.max_annotations_per_batch == 10

    def test_default_optimization_triggers(self):
        config = AutomationRulesConfig()
        o = config.optimization_triggers
        assert o.min_annotations_for_optimization == 50
        assert o.optimization_improvement_threshold == 0.05
        assert o.min_days_between_optimizations == 1
        assert o.span_eval_lookback_hours == 2
        assert o.annotation_lookback_hours == 24
        assert o.span_eval_batch_size == 100
        assert o.max_annotations_per_cycle == 100

    def test_default_feedback(self):
        config = AutomationRulesConfig()
        f = config.feedback
        assert f.poll_interval_minutes == 15
        assert f.min_annotations_for_update == 10
        assert f.quality_map == {
            "correct_routing": 0.9,
            "wrong_routing": 0.3,
            "ambiguous": 0.6,
            "insufficient_info": 0.5,
        }

    def test_default_intervals(self):
        config = AutomationRulesConfig()
        i = config.intervals
        assert i.span_eval_interval_minutes == 15
        assert i.annotation_interval_minutes == 30
        assert i.feedback_interval_minutes == 15
        assert i.metrics_report_interval_seconds == 300


class TestAutomationRulesConfigValidation:
    """Verify Pydantic validates input correctly."""

    def test_invalid_confidence_type(self):
        with pytest.raises(ValidationError):
            AnnotationThresholdsConfig(confidence_threshold="not_a_number")

    def test_override_single_field(self):
        config = AutomationRulesConfig(
            annotation_thresholds={"confidence_threshold": 0.8}
        )
        assert config.annotation_thresholds.confidence_threshold == 0.8
        # Other fields keep defaults
        assert config.annotation_thresholds.very_low_confidence == 0.3

    def test_partial_nested_override(self):
        config = AutomationRulesConfig(
            optimization_triggers={"min_annotations_for_optimization": 100}
        )
        assert config.optimization_triggers.min_annotations_for_optimization == 100
        assert config.optimization_triggers.span_eval_lookback_hours == 2

    def test_custom_quality_map(self):
        config = AutomationRulesConfig(
            feedback={
                "quality_map": {
                    "correct_routing": 1.0,
                    "wrong_routing": 0.0,
                    "ambiguous": 0.5,
                    "insufficient_info": 0.4,
                }
            }
        )
        assert config.feedback.quality_map["correct_routing"] == 1.0
        assert config.feedback.quality_map["wrong_routing"] == 0.0


class TestAutomationRulesConfigRoundTrip:
    """Verify round-trip serialization/deserialization."""

    def test_dict_round_trip(self):
        original = AutomationRulesConfig(
            annotation_thresholds={"confidence_threshold": 0.7},
            intervals={"span_eval_interval_minutes": 20},
        )
        serialized = original.to_dict()
        restored = AutomationRulesConfig.from_dict(serialized)

        assert restored.annotation_thresholds.confidence_threshold == 0.7
        assert restored.intervals.span_eval_interval_minutes == 20
        assert restored.to_dict() == serialized

    def test_json_file_round_trip(self, tmp_path):
        original = AutomationRulesConfig(
            annotation_thresholds={"confidence_threshold": 0.55},
            feedback={"poll_interval_minutes": 5},
        )

        filepath = tmp_path / "rules.json"
        original.save(filepath)

        assert filepath.exists()
        restored = AutomationRulesConfig.from_file(filepath)
        assert restored.annotation_thresholds.confidence_threshold == 0.55
        assert restored.feedback.poll_interval_minutes == 5

    def test_yaml_file_round_trip(self, tmp_path):
        original = AutomationRulesConfig(
            optimization_triggers={"min_annotations_for_optimization": 25}
        )

        filepath = tmp_path / "rules.yaml"
        original.save(filepath)

        assert filepath.exists()
        restored = AutomationRulesConfig.from_file(filepath)
        assert restored.optimization_triggers.min_annotations_for_optimization == 25

    def test_from_file_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AutomationRulesConfig.from_file(tmp_path / "nonexistent.json")

    def test_from_file_bad_extension_raises(self, tmp_path):
        bad = tmp_path / "rules.txt"
        bad.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported"):
            AutomationRulesConfig.from_file(bad)


class TestConfigPropagationToAnnotationAgent:
    """Verify AutomationRulesConfig propagates to AnnotationAgent."""

    @patch(
        "cogniverse_agents.routing.annotation_agent.get_telemetry_manager",
    )
    def test_agent_uses_config_thresholds(self, mock_tm):
        mock_tm.return_value = MagicMock()
        mock_tm.return_value.config.get_project_name.return_value = "test-project"
        mock_tm.return_value.get_provider.return_value = MagicMock()

        rules = AutomationRulesConfig(
            annotation_thresholds={
                "confidence_threshold": 0.8,
                "very_low_confidence": 0.2,
                "boundary_low": 0.7,
                "boundary_high": 0.85,
                "failure_lookback_hours": 12,
                "max_annotations_per_run": 25,
            }
        )

        agent = AnnotationAgent(tenant_id="test", automation_rules=rules)

        assert agent.confidence_threshold == 0.8
        assert agent.very_low_confidence == 0.2
        assert agent.boundary_low == 0.7
        assert agent.boundary_high == 0.85
        assert agent.failure_lookback_hours == 12
        assert agent.max_annotations_per_run == 25

    @patch(
        "cogniverse_agents.routing.annotation_agent.get_telemetry_manager",
    )
    def test_agent_without_config_uses_kwargs(self, mock_tm):
        mock_tm.return_value = MagicMock()
        mock_tm.return_value.config.get_project_name.return_value = "test-project"
        mock_tm.return_value.get_provider.return_value = MagicMock()

        agent = AnnotationAgent(
            tenant_id="test",
            confidence_threshold=0.5,
            failure_lookback_hours=48,
            max_annotations_per_run=100,
        )

        assert agent.confidence_threshold == 0.5
        assert agent.failure_lookback_hours == 48
        assert agent.max_annotations_per_run == 100


class TestConfigPropagationToFeedbackLoop:
    """Verify AutomationRulesConfig propagates to AnnotationFeedbackLoop."""

    def test_feedback_loop_uses_config(self):
        rules = AutomationRulesConfig(
            feedback={
                "poll_interval_minutes": 5,
                "min_annotations_for_update": 3,
                "quality_map": {
                    "correct_routing": 1.0,
                    "wrong_routing": 0.1,
                    "ambiguous": 0.5,
                    "insufficient_info": 0.4,
                },
            }
        )

        mock_optimizer = MagicMock()
        loop = AnnotationFeedbackLoop(
            optimizer=mock_optimizer,
            tenant_id="test",
            automation_rules=rules,
        )

        assert loop.poll_interval_minutes == 5
        assert loop.min_annotations_for_update == 3
        assert loop._quality_map["correct_routing"] == 1.0
        assert loop._quality_map["wrong_routing"] == 0.1

    def test_feedback_loop_without_config_uses_kwargs(self):
        mock_optimizer = MagicMock()
        loop = AnnotationFeedbackLoop(
            optimizer=mock_optimizer,
            tenant_id="test",
            poll_interval_minutes=20,
            min_annotations_for_update=5,
        )

        assert loop.poll_interval_minutes == 20
        assert loop.min_annotations_for_update == 5
        # Default quality map
        assert loop._quality_map["correct_routing"] == 0.9


class TestConfigBehaviorChange:
    """Verify that changing config values actually changes annotation behavior."""

    @patch(
        "cogniverse_agents.routing.annotation_agent.get_telemetry_manager",
    )
    def test_higher_threshold_flags_more_spans(self, mock_tm):
        """With a higher confidence threshold, more spans are flagged for annotation."""
        mock_tm.return_value = MagicMock()
        mock_tm.return_value.config.get_project_name.return_value = "test"
        mock_tm.return_value.get_provider.return_value = MagicMock()

        from cogniverse_evaluation.evaluators.routing_evaluator import RoutingOutcome

        # Default threshold (0.6, boundary [0.6, 0.75]):
        # span with confidence 0.8 + SUCCESS → not flagged (above boundary_high)
        default_agent = AnnotationAgent(tenant_id="test")
        needs, priority, reason = default_agent._needs_annotation(
            confidence=0.8,
            outcome=RoutingOutcome.SUCCESS,
            outcome_details={},
            span_row=MagicMock(),
        )
        assert not needs  # 0.8 > 0.75, above boundary range

        # Custom threshold (0.9, boundary [0.85, 0.95]):
        # same span → flagged as MEDIUM (0.8 < 0.9 confidence_threshold)
        strict_rules = AutomationRulesConfig(
            annotation_thresholds={
                "confidence_threshold": 0.9,
                "boundary_low": 0.85,
                "boundary_high": 0.95,
            }
        )
        strict_agent = AnnotationAgent(tenant_id="test", automation_rules=strict_rules)
        needs, priority, reason = strict_agent._needs_annotation(
            confidence=0.8,
            outcome=RoutingOutcome.SUCCESS,
            outcome_details={},
            span_row=MagicMock(),
        )
        assert needs
        assert priority == AnnotationPriority.MEDIUM

    @patch(
        "cogniverse_agents.routing.annotation_agent.get_telemetry_manager",
    )
    def test_custom_very_low_threshold(self, mock_tm):
        """Changing very_low_confidence shifts the HIGH priority boundary."""
        mock_tm.return_value = MagicMock()
        mock_tm.return_value.config.get_project_name.return_value = "test"
        mock_tm.return_value.get_provider.return_value = MagicMock()

        from cogniverse_evaluation.evaluators.routing_evaluator import RoutingOutcome

        # Default (0.3): confidence 0.25 → HIGH
        default_agent = AnnotationAgent(tenant_id="test")
        needs, priority, _ = default_agent._needs_annotation(
            confidence=0.25,
            outcome=RoutingOutcome.SUCCESS,
            outcome_details={},
            span_row=MagicMock(),
        )
        assert needs
        assert priority == AnnotationPriority.HIGH

        # Raise very_low to 0.1: same span → no longer very_low HIGH,
        # but still gets flagged as MEDIUM (below confidence_threshold)
        lenient_rules = AutomationRulesConfig(
            annotation_thresholds={"very_low_confidence": 0.1}
        )
        lenient_agent = AnnotationAgent(
            tenant_id="test", automation_rules=lenient_rules
        )
        needs, priority, _ = lenient_agent._needs_annotation(
            confidence=0.25,
            outcome=RoutingOutcome.SUCCESS,
            outcome_details={},
            span_row=MagicMock(),
        )
        assert needs
        assert priority == AnnotationPriority.MEDIUM

    def test_quality_map_change_affects_scoring(self):
        """Changing the quality map changes score output from feedback loop."""
        rules = AutomationRulesConfig(
            feedback={
                "quality_map": {
                    "correct_routing": 0.99,
                    "wrong_routing": 0.01,
                    "ambiguous": 0.5,
                    "insufficient_info": 0.3,
                }
            }
        )
        loop = AnnotationFeedbackLoop(
            optimizer=MagicMock(), tenant_id="test", automation_rules=rules
        )

        assert loop._label_to_search_quality("correct_routing") == 0.99
        assert loop._label_to_search_quality("wrong_routing") == 0.01
        assert loop._label_to_search_quality("insufficient_info") == 0.3


class TestConfigFromConfigJson:
    """Verify loading from the actual config.json format."""

    def test_load_from_config_json_section(self):
        """Simulate loading the automation_rules section from config.json."""
        config_section = {
            "annotation_thresholds": {
                "confidence_threshold": 0.6,
                "very_low_confidence": 0.3,
                "boundary_low": 0.6,
                "boundary_high": 0.75,
                "failure_lookback_hours": 24,
                "max_annotations_per_run": 50,
                "max_annotations_per_batch": 10,
            },
            "optimization_triggers": {
                "min_annotations_for_optimization": 50,
                "optimization_improvement_threshold": 0.05,
                "min_days_between_optimizations": 1,
                "span_eval_lookback_hours": 2,
                "annotation_lookback_hours": 24,
                "span_eval_batch_size": 100,
                "max_annotations_per_cycle": 100,
            },
            "feedback": {
                "poll_interval_minutes": 15,
                "min_annotations_for_update": 10,
                "quality_map": {
                    "correct_routing": 0.9,
                    "wrong_routing": 0.3,
                    "ambiguous": 0.6,
                    "insufficient_info": 0.5,
                },
            },
            "intervals": {
                "span_eval_interval_minutes": 15,
                "annotation_interval_minutes": 30,
                "feedback_interval_minutes": 15,
                "metrics_report_interval_seconds": 300,
            },
        }

        config = AutomationRulesConfig.from_dict(config_section)
        assert config.annotation_thresholds.confidence_threshold == 0.6
        assert config.optimization_triggers.min_annotations_for_optimization == 50
        assert config.feedback.quality_map["correct_routing"] == 0.9
        assert config.intervals.annotation_interval_minutes == 30

    def test_empty_dict_uses_defaults(self):
        """An empty dict should produce a valid config with all defaults."""
        config = AutomationRulesConfig.from_dict({})
        assert config.annotation_thresholds.confidence_threshold == 0.6
        assert config.intervals.span_eval_interval_minutes == 15
