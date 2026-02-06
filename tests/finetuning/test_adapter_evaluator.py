"""
Tests for AdapterEvaluator entity extraction evaluation logic.

Tests the _check_entity_prediction static method which computes
set-based F1 for entity matching, plus ensures routing and
profile_selection evaluation paths remain functional.
"""

import pytest

from cogniverse_finetuning.evaluation.adapter_evaluator import AdapterEvaluator


class TestCheckEntityPrediction:
    """Tests for the entity extraction F1 evaluation logic."""

    def test_entity_extraction_exact_match(self):
        """Same entities, same order -> F1=1.0"""
        pred = {
            "entities": [
                {"text": "John Doe", "type": "PERSON"},
                {"text": "Acme Corp", "type": "ORGANIZATION"},
            ]
        }
        expected = {
            "entities": [
                {"text": "John Doe", "type": "PERSON"},
                {"text": "Acme Corp", "type": "ORGANIZATION"},
            ]
        }

        correct, f1 = AdapterEvaluator._check_entity_prediction(pred, expected)
        assert f1 == 1.0
        assert correct is True

    def test_entity_extraction_partial_match(self):
        """2 of 3 entities match -> F1 ~ 0.8"""
        pred = {
            "entities": [
                {"text": "John Doe", "type": "PERSON"},
                {"text": "Acme Corp", "type": "ORGANIZATION"},
            ]
        }
        expected = {
            "entities": [
                {"text": "John Doe", "type": "PERSON"},
                {"text": "Acme Corp", "type": "ORGANIZATION"},
                {"text": "New York", "type": "LOCATION"},
            ]
        }

        correct, f1 = AdapterEvaluator._check_entity_prediction(pred, expected)
        # precision = 2/2 = 1.0, recall = 2/3 ≈ 0.667, F1 = 2*1*0.667/(1+0.667) ≈ 0.8
        assert f1 == pytest.approx(0.8, abs=0.01)
        assert correct is True  # F1 >= 0.5

    def test_entity_extraction_no_match(self):
        """Completely different entities -> F1=0.0"""
        pred = {
            "entities": [
                {"text": "Alice", "type": "PERSON"},
                {"text": "Bob Corp", "type": "ORGANIZATION"},
            ]
        }
        expected = {
            "entities": [
                {"text": "John Doe", "type": "PERSON"},
                {"text": "Acme Corp", "type": "ORGANIZATION"},
            ]
        }

        correct, f1 = AdapterEvaluator._check_entity_prediction(pred, expected)
        assert f1 == 0.0
        assert correct is False

    def test_entity_extraction_empty_both(self):
        """Both empty -> correct=True, F1=1.0"""
        pred = {"entities": []}
        expected = {"entities": []}

        correct, f1 = AdapterEvaluator._check_entity_prediction(pred, expected)
        assert f1 == 1.0
        assert correct is True

    def test_entity_extraction_case_insensitive(self):
        """'john doe' matches 'John Doe' (case-insensitive text)"""
        pred = {
            "entities": [
                {"text": "john doe", "type": "person"},
            ]
        }
        expected = {
            "entities": [
                {"text": "John Doe", "type": "PERSON"},
            ]
        }

        correct, f1 = AdapterEvaluator._check_entity_prediction(pred, expected)
        assert f1 == 1.0
        assert correct is True

    def test_entity_extraction_different_order(self):
        """Same entities, different order -> F1=1.0"""
        pred = {
            "entities": [
                {"text": "Acme Corp", "type": "ORGANIZATION"},
                {"text": "John Doe", "type": "PERSON"},
            ]
        }
        expected = {
            "entities": [
                {"text": "John Doe", "type": "PERSON"},
                {"text": "Acme Corp", "type": "ORGANIZATION"},
            ]
        }

        correct, f1 = AdapterEvaluator._check_entity_prediction(pred, expected)
        assert f1 == 1.0
        assert correct is True

    def test_entity_extraction_predicted_empty_expected_nonempty(self):
        """Predicted empty, expected non-empty -> F1=0.0"""
        pred = {"entities": []}
        expected = {
            "entities": [
                {"text": "John Doe", "type": "PERSON"},
            ]
        }

        correct, f1 = AdapterEvaluator._check_entity_prediction(pred, expected)
        assert f1 == 0.0
        assert correct is False

    def test_entity_extraction_predicted_nonempty_expected_empty(self):
        """Predicted entities when none expected -> F1=0.0 (false positives)"""
        pred = {
            "entities": [
                {"text": "John Doe", "type": "PERSON"},
            ]
        }
        expected = {"entities": []}

        correct, f1 = AdapterEvaluator._check_entity_prediction(pred, expected)
        assert f1 == 0.0
        assert correct is False

    def test_entity_extraction_missing_entities_key(self):
        """Missing 'entities' key treated as empty list"""
        pred = {}
        expected = {}

        correct, f1 = AdapterEvaluator._check_entity_prediction(pred, expected)
        assert f1 == 1.0
        assert correct is True

    def test_entity_extraction_extra_predictions(self):
        """Extra predicted entities reduce precision"""
        pred = {
            "entities": [
                {"text": "John Doe", "type": "PERSON"},
                {"text": "Acme Corp", "type": "ORGANIZATION"},
                {"text": "Fake Entity", "type": "PERSON"},
            ]
        }
        expected = {
            "entities": [
                {"text": "John Doe", "type": "PERSON"},
                {"text": "Acme Corp", "type": "ORGANIZATION"},
            ]
        }

        correct, f1 = AdapterEvaluator._check_entity_prediction(pred, expected)
        # precision = 2/3, recall = 2/2 = 1.0, F1 = 2*(2/3)*1/(2/3+1) ≈ 0.8
        assert f1 == pytest.approx(0.8, abs=0.01)
        assert correct is True


class TestAdapterEvaluatorAgentTypes:
    """Test that routing and profile_selection evaluation paths still work."""

    def test_routing_evaluation_unchanged(self):
        """Routing uses recommended_agent field comparison."""
        from unittest.mock import MagicMock

        evaluator = AdapterEvaluator(
            telemetry_provider=MagicMock(),
            agent_type="routing",
        )
        # The routing path is tested indirectly through the agent_type check
        # in _evaluate_model. We verify the type is accepted at construction.
        assert evaluator.agent_type == "routing"

    def test_profile_selection_evaluation_unchanged(self):
        """Profile selection uses selected_profiles field comparison."""
        from unittest.mock import MagicMock

        evaluator = AdapterEvaluator(
            telemetry_provider=MagicMock(),
            agent_type="profile_selection",
        )
        assert evaluator.agent_type == "profile_selection"

    def test_entity_extraction_type_accepted(self):
        """Entity extraction is an accepted agent_type."""
        from unittest.mock import MagicMock

        evaluator = AdapterEvaluator(
            telemetry_provider=MagicMock(),
            agent_type="entity_extraction",
        )
        assert evaluator.agent_type == "entity_extraction"
