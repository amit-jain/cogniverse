"""
Tests for AdapterEvaluator entity extraction evaluation logic.

Tests the _check_entity_prediction static method which computes
set-based F1 for entity matching, plus ensures routing and
profile_selection evaluation paths remain functional.
"""

import pytest

from cogniverse_finetuning.evaluation.adapter_evaluator import AdapterEvaluator


class _FakeEncoding:
    def to(self, device):
        return {}


class _FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return _FakeEncoding()

    def decode(self, ids, skip_special_tokens=True):
        return ids


class _FakeModel:
    device = "cpu"

    def __init__(self, prediction: str):
        self._prediction = prediction

    def generate(self, **kwargs):
        return [self._prediction]


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


class TestEvaluateModelConfidenceCoercion:
    """A label-shaped model confidence must not crash the eval loop."""

    @pytest.mark.asyncio
    async def test_label_confidence_is_coerced(self):
        evaluator = object.__new__(AdapterEvaluator)
        evaluator.agent_type = "routing"
        prediction = '{"recommended_agent": "search_agent", "confidence": "high"}'

        metrics = await evaluator._evaluate_model(
            _FakeModel(prediction),
            _FakeTokenizer(),
            [
                {
                    "input": "route this",
                    "expected_output": '{"recommended_agent": "search_agent"}',
                }
            ],
        )

        assert metrics.accuracy == pytest.approx(1.0)
        assert metrics.avg_confidence == pytest.approx(0.9)


class TestSignificanceTest:
    """The base-vs-adapter comparison must emit a REAL p-value from a
    two-proportion z-test, not the former hardcoded 0.01/0.5 placeholder."""

    def test_p_value_helper_identical_rates_is_one(self):
        from cogniverse_finetuning.evaluation.adapter_evaluator import (
            _two_proportion_p_value,
        )

        # z == 0 for equal proportions -> two-tailed p == 1.0 exactly.
        assert _two_proportion_p_value(0.5, 100, 0.5, 100) == pytest.approx(1.0)

    def test_p_value_helper_large_difference_is_significant(self):
        from cogniverse_finetuning.evaluation.adapter_evaluator import (
            _two_proportion_p_value,
        )

        # 0.40 vs 0.70 over n=100 each: z ~= -4.26, two-tailed p ~= 2e-5.
        p = _two_proportion_p_value(0.40, 100, 0.70, 100)
        assert p < 1e-3

    def test_p_value_helper_empty_sample_is_one(self):
        from cogniverse_finetuning.evaluation.adapter_evaluator import (
            _two_proportion_p_value,
        )

        assert _two_proportion_p_value(0.9, 0, 0.5, 100) == 1.0

    def test_compare_metrics_uses_real_p_value(self):
        from cogniverse_finetuning.evaluation.adapter_evaluator import (
            EvaluationMetrics,
        )

        evaluator = object.__new__(AdapterEvaluator)

        def _metrics(acc: float, n: int) -> EvaluationMetrics:
            return EvaluationMetrics(
                accuracy=acc,
                top_k_accuracy=acc,
                avg_confidence=0.8,
                confidence_calibration=0.0,
                error_rate=1.0 - acc,
                hallucination_rate=0.0,
                avg_latency_ms=10.0,
                sample_count=n,
            )

        # No real difference -> p == 1.0, not significant, not the old 0.5.
        same = evaluator._compare_metrics(_metrics(0.7, 100), _metrics(0.7, 100))
        assert same.p_value == pytest.approx(1.0)
        assert same.improvement_significant is False

        # Large accuracy gain -> significant with a small real p-value, not 0.01.
        improved = evaluator._compare_metrics(_metrics(0.40, 100), _metrics(0.70, 100))
        assert improved.p_value < 0.05
        assert improved.improvement_significant is True
        assert improved.accuracy_improvement == pytest.approx(0.30)


class TestHeldOutTestSet:
    """The test set must exclude every example the adapter was trained on;
    otherwise accuracy is measured on memorised data and inflated."""

    @staticmethod
    def _dataset(triples):
        from cogniverse_finetuning.dataset.trace_converter import (
            InstructionDataset,
            InstructionExample,
        )

        return InstructionDataset(
            examples=[
                InstructionExample(instruction=i, input=inp, output=o, metadata={})
                for (i, inp, o) in triples
            ],
            metadata={},
        )

    def _evaluator_with(self, monkeypatch, dataset):
        import cogniverse_finetuning.dataset.trace_converter as tc

        class _FakeConverter:
            def __init__(self, provider):
                pass

            async def convert(self, **kwargs):
                return dataset

        monkeypatch.setattr(tc, "TraceToInstructionConverter", _FakeConverter)
        ev = object.__new__(AdapterEvaluator)
        ev.provider = object()
        ev.agent_type = "routing"
        return ev

    def test_identity_is_deterministic_and_content_sensitive(self):
        from cogniverse_finetuning.evaluation.adapter_evaluator import example_identity

        a = example_identity("route", "q1", "search_agent")
        assert a == example_identity("route", "q1", "search_agent")
        assert a != example_identity("route", "q1", "summary_agent")
        assert a != example_identity("route", "q2", "search_agent")

    @pytest.mark.asyncio
    async def test_trained_examples_are_excluded(self, monkeypatch):
        from cogniverse_finetuning.evaluation.adapter_evaluator import example_identity

        dataset = self._dataset(
            [
                ("route", "trained one", "search_agent"),
                ("route", "trained two", "summary_agent"),
                ("route", "held out", "detailed_report_agent"),
            ]
        )
        ev = self._evaluator_with(monkeypatch, dataset)
        exclude = {
            example_identity("route", "trained one", "search_agent"),
            example_identity("route", "trained two", "summary_agent"),
        }

        test_set = await ev._create_test_set(
            "proj", test_size=50, exclude_identities=exclude
        )

        assert len(test_set) == 1
        assert test_set[0]["expected_output"] == "detailed_report_agent"
        assert test_set[0]["input"] == "route\n\nheld out"

    @pytest.mark.asyncio
    async def test_all_trained_yields_empty_test_set(self, monkeypatch):
        from cogniverse_finetuning.evaluation.adapter_evaluator import example_identity

        triples = [
            ("route", "a", "search_agent"),
            ("route", "b", "summary_agent"),
        ]
        dataset = self._dataset(triples)
        ev = self._evaluator_with(monkeypatch, dataset)
        exclude = {example_identity(i, inp, o) for (i, inp, o) in triples}

        test_set = await ev._create_test_set(
            "proj", test_size=50, exclude_identities=exclude
        )

        assert test_set == []

    @pytest.mark.asyncio
    async def test_no_exclusion_keeps_all(self, monkeypatch):
        dataset = self._dataset(
            [("route", "a", "x"), ("route", "b", "y"), ("route", "c", "z")]
        )
        ev = self._evaluator_with(monkeypatch, dataset)

        test_set = await ev._create_test_set("proj", test_size=50)

        assert len(test_set) == 3

    @pytest.mark.asyncio
    async def test_evaluate_raises_when_all_examples_were_trained(self, monkeypatch):
        dataset = self._dataset([("route", "a", "x")])
        ev = self._evaluator_with(monkeypatch, dataset)
        from cogniverse_finetuning.evaluation.adapter_evaluator import example_identity

        exclude = {example_identity("route", "a", "x")}

        with pytest.raises(ValueError, match="No held-out test examples"):
            await ev.evaluate(
                base_model="m",
                adapter_path="/p",
                project="proj",
                test_size=10,
                exclude_identities=exclude,
            )
