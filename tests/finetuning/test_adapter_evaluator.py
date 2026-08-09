"""
Tests for AdapterEvaluator entity extraction evaluation logic.

Tests the _check_entity_prediction static method which computes
set-based F1 for entity matching, plus ensures routing and
profile_selection evaluation paths remain functional.
"""

import asyncio
import math
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from cogniverse_finetuning.evaluation import adapter_evaluator
from cogniverse_finetuning.evaluation.adapter_evaluator import AdapterEvaluator


class _FakeTokenIds:
    shape = (1, 2)


class _FakeEncoding(dict):
    def __init__(self):
        super().__init__(input_ids=_FakeTokenIds())

    def to(self, device):
        return self


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.encoded_text = None
        self.decoded_token_ids = None

    def __call__(self, text, **kwargs):
        self.encoded_text = text
        return _FakeEncoding()

    def decode(self, ids, skip_special_tokens=True):
        self.decoded_token_ids = ids
        return ids[-1]


class _FakeModel:
    device = "cpu"

    def __init__(self, prediction: str | list[str], probability: float = 0.8):
        self._predictions = prediction if isinstance(prediction, list) else [prediction]
        self._probability = probability

    def generate(self, **kwargs):
        assert kwargs["num_beams"] == 3
        assert kwargs["num_return_sequences"] == 3
        assert kwargs["return_dict_in_generate"] is True
        assert kwargs["output_scores"] is True
        predictions = self._predictions + [self._predictions[-1]] * (
            3 - len(self._predictions)
        )
        return SimpleNamespace(
            sequences=[[101, 102, prediction] for prediction in predictions[:3]],
            scores=(object(),),
        )

    def compute_transition_scores(
        self, sequences, scores, normalize_logits, beam_indices=None
    ):
        assert normalize_logits is True
        assert len(sequences) == 3
        transition_scores = torch.zeros((3, 3))
        transition_scores[:, 0] = math.log(self._probability)
        return transition_scores


class TestAdapterInferenceBoundary:
    @pytest.mark.asyncio
    async def test_decodes_only_generated_continuation_and_scores_exact_prediction(
        self,
    ):
        evaluator = object.__new__(AdapterEvaluator)
        evaluator.agent_type = "routing"
        tokenizer = _FakeTokenizer()
        prediction = '{"recommended_agent":"search_agent"}'

        metrics = await evaluator._evaluate_model(
            _FakeModel(prediction),
            tokenizer,
            [
                {
                    "input": (
                        "### Instruction:\nRoute the query\n\n"
                        "### Input:\nFind the launch video\n\n"
                        "### Response:"
                    ),
                    "expected_output": prediction,
                }
            ],
        )

        assert tokenizer.encoded_text == (
            "### Instruction:\nRoute the query\n\n"
            "### Input:\nFind the launch video\n\n"
            "### Response:"
        )
        assert tokenizer.decoded_token_ids == [prediction]
        assert metrics.accuracy == 1.0
        assert metrics.top_k_accuracy == 1.0
        assert metrics.avg_confidence == pytest.approx(0.8)
        assert metrics.confidence_calibration == pytest.approx(0.8)
        assert metrics.error_rate == 0.0
        assert metrics.hallucination_rate == 0.0
        assert metrics.correctness == (True,)

    @pytest.mark.asyncio
    async def test_top_k_scores_a_correct_nonleading_candidate_without_changing_top_one(
        self,
    ):
        evaluator = object.__new__(AdapterEvaluator)
        evaluator.agent_type = "routing"

        metrics = await evaluator._evaluate_model(
            _FakeModel(
                [
                    '{"recommended_agent":"summary_agent"}',
                    '{"recommended_agent":"search_agent"}',
                    '{"recommended_agent":"detailed_report_agent"}',
                ],
                probability=0.25,
            ),
            _FakeTokenizer(),
            [
                {
                    "input": "route this",
                    "expected_output": '{"recommended_agent":"search_agent"}',
                }
            ],
        )

        assert metrics.accuracy == 0.0
        assert metrics.top_k_accuracy == 1.0
        assert metrics.avg_confidence == pytest.approx(0.25)
        assert metrics.confidence_calibration == pytest.approx(0.75)
        assert metrics.correctness == (False,)


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
        assert correct is False

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

    def test_relationship_mismatch_fails_exactness_and_reduces_structured_f1(self):
        pred = {
            "entities": [
                {"text": "PyTorch", "type": "PRODUCT"},
                {"text": "Meta AI", "type": "ORG"},
            ],
            "relationships": [
                {"source": "Meta AI", "target": "PyTorch", "type": "owns"}
            ],
        }
        expected = {
            "entities": [
                {"text": "PyTorch", "type": "PRODUCT"},
                {"text": "Meta AI", "type": "ORG"},
            ],
            "relationships": [
                {"source": "Meta AI", "target": "PyTorch", "type": "created"}
            ],
        }

        correct, score = AdapterEvaluator._check_entity_prediction(pred, expected)

        assert correct is False
        assert score == pytest.approx(2 / 3)

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
        assert correct is False


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

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("agent_type", "expected", "prediction"),
        [
            (
                "routing",
                '{"recommended_agent":"search_agent"}',
                '{"recommended_agent":"search_agent"}',
            ),
            (
                "routing",
                '{"recommended_agent":"search_agent"}',
                '{"recommended_agent":"summary_agent"}',
            ),
            (
                "profile_selection",
                '{"selected_profile":"video_colpali"}',
                '{"selected_profile":"video_colpali"}',
            ),
            (
                "profile_selection",
                '{"selected_profile":"video_colpali"}',
                '{"selected_profile":"video_videoprism"}',
            ),
        ],
    )
    async def test_exact_label_scores_one_and_wrong_label_scores_zero(
        self, agent_type, expected, prediction
    ):
        evaluator = object.__new__(AdapterEvaluator)
        evaluator.agent_type = agent_type

        metrics = await evaluator._evaluate_model(
            _FakeModel(prediction),
            _FakeTokenizer(),
            [{"input": "evaluate", "expected_output": expected}],
        )

        assert metrics.accuracy == (1.0 if prediction == expected else 0.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("agent_type", "expected"),
        [
            ("routing", '{"recommended_agent":"search_agent"}'),
            ("profile_selection", '{"selected_profile":"video_colpali"}'),
        ],
    )
    async def test_missing_label_never_scores_as_none_equals_none(
        self, agent_type, expected
    ):
        evaluator = object.__new__(AdapterEvaluator)
        evaluator.agent_type = agent_type

        metrics = await evaluator._evaluate_model(
            _FakeModel("{}"),
            _FakeTokenizer(),
            [{"input": "evaluate", "expected_output": expected}],
        )

        assert metrics.accuracy == 0.0
        assert metrics.hallucination_rate == 1.0


class TestEvaluateModelSequenceConfidence:
    """Categorical confidence is the generated sequence probability."""

    @pytest.mark.asyncio
    async def test_exact_routing_label_has_unit_score(self):
        evaluator = object.__new__(AdapterEvaluator)
        evaluator.agent_type = "routing"
        prediction = '{"recommended_agent":"search_agent"}'

        metrics = await evaluator._evaluate_model(
            _FakeModel(prediction),
            _FakeTokenizer(),
            [
                {
                    "input": "route this",
                    "expected_output": '{"recommended_agent":"search_agent"}',
                }
            ],
        )

        assert metrics.accuracy == pytest.approx(1.0)
        assert metrics.avg_confidence == pytest.approx(0.8)


class TestSignificanceTest:
    """Base and adapter outcomes are compared as paired observations."""

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

    def test_compare_metrics_uses_exact_mcnemar_p_value(self):
        from cogniverse_finetuning.evaluation.adapter_evaluator import (
            EvaluationMetrics,
        )

        evaluator = object.__new__(AdapterEvaluator)

        def _metrics(outcomes: tuple[bool, ...]) -> EvaluationMetrics:
            accuracy = sum(outcomes) / len(outcomes)
            return EvaluationMetrics(
                accuracy=accuracy,
                top_k_accuracy=accuracy,
                avg_confidence=0.8,
                confidence_calibration=0.0,
                error_rate=1.0 - accuracy,
                hallucination_rate=0.0,
                avg_latency_ms=10.0,
                sample_count=len(outcomes),
                correctness=outcomes,
            )

        shared = (True,) * 70 + (False,) * 30
        same = evaluator._compare_metrics(_metrics(shared), _metrics(shared))
        assert same.p_value == pytest.approx(1.0)
        assert same.improvement_significant is False

        base = (True,) * 40 + (False,) * 60
        adapter = (True,) * 70 + (False,) * 30
        improved = evaluator._compare_metrics(_metrics(base), _metrics(adapter))
        assert improved.p_value < 0.05
        assert improved.improvement_significant is True
        assert improved.accuracy_improvement == pytest.approx(0.30)

    def test_compare_metrics_rejects_unpaired_outcomes(self):
        from cogniverse_finetuning.evaluation.adapter_evaluator import (
            EvaluationMetrics,
        )

        evaluator = object.__new__(AdapterEvaluator)

        def _metrics(outcomes):
            return EvaluationMetrics(
                accuracy=0.5,
                top_k_accuracy=0.5,
                avg_confidence=0.5,
                confidence_calibration=0.5,
                error_rate=0.5,
                hallucination_rate=0.0,
                avg_latency_ms=1.0,
                sample_count=len(outcomes),
                correctness=outcomes,
            )

        with pytest.raises(ValueError, match="same non-empty held-out examples"):
            evaluator._compare_metrics(_metrics((True, False)), _metrics((True,)))


class TestAdapterEvaluatorAsyncBoundaries:
    @pytest.mark.asyncio
    async def test_concurrent_model_evaluations_keep_predictions_isolated(self):
        barrier = threading.Barrier(2)

        class _BarrierModel(_FakeModel):
            def generate(self, **kwargs):
                barrier.wait(timeout=2)
                return super().generate(**kwargs)

        first = object.__new__(AdapterEvaluator)
        first.agent_type = "routing"
        second = object.__new__(AdapterEvaluator)
        second.agent_type = "routing"
        expected = '{"recommended_agent":"search_agent"}'

        first_metrics, second_metrics = await asyncio.gather(
            first._evaluate_model(
                _BarrierModel(expected),
                _FakeTokenizer(),
                [{"input": "first", "expected_output": expected}],
            ),
            second._evaluate_model(
                _BarrierModel('{"recommended_agent":"summary_agent"}'),
                _FakeTokenizer(),
                [{"input": "second", "expected_output": expected}],
            ),
        )

        assert first_metrics.correctness == (True,)
        assert second_metrics.correctness == (False,)

    @pytest.mark.asyncio
    async def test_hung_generation_times_out_with_example_context(self, monkeypatch):
        release = threading.Event()

        class _HungModel(_FakeModel):
            def generate(self, **kwargs):
                release.wait(timeout=1)
                return super().generate(**kwargs)

        monkeypatch.setattr(adapter_evaluator, "_MODEL_OPERATION_TIMEOUT_S", 0.01)
        evaluator = object.__new__(AdapterEvaluator)
        evaluator.agent_type = "routing"

        try:
            with pytest.raises(
                TimeoutError, match="generating evaluation example 1 timed out"
            ):
                await evaluator._evaluate_model(
                    _HungModel('{"recommended_agent":"search_agent"}'),
                    _FakeTokenizer(),
                    [
                        {
                            "input": "blocked",
                            "expected_output": '{"recommended_agent":"search_agent"}',
                        }
                    ],
                )
        finally:
            release.set()
            await asyncio.sleep(0.02)

    @pytest.mark.asyncio
    async def test_slow_generation_does_not_block_event_loop(self):
        class _SlowModel(_FakeModel):
            def generate(self, **kwargs):
                time.sleep(0.05)
                return super().generate(**kwargs)

        evaluator = object.__new__(AdapterEvaluator)
        evaluator.agent_type = "routing"
        ticks = 0

        async def ticker():
            nonlocal ticks
            for _ in range(3):
                await asyncio.sleep(0.01)
                ticks += 1

        metrics, _ = await asyncio.gather(
            evaluator._evaluate_model(
                _SlowModel('{"recommended_agent":"search_agent"}'),
                _FakeTokenizer(),
                [
                    {
                        "input": "slow",
                        "expected_output": '{"recommended_agent":"search_agent"}',
                    }
                ],
            ),
            ticker(),
        )

        assert ticks == 3
        assert metrics.correctness == (True,)


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
        a = adapter_evaluator.training_example_identity(
            "routing", "q1", '{"recommended_agent":"search_agent"}'
        )
        assert a == adapter_evaluator.training_example_identity(
            "routing", "q1", '{"recommended_agent": "search_agent"}'
        )
        assert a != adapter_evaluator.training_example_identity(
            "routing", "q1", '{"recommended_agent":"summary_agent"}'
        )
        assert a != adapter_evaluator.training_example_identity(
            "routing", "q2", '{"recommended_agent":"search_agent"}'
        )

    @pytest.mark.asyncio
    async def test_held_out_example_uses_exact_alpaca_inference_prefix(
        self, monkeypatch
    ):
        dataset = self._dataset(
            [
                (
                    "Route the query",
                    "Find the launch video",
                    '{"recommended_agent":"search_agent"}',
                )
            ]
        )
        evaluator = self._evaluator_with(monkeypatch, dataset)

        test_set = await evaluator._create_test_set("proj", test_size=1)

        assert test_set == [
            {
                "input": (
                    "### Instruction:\nRoute the query\n\n"
                    "### Input:\nFind the launch video\n\n"
                    "### Response:"
                ),
                "expected_output": '{"recommended_agent":"search_agent"}',
                "metadata": {},
            }
        ]

    @pytest.mark.asyncio
    async def test_trained_examples_are_excluded(self, monkeypatch):
        dataset = self._dataset(
            [
                ("route", "trained one", '{"recommended_agent":"search_agent"}'),
                ("route", "trained two", '{"recommended_agent":"summary_agent"}'),
                (
                    "route",
                    "held out",
                    '{"recommended_agent":"detailed_report_agent"}',
                ),
            ]
        )
        ev = self._evaluator_with(monkeypatch, dataset)
        exclude = {
            adapter_evaluator.training_example_identity(
                "routing", "trained one", '{"recommended_agent":"search_agent"}'
            ),
            adapter_evaluator.training_example_identity(
                "routing", "trained two", '{"recommended_agent":"summary_agent"}'
            ),
        }

        test_set = await ev._create_test_set(
            "proj", test_size=50, exclude_identities=exclude
        )

        assert len(test_set) == 1
        assert test_set[0]["expected_output"] == (
            '{"recommended_agent":"detailed_report_agent"}'
        )
        assert test_set[0]["input"] == (
            "### Instruction:\nroute\n\n### Input:\nheld out\n\n### Response:"
        )

    @pytest.mark.asyncio
    async def test_all_trained_yields_empty_test_set(self, monkeypatch):
        triples = [
            ("route", "a", '{"recommended_agent":"search_agent"}'),
            ("route", "b", '{"recommended_agent":"summary_agent"}'),
        ]
        dataset = self._dataset(triples)
        ev = self._evaluator_with(monkeypatch, dataset)
        exclude = {
            adapter_evaluator.training_example_identity("routing", prompt, output)
            for _, prompt, output in triples
        }

        test_set = await ev._create_test_set(
            "proj", test_size=50, exclude_identities=exclude
        )

        assert test_set == []

    @pytest.mark.asyncio
    async def test_no_exclusion_keeps_all(self, monkeypatch):
        dataset = self._dataset(
            [
                ("route", "a", '{"recommended_agent":"x"}'),
                ("route", "b", '{"recommended_agent":"y"}'),
                ("route", "c", '{"recommended_agent":"z"}'),
            ]
        )
        ev = self._evaluator_with(monkeypatch, dataset)

        test_set = await ev._create_test_set("proj", test_size=50)

        assert len(test_set) == 3

    @pytest.mark.asyncio
    async def test_evaluate_raises_when_all_examples_were_trained(self, monkeypatch):
        dataset = self._dataset([("route", "a", '{"recommended_agent":"x"}')])
        ev = self._evaluator_with(monkeypatch, dataset)
        exclude = {
            adapter_evaluator.training_example_identity(
                "routing", "a", '{"recommended_agent":"x"}'
            )
        }

        with pytest.raises(ValueError, match="No held-out test examples"):
            await ev.evaluate(
                base_model="m",
                adapter_path="/p",
                project="proj",
                test_size=10,
                exclude_identities=exclude,
            )
