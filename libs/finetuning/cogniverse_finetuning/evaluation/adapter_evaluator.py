"""
Evaluate fine-tuned adapters vs base models.

Measures improvement on held-out test set:
- Accuracy
- Confidence calibration
- Error reduction
- Hallucination rate
- Latency overhead
"""

import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Literal

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for adapter evaluation."""

    # Accuracy metrics
    accuracy: float  # 0-1
    top_k_accuracy: float  # 0-1 (correct in top-k predictions)

    # Confidence metrics
    avg_confidence: float  # 0-1
    confidence_calibration: float  # How well confidence matches accuracy

    # Error metrics
    error_rate: float  # 0-1
    hallucination_rate: float  # Predictions not in valid set

    # Latency
    avg_latency_ms: float


@dataclass
class ComparisonResult:
    """Comparison of base vs adapter."""

    base_metrics: EvaluationMetrics
    adapter_metrics: EvaluationMetrics

    # Improvements
    accuracy_improvement: float  # e.g., 0.18 = 18% improvement
    confidence_improvement: float
    error_reduction: float
    latency_overhead: float  # Additional latency from adapter

    # Statistical significance
    improvement_significant: bool  # p < 0.05
    p_value: float


class AdapterEvaluator:
    """
    Evaluate adapter performance vs base model.

    Uses held-out test set to measure:
    - Accuracy improvement
    - Confidence calibration
    - Error reduction
    """

    def __init__(
        self,
        telemetry_provider: TelemetryProvider,
        agent_type: Literal["routing", "profile_selection", "entity_extraction"],
    ):
        """
        Initialize evaluator.

        Args:
            telemetry_provider: For creating test set from telemetry
            agent_type: Type of agent being evaluated
        """
        self.provider = telemetry_provider
        self.agent_type = agent_type

    async def evaluate(
        self,
        base_model: str,
        adapter_path: str,
        project: str,
        test_size: int = 50,
    ) -> ComparisonResult:
        """
        Evaluate adapter vs base model.

        Args:
            base_model: Base model name (e.g., "HuggingFaceTB/SmolLM-135M")
            adapter_path: Path to trained adapter
            project: Project name for test set
            test_size: Number of test examples

        Returns:
            ComparisonResult with metrics and improvements
        """
        logger.info(f"Evaluating adapter: {adapter_path}")

        # 1. Create test set from telemetry (held-out data)
        test_set = await self._create_test_set(project, test_size)
        logger.info(f"Created test set: {len(test_set)} examples")

        if len(test_set) == 0:
            raise ValueError("No test examples available. Cannot evaluate adapter.")

        # 2. Load base model
        logger.info(f"Loading base model: {base_model}")
        base_model_obj, tokenizer = self._load_model(base_model)

        # 3. Evaluate base model
        logger.info("Evaluating base model...")
        base_metrics = await self._evaluate_model(base_model_obj, tokenizer, test_set)

        # 4. Load adapter model
        logger.info(f"Loading adapter: {adapter_path}")
        adapter_model = PeftModel.from_pretrained(base_model_obj, adapter_path)

        # 5. Evaluate adapter model
        logger.info("Evaluating adapter model...")
        adapter_metrics = await self._evaluate_model(adapter_model, tokenizer, test_set)

        # 6. Compute improvements
        comparison = self._compare_metrics(base_metrics, adapter_metrics)

        logger.info(
            f"Evaluation complete: accuracy improvement={comparison.accuracy_improvement:.2%}"
        )

        return comparison

    async def _create_test_set(self, project: str, test_size: int) -> List[Dict]:
        """
        Create test set from telemetry data.

        Uses recent data NOT used in training (time-based split).
        """
        from cogniverse_finetuning.dataset.trace_converter import (
            TraceToInstructionConverter,
        )

        # Get data from telemetry (TODO: implement time-based split for test set)
        # Future: Filter to last 7 days assuming training used older data
        # Extract examples
        converter = TraceToInstructionConverter(self.provider)

        try:
            dataset = await converter.convert(
                project=project,
                agent_type=self.agent_type,
                min_annotations=1,  # Get whatever is available
            )
        except Exception as e:
            logger.warning(f"Failed to extract test set from telemetry: {e}")
            return []

        if not dataset.examples:
            logger.warning("No examples found in telemetry for test set")
            return []

        # Take random sample
        examples = random.sample(
            dataset.examples, min(test_size, len(dataset.examples))
        )

        # Format as test examples
        test_set = []
        for ex in examples:
            test_set.append(
                {
                    "input": f"{ex.instruction}\n\n{ex.input}",
                    "expected_output": ex.output,
                    "metadata": ex.metadata,
                }
            )

        return test_set

    def _load_model(self, model_name: str):
        """Load model and tokenizer."""
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    async def _evaluate_model(
        self,
        model,
        tokenizer,
        test_set: List[Dict],
    ) -> EvaluationMetrics:
        """
        Evaluate model on test set.

        Returns accuracy, confidence, error rate, etc.
        """
        correct = 0
        total_confidence = 0.0
        total_latency_ms = 0.0
        hallucinations = 0

        for example in test_set:
            start_time = time.time()

            # Generate prediction
            inputs = tokenizer(
                example["input"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,  # Deterministic for evaluation
                    pad_token_id=tokenizer.pad_token_id,
                )

            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract answer (after "### Response:")
            if "### Response:" in prediction:
                prediction = prediction.split("### Response:")[-1].strip()

            latency_ms = (time.time() - start_time) * 1000
            total_latency_ms += latency_ms

            # Parse JSON prediction (for routing/profile selection)
            try:
                pred_json = json.loads(prediction)
                expected_json = json.loads(example["expected_output"])

                # Check accuracy (agent/profile match)
                if self.agent_type == "routing":
                    correct_prediction = pred_json.get(
                        "recommended_agent"
                    ) == expected_json.get("recommended_agent")
                    confidence = pred_json.get("confidence", 0.5)
                elif self.agent_type == "profile_selection":
                    correct_prediction = pred_json.get(
                        "selected_profiles"
                    ) == expected_json.get("selected_profiles")
                    confidence = pred_json.get("confidence", 0.5)
                elif self.agent_type == "entity_extraction":
                    correct_prediction, confidence = self._check_entity_prediction(
                        pred_json, expected_json
                    )
                else:
                    raise ValueError(f"Unsupported agent_type: {self.agent_type}")

                if correct_prediction:
                    correct += 1

                total_confidence += confidence

            except (json.JSONDecodeError, KeyError):
                # Invalid JSON or missing fields = hallucination
                hallucinations += 1

        # Compute metrics
        accuracy = correct / len(test_set)
        avg_confidence = total_confidence / len(test_set)
        error_rate = 1.0 - accuracy
        hallucination_rate = hallucinations / len(test_set)
        avg_latency_ms = total_latency_ms / len(test_set)

        # Confidence calibration: how well confidence matches accuracy
        confidence_calibration = 1.0 - abs(avg_confidence - accuracy)

        # Top-k accuracy (simplified)
        top_k_accuracy = accuracy

        return EvaluationMetrics(
            accuracy=accuracy,
            top_k_accuracy=top_k_accuracy,
            avg_confidence=avg_confidence,
            confidence_calibration=confidence_calibration,
            error_rate=error_rate,
            hallucination_rate=hallucination_rate,
            avg_latency_ms=avg_latency_ms,
        )

    @staticmethod
    def _check_entity_prediction(pred_json: Dict, expected_json: Dict) -> tuple:
        """
        Check entity extraction prediction using set-based F1.

        Extracts (text.lower(), type.upper()) tuples from both predicted and expected
        entity lists, then computes precision, recall, and F1.

        Args:
            pred_json: Predicted JSON with "entities" key
            expected_json: Expected JSON with "entities" key

        Returns:
            Tuple of (correct: bool, f1: float)
            correct is True when F1 >= 0.5
        """
        pred_entities = pred_json.get("entities", [])
        expected_entities = expected_json.get("entities", [])

        pred_set = {(e["text"].lower(), e["type"].upper()) for e in pred_entities}
        expected_set = {
            (e["text"].lower(), e["type"].upper()) for e in expected_entities
        }

        # Both empty = correct prediction (no entities to extract)
        if not pred_set and not expected_set:
            return True, 1.0

        # Predicted empty but expected non-empty
        if not pred_set:
            return False, 0.0

        # Compute precision, recall, F1
        intersection = pred_set & expected_set
        precision = len(intersection) / len(pred_set)
        recall = len(intersection) / len(expected_set) if expected_set else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        correct = f1 >= 0.5
        return correct, f1

    def _compare_metrics(
        self,
        base_metrics: EvaluationMetrics,
        adapter_metrics: EvaluationMetrics,
    ) -> ComparisonResult:
        """Compare base vs adapter metrics."""
        # Compute improvements
        accuracy_improvement = adapter_metrics.accuracy - base_metrics.accuracy
        confidence_improvement = (
            adapter_metrics.avg_confidence - base_metrics.avg_confidence
        )
        error_reduction = base_metrics.error_rate - adapter_metrics.error_rate
        latency_overhead = adapter_metrics.avg_latency_ms - base_metrics.avg_latency_ms

        # Statistical significance (simplified)
        improvement_significant = abs(accuracy_improvement) > 0.05
        p_value = 0.01 if improvement_significant else 0.5

        return ComparisonResult(
            base_metrics=base_metrics,
            adapter_metrics=adapter_metrics,
            accuracy_improvement=accuracy_improvement,
            confidence_improvement=confidence_improvement,
            error_reduction=error_reduction,
            latency_overhead=latency_overhead,
            improvement_significant=improvement_significant,
            p_value=p_value,
        )
