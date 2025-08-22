"""
Inspect AI Integration for structured evaluation

This module shows how Inspect AI helps in the evaluation framework
"""

import logging
from typing import List, Dict, Any, Optional

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, Dataset
from inspect_ai.solver import Solver, solver, TaskState
from inspect_ai.scorer import Scorer, scorer, Score

from .span_evaluator import SpanEvaluator
from .phoenix_experiments import VideoRetrievalExperiment
from .evaluators.reference_free import create_reference_free_evaluators

logger = logging.getLogger(__name__)


@solver
class RetrievalQualitySolver(Solver):
    """
    Inspect AI solver that evaluates retrieval quality using reference-free metrics
    """

    def __init__(self, evaluators: Optional[Dict[str, Any]] = None):
        self.evaluators = evaluators or create_reference_free_evaluators()

    async def solve(self, state: TaskState) -> TaskState:
        """
        Evaluate the retrieval results in the task state

        Args:
            state: Current task state with retrieval results

        Returns:
            Updated task state with evaluation results
        """
        # Extract retrieval results from state
        results = state.metadata.get("retrieval_results", [])
        query = state.input.text

        # Run evaluations
        evaluation_results = {}
        for eval_name, evaluator in self.evaluators.items():
            try:
                eval_result = await evaluator.evaluate(input=query, output=results)
                evaluation_results[eval_name] = {
                    "score": eval_result.score,
                    "label": eval_result.label,
                    "explanation": eval_result.explanation,
                }
            except Exception as e:
                logger.error(f"Evaluator {eval_name} failed: {e}")
                evaluation_results[eval_name] = {"error": str(e)}

        # Update state with evaluation results
        state.metadata["evaluations"] = evaluation_results

        # Generate overall assessment
        scores = [r["score"] for r in evaluation_results.values() if "score" in r]
        if scores:
            state.metadata["overall_score"] = sum(scores) / len(scores)

        return state


@scorer
class CompositeRetrievalScorer(Scorer):
    """
    Inspect AI scorer that combines multiple evaluation metrics
    """

    async def score(self, state: TaskState) -> Score:
        """
        Generate a composite score from evaluation results

        Args:
            state: Task state with evaluation results

        Returns:
            Composite score
        """
        evaluations = state.metadata.get("evaluations", {})

        if not evaluations:
            return Score(
                value=0.0,
                answer="no_evaluations",
                explanation="No evaluation results found",
            )

        # Calculate composite score
        scores = []
        explanations = []

        for eval_name, eval_result in evaluations.items():
            if "score" in eval_result:
                scores.append(eval_result["score"])
                explanations.append(f"{eval_name}: {eval_result['score']:.3f}")

        if not scores:
            return Score(
                value=0.0,
                answer="evaluation_failed",
                explanation="All evaluations failed",
            )

        composite_score = sum(scores) / len(scores)

        # Determine answer based on score
        if composite_score >= 0.8:
            answer = "excellent"
        elif composite_score >= 0.6:
            answer = "good"
        elif composite_score >= 0.4:
            answer = "fair"
        else:
            answer = "poor"

        return Score(
            value=composite_score,
            answer=answer,
            explanation=" | ".join(explanations),
            metadata={
                "individual_scores": {
                    name: result.get("score", -1)
                    for name, result in evaluations.items()
                }
            },
        )


@task
def evaluate_span_quality(
    span_data_path: Optional[str] = None, evaluator_names: Optional[List[str]] = None
) -> Task:
    """
    Inspect AI task for evaluating span quality

    This demonstrates how Inspect provides:
    1. Structured task definition
    2. Composable solvers for evaluation steps
    3. Flexible scoring mechanisms
    4. Integration with evaluation framework

    Args:
        span_data_path: Path to span data (or use recent spans)
        evaluator_names: Evaluators to use

    Returns:
        Inspect Task for span evaluation
    """
    # Create dataset from spans
    span_evaluator = SpanEvaluator()
    spans_df = span_evaluator.get_recent_spans(hours=24)

    # Convert spans to Inspect samples
    samples = []
    for _, span in spans_df.iterrows():
        attributes = span.get("attributes", {})
        outputs = span.get("outputs", {})

        sample = Sample(
            input=attributes.get("query", ""),
            target="",  # No target for reference-free evaluation
            metadata={
                "span_id": span.get("span_id"),
                "retrieval_results": outputs.get("results", []),
                "attributes": attributes,
            },
        )
        samples.append(sample)

    dataset = Dataset(name="span_evaluation", samples=samples)

    # Create solvers
    solvers = [
        RetrievalQualitySolver(
            evaluators={
                name: create_reference_free_evaluators()[name]
                for name in (evaluator_names or ["relevance", "diversity"])
            }
        )
    ]

    # Create scorer
    scorer = CompositeRetrievalScorer()

    return Task(
        dataset=dataset,
        solver=solvers,
        scorer=scorer,
        metadata={"task_type": "span_quality_evaluation", "num_spans": len(samples)},
    )


@task
def experiment_comparison_task(profiles: List[str], strategies: List[str]) -> Task:
    """
    Inspect AI task for comparing different configurations

    This shows how Inspect helps with:
    1. Systematic experiment design
    2. Controlled comparison of configurations
    3. Statistical analysis of results

    Args:
        profiles: Video processing profiles to test
        strategies: Ranking strategies to test

    Returns:
        Inspect Task for configuration comparison
    """
    # Create test queries
    test_queries = [
        "person in winter scene",
        "mechanical equipment",
        "abstract concept",
        "temporal event at specific time",
    ]

    samples = []
    for query in test_queries:
        for profile in profiles:
            for strategy in strategies:
                sample = Sample(
                    input=query,
                    target="",
                    metadata={
                        "profile": profile,
                        "strategy": strategy,
                        "config": f"{profile}_{strategy}",
                    },
                )
                samples.append(sample)

    dataset = Dataset(name="configuration_comparison", samples=samples)

    # This would use a custom solver that runs retrieval with different configs
    # and a scorer that compares results

    return Task(
        dataset=dataset,
        solver=[],  # Would add configuration-specific solvers
        scorer=CompositeRetrievalScorer(),
        metadata={
            "task_type": "configuration_comparison",
            "profiles": profiles,
            "strategies": strategies,
        },
    )


class InspectEvaluationOrchestrator:
    """
    Shows how Inspect AI helps orchestrate the evaluation process
    """

    def __init__(self):
        self.span_evaluator = SpanEvaluator()
        self.experiment_manager = VideoRetrievalExperiment()

    async def demonstrate_inspect_benefits(self) -> Dict[str, Any]:
        """
        Demonstrate how Inspect AI helps in the evaluation framework

        Returns:
            Summary of Inspect AI benefits
        """
        benefits = {
            "structured_evaluation": {
                "description": "Inspect provides structured task definitions",
                "example": "Tasks combine datasets, solvers, and scorers",
                "benefit": "Reproducible and modular evaluation pipelines",
            },
            "composable_solvers": {
                "description": "Solvers can be chained for complex evaluation logic",
                "example": "RetrievalQualitySolver runs multiple evaluators",
                "benefit": "Flexible evaluation workflows",
            },
            "flexible_scoring": {
                "description": "Scorers can implement custom metrics",
                "example": "CompositeRetrievalScorer combines multiple metrics",
                "benefit": "Domain-specific evaluation criteria",
            },
            "integration_ready": {
                "description": "Works seamlessly with Phoenix and other tools",
                "example": "Span data flows through Inspect tasks",
                "benefit": "Unified evaluation ecosystem",
            },
            "async_support": {
                "description": "Native async/await for efficient evaluation",
                "example": "Concurrent evaluation of multiple spans",
                "benefit": "Scalable evaluation pipelines",
            },
            "result_tracking": {
                "description": "Built-in result logging and analysis",
                "example": "Evaluation logs with detailed metrics",
                "benefit": "Easy debugging and improvement",
            },
        }

        # Run example evaluation to show it working
        task = evaluate_span_quality(evaluator_names=["relevance", "diversity"])

        # This would run the evaluation
        # eval_log = await eval_async(task, model="cogniverse")

        return {
            "inspect_benefits": benefits,
            "integration_points": {
                "span_evaluation": "Evaluate existing traces with reference-free metrics",
                "golden_dataset": "Compare against golden datasets when available",
                "experiments": "Run controlled experiments with Phoenix",
                "reporting": "Generate structured evaluation reports",
            },
            "key_advantage": "Inspect AI provides the scaffolding for building sophisticated evaluation pipelines that can grow with your needs",
        }


# Example usage showing the complete flow
async def demonstrate_complete_evaluation_flow():
    """
    Show how all components work together
    """
    orchestrator = InspectEvaluationOrchestrator()

    # 1. Evaluate existing spans (reference-free)
    logger.info("Step 1: Evaluating existing spans")
    span_results = await orchestrator.span_evaluator.run_evaluation_pipeline(
        hours=24, evaluator_names=["relevance", "diversity"]
    )

    # 2. Run Phoenix experiments with golden dataset
    logger.info("Step 2: Running Phoenix experiments")
    experiment_results = (
        await orchestrator.experiment_manager.run_comparison_experiments(
            profiles=["frame_based_colpali"],
            strategies=["binary_binary", "float_float"],
        )
    )

    # 3. Show Inspect AI benefits
    logger.info("Step 3: Demonstrating Inspect AI integration")
    inspect_benefits = await orchestrator.demonstrate_inspect_benefits()

    return {
        "span_evaluation": span_results,
        "experiments": experiment_results,
        "inspect_benefits": inspect_benefits,
    }
