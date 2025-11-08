#!/usr/bin/env python3
# tests/gliner_optimizer.py
"""
Advanced GLiNER Optimization Framework
Tries multiple optimization strategies to improve GLiNER performance beyond baseline.
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import dspy

# Add src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

from cogniverse_agents.tools.query_analyzer import QueryAnalyzer  # noqa: E402
from cogniverse_core.config.utils import (
    create_default_config_manager,  # noqa: E402
    get_config,
)


@dataclass
class OptimizationResult:
    """Results from an optimization experiment."""

    strategy: str
    accuracy: float
    config: dict[str, Any]
    runtime: float
    details: dict[str, Any]


class QueryExample(dspy.Example):
    """DSPy-compatible query training example."""

    def __init__(self, query, needs_video, needs_text, temporal_pattern=None):
        super().__init__(
            query=query,
            needs_video_search=needs_video,
            needs_text_search=needs_text,
            temporal_pattern=temporal_pattern if temporal_pattern else "null",
        )
        self.query = query
        self.needs_video = needs_video
        self.needs_text = needs_text
        self.temporal_pattern = temporal_pattern


def create_query_example(query, needs_video, needs_text, temporal_pattern=None):
    """Create a properly formatted DSPy example."""
    example = dspy.Example(
        query=query,
        needs_video_search=needs_video,
        needs_text_search=needs_text,
        temporal_pattern=temporal_pattern if temporal_pattern else "null",
    ).with_inputs("query")

    # Add compatibility attributes for our metric
    example.needs_video = needs_video
    example.needs_text = needs_text
    example.temporal_pattern = temporal_pattern

    return example


class GLiNERRouterSignature(dspy.Signature):
    """GLiNER-based routing signature for optimization."""

    query = dspy.InputField(
        desc="User query to analyze for content routing and temporal patterns"
    )
    needs_video_search = dspy.OutputField(
        desc="True if query needs video/visual content search, False otherwise"
    )
    needs_text_search = dspy.OutputField(
        desc="True if query needs text/document search, False otherwise"
    )
    temporal_pattern = dspy.OutputField(
        desc="Temporal pattern extracted from query (e.g., 'yesterday', 'last_week') or 'null' if none"
    )
    reasoning = dspy.OutputField(
        desc="Brief explanation of routing and temporal decisions"
    )


class GLiNERRouter(dspy.Module):
    """GLiNER-based router for DSPy optimization."""

    def __init__(self, analyzer: QueryAnalyzer):
        super().__init__()
        self.analyzer = analyzer
        self.predictor = dspy.Predict(GLiNERRouterSignature)

    def forward(self, query: str):
        """Process a query using GLiNER and return routing decision."""
        try:
            # Use GLiNER for analysis
            self.analyzer.set_mode("gliner_only")
            result = asyncio.run(self.analyzer.analyze_query(query))

            # Convert to DSPy prediction format
            return dspy.Prediction(
                needs_video_search=result.get("needs_video_search", False),
                needs_text_search=result.get("needs_text_search", False),
                temporal_pattern=result.get("temporal_pattern", "null"),
                reasoning=f"GLiNER entities: {len(result.get('gliner_entities', []))}",
            )

        except Exception as e:
            print(f"⚠️ GLiNER routing failed: {e}")
            return dspy.Prediction(
                needs_video_search=False,
                needs_text_search=True,
                temporal_pattern="null",
                reasoning="Error in GLiNER processing",
            )


class GLiNERMetric:
    """Metric for evaluating GLiNER routing performance."""

    def __call__(self, example, prediction, trace=None) -> float:
        """Calculate accuracy score for a prediction."""
        score = 0.0
        total_weight = 3.0

        # Get expected values from example
        expected_video = getattr(
            example, "needs_video_search", getattr(example, "needs_video", False)
        )
        expected_text = getattr(
            example, "needs_text_search", getattr(example, "needs_text", False)
        )
        expected_temporal = getattr(example, "temporal_pattern", None)

        # Get actual values from prediction
        actual_video = getattr(prediction, "needs_video_search", False)
        actual_text = getattr(prediction, "needs_text_search", False)
        actual_temporal = getattr(prediction, "temporal_pattern", None)

        # Handle null temporal pattern
        if expected_temporal == "null":
            expected_temporal = None
        if actual_temporal == "null":
            actual_temporal = None

        # Routing accuracy (2 components)
        if actual_video == expected_video:
            score += 1.0
        if actual_text == expected_text:
            score += 1.0

        # Temporal accuracy (1 component)
        if actual_temporal == expected_temporal:
            score += 1.0

        return score / total_weight


class AdvancedGLiNEROptimizer:
    """Advanced GLiNER optimizer with multiple strategies."""

    def __init__(self):
        config_manager = create_default_config_manager()
        self.config = get_config(tenant_id="default", config_manager=config_manager)
        self.analyzer = QueryAnalyzer()
        self.results = []

        # Extended label candidates for comprehensive optimization
        self.label_sets = {
            "baseline": [
                "video_content",
                "text_content",
                "temporal_phrase",
                "date_value",
                "content_request",
            ],
            "specific": [
                "video_content",
                "visual_content",
                "document_content",
                "text_information",
                "time_reference",
                "date_pattern",
                "search_intent",
            ],
            "comprehensive": [
                "video_content",
                "visual_content",
                "media_content",
                "document_content",
                "text_information",
                "written_material",
                "time_reference",
                "temporal_phrase",
                "search_intent",
                "content_type",
            ],
            "minimal": [
                "video_content",
                "document_content",
                "temporal_phrase",
                "search_intent",
            ],
            "domain_focused": [
                "video",
                "document",
                "footage",
                "report",
                "clip",
                "article",
                "yesterday",
                "today",
                "week",
                "month",
                "find",
                "search",
            ],
            "entity_rich": [
                "multimedia_content",
                "textual_content",
                "visual_media",
                "written_document",
                "temporal_expression",
                "time_indicator",
                "search_query",
                "content_request",
                "information_need",
                "media_type",
            ],
            "semantic": [
                "visual_search_intent",
                "textual_search_intent",
                "temporal_constraint",
                "content_specification",
                "media_preference",
            ],
            "fine_grained": [
                "video_clip",
                "video_recording",
                "document_file",
                "text_article",
                "time_yesterday",
                "time_week",
                "search_find",
                "content_both",
            ],
        }

        self.threshold_ranges = {
            "conservative": [0.1, 0.2, 0.3],
            "balanced": [0.3, 0.4, 0.5],
            "strict": [0.5, 0.6, 0.7],
            "adaptive": [0.15, 0.25, 0.35, 0.45],
            "fine": [0.12, 0.18, 0.22, 0.28, 0.32, 0.38],
        }

        # Available GLiNER models for testing
        self.available_models = [
            "urchade/gliner_small-v2.1",
            "urchade/gliner_medium-v2.1",
            "urchade/gliner_large-v2.1",
            "urchade/gliner_multi-v2.1",
        ]

    def load_training_examples(self) -> list[QueryExample]:
        """Load training examples from test queries."""
        queries = []
        test_file = os.path.join(os.path.dirname(__file__), "test_queries.txt")

        try:
            with open(test_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    try:
                        parts = line.split(",", 2)
                        if len(parts) != 3:
                            continue

                        query = parts[0].strip()
                        expected_routing = parts[1].strip()
                        expected_temporal = parts[2].strip()

                        # Convert routing to boolean format
                        if expected_routing == "video":
                            needs_video, needs_text = True, False
                        elif expected_routing == "text":
                            needs_video, needs_text = False, True
                        elif expected_routing == "both":
                            needs_video, needs_text = True, True
                        else:
                            continue

                        # Handle temporal pattern
                        if expected_temporal.lower() in ["none", "null", ""]:
                            expected_temporal = None

                        queries.append(
                            QueryExample(
                                query=query,
                                needs_video=needs_video,
                                needs_text=needs_text,
                                temporal_pattern=expected_temporal,
                            )
                        )

                    except Exception:
                        continue
        except FileNotFoundError:
            print("⚠️ test_queries.txt not found, using default examples")
            # Fallback examples
            queries = [
                QueryExample("Show me videos of cats", True, False, None),
                QueryExample("Find documents about AI", False, True, None),
                QueryExample("Search for content about the project", True, True, None),
                QueryExample("Videos from yesterday", True, False, "yesterday"),
                QueryExample("Find reports from last week", False, True, "last_week"),
            ]

        return queries

    async def evaluate_configuration(
        self,
        labels: list[str],
        threshold: float,
        model_name: str,
        test_examples: list[QueryExample],
    ) -> float:
        """Evaluate a specific GLiNER configuration."""
        # Store original configuration
        original_labels = self.analyzer.gliner_labels
        original_threshold = self.analyzer.gliner_threshold

        try:
            # Apply new configuration
            self.analyzer.gliner_labels = labels
            self.analyzer.gliner_threshold = threshold

            # Load specified model
            success = self.analyzer.switch_gliner_model(model_name)
            if not success:
                return 0.0

            self.analyzer.set_mode("gliner_only")

            correct = 0
            total = len(test_examples)

            for example in test_examples:
                try:
                    result = await self.analyzer.analyze_query(example.query)

                    video_correct = (
                        result.get("needs_video_search", False) == example.needs_video
                    )
                    text_correct = (
                        result.get("needs_text_search", False) == example.needs_text
                    )
                    temporal_correct = (
                        result.get("temporal_pattern") == example.temporal_pattern
                    )

                    if video_correct and text_correct and temporal_correct:
                        correct += 1

                except Exception:
                    continue

            return correct / total if total > 0 else 0.0

        finally:
            # Restore original configuration
            self.analyzer.gliner_labels = original_labels
            self.analyzer.gliner_threshold = original_threshold

    async def grid_search_optimization(
        self, test_examples: list[QueryExample]
    ) -> OptimizationResult:
        """Comprehensive grid search across all configurations."""
        print("🔍 Running Grid Search Optimization")
        print("-" * 50)

        start_time = time.time()
        best_score = 0.0
        best_config = None
        configurations_tested = 0

        for model_name in self.available_models:
            print(f"\n🤖 Testing model: {model_name.split('/')[-1]}")

            for label_set_name, labels in self.label_sets.items():
                print(f"  📋 Label set '{label_set_name}': {len(labels)} labels")

                for threshold_range_name, thresholds in self.threshold_ranges.items():
                    for threshold in thresholds:
                        configurations_tested += 1

                        try:
                            score = await self.evaluate_configuration(
                                labels, threshold, model_name, test_examples
                            )

                            print(
                                f"    🎯 {threshold_range_name} threshold {threshold}: {score:.1%}"
                            )

                            if score > best_score:
                                best_score = score
                                best_config = {
                                    "model": model_name,
                                    "labels": labels,
                                    "threshold": threshold,
                                    "label_set": label_set_name,
                                    "threshold_range": threshold_range_name,
                                }
                                print(f"    🌟 New best: {score:.1%}")

                        except Exception as e:
                            print(f"    ❌ Error: {e}")

        runtime = time.time() - start_time

        return OptimizationResult(
            strategy="grid_search",
            accuracy=best_score,
            config=best_config,
            runtime=runtime,
            details={"configurations_tested": configurations_tested},
        )

    async def focused_optimization(
        self, test_examples: list[QueryExample]
    ) -> OptimizationResult:
        """Focused optimization on promising configurations."""
        print("🎯 Running Focused Optimization")
        print("-" * 50)

        start_time = time.time()

        # Test promising combinations based on domain knowledge
        promising_configs = [
            {
                "model": "urchade/gliner_medium-v2.1",
                "labels": self.label_sets["comprehensive"],
                "threshold": 0.25,
            },
            {
                "model": "urchade/gliner_large-v2.1",
                "labels": self.label_sets["specific"],
                "threshold": 0.3,
            },
            {
                "model": "urchade/gliner_medium-v2.1",
                "labels": self.label_sets["domain_focused"],
                "threshold": 0.2,
            },
            {
                "model": "urchade/gliner_large-v2.1",
                "labels": self.label_sets["entity_rich"],
                "threshold": 0.35,
            },
            {
                "model": "urchade/gliner_medium-v2.1",
                "labels": self.label_sets["fine_grained"],
                "threshold": 0.15,
            },
        ]

        best_score = 0.0
        best_config = None

        for i, config in enumerate(promising_configs):
            try:
                score = await self.evaluate_configuration(
                    config["labels"],
                    config["threshold"],
                    config["model"],
                    test_examples,
                )

                print(
                    f"  Config {i+1}: {score:.1%} ({config['model'].split('/')[-1]}, {len(config['labels'])} labels, {config['threshold']} threshold)"
                )

                if score > best_score:
                    best_score = score
                    best_config = config
                    print(f"  🌟 New best: {score:.1%}")

            except Exception as e:
                print(f"  ❌ Error with config {i+1}: {e}")

        runtime = time.time() - start_time

        return OptimizationResult(
            strategy="focused",
            accuracy=best_score,
            config=best_config,
            runtime=runtime,
            details={"configs_tested": len(promising_configs)},
        )

    async def adaptive_optimization(
        self, test_examples: list[QueryExample]
    ) -> OptimizationResult:
        """Adaptive optimization that learns from results."""
        print("🧠 Running Adaptive Optimization")
        print("-" * 50)

        start_time = time.time()

        # Start with baseline and adapt based on performance
        current_config = {
            "model": "urchade/gliner_medium-v2.1",
            "labels": self.label_sets["baseline"],
            "threshold": 0.3,
        }

        best_score = await self.evaluate_configuration(
            current_config["labels"],
            current_config["threshold"],
            current_config["model"],
            test_examples,
        )
        best_config = current_config.copy()

        print(f"  Baseline: {best_score:.1%}")

        # Adaptive iterations
        iterations = 5
        for iteration in range(iterations):
            print(f"\n  Iteration {iteration + 1}/{iterations}")

            # Try different models if score is low
            if best_score < 0.6:
                for model in ["urchade/gliner_large-v2.1", "urchade/gliner_multi-v2.1"]:
                    test_config = current_config.copy()
                    test_config["model"] = model

                    score = await self.evaluate_configuration(
                        test_config["labels"],
                        test_config["threshold"],
                        test_config["model"],
                        test_examples,
                    )

                    print(f"    Model {model.split('/')[-1]}: {score:.1%}")

                    if score > best_score:
                        best_score = score
                        best_config = test_config.copy()
                        print(f"    🌟 Better model found: {score:.1%}")

            # Try different label sets
            for label_set_name, labels in self.label_sets.items():
                if len(labels) != len(
                    best_config["labels"]
                ):  # Try different complexity
                    test_config = best_config.copy()
                    test_config["labels"] = labels

                    score = await self.evaluate_configuration(
                        test_config["labels"],
                        test_config["threshold"],
                        test_config["model"],
                        test_examples,
                    )

                    print(f"    Labels {label_set_name}: {score:.1%}")

                    if score > best_score:
                        best_score = score
                        best_config = test_config.copy()
                        print(f"    🌟 Better labels found: {score:.1%}")
                        break

            # Fine-tune threshold
            current_threshold = best_config["threshold"]
            for delta in [-0.1, -0.05, 0.05, 0.1]:
                new_threshold = max(0.1, min(0.7, current_threshold + delta))
                if new_threshold != current_threshold:
                    test_config = best_config.copy()
                    test_config["threshold"] = new_threshold

                    score = await self.evaluate_configuration(
                        test_config["labels"],
                        test_config["threshold"],
                        test_config["model"],
                        test_examples,
                    )

                    if score > best_score:
                        best_score = score
                        best_config = test_config.copy()
                        print(f"    🌟 Better threshold {new_threshold}: {score:.1%}")
                        break

        runtime = time.time() - start_time

        return OptimizationResult(
            strategy="adaptive",
            accuracy=best_score,
            config=best_config,
            runtime=runtime,
            details={"iterations": iterations},
        )

    async def run_comprehensive_optimization(self) -> list[OptimizationResult]:
        """Run all optimization strategies and compare results."""
        print("🚀 Comprehensive GLiNER Optimization")
        print("=" * 80)

        # Load data
        examples = self.load_training_examples()
        # Use a subset for faster testing
        test_examples = examples[:15] if len(examples) > 15 else examples

        print(f"📚 Dataset: {len(test_examples)} test examples")

        # Run optimization strategies
        print("\n🔬 Running optimization strategies...")

        # 1. Grid Search (comprehensive)
        grid_result = await self.grid_search_optimization(test_examples)
        self.results.append(grid_result)

        # 2. Focused optimization (promising configs)
        focused_result = await self.focused_optimization(test_examples)
        self.results.append(focused_result)

        # 3. Adaptive optimization
        adaptive_result = await self.adaptive_optimization(test_examples)
        self.results.append(adaptive_result)

        return self.results

    def print_optimization_summary(self):
        """Print comprehensive optimization results."""
        print("\n" + "=" * 80)
        print("🏆 GLINER OPTIMIZATION RESULTS")
        print("=" * 80)

        if not self.results:
            print("❌ No optimization results available")
            return

        # Sort by accuracy
        sorted_results = sorted(self.results, key=lambda x: x.accuracy, reverse=True)

        print("📊 Strategy Performance Rankings:")
        for i, result in enumerate(sorted_results, 1):
            print(
                f"{i:2d}. {result.strategy:<15} {result.accuracy:>6.1%} ({result.runtime:>5.1f}s)"
            )

        # Best result details
        best = sorted_results[0]
        print(f"\n🥇 Best Strategy: {best.strategy}")
        print(f"   📈 Accuracy: {best.accuracy:.1%}")
        print(f"   ⏱️  Runtime: {best.runtime:.1f}s")

        if best.config:
            print("   ⚙️  Best Config:")
            print(f"      • Model: {best.config.get('model', 'N/A')}")
            print(f"      • Labels: {len(best.config.get('labels', []))} labels")
            print(f"      • Threshold: {best.config.get('threshold', 'N/A')}")
            print(f"      • Label Set: {best.config.get('label_set', 'N/A')}")

        print(f"   📝 Details: {best.details}")

        # Performance analysis
        print("\n📈 Performance Analysis:")
        accuracies = [r.accuracy for r in self.results]
        if accuracies:
            print(f"   • Best:    {max(accuracies):.1%}")
            print(f"   • Average: {sum(accuracies)/len(accuracies):.1%}")
            print(f"   • Range:   {max(accuracies) - min(accuracies):.1%}")

        # Speed analysis
        runtimes = [r.runtime for r in self.results]
        if runtimes:
            print("\n⏱️  Runtime Analysis:")
            print(f"   • Fastest: {min(runtimes):.1f}s")
            print(f"   • Slowest: {max(runtimes):.1f}s")
            print(f"   • Average: {sum(runtimes)/len(runtimes):.1f}s")


async def main():
    """Run comprehensive GLiNER optimization."""
    optimizer = AdvancedGLiNEROptimizer()

    print(f"🔧 Configured {len(optimizer.label_sets)} label sets")
    print(f"🎯 Testing {len(optimizer.available_models)} models")

    # Run all optimization strategies
    results = await optimizer.run_comprehensive_optimization()

    # Print summary
    optimizer.print_optimization_summary()

    # Save results for analysis
    results_file = "gliner_optimization_results.json"
    try:
        with open(results_file, "w") as f:
            json.dump(
                [
                    {
                        "strategy": r.strategy,
                        "accuracy": r.accuracy,
                        "config": r.config,
                        "runtime": r.runtime,
                        "details": r.details,
                    }
                    for r in results
                ],
                f,
                indent=2,
            )

        print(f"\n💾 Results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️ Could not save results: {e}")


if __name__ == "__main__":
    asyncio.run(main())
