#!/usr/bin/env python3
"""
Simple DSPy GLiNER Optimizer - No LLM Configuration Required

This is a simplified version that tests DSPy optimization without complex LLM setup.
It focuses on comparing optimization strategies rather than full DSPy features.
"""

import asyncio
import json
import os
import random
import sys
import time
from typing import Any

# Add src directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))

from src.common.config import get_config  # noqa: E402
from src.tools.query_analyzer import QueryAnalyzer  # noqa: E402


class SimpleGLiNEROptimizer:
    """
    Simple GLiNER optimizer that tests different configurations
    and compares with manual optimization approach.
    """

    def __init__(self):
        self.config = get_config()
        self.analyzer = QueryAnalyzer()

        # Test configurations (similar to DSPy's approach but simpler)
        self.test_configs = [
            {
                "name": "baseline",
                "model": "urchade/gliner_medium-v2.1",
                "labels": [
                    "video_content",
                    "text_content",
                    "temporal_phrase",
                    "date_value",
                    "content_request",
                ],
                "threshold": 0.3,
            },
            {
                "name": "comprehensive",
                "model": "urchade/gliner_large-v2.1",
                "labels": [
                    "video_content",
                    "visual_content",
                    "document_content",
                    "text_information",
                    "time_reference",
                    "date_pattern",
                    "search_intent",
                ],
                "threshold": 0.25,
            },
            {
                "name": "domain_focused",
                "model": "urchade/gliner_medium-v2.1",
                "labels": [
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
                "threshold": 0.2,
            },
            {
                "name": "minimal_fast",
                "model": "urchade/gliner_small-v2.1",
                "labels": [
                    "video_content",
                    "document_content",
                    "temporal_phrase",
                    "search_intent",
                ],
                "threshold": 0.4,
            },
            {
                "name": "entity_rich",
                "model": "urchade/gliner_large-v2.1",
                "labels": [
                    "multimedia_content",
                    "textual_content",
                    "visual_media",
                    "written_document",
                    "temporal_expression",
                    "time_indicator",
                    "search_query",
                    "content_request",
                ],
                "threshold": 0.35,
            },
        ]

    def load_test_queries(self) -> list[dict[str, Any]]:
        """Load test queries from file or use defaults."""
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

                        # Convert routing format
                        if expected_routing == "video":
                            needs_video, needs_text = True, False
                        elif expected_routing == "text":
                            needs_video, needs_text = False, True
                        elif expected_routing == "both":
                            needs_video, needs_text = True, True
                        else:
                            continue

                        # Handle temporal
                        if expected_temporal.lower() in ["none", "null", ""]:
                            expected_temporal = None

                        queries.append(
                            {
                                "query": query,
                                "needs_video": needs_video,
                                "needs_text": needs_text,
                                "temporal_pattern": expected_temporal,
                            }
                        )

                    except Exception:
                        continue

        except FileNotFoundError:
            print("⚠️ test_queries.txt not found, using default examples")
            queries = [
                {
                    "query": "Show me videos of robots",
                    "needs_video": True,
                    "needs_text": False,
                    "temporal_pattern": None,
                },
                {
                    "query": "Find documents about AI",
                    "needs_video": False,
                    "needs_text": True,
                    "temporal_pattern": None,
                },
                {
                    "query": "Search for content about the project",
                    "needs_video": True,
                    "needs_text": True,
                    "temporal_pattern": None,
                },
                {
                    "query": "Videos from yesterday",
                    "needs_video": True,
                    "needs_text": False,
                    "temporal_pattern": "yesterday",
                },
                {
                    "query": "Find reports from last week",
                    "needs_video": False,
                    "needs_text": True,
                    "temporal_pattern": "last_week",
                },
                {
                    "query": "Show me clips about machine learning",
                    "needs_video": True,
                    "needs_text": False,
                    "temporal_pattern": None,
                },
                {
                    "query": "Search documents about neural networks",
                    "needs_video": False,
                    "needs_text": True,
                    "temporal_pattern": None,
                },
                {
                    "query": "Find content from today",
                    "needs_video": True,
                    "needs_text": True,
                    "temporal_pattern": "today",
                },
                {
                    "query": "Show videos and reports about deep learning",
                    "needs_video": True,
                    "needs_text": True,
                    "temporal_pattern": None,
                },
                {
                    "query": "Search for footage from this morning",
                    "needs_video": True,
                    "needs_text": False,
                    "temporal_pattern": "this_morning",
                },
            ]

        return queries

    async def evaluate_configuration(
        self, config: dict[str, Any], test_queries: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Evaluate a specific GLiNER configuration."""
        # Store original configuration
        original_labels = self.analyzer.gliner_labels
        original_threshold = self.analyzer.gliner_threshold

        try:
            # Apply configuration
            self.analyzer.gliner_labels = config["labels"]
            self.analyzer.gliner_threshold = config["threshold"]

            # Load model
            success = self.analyzer.switch_gliner_model(config["model"])
            if not success:
                return {"accuracy": 0.0, "error": "Failed to load model"}

            self.analyzer.set_mode("gliner_only")

            # Test queries
            correct = 0
            total = len(test_queries)
            detailed_results = []

            start_time = time.time()

            for query_data in test_queries:
                try:
                    result = await self.analyzer.analyze_query(query_data["query"])

                    video_correct = (
                        result.get("needs_video_search", False)
                        == query_data["needs_video"]
                    )
                    text_correct = (
                        result.get("needs_text_search", False)
                        == query_data["needs_text"]
                    )
                    temporal_correct = (
                        result.get("temporal_pattern") == query_data["temporal_pattern"]
                    )

                    # Calculate score (3 components)
                    score = sum([video_correct, text_correct, temporal_correct]) / 3.0

                    if score == 1.0:
                        correct += 1

                    detailed_results.append(
                        {
                            "query": query_data["query"],
                            "expected": query_data,
                            "actual": result,
                            "score": score,
                            "video_correct": video_correct,
                            "text_correct": text_correct,
                            "temporal_correct": temporal_correct,
                        }
                    )

                except Exception as e:
                    detailed_results.append(
                        {"query": query_data["query"], "error": str(e), "score": 0.0}
                    )
                    continue

            eval_time = time.time() - start_time
            accuracy = correct / total if total > 0 else 0.0

            return {
                "accuracy": accuracy,
                "eval_time": eval_time,
                "correct": correct,
                "total": total,
                "detailed_results": detailed_results,
            }

        finally:
            # Restore original configuration
            self.analyzer.gliner_labels = original_labels
            self.analyzer.gliner_threshold = original_threshold

    async def run_optimization(self) -> dict[str, Any]:
        """Run optimization across all test configurations."""
        print("🚀 Simple GLiNER Optimization")
        print("=" * 50)

        # Load test data
        test_queries = self.load_test_queries()
        print(f"📚 Testing with {len(test_queries)} queries")

        # Split into train/test for more realistic evaluation
        random.shuffle(test_queries)
        split_point = int(0.8 * len(test_queries))
        train_queries = test_queries[:split_point]
        test_queries = test_queries[split_point:]

        print(f"📊 Train: {len(train_queries)}, Test: {len(test_queries)} queries")

        results = {}

        # Test each configuration
        for config in self.test_configs:
            print(f"\n🔧 Testing configuration: {config['name']}")
            print(f"   Model: {config['model'].split('/')[-1]}")
            print(f"   Labels: {len(config['labels'])} labels")
            print(f"   Threshold: {config['threshold']}")

            # Evaluate on test set
            result = await self.evaluate_configuration(config, test_queries)

            if "error" in result:
                print(f"   ❌ {result['error']}")
                results[config["name"]] = result
                continue

            accuracy = result["accuracy"]
            eval_time = result["eval_time"]

            print(
                f"   📈 Accuracy: {accuracy:.1%} ({result['correct']}/{result['total']})"
            )
            print(f"   ⏱️ Eval time: {eval_time:.2f}s")

            # Store results
            results[config["name"]] = {
                "config": config,
                "accuracy": accuracy,
                "eval_time": eval_time,
                "correct": result["correct"],
                "total": result["total"],
                "detailed_results": result["detailed_results"],
            }

        return results

    def print_optimization_summary(self, results: dict[str, Any]):
        """Print comprehensive optimization results."""
        print("\n" + "=" * 60)
        print("🏆 SIMPLE GLINER OPTIMIZATION RESULTS")
        print("=" * 60)

        # Filter out errors
        valid_results = {
            k: v for k, v in results.items() if "accuracy" in v and "error" not in v
        }

        if not valid_results:
            print("❌ No valid results")
            return

        # Sort by accuracy
        sorted_results = sorted(
            valid_results.items(), key=lambda x: x[1]["accuracy"], reverse=True
        )

        print("📊 Configuration Performance Rankings:")
        for i, (name, result) in enumerate(sorted_results, 1):
            config = result["config"]
            print(
                f"{i:2d}. {name:<15} {result['accuracy']:>6.1%} "
                f"({result['eval_time']:>5.2f}s) - {config['model'].split('/')[-1]}"
            )

        # Best result details
        best_name, best_result = sorted_results[0]
        best_config = best_result["config"]

        print(f"\n🥇 Best Configuration: {best_name}")
        print(f"   📈 Accuracy: {best_result['accuracy']:.1%}")
        print(f"   🤖 Model: {best_config['model']}")
        print(f"   🏷️ Labels: {len(best_config['labels'])} labels")
        print(f"   🎯 Threshold: {best_config['threshold']}")
        print(f"   ⏱️ Eval time: {best_result['eval_time']:.2f}s")

        # Performance analysis
        accuracies = [r["accuracy"] for r in valid_results.values()]
        eval_times = [r["eval_time"] for r in valid_results.values()]

        print("\n📈 Performance Analysis:")
        print(f"   • Best accuracy: {max(accuracies):.1%}")
        print(f"   • Average accuracy: {sum(accuracies)/len(accuracies):.1%}")
        print(f"   • Accuracy range: {max(accuracies) - min(accuracies):.1%}")
        print(f"   • Fastest eval: {min(eval_times):.2f}s")
        print(f"   • Slowest eval: {max(eval_times):.2f}s")

        # Show some detailed results for best config
        if best_result["detailed_results"]:
            print(f"\n🔍 Sample Results for Best Config ({best_name}):")
            for i, detail in enumerate(best_result["detailed_results"][:3]):
                if "error" not in detail:
                    print(f"   Query: \"{detail['query']}\"")
                    print(
                        f"   Score: {detail['score']:.2f} (V:{detail['video_correct']}, T:{detail['text_correct']}, Temp:{detail['temporal_correct']})"
                    )
                    print()

        return sorted_results


async def main():
    """Run simple GLiNER optimization."""
    optimizer = SimpleGLiNEROptimizer()

    print("🔧 Simple GLiNER Optimization")
    print(f"🧪 Testing {len(optimizer.test_configs)} configurations")

    # Run optimization
    results = await optimizer.run_optimization()

    # Print summary
    sorted_results = optimizer.print_optimization_summary(results)

    # Save results
    try:
        with open("simple_gliner_optimization_results.json", "w") as f:
            # Remove detailed results for JSON serialization
            json_results = {}
            for name, result in results.items():
                if "detailed_results" in result:
                    json_result = result.copy()
                    json_result.pop("detailed_results", None)
                    json_results[name] = json_result
                else:
                    json_results[name] = result

            json.dump(json_results, f, indent=2)

        print("\n💾 Results saved to: simple_gliner_optimization_results.json")

    except Exception as e:
        print(f"⚠️ Could not save results: {e}")

    # Compare with manual optimization (if results exist)
    try:
        manual_results_file = "gliner_optimization_results.json"
        if os.path.exists(manual_results_file):
            with open(manual_results_file) as f:
                manual_results = json.load(f)

            print("\n🔄 Comparison with Manual Optimization:")
            print(f"   Manual best: {max(r['accuracy'] for r in manual_results):.1%}")
            print(f"   Simple best: {sorted_results[0][1]['accuracy']:.1%}")

            # Show which approach is better
            manual_best = max(r["accuracy"] for r in manual_results)
            simple_best = sorted_results[0][1]["accuracy"]

            if simple_best > manual_best:
                print(
                    f"   🏆 Simple optimization wins by {simple_best - manual_best:.1%}"
                )
            else:
                print(
                    f"   🏆 Manual optimization wins by {manual_best - simple_best:.1%}"
                )

    except Exception as e:
        print(f"⚠️ Could not compare with manual results: {e}")


if __name__ == "__main__":
    asyncio.run(main())
