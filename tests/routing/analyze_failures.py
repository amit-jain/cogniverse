#!/usr/bin/env python3
# tests/analyze_failures.py
"""
Analyze current failures to identify optimization opportunities.
"""

import asyncio
import sys
import os
from collections import defaultdict

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from tools.query_analyzer import QueryAnalyzer


def load_test_queries(filename="test_queries.txt"):
    """Load test queries from file."""
    queries = []
    test_file = os.path.join(os.path.dirname(__file__), filename)

    with open(test_file, "r") as f:
        for line_num, line in enumerate(f, 1):
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

                queries.append((query, needs_video, needs_text, expected_temporal))

            except Exception as e:
                print(f"⚠️ Error parsing line {line_num}: {e}")

    return queries


async def analyze_mode_failures(analyzer, mode_name, test_queries):
    """Analyze failures for a specific mode."""
    print(f"\n🔍 Analyzing {mode_name.upper()} failures")
    print("-" * 50)

    analyzer.set_mode(mode_name.lower().replace("-", "_"))

    routing_failures = []
    temporal_failures = []
    overall_failures = []

    for query, expected_video, expected_text, expected_temporal in test_queries:
        try:
            result = await analyzer.analyze_query(query)

            # Check results
            actual_video = result.get("needs_video_search", False)
            actual_text = result.get("needs_text_search", False)
            actual_temporal = result.get("temporal_pattern")

            # Analyze failures
            video_correct = actual_video == expected_video
            text_correct = actual_text == expected_text
            temporal_correct = actual_temporal == expected_temporal

            if not (video_correct and text_correct):
                routing_failures.append(
                    {
                        "query": query,
                        "expected": (expected_video, expected_text),
                        "actual": (actual_video, actual_text),
                        "video_error": not video_correct,
                        "text_error": not text_correct,
                    }
                )

            if not temporal_correct:
                temporal_failures.append(
                    {
                        "query": query,
                        "expected": expected_temporal,
                        "actual": actual_temporal,
                    }
                )

            if not (video_correct and text_correct and temporal_correct):
                overall_failures.append(
                    {
                        "query": query,
                        "routing_error": not (video_correct and text_correct),
                        "temporal_error": not temporal_correct,
                        "inference_method": result.get("inference_method", "unknown"),
                        "entities": result.get("gliner_entities", []),
                        "error": result.get("error"),
                    }
                )

        except Exception as e:
            overall_failures.append({"query": query, "exception": str(e)})

    return {
        "mode": mode_name,
        "routing_failures": routing_failures,
        "temporal_failures": temporal_failures,
        "overall_failures": overall_failures,
        "total_queries": len(test_queries),
    }


def analyze_routing_patterns(failures):
    """Analyze patterns in routing failures."""
    print(f"\n📊 Routing Failure Patterns ({len(failures)} failures)")
    print("-" * 30)

    # Categorize failures
    video_false_positives = []  # Should be False, got True
    video_false_negatives = []  # Should be True, got False
    text_false_positives = []
    text_false_negatives = []

    for failure in failures:
        expected_video, expected_text = failure["expected"]
        actual_video, actual_text = failure["actual"]

        if expected_video and not actual_video:
            video_false_negatives.append(failure)
        elif not expected_video and actual_video:
            video_false_positives.append(failure)

        if expected_text and not actual_text:
            text_false_negatives.append(failure)
        elif not expected_text and actual_text:
            text_false_positives.append(failure)

    print(f"Video False Negatives (missed video): {len(video_false_negatives)}")
    for f in video_false_negatives[:3]:  # Show first 3
        print(f"  - '{f['query']}'")

    print(f"Video False Positives (wrong video): {len(video_false_positives)}")
    for f in video_false_positives[:3]:
        print(f"  - '{f['query']}'")

    print(f"Text False Negatives (missed text): {len(text_false_negatives)}")
    for f in text_false_negatives[:3]:
        print(f"  - '{f['query']}'")

    print(f"Text False Positives (wrong text): {len(text_false_positives)}")
    for f in text_false_positives[:3]:
        print(f"  - '{f['query']}'")


def analyze_temporal_patterns(failures):
    """Analyze patterns in temporal failures."""
    print(f"\n📅 Temporal Failure Patterns ({len(failures)} failures)")
    print("-" * 30)

    missed_patterns = defaultdict(list)
    wrong_patterns = defaultdict(list)

    for failure in failures:
        expected = failure["expected"]
        actual = failure["actual"]

        if expected and not actual:
            missed_patterns[expected].append(failure["query"])
        elif actual and not expected:
            wrong_patterns[actual].append(failure["query"])
        elif expected and actual and expected != actual:
            print(
                f"Pattern mismatch: '{failure['query']}' expected {expected}, got {actual}"
            )

    print("Most missed temporal patterns:")
    for pattern, queries in sorted(
        missed_patterns.items(), key=lambda x: len(x[1]), reverse=True
    )[:5]:
        print(f"  {pattern}: {len(queries)} queries")
        for query in queries[:2]:  # Show examples
            print(f"    - '{query}'")

    print("Most wrong temporal patterns:")
    for pattern, queries in sorted(
        wrong_patterns.items(), key=lambda x: len(x[1]), reverse=True
    )[:3]:
        print(f"  {pattern}: {len(queries)} queries")


async def main():
    """Analyze failures across different modes."""
    print("🔬 Query Analysis Failure Analysis")
    print("=" * 60)

    # Load test queries
    test_queries = load_test_queries()
    print(f"📚 Loaded {len(test_queries)} test queries")

    analyzer = QueryAnalyzer()

    # Analyze different modes
    modes_to_test = ["llm_only", "gliner_only", "hybrid"]

    all_results = {}

    for mode in modes_to_test:
        try:
            if mode == "gliner_only":
                # Load medium model for GLiNER testing
                success = analyzer.switch_gliner_model("urchade/gliner_medium-v2.1")
                if not success:
                    print(f"⚠️ Skipping {mode} - model loading failed")
                    continue

            results = await analyze_mode_failures(analyzer, mode, test_queries)
            all_results[mode] = results

            print(f"\n📊 {mode.upper()} Summary:")
            print(
                f"   Routing failures: {len(results['routing_failures'])}/{results['total_queries']}"
            )
            print(
                f"   Temporal failures: {len(results['temporal_failures'])}/{results['total_queries']}"
            )
            print(
                f"   Overall failures: {len(results['overall_failures'])}/{results['total_queries']}"
            )

            # Detailed analysis
            if results["routing_failures"]:
                analyze_routing_patterns(results["routing_failures"])

            if results["temporal_failures"]:
                analyze_temporal_patterns(results["temporal_failures"])

        except Exception as e:
            print(f"❌ Error analyzing {mode}: {e}")

    # Compare modes
    print("\n" + "=" * 60)
    print("🆚 MODE COMPARISON")
    print("=" * 60)

    for mode, results in all_results.items():
        routing_acc = (
            (results["total_queries"] - len(results["routing_failures"]))
            / results["total_queries"]
            * 100
        )
        temporal_acc = (
            (results["total_queries"] - len(results["temporal_failures"]))
            / results["total_queries"]
            * 100
        )
        overall_acc = (
            (results["total_queries"] - len(results["overall_failures"]))
            / results["total_queries"]
            * 100
        )

        print(
            f"{mode:<12} | Routing: {routing_acc:5.1f}% | Temporal: {temporal_acc:5.1f}% | Overall: {overall_acc:5.1f}%"
        )

    # Suggestions for improvement
    print("\n💡 IMPROVEMENT SUGGESTIONS")
    print("=" * 30)

    if "llm_only" in all_results:
        llm_failures = all_results["llm_only"]["overall_failures"]
        llm_errors = [f for f in llm_failures if f.get("error")]
        print("LLM Improvements needed:")
        print(f"  - Fix {len(llm_errors)} parsing/timeout errors")
        print(
            f"  - Improve prompt for {len(all_results['llm_only']['routing_failures'])} routing errors"
        )
        print(
            f"  - Better temporal patterns for {len(all_results['llm_only']['temporal_failures'])} errors"
        )

    if "gliner_only" in all_results:
        print("GLiNER Improvements needed:")
        print(
            f"  - Better labels/threshold for {len(all_results['gliner_only']['routing_failures'])} routing errors"
        )
        print(
            f"  - Enhanced temporal recognition for {len(all_results['gliner_only']['temporal_failures'])} errors"
        )


if __name__ == "__main__":
    asyncio.run(main())
