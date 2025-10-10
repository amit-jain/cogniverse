#!/usr/bin/env python3
# tests/test_combined_routing.py
"""
Combined LLM and GLiNER Routing Test Suite
Tests both LLM and GLiNER models on query analysis, routing decisions, and temporal extraction.
"""

import asyncio
import csv
import datetime
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cogniverse_core.common.config_utils import get_config
# from cogniverse_agents.tools.query_analyzer import QueryAnalyzer  # Module removed

import pytest
pytestmark = pytest.mark.skip(reason="QueryAnalyzer module removed - test needs rewrite")


@dataclass
class TestQuery:
    """Test query with expected results."""

    query: str
    expected_routing: str
    expected_temporal: str


@dataclass
class TestResult:
    """Result of testing a query with a model."""

    query: str
    model_type: str  # "llm" or "gliner"
    model_name: str
    expected_routing: str
    expected_temporal: str
    actual_routing: str
    actual_temporal: str
    routing_correct: bool
    temporal_correct: bool
    response_time: float
    error: str = ""


@dataclass
class ModelScore:
    """Overall score for a model."""

    model_type: str
    model_name: str
    total_queries: int
    routing_accuracy: float
    temporal_accuracy: float
    overall_accuracy: float
    avg_response_time: float
    errors: int


class CombinedRoutingTester:
    """Test suite for evaluating both LLM and GLiNER routing and temporal extraction."""

    def __init__(self):
        self.config = get_config()

    def load_test_queries(self, filename: str = "test_queries.txt") -> list[TestQuery]:
        """Load test queries from file."""
        queries = []

        # Look for test_queries.txt in tests directory or current directory
        test_file_paths = [
            filename,  # Current directory
            f"tests/{filename}",  # From project root
            f"../{filename}",  # From subdirectory
        ]

        test_file = None
        for path in test_file_paths:
            if Path(path).exists():
                test_file = path
                break

        if not test_file:
            print(f"❌ Test queries file not found in any of: {test_file_paths}")
            sys.exit(1)

        try:
            with open(test_file) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # Skip comments and empty lines
                    if not line or line.startswith("#"):
                        continue

                    # Parse CSV format: query, expected_routing, expected_temporal
                    parts = [part.strip() for part in line.split(",")]

                    if len(parts) != 3:
                        print(f"⚠️  Line {line_num}: Invalid format, skipping: {line}")
                        continue

                    query = TestQuery(
                        query=parts[0],
                        expected_routing=parts[1],
                        expected_temporal=parts[2],
                    )
                    queries.append(query)

        except FileNotFoundError:
            print(f"❌ Test queries file not found: {test_file}")
            sys.exit(1)

        print(f"✅ Loaded {len(queries)} test queries from {test_file}")
        return queries

    def normalize_routing(self, routing_result: dict[str, Any]) -> str:
        """Convert routing analysis to normalized format."""
        needs_video = routing_result.get("needs_video_search", False)
        needs_text = routing_result.get("needs_text_search", False)

        if needs_video and needs_text:
            return "both"
        elif needs_video:
            return "video"
        elif needs_text:
            return "text"
        else:
            return "video"  # Default fallback

    def normalize_temporal_llm(self, temporal_info: dict[str, Any]) -> str:
        """Convert LLM temporal analysis to normalized format."""
        if not temporal_info:
            return "none"

        # Return the detected pattern name, not the resolved dates
        detected_pattern = temporal_info.get("detected_pattern")
        if detected_pattern:
            return detected_pattern

        # Fallback to checking for specific date patterns
        start_date = temporal_info.get("start_date")
        end_date = temporal_info.get("end_date")

        if start_date and end_date and start_date != end_date:
            # This is a date range, but we want the pattern type, not dates
            return "date_range"
        elif start_date:
            # Single date - check if it's a specific date pattern
            if re.match(r"\d{4}-\d{2}-\d{2}", start_date):
                return start_date  # Keep specific dates as-is
            else:
                return "specific_date"
        else:
            return "none"

    def normalize_temporal_gliner(self, result: dict[str, Any]) -> str:
        """Convert GLiNER temporal analysis to normalized format."""
        return result.get("temporal_pattern", "none")

    def calculate_temporal_match(self, expected: str, actual: str) -> bool:
        """Calculate if temporal extraction matches expected result."""
        if expected == actual:
            return True

        # Normalize both to lowercase for comparison
        expected_lower = expected.lower()
        actual_lower = actual.lower()

        if expected_lower == actual_lower:
            return True

        # Handle date range formats: 2024-01-10_to_2024-01-20 vs between_2024_01_10_and_2024_01_20
        if "_to_" in expected_lower and (
            "between" in actual_lower or "_and_" in actual_lower
        ):
            # Extract dates from expected format: 2024-01-10_to_2024-01-20
            if "_to_" in expected_lower:
                start_date, end_date = expected_lower.split("_to_")
                # Check if actual contains both dates
                if (
                    start_date.replace("-", "_") in actual_lower
                    and end_date.replace("-", "_") in actual_lower
                ):
                    return True
                if start_date in actual_lower and end_date in actual_lower:
                    return True

        # Handle month-year formats: january_2024 vs January 2024
        if "_" in expected_lower and " " in actual_lower:
            expected_normalized = expected_lower.replace("_", " ")
            if expected_normalized == actual_lower:
                return True

        # Handle specific date formats
        if expected == "2024-01-15":
            date_patterns = [
                "2024-01-15",
                "01/15/2024",
                "15/01/2024",
                "january_15_2024",
            ]
            return any(pattern in actual.lower() for pattern in date_patterns)

        # Handle relative dates
        relative_matches = {
            "yesterday": ["yesterday", "1_day_ago"],
            "last_week": ["last_week", "past_week", "7_days_ago"],
            "last_month": ["last_month", "past_month", "30_days_ago"],
            "this_week": ["this_week", "current_week"],
            "this_month": ["this_month", "current_month"],
            "past_7_days": ["past_7_days", "seven_days_ago", "last_7_days"],
        }

        for key, variants in relative_matches.items():
            if expected_lower == key and any(
                variant in actual_lower for variant in variants
            ):
                return True

        # Debug output for unmatched patterns
        print(
            f"         🔍 TEMPORAL MISMATCH: expected='{expected}' vs actual='{actual}'"
        )

        return False

    # =================== LLM TESTING METHODS ===================

    async def test_query_with_llm_model(
        self, query: TestQuery, model_name: str
    ) -> TestResult:
        """Test a single query with a specific LLM model."""
        start_time = time.time()

        try:
            # Save original config state
            original_model = self.config.get("local_llm_model")

            # Set LLM model
            self.config.set("local_llm_model", model_name)

            # Create new analyzer instance for this model
            test_analyzer = QueryAnalyzer()

            # CRITICAL: Explicitly set mode to LLM (like legacy GLiNER test does)
            test_analyzer.set_mode("llm_only")

            # Store original JSON extraction method to intercept raw responses
            original_extract_json = (
                test_analyzer.inference_engine.extract_json_from_response
            )
            raw_response_debug = {"response": None}

            def debug_extract_json(response_text):
                raw_response_debug["response"] = response_text
                # Simplified debug - just show if think tags detected and key info
                if response_text.startswith("<think>"):
                    print(f"         ⚠️ THINK TAG DETECTED in {model_name}")
                else:
                    print(f"         ✅ Clean response from {model_name}")

                result = original_extract_json(response_text)
                if result:
                    print(
                        f"         ✅ JSON extracted: video={result.get('needs_video_search')}, text={result.get('needs_text_search')}, temporal='{result.get('temporal_pattern')}'"
                    )
                else:
                    print("         ❌ JSON extraction failed")
                return result

            test_analyzer.inference_engine.extract_json_from_response = (
                debug_extract_json
            )

            # Run query analysis with more detailed debugging
            analysis = await test_analyzer.analyze_query(query.query)

            # DEBUG: Check raw LLM response for think tags
            print(f"      🔍 DEBUG LLM Response for {model_name}:")
            print(f"         Analysis keys: {list(analysis.keys())}")
            print(f"         LLM success: {analysis.get('llm_success', False)}")
            print(
                f"         Inference method: {analysis.get('inference_method', 'unknown')}"
            )

            # Show raw response if captured - simplified
            if raw_response_debug["response"]:
                # Already shown above in debug_extract_json
                pass

            # Check for empty or invalid responses
            needs_video = analysis.get("needs_video_search", None)
            needs_text = analysis.get("needs_text_search", None)
            if needs_video is None or needs_text is None:
                print(
                    f"         ❌ MISSING ROUTING DATA: video={needs_video}, text={needs_text}"
                )
                analysis["error"] = "Missing routing decisions from LLM response"
                analysis["llm_success"] = False

            # Check temporal extraction - if query expects temporal but none extracted, it's an error
            temporal_info = analysis.get("temporal_info", {})
            expected_temporal = query.expected_temporal
            if expected_temporal != "none" and not temporal_info:
                print(
                    f"         ❌ MISSING TEMPORAL DATA: expected '{expected_temporal}' but got none"
                )
                analysis["error"] = (
                    f"Expected temporal pattern '{expected_temporal}' but extracted none"
                )
                analysis["llm_success"] = False
            elif not temporal_info:
                print(
                    f"         ⚠️ No temporal info extracted (expected: {expected_temporal})"
                )
            else:
                print(f"         ✅ Temporal info: {temporal_info}")

            if analysis.get("error"):
                print(f"         ERROR: {analysis.get('error')}")

            # Restore original method
            test_analyzer.inference_engine.extract_json_from_response = (
                original_extract_json
            )
            print()

            # Restore original config
            self.config.set("local_llm_model", original_model)

            response_time = time.time() - start_time

            # Check if analysis failed
            if analysis.get("error") or not analysis.get("llm_success", True):
                return TestResult(
                    query=query.query,
                    model_type="llm",
                    model_name=model_name,
                    expected_routing=query.expected_routing,
                    expected_temporal=query.expected_temporal,
                    actual_routing="error",
                    actual_temporal="error",
                    routing_correct=False,
                    temporal_correct=False,
                    response_time=response_time,
                    error=analysis.get("error", "LLM analysis failed"),
                )

            # Extract results
            actual_routing = self.normalize_routing(analysis)
            actual_temporal = self.normalize_temporal_llm(
                analysis.get("temporal_info", {})
            )

            # Calculate accuracy
            routing_correct = actual_routing == query.expected_routing
            temporal_correct = self.calculate_temporal_match(
                query.expected_temporal, actual_temporal
            )

            return TestResult(
                query=query.query,
                model_type="llm",
                model_name=model_name,
                expected_routing=query.expected_routing,
                expected_temporal=query.expected_temporal,
                actual_routing=actual_routing,
                actual_temporal=actual_temporal,
                routing_correct=routing_correct,
                temporal_correct=temporal_correct,
                response_time=response_time,
            )

        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                query=query.query,
                model_type="llm",
                model_name=model_name,
                expected_routing=query.expected_routing,
                expected_temporal=query.expected_temporal,
                actual_routing="error",
                actual_temporal="error",
                routing_correct=False,
                temporal_correct=False,
                response_time=response_time,
                error=str(e),
            )

    async def test_llm_model(
        self, model_name: str, queries: list[TestQuery]
    ) -> list[TestResult]:
        """Test all queries with a specific LLM model."""
        print(f"\n🧪 Testing LLM: {model_name}")
        print(f"📊 Running {len(queries)} queries...")

        results = []

        for i, query in enumerate(queries, 1):
            print(f"   Query {i}/{len(queries)}: {query.query[:50]}...")

            result = await self.test_query_with_llm_model(query, model_name)
            results.append(result)

            # Small delay to avoid overwhelming the model
            await asyncio.sleep(0.1)

        return results

    # =================== GLiNER TESTING METHODS ===================

    def convert_queries_to_gliner_format(self, queries: list[TestQuery]) -> list[tuple]:
        """Convert TestQuery objects to GLiNER test format."""
        gliner_queries = []
        for query in queries:
            # Convert routing format to boolean format
            expected_video = query.expected_routing in ["video", "both"]
            expected_text = query.expected_routing in ["text", "both"]
            gliner_queries.append(
                (query.query, expected_video, expected_text, query.expected_temporal)
            )
        return gliner_queries

    async def test_query_with_gliner_model(
        self, analyzer: "QueryAnalyzer", query_tuple: tuple, model_name: str  # type: ignore
    ) -> TestResult:
        """Test a single query with GLiNER model."""
        query, expected_video, expected_text, expected_temporal = query_tuple
        start_time = time.time()

        try:
            result = await analyzer.analyze_query(query)
            response_time = time.time() - start_time

            # LOG ENTITY DETECTION RESULTS
            print("      🔍 GLiNER Detection Results:")
            print(f"         Query: {query}")
            print(f"         Raw result keys: {list(result.keys())}")
            if "entities" in result:
                print(f"         Detected entities: {result.get('entities', [])}")
            if "gliner_entities" in result:
                print(f"         GLiNER entities: {result.get('gliner_entities', [])}")
            print(
                f"         needs_video_search: {result.get('needs_video_search', False)}"
            )
            print(
                f"         needs_text_search: {result.get('needs_text_search', False)}"
            )
            print(f"         temporal_pattern: {result.get('temporal_pattern')}")
            print(f"         inference_method: {result.get('inference_method')}")
            if result.get("error"):
                print(f"         ERROR: {result.get('error')}")
            print()

            # Extract results
            actual_video = result.get("needs_video_search", False)
            actual_text = result.get("needs_text_search", False)
            actual_temporal = self.normalize_temporal_gliner(result)

            # Convert to routing format
            if actual_video and actual_text:
                actual_routing = "both"
            elif actual_video:
                actual_routing = "video"
            elif actual_text:
                actual_routing = "text"
            else:
                actual_routing = "video"  # Default

            if expected_video and expected_text:
                expected_routing = "both"
            elif expected_video:
                expected_routing = "video"
            elif expected_text:
                expected_routing = "text"
            else:
                expected_routing = "video"

            # Check correctness
            routing_correct = actual_routing == expected_routing
            temporal_correct = self.calculate_temporal_match(
                expected_temporal, actual_temporal
            )

            return TestResult(
                query=query,
                model_type="gliner",
                model_name=model_name,
                expected_routing=expected_routing,
                expected_temporal=expected_temporal,
                actual_routing=actual_routing,
                actual_temporal=actual_temporal,
                routing_correct=routing_correct,
                temporal_correct=temporal_correct,
                response_time=response_time,
                error=result.get("error", ""),
            )

        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                query=query,
                model_type="gliner",
                model_name=model_name,
                expected_routing="error",
                expected_temporal="error",
                actual_routing="error",
                actual_temporal="error",
                routing_correct=False,
                temporal_correct=False,
                response_time=response_time,
                error=str(e),
            )

    async def test_gliner_model(
        self, model_name: str, query_tuples: list[tuple]
    ) -> list[TestResult]:
        """Test all queries with a specific GLiNER model."""
        print(f"\n🤖 Testing GLiNER: {model_name.split('/')[-1]}")
        print(f"📊 Running {len(query_tuples)} queries...")

        # Create analyzer and switch to this model
        analyzer = QueryAnalyzer()
        success = analyzer.switch_gliner_model(model_name)
        if not success:
            print(f"❌ Failed to load GLiNER model: {model_name}")
            return []

        analyzer.set_mode("gliner_only")

        results = []

        for i, query_tuple in enumerate(query_tuples, 1):
            query = query_tuple[0]
            print(f"   Query {i}/{len(query_tuples)}: {query[:50]}...")

            result = await self.test_query_with_gliner_model(
                analyzer, query_tuple, model_name
            )
            results.append(result)

            # Small delay
            await asyncio.sleep(0.1)

        return results

    # =================== SCORING AND REPORTING ===================

    def calculate_score(self, results: list[TestResult]) -> ModelScore:
        """Calculate overall score for a model."""
        if not results:
            return None

        first_result = results[0]
        model_type = first_result.model_type
        model_name = first_result.model_name

        total = len(results)
        routing_correct = sum(1 for r in results if r.routing_correct)
        temporal_correct = sum(1 for r in results if r.temporal_correct)
        errors = sum(1 for r in results if r.error)

        routing_accuracy = (routing_correct / total) * 100 if total > 0 else 0
        temporal_accuracy = (temporal_correct / total) * 100 if total > 0 else 0
        overall_accuracy = (
            ((routing_correct + temporal_correct) / (total * 2)) * 100
            if total > 0
            else 0
        )

        avg_response_time = (
            sum(r.response_time for r in results) / total if total > 0 else 0
        )

        return ModelScore(
            model_type=model_type,
            model_name=model_name,
            total_queries=total,
            routing_accuracy=routing_accuracy,
            temporal_accuracy=temporal_accuracy,
            overall_accuracy=overall_accuracy,
            avg_response_time=avg_response_time,
            errors=errors,
        )

    def save_detailed_results(
        self, all_results: list[TestResult], filename: str = "combined_test_results.csv"
    ):
        """Save detailed results to CSV."""
        with open(filename, "w", newline="") as csvfile:
            fieldnames = [
                "model_type",
                "model_name",
                "query",
                "expected_routing",
                "expected_temporal",
                "actual_routing",
                "actual_temporal",
                "routing_correct",
                "temporal_correct",
                "response_time",
                "error",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in all_results:
                writer.writerow(
                    {
                        "model_type": result.model_type,
                        "model_name": result.model_name,
                        "query": result.query,
                        "expected_routing": result.expected_routing,
                        "expected_temporal": result.expected_temporal,
                        "actual_routing": result.actual_routing,
                        "actual_temporal": result.actual_temporal,
                        "routing_correct": result.routing_correct,
                        "temporal_correct": result.temporal_correct,
                        "response_time": result.response_time,
                        "error": result.error,
                    }
                )

        print(f"📄 Detailed results saved to: {filename}")

    def save_summary_report(
        self,
        scores: list[ModelScore],
        filename: str = "combined_comparison_report.json",
    ):
        """Save summary report to JSON."""
        report = {
            "test_date": datetime.datetime.now().isoformat(),
            "models_tested": len(scores),
            "total_queries": scores[0].total_queries if scores else 0,
            "scores": [asdict(score) for score in scores],
            "ranking": {
                "by_overall_accuracy": sorted(
                    scores, key=lambda x: x.overall_accuracy, reverse=True
                ),
                "by_routing_accuracy": sorted(
                    scores, key=lambda x: x.routing_accuracy, reverse=True
                ),
                "by_temporal_accuracy": sorted(
                    scores, key=lambda x: x.temporal_accuracy, reverse=True
                ),
                "by_response_time": sorted(scores, key=lambda x: x.avg_response_time),
            },
        }

        # Convert ranking to dict format
        for category in report["ranking"]:
            report["ranking"][category] = [
                asdict(score) for score in report["ranking"][category]
            ]

        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        print(f"📊 Summary report saved to: {filename}")

    def print_results(self, scores: list[ModelScore]):
        """Print formatted results to console."""
        print("\n" + "=" * 80)
        print("🏆 COMBINED LLM & GLiNER ROUTING TEST RESULTS")
        print("=" * 80)

        # Sort by overall accuracy
        scores.sort(key=lambda x: x.overall_accuracy, reverse=True)

        # Group by model type
        llm_scores = [s for s in scores if s.model_type == "llm"]
        gliner_scores = [s for s in scores if s.model_type == "gliner"]

        print(
            f"\n📊 OVERALL RANKING (Total Queries: {scores[0].total_queries if scores else 0})"
        )
        print("-" * 80)

        for i, score in enumerate(scores, 1):
            model_display = f"{score.model_type.upper()}: {score.model_name.split('/')[-1] if score.model_type == 'gliner' else score.model_name}"
            print(f"{i}. {model_display}")
            print(f"   Overall Accuracy: {score.overall_accuracy:.1f}%")
            print(f"   Routing Accuracy: {score.routing_accuracy:.1f}%")
            print(f"   Temporal Accuracy: {score.temporal_accuracy:.1f}%")
            print(f"   Avg Response Time: {score.avg_response_time:.2f}s")
            print(f"   Errors: {score.errors}")
            print()

        # Category breakdowns
        if llm_scores:
            print("🧠 LLM MODELS")
            print("-" * 40)
            for i, score in enumerate(llm_scores, 1):
                print(f"{i}. {score.model_name}: {score.overall_accuracy:.1f}%")

        if gliner_scores:
            print("\n🤖 GLiNER MODELS")
            print("-" * 40)
            for i, score in enumerate(gliner_scores, 1):
                print(
                    f"{i}. {score.model_name.split('/')[-1]}: {score.overall_accuracy:.1f}%"
                )

        # Best performers
        if scores:
            best_overall = max(scores, key=lambda x: x.overall_accuracy)
            best_routing = max(scores, key=lambda x: x.routing_accuracy)
            best_temporal = max(scores, key=lambda x: x.temporal_accuracy)
            fastest = min(scores, key=lambda x: x.avg_response_time)

            print("\n🥇 BEST PERFORMERS")
            print("-" * 40)
            print(
                f"Overall Accuracy: {best_overall.model_type.upper()}:{best_overall.model_name.split('/')[-1] if best_overall.model_type == 'gliner' else best_overall.model_name} ({best_overall.overall_accuracy:.1f}%)"
            )
            print(
                f"Routing Accuracy: {best_routing.model_type.upper()}:{best_routing.model_name.split('/')[-1] if best_routing.model_type == 'gliner' else best_routing.model_name} ({best_routing.routing_accuracy:.1f}%)"
            )
            print(
                f"Temporal Accuracy: {best_temporal.model_type.upper()}:{best_temporal.model_name.split('/')[-1] if best_temporal.model_type == 'gliner' else best_temporal.model_name} ({best_temporal.temporal_accuracy:.1f}%)"
            )
            print(
                f"Fastest Response: {fastest.model_type.upper()}:{fastest.model_name.split('/')[-1] if fastest.model_type == 'gliner' else fastest.model_name} ({fastest.avg_response_time:.2f}s)"
            )

    async def run_combined_test(
        self,
        llm_models: list[str] = None,
        test_gliner: bool = True,
        queries: list[TestQuery] = None,
    ):
        """Run comprehensive test of both LLM and GLiNER models."""
        print("🚀 Starting Combined LLM & GLiNER Routing Test Suite")
        print("=" * 60)

        # Load test queries or use provided ones
        if queries is None:
            queries = self.load_test_queries()

        all_results = []
        scores = []

        # Test LLM models
        if llm_models:
            print(f"\n🧠 Testing {len(llm_models)} LLM Models")
            print("=" * 60)

            for model in llm_models:
                try:
                    results = await self.test_llm_model(model, queries)
                    score = self.calculate_score(results)

                    all_results.extend(results)
                    scores.append(score)

                    print(f"✅ {model}: {score.overall_accuracy:.1f}% accuracy")

                except Exception as e:
                    print(f"❌ {model}: Failed - {e}")

        # Test GLiNER models
        if test_gliner:
            available_gliner = self.config.get("query_inference_engine", {}).get(
                "available_gliner_models", []
            )

            if available_gliner:
                print(f"\n🤖 Testing {len(available_gliner)} GLiNER Models")
                print("=" * 60)

                # Convert queries to GLiNER format
                gliner_queries = self.convert_queries_to_gliner_format(queries)

                for model in available_gliner:
                    try:
                        results = await self.test_gliner_model(model, gliner_queries)
                        if results:  # Only calculate if we got results
                            score = self.calculate_score(results)

                            all_results.extend(results)
                            scores.append(score)

                            print(
                                f"✅ {model.split('/')[-1]}: {score.overall_accuracy:.1f}% accuracy"
                            )

                    except Exception as e:
                        print(f"❌ {model}: Failed - {e}")
            else:
                print("⚠️ No GLiNER models configured")

        # Save results and print summary
        if all_results:
            self.save_detailed_results(all_results)
            self.save_summary_report(scores)
            self.print_results(scores)
        else:
            print("❌ No results to display")

        return scores


async def main():
    """Main test execution."""
    # LLM Models to test - All available models
    llm_models_to_test = [
        "deepseek-r1:1.5b",  # DeepSeek provider
        "gemma3:1b",  # Gemma provider
        "deepseek-r1:7b",
        "deepseek-r1:8b",
        "gemma3:4b",
        "gemma3:12b",
        "qwen3:0.6b",
        "qwen3:1.7b",
        "qwen3:4b",
        "qwen3:8b",
    ]

    print("🧪 Combined LLM & GLiNER Routing Test Suite")
    print("=" * 60)

    # Allow user to select mode
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg == "llm-only":
            print("🧠 Testing LLM models only")
            tester = CombinedRoutingTester()
            await tester.run_combined_test(
                llm_models=llm_models_to_test, test_gliner=False
            )
        elif arg == "gliner-only":
            print("🤖 Testing GLiNER models only")
            tester = CombinedRoutingTester()
            await tester.run_combined_test(llm_models=None, test_gliner=True)
        elif arg == "quick":
            print("⚡ Quick test with limited models")
            tester = CombinedRoutingTester()
            # Load all queries and randomly select 6 for quick test
            all_queries = tester.load_test_queries()
            import random

            quick_queries = random.sample(all_queries, min(6, len(all_queries)))

            # Test with limited queries
            print(
                f"📝 Using {len(quick_queries)} randomly selected queries for quick test"
            )

            # Use all models for quick test including GLiNER
            await tester.run_combined_test(
                llm_models=llm_models_to_test, test_gliner=True, queries=quick_queries
            )
        else:
            print(
                "❌ Unknown option. Use: llm-only, gliner-only, quick, or no argument for full test"
            )
            return
    else:
        print("🚀 Testing all models (LLM + GLiNER)")
        tester = CombinedRoutingTester()
        await tester.run_combined_test(llm_models=llm_models_to_test, test_gliner=True)

    print("\n🎉 Testing completed! Check the generated files for detailed results.")


if __name__ == "__main__":
    asyncio.run(main())
