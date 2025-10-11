"""
Custom solvers for Cogniverse evaluation with Inspect AI
"""

import logging
import time
from typing import Any

from inspect_ai.solver import Solver, solver

from cogniverse_agents.search.service import SearchService
from cogniverse_core.config.utils import get_config

logger = logging.getLogger(__name__)


# Factory functions for Inspect AI registration
@solver(name="cogniverse_retrieval_solver")
def cogniverse_retrieval_solver(profiles: list[str], strategies: list[str]) -> Solver:
    """Create a Cogniverse retrieval solver for Inspect AI."""
    return CogniverseRetrievalSolver(profiles, strategies)


class CogniverseRetrievalSolver(Solver):
    """Custom solver for Cogniverse retrieval evaluation"""

    def __init__(self, profiles: list[str], strategies: list[str]):
        self.profiles = profiles
        self.strategies = strategies
        self.config = get_config()
        self.search_services = {}

        # Initialize search services for each profile
        for profile in profiles:
            try:
                self.search_services[profile] = SearchService(self.config, profile)
                logger.info(f"Initialized SearchService for profile: {profile}")
            except Exception as e:
                logger.error(f"Failed to initialize SearchService for {profile}: {e}")

    async def __call__(self, state, generate):
        """Execute retrieval for all profile/strategy combinations"""
        query = state.input.text if hasattr(state.input, "text") else str(state.input)
        results = {}

        # Test across profiles and strategies
        for profile in self.profiles:
            if profile not in self.search_services:
                logger.warning(f"Skipping profile {profile} - no search service")
                continue

            search_service = self.search_services[profile]

            for strategy in self.strategies:
                config_key = f"{profile}_{strategy}"

                # Execute search with OpenTelemetry tracing
                from opentelemetry import trace

                tracer = trace.get_tracer(__name__)

                with tracer.start_as_current_span("retrieval_solve") as span:
                    span.set_attribute("profile", profile)
                    span.set_attribute("strategy", strategy)
                    span.set_attribute("query", query)

                    start_time = time.time()

                    # Execute search (Phoenix experiment system handles project isolation)
                    search_results = search_service.search(
                        query=query, top_k=10, ranking_strategy=strategy
                    )

                    # Convert results to serializable format
                    serialized_results = []
                    for result in search_results:
                        result_dict = result.to_dict()
                        # Extract video ID from document
                        video_id = result_dict.get(
                            "source_id", result_dict["document_id"].split("_")[0]
                        )
                        serialized_results.append(
                            {
                                "video_id": video_id,
                                "score": result_dict["score"],
                                "document_id": result_dict["document_id"],
                            }
                        )

                    results[config_key] = serialized_results

                    # Add span attributes
                    latency_ms = (time.time() - start_time) * 1000
                    span.set_attribute("num_results", len(serialized_results))
                    span.set_attribute("latency_ms", latency_ms)
                    if serialized_results:
                        span.set_attribute("top_score", serialized_results[0]["score"])

                    logger.info(
                        f"Search completed for {config_key}: {len(serialized_results)} results in {latency_ms:.2f}ms"
                    )

        # Store results in state metadata
        if not hasattr(state, "metadata"):
            state.metadata = {}
        state.metadata["retrieval_results"] = results

        return state


class ResultRankingAnalyzer(Solver):
    """Analyze ranking quality across different strategies"""

    async def __call__(self, state, generate):
        """Analyze ranking quality"""
        if not hasattr(state, "metadata") or "retrieval_results" not in state.metadata:
            logger.warning("No retrieval results to analyze")
            return state

        results = state.metadata["retrieval_results"]

        # Get expected videos from metadata
        expected = []
        if hasattr(state, "metadata") and "expected_videos" in state.metadata:
            expected = state.metadata["expected_videos"]
        elif (
            hasattr(state.input, "metadata")
            and "expected_videos" in state.input.metadata
        ):
            expected = state.input.metadata["expected_videos"]

        rankings = {}
        for config, search_results in results.items():
            ranking_analysis = self._analyze_ranking(search_results, expected)
            rankings[config] = ranking_analysis

        state.metadata["ranking_analysis"] = rankings
        return state

    def _analyze_ranking(
        self, search_results: list[dict], expected: list[str]
    ) -> dict[str, Any]:
        """Analyze ranking quality"""
        if not search_results:
            return {
                "total_results": 0,
                "relevant_found": 0,
                "first_relevant_position": -1,
                "precision_at_5": 0.0,
                "precision_at_10": 0.0,
            }

        # Extract video IDs
        result_videos = [r["video_id"] for r in search_results]

        # Find relevant results
        relevant_found = [vid for vid in result_videos if vid in expected]

        # Find position of first relevant result
        first_relevant_pos = -1
        for i, vid in enumerate(result_videos):
            if vid in expected:
                first_relevant_pos = i + 1
                break

        # Calculate precision at k
        precision_at_5 = len([v for v in result_videos[:5] if v in expected]) / min(
            5, len(result_videos)
        )
        precision_at_10 = len([v for v in result_videos[:10] if v in expected]) / min(
            10, len(result_videos)
        )

        return {
            "total_results": len(search_results),
            "relevant_found": len(relevant_found),
            "first_relevant_position": first_relevant_pos,
            "precision_at_5": precision_at_5,
            "precision_at_10": precision_at_10,
            "result_order": result_videos[:10],  # Top 10 for inspection
        }


class RelevanceJudgmentCollector(Solver):
    """Collect relevance judgments for results"""

    async def __call__(self, state, generate):
        """Collect relevance judgments"""
        if not hasattr(state, "metadata") or "retrieval_results" not in state.metadata:
            return state

        results = state.metadata["retrieval_results"]

        # Get expected videos
        expected = []
        if hasattr(state, "metadata") and "expected_videos" in state.metadata:
            expected = state.metadata["expected_videos"]
        elif (
            hasattr(state.input, "metadata")
            and "expected_videos" in state.input.metadata
        ):
            expected = state.input.metadata["expected_videos"]

        judgments = {}
        for config, search_results in results.items():
            config_judgments = []
            for result in search_results[:10]:  # Judge top 10
                video_id = result["video_id"]
                relevance = 1 if video_id in expected else 0
                config_judgments.append(
                    {
                        "video_id": video_id,
                        "score": result["score"],
                        "relevance": relevance,
                    }
                )
            judgments[config] = config_judgments

        state.metadata["relevance_judgments"] = judgments
        return state


class TemporalQueryProcessor(Solver):
    """Process temporal queries to extract time ranges"""

    async def __call__(self, state, generate):
        """Process temporal query"""
        query = state.input.text if hasattr(state.input, "text") else str(state.input)

        # Simple temporal keyword extraction
        temporal_info = self._extract_temporal_info(query)

        if not hasattr(state, "metadata"):
            state.metadata = {}
        state.metadata["temporal_info"] = temporal_info

        return state

    def _extract_temporal_info(self, query: str) -> dict[str, Any]:
        """Extract temporal information from query"""
        query_lower = query.lower()

        temporal_info = {
            "has_temporal": False,
            "time_type": None,
            "extracted_range": None,
        }

        # Check for absolute time references
        if "first" in query_lower and "seconds" in query_lower:
            # Extract number of seconds
            import re

            match = re.search(r"first (\d+) seconds", query_lower)
            if match:
                seconds = int(match.group(1))
                temporal_info["has_temporal"] = True
                temporal_info["time_type"] = "absolute"
                temporal_info["extracted_range"] = [0, seconds]

        elif "last" in query_lower and "seconds" in query_lower:
            import re

            match = re.search(r"last (\d+) seconds", query_lower)
            if match:
                seconds = int(match.group(1))
                temporal_info["has_temporal"] = True
                temporal_info["time_type"] = "absolute"
                temporal_info["extracted_range"] = [-seconds, -1]

        elif "beginning" in query_lower or "start" in query_lower:
            temporal_info["has_temporal"] = True
            temporal_info["time_type"] = "relative"
            temporal_info["extracted_range"] = [0, 0.2]  # First 20%

        elif "end" in query_lower or "ending" in query_lower:
            temporal_info["has_temporal"] = True
            temporal_info["time_type"] = "relative"
            temporal_info["extracted_range"] = [0.8, 1.0]  # Last 20%

        elif "middle" in query_lower:
            temporal_info["has_temporal"] = True
            temporal_info["time_type"] = "relative"
            temporal_info["extracted_range"] = [0.4, 0.6]  # Middle 20%

        return temporal_info


class TimeRangeExtractor(Solver):
    """Extract specific time ranges from videos"""

    async def __call__(self, state, generate):
        """Extract time range"""
        if not hasattr(state, "metadata") or "temporal_info" not in state.metadata:
            return state

        temporal_info = state.metadata["temporal_info"]

        if temporal_info["has_temporal"] and temporal_info["extracted_range"]:
            # Modify query to include temporal constraints
            original_query = (
                state.input.text if hasattr(state.input, "text") else str(state.input)
            )
            time_range = temporal_info["extracted_range"]

            # Add temporal constraint to metadata for retrieval
            state.metadata["time_constraint"] = time_range

            # Log temporal extraction
            logger.info(
                f"Extracted time range {time_range} from query: {original_query}"
            )

        return state


class VisualQueryEncoder(Solver):
    """Encode visual aspects of queries"""

    async def __call__(self, state, generate):
        """Encode visual query"""
        # Extract visual description from input
        input_text = (
            state.input.text if hasattr(state.input, "text") else str(state.input)
        )

        if "|" in input_text:
            # Multimodal input format: text|visual
            text_part, visual_part = input_text.split("|", 1)

            if not hasattr(state, "metadata"):
                state.metadata = {}

            state.metadata["visual_description"] = visual_part.strip()
            state.metadata["text_query"] = text_part.strip()

        return state


class TextQueryEncoder(Solver):
    """Encode textual aspects of queries"""

    async def __call__(self, state, generate):
        """Encode text query"""
        # Text encoding already handled by VisualQueryEncoder
        # This solver could add additional text processing if needed
        return state


class CrossModalAlignmentChecker(Solver):
    """Check alignment between visual and text modalities"""

    async def __call__(self, state, generate):
        """Check cross-modal alignment"""
        if not hasattr(state, "metadata"):
            return state

        visual_desc = state.metadata.get("visual_description", "")
        text_query = state.metadata.get("text_query", "")

        # Simple keyword-based alignment check
        alignment_score = self._calculate_alignment(text_query, visual_desc)

        state.metadata["alignment_score"] = alignment_score
        state.metadata["alignment_check"] = alignment_score > 0.5

        return state

    def _calculate_alignment(self, text: str, visual: str) -> float:
        """Calculate simple alignment score between text and visual descriptions"""
        text_words = set(text.lower().split())
        visual_words = set(visual.lower().split())

        if not text_words or not visual_words:
            return 0.0

        # Calculate Jaccard similarity
        intersection = text_words & visual_words
        union = text_words | visual_words

        if not union:
            return 0.0

        return len(intersection) / len(union)


class FailureAnalyzer(Solver):
    """Analyze failure cases in retrieval"""

    async def __call__(self, state, generate):
        """Analyze failures"""
        if not hasattr(state, "metadata") or "retrieval_results" not in state.metadata:
            return state

        results = state.metadata["retrieval_results"]
        expected = state.metadata.get("expected_videos", [])

        failures = {}
        for config, search_results in results.items():
            if not search_results:
                failures[config] = {
                    "type": "no_results",
                    "details": "Search returned no results",
                }
            else:
                result_videos = [r["video_id"] for r in search_results[:10]]
                if not any(vid in expected for vid in result_videos):
                    failures[config] = {
                        "type": "no_relevant_in_top_10",
                        "details": f"Expected {expected}, got {result_videos[:5]}",
                    }

        state.metadata["failure_analysis"] = failures
        return state


class ErrorPatternDetector(Solver):
    """Detect patterns in errors and failures"""

    async def __call__(self, state, generate):
        """Detect error patterns"""
        if not hasattr(state, "metadata") or "failure_analysis" not in state.metadata:
            return state

        failures = state.metadata["failure_analysis"]

        # Detect patterns
        patterns = {
            "no_results_count": 0,
            "no_relevant_count": 0,
            "affected_profiles": [],
            "affected_strategies": [],
        }

        for config, failure in failures.items():
            profile, strategy = config.split("_", 1)

            if failure["type"] == "no_results":
                patterns["no_results_count"] += 1
            elif failure["type"] == "no_relevant_in_top_10":
                patterns["no_relevant_count"] += 1

            if profile not in patterns["affected_profiles"]:
                patterns["affected_profiles"].append(profile)
            if strategy not in patterns["affected_strategies"]:
                patterns["affected_strategies"].append(strategy)

        state.metadata["error_patterns"] = patterns
        return state
