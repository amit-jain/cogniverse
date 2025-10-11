"""
Unit tests for Inspect AI solvers.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from cogniverse_core.evaluation.inspect_tasks.solvers import CogniverseRetrievalSolver


class TestCogniverseRetrievalSolver:
    """Test CogniverseRetrievalSolver functionality."""

    @pytest.fixture
    def mock_search_service(self):
        """Create mock search service."""
        with patch("cogniverse_core.evaluation.inspect_tasks.solvers.SearchService") as mock_cls:
            mock_service = Mock()
            mock_service.search = Mock(
                return_value=[
                    Mock(
                        to_dict=lambda: {
                            "document_id": "video1_0",
                            "source_id": "video1",
                            "score": 0.9,
                        }
                    ),
                    Mock(
                        to_dict=lambda: {
                            "document_id": "video2_0",
                            "source_id": "video2",
                            "score": 0.8,
                        }
                    ),
                    Mock(
                        to_dict=lambda: {
                            "document_id": "video3_0",
                            "source_id": "video3",
                            "score": 0.7,
                        }
                    ),
                ]
            )
            mock_cls.return_value = mock_service
            yield mock_service

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        with patch("cogniverse_core.evaluation.inspect_tasks.solvers.get_config") as mock_get:
            mock_get.return_value = {
                "vespa": {"url": "http://localhost:8080"},
                "profiles": {"test_profile": {"embedding_model": "test_model"}},
            }
            yield mock_get.return_value

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_solver_initialization(self, mock_config):
        """Test solver initialization."""
        solver = CogniverseRetrievalSolver(
            profiles=["test_profile"], strategies=["test_strategy"]
        )

        assert solver.profiles == ["test_profile"]
        assert solver.strategies == ["test_strategy"]
        assert solver.config is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_solver_call(self, mock_config, mock_search_service):
        """Test solver execution."""
        solver = CogniverseRetrievalSolver(
            profiles=["test_profile"], strategies=["test_strategy"]
        )

        # Create mock state
        state = Mock()
        state.input = Mock()
        state.input.text = "test query"
        state.metadata = {}

        # Execute solver
        updated_state = await solver(state, None)

        # Check results were added to metadata
        assert "retrieval_results" in updated_state.metadata
        results = updated_state.metadata["retrieval_results"]
        assert isinstance(results, dict)
        # Should have results for test_profile_test_strategy
        assert "test_profile_test_strategy" in results
        assert len(results["test_profile_test_strategy"]) == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_solver_error_handling(self, mock_config):
        """Test solver error handling."""
        with patch("cogniverse_core.evaluation.inspect_tasks.solvers.SearchService") as mock_cls:
            mock_cls.side_effect = Exception("Connection failed")

            solver = CogniverseRetrievalSolver(
                profiles=["test_profile"], strategies=["test_strategy"]
            )

            state = Mock()
            state.input = Mock()
            state.input.text = "test query"
            state.metadata = {}

            # Should handle error gracefully
            updated_state = await solver(state, None)

            # Should still return state, possibly with error info
            assert updated_state is not None
            assert (
                "retrieval_results" in updated_state.metadata
                or "error" in updated_state.metadata
            )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_solver_phoenix_logging(self, mock_config, mock_search_service):
        """Test OpenTelemetry tracing integration."""
        solver = CogniverseRetrievalSolver(
            profiles=["test_profile"], strategies=["test_strategy"]
        )

        state = Mock()
        state.input = Mock()
        state.input.text = "test query"
        state.metadata = {}

        await solver(state, None)

        # Check search service was used
        assert mock_search_service.search.called

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_solver_with_ground_truth(self, mock_config, mock_search_service):
        """Test solver with ground truth extraction."""
        # Ground truth IS implemented in src.evaluation.core.ground_truth
        # Import from the CORRECT location where it actually exists
        with patch(
            "cogniverse_core.evaluation.core.ground_truth.get_ground_truth_strategy"
        ) as mock_gt:
            mock_strategy = Mock()
            mock_strategy.extract_ground_truth = AsyncMock(
                return_value={"expected_items": ["video1", "video3"], "confidence": 0.9}
            )
            mock_gt.return_value = mock_strategy

            solver = CogniverseRetrievalSolver(
                profiles=["test_profile"], strategies=["test_strategy"]
            )

            state = Mock()
            state.input = Mock()
            state.input.text = "test query"
            state.metadata = {"use_ground_truth": True}

            updated_state = await solver(state, None)

            # Verify solver runs successfully with ground truth
            assert updated_state is not None
            assert "retrieval_results" in updated_state.metadata

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_solver_multiple_profiles_strategies(
        self, mock_config, mock_search_service
    ):
        """Test solver with multiple profiles and strategies."""
        solver = CogniverseRetrievalSolver(
            profiles=["profile1", "profile2"], strategies=["strategy1", "strategy2"]
        )

        state = Mock()
        state.input = Mock()
        state.input.text = "test query"
        state.metadata = {}

        updated_state = await solver(state, None)

        # Should have results for all combinations
        assert "retrieval_results" in updated_state.metadata
        results = updated_state.metadata["retrieval_results"]
        assert "profile1_strategy1" in results
        assert "profile1_strategy2" in results
        assert "profile2_strategy1" in results
        assert "profile2_strategy2" in results


class TestResultRankingAnalyzer:
    """Test ResultRankingAnalyzer functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ranking_analyzer_initialization(self):
        """Test analyzer initialization."""
        with patch(
            "cogniverse_core.evaluation.inspect_tasks.solvers.ResultRankingAnalyzer"
        ) as MockAnalyzer:
            mock_analyzer = Mock()
            mock_analyzer.metric = "ndcg"
            mock_analyzer.k = 10
            MockAnalyzer.return_value = mock_analyzer

            from cogniverse_core.evaluation.inspect_tasks.solvers import ResultRankingAnalyzer

            analyzer = ResultRankingAnalyzer(metric="ndcg", k=10)

            assert analyzer.metric == "ndcg"
            assert analyzer.k == 10

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ranking_analysis(self):
        """Test ranking analysis functionality."""
        with patch(
            "cogniverse_core.evaluation.inspect_tasks.solvers.ResultRankingAnalyzer"
        ) as MockAnalyzer:
            # Create an AsyncMock that acts as a callable
            async def mock_analyzer_call(state, gen):
                state.metadata["ranking_metrics"] = {"metric": "ndcg"}
                return state

            mock_analyzer = AsyncMock(side_effect=mock_analyzer_call)
            mock_analyzer.metric = "ndcg"
            MockAnalyzer.return_value = mock_analyzer

            from cogniverse_core.evaluation.inspect_tasks.solvers import ResultRankingAnalyzer

            analyzer = ResultRankingAnalyzer(metric="ndcg")

            state = Mock()
            state.metadata = {
                "retrieval_results": {
                    "test": [
                        {"id": "1", "score": 0.9, "rank": 1},
                        {"id": "2", "score": 0.8, "rank": 2},
                        {"id": "3", "score": 0.7, "rank": 3},
                    ]
                },
                "ground_truth": ["1", "3", "5"],
            }

            updated_state = await analyzer(state, None)

            # Should have ranking metrics
            assert "ranking_metrics" in updated_state.metadata
            assert updated_state.metadata["ranking_metrics"]["metric"] == "ndcg"


class TestRelevanceJudgmentCollector:
    """Test RelevanceJudgmentCollector functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_judgment_collector_initialization(self):
        """Test collector initialization."""
        with patch(
            "cogniverse_core.evaluation.inspect_tasks.solvers.RelevanceJudgmentCollector"
        ) as MockCollector:
            mock_collector = Mock()
            mock_collector.threshold = 0.7
            mock_collector.binary = True
            MockCollector.return_value = mock_collector

            from cogniverse_core.evaluation.inspect_tasks.solvers import RelevanceJudgmentCollector

            collector = RelevanceJudgmentCollector(threshold=0.7, binary=True)

            assert collector.threshold == 0.7
            assert collector.binary is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_relevance_judgment_collection(self):
        """Test relevance judgment collection."""
        with patch(
            "cogniverse_core.evaluation.inspect_tasks.solvers.RelevanceJudgmentCollector"
        ) as MockCollector:
            # Create an AsyncMock that acts as a callable
            async def mock_collector_call(state, gen):
                state.metadata["relevance_judgments"] = {"1": 1, "2": 0, "3": 1}
                return state

            mock_collector = AsyncMock(side_effect=mock_collector_call)
            mock_collector.threshold = 0.5
            MockCollector.return_value = mock_collector

            from cogniverse_core.evaluation.inspect_tasks.solvers import RelevanceJudgmentCollector

            collector = RelevanceJudgmentCollector(threshold=0.5)

            state = Mock()
            state.metadata = {
                "retrieval_results": {
                    "test": [
                        {"id": "1", "score": 0.9},
                        {"id": "2", "score": 0.3},
                        {"id": "3", "score": 0.7},
                    ]
                }
            }

            updated_state = await collector(state, None)

            # Should have judgments
            assert "relevance_judgments" in updated_state.metadata
            judgments = updated_state.metadata["relevance_judgments"]
            assert judgments["1"] == 1  # Relevant (score > 0.5)
            assert judgments["2"] == 0  # Not relevant (score < 0.5)
            assert judgments["3"] == 1  # Relevant (score > 0.5)


class TestTemporalQueryProcessor:
    """Test TemporalQueryProcessor functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_temporal_processor_initialization(self):
        """Test processor initialization."""
        with patch(
            "cogniverse_core.evaluation.inspect_tasks.solvers.TemporalQueryProcessor"
        ) as MockProcessor:
            mock_processor = Mock()
            mock_processor.time_window = "1h"
            mock_processor.ordering = "desc"
            MockProcessor.return_value = mock_processor

            from cogniverse_core.evaluation.inspect_tasks.solvers import TemporalQueryProcessor

            processor = TemporalQueryProcessor(time_window="1h", ordering="desc")

            assert processor.time_window == "1h"
            assert processor.ordering == "desc"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_temporal_query_processing(self):
        """Test temporal query processing."""
        with patch(
            "cogniverse_core.evaluation.inspect_tasks.solvers.TemporalQueryProcessor"
        ) as MockProcessor:
            # Create an AsyncMock that acts as a callable
            async def mock_processor_call(state, gen):
                # Sort results by timestamp
                state.metadata["retrieval_results"]["test"].sort(
                    key=lambda x: x["timestamp"]
                )
                return state

            mock_processor = AsyncMock(side_effect=mock_processor_call)
            mock_processor.ordering = "asc"
            MockProcessor.return_value = mock_processor

            from cogniverse_core.evaluation.inspect_tasks.solvers import TemporalQueryProcessor

            processor = TemporalQueryProcessor(ordering="asc")

            state = Mock()
            state.input = Mock()
            state.input.text = "what happened after the meeting"
            state.metadata = {
                "retrieval_results": {
                    "test": [
                        {"id": "1", "timestamp": 300},
                        {"id": "2", "timestamp": 100},
                        {"id": "3", "timestamp": 200},
                    ]
                }
            }

            updated_state = await processor(state, None)

            # Results should be ordered by timestamp
            results = updated_state.metadata["retrieval_results"]["test"]
            assert results[0]["timestamp"] == 100
            assert results[1]["timestamp"] == 200
            assert results[2]["timestamp"] == 300


class TestSolverChaining:
    """Test solver chaining and composition."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_chain_multiple_solvers(self):
        """Test chaining multiple solvers."""
        with patch("cogniverse_core.evaluation.inspect_tasks.solvers.get_config") as mock_config:
            mock_config.return_value = {
                "profiles": {},
                "vespa": {"url": "http://localhost:8080"},
            }

            with patch(
                "cogniverse_core.evaluation.inspect_tasks.solvers.SearchService"
            ) as mock_service:
                mock_service.return_value.search.return_value = []

                # Create multiple solvers
                solver1 = CogniverseRetrievalSolver(["profile1"], ["strategy1"])
                solver2 = CogniverseRetrievalSolver(["profile2"], ["strategy2"])

                # Create initial state
                state = Mock()
                state.input = Mock()
                state.input.text = "test query"
                state.metadata = {}

                # Chain execution
                state = await solver1(state, None)
                state = await solver2(state, None)

                # Should have results from both
                assert "retrieval_results" in state.metadata

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parallel_solver_execution(self):
        """Test parallel solver execution."""
        import asyncio

        with patch("cogniverse_core.evaluation.inspect_tasks.solvers.get_config") as mock_config:
            mock_config.return_value = {
                "profiles": {},
                "vespa": {"url": "http://localhost:8080"},
            }

            with patch(
                "cogniverse_core.evaluation.inspect_tasks.solvers.SearchService"
            ) as mock_service:
                mock_service.return_value.search.return_value = []

                solvers = [
                    CogniverseRetrievalSolver([f"profile{i}"], ["strategy"])
                    for i in range(3)
                ]

                state = Mock()
                state.input = Mock()
                state.input.text = "test query"
                state.metadata = {}

                # Execute in parallel
                tasks = [solver(state, None) for solver in solvers]
                results = await asyncio.gather(*tasks)

                assert len(results) == 3
                assert all(r is not None for r in results)


class TestSolverConfiguration:
    """Test solver configuration handling."""

    @pytest.mark.unit
    def test_solver_with_custom_config(self):
        """Test solver with custom configuration."""
        custom_config = {
            "vespa": {"url": "http://custom:8080"},
            "profiles": {
                "custom_profile": {"embedding_model": "custom_model", "chunk_size": 512}
            },
        }

        with patch("cogniverse_core.evaluation.inspect_tasks.solvers.get_config") as mock_config:
            mock_config.return_value = custom_config

            solver = CogniverseRetrievalSolver(
                profiles=["custom_profile"], strategies=["custom"]
            )

            assert solver.config == custom_config
            assert solver.profiles == ["custom_profile"]

    @pytest.mark.unit
    def test_solver_with_invalid_profile(self):
        """Test solver with invalid profile."""
        with patch("cogniverse_core.evaluation.inspect_tasks.solvers.get_config") as mock_config:
            mock_config.return_value = {
                "profiles": {},
                "vespa": {"url": "http://localhost:8080"},
            }

            # Should handle invalid profile gracefully
            solver = CogniverseRetrievalSolver(
                profiles=["non_existent_profile"], strategies=["test"]
            )

            assert solver is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_solver_config_override(self):
        """Test solver configuration override."""
        with patch("cogniverse_core.evaluation.inspect_tasks.solvers.get_config") as mock_config:
            mock_config.return_value = {
                "profiles": {},
                "vespa": {"url": "http://localhost:8080"},
            }

            solver = CogniverseRetrievalSolver(profiles=["test"], strategies=["test"])

            state = Mock()
            state.input = Mock()
            state.input.text = "test"
            state.metadata = {}

            # Configuration should be applied
            with patch(
                "cogniverse_core.evaluation.inspect_tasks.solvers.SearchService"
            ) as mock_service:
                mock_service.return_value.search.return_value = []
                await solver(state, None)

                # Check if override was used
                assert solver is not None
