"""
Unit tests for LazyModalityExecutor
"""


import pytest

from src.app.routing.lazy_executor import LazyModalityExecutor
from src.app.search.multi_modal_reranker import QueryModality


class TestLazyModalityExecutor:
    """Test LazyModalityExecutor functionality"""

    @pytest.fixture
    def executor(self):
        """Create executor instance"""
        return LazyModalityExecutor(default_quality_threshold=0.8)

    @pytest.fixture
    def mock_modality_executor(self):
        """Create mock modality executor"""
        async def executor(query, modality, context):
            # Return different results based on modality
            return {
                "results": [
                    {"content": f"result_{i}", "confidence": 0.9}
                    for i in range(10)
                ],
                "confidence": 0.9,
            }
        return executor

    def test_initialization(self, executor):
        """Test executor initialization"""
        assert executor.default_quality_threshold == 0.8
        assert executor.execution_stats["total_executions"] == 0

    def test_modality_cost_ranking(self, executor):
        """Test modality cost ranking"""
        # TEXT should be cheapest
        assert executor.MODALITY_COST[QueryModality.TEXT] < executor.MODALITY_COST[QueryModality.VIDEO]
        assert executor.MODALITY_COST[QueryModality.DOCUMENT] < executor.MODALITY_COST[QueryModality.VIDEO]
        # VIDEO should be cheaper than AUDIO
        assert executor.MODALITY_COST[QueryModality.VIDEO] < executor.MODALITY_COST[QueryModality.AUDIO]

    @pytest.mark.asyncio
    async def test_execute_single_modality(self, executor, mock_modality_executor):
        """Test executing single modality"""
        result = await executor.execute_with_lazy_evaluation(
            query="test query",
            modalities=[QueryModality.VIDEO],
            context={},
            modality_executor=mock_modality_executor,
        )

        assert QueryModality.VIDEO in result["results"]
        assert len(result["executed_modalities"]) == 1
        assert len(result["skipped_modalities"]) == 0
        assert result["early_stopped"] is False

    @pytest.mark.asyncio
    async def test_execute_multiple_modalities_in_cost_order(self, executor, mock_modality_executor):
        """Test modalities executed in cost order"""
        # Provide modalities in wrong order
        modalities = [QueryModality.VIDEO, QueryModality.TEXT, QueryModality.AUDIO]

        result = await executor.execute_with_lazy_evaluation(
            query="test",
            modalities=modalities,
            context={"quality_threshold": 0.0, "min_results_required": 100},  # Don't stop early
            modality_executor=mock_modality_executor,
        )

        # Should be executed in cost order: TEXT, VIDEO, AUDIO
        executed = result["executed_modalities"]
        assert len(executed) == 3
        assert executed[0] == "text"
        assert executed[1] == "video"
        assert executed[2] == "audio"

    @pytest.mark.asyncio
    async def test_early_stopping_with_sufficient_results(self, executor):
        """Test early stopping when results are sufficient"""
        async def executor_with_good_results(query, modality, context):
            # Return high-quality results
            return {
                "results": [
                    {"content": f"result_{i}", "confidence": 0.95}
                    for i in range(20)
                ],
                "confidence": 0.95,
            }

        modalities = [QueryModality.TEXT, QueryModality.VIDEO, QueryModality.AUDIO]

        result = await executor.execute_with_lazy_evaluation(
            query="test",
            modalities=modalities,
            context={"quality_threshold": 0.8, "min_results_required": 5},
            modality_executor=executor_with_good_results,
        )

        # Should stop early after TEXT (cheap and sufficient)
        assert result["early_stopped"] is True
        assert len(result["executed_modalities"]) < len(modalities)
        assert len(result["skipped_modalities"]) > 0

    @pytest.mark.asyncio
    async def test_no_early_stopping_with_insufficient_results(self, executor):
        """Test no early stopping when results insufficient"""
        async def executor_with_poor_results(query, modality, context):
            # Return low-quality results
            return {
                "results": [
                    {"content": f"result_{i}", "confidence": 0.3}
                    for i in range(2)
                ],
                "confidence": 0.3,
            }

        modalities = [QueryModality.TEXT, QueryModality.VIDEO]

        result = await executor.execute_with_lazy_evaluation(
            query="test",
            modalities=modalities,
            context={"quality_threshold": 0.8, "min_results_required": 10},
            modality_executor=executor_with_poor_results,
        )

        # Should execute all modalities
        assert result["early_stopped"] is False
        assert len(result["executed_modalities"]) == len(modalities)
        assert len(result["skipped_modalities"]) == 0

    @pytest.mark.asyncio
    async def test_early_stop_after_expensive_modality(self, executor):
        """Test early stopping after executing expensive modality"""
        async def executor_func(query, modality, context):
            if modality in [QueryModality.VIDEO, QueryModality.TEXT]:
                # Both return great results
                return {
                    "results": [{"content": f"{modality.value}_{i}", "confidence": 0.95} for i in range(15)],
                    "confidence": 0.95,
                }
            else:
                return {"results": [], "confidence": 0.5}

        modalities = [QueryModality.TEXT, QueryModality.VIDEO, QueryModality.AUDIO]

        result = await executor.execute_with_lazy_evaluation(
            query="test",
            modalities=modalities,
            context={"quality_threshold": 0.8, "min_results_required": 5},
            modality_executor=executor_func,
        )

        # Should stop early after TEXT or VIDEO (good results from cheaper modality)
        assert result["early_stopped"] is True
        assert QueryModality.AUDIO.value in result["skipped_modalities"]

    @pytest.mark.asyncio
    async def test_count_results_with_list(self, executor, mock_modality_executor):
        """Test counting results from list"""
        result_list = [{"item": 1}, {"item": 2}, {"item": 3}]
        count = executor._count_results(result_list)
        assert count == 3

    @pytest.mark.asyncio
    async def test_count_results_with_dict(self, executor, mock_modality_executor):
        """Test counting results from dict"""
        result_dict = {"results": [1, 2, 3, 4, 5]}
        count = executor._count_results(result_dict)
        assert count == 5

    @pytest.mark.asyncio
    async def test_calculate_avg_confidence_from_results(self, executor):
        """Test calculating average confidence"""
        results = {
            QueryModality.VIDEO: {
                "results": [
                    {"confidence": 0.9},
                    {"confidence": 0.8},
                    {"confidence": 0.7},
                ]
            }
        }

        avg = executor._calculate_avg_confidence(results)
        assert 0.79 < avg < 0.81  # Average of 0.9, 0.8, 0.7

    @pytest.mark.asyncio
    async def test_calculate_avg_confidence_no_confidence_field(self, executor):
        """Test calculating confidence when no confidence field"""
        results = {
            QueryModality.VIDEO: {
                "results": [
                    {"data": "item1"},
                    {"data": "item2"},
                ]
            }
        }

        avg = executor._calculate_avg_confidence(results)
        assert avg == 0.7  # Default

    @pytest.mark.asyncio
    async def test_calculate_avg_confidence_with_score_field(self, executor):
        """Test using score field as fallback for confidence"""
        results = {
            QueryModality.VIDEO: {
                "results": [
                    {"score": 0.95},
                    {"score": 0.85},
                ]
            }
        }

        avg = executor._calculate_avg_confidence(results)
        assert 0.89 < avg < 0.91  # Average of 0.95, 0.85

    @pytest.mark.asyncio
    async def test_get_stats(self, executor, mock_modality_executor):
        """Test getting execution statistics"""
        # Execute once with early stop
        modalities = [QueryModality.TEXT, QueryModality.VIDEO, QueryModality.AUDIO]

        await executor.execute_with_lazy_evaluation(
            query="test",
            modalities=modalities,
            context={"quality_threshold": 0.7},
            modality_executor=mock_modality_executor,
        )

        stats = executor.get_stats()

        assert stats["total_executions"] == 1
        assert "early_stop_rate" in stats
        assert "avg_modalities_skipped" in stats

    @pytest.mark.asyncio
    async def test_reset_stats(self, executor, mock_modality_executor):
        """Test resetting statistics"""
        await executor.execute_with_lazy_evaluation(
            query="test",
            modalities=[QueryModality.TEXT],
            context={},
            modality_executor=mock_modality_executor,
        )

        executor.reset_stats()

        stats = executor.get_stats()
        assert stats["total_executions"] == 0
        assert stats["early_stops"] == 0

    @pytest.mark.asyncio
    async def test_total_cost_calculation(self, executor, mock_modality_executor):
        """Test total cost calculation"""
        modalities = [QueryModality.TEXT, QueryModality.VIDEO]

        result = await executor.execute_with_lazy_evaluation(
            query="test",
            modalities=modalities,
            context={"quality_threshold": 0.0, "min_results_required": 100},  # Execute all
            modality_executor=mock_modality_executor,
        )

        expected_cost = executor.MODALITY_COST[QueryModality.TEXT] + executor.MODALITY_COST[QueryModality.VIDEO]
        assert result["total_cost"] == expected_cost


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
