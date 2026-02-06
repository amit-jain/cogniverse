"""
Unit tests for OptimizerCoordinator facade.
"""

from unittest.mock import MagicMock, patch

import pytest

from cogniverse_agents.routing.optimizer_coordinator import (
    OptimizationType,
    OptimizerCoordinator,
)


class TestOptimizerCoordinator:
    """Test OptimizerCoordinator facade pattern and lazy loading."""

    @pytest.fixture
    def coordinator(self, tmp_path):
        """Create coordinator with temp optimization dir."""
        return OptimizerCoordinator(
            optimization_dir=str(tmp_path / "optimization"),
            tenant_id="test_tenant",
        )

    # --- Initialization ---

    def test_initialization(self, coordinator):
        """Test initial state: all optimizers are None."""
        assert coordinator.tenant_id == "test_tenant"
        assert coordinator._routing_optimizer is None
        assert coordinator._modality_optimizer is None
        assert coordinator._cross_modal_optimizer is None
        assert coordinator._unified_optimizer is None

    # --- Lazy Loading ---

    @patch(
        "cogniverse_agents.routing.advanced_optimizer.AdvancedRoutingOptimizer",
    )
    def test_lazy_load_routing(self, mock_cls, coordinator):
        """First call creates AdvancedRoutingOptimizer; second reuses."""
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        opt1 = coordinator._get_routing_optimizer()
        opt2 = coordinator._get_routing_optimizer()

        assert opt1 is mock_instance
        assert opt1 is opt2
        mock_cls.assert_called_once_with(
            tenant_id="test_tenant",
            base_storage_dir=coordinator.optimization_dir,
        )

    @patch(
        "cogniverse_agents.routing.modality_optimizer.ModalityOptimizer",
    )
    def test_lazy_load_modality(self, mock_cls, coordinator):
        """First call creates ModalityOptimizer; second reuses."""
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        opt1 = coordinator._get_modality_optimizer()
        opt2 = coordinator._get_modality_optimizer()

        assert opt1 is mock_instance
        assert opt1 is opt2
        mock_cls.assert_called_once()

    @patch(
        "cogniverse_agents.routing.cross_modal_optimizer.CrossModalOptimizer",
    )
    def test_lazy_load_cross_modal(self, mock_cls, coordinator):
        """First call creates CrossModalOptimizer; second reuses."""
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        opt1 = coordinator._get_cross_modal_optimizer()
        opt2 = coordinator._get_cross_modal_optimizer()

        assert opt1 is mock_instance
        assert opt1 is opt2
        mock_cls.assert_called_once()

    @patch(
        "cogniverse_agents.workflow_intelligence.WorkflowIntelligence",
    )
    @patch(
        "cogniverse_agents.routing.unified_optimizer.UnifiedOptimizer",
    )
    @patch(
        "cogniverse_agents.routing.advanced_optimizer.AdvancedRoutingOptimizer",
    )
    def test_lazy_load_unified(
        self, mock_routing_cls, mock_unified_cls, mock_wf_cls, coordinator
    ):
        """Unified creates routing optimizer + WorkflowIntelligence + UnifiedOptimizer."""
        mock_routing = MagicMock()
        mock_routing_cls.return_value = mock_routing
        mock_wf = MagicMock()
        mock_wf_cls.return_value = mock_wf
        mock_unified = MagicMock()
        mock_unified_cls.return_value = mock_unified

        opt = coordinator._get_unified_optimizer()

        assert opt is mock_unified
        # Routing optimizer should have been created as dependency
        assert coordinator._routing_optimizer is mock_routing
        mock_unified_cls.assert_called_once_with(
            routing_optimizer=mock_routing,
            workflow_intelligence=mock_wf,
        )

    # --- optimize() dispatch ---

    @patch(
        "cogniverse_agents.routing.advanced_optimizer.AdvancedRoutingOptimizer",
    )
    def test_optimize_routing(self, mock_cls, coordinator):
        """ROUTING type dispatches to optimizer.optimize_routing()."""
        mock_optimizer = MagicMock()
        mock_optimizer.optimize_routing.return_value = {"status": "success"}
        mock_cls.return_value = mock_optimizer

        data = [{"query": "test", "outcome": "good"}]
        result = coordinator.optimize(type=OptimizationType.ROUTING, training_data=data)

        assert result == {"status": "success"}
        mock_optimizer.optimize_routing.assert_called_once_with(training_data=data)

    @patch(
        "cogniverse_agents.routing.modality_optimizer.ModalityOptimizer",
    )
    def test_optimize_modality(self, mock_cls, coordinator):
        """MODALITY type dispatches to optimizer.train_modality_model()."""
        mock_optimizer = MagicMock()
        mock_optimizer.train_modality_model.return_value = {"trained": True}
        mock_cls.return_value = mock_optimizer

        data = [{"example": 1}]
        result = coordinator.optimize(
            type=OptimizationType.MODALITY,
            training_data=data,
            modality="video",
        )

        assert result == {"trained": True}
        mock_optimizer.train_modality_model.assert_called_once_with(
            modality="video", training_data=data
        )

    @patch(
        "cogniverse_agents.routing.modality_optimizer.ModalityOptimizer",
    )
    def test_optimize_modality_missing_param(self, mock_cls, coordinator):
        """MODALITY without modality kwarg raises ValueError."""
        mock_cls.return_value = MagicMock()

        with pytest.raises(
            ValueError, match="modality parameter required for MODALITY optimization"
        ):
            coordinator.optimize(
                type=OptimizationType.MODALITY,
                training_data=[{"x": 1}],
            )

    @patch(
        "cogniverse_agents.routing.cross_modal_optimizer.CrossModalOptimizer",
    )
    def test_optimize_cross_modal(self, mock_cls, coordinator):
        """CROSS_MODAL type dispatches to optimizer.optimize_fusion()."""
        mock_optimizer = MagicMock()
        mock_optimizer.optimize_fusion.return_value = {"fused": True}
        mock_cls.return_value = mock_optimizer

        data = [{"example": 1}]
        result = coordinator.optimize(
            type=OptimizationType.CROSS_MODAL, training_data=data
        )

        assert result == {"fused": True}
        mock_optimizer.optimize_fusion.assert_called_once_with(training_data=data)

    @patch(
        "cogniverse_agents.workflow_intelligence.WorkflowIntelligence",
    )
    @patch(
        "cogniverse_agents.routing.unified_optimizer.UnifiedOptimizer",
    )
    @patch(
        "cogniverse_agents.routing.advanced_optimizer.AdvancedRoutingOptimizer",
    )
    def test_optimize_unified(
        self, mock_routing_cls, mock_unified_cls, mock_wf_cls, coordinator
    ):
        """UNIFIED type dispatches to optimizer.optimize_unified()."""
        mock_routing_cls.return_value = MagicMock()
        mock_wf_cls.return_value = MagicMock()
        mock_unified = MagicMock()
        mock_unified.optimize_unified.return_value = {"unified": True}
        mock_unified_cls.return_value = mock_unified

        data = [{"example": 1}]
        result = coordinator.optimize(type=OptimizationType.UNIFIED, training_data=data)

        assert result == {"unified": True}
        mock_unified.optimize_unified.assert_called_once_with(training_data=data)

    def test_optimize_invalid_type(self, coordinator):
        """Unsupported optimization type raises ValueError."""
        # Force an invalid type through the else branch
        # OptimizationType.ORCHESTRATION is defined but not handled in optimize()
        with pytest.raises(ValueError, match="Unsupported optimization type"):
            coordinator.optimize(
                type=OptimizationType.ORCHESTRATION,
                training_data=[],
            )

    # --- get_optimizer() ---

    @patch(
        "cogniverse_agents.routing.cross_modal_optimizer.CrossModalOptimizer",
    )
    @patch(
        "cogniverse_agents.routing.modality_optimizer.ModalityOptimizer",
    )
    @patch(
        "cogniverse_agents.routing.advanced_optimizer.AdvancedRoutingOptimizer",
    )
    def test_get_optimizer(
        self, mock_routing_cls, mock_modality_cls, mock_cross_cls, coordinator
    ):
        """get_optimizer() returns correct optimizer for each type."""
        mock_routing = MagicMock()
        mock_routing_cls.return_value = mock_routing
        mock_modality = MagicMock()
        mock_modality_cls.return_value = mock_modality
        mock_cross = MagicMock()
        mock_cross_cls.return_value = mock_cross

        assert coordinator.get_optimizer(OptimizationType.ROUTING) is mock_routing
        assert coordinator.get_optimizer(OptimizationType.MODALITY) is mock_modality
        assert coordinator.get_optimizer(OptimizationType.CROSS_MODAL) is mock_cross

    def test_get_optimizer_invalid_type(self, coordinator):
        """get_optimizer() with unsupported type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported optimizer type"):
            coordinator.get_optimizer(OptimizationType.ORCHESTRATION)

    # --- get_optimization_status() ---

    def test_get_optimization_status_empty(self, coordinator):
        """Status reports no loaded optimizers initially."""
        status = coordinator.get_optimization_status()

        assert status["tenant_id"] == "test_tenant"
        assert status["loaded_optimizers"] == []

    @patch(
        "cogniverse_agents.routing.modality_optimizer.ModalityOptimizer",
    )
    @patch(
        "cogniverse_agents.routing.advanced_optimizer.AdvancedRoutingOptimizer",
    )
    def test_get_optimization_status_after_use(
        self, mock_routing_cls, mock_modality_cls, coordinator
    ):
        """Status reflects loaded optimizers after calls."""
        mock_routing_cls.return_value = MagicMock()
        mock_modality_cls.return_value = MagicMock()

        coordinator._get_routing_optimizer()
        coordinator._get_modality_optimizer()

        status = coordinator.get_optimization_status()
        assert "routing" in status["loaded_optimizers"]
        assert "modality" in status["loaded_optimizers"]
        assert "cross_modal" not in status["loaded_optimizers"]
        assert "unified" not in status["loaded_optimizers"]
