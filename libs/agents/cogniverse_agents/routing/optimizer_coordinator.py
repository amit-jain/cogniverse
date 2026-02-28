"""
OptimizerCoordinator - Unified interface for all routing optimizers.

Provides a facade pattern to route optimization requests to the appropriate
specialized optimizer without requiring callers to know which optimizer to use.
"""

import logging
from enum import Enum
from typing import Any, Dict, List

from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimization supported"""

    ROUTING = "routing"  # AdvancedRoutingOptimizer
    MODALITY = "modality"  # ModalityOptimizer
    CROSS_MODAL = "cross_modal"  # CrossModalOptimizer
    UNIFIED = "unified"  # UnifiedOptimizer
    ORCHESTRATION = "orchestration"  # For workflow optimization


class OptimizerCoordinator:
    """
    Facade for all routing optimizers.

    Routes optimization requests to the appropriate specialized optimizer
    based on the optimization type and context.

    Example:
        coordinator = OptimizerCoordinator()

        # Automatically routes to AdvancedRoutingOptimizer
        coordinator.optimize(
            type=OptimizationType.ROUTING,
            training_data=data
        )

        # Automatically routes to ModalityOptimizer
        coordinator.optimize(
            type=OptimizationType.MODALITY,
            modality="video",
            training_data=data
        )
    """

    def __init__(
        self,
        llm_config: LLMEndpointConfig,
        telemetry_provider: TelemetryProvider,
        tenant_id: str = "default",
    ):
        """
        Initialize optimizer coordinator.

        Args:
            llm_config: LLM endpoint configuration for optimizers
            telemetry_provider: Telemetry provider for artifact persistence
            tenant_id: Tenant identifier for multi-tenant setups
        """
        self.llm_config = llm_config
        self.telemetry_provider = telemetry_provider
        self.tenant_id = tenant_id

        # Lazy-load optimizers on demand
        self._routing_optimizer = None
        self._modality_optimizer = None
        self._cross_modal_optimizer = None
        self._unified_optimizer = None

        logger.info(
            f"Initialized OptimizerCoordinator (tenant={tenant_id})"
        )

    def _get_routing_optimizer(self):
        """Lazy-load AdvancedRoutingOptimizer"""
        if self._routing_optimizer is None:
            from cogniverse_agents.routing.advanced_optimizer import (
                AdvancedRoutingOptimizer,
            )

            self._routing_optimizer = AdvancedRoutingOptimizer(
                tenant_id=self.tenant_id,
                llm_config=self.llm_config,
                telemetry_provider=self.telemetry_provider,
            )
        return self._routing_optimizer

    def _get_modality_optimizer(self):
        """Lazy-load ModalityOptimizer"""
        if self._modality_optimizer is None:
            from cogniverse_agents.routing.modality_optimizer import ModalityOptimizer

            self._modality_optimizer = ModalityOptimizer(
                llm_config=self.llm_config,
                telemetry_provider=self.telemetry_provider,
                tenant_id=self.tenant_id,
            )
        return self._modality_optimizer

    def _get_cross_modal_optimizer(self):
        """Lazy-load CrossModalOptimizer"""
        if self._cross_modal_optimizer is None:
            from cogniverse_agents.routing.cross_modal_optimizer import (
                CrossModalOptimizer,
            )

            self._cross_modal_optimizer = CrossModalOptimizer(
                telemetry_provider=self.telemetry_provider,
                tenant_id=self.tenant_id,
            )
        return self._cross_modal_optimizer

    def _get_unified_optimizer(self):
        """Lazy-load UnifiedOptimizer"""
        if self._unified_optimizer is None:
            from cogniverse_agents.routing.unified_optimizer import UnifiedOptimizer
            from cogniverse_agents.workflow.intelligence import WorkflowIntelligence

            # UnifiedOptimizer requires routing_optimizer and workflow_intelligence
            routing_optimizer = self._get_routing_optimizer()
            workflow_intelligence = WorkflowIntelligence(
                telemetry_provider=self.telemetry_provider,
                tenant_id=self.tenant_id,
            )

            self._unified_optimizer = UnifiedOptimizer(
                routing_optimizer=routing_optimizer,
                workflow_intelligence=workflow_intelligence,
            )
        return self._unified_optimizer

    def optimize(
        self,
        type: OptimizationType,
        training_data: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run optimization using the appropriate optimizer.

        Args:
            type: Type of optimization to perform
            training_data: Training examples
            **kwargs: Additional optimizer-specific parameters

        Returns:
            Optimization results

        Raises:
            ValueError: If optimization type is not supported
        """
        logger.info(
            f"Coordinating {type.value} optimization with {len(training_data)} examples"
        )

        if type == OptimizationType.ROUTING:
            optimizer = self._get_routing_optimizer()
            return optimizer.optimize_routing(training_data=training_data, **kwargs)

        elif type == OptimizationType.MODALITY:
            optimizer = self._get_modality_optimizer()
            modality = kwargs.pop("modality", None)
            if not modality:
                raise ValueError(
                    "modality parameter required for MODALITY optimization"
                )
            return optimizer.train_modality_model(
                modality=modality, training_data=training_data, **kwargs
            )

        elif type == OptimizationType.CROSS_MODAL:
            optimizer = self._get_cross_modal_optimizer()
            return optimizer.optimize_fusion(training_data=training_data, **kwargs)

        elif type == OptimizationType.UNIFIED:
            optimizer = self._get_unified_optimizer()
            return optimizer.optimize_unified(training_data=training_data, **kwargs)

        else:
            raise ValueError(f"Unsupported optimization type: {type}")

    def get_optimizer(self, type: OptimizationType):
        """
        Get direct access to a specific optimizer.

        Use this when you need to call optimizer-specific methods
        that aren't exposed through the coordinator.

        Args:
            type: Optimizer type to retrieve

        Returns:
            The requested optimizer instance
        """
        if type == OptimizationType.ROUTING:
            return self._get_routing_optimizer()
        elif type == OptimizationType.MODALITY:
            return self._get_modality_optimizer()
        elif type == OptimizationType.CROSS_MODAL:
            return self._get_cross_modal_optimizer()
        elif type == OptimizationType.UNIFIED:
            return self._get_unified_optimizer()
        else:
            raise ValueError(f"Unsupported optimizer type: {type}")

    def get_optimization_status(self) -> Dict[str, Any]:
        """Get status of all loaded optimizers"""
        status = {
            "tenant_id": self.tenant_id,
            "loaded_optimizers": [],
        }

        if self._routing_optimizer:
            status["loaded_optimizers"].append("routing")
        if self._modality_optimizer:
            status["loaded_optimizers"].append("modality")
        if self._cross_modal_optimizer:
            status["loaded_optimizers"].append("cross_modal")
        if self._unified_optimizer:
            status["loaded_optimizers"].append("unified")

        return status
