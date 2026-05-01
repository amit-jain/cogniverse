"""
Synthetic Data Generation Package

Generates synthetic training examples for all optimizers in the system.
Uses backend schema introspection and agent-based profile selection.
"""

from cogniverse_synthetic.api import configure_service, router
from cogniverse_synthetic.registry import OPTIMIZER_REGISTRY, OptimizerConfig
from cogniverse_synthetic.schemas import (
    ProfileSelectionExampleSchema,
    RoutingExperienceSchema,
    SyntheticDataRequest,
    SyntheticDataResponse,
    WorkflowExecutionSchema,
)
from cogniverse_synthetic.service import SyntheticDataService

__all__ = [
    "OPTIMIZER_REGISTRY",
    "OptimizerConfig",
    "ProfileSelectionExampleSchema",
    "RoutingExperienceSchema",
    "WorkflowExecutionSchema",
    "SyntheticDataRequest",
    "SyntheticDataResponse",
    "SyntheticDataService",
    "router",
    "configure_service",
]
