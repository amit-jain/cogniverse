"""
Optimizer Registry

Maps optimizer names to their generator classes, schemas, and configurations.
Central registry for all synthetic data generation capabilities.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Type

from pydantic import BaseModel

from cogniverse_synthetic.schemas import (
    FusionHistorySchema,
    ModalityExampleSchema,
    RoutingExperienceSchema,
    WorkflowExecutionSchema,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """
    Configuration for an optimizer's synthetic data generation

    Defines all metadata needed to generate synthetic data for a specific optimizer.
    """

    name: str
    description: str
    schema_class: Type[BaseModel]
    generator_class_name: str  # String name to avoid circular imports
    backend_query_strategy: str
    agent_mapping_required: bool
    default_sample_size: int = 200
    default_generation_count: int = 100

    def __repr__(self) -> str:
        return (
            f"OptimizerConfig(name='{self.name}', "
            f"schema={self.schema_class.__name__}, "
            f"strategy='{self.backend_query_strategy}')"
        )


# Optimizer Registry
# Maps optimizer names to their configurations
OPTIMIZER_REGISTRY: Dict[str, OptimizerConfig] = {
    "modality": OptimizerConfig(
        name="modality",
        description="Per-modality routing optimization. Learns which agents handle different content types (video, document, image, audio) most effectively.",
        schema_class=ModalityExampleSchema,
        generator_class_name="ModalityGenerator",
        backend_query_strategy="by_modality",
        agent_mapping_required=True,
        default_sample_size=200,
        default_generation_count=100,
    ),
    "cross_modal": OptimizerConfig(
        name="cross_modal",
        description="Multi-modal fusion optimization. Learns when combining multiple modalities (e.g., video + document) improves results and by how much.",
        schema_class=FusionHistorySchema,
        generator_class_name="CrossModalGenerator",
        backend_query_strategy="cross_modal_pairs",
        agent_mapping_required=False,
        default_sample_size=200,
        default_generation_count=100,
    ),
    "routing": OptimizerConfig(
        name="routing",
        description="Advanced routing with entity extraction. Learns to route queries based on extracted entities, relationships, and semantic understanding.",
        schema_class=RoutingExperienceSchema,
        generator_class_name="RoutingGenerator",
        backend_query_strategy="entity_rich",
        agent_mapping_required=True,
        default_sample_size=200,
        default_generation_count=100,
    ),
    "workflow": OptimizerConfig(
        name="workflow",
        description="Workflow execution pattern optimization. Learns optimal agent sequences and parallel execution strategies for complex multi-step tasks.",
        schema_class=WorkflowExecutionSchema,
        generator_class_name="WorkflowGenerator",
        backend_query_strategy="multi_modal_sequences",
        agent_mapping_required=True,
        default_sample_size=200,
        default_generation_count=100,
    ),
    "unified": OptimizerConfig(
        name="unified",
        description="Unified routing and orchestration optimization. Combines routing decisions with workflow planning for end-to-end optimization.",
        schema_class=WorkflowExecutionSchema,
        generator_class_name="WorkflowGenerator",  # Same as workflow
        backend_query_strategy="multi_modal_sequences",
        agent_mapping_required=True,
        default_sample_size=200,
        default_generation_count=100,
    ),
}


def get_optimizer_config(optimizer_name: str) -> OptimizerConfig:
    """
    Get configuration for an optimizer

    Args:
        optimizer_name: Name of the optimizer

    Returns:
        OptimizerConfig for the optimizer

    Raises:
        ValueError: If optimizer name is not in registry
    """
    if optimizer_name not in OPTIMIZER_REGISTRY:
        available = ", ".join(OPTIMIZER_REGISTRY.keys())
        raise ValueError(
            f"Unknown optimizer: '{optimizer_name}'. Available optimizers: {available}"
        )

    return OPTIMIZER_REGISTRY[optimizer_name]


def list_optimizers() -> Dict[str, str]:
    """
    List all registered optimizers with descriptions

    Returns:
        Dictionary mapping optimizer names to descriptions
    """
    return {name: config.description for name, config in OPTIMIZER_REGISTRY.items()}


def get_optimizer_schema(optimizer_name: str) -> Type[BaseModel]:
    """
    Get the Pydantic schema class for an optimizer

    Args:
        optimizer_name: Name of the optimizer

    Returns:
        Pydantic BaseModel class for the optimizer's data schema

    Raises:
        ValueError: If optimizer name is not in registry
    """
    config = get_optimizer_config(optimizer_name)
    return config.schema_class


def validate_optimizer_exists(optimizer_name: str) -> bool:
    """
    Check if an optimizer exists in the registry

    Args:
        optimizer_name: Name of the optimizer

    Returns:
        True if optimizer exists, False otherwise
    """
    return optimizer_name in OPTIMIZER_REGISTRY


logger.info(f"Loaded optimizer registry with {len(OPTIMIZER_REGISTRY)} optimizers")
