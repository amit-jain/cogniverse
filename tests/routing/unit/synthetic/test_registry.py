"""
Unit tests for optimizer registry
"""

import pytest

from cogniverse_synthetic.registry import (
    OPTIMIZER_REGISTRY,
    OptimizerConfig,
    get_optimizer_config,
    get_optimizer_schema,
    list_optimizers,
    validate_optimizer_exists,
)
from cogniverse_synthetic.schemas import (
    FusionHistorySchema,
    ModalityExampleSchema,
    RoutingExperienceSchema,
    WorkflowExecutionSchema,
)


class TestOptimizerConfig:
    """Test OptimizerConfig dataclass"""

    def test_optimizer_config_creation(self):
        """Test creating OptimizerConfig"""
        config = OptimizerConfig(
            name="test",
            description="Test optimizer",
            schema_class=ModalityExampleSchema,
            generator_class_name="TestGenerator",
            backend_query_strategy="test_strategy",
            agent_mapping_required=True,
        )

        assert config.name == "test"
        assert config.description == "Test optimizer"
        assert config.schema_class == ModalityExampleSchema
        assert config.generator_class_name == "TestGenerator"
        assert config.backend_query_strategy == "test_strategy"
        assert config.agent_mapping_required is True
        assert config.default_sample_size == 200
        assert config.default_generation_count == 100

    def test_optimizer_config_repr(self):
        """Test OptimizerConfig string representation"""
        config = OptimizerConfig(
            name="test",
            description="Test",
            schema_class=ModalityExampleSchema,
            generator_class_name="TestGenerator",
            backend_query_strategy="test_strategy",
            agent_mapping_required=True,
        )

        repr_str = repr(config)
        assert "test" in repr_str
        assert "ModalityExampleSchema" in repr_str
        assert "test_strategy" in repr_str


class TestOptimizerRegistry:
    """Test OPTIMIZER_REGISTRY structure and contents"""

    def test_registry_not_empty(self):
        """Test registry contains optimizers"""
        assert len(OPTIMIZER_REGISTRY) > 0

    def test_registry_has_required_optimizers(self):
        """Test registry contains all required optimizers"""
        required_optimizers = [
            "modality",
            "cross_modal",
            "routing",
            "workflow",
            "unified",
        ]

        for optimizer in required_optimizers:
            assert optimizer in OPTIMIZER_REGISTRY, f"Missing optimizer: {optimizer}"

    def test_modality_optimizer_config(self):
        """Test modality optimizer configuration"""
        config = OPTIMIZER_REGISTRY["modality"]

        assert config.name == "modality"
        assert config.schema_class == ModalityExampleSchema
        assert config.generator_class_name == "ModalityGenerator"
        assert config.backend_query_strategy == "by_modality"
        assert config.agent_mapping_required is True
        assert len(config.description) > 0

    def test_cross_modal_optimizer_config(self):
        """Test cross_modal optimizer configuration"""
        config = OPTIMIZER_REGISTRY["cross_modal"]

        assert config.name == "cross_modal"
        assert config.schema_class == FusionHistorySchema
        assert config.generator_class_name == "CrossModalGenerator"
        assert config.backend_query_strategy == "cross_modal_pairs"
        assert config.agent_mapping_required is False
        assert len(config.description) > 0

    def test_routing_optimizer_config(self):
        """Test routing optimizer configuration"""
        config = OPTIMIZER_REGISTRY["routing"]

        assert config.name == "routing"
        assert config.schema_class == RoutingExperienceSchema
        assert config.generator_class_name == "RoutingGenerator"
        assert config.backend_query_strategy == "entity_rich"
        assert config.agent_mapping_required is True
        assert len(config.description) > 0

    def test_workflow_optimizer_config(self):
        """Test workflow optimizer configuration"""
        config = OPTIMIZER_REGISTRY["workflow"]

        assert config.name == "workflow"
        assert config.schema_class == WorkflowExecutionSchema
        assert config.generator_class_name == "WorkflowGenerator"
        assert config.backend_query_strategy == "multi_modal_sequences"
        assert config.agent_mapping_required is True
        assert len(config.description) > 0

    def test_unified_optimizer_config(self):
        """Test unified optimizer configuration"""
        config = OPTIMIZER_REGISTRY["unified"]

        assert config.name == "unified"
        assert config.schema_class == WorkflowExecutionSchema
        assert config.generator_class_name == "WorkflowGenerator"
        assert config.backend_query_strategy == "multi_modal_sequences"
        assert config.agent_mapping_required is True
        assert len(config.description) > 0

    def test_all_configs_have_descriptions(self):
        """Test all optimizer configs have non-empty descriptions"""
        for name, config in OPTIMIZER_REGISTRY.items():
            assert len(config.description) > 0, f"Optimizer {name} has no description"

    def test_all_configs_have_valid_schemas(self):
        """Test all optimizer configs have valid Pydantic schemas"""
        from pydantic import BaseModel

        for name, config in OPTIMIZER_REGISTRY.items():
            assert issubclass(
                config.schema_class, BaseModel
            ), f"Optimizer {name} schema is not a Pydantic BaseModel"

    def test_all_configs_have_generator_names(self):
        """Test all optimizer configs have generator class names"""
        for name, config in OPTIMIZER_REGISTRY.items():
            assert (
                len(config.generator_class_name) > 0
            ), f"Optimizer {name} has no generator class name"
            assert config.generator_class_name.endswith(
                "Generator"
            ), f"Optimizer {name} generator name doesn't end with 'Generator'"

    def test_all_configs_have_query_strategies(self):
        """Test all optimizer configs have backend query strategies"""
        for name, config in OPTIMIZER_REGISTRY.items():
            assert (
                len(config.backend_query_strategy) > 0
            ), f"Optimizer {name} has no backend query strategy"


class TestGetOptimizerConfig:
    """Test get_optimizer_config function"""

    def test_get_valid_optimizer_config(self):
        """Test getting config for valid optimizer"""
        config = get_optimizer_config("modality")

        assert isinstance(config, OptimizerConfig)
        assert config.name == "modality"

    def test_get_invalid_optimizer_config(self):
        """Test getting config for invalid optimizer raises error"""
        with pytest.raises(ValueError) as exc_info:
            get_optimizer_config("nonexistent")

        assert "Unknown optimizer" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)
        assert "Available optimizers" in str(exc_info.value)

    def test_get_all_optimizer_configs(self):
        """Test getting all optimizer configs"""
        for optimizer_name in OPTIMIZER_REGISTRY.keys():
            config = get_optimizer_config(optimizer_name)
            assert config.name == optimizer_name


class TestListOptimizers:
    """Test list_optimizers function"""

    def test_list_optimizers_returns_dict(self):
        """Test list_optimizers returns dictionary"""
        optimizers = list_optimizers()

        assert isinstance(optimizers, dict)
        assert len(optimizers) > 0

    def test_list_optimizers_contains_all_optimizers(self):
        """Test list_optimizers contains all registry optimizers"""
        optimizers = list_optimizers()

        for name in OPTIMIZER_REGISTRY.keys():
            assert name in optimizers

    def test_list_optimizers_has_descriptions(self):
        """Test all listed optimizers have descriptions"""
        optimizers = list_optimizers()

        for name, description in optimizers.items():
            assert isinstance(description, str)
            assert len(description) > 0


class TestGetOptimizerSchema:
    """Test get_optimizer_schema function"""

    def test_get_valid_optimizer_schema(self):
        """Test getting schema for valid optimizer"""
        from pydantic import BaseModel

        schema = get_optimizer_schema("modality")

        assert issubclass(schema, BaseModel)
        assert schema == ModalityExampleSchema

    def test_get_invalid_optimizer_schema(self):
        """Test getting schema for invalid optimizer raises error"""
        with pytest.raises(ValueError):
            get_optimizer_schema("nonexistent")

    def test_get_all_optimizer_schemas(self):
        """Test getting schemas for all optimizers"""
        expected_schemas = {
            "modality": ModalityExampleSchema,
            "cross_modal": FusionHistorySchema,
            "routing": RoutingExperienceSchema,
            "workflow": WorkflowExecutionSchema,
            "unified": WorkflowExecutionSchema,
        }

        for optimizer_name, expected_schema in expected_schemas.items():
            schema = get_optimizer_schema(optimizer_name)
            assert schema == expected_schema


class TestValidateOptimizerExists:
    """Test validate_optimizer_exists function"""

    def test_validate_existing_optimizer(self):
        """Test validating existing optimizer returns True"""
        assert validate_optimizer_exists("modality") is True
        assert validate_optimizer_exists("cross_modal") is True
        assert validate_optimizer_exists("routing") is True
        assert validate_optimizer_exists("workflow") is True
        assert validate_optimizer_exists("unified") is True

    def test_validate_nonexistent_optimizer(self):
        """Test validating nonexistent optimizer returns False"""
        assert validate_optimizer_exists("nonexistent") is False
        assert validate_optimizer_exists("invalid") is False
        assert validate_optimizer_exists("") is False

    def test_validate_all_registry_optimizers(self):
        """Test all registry optimizers validate as existing"""
        for optimizer_name in OPTIMIZER_REGISTRY.keys():
            assert validate_optimizer_exists(optimizer_name) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
