"""
Integration tests for SyntheticDataService

Tests the main service orchestrator end-to-end.
"""

import pytest

from cogniverse_foundation.config.unified_config import (
    DSPyModuleConfig,
    OptimizerGenerationConfig,
    SyntheticGeneratorConfig,
)
from cogniverse_synthetic.schemas import SyntheticDataRequest
from cogniverse_synthetic.service import SyntheticDataService

pytestmark = [pytest.mark.unit]


def create_test_generator_config():
    """Create test generator configuration with all required optimizer configs"""
    return SyntheticGeneratorConfig(
        tenant_id="test:unit",
        optimizer_configs={
            "routing": OptimizerGenerationConfig(
                optimizer_type="routing",
                dspy_modules={
                    "query_generator": DSPyModuleConfig(
                        signature_class="cogniverse_synthetic.dspy_signatures.GenerateEntityQuery",
                        module_type="Predict",
                    )
                },
            ),
        },
    )


class TestSyntheticDataService:
    """Integration tests for SyntheticDataService"""

    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test service can be initialized"""
        service = SyntheticDataService(generator_config=create_test_generator_config())
        assert service.profile_selector is not None
        assert service.backend_querier is not None
        assert service.pattern_extractor is not None
        assert service.agent_inferrer is not None
        # Generators are initialized lazily, so check starts empty
        assert isinstance(service.generators, dict)

    @pytest.mark.asyncio
    async def test_service_with_backend(self):
        """Test service can be initialized with Backend interface"""
        mock_backend = type(
            "MockBackend", (), {"query_metadata_documents": lambda *args, **kwargs: []}
        )()
        service = SyntheticDataService(
            backend=mock_backend, generator_config=create_test_generator_config()
        )
        assert service.backend == mock_backend

    @pytest.mark.asyncio
    async def test_service_with_backend_config(self):
        """Test service with backend configuration"""
        from cogniverse_foundation.config.unified_config import BackendConfig

        config = BackendConfig(profiles={}, tenant_id="test:unit")
        service = SyntheticDataService(
            backend_config=config, generator_config=create_test_generator_config()
        )
        assert service.backend_config == config

    @pytest.mark.asyncio
    async def test_generate_profile_examples(self):
        """Test generating profile selection examples"""
        service = SyntheticDataService(generator_config=create_test_generator_config())

        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="profile", count=10
        )

        response = await service.generate(request)

        assert response.optimizer == "profile"
        assert response.count == 10
        assert response.schema_name == "ProfileSelectionExampleSchema"
        assert len(response.data) == 10
        assert isinstance(response.selected_profiles, list)
        assert len(response.selected_profiles) > 0
        assert isinstance(response.metadata, dict)
        assert isinstance(response.profile_selection_reasoning, str)

    @pytest.mark.asyncio
    async def test_generate_routing_examples(self):
        """Test generating routing experience examples"""
        service = SyntheticDataService(generator_config=create_test_generator_config())

        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="routing", count=12
        )

        response = await service.generate(request)

        assert response.optimizer == "routing"
        assert response.count == 12
        assert response.schema_name == "RoutingExperienceSchema"
        assert len(response.data) == 12

    @pytest.mark.asyncio
    async def test_generate_workflow_examples(self):
        """Test generating workflow execution examples"""
        service = SyntheticDataService(generator_config=create_test_generator_config())

        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="workflow", count=20
        )

        response = await service.generate(request)

        assert response.optimizer == "workflow"
        assert response.count == 20
        assert response.schema_name == "WorkflowExecutionSchema"
        assert len(response.data) == 20

    @pytest.mark.asyncio
    async def test_generate_with_custom_sample_size(self):
        """Test generation with custom sample size"""
        service = SyntheticDataService(generator_config=create_test_generator_config())

        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="profile", count=10, vespa_sample_size=50
        )

        response = await service.generate(request)

        assert response.count == 10
        assert response.metadata["vespa_sample_size"] == 50

    @pytest.mark.asyncio
    async def test_generate_with_max_profiles(self):
        """Test generation with max profiles setting"""
        service = SyntheticDataService(generator_config=create_test_generator_config())

        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="profile", count=5, max_profiles=1
        )

        response = await service.generate(request)

        assert response.count == 5
        assert len(response.selected_profiles) <= 1

    @pytest.mark.asyncio
    async def test_generate_with_strategies(self):
        """Test generation with specific sampling strategies"""
        service = SyntheticDataService(generator_config=create_test_generator_config())

        request = SyntheticDataRequest(
            tenant_id="test:unit",
            optimizer="routing",
            count=8,
            strategies=["entity_rich"],
        )

        response = await service.generate(request)

        assert response.count == 8

    @pytest.mark.asyncio
    async def test_generate_invalid_optimizer(self):
        """Test generation with invalid optimizer name"""
        service = SyntheticDataService(generator_config=create_test_generator_config())

        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="nonexistent_optimizer", count=10
        )

        with pytest.raises(ValueError, match="Unknown optimizer"):
            await service.generate(request)

    @pytest.mark.asyncio
    async def test_get_optimizer_info(self):
        """Test getting optimizer information"""
        service = SyntheticDataService(generator_config=create_test_generator_config())

        info = service.get_optimizer_info("profile")

        assert info["name"] == "profile"
        assert "description" in info
        assert info["schema"] == "ProfileSelectionExampleSchema"
        assert info["generator"] == "ProfileGenerator"
        assert info["backend_strategy"] == "diverse"
        assert info["requires_agent_mapping"] is False
        assert "defaults" in info
        # Note: "generator_info" is only present if generator has been initialized (lazy init)

    @pytest.mark.asyncio
    async def test_get_optimizer_info_all_optimizers(self):
        """Test getting info for all optimizers"""
        service = SyntheticDataService(generator_config=create_test_generator_config())

        for optimizer_name in ["routing", "workflow", "profile", "unified"]:
            info = service.get_optimizer_info(optimizer_name)
            assert info["name"] == optimizer_name
            assert "description" in info
            assert "schema" in info
            assert "generator" in info

    @pytest.mark.asyncio
    async def test_list_all_optimizers(self):
        """Test listing all available optimizers"""
        service = SyntheticDataService(generator_config=create_test_generator_config())

        all_optimizers = service.list_all_optimizers()

        assert len(all_optimizers) >= 3
        assert "routing" in all_optimizers
        assert "workflow" in all_optimizers
        assert "profile" in all_optimizers

        for name, info in all_optimizers.items():
            assert "name" in info
            assert "description" in info
            assert "schema" in info

    @pytest.mark.asyncio
    async def test_service_orchestration_flow(self):
        """Test complete service orchestration flow"""
        # This test validates the entire pipeline:
        # Request -> Profile Selection -> Backend Query -> Generation -> Response

        service = SyntheticDataService(generator_config=create_test_generator_config())

        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="routing", count=8, vespa_sample_size=20
        )

        response = await service.generate(request)

        # Validate response structure
        assert response.optimizer == "routing"
        assert response.count == 8
        assert len(response.data) == 8
        assert len(response.selected_profiles) > 0

        # Validate metadata
        assert "sampled_content_count" in response.metadata
        assert response.metadata["target_count"] == 8

        # Validate examples are proper dicts
        for example in response.data:
            assert isinstance(example, dict)
            assert "query" in example
            assert "entities" in example
            assert "enhanced_query" in example


class TestServiceErrorHandling:
    """Test error handling in SyntheticDataService"""

    @pytest.mark.asyncio
    async def test_invalid_optimizer_in_generate(self):
        """Test error handling for invalid optimizer"""
        service = SyntheticDataService(generator_config=create_test_generator_config())

        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="invalid_name", count=10
        )

        with pytest.raises(ValueError) as exc_info:
            await service.generate(request)

        assert "Unknown optimizer" in str(exc_info.value)
        assert "invalid_name" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_optimizer_in_get_info(self):
        """Test error handling for get_optimizer_info"""
        service = SyntheticDataService(generator_config=create_test_generator_config())

        with pytest.raises(ValueError) as exc_info:
            service.get_optimizer_info("nonexistent")

        assert "Unknown optimizer" in str(exc_info.value)


class TestServiceWithBackendConfig:
    """Test service with various backend configurations"""

    @pytest.mark.asyncio
    async def test_service_uses_backend_config_profiles(self):
        """Test that service uses profiles from backend config"""
        from cogniverse_foundation.config.unified_config import BackendConfig

        config = BackendConfig(profiles={}, tenant_id="test:unit")

        service = SyntheticDataService(
            backend_config=config, generator_config=create_test_generator_config()
        )

        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="profile", count=5
        )

        response = await service.generate(request)

        # Should use profiles from config
        assert len(response.selected_profiles) > 0

    @pytest.mark.asyncio
    async def test_service_without_backend_config_uses_defaults(self):
        """Test service falls back to default profiles when no config"""
        service = SyntheticDataService(generator_config=create_test_generator_config())

        request = SyntheticDataRequest(
            tenant_id="test:unit", optimizer="profile", count=5
        )

        response = await service.generate(request)

        # Should use default profiles
        assert len(response.selected_profiles) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
