"""
Unit tests for BaseGenerator
"""

from typing import Any, Dict, List

import pytest
from pydantic import BaseModel

from cogniverse_synthetic.generators.base import BaseGenerator

pytestmark = [pytest.mark.unit]


class MockSchema(BaseModel):
    """Mock schema for testing"""

    value: str


class ConcreteGenerator(BaseGenerator):
    """Concrete implementation of BaseGenerator for testing"""

    async def generate(
        self, sampled_content: List[Dict[str, Any]], target_count: int, **kwargs
    ) -> List[BaseModel]:
        """Simple implementation that returns mock data"""
        self.validate_inputs(sampled_content, target_count)
        return [MockSchema(value=f"item_{i}") for i in range(target_count)]


class TestBaseGenerator:
    """Test BaseGenerator abstract class and interface"""

    def test_concrete_generator_initialization(self):
        """Test initializing concrete generator"""
        generator = ConcreteGenerator()

        assert generator.agent_inferrer is None
        assert not hasattr(generator, "pattern_extractor")

    def test_generator_with_utilities(self):
        """Test initializing generator with utilities"""
        mock_inferrer = object()

        generator = ConcreteGenerator(agent_inferrer=mock_inferrer)

        assert generator.agent_inferrer is mock_inferrer
        with pytest.raises(TypeError):
            ConcreteGenerator(pattern_extractor=object())

    @pytest.mark.asyncio
    async def test_generate_method(self):
        """Test generate method implementation"""
        generator = ConcreteGenerator()

        sampled_content = [{"title": "test", "description": "test content"}]
        result = await generator.generate(sampled_content, target_count=5)

        assert len(result) == 5
        assert all(isinstance(item, MockSchema) for item in result)
        assert result[0].value == "item_0"
        assert result[4].value == "item_4"

    def test_validate_inputs_valid(self):
        """Test validate_inputs with valid inputs"""
        generator = ConcreteGenerator()

        # Should not raise exception
        generator.validate_inputs(sampled_content=[{"test": "data"}], target_count=10)

    def test_validate_inputs_invalid_count(self):
        """Test validate_inputs with invalid count"""
        generator = ConcreteGenerator()

        with pytest.raises(ValueError) as exc_info:
            generator.validate_inputs(
                sampled_content=[{"test": "data"}], target_count=0
            )

        assert str(exc_info.value) == ("target_count must be a positive integer, got 0")

        with pytest.raises(ValueError):
            generator.validate_inputs(
                sampled_content=[{"test": "data"}], target_count=-1
            )

    @pytest.mark.parametrize("target_count", [True, False, 1.0, "1", None])
    def test_validate_inputs_requires_positive_integer_count(self, target_count):
        generator = ConcreteGenerator()

        with pytest.raises(ValueError) as error:
            generator.validate_inputs(
                sampled_content=[{"test": "data"}], target_count=target_count
            )

        assert str(error.value) == (
            f"target_count must be a positive integer, got {target_count!r}"
        )

    def test_validate_inputs_empty_content(self):
        """Empty backend samples cannot produce grounded training data."""
        generator = ConcreteGenerator()

        with pytest.raises(
            ValueError,
            match="sampled_content must be a non-empty list of dict records",
        ):
            generator.validate_inputs(sampled_content=[], target_count=10)

    @pytest.mark.parametrize(
        "sampled_content",
        [None, {}, "content", ({},), ["content"], [{}, 1]],
    )
    def test_validate_inputs_requires_non_empty_list_of_records(self, sampled_content):
        generator = ConcreteGenerator()

        with pytest.raises(ValueError) as error:
            generator.validate_inputs(sampled_content=sampled_content, target_count=1)

        assert str(error.value) == (
            "sampled_content must be a non-empty list of dict records"
        )

    def test_get_generator_info(self):
        """Test get_generator_info method"""
        generator = ConcreteGenerator()

        info = generator.get_generator_info()

        assert info == {"name": "ConcreteGenerator", "has_agent_inferrer": False}

    def test_get_generator_info_with_utilities(self):
        """Test get_generator_info with utilities"""
        generator = ConcreteGenerator(agent_inferrer=object())

        info = generator.get_generator_info()

        assert info == {"name": "ConcreteGenerator", "has_agent_inferrer": True}

    def test_base_generator_cannot_be_instantiated(self):
        """Test that BaseGenerator cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseGenerator()

    @pytest.mark.asyncio
    async def test_generate_with_kwargs(self):
        """Test generate method accepts additional kwargs"""
        generator = ConcreteGenerator()

        result = await generator.generate(
            sampled_content=[{"test": "data"}],
            target_count=3,
            custom_param="value",
            another_param=123,
        )

        assert len(result) == 3


class IncompleteGenerator(BaseGenerator):
    """Generator that doesn't implement generate() - should fail"""

    pass


class TestBaseGeneratorAbstract:
    """Test BaseGenerator abstract method enforcement"""

    def test_incomplete_generator_cannot_be_instantiated(self):
        """Test that incomplete generator implementation fails"""
        with pytest.raises(TypeError) as exc_info:
            IncompleteGenerator()

        assert "abstract" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
