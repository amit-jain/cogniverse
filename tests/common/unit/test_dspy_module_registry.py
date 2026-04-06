"""
Unit tests for DSPy Module and Optimizer Registries.
"""

import dspy
import pytest

from cogniverse_core.common.dspy_module_registry import (
    DSPyModuleRegistry,
    DSPyOptimizerRegistry,
)
from cogniverse_foundation.config.agent_config import DSPyModuleType, OptimizerType


class TestSignature(dspy.Signature):
    """Test signature for module creation"""

    input_text = dspy.InputField()
    output_text = dspy.OutputField()


class TestDSPyModuleRegistry:
    """Test DSPyModuleRegistry"""

    def test_get_module_class_predict(self):
        """Test getting Predict module class"""
        module_class = DSPyModuleRegistry.get_module_class(DSPyModuleType.PREDICT)
        assert module_class == dspy.Predict

    def test_get_module_class_chain_of_thought(self):
        """Test getting ChainOfThought module class"""
        module_class = DSPyModuleRegistry.get_module_class(
            DSPyModuleType.CHAIN_OF_THOUGHT
        )
        assert module_class == dspy.ChainOfThought

    def test_get_module_class_react(self):
        """Test getting ReAct module class"""
        module_class = DSPyModuleRegistry.get_module_class(DSPyModuleType.REACT)
        assert module_class == dspy.ReAct

    def test_create_module_predict(self):
        """Test creating Predict module"""
        module = DSPyModuleRegistry.create_module(DSPyModuleType.PREDICT, TestSignature)

        assert isinstance(module, dspy.Predict)
        assert module.signature == TestSignature

    def test_create_module_chain_of_thought(self):
        """Test creating ChainOfThought module"""
        module = DSPyModuleRegistry.create_module(
            DSPyModuleType.CHAIN_OF_THOUGHT, TestSignature
        )

        assert isinstance(module, dspy.ChainOfThought)

    def test_create_module_with_custom_params(self):
        """Test creating module with custom parameters"""
        module = DSPyModuleRegistry.create_module(
            DSPyModuleType.PREDICT, TestSignature, max_retries=5
        )

        assert isinstance(module, dspy.Predict)

    def test_register_custom_module_type(self):
        """Test registering custom module type"""

        class CustomModule(dspy.Module):
            def __init__(self, signature):
                super().__init__()
                self.signature = signature

        # Create custom enum value (would need to extend DSPyModuleType)
        # For now, test the register method exists and works with existing enum

        # Register predict again with custom class
        DSPyModuleRegistry.register_module(DSPyModuleType.PREDICT, CustomModule)

        # Verify it was updated
        module_class = DSPyModuleRegistry.get_module_class(DSPyModuleType.PREDICT)
        assert module_class == CustomModule

        # Restore original
        DSPyModuleRegistry.register_module(DSPyModuleType.PREDICT, dspy.Predict)


class TestDSPyOptimizerRegistry:
    """Test DSPyOptimizerRegistry"""

    def test_get_optimizer_class_bootstrap_few_shot(self):
        """Test getting BootstrapFewShot optimizer class"""
        optimizer_class = DSPyOptimizerRegistry.get_optimizer_class(
            OptimizerType.BOOTSTRAP_FEW_SHOT
        )
        assert optimizer_class == dspy.BootstrapFewShot

    def test_get_optimizer_class_copro(self):
        """Test getting COPRO optimizer class"""
        optimizer_class = DSPyOptimizerRegistry.get_optimizer_class(OptimizerType.COPRO)
        assert optimizer_class == dspy.COPRO

    def test_get_optimizer_class_mipro_v2(self):
        """Test getting MIPROv2 optimizer class"""
        optimizer_class = DSPyOptimizerRegistry.get_optimizer_class(
            OptimizerType.MIPRO_V2
        )
        assert optimizer_class == dspy.MIPROv2

    def test_register_custom_optimizer_type(self):
        """Test registering custom optimizer type"""

        class CustomOptimizer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        # Register copro again with custom class
        DSPyOptimizerRegistry.register_optimizer(OptimizerType.COPRO, CustomOptimizer)

        # Verify it was updated
        optimizer_class = DSPyOptimizerRegistry.get_optimizer_class(OptimizerType.COPRO)
        assert optimizer_class == CustomOptimizer

        # Restore original
        DSPyOptimizerRegistry.register_optimizer(OptimizerType.COPRO, dspy.COPRO)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
