"""
Unit tests for DSPy Module and Optimizer Registries.
"""

import dspy
import pytest

from cogniverse_core.common.agent_config import DSPyModuleType, OptimizerType
from cogniverse_core.common.dspy_module_registry import DSPyModuleRegistry, DSPyOptimizerRegistry


class TestSignature(dspy.Signature):
    """Test signature for module creation"""

    input_text = dspy.InputField()
    output_text = dspy.OutputField()


class TestDSPyModuleRegistry:
    """Test DSPyModuleRegistry"""

    def test_all_module_types_registered(self):
        """Test all module types are registered"""
        modules = DSPyModuleRegistry.list_modules()

        expected_modules = {
            "predict": "Predict",
            "chain_of_thought": "ChainOfThought",
            "react": "ReAct",
            "multi_chain_comparison": "MultiChainComparison",
            "program_of_thought": "ProgramOfThought",
        }

        assert modules == expected_modules

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

    def test_all_optimizer_types_registered(self):
        """Test all optimizer types are registered"""
        optimizers = DSPyOptimizerRegistry.list_optimizers()

        expected_optimizers = {
            "bootstrap_few_shot": "BootstrapFewShot",
            "labeled_few_shot": "LabeledFewShot",
            "bootstrap_few_shot_with_random_search": "BootstrapFewShotWithRandomSearch",
            "copro": "COPRO",
            "mipro_v2": "MIPROv2",
        }

        assert optimizers == expected_optimizers

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

    def test_create_optimizer_bootstrap_few_shot(self):
        """Test creating BootstrapFewShot optimizer"""

        # Create optimizer without mock module (not needed for this test)
        optimizer = DSPyOptimizerRegistry.create_optimizer(
            OptimizerType.BOOTSTRAP_FEW_SHOT,
            metric=lambda x, y: 1.0,
            max_bootstrapped_demos=2,
            max_labeled_demos=4,
        )

        assert optimizer is not None
        assert isinstance(optimizer, dspy.BootstrapFewShot)

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
