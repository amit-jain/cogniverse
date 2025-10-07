"""Global pytest configuration for test isolation"""
import pytest


@pytest.fixture(autouse=True, scope="function")
def cleanup_dspy_state():
    """Clean up DSPy state between tests to prevent isolation issues"""
    yield
    # Clean up any DSPy state after each test
    try:
        import dspy
        # Reset settings to prevent state pollution
        # Don't call configure() as it breaks async task isolation
        # Instead, directly clear the internal state
        if hasattr(dspy.settings, '_instance'):
            dspy.settings._instance = None
    except (ImportError, AttributeError, RuntimeError):
        # RuntimeError can occur if called from different async context
        pass


@pytest.fixture(autouse=True, scope="function")
def cleanup_vlm_state():
    """Clean up VLM interface state between tests"""
    yield
    # Clean up any cached VLM instances
    try:
        from src.common.vlm_interface import VLMInterface
        # Clear any class-level state if it exists
        if hasattr(VLMInterface, '_instance'):
            VLMInterface._instance = None
    except (ImportError, AttributeError):
        pass
