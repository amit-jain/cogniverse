"""
Centralized factory for creating DSPy LM instances.

Every dspy.LM() call in the codebase MUST go through create_dspy_lm().
This is the single chokepoint for LLM instantiation, making it trivial
to add instrumentation, logging, or caching in one place.
"""

import logging

import dspy

from cogniverse_foundation.config.unified_config import LLMEndpointConfig

logger = logging.getLogger(__name__)


def create_dspy_lm(config: LLMEndpointConfig) -> dspy.LM:
    """
    Create a dspy.LM instance from an LLMEndpointConfig.

    This is the ONLY place dspy.LM() should be called in the entire codebase.

    Args:
        config: LLM endpoint configuration with model, api_base, api_key, etc.

    Returns:
        Configured dspy.LM instance

    Raises:
        ValueError: If config.model is empty or None
    """
    if not config.model:
        raise ValueError(
            "LLMEndpointConfig.model is required (e.g., 'ollama/smollm3:3b')"
        )

    kwargs: dict = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    if config.api_base is not None:
        kwargs["api_base"] = config.api_base

    if config.api_key is not None:
        kwargs["api_key"] = config.api_key

    logger.info(
        "Creating dspy.LM: model=%s, api_base=%s, temperature=%s, max_tokens=%s",
        config.model,
        config.api_base or "(default)",
        config.temperature,
        config.max_tokens,
    )

    return dspy.LM(config.model, **kwargs)
