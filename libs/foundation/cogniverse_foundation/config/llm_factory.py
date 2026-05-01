"""
Centralized factory for creating DSPy LM instances.

Every dspy.LM() call in the codebase MUST go through create_dspy_lm().
This is the single chokepoint for LLM instantiation, making it trivial
to add instrumentation, logging, or caching in one place.

The factory does no string manipulation on ``LLMEndpointConfig.model``.
The contract is that ``config.model`` already carries whatever
litellm-recognised string the caller wants — Helm renders it into
``config.json`` (see ``cogniverse.primaryLLMModel`` and
``cogniverse.teacherLLMModel`` in templates/_helpers.tpl), or callers
build the ``LLMEndpointConfig`` directly. The factory's job is to wire
api_base / api_key / extra_body / sampling onto the dspy.LM and emit
one well-formed log line per construction.
"""

import logging

import dspy

from cogniverse_foundation.config.unified_config import LLMEndpointConfig

logger = logging.getLogger(__name__)


def create_dspy_lm(config: LLMEndpointConfig) -> dspy.LM:
    """
    Create a dspy.LM instance from an LLMEndpointConfig.

    Args:
        config: LLM endpoint configuration. ``config.model`` is passed
            through to dspy.LM verbatim.

    Returns:
        Configured dspy.LM instance.

    Raises:
        ValueError: If config.model is empty or None.
    """
    if not config.model:
        raise ValueError("LLMEndpointConfig.model is required and must be non-empty")

    kwargs: dict = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }

    if config.api_base is not None:
        kwargs["api_base"] = config.api_base

    if config.api_key is not None:
        kwargs["api_key"] = config.api_key

    if config.extra_body is not None:
        kwargs["extra_body"] = config.extra_body

    logger.info(
        "Creating dspy.LM: model=%s api_base=%s temperature=%s max_tokens=%s",
        config.model,
        config.api_base or "(default)",
        config.temperature,
        config.max_tokens,
    )

    return dspy.LM(config.model, **kwargs)
