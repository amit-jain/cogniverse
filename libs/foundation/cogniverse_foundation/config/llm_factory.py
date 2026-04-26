"""
Centralized factory for creating DSPy LM instances.

Every dspy.LM() call in the codebase MUST go through create_dspy_lm().
This is the single chokepoint for LLM instantiation, making it trivial
to add instrumentation, logging, or caching in one place — and the
single place where the chart's ``llm.engine`` deployment switch is
applied to the model id.

Engine resolution: the factory reads ``system_config.llm_engine`` (set
from the chart's ``LLM_ENGINE`` env var at runtime startup) and prepends
the matching DSPy/litellm prefix. ``LLMEndpointConfig.model`` should
carry the BARE model id (``qwen3:4b`` or ``Qwen/Qwen2.5-7B-Instruct``);
configs that still ship pre-prefixed ids (``ollama/qwen3:4b``) are
handled idempotently by ``format_dspy_model`` — known prefixes are
stripped before re-prefixing.
"""

import logging

import dspy

from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.dspy import format_dspy_model

logger = logging.getLogger(__name__)


def create_dspy_lm(config: LLMEndpointConfig) -> dspy.LM:
    """
    Create a dspy.LM instance from an LLMEndpointConfig.

    Looks up the deployment engine from system_config and applies the
    matching litellm prefix (ollama_chat / hosted_vllm / openai) so
    callers don't need to know about engines at all.

    Args:
        config: LLM endpoint configuration with bare model id, api_base,
            api_key, etc.

    Returns:
        Configured dspy.LM instance.

    Raises:
        ValueError: If config.model is empty or None.
    """
    if not config.model:
        raise ValueError(
            "LLMEndpointConfig.model is required (e.g. 'qwen3:4b' — "
            "bare id; the factory prepends the engine prefix)"
        )

    # Lazy import to avoid a circular import at module load
    # (config.utils imports back into config.*).
    from cogniverse_foundation.config.utils import create_default_config_manager

    engine = create_default_config_manager().get_system_config().llm_engine
    model_id = format_dspy_model(config.model, engine)

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
        "Creating dspy.LM: model=%s engine=%s api_base=%s temperature=%s max_tokens=%s",
        model_id,
        engine,
        config.api_base or "(default)",
        config.temperature,
        config.max_tokens,
    )

    return dspy.LM(model_id, **kwargs)
