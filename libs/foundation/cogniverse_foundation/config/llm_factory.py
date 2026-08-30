"""
Centralized factory for creating DSPy LM instances.

Every dspy.LM() call in the codebase MUST go through create_dspy_lm().
This is the single chokepoint for LLM instantiation, making it trivial
to add instrumentation, logging, or caching in one place. Clients that
cannot be a dspy.LM (Mem0's OpenAI provider, litellm.completion, a raw
chat-completions POST) resolve their key through resolve_inference_api_key()
so every OpenAI-compatible call authenticates the same way.

The factory does no string manipulation on ``LLMEndpointConfig.model``.
The contract is that ``config.model`` already carries whatever
litellm-recognised string the caller wants — Helm renders it into
``config.json`` (see ``cogniverse.primaryLLMModel`` and
``cogniverse.teacherLLMModel`` in templates/_helpers.tpl), or callers
build the ``LLMEndpointConfig`` directly. The factory's job is to wire
api_base / api_key / extra_body / sampling onto the dspy.LM and emit
one well-formed log line per construction.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from cogniverse_foundation.config.bootstrap import (
    INFERENCE_API_KEY_ENV,
    inference_api_key_from_environment,
)
from cogniverse_foundation.config.inference_auth import (
    endpoint_root,
    is_modal_inference_url,
)
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

if TYPE_CHECKING:
    import dspy

logger = logging.getLogger(__name__)

# dspy (and its litellm dependency) is imported lazily inside create_dspy_lm:
# importing it at module level cost ~2.2s on every process that touches
# cogniverse_foundation.config (every CLI, the worker pod), even on paths that
# never build a dspy.LM. The annotation below is a string via the __future__
# import, so it needs no import at definition time.


# What the chart renders for an endpoint whose real key lives in a Secret.
_PLACEHOLDER_API_KEY = "placeholder-no-auth-needed"


def _optional_environment_bearer() -> str | None:
    """The synced inference bearer, or None when the environment has none.

    Narrow by construction rather than by catching: the reader raises on an
    absent or untrimmed value, and only the absent case is a legitimate "no
    bearer configured". An untrimmed one is a misconfiguration and must still
    raise from the caller.
    """
    raw = os.environ.get(INFERENCE_API_KEY_ENV)
    if raw is None or raw == "":
        return None
    return inference_api_key_from_environment()


def resolve_inference_api_key(api_base: str | None, api_key: str | None) -> str | None:
    """The key an OpenAI-compatible client must send to ``api_base``.

    An explicitly configured key wins. With no key (or the chart's
    placeholder, which it emits because config.json renders into a ConfigMap
    that cannot hold a secret) the environment bearer is sent whenever one
    exists rather than only when ``api_base`` looks external: agent traffic
    addresses the in-cluster semantic router, which forwards to Modal, so the
    caller cannot know the ultimate upstream, and a self-hosted vLLM ignores
    an Authorization header it does not check. A Modal endpoint with no
    bearer raises here, naming the variable, instead of server-side on the
    first call. Without an ``api_base`` the configured key passes through
    untouched so the provider SDK's own resolution applies.
    """
    if api_base is None:
        return api_key
    if api_key not in (None, _PLACEHOLDER_API_KEY):
        return api_key
    bearer = _optional_environment_bearer()
    if bearer is not None:
        return bearer
    if is_modal_inference_url(endpoint_root(api_base)):
        return inference_api_key_from_environment()
    if api_key is not None:
        return api_key
    # The OpenAI client refuses to construct without a key even though
    # self-hosted OAI-compat servers (vLLM, Ollama) ignore it. A real
    # endpoint that enforces auth rejects the placeholder server-side
    # with a clear 401 instead of a client-side construction error.
    return "not-required"


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

    import dspy

    kwargs: dict = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout": config.request_timeout,
        "num_retries": config.num_retries,
    }

    if config.api_base is not None:
        kwargs["api_base"] = config.api_base

    api_key = resolve_inference_api_key(config.api_base, config.api_key)
    if api_key is not None:
        kwargs["api_key"] = api_key

    # Merge config.seed into extra_body when set. vLLM's OpenAI-compat
    # layer reads ``seed`` from the request body and uses it for the
    # sampling RNG; combined with ``temperature=0`` this gives
    # byte-stable output across runs (modulo vLLM batching state).
    extra_body = dict(config.extra_body or {})
    if config.seed is not None:
        extra_body["seed"] = int(config.seed)
    if extra_body:
        kwargs["extra_body"] = extra_body

    # Static per-endpoint HTTP headers (e.g. routing metadata for a gateway
    # in front of the backend). litellm sends these on every request. An
    # empty/None dict is omitted so no header block hits the wire.
    if config.extra_headers:
        kwargs["extra_headers"] = dict(config.extra_headers)

    logger.info(
        "Creating dspy.LM: model=%s api_base=%s temperature=%s max_tokens=%s seed=%s",
        config.model,
        config.api_base or "(default)",
        config.temperature,
        config.max_tokens,
        config.seed,
    )

    return dspy.LM(config.model, **kwargs)
