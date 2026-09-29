"""Inspect AI model provider that calls the LM through litellm.

``inspect_model(endpoint)`` returns an Inspect AI ``Model`` for an
``LLMEndpointConfig``; the provider is also addressable by name as
``cogniverse/<litellm model>`` (for example ``cogniverse/openai/gpt-4o``).
Requests carry the endpoint's api_base, key resolution, extra body, headers,
sampling and timeout exactly as ``create_dspy_lm`` sends them. Inspect AI owns
retries: litellm makes one attempt and ``should_retry`` classifies failures.
Text and image messages are supported; tool calls are not.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import litellm
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ContentImage,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    Model,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    RetryDecision,
    StopReason,
    get_model,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from cogniverse_foundation.config.llm_factory import dspy_lm_kwargs
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

PROVIDER_NAME = "cogniverse"

_STOP_REASONS: dict[str, StopReason] = {
    "stop": "stop",
    "length": "max_tokens",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "content_filter",
}

_TRANSIENT_ERRORS = (
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
)


def _content_part(content: Any) -> dict[str, Any] | None:
    if isinstance(content, ContentText):
        return {"type": "text", "text": content.text}
    if isinstance(content, ContentImage):
        image_url: dict[str, Any] = {"url": content.image}
        if content.detail != "auto":
            image_url["detail"] = content.detail
        return {"type": "image_url", "image_url": image_url}
    if isinstance(content, ContentReasoning):
        return None
    raise ValueError(
        f"The {PROVIDER_NAME} model provider does not support {content.type} content"
    )


def _message(message: ChatMessage) -> dict[str, Any]:
    if isinstance(message.content, str):
        return {"role": message.role, "content": message.content}
    parts = [
        part
        for part in (_content_part(content) for content in message.content)
        if part is not None
    ]
    return {"role": message.role, "content": parts}


class CogniverseModelAPI(ModelAPI):
    """Inspect AI ``ModelAPI`` backed by one ``LLMEndpointConfig``."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **endpoint_fields: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )
        self._endpoint = LLMEndpointConfig(
            model=model_name, api_base=base_url, api_key=api_key, **endpoint_fields
        )
        self._request = dspy_lm_kwargs(self._endpoint)
        self._request["num_retries"] = 0

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        if tools or any(
            isinstance(message, ChatMessageTool)
            or (isinstance(message, ChatMessageAssistant) and message.tool_calls)
            for message in input
        ):
            raise ValueError(
                f"The {PROVIDER_NAME} model provider does not support tool calls"
            )

        request = dict(self._request)
        if config.temperature is not None:
            request["temperature"] = config.temperature
        if config.max_tokens is not None:
            request["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            request["top_p"] = config.top_p
        if config.stop_seqs:
            request["stop"] = list(config.stop_seqs)
        if config.seed is not None:
            request["extra_body"] = {
                **request.get("extra_body", {}),
                "seed": config.seed,
            }
        if config.timeout is not None:
            request["timeout"] = config.timeout

        try:
            response = await litellm.acompletion(
                model=self._endpoint.model,
                messages=[_message(message) for message in input],
                **request,
            )
        except litellm.ContextWindowExceededError as exc:
            return ModelOutput.from_content(
                model=self.model_name,
                content="",
                stop_reason="model_length",
                error=str(exc),
            )

        choice = response.choices[0]
        usage = response.usage
        return ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(
                        content=choice.message.content or "",
                        model=self.model_name,
                        source="generate",
                    ),
                    stop_reason=_STOP_REASONS.get(choice.finish_reason, "unknown"),
                )
            ],
            usage=ModelUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            ),
        )

    def should_retry(self, ex: Exception) -> RetryDecision:
        if isinstance(ex, litellm.RateLimitError):
            return RetryDecision.rate_limit()
        if isinstance(ex, _TRANSIENT_ERRORS):
            return RetryDecision.transient()
        return RetryDecision.no()

    def is_auth_failure(self, ex: Exception) -> bool:
        return isinstance(ex, litellm.AuthenticationError)


@modelapi(name=PROVIDER_NAME)
def cogniverse() -> type[ModelAPI]:
    return CogniverseModelAPI


def inspect_model(
    endpoint: LLMEndpointConfig, config: GenerateConfig | None = None
) -> Model:
    """An Inspect AI ``Model`` that sends every request to ``endpoint``."""
    endpoint_fields = {
        field.name: getattr(endpoint, field.name)
        for field in dataclasses.fields(endpoint)
        if field.name not in ("model", "api_base", "api_key")
    }
    return get_model(
        f"{PROVIDER_NAME}/{endpoint.model}",
        base_url=endpoint.api_base,
        api_key=endpoint.api_key,
        config=config or GenerateConfig(),
        memoize=False,
        **endpoint_fields,
    )
