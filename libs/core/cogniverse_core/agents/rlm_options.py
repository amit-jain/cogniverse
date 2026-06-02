"""RLM (Recursive Language Model) configuration for query-level control.

RLM (Recursive Language Models) is an inference paradigm from MIT that enables
LLMs to handle near-infinite context by programmatically examining, decomposing,
and recursively calling themselves.

This module provides RLMOptions for query-level RLM configuration, enabling
A/B testing between RLM and standard inference.

References:
    - DSPy: https://github.com/stanfordnlp/dspy (RLM module in 3.1.0+)
"""

from typing import Optional

from pydantic import BaseModel, Field


class RLMOptions(BaseModel):
    """
    Optional RLM configuration per query.

    RLM enables LLMs to handle near-infinite context via recursive decomposition.
    Enable explicitly for A/B testing or use auto_detect for threshold-based activation.

    Attributes:
        enabled: Explicitly enable RLM inference for this query
        auto_detect: Auto-enable RLM when context exceeds threshold
        context_threshold: Context size (chars) threshold for auto_detect mode
        max_iterations: Maximum REPL iterations for RLM (1-10)
        backend: LLM backend for RLM (openai, anthropic, litellm)
        model: Model override for RLM (defaults to agent's model)

    Usage:
        # Explicit enable (A/B test group B)
        rlm = RLMOptions(enabled=True, max_iterations=3)

        # Auto-detect based on context size
        rlm = RLMOptions(auto_detect=True, context_threshold=50_000)

        # Standard search (no RLM) - Group A
        input = SearchInput(query="...", rlm=None)

        # RLM-enabled search - Group B
        input = SearchInput(query="...", rlm=RLMOptions(enabled=True))

    Example A/B Testing:
        # Compare in telemetry/Phoenix dashboard:
        # - rlm_enabled=true vs rlm_enabled=false
        # - Latency distribution
        # - Token usage
        # - Response quality
    """

    enabled: bool = Field(
        default=False, description="Explicitly enable RLM inference for this query"
    )
    auto_detect: bool = Field(
        default=False, description="Auto-enable RLM when context exceeds threshold"
    )
    context_threshold: int = Field(
        default=50_000,
        description="Context size (chars) threshold for auto_detect mode",
    )
    max_iterations: int = Field(
        default=3, ge=1, le=10, description="Maximum REPL iterations for RLM (1-10)"
    )
    max_llm_calls: int = Field(
        default=30, ge=1, le=100, description="Maximum LLM sub-calls for RLM (1-100)"
    )
    timeout_seconds: int = Field(
        default=300,
        ge=10,
        le=1800,
        description="Timeout for RLM processing (10-1800 seconds)",
    )
    backend: str = Field(
        default="openai", description="LLM backend for RLM (openai, anthropic, litellm)"
    )
    model: Optional[str] = Field(
        default=None, description="Model override for RLM (defaults to agent's model)"
    )
    api_base: Optional[str] = Field(
        default=None,
        description=(
            "Override the LLM endpoint URL (e.g. ``http://localhost:29010/v1`` for "
            "a local vLLM). When ``None``, the mixin inherits the agent's primary "
            "LLM ``api_base``; passing a value here lets per-query RLM hit a "
            "separate endpoint without rewiring the agent."
        ),
    )
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "Override the LLM API key. ``None`` means inherit the agent's primary "
            "key (or pass nothing for local OAI-compat servers that ignore keys)."
        ),
    )
    include_trajectory: bool = Field(
        default=False,
        description=(
            "When true, attach a structured (bounded) trajectory list to "
            "RLMResult.trajectory so callers can audit the recursion. The "
            "trajectory is also written to Phoenix span attributes regardless "
            "of this flag for server-side debugging."
        ),
    )
    trajectory_max_entries: int = Field(
        default=32,
        ge=1,
        le=200,
        description=(
            "Maximum number of REPL iteration entries to keep in the captured "
            "trajectory. Bounds payload size when include_trajectory=True and "
            "the size of the Phoenix span attribute always written."
        ),
    )

    def should_use_rlm(self, context_size: int) -> bool:
        """
        Determine if RLM should be used based on config and context size.

        Args:
            context_size: Size of context in characters

        Returns:
            True if RLM should be used for this query
        """
        if self.enabled:
            return True
        if self.auto_detect and context_size > self.context_threshold:
            return True
        return False
