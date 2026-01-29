"""Mixin for agents that need RLM capabilities.

RLM (Recursive Language Models) enables near-infinite context processing.
This mixin provides query-configurable RLM inference for A/B testing.

Features:
    - Query-configurable RLM via RLMOptions
    - Optional EventQueue integration for real-time progress tracking
    - Cancellation support via CancellationToken
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from cogniverse_agents.inference.rlm_inference import RLMInference, RLMResult
from cogniverse_core.agents.rlm_options import RLMOptions

if TYPE_CHECKING:
    from cogniverse_core.events import EventQueue

logger = logging.getLogger(__name__)


class RLMAwareMixin:
    """
    Mixin providing RLM inference capabilities to agents.

    RLM is **query-configurable** via RLMOptions in the input schema.
    This enables A/B testing between RLM and standard inference.

    Usage in agent:
        class SearchAgent(RLMAwareMixin, A2AAgent[SearchInput, ...]):
            async def _process_impl(self, input: SearchInput):
                # Check if RLM should be used for this query
                context = self._build_context(results)

                if self.should_use_rlm_for_query(input.rlm, context):
                    result = self.process_with_rlm(
                        query=input.query,
                        context=context,
                        rlm_options=input.rlm
                    )
                    return self._build_output(result)
                else:
                    return await self._standard_processing(input)

    A/B Testing Example:
        # Group A: Standard inference (rlm=None)
        input_a = SearchInput(query="...", rlm=None)

        # Group B: RLM inference
        input_b = SearchInput(query="...", rlm=RLMOptions(enabled=True))

        # Compare telemetry metrics in Phoenix dashboard
    """

    _rlm_instance: Optional[RLMInference] = None

    def get_rlm(
        self,
        backend: str = "openai",
        model: str = "gpt-4o",
        max_iterations: int = 10,
        max_llm_calls: int = 30,
        timeout_seconds: int = 300,
        event_queue: Optional["EventQueue"] = None,
        task_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> RLMInference:
        """Get or create RLM inference instance with specified config.

        Args:
            backend: LLM backend (openai, anthropic, litellm)
            model: Model name
            max_iterations: Maximum REPL iteration loops
            max_llm_calls: Maximum LLM sub-calls
            timeout_seconds: Timeout for RLM processing
            event_queue: Optional EventQueue for real-time progress events
            task_id: Task identifier for events
            tenant_id: Tenant identifier for events

        Returns:
            RLMInference instance (with InstrumentedRLM if event_queue provided)
        """
        # Always create new instance when event_queue is provided
        # (event_queue/task_id may change per request)
        if event_queue:
            return RLMInference(
                backend=backend,
                model=model,
                max_iterations=max_iterations,
                max_llm_calls=max_llm_calls,
                timeout_seconds=timeout_seconds,
                event_queue=event_queue,
                task_id=task_id,
                tenant_id=tenant_id or getattr(self, "tenant_id", "default"),
            )

        # Create cached instance if config changed (no event_queue)
        if (
            self._rlm_instance is None
            or self._rlm_instance.backend != backend
            or self._rlm_instance.model != model
            or self._rlm_instance.max_iterations != max_iterations
            or self._rlm_instance.max_llm_calls != max_llm_calls
            or self._rlm_instance.timeout_seconds != timeout_seconds
        ):
            self._rlm_instance = RLMInference(
                backend=backend,
                model=model,
                max_iterations=max_iterations,
                max_llm_calls=max_llm_calls,
                timeout_seconds=timeout_seconds,
            )
        return self._rlm_instance

    def should_use_rlm_for_query(
        self,
        rlm_options: Optional[RLMOptions],
        context: str,
    ) -> bool:
        """
        Determine if RLM should be used based on query config and context.

        Args:
            rlm_options: RLMOptions from query input (None = disabled)
            context: The context string that would be processed

        Returns:
            True if RLM should be used for this query
        """
        if rlm_options is None:
            return False

        return rlm_options.should_use_rlm(len(context))

    def process_with_rlm(
        self,
        query: str,
        context: str,
        rlm_options: RLMOptions,
        system_prompt: Optional[str] = None,
        event_queue: Optional["EventQueue"] = None,
        task_id: Optional[str] = None,
    ) -> RLMResult:
        """
        Process query using RLM with the specified options.

        Args:
            query: User query
            context: Large context to process
            rlm_options: RLM configuration from query
            system_prompt: Optional system instructions
            event_queue: Optional EventQueue for real-time progress events
            task_id: Task identifier for events

        Returns:
            RLMResult with answer and telemetry data
        """
        rlm = self.get_rlm(
            backend=rlm_options.backend,
            model=rlm_options.model or "gpt-4o",
            max_iterations=rlm_options.max_depth,
            max_llm_calls=rlm_options.max_llm_calls,
            timeout_seconds=rlm_options.timeout_seconds,
            event_queue=event_queue,
            task_id=task_id,
            tenant_id=getattr(self, "tenant_id", "default"),
        )

        logger.info(
            f"Using RLM inference (max_iterations={rlm_options.max_depth}, "
            f"max_llm_calls={rlm_options.max_llm_calls}, "
            f"timeout={rlm_options.timeout_seconds}s, context={len(context)} chars, "
            f"events={'enabled' if event_queue else 'disabled'})"
        )

        return rlm.process(
            query=query,
            context=context,
            system_prompt=system_prompt,
        )

    def process_results_with_rlm(
        self,
        query: str,
        results: list[dict],
        rlm_options: RLMOptions,
        event_queue: Optional["EventQueue"] = None,
        task_id: Optional[str] = None,
    ) -> RLMResult:
        """
        Process search results using RLM.

        Args:
            query: User query
            results: List of search result dicts
            rlm_options: RLM configuration from query
            event_queue: Optional EventQueue for real-time progress events
            task_id: Task identifier for events

        Returns:
            RLMResult with synthesized answer
        """
        rlm = self.get_rlm(
            backend=rlm_options.backend,
            model=rlm_options.model or "gpt-4o",
            max_iterations=rlm_options.max_depth,
            max_llm_calls=rlm_options.max_llm_calls,
            timeout_seconds=rlm_options.timeout_seconds,
            event_queue=event_queue,
            task_id=task_id,
            tenant_id=getattr(self, "tenant_id", "default"),
        )

        logger.info(
            f"Using RLM for result aggregation ({len(results)} results, "
            f"max_iterations={rlm_options.max_depth}, timeout={rlm_options.timeout_seconds}s, "
            f"events={'enabled' if event_queue else 'disabled'})"
        )

        return rlm.process_search_results(
            query=query,
            results=results,
        )

    def get_rlm_telemetry(
        self,
        rlm_result: Optional[RLMResult],
        context_size: int,
    ) -> Dict[str, Any]:
        """
        Build telemetry dict for A/B testing comparison.

        Args:
            rlm_result: Result from RLM (None if RLM not used)
            context_size: Size of context in characters

        Returns:
            Dict with telemetry metrics for tracking
        """
        if rlm_result:
            return {
                **rlm_result.to_telemetry_dict(),
                "context_size_chars": context_size,
            }
        else:
            return {
                "rlm_enabled": False,
                "context_size_chars": context_size,
            }
