"""RLM (Recursive Language Model) inference wrapper for long-context tasks.

RLM enables LLMs to handle near-infinite context by:
1. Storing context as a Python variable (not in LLM prompt)
2. LLM generates code to inspect, filter, and partition context
3. Spawns sub-LLMs recursively to process partitions
4. Aggregates results via SUBMIT({fields})

This module uses DSPy's built-in RLM module (available in dspy-ai>=3.1.0).

References:
    - Paper: https://arxiv.org/abs/2512.24601
    - DSPy: https://github.com/stanfordnlp/dspy
"""

import concurrent.futures
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import dspy

logger = logging.getLogger(__name__)


class RLMTimeoutError(TimeoutError):
    """Raised when RLM processing exceeds the configured timeout."""

    pass


@dataclass
class RLMResult:
    """Result from RLM inference with telemetry data.

    Attributes:
        answer: The final answer from RLM processing
        depth_reached: Actual recursion depth used
        total_calls: Number of LLM sub-calls made
        tokens_used: Total tokens across all calls
        latency_ms: End-to-end processing latency
        metadata: Additional metadata from RLM processing
    """

    answer: str
    depth_reached: int
    total_calls: int
    tokens_used: int
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_telemetry_dict(self) -> Dict[str, Any]:
        """Export metrics for telemetry/A/B testing.

        Returns:
            Dict with RLM telemetry metrics
        """
        return {
            "rlm_enabled": True,
            "rlm_depth_reached": self.depth_reached,
            "rlm_total_calls": self.total_calls,
            "rlm_tokens_used": self.tokens_used,
            "rlm_latency_ms": self.latency_ms,
        }


class RLMInference:
    """
    RLM inference using DSPy's built-in RLM module.

    Implements the Recursive Language Model pattern for handling
    large contexts that exceed model limits:
    - Large video frame analysis (many frames)
    - Multi-document aggregation
    - Long transcript processing

    Example:
        rlm = RLMInference(backend="openai", model="gpt-4o", max_iterations=10)
        result = rlm.process(
            query="Summarize the main findings",
            context=large_context_string,
        )
        print(f"Answer: {result.answer}")
        print(f"Depth: {result.depth_reached}, Calls: {result.total_calls}")
    """

    def __init__(
        self,
        backend: str = "openai",
        model: str = "gpt-4o",
        max_iterations: int = 10,
        max_llm_calls: int = 30,
        sandbox: str = "local",
        api_key: Optional[str] = None,
        timeout_seconds: Optional[int] = 300,
    ):
        """Initialize RLM inference wrapper.

        Args:
            backend: LLM backend (openai, anthropic, ollama)
            model: Model name to use
            max_iterations: Maximum REPL interaction loops (default: 10)
            max_llm_calls: Limit on sub-LLM queries (default: 30)
            sandbox: Execution sandbox (local only for now)
            api_key: Optional API key for the backend
            timeout_seconds: Maximum time for RLM processing (default: 300s/5min)
        """
        self.backend = backend
        self.model = model
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls
        self.sandbox = sandbox
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._rlm = None  # Lazy initialization

    @property
    def max_depth(self) -> int:
        """Alias for max_iterations for backward compatibility."""
        return self.max_iterations

    def _create_lm(self):
        """Create DSPy LM based on backend."""
        if self.backend == "ollama":
            return dspy.LM(
                f"ollama_chat/{self.model}", api_base="http://localhost:11434"
            )
        elif self.backend == "litellm":
            # litellm format: ollama/model or openai/model
            return dspy.LM(self.model, api_key=self._api_key)
        elif self.backend == "openai":
            return dspy.LM(f"openai/{self.model}", api_key=self._api_key)
        elif self.backend == "anthropic":
            return dspy.LM(f"anthropic/{self.model}", api_key=self._api_key)
        else:
            # Default to litellm format
            return dspy.LM(self.model, api_key=self._api_key)

    def _get_rlm(self):
        """Get or create DSPy RLM instance."""
        if self._rlm is None:
            self._lm = self._create_lm()
            self._rlm = dspy.RLM(
                "context, query -> answer",
                max_iterations=self.max_iterations,
                max_llm_calls=self.max_llm_calls,
                verbose=False,
            )
        return self._rlm

    def _execute_rlm(self, rlm, full_query: str, context: str) -> Any:
        """Execute RLM in a thread-safe manner.

        This is separated for timeout handling via ThreadPoolExecutor.
        """
        with dspy.context(lm=self._lm):
            return rlm(context=context, query=full_query)

    def process(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> RLMResult:
        """
        Process query with potentially huge context using DSPy RLM.

        Args:
            query: The user query/task
            context: Large context (can be 100K+ chars)
            system_prompt: Optional system instructions

        Returns:
            RLMResult with answer and telemetry metadata

        Raises:
            RLMTimeoutError: If processing exceeds timeout_seconds
        """
        start_time = time.time()

        rlm = self._get_rlm()

        # Include system prompt in query if provided
        full_query = f"{system_prompt}\n\n{query}" if system_prompt else query

        try:
            # Execute RLM with timeout protection
            if self.timeout_seconds:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self._execute_rlm, rlm, full_query, context
                    )
                    try:
                        result = future.result(timeout=self.timeout_seconds)
                    except concurrent.futures.TimeoutError:
                        raise RLMTimeoutError(
                            f"RLM processing exceeded timeout of {self.timeout_seconds}s"
                        )
            else:
                # No timeout - execute directly
                result = self._execute_rlm(rlm, full_query, context)

            answer = result.answer if hasattr(result, "answer") else str(result)

            # Extract trajectory info for telemetry
            trajectory = getattr(result, "trajectory", [])
            depth_reached = len(trajectory) if trajectory else 1
            total_calls = len(trajectory) if trajectory else 1

        except RLMTimeoutError:
            raise
        except Exception as e:
            logger.warning(f"RLM execution failed: {e}")
            raise

        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            f"RLM completed: depth={depth_reached}, calls={total_calls}, "
            f"latency={latency_ms:.0f}ms"
        )

        return RLMResult(
            answer=answer,
            depth_reached=depth_reached,
            total_calls=total_calls,
            tokens_used=0,  # DSPy doesn't expose token counts easily
            latency_ms=latency_ms,
            metadata={
                "context_size_chars": len(context),
                "query": query[:100],
                "backend": self.backend,
                "model": self.model,
                "timeout_seconds": self.timeout_seconds,
            },
        )

    def process_documents(
        self,
        query: str,
        documents: list[dict],
        doc_key: str = "content",
    ) -> RLMResult:
        """Process query over multiple documents.

        Args:
            query: User query
            documents: List of document dicts
            doc_key: Key to extract content from each document

        Returns:
            RLMResult with aggregated answer
        """
        context = "\n\n".join(
            [
                f"=== Document {i} ===\n{doc.get(doc_key, '')}"
                for i, doc in enumerate(documents)
            ]
        )

        return self.process(
            query=query,
            context=context,
            system_prompt="Search through documents to answer. Focus on relevant sections.",
        )

    def process_search_results(
        self,
        query: str,
        results: list[dict],
    ) -> RLMResult:
        """Process query over search results.

        Args:
            query: User query
            results: List of search result dicts

        Returns:
            RLMResult with synthesized answer
        """
        context = "\n\n".join(
            [
                f"=== Result {i + 1} (score: {r.get('score', 0):.3f}) ===\n"
                f"ID: {r.get('id', 'unknown')}\n"
                f"Content: {r.get('content', r.get('summary', str(r)))}"
                for i, r in enumerate(results)
            ]
        )

        return self.process(
            query=f"Synthesize answer for: {query}",
            context=context,
            system_prompt="Analyze search results and provide comprehensive answer.",
        )
