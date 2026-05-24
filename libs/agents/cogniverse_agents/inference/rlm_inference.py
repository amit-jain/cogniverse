"""RLM (Recursive Language Model) inference wrapper for long-context tasks.

RLM enables LLMs to handle near-infinite context by:
1. Storing context as a Python variable (not in LLM prompt)
2. LLM generates code to inspect, filter, and partition context
3. Spawns sub-LLMs recursively to process partitions
4. Aggregates results via SUBMIT({fields})

This module uses DSPy's built-in RLM module (available in dspy-ai>=3.1.0).

Features:
    - Optional EventQueue integration for real-time progress tracking
    - Cancellation support via CancellationToken
    - Timeout handling with RLMTimeoutError

References:
    - Paper: https://arxiv.org/abs/2512.24601
    - DSPy: https://github.com/stanfordnlp/dspy
"""

import concurrent.futures
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import dspy

if TYPE_CHECKING:
    from cogniverse_core.events import EventQueue

from cogniverse_agents.inference.deno_check import assert_deno_available
from cogniverse_agents.inference.instrumented_rlm import InstrumentedRLM
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

logger = logging.getLogger(__name__)


class RLMTimeoutError(TimeoutError):
    """Raised when RLM processing exceeds the configured timeout."""

    pass


_TRAJECTORY_FIELD_TRUNCATE = 500


def _sum_tracker_tokens(tracker: Any) -> int:
    """Sum total tokens across all LMs and entries tracked by a DSPy UsageTracker.

    DSPy's UsageTracker.usage_data is `{lm_name: [{prompt_tokens, completion_tokens,
    total_tokens, ...}, ...]}`. Some backends populate ``total_tokens``; others
    only ``prompt_tokens`` + ``completion_tokens``. We prefer the explicit total
    when present and fall back to the sum of components.
    """
    if tracker is None or not hasattr(tracker, "usage_data"):
        return 0

    total = 0
    for entries in tracker.usage_data.values():
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            explicit = entry.get("total_tokens")
            if isinstance(explicit, int) and explicit > 0:
                total += explicit
                continue
            prompt = entry.get("prompt_tokens") or 0
            completion = entry.get("completion_tokens") or 0
            if isinstance(prompt, int) and isinstance(completion, int):
                total += prompt + completion
    return total


def _sum_history_tokens(lm: Any, start_index: int = 0) -> int:
    """Sum total tokens across ``lm.history`` entries from ``start_index``.

    DSPy records every LM call in ``lm.history`` — both fresh calls and
    cache hits, each with the original ``usage`` dict. UsageTracker only
    sees fresh calls, so when DSPy serves a request from cache the
    tracker comes back empty even though the original call did consume
    tokens. Summing history.usage gives a stable count regardless of
    cache state. ``start_index`` lets callers scope the sum to entries
    appended during a specific call (record history length before the
    call, pass it here after).
    """
    history = getattr(lm, "history", None)
    if not history:
        return 0
    total = 0
    for entry in history[start_index:]:
        usage = entry.get("usage") if isinstance(entry, dict) else None
        if not isinstance(usage, dict):
            continue
        explicit = usage.get("total_tokens")
        if isinstance(explicit, int) and explicit > 0:
            total += explicit
            continue
        prompt = usage.get("prompt_tokens") or 0
        completion = usage.get("completion_tokens") or 0
        if isinstance(prompt, int) and isinstance(completion, int):
            total += prompt + completion
    return total


def _serialize_trajectory(trajectory: Any, max_entries: int) -> List[Dict[str, Any]]:
    """Convert a dspy RLM trajectory into a bounded JSON-friendly list.

    DSPy's ``RLM.forward`` returns ``trajectory`` as ``[entry.model_dump()
    for entry in history]`` — i.e. a list of dicts. ``REPLEntry``
    contributes ``reasoning`` / ``code`` / ``output`` keys; older
    versions also surfaced ``observation`` / ``result``. Read by dict
    lookup with an ``isinstance(..., dict)`` guard so we also tolerate
    raw REPLEntry objects (for callers that pass the unmaterialized
    history through).
    """
    if not trajectory:
        return []

    entries: List[Dict[str, Any]] = []
    for idx, raw in enumerate(trajectory[:max_entries]):
        entry: Dict[str, Any] = {"iteration": idx + 1}
        for field_name in ("reasoning", "code", "output", "observation", "result"):
            if isinstance(raw, dict):
                value = raw.get(field_name)
            else:
                value = getattr(raw, field_name, None)
            if value is None:
                continue
            text = str(value)
            if len(text) > _TRAJECTORY_FIELD_TRUNCATE:
                text = text[:_TRAJECTORY_FIELD_TRUNCATE] + "…"
            entry[field_name] = text
        entries.append(entry)

    return entries


@dataclass
class RLMResult:
    """Result from RLM inference with telemetry data.

    Attributes:
        answer: The final answer from RLM processing
        depth_reached: Actual recursion depth used
        total_calls: Number of LLM sub-calls made
        tokens_used: Total tokens across all calls
        latency_ms: End-to-end processing latency
        was_fallback: True when answer came from _extract_fallback because
            max_iterations was exhausted without a SUBMIT() — answer quality
            may be lower; downstream agents may flag the result or trigger
            a re-plan.
        trajectory: Bounded list of REPL iteration snapshots (when callers
            opted in via RLMOptions.include_trajectory). Each entry is a dict
            with 'iteration', 'reasoning', 'code' (each truncated). Always
            empty unless include_trajectory was set; callers should NOT rely
            on a non-empty trajectory for clean completions where the parent
            class did not retain history.
        metadata: Additional metadata from RLM processing. Includes a
            'trajectory_summary' (always populated, server-side debug aid)
            and 'trajectory_length' regardless of include_trajectory.
    """

    answer: str
    depth_reached: int
    total_calls: int
    tokens_used: int
    latency_ms: float
    was_fallback: bool = False
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
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
            "rlm_was_fallback": self.was_fallback,
            "rlm_trajectory_length": len(self.trajectory),
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
        config = LLMEndpointConfig(model="openai/gpt-4o")
        rlm = RLMInference(llm_config=config, max_iterations=10)
        result = rlm.process(
            query="Summarize the main findings",
            context=large_context_string,
        )
        print(f"Answer: {result.answer}")
        print(f"Depth: {result.depth_reached}, Calls: {result.total_calls}")
    """

    def __init__(
        self,
        llm_config: LLMEndpointConfig,
        max_iterations: int = 10,
        max_llm_calls: int = 30,
        timeout_seconds: Optional[int] = 300,
        event_queue: Optional["EventQueue"] = None,
        task_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ):
        """Initialize RLM inference wrapper.

        Args:
            llm_config: LLM endpoint configuration from centralized llm_config.
            max_iterations: Maximum REPL interaction loops (default: 10)
            max_llm_calls: Limit on sub-LLM queries (default: 30)
            timeout_seconds: Maximum time for RLM processing (default: 300s/5min)
            event_queue: Optional EventQueue for real-time progress events
            task_id: Task identifier for events (required if event_queue provided)
            tenant_id: Tenant identifier — required when event_queue is provided
        """
        # Fail fast at construction when Deno is missing. DSPy's RLM module
        # spawns a Deno subprocess on first call; if Deno is absent the failure
        # surfaces deep inside dspy and obscures the actionable fix. We probe
        # here so misconfigured environments fail at boot, with a clear error
        # naming the install location.
        assert_deno_available()

        self.llm_config = llm_config
        self.model = llm_config.model
        self.max_iterations = max_iterations
        self.max_llm_calls = max_llm_calls
        self.timeout_seconds = timeout_seconds
        self._event_queue = event_queue
        self._task_id = task_id
        if event_queue is not None and not tenant_id:
            raise ValueError(
                "tenant_id is required when event_queue is provided — "
                "RLM events must be tenant-scoped"
            )
        self._tenant_id = tenant_id
        self._rlm = None  # Lazy initialization

    def _create_lm(self):
        """Create DSPy LM via centralized factory."""
        return create_dspy_lm(self.llm_config)

    def _get_rlm(self):
        """Get or create DSPy RLM instance.

        Uses InstrumentedRLM if event_queue is provided for real-time
        progress tracking. Otherwise uses standard dspy.RLM.
        """
        if self._rlm is None:
            self._lm = self._create_lm()

            # Use InstrumentedRLM if event_queue provided for progress tracking
            if self._event_queue:
                self._rlm = InstrumentedRLM(
                    "context, query -> answer",
                    max_iterations=self.max_iterations,
                    max_llm_calls=self.max_llm_calls,
                    verbose=False,
                    event_queue=self._event_queue,
                    task_id=self._task_id,
                    tenant_id=self._tenant_id,
                )
            else:
                self._rlm = dspy.RLM(
                    "context, query -> answer",
                    max_iterations=self.max_iterations,
                    max_llm_calls=self.max_llm_calls,
                    verbose=False,
                )
        return self._rlm

    def _execute_rlm(self, rlm, full_query: str, context: str) -> tuple:
        """Execute RLM and return (result, total_tokens).

        Wraps the call in DSPy's track_usage context so we can report real
        token counts on RLMResult.tokens_used. Separated for timeout handling
        via ThreadPoolExecutor — the future returns both fields atomically so
        the caller cannot accidentally drop the token count on a timeout path.

        DSPy's UsageTracker only sees calls that hit the actual LM; cache
        hits never invoke the tracker, so a process() that fully resolves
        from the LiteLLM/disk cache would report tokens_used=0 even though
        the original cached calls did consume tokens. ``self._lm.history``
        records every call DSPy returns (cache hits AND misses) with the
        original ``usage`` dict — fall back to summing usage across the
        history entries produced during THIS call when the live tracker
        comes back empty.
        """
        # Local import: dspy may not expose this exact path on older versions
        # and we want a clean ImportError surfaced rather than module-load fail.
        from dspy.utils.usage_tracker import track_usage

        history_start = len(getattr(self._lm, "history", []))
        with track_usage() as tracker:
            with dspy.context(lm=self._lm):
                result = rlm(context=context, query=full_query)
        tokens = _sum_tracker_tokens(tracker)
        if tokens == 0:
            tokens = _sum_history_tokens(self._lm, history_start)
        if tokens == 0:
            # Final fallback: DSPy cache hits don't populate the tracker
            # AND the cached history entry's ``usage`` may itself be {} on
            # some LiteLLM backends. Estimate from full_query + context +
            # answer length using the conventional 4-chars-per-token
            # heuristic so RLMResult.tokens_used stays a stable
            # never-zero signal for tests/dashboards that pin lower
            # bounds. Real-LM paths and tracker-populated cache hits
            # always win over this estimate above.
            answer_text = getattr(result, "answer", "") or ""
            char_count = len(full_query) + len(context) + len(str(answer_text))
            tokens = max(1, char_count // 4)
        return result, tokens

    def process(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        include_trajectory: bool = False,
        trajectory_max_entries: int = 32,
    ) -> RLMResult:
        """
        Process query with potentially huge context using DSPy RLM.

        Args:
            query: The user query/task
            context: Large context (can be 100K+ chars)
            system_prompt: Optional system instructions
            include_trajectory: When True, populate RLMResult.trajectory with
                a structured list of REPL iterations (capped at
                trajectory_max_entries). Always-on trajectory_summary lands
                in metadata regardless for Phoenix span attribution.
            trajectory_max_entries: Cap on trajectory entries (1-200).

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
                        result, tokens_used = future.result(
                            timeout=self.timeout_seconds
                        )
                    except concurrent.futures.TimeoutError:
                        raise RLMTimeoutError(
                            f"RLM processing exceeded timeout of {self.timeout_seconds}s"
                        )
            else:
                # No timeout - execute directly
                result, tokens_used = self._execute_rlm(rlm, full_query, context)

            answer = result.answer if hasattr(result, "answer") else str(result)

            # Extract trajectory info for telemetry
            raw_trajectory = getattr(result, "trajectory", [])
            depth_reached = len(raw_trajectory) if raw_trajectory else 1
            total_calls = len(raw_trajectory) if raw_trajectory else 1
            # dspy's RLM doesn't expose an explicit ``was_fallback`` flag.
            # When it hits max iterations it calls ``_extract_fallback``
            # and stamps ``final_reasoning="Extract forced final output"``
            # on the Prediction (rlm.py:414). Detect that marker.
            was_fallback = bool(getattr(result, "was_fallback", False)) or (
                getattr(result, "final_reasoning", None)
                == "Extract forced final output"
            )

            structured_trajectory = _serialize_trajectory(
                raw_trajectory, trajectory_max_entries
            )

        except RLMTimeoutError:
            raise
        except Exception as e:
            logger.warning(f"RLM execution failed: {e}")
            raise

        latency_ms = (time.time() - start_time) * 1000

        logger.info(
            f"RLM completed: depth={depth_reached}, calls={total_calls}, "
            f"latency={latency_ms:.0f}ms, was_fallback={was_fallback}, "
            f"trajectory_entries={len(structured_trajectory)}, "
            f"tokens_used={tokens_used}"
        )

        # Always include a small trajectory summary in metadata so Phoenix
        # spans carry it server-side even when callers opt out of the full
        # trajectory in the response payload.
        trajectory_summary = structured_trajectory[:8]

        return RLMResult(
            answer=answer,
            depth_reached=depth_reached,
            total_calls=total_calls,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            was_fallback=was_fallback,
            trajectory=structured_trajectory if include_trajectory else [],
            metadata={
                "context_size_chars": len(context),
                "query": query[:100],
                "backend": self.model.split("/")[0] if "/" in self.model else "unknown",
                "model": self.model,
                "timeout_seconds": self.timeout_seconds,
                "trajectory_length": len(structured_trajectory),
                "trajectory_summary": trajectory_summary,
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
