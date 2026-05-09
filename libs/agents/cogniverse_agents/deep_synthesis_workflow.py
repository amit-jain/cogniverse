"""DeepSynthesisWorkflow (B.7).

Opt-in entry point for queries that need recursive multi-agent
orchestration over a knowledge subgraph that doesn't fit in a normal
plan ("compare these 50 documents across 5 tenants and produce a
unified report"). Runs the orchestrator inside an RLM-style trajectory:
fan out to sub-agents, let the RLM summariser decide whether it has
enough material, and either submit an answer or request another
fan-out round.

This is **not** the default execution path — see the C1 analysis in
the plan. The default Orchestrator stays plan-then-act with parallel
sub-agent fan-out. ``DeepSynthesisWorkflow`` is a separate class so:

  * its cost (rate limit, hard call cap) is local and explicit
  * the default path's Phoenix trace shape stays clean (parent → children)
  * A/B (B.5) can compare deep-synthesis vs flat orchestration on a
    curated benchmark before promoting it broadly

Invariants enforced in V1:
  * **Per-tenant rate limit** (default 5 invocations / hour).
  * **Hard cap on total LLM calls per invocation** (default 200).
  * **Iteration cap** so a runaway trajectory always terminates.
  * **Bounded fan-out** per round (the workflow never dispatches more
    than ``max_subagent_calls_per_round`` parallel sub-agent calls).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional

from cogniverse_agents.inference.rlm_inference import RLMInference

logger = logging.getLogger(__name__)


_DEFAULT_RATE_LIMIT_PER_HOUR = 5
_DEFAULT_HARD_CALL_CAP = 200
_DEFAULT_MAX_ITERATIONS = 8
_DEFAULT_MAX_SUBAGENT_CALLS_PER_ROUND = 6
_RATE_WINDOW_SECONDS = 3600.0
SUBMIT_TOKEN = "SUBMIT()"


# Sub-agent dispatcher contract: caller-provided async callable that takes a
# (query, sub_agent_name) tuple and returns a string snippet for the trajectory.
SubAgentDispatcher = Callable[[str, str], Awaitable[str]]


class DeepSynthesisError(Exception):
    """Base class for deep-synthesis failures."""


class RateLimitedError(DeepSynthesisError):
    """Raised when the per-tenant rate limit denies an invocation."""


@dataclass
class DeepSynthesisConfig:
    rate_limit_per_hour: int = _DEFAULT_RATE_LIMIT_PER_HOUR
    hard_call_cap: int = _DEFAULT_HARD_CALL_CAP
    max_iterations: int = _DEFAULT_MAX_ITERATIONS
    max_subagent_calls_per_round: int = _DEFAULT_MAX_SUBAGENT_CALLS_PER_ROUND


@dataclass
class DeepSynthesisResult:
    """Outcome of a single deep-synthesis invocation."""

    answer: str
    iterations_used: int
    subagent_calls_made: int
    llm_calls_used: int
    was_capped: bool = False
    was_submitted: bool = False
    was_rate_limited: bool = False
    trajectory: List[Dict[str, Any]] = field(default_factory=list)


class DeepSynthesisRateLimiter:
    """Per-tenant sliding-window rate limiter for ``DeepSynthesisWorkflow``.

    Holds a per-tenant ``deque`` of invocation timestamps. ``try_acquire``
    drops anything older than ``_RATE_WINDOW_SECONDS`` first, then admits
    the call when the in-window count is below ``rate_limit_per_hour``.
    """

    def __init__(self, rate_limit_per_hour: int = _DEFAULT_RATE_LIMIT_PER_HOUR) -> None:
        if rate_limit_per_hour < 1:
            raise ValueError("rate_limit_per_hour must be >= 1")
        self._limit = rate_limit_per_hour
        self._buckets: Dict[str, Deque[float]] = {}
        self._lock = asyncio.Lock()

    async def try_acquire(self, tenant_id: str) -> bool:
        async with self._lock:
            bucket = self._buckets.setdefault(tenant_id, deque())
            now = time.time()
            cutoff = now - _RATE_WINDOW_SECONDS
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self._limit:
                return False
            bucket.append(now)
            return True

    async def remaining(self, tenant_id: str) -> int:
        async with self._lock:
            bucket = self._buckets.get(tenant_id)
            if bucket is None:
                return self._limit
            now = time.time()
            cutoff = now - _RATE_WINDOW_SECONDS
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            return max(0, self._limit - len(bucket))


class DeepSynthesisWorkflow:
    """Recursive multi-agent synthesis with hard cost bounds.

    Args:
        rlm: An ``RLMInference`` instance (caller controls model + LLM
            config). The workflow consumes ``rlm.process`` for the
            "have I got enough yet?" decision.
        sub_agent_dispatcher: Async callable that fans a sub-query out to
            a named sub-agent and returns a textual snippet. The
            workflow asks the RLM what to ask next; the dispatcher
            actually does the call. Keeping it injected lets tests swap
            in a deterministic stub.
        config: Bounds (rate limit, call cap, iteration cap).
        rate_limiter: Optional injected limiter for tests; when omitted,
            one is built from ``config.rate_limit_per_hour``.
    """

    def __init__(
        self,
        rlm: RLMInference,
        sub_agent_dispatcher: SubAgentDispatcher,
        config: Optional[DeepSynthesisConfig] = None,
        rate_limiter: Optional[DeepSynthesisRateLimiter] = None,
    ) -> None:
        self._rlm = rlm
        self._dispatch = sub_agent_dispatcher
        self._config = config or DeepSynthesisConfig()
        self._rate = rate_limiter or DeepSynthesisRateLimiter(
            self._config.rate_limit_per_hour
        )

    @property
    def rate_limiter(self) -> DeepSynthesisRateLimiter:
        return self._rate

    async def run(
        self,
        *,
        query: str,
        tenant_id: str,
        seed_subagents: List[str],
    ) -> DeepSynthesisResult:
        """Execute a single deep-synthesis invocation under the configured bounds.

        Args:
            query: The original user query.
            tenant_id: Tenant for rate-limit accounting.
            seed_subagents: Sub-agent names to fan out to in iteration 0.
                Subsequent rounds let the RLM request more sub-agent
                calls by emitting ``ASK(<subagent>: <subquery>)`` tokens
                in its iteration text.
        """
        if not await self._rate.try_acquire(tenant_id):
            logger.warning(
                "DeepSynthesisWorkflow: tenant=%s rate-limited (limit=%d/hour)",
                tenant_id,
                self._config.rate_limit_per_hour,
            )
            return DeepSynthesisResult(
                answer="",
                iterations_used=0,
                subagent_calls_made=0,
                llm_calls_used=0,
                was_rate_limited=True,
            )

        trajectory: List[Dict[str, Any]] = []
        subagent_calls_made = 0
        llm_calls_used = 0
        iterations_used = 0
        gathered: List[str] = []

        # Iteration 0 fan-out: dispatch the seed sub-agents in parallel,
        # capped at max_subagent_calls_per_round.
        seed = seed_subagents[: self._config.max_subagent_calls_per_round]
        if seed:
            results = await self._fan_out(query, seed)
            subagent_calls_made += len(results)
            for name, snippet in results:
                gathered.append(f"[{name}] {snippet}")
                trajectory.append(
                    {"iter": 0, "kind": "subagent", "name": name, "snippet": snippet}
                )

        # Iteration loop: ask the RLM whether to submit or request more.
        for it in range(1, self._config.max_iterations + 1):
            iterations_used = it

            if subagent_calls_made + llm_calls_used >= self._config.hard_call_cap:
                trajectory.append({"iter": it, "kind": "cap_reached"})
                return DeepSynthesisResult(
                    answer="\n\n".join(gathered) if gathered else "",
                    iterations_used=iterations_used,
                    subagent_calls_made=subagent_calls_made,
                    llm_calls_used=llm_calls_used,
                    was_capped=True,
                    trajectory=trajectory,
                )

            context = "\n\n".join(gathered) if gathered else "(no evidence yet)"
            rlm_result = await asyncio.to_thread(
                self._rlm.process,
                query=query,
                context=context,
            )
            llm_calls_used += 1
            iter_text = (rlm_result.answer or "").strip()
            trajectory.append(
                {"iter": it, "kind": "rlm_step", "answer": iter_text[:1000]}
            )

            if SUBMIT_TOKEN in iter_text:
                # Strip the SUBMIT() marker; whatever precedes it is the answer.
                answer = iter_text.replace(SUBMIT_TOKEN, "").strip() or "\n\n".join(
                    gathered
                )
                return DeepSynthesisResult(
                    answer=answer,
                    iterations_used=iterations_used,
                    subagent_calls_made=subagent_calls_made,
                    llm_calls_used=llm_calls_used,
                    was_submitted=True,
                    trajectory=trajectory,
                )

            # Parse ASK(<subagent>: <subquery>) tokens; cap per round.
            asks = self._parse_asks(iter_text)[
                : self._config.max_subagent_calls_per_round
            ]
            if not asks:
                # Trajectory stalled — return what we have, marked as capped.
                trajectory.append({"iter": it, "kind": "stalled_no_asks"})
                return DeepSynthesisResult(
                    answer=iter_text or "\n\n".join(gathered),
                    iterations_used=iterations_used,
                    subagent_calls_made=subagent_calls_made,
                    llm_calls_used=llm_calls_used,
                    was_capped=True,
                    trajectory=trajectory,
                )

            # Dispatch the asks in parallel, also bounded by the cap.
            remaining = self._config.hard_call_cap - (
                subagent_calls_made + llm_calls_used
            )
            asks = asks[: max(0, remaining)]
            if not asks:
                trajectory.append({"iter": it, "kind": "cap_blocks_asks"})
                return DeepSynthesisResult(
                    answer=iter_text or "\n\n".join(gathered),
                    iterations_used=iterations_used,
                    subagent_calls_made=subagent_calls_made,
                    llm_calls_used=llm_calls_used,
                    was_capped=True,
                    trajectory=trajectory,
                )
            ask_pairs = [(subq, name) for (name, subq) in asks]
            results = await self._fan_out_pairs(ask_pairs)
            subagent_calls_made += len(results)
            for name, snippet in results:
                gathered.append(f"[{name}] {snippet}")
                trajectory.append(
                    {"iter": it, "kind": "subagent", "name": name, "snippet": snippet}
                )

        # Iteration cap exhausted without SUBMIT.
        trajectory.append({"iter": iterations_used, "kind": "iteration_cap_exhausted"})
        return DeepSynthesisResult(
            answer="\n\n".join(gathered) if gathered else "",
            iterations_used=iterations_used,
            subagent_calls_made=subagent_calls_made,
            llm_calls_used=llm_calls_used,
            was_capped=True,
            trajectory=trajectory,
        )

    async def _fan_out(self, query: str, sub_agents: List[str]) -> List[tuple]:
        return await self._fan_out_pairs([(query, name) for name in sub_agents])

    async def _fan_out_pairs(self, pairs: List[tuple]) -> List[tuple]:
        """Dispatch many (query, sub_agent_name) calls in parallel."""
        tasks = [self._safe_dispatch(q, n) for (q, n) in pairs]
        results = await asyncio.gather(*tasks)
        return [(name, snippet) for (name, snippet) in results if snippet is not None]

    async def _safe_dispatch(self, q: str, name: str) -> tuple:
        try:
            snippet = await self._dispatch(q, name)
        except Exception as exc:
            logger.debug("DeepSynth: sub-agent %s failed: %s", name, exc)
            snippet = None
        return name, snippet

    @staticmethod
    def _parse_asks(text: str) -> List[tuple]:
        """Pull ``ASK(<subagent>: <subquery>)`` invocations out of an RLM step.

        Tolerates whitespace and trailing punctuation. Skips malformed
        markers without raising — a weird trajectory should not crash
        the workflow.
        """
        out: List[tuple] = []
        cursor = 0
        marker = "ASK("
        while True:
            start = text.find(marker, cursor)
            if start == -1:
                break
            end = text.find(")", start + len(marker))
            if end == -1:
                break
            inner = text[start + len(marker) : end].strip()
            if ":" in inner:
                name, _, subq = inner.partition(":")
                name = name.strip()
                subq = subq.strip()
                if name and subq:
                    out.append((name, subq))
            cursor = end + 1
        return out
