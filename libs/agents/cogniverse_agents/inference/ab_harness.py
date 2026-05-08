"""RLM A/B harness (B.5).

Runs the same query through both arms — once without RLM (a single LM call
on the raw context), once with RLM (recursive REPL) — and returns a
typed comparison so callers can A/B test RLM's quality vs cost trade-off
on real workloads.

Both arms share an ``ab_id`` (stamped in result metadata) so Phoenix
spans correlate across the pair, and the dashboard's A/B tile can
aggregate the deltas without joining on opaque trace ids.

Caller-supplied ``judge`` is optional. When provided, the harness invokes
it on each arm's answer to produce a per-arm quality score. Without a
judge, callers can still compare latency + tokens + fallback rate.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import dspy

if TYPE_CHECKING:
    from cogniverse_core.events import EventQueue

from cogniverse_agents.inference.rlm_inference import RLMInference, RLMResult
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig

logger = logging.getLogger(__name__)


@dataclass
class ABArmResult:
    """Per-arm output from an A/B run."""

    arm: str  # "with_rlm" | "without_rlm"
    answer: str
    latency_ms: float
    tokens_used: int
    was_fallback: bool
    judge_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABComparison:
    """Side-by-side comparison of the two arms.

    Latency / token deltas are positive when the with-RLM arm is *more*
    expensive than the without-RLM arm (the typical sign — RLM trades
    cost for quality). Judge delta is positive when with-RLM scores
    higher; it is None when no judge was supplied.
    """

    latency_delta_ms: float
    tokens_delta: int
    judge_delta: Optional[float]
    rlm_was_fallback: bool


@dataclass
class ABResult:
    """Full A/B run: both arms + the side-by-side comparison."""

    ab_id: str
    query: str
    context_size_chars: int
    without_rlm: ABArmResult
    with_rlm: ABArmResult
    comparison: ABComparison

    def to_telemetry_dict(self) -> Dict[str, Any]:
        """Flatten into a single dict suitable for a Phoenix span attribute set."""
        return {
            "ab_id": self.ab_id,
            "ab_query": self.query[:120],
            "ab_context_chars": self.context_size_chars,
            "ab_without_rlm_latency_ms": self.without_rlm.latency_ms,
            "ab_without_rlm_tokens": self.without_rlm.tokens_used,
            "ab_without_rlm_judge": self.without_rlm.judge_score,
            "ab_with_rlm_latency_ms": self.with_rlm.latency_ms,
            "ab_with_rlm_tokens": self.with_rlm.tokens_used,
            "ab_with_rlm_judge": self.with_rlm.judge_score,
            "ab_with_rlm_was_fallback": self.with_rlm.was_fallback,
            "ab_latency_delta_ms": self.comparison.latency_delta_ms,
            "ab_tokens_delta": self.comparison.tokens_delta,
            "ab_judge_delta": self.comparison.judge_delta,
        }


# Caller-supplied judge: takes (query, context, answer) and returns a score
# in [0.0, 1.0]. When None, judge_score / judge_delta stay None.
JudgeFn = Callable[[str, str, str], float]


class RLMABRunner:
    """A/B harness for comparing RLM-on vs RLM-off arms on the same query.

    Args:
        llm_config: LLM backend to use for both arms (same model so the
            comparison isolates the RLM machinery, not the model).
        judge: Optional quality scorer. Receives (query, context, answer)
            and returns a 0.0–1.0 score. When None, only latency / token /
            fallback are compared.
        timeout_seconds: Per-arm timeout. RLM also bounded internally by
            its own ``timeout_seconds``.
        rlm_max_iterations: ``RLMInference.max_iterations`` for the RLM arm.
        rlm_max_llm_calls: ``RLMInference.max_llm_calls`` for the RLM arm.
        event_queue: Optional EventQueue forwarded to RLMInference for
            real-time progress events on the RLM arm.
        tenant_id: Required when ``event_queue`` is provided.
    """

    def __init__(
        self,
        llm_config: LLMEndpointConfig,
        judge: Optional[JudgeFn] = None,
        timeout_seconds: int = 300,
        rlm_max_iterations: int = 10,
        rlm_max_llm_calls: int = 30,
        event_queue: "Optional[EventQueue]" = None,
        tenant_id: Optional[str] = None,
    ) -> None:
        self._llm_config = llm_config
        self._judge = judge
        self._timeout_seconds = timeout_seconds
        self._rlm_max_iterations = rlm_max_iterations
        self._rlm_max_llm_calls = rlm_max_llm_calls
        self._event_queue = event_queue
        self._tenant_id = tenant_id

    def run(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> ABResult:
        """Run both arms sequentially and return the side-by-side comparison.

        The without-RLM arm goes through a single ``dspy.Predict`` call on
        a ``"context, query -> answer"`` signature so both arms see the
        same input shape. RLM uses :class:`RLMInference`.
        """
        ab_id = uuid.uuid4().hex
        logger.info(
            "RLM A/B run id=%s query=%r context_chars=%d",
            ab_id,
            query[:80],
            len(context),
        )

        without = self._run_without_rlm(query, context, system_prompt)
        with_rlm = self._run_with_rlm(query, context, system_prompt)
        without.metadata["ab_id"] = ab_id
        with_rlm.metadata["ab_id"] = ab_id

        judge_delta: Optional[float] = None
        if without.judge_score is not None and with_rlm.judge_score is not None:
            judge_delta = with_rlm.judge_score - without.judge_score

        comparison = ABComparison(
            latency_delta_ms=with_rlm.latency_ms - without.latency_ms,
            tokens_delta=with_rlm.tokens_used - without.tokens_used,
            judge_delta=judge_delta,
            rlm_was_fallback=with_rlm.was_fallback,
        )
        return ABResult(
            ab_id=ab_id,
            query=query,
            context_size_chars=len(context),
            without_rlm=without,
            with_rlm=with_rlm,
            comparison=comparison,
        )

    def _run_without_rlm(
        self, query: str, context: str, system_prompt: Optional[str]
    ) -> ABArmResult:
        """Single LM call on the raw context — the no-RLM baseline."""
        from dspy.utils.usage_tracker import track_usage

        full_query = f"{system_prompt}\n\n{query}" if system_prompt else query
        lm = create_dspy_lm(self._llm_config)
        predict = dspy.Predict("context, query -> answer")

        start = time.time()
        with track_usage() as tracker:
            with dspy.context(lm=lm):
                result = predict(context=context, query=full_query)
        latency_ms = (time.time() - start) * 1000.0

        # Sum tokens via the same helper RLMInference uses.
        from cogniverse_agents.inference.rlm_inference import _sum_tracker_tokens

        tokens = _sum_tracker_tokens(tracker)
        answer = getattr(result, "answer", "") or ""

        judge_score = (
            self._judge(query, context, answer) if self._judge is not None else None
        )

        return ABArmResult(
            arm="without_rlm",
            answer=str(answer),
            latency_ms=latency_ms,
            tokens_used=int(tokens),
            was_fallback=False,
            judge_score=judge_score,
            metadata={"backend": self._llm_config.model},
        )

    def _run_with_rlm(
        self, query: str, context: str, system_prompt: Optional[str]
    ) -> ABArmResult:
        """RLM arm via RLMInference — the recursive baseline."""
        rlm = RLMInference(
            llm_config=self._llm_config,
            max_iterations=self._rlm_max_iterations,
            max_llm_calls=self._rlm_max_llm_calls,
            timeout_seconds=self._timeout_seconds,
            event_queue=self._event_queue,
            tenant_id=self._tenant_id,
        )
        result: RLMResult = rlm.process(
            query=query,
            context=context,
            system_prompt=system_prompt,
        )

        judge_score = (
            self._judge(query, context, result.answer)
            if self._judge is not None
            else None
        )

        return ABArmResult(
            arm="with_rlm",
            answer=result.answer,
            latency_ms=result.latency_ms,
            tokens_used=result.tokens_used,
            was_fallback=result.was_fallback,
            judge_score=judge_score,
            metadata=dict(result.metadata),
        )
