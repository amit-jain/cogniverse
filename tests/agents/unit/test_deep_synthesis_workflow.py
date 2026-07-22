"""Unit tests for DeepSynthesisWorkflow."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from cogniverse_agents.deep_synthesis_workflow import (
    SUBMIT_TOKEN,
    DeepSynthesisConfig,
    DeepSynthesisRateLimiter,
    DeepSynthesisWorkflow,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@dataclass
class _StubRLMResult:
    answer: str
    tokens_used: int = 0
    was_fallback: bool = False
    iterations: int = 1
    trajectory: list = None


class _StubRLM:
    """RLMInference stub that scripts a sequence of step answers."""

    def __init__(self, scripted_answers: List[str]) -> None:
        self._answers = list(scripted_answers)
        self.calls: List[Dict[str, Any]] = []

    def process(self, *, query: str, context: str = "", **_kw):
        self.calls.append({"query": query, "context": context})
        if not self._answers:
            return _StubRLMResult(answer=f"FINAL{SUBMIT_TOKEN}")
        return _StubRLMResult(answer=self._answers.pop(0))


def _dispatcher_returning(snippets_by_name: Dict[str, str]):
    """Builds a sub-agent dispatcher that returns scripted snippets."""

    async def _dispatch(query: str, name: str) -> str:
        return snippets_by_name.get(name, f"empty:{name}")

    return _dispatch


def _flaky_dispatcher(failing_names: set):
    """A dispatcher that raises for any sub-agent in failing_names."""

    async def _dispatch(query: str, name: str) -> str:
        if name in failing_names:
            raise RuntimeError(f"simulated failure: {name}")
        return f"ok:{name}"

    return _dispatch


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_under_limit_allowed(self):
        rl = DeepSynthesisRateLimiter(rate_limit_per_hour=3)
        assert await rl.try_acquire("acme")
        assert await rl.try_acquire("acme")
        assert await rl.try_acquire("acme")

    @pytest.mark.asyncio
    async def test_over_limit_denied(self):
        rl = DeepSynthesisRateLimiter(rate_limit_per_hour=2)
        assert await rl.try_acquire("acme")
        assert await rl.try_acquire("acme")
        assert (await rl.try_acquire("acme")) is False

    @pytest.mark.asyncio
    async def test_per_tenant_isolation(self):
        rl = DeepSynthesisRateLimiter(rate_limit_per_hour=1)
        assert await rl.try_acquire("acme")
        # Different tenant → independent budget.
        assert await rl.try_acquire("globex")

    @pytest.mark.asyncio
    async def test_remaining_tracks_usage(self):
        rl = DeepSynthesisRateLimiter(rate_limit_per_hour=5)
        assert await rl.remaining("acme") == 5
        await rl.try_acquire("acme")
        await rl.try_acquire("acme")
        assert await rl.remaining("acme") == 3

    @pytest.mark.asyncio
    async def test_window_expiry_releases_capacity(self, monkeypatch):
        # Force the rate window to a tiny value via patching the constant.
        from cogniverse_agents import deep_synthesis_workflow as dsw

        monkeypatch.setattr(dsw, "_RATE_WINDOW_SECONDS", 0.05)
        rl = DeepSynthesisRateLimiter(rate_limit_per_hour=1)
        assert await rl.try_acquire("acme")
        assert (await rl.try_acquire("acme")) is False
        await asyncio.sleep(0.1)
        # Old timestamp aged out → capacity returned.
        assert await rl.try_acquire("acme")

    def test_invalid_limit_rejected(self):
        with pytest.raises(ValueError):
            DeepSynthesisRateLimiter(rate_limit_per_hour=0)


@pytest.mark.asyncio
class TestWorkflowSubmitPath:
    async def test_submits_immediately_on_first_step(self):
        rlm = _StubRLM([f"answer: 42 {SUBMIT_TOKEN}"])
        wf = DeepSynthesisWorkflow(
            rlm=rlm,
            sub_agent_dispatcher=_dispatcher_returning({"search": "vespa hit"}),
        )
        out = await wf.run(query="q", tenant_id="acme", seed_subagents=["search"])
        assert out.was_submitted is True
        assert out.iterations_used == 1
        # 1 seed dispatch + 1 RLM call.
        assert out.subagent_calls_made == 1
        assert out.llm_calls_used == 1
        assert "answer: 42" in out.answer
        assert SUBMIT_TOKEN not in out.answer  # token stripped

    async def test_submits_after_one_ask_round(self):
        # Step 1: ask the document agent for more; step 2: submit.
        rlm = _StubRLM(
            [
                "ASK(document_agent: get the EU table)",
                f"final answer with the table {SUBMIT_TOKEN}",
            ]
        )
        wf = DeepSynthesisWorkflow(
            rlm=rlm,
            sub_agent_dispatcher=_dispatcher_returning(
                {"search": "search hit", "document_agent": "EU table content"}
            ),
        )
        out = await wf.run(query="q", tenant_id="acme", seed_subagents=["search"])
        assert out.was_submitted is True
        # 1 seed + 1 ASK = 2 sub-agent calls. 2 RLM calls.
        assert out.subagent_calls_made == 2
        assert out.llm_calls_used == 2
        # Trajectory mentions the document_agent fan-out.
        assert any(
            entry.get("kind") == "subagent" and entry.get("name") == "document_agent"
            for entry in out.trajectory
        )


@pytest.mark.asyncio
class TestWorkflowBounds:
    async def test_rate_limit_short_circuits(self):
        # Pre-exhaust the limiter for tenant=acme, then run.
        cfg = DeepSynthesisConfig(rate_limit_per_hour=1)
        limiter = DeepSynthesisRateLimiter(rate_limit_per_hour=1)
        await limiter.try_acquire("acme")
        rlm = _StubRLM([f"x {SUBMIT_TOKEN}"])
        wf = DeepSynthesisWorkflow(
            rlm=rlm,
            sub_agent_dispatcher=_dispatcher_returning({"s": "ok"}),
            config=cfg,
            rate_limiter=limiter,
        )
        out = await wf.run(query="q", tenant_id="acme", seed_subagents=["s"])
        assert out.was_rate_limited is True
        assert out.subagent_calls_made == 0
        assert out.llm_calls_used == 0
        # RLM was never even consulted.
        assert rlm.calls == []

    async def test_iteration_cap_terminates_runaway(self):
        # RLM never submits; just keeps asking. Cap iterations at 2.
        rlm = _StubRLM(
            [
                "ASK(s: more1)",
                "ASK(s: more2)",
                "ASK(s: more3)",
                "ASK(s: more4)",
            ]
        )
        wf = DeepSynthesisWorkflow(
            rlm=rlm,
            sub_agent_dispatcher=_dispatcher_returning({"s": "ok"}),
            config=DeepSynthesisConfig(max_iterations=2),
        )
        out = await wf.run(query="q", tenant_id="acme", seed_subagents=["s"])
        assert out.was_submitted is False
        assert out.was_capped is True
        assert out.iterations_used == 2
        # 1 seed + 2 ASK rounds * 1 sub-agent = 3 dispatches.
        assert out.subagent_calls_made == 3

    async def test_hard_call_cap_short_circuits(self):
        # Set a tiny call cap so we hit it immediately.
        rlm = _StubRLM(["ASK(s: more1)", "ASK(s: more2)", "ASK(s: more3)"])
        wf = DeepSynthesisWorkflow(
            rlm=rlm,
            sub_agent_dispatcher=_dispatcher_returning({"s": "ok"}),
            config=DeepSynthesisConfig(hard_call_cap=2, max_iterations=10),
        )
        out = await wf.run(query="q", tenant_id="acme", seed_subagents=["s"])
        # Initial seed (1) + 1 RLM call = 2 → cap reached at start of iter 2.
        assert out.was_capped is True
        assert (out.subagent_calls_made + out.llm_calls_used) >= 2

    async def test_max_subagent_calls_per_round_caps_seed_fanout(self):
        # 10 seed agents but cap is 3.
        rlm = _StubRLM([f"done {SUBMIT_TOKEN}"])
        wf = DeepSynthesisWorkflow(
            rlm=rlm,
            sub_agent_dispatcher=_dispatcher_returning(
                {f"a{i}": "ok" for i in range(10)}
            ),
            config=DeepSynthesisConfig(max_subagent_calls_per_round=3),
        )
        out = await wf.run(
            query="q",
            tenant_id="acme",
            seed_subagents=[f"a{i}" for i in range(10)],
        )
        # Only the first 3 seed agents got dispatched.
        assert out.subagent_calls_made == 3

    async def test_stalled_no_asks_terminates(self):
        # RLM returns text without ASK() and without SUBMIT() — workflow
        # must not loop forever; it returns capped with whatever it has.
        rlm = _StubRLM(["I have no more questions but no answer either"])
        wf = DeepSynthesisWorkflow(
            rlm=rlm,
            sub_agent_dispatcher=_dispatcher_returning({"s": "ok"}),
            config=DeepSynthesisConfig(max_iterations=5),
        )
        out = await wf.run(query="q", tenant_id="acme", seed_subagents=["s"])
        assert out.was_capped is True
        assert out.was_submitted is False
        # Trajectory records the stall.
        assert any(entry.get("kind") == "stalled_no_asks" for entry in out.trajectory)


@pytest.mark.asyncio
class TestSubagentFailureTolerance:
    async def test_failing_subagent_dropped_others_kept(self):
        rlm = _StubRLM([f"done {SUBMIT_TOKEN}"])
        wf = DeepSynthesisWorkflow(
            rlm=rlm,
            sub_agent_dispatcher=_flaky_dispatcher({"broken"}),
        )
        out = await wf.run(
            query="q",
            tenant_id="acme",
            seed_subagents=["good", "broken", "good2"],
        )
        # broken is silently dropped; the other two count.
        assert out.subagent_calls_made == 2

    async def test_all_subagents_failing_still_runs_rlm(self):
        rlm = _StubRLM([f"empty-handed {SUBMIT_TOKEN}"])
        wf = DeepSynthesisWorkflow(
            rlm=rlm,
            sub_agent_dispatcher=_flaky_dispatcher({"a", "b"}),
        )
        out = await wf.run(query="q", tenant_id="acme", seed_subagents=["a", "b"])
        assert out.subagent_calls_made == 0
        assert out.llm_calls_used == 1
        assert out.was_submitted is True


class TestParseAsks:
    def test_single_ask(self):
        out = DeepSynthesisWorkflow._parse_asks("ASK(s: hello)")
        assert out == [("s", "hello")]

    def test_multiple_asks(self):
        out = DeepSynthesisWorkflow._parse_asks(
            "I'd like ASK(a: q1) and also ASK(b: q2)"
        )
        assert out == [("a", "q1"), ("b", "q2")]

    def test_malformed_marker_skipped(self):
        # Missing closing paren — skipped without error.
        out = DeepSynthesisWorkflow._parse_asks("ASK(s: hello and more")
        assert out == []

    def test_no_colon_skipped(self):
        out = DeepSynthesisWorkflow._parse_asks("ASK(no colon here)")
        assert out == []

    def test_no_marker_returns_empty(self):
        assert DeepSynthesisWorkflow._parse_asks("just text") == []
