"""Phase 11 — DeepSynthesisWorkflow composition end-to-end.

Pins the four cost-bound invariants documented in the workflow's
module docstring:

  * Per-tenant sliding-window rate limit (12-per-hour denies the 13th).
  * Hard cap on total LLM + sub-agent calls per invocation.
  * Iteration cap so a runaway trajectory always terminates.
  * Bounded fan-out per round.

Plus the cooperative happy path: a real RLM (in-cluster vLLM) over a
seeded set of sub-agents converges to ``was_submitted=True`` within
the configured ceilings.

For the cost-bound tests, the workflow's module docstring explicitly
documents the dispatcher and the RLM as test seams ("Keeping it
injected lets tests swap in a deterministic stub"). The tests use
those seams instead of poking the LM into emitting magic markers,
because the *contract under test* is the workflow's control flow,
not the LM's prompt-following ability.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List

import httpx
import pytest

from cogniverse_agents.deep_synthesis_workflow import (
    SUBMIT_TOKEN,
    DeepSynthesisConfig,
    DeepSynthesisRateLimiter,
    DeepSynthesisWorkflow,
)
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from tests.e2e.conftest import run_async, skip_if_no_runtime, unique_id

# ---------------------------------------------------------------------------
# vLLM port-forward fixture (mirrors test_rlm_telemetry_e2e.py:38-94 pattern)
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def vllm_student_url() -> Iterator[str]:
    """kubectl port-forward to cogniverse-vllm-llm-student:8000."""
    deno_bin = os.path.expanduser("~/.deno/bin")
    if os.path.isdir(deno_bin) and deno_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{deno_bin}:{os.environ.get('PATH', '')}"

    port = _free_port()
    proc = subprocess.Popen(
        [
            "kubectl",
            "port-forward",
            "-n",
            "cogniverse",
            "svc/cogniverse-vllm-llm-student",
            f"{port}:8000",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}/v1"
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                r = httpx.get(f"{base}/models", timeout=2)
                if r.status_code == 200 and r.json().get("data"):
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            pytest.fail(
                f"kubectl port-forward to vllm-llm-student never came up "
                f"on {base} within 30s — DeepSynthesis test can't run"
            )
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def llm_config(vllm_student_url: str) -> LLMEndpointConfig:
    return LLMEndpointConfig(
        model="openai/google/gemma-4-e4b-it",
        api_base=vllm_student_url,
        api_key="not-required",
        temperature=0.1,
        max_tokens=512,
    )


# ---------------------------------------------------------------------------
# Stubs for cost-bound tests
# ---------------------------------------------------------------------------


@dataclass
class _StubRLMResult:
    """Minimal RLMResult shape consumed by DeepSynthesisWorkflow.

    The workflow only reads ``.answer``; building a full RLMResult would
    pull in dspy/Deno just for an attribute string the workflow then
    unpacks. The dataclass keeps the seam honest about what it's
    standing in for.
    """

    answer: str


class _ScriptedRLM:
    """RLM stub: yields a scripted sequence of answers across .process calls.

    Used to drive the workflow into the exact control-flow branch under
    test (always-ask, ask-then-submit, or stall). A real LM is too
    non-deterministic for cap/fan-out invariants; a scripted seam keeps
    those invariants testable while still exercising the real workflow.
    """

    def __init__(self, scripted_answers: List[str]) -> None:
        self._answers = list(scripted_answers)
        self.process_calls = 0

    def process(self, *, query: str, context: str) -> _StubRLMResult:
        self.process_calls += 1
        if self._answers:
            return _StubRLMResult(answer=self._answers.pop(0))
        return _StubRLMResult(answer="(scripted RLM exhausted)")


# ---------------------------------------------------------------------------
# 1. DeepSynthesisOverHundredDocuments — real RLM + Mem0-backed dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestDeepSynthesisOverHundredDocuments:
    """Pre-write 100 external_doc memories under one tenant. Workflow's
    dispatcher fetches matching memories per sub-query keyword → real
    RLM gets real evidence.

    Pins:
      * Workflow terminates within hard_call_cap=50 + max_iterations=4.
      * subagent_calls_made >= 5 (all 5 seed agents fire).
      * Exactly one terminal branch (was_submitted XOR was_capped XOR
        rate_limited).
      * answer length in [50, 50_000] (non-trivial output bound).
      * answer contains "lithium" or "lithium-ion" (catches the
        regression where the LM ignores the seeded evidence and emits
        unrelated text).
      * trajectory's iteration-0 names exactly match seed_subagents.
    """

    def test_workflow_terminates_within_bounds_with_real_lm(
        self, llm_config: LLMEndpointConfig
    ) -> None:
        run_async(self._async_workflow_terminates(llm_config))

    async def _async_workflow_terminates(self, llm_config: LLMEndpointConfig) -> None:
        from pathlib import Path

        from cogniverse_agents.inference.rlm_inference import RLMInference
        from cogniverse_core.memory.manager import Mem0MemoryManager
        from cogniverse_core.memory.provenance import (
            CitationRef,
            DerivationKind,
            attach_to_metadata,
            make_provenance,
        )
        from cogniverse_core.memory.schema import build_default_registry
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_foundation.config.unified_config import SystemConfig
        from cogniverse_vespa.config.config_store import VespaConfigStore

        Mem0MemoryManager._instances.clear()
        tenant_id = unique_id("rlm_deep") + ":t1"
        cm = ConfigManager(
            store=VespaConfigStore(backend_url="http://localhost", backend_port=8080)
        )
        # In-memory only: cm.set_system_config would persist a denseon-only
        # localhost URL map into config_metadata and starve the in-cluster
        # ingestor (which reads inference_service_urls from the same store).
        cm._system_config_cache = SystemConfig(  # noqa: SLF001
            backend_url="http://localhost",
            backend_port=8080,
            inference_service_urls={"denseon": "http://localhost:29006"},
        )
        mm = Mem0MemoryManager(tenant_id=tenant_id)
        mm.initialize(
            backend_host="http://localhost",
            backend_port=8080,
            backend_config_port=19071,
            base_schema_name="agent_memories",
            llm_model="google/gemma-4-e4b-it",
            embedding_model="lightonai/DenseOn",
            llm_base_url=("http://cogniverse-vllm-llm-student.cogniverse:8000/v1"),
            embedder_base_url="http://localhost:29006",
            auto_create_schema=True,
            config_manager=cm,
            schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
            knowledge_registry=build_default_registry(),
        )

        # Five topical buckets, 20 docs each = 100 total external_doc
        # memories under one tenant.
        topics = {
            "market": "Lithium demand tripled between 2020 and 2024.",
            "geology": "Lithium reserves are concentrated in South America.",
            "supply": "Lithium recycling remains an industrial challenge.",
            "ev": "Lithium-ion batteries power most modern EVs.",
            "policy": "Several governments classify lithium as critical.",
        }
        try:
            written: Dict[str, List[str]] = {topic: [] for topic in topics}
            for topic, base_content in topics.items():
                for i in range(20):
                    prov = make_provenance(
                        written_by="agent:phase11",
                        derivation_kind=DerivationKind.DIRECT_INGEST,
                        confidence=0.9,
                        derived_from=[CitationRef.external(f"phase11://{topic}/{i}")],
                    )
                    metadata = attach_to_metadata(
                        {"kind": "external_doc", "subject_key": f"{topic}.{i}"},
                        prov,
                    )
                    mid = mm.add_memory(
                        content=f"{base_content} (doc {topic}#{i})",
                        tenant_id=tenant_id,
                        agent_name=f"{topic}_agent",
                        metadata=metadata,
                        infer=False,
                    )
                    assert mid is not None
                    written[topic].append(mid)

            # Dispatcher: each sub-agent name names a topic. The
            # dispatcher returns the first matching memory's content
            # so the workflow consumes real Mem0-backed evidence.
            async def dispatcher(query: str, name: str) -> str:
                topic = name.replace("_agent", "")
                ids = written.get(topic, [])
                if not ids:
                    return f"(no data for {name})"
                memory = mm.memory.get(ids[0])
                if isinstance(memory, dict):
                    return (
                        memory.get("memory")
                        or memory.get("content")
                        or f"(empty memory for {name})"
                    )
                return f"(memory shape unexpected for {name})"

            rlm = RLMInference(
                llm_config=llm_config,
                max_iterations=4,
                max_llm_calls=20,
                timeout_seconds=600,
            )
            wf = DeepSynthesisWorkflow(
                rlm=rlm,
                sub_agent_dispatcher=dispatcher,
                config=DeepSynthesisConfig(
                    rate_limit_per_hour=10,
                    hard_call_cap=50,
                    max_iterations=4,
                    max_subagent_calls_per_round=5,
                ),
            )
            seed = [f"{t}_agent" for t in topics]
            result = await wf.run(
                query=(
                    "Synthesise the role of lithium in modern energy "
                    "across markets, geology, supply, EVs, and policy."
                ),
                tenant_id=tenant_id,
                seed_subagents=seed,
            )

            # Exactly one terminal branch fires.
            terminal_flags = (
                result.was_submitted,
                result.was_capped,
                result.was_rate_limited,
            )
            assert sum(bool(f) for f in terminal_flags) == 1, terminal_flags
            assert result.was_rate_limited is False, result

            # Iteration 0 fires all 5 seed sub-agents.
            assert result.subagent_calls_made >= 5, result.subagent_calls_made
            # Hard cap honoured.
            total = result.subagent_calls_made + result.llm_calls_used
            assert total <= 50, (total, result)
            # Iteration count strictly bounded by config.
            assert 1 <= result.iterations_used <= 4, result.iterations_used

            # Iteration 0 trajectory carries one entry per seed agent;
            # exact name set pinned.
            iter0_names = sorted(
                t["name"]
                for t in result.trajectory
                if t["kind"] == "subagent" and t["iter"] == 0
            )
            assert iter0_names == sorted(seed), iter0_names

            # Answer non-trivial AND topical (catches a regression
            # where the LM returns unrelated text or echoes the prompt).
            assert 50 <= len(result.answer) <= 50_000, len(result.answer)
            assert "lithium" in result.answer.lower(), result.answer[:300]
        finally:
            for topic in topics:
                try:
                    mm.clear_agent_memory(tenant_id, f"{topic}_agent")
                except Exception:
                    pass
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 2. Rate limiter denies the 13th call in a 12/hour window
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestRateLimiterPerTenantSlidingWindow:
    """12 admits return True; 13th is False; a different tenant still admits."""

    def test_thirteenth_call_denied_separate_tenant_admitted(self) -> None:
        run_async(self._async_thirteenth_call_denied())

    async def _async_thirteenth_call_denied(self) -> None:
        rl = DeepSynthesisRateLimiter(rate_limit_per_hour=12)
        tenant_a = unique_id("rlm_rl") + ":t1"
        tenant_b = unique_id("rlm_rl") + ":t2"

        admits_a = [await rl.try_acquire(tenant_a) for _ in range(12)]
        assert admits_a == [True] * 12
        # 13th is denied for tenant A.
        assert await rl.try_acquire(tenant_a) is False
        # Tenant B has its own bucket — first acquire admits.
        assert await rl.try_acquire(tenant_b) is True
        # remaining() returns the exact quota left in the window.
        assert await rl.remaining(tenant_a) == 0
        assert await rl.remaining(tenant_b) == 11

    def test_constructor_rejects_zero_or_negative_limit(self) -> None:
        # Pure sync — no awaits — but kept inside the class for grouping.
        with pytest.raises(ValueError, match="rate_limit_per_hour must be >= 1"):
            DeepSynthesisRateLimiter(rate_limit_per_hour=0)


# ---------------------------------------------------------------------------
# 3. Hard call cap halts the trajectory at the exact bound
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestHardCallCapTrips:
    """RLM always asks for one more sub-agent → cap=3 trips on iteration 2."""

    def test_cap_set_to_three_stops_at_three_calls(self) -> None:
        run_async(self._async_cap_set_to_three())

    async def _async_cap_set_to_three(self) -> None:
        # The RLM never SUBMITs — it always asks for one more sub-agent.
        # The workflow should trip cap_reached on iteration 2 (after
        # iteration 0 used 1 sub-agent, then 1 RLM step + 1 sub-agent).
        scripted_rlm = _ScriptedRLM(scripted_answers=["ASK(more: dig deeper)"] * 8)

        async def dispatcher(query: str, name: str) -> str:
            return f"snippet from {name}"

        wf = DeepSynthesisWorkflow(
            rlm=scripted_rlm,  # type: ignore[arg-type]
            sub_agent_dispatcher=dispatcher,
            config=DeepSynthesisConfig(
                rate_limit_per_hour=100,
                hard_call_cap=3,
                max_iterations=10,
                max_subagent_calls_per_round=5,
            ),
        )
        tenant_id = unique_id("rlm_cap") + ":t1"
        result = await wf.run(
            query="dig forever",
            tenant_id=tenant_id,
            seed_subagents=["seed_a"],
        )
        # Iteration 0: 1 sub-agent (seed_a) → 1 call.
        # Iteration 1: 1 RLM step (call #2) + 1 sub-agent (call #3).
        # Iteration 2: cap_reached BEFORE doing more work.
        assert result.was_capped is True, result
        assert result.was_submitted is False, result
        assert result.subagent_calls_made == 2, result.subagent_calls_made
        assert result.llm_calls_used == 1, result.llm_calls_used
        assert result.subagent_calls_made + result.llm_calls_used == 3
        # The cap_reached marker is the last trajectory entry.
        assert result.trajectory[-1] == {"iter": 2, "kind": "cap_reached"}


# ---------------------------------------------------------------------------
# 4. Bounded fan-out per round
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestBoundedFanOutPerRound:
    """RLM asks for 10 sub-agents in one round → only 2 fire (cap)."""

    def test_per_round_cap_is_two_drops_eight_extras(self) -> None:
        run_async(self._async_per_round_cap())

    async def _async_per_round_cap(self) -> None:
        # Single RLM step requests 10 ASKs; max_subagent_calls_per_round=2
        # should pick the first 2 only. Then SUBMIT to terminate cleanly.
        ten_asks = " ".join(f"ASK(agent_{i}: q{i})" for i in range(10))
        scripted_rlm = _ScriptedRLM(
            scripted_answers=[
                ten_asks,  # iter 1: requests 10 asks → only 2 fire
                f"final answer here {SUBMIT_TOKEN}",  # iter 2: submits
            ]
        )
        dispatched: List[str] = []

        async def dispatcher(query: str, name: str) -> str:
            dispatched.append(name)
            return f"snippet from {name}"

        wf = DeepSynthesisWorkflow(
            rlm=scripted_rlm,  # type: ignore[arg-type]
            sub_agent_dispatcher=dispatcher,
            config=DeepSynthesisConfig(
                rate_limit_per_hour=100,
                hard_call_cap=200,
                max_iterations=5,
                max_subagent_calls_per_round=2,
            ),
        )
        tenant_id = unique_id("rlm_fan") + ":t1"
        result = await wf.run(
            query="bounded fan-out",
            tenant_id=tenant_id,
            seed_subagents=["seed_only"],
        )
        # Iteration 0 dispatched seed_only (1 sub-agent).
        # Iteration 1 RLM emits 10 asks but only the first 2 dispatch.
        # Iteration 2 RLM submits.
        assert result.was_submitted is True, result
        assert result.was_capped is False, result
        # subagent_calls_made = 1 (seed) + 2 (round-1 cap) = 3.
        assert result.subagent_calls_made == 3, result.subagent_calls_made
        # Of the 11 names the RLM and seed referenced, exactly 3 dispatched:
        # the seed plus the first 2 of the 10 asks (agent_0, agent_1).
        # asyncio.gather preserves output order, but the in-coroutine
        # append order is concurrency-dependent — check membership not
        # order for the round-1 batch.
        assert dispatched[0] == "seed_only", dispatched
        assert sorted(dispatched[1:]) == ["agent_0", "agent_1"], dispatched
        # The final answer strips SUBMIT() and pins the prefix.
        assert result.answer == "final answer here"


# ---------------------------------------------------------------------------
# 5. Rate-limited invocation returns was_rate_limited without running
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestRateLimitedInvocationReturnsFlag:
    """Once the limiter denies, .run returns immediately with was_rate_limited."""

    def test_run_after_quota_exhausted_returns_flag(self) -> None:
        run_async(self._async_run_after_quota_exhausted())

    async def _async_run_after_quota_exhausted(self) -> None:
        # Limit = 1; first acquire admits, second denies.
        rl = DeepSynthesisRateLimiter(rate_limit_per_hour=1)
        scripted_rlm = _ScriptedRLM(scripted_answers=[f"answer {SUBMIT_TOKEN}"])
        dispatched: List[str] = []

        async def dispatcher(query: str, name: str) -> str:
            dispatched.append(name)
            return "snippet"

        wf = DeepSynthesisWorkflow(
            rlm=scripted_rlm,  # type: ignore[arg-type]
            sub_agent_dispatcher=dispatcher,
            config=DeepSynthesisConfig(
                rate_limit_per_hour=1,
                hard_call_cap=10,
                max_iterations=2,
                max_subagent_calls_per_round=1,
            ),
            rate_limiter=rl,
        )
        tenant_id = unique_id("rlm_rlex") + ":t1"

        first = await wf.run(
            query="first call", tenant_id=tenant_id, seed_subagents=["a"]
        )
        # First call ran successfully — submitted.
        assert first.was_submitted is True, first
        assert first.was_rate_limited is False, first
        assert dispatched == ["a"], dispatched

        second = await wf.run(
            query="second call", tenant_id=tenant_id, seed_subagents=["b"]
        )
        # Second call: rate-limited, returns immediately, no dispatch.
        assert second.was_rate_limited is True, second
        assert second.was_submitted is False, second
        assert second.iterations_used == 0
        assert second.subagent_calls_made == 0
        assert second.llm_calls_used == 0
        assert second.answer == ""
        # Dispatcher was NOT invoked for "b".
        assert dispatched == ["a"], dispatched
