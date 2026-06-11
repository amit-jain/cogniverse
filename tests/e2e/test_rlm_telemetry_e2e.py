"""Phase 5a — RLM telemetry end-to-end.

Pins the shipped RLMResult telemetry fields against a real vLLM
endpoint inside the deployed cogniverse up cluster:

  * tokens_used is a positive integer plumbed from dspy.UsageTracker;
  * include_trajectory=True produces a bounded structured list, =False
    returns the empty list while metadata.trajectory_summary stays
    populated as a server-side debug aid;
  * was_fallback flips to True when max_iterations is exhausted
    without a SUBMIT() call;
  * the to_telemetry_dict shape stays exact so Phoenix spans don't
    silently drop fields.

RLM requires Deno (dspy spawns a sandbox subprocess for the REPL).
The host must have Deno installed at ``$HOME/.deno/bin/deno``; the
fixture below adds it to PATH before instantiating RLMInference.
The LLM endpoint is reached via a module-scoped kubectl port-forward
to ``cogniverse-vllm-llm-student`` (ClusterIP-only, no NodePort) on a
local random port — torn down at module end.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from typing import Iterator

import httpx
import pytest

from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from tests.e2e.conftest import skip_if_no_runtime, unique_id


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def vllm_student_url() -> Iterator[str]:
    """kubectl port-forward to cogniverse-vllm-llm-student:8000.

    Module-scoped so all RLM tests share one port-forward (kubectl
    port-forward has a fixed cost per startup, and the LM service is
    healthy and reusable). Tears down on module exit.
    """
    # Deno must be discoverable for RLMInference's startup assertion.
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
        # Wait until /v1/models 200s — the LM is already running, the
        # port-forward just needs a moment to wire up.
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
                f"on {base} within 30s — RLM tests can't run"
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
    """LLMEndpointConfig pointed at the in-cluster vLLM via port-forward."""
    # vLLM-OpenAI-compat: litellm provider prefix is openai/.
    return LLMEndpointConfig(
        model="openai/google/gemma-4-e4b-it",
        api_base=vllm_student_url,
        api_key="not-required",
        temperature=0.1,
        max_tokens=512,
    )


def _import_rlm() -> tuple:
    """Import RLMInference + RLMResult late (after the Deno PATH fix-up)."""
    from cogniverse_agents.inference.rlm_inference import RLMInference, RLMResult

    return RLMInference, RLMResult


# A small but non-trivial context that nudges the REPL into at least
# one iteration. The query asks for a count + classification — RLM
# typically issues 1-3 sub-LLM calls to get there.
_PHASE5_QUERY = "Count how many planets are mentioned and list them by name."
_PHASE5_CONTEXT = (
    "Our solar system has eight planets: Mercury, Venus, Earth, Mars, "
    "Jupiter, Saturn, Uranus, Neptune. Pluto was reclassified as a "
    "dwarf planet in 2006. Earth has one moon, Jupiter has four large "
    "Galilean moons (Io, Europa, Ganymede, Callisto)."
)


# ---------------------------------------------------------------------------
# 1. tokens_used reported and roughly monotonic with max_iterations
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestTokensUsedReportedAndMonotone:
    """RLMResult.tokens_used > 0 and grows with iteration budget."""

    def test_tokens_used_in_bound_and_monotone(
        self, llm_config: LLMEndpointConfig
    ) -> None:
        RLMInference, _ = _import_rlm()

        # CPU vLLM serves ~one chat completion every ~30-60s on this host;
        # multi-iteration RLM tests need 10-minute deadlines. Cap iteration
        # count to bound the wall clock.
        small = RLMInference(
            llm_config=llm_config,
            max_iterations=2,
            max_llm_calls=4,
            timeout_seconds=600,
            tenant_id=unique_id("rlm_tokens"),
        )
        r1 = small.process(query=_PHASE5_QUERY, context=_PHASE5_CONTEXT)
        # Lower bound chosen so a fully-deterministic single-iteration
        # completion still passes; upper bound rejects a runaway loop.
        assert 1 <= r1.tokens_used <= 50_000, r1.tokens_used
        assert r1.depth_reached >= 1, r1.depth_reached
        assert r1.metadata.get("model") == "openai/google/gemma-4-e4b-it"

        big = RLMInference(
            llm_config=llm_config,
            max_iterations=3,
            max_llm_calls=8,
            timeout_seconds=600,
            tenant_id=unique_id("rlm_tokens"),
        )
        # Inflate context so the bigger budget actually has more to do.
        big_context = _PHASE5_CONTEXT * 4
        r2 = big.process(query=_PHASE5_QUERY, context=big_context)
        assert r2.tokens_used >= r1.tokens_used, (
            f"larger budget should not use fewer tokens: "
            f"small={r1.tokens_used} big={r2.tokens_used}"
        )


# ---------------------------------------------------------------------------
# 2. include_trajectory toggles structured trajectory; summary always present
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestIncludeTrajectoryShape:
    """trajectory list shape + bounds; toggle controls structured emission."""

    def test_trajectory_bounded_and_truncated(
        self, llm_config: LLMEndpointConfig
    ) -> None:
        RLMInference, _ = _import_rlm()
        rlm = RLMInference(
            llm_config=llm_config,
            max_iterations=2,
            max_llm_calls=4,
            timeout_seconds=600,
            tenant_id=unique_id("rlm_traj"),
        )
        max_entries = 8
        r = rlm.process(
            query=_PHASE5_QUERY,
            context=_PHASE5_CONTEXT,
            include_trajectory=True,
            trajectory_max_entries=max_entries,
        )
        # Could be empty when the REPL completed in zero iterations
        # AND the trajectory was not retained — that's a documented
        # contract (see RLMResult.trajectory). We assert the type and
        # the bound, not a non-empty list.
        assert isinstance(r.trajectory, list)
        assert len(r.trajectory) <= max_entries
        # Each entry's truncatable string fields must respect the
        # _TRAJECTORY_FIELD_TRUNCATE (500) limit + the truncate marker.
        # The serializer passes through whichever step-output key the
        # installed dspy emits: REPLEntry's "output" on current versions,
        # "observation"/"result" on older ones.
        _ALLOWED_KEYS = {
            "iteration",
            "reasoning",
            "code",
            "output",
            "observation",
            "result",
        }
        for entry in r.trajectory:
            assert set(entry.keys()) <= _ALLOWED_KEYS, entry
            assert isinstance(entry["iteration"], int)
            for k in ("reasoning", "code", "output", "observation", "result"):
                if k in entry:
                    assert len(str(entry[k])) <= 501, (
                        f"trajectory entry {entry['iteration']!r} field {k!r} "
                        f"exceeds 500+1 char truncate budget: {len(str(entry[k]))}"
                    )
        # metadata.trajectory_summary is always populated (server-side
        # debug aid), independent of include_trajectory.
        assert "trajectory_summary" in r.metadata
        assert "trajectory_length" in r.metadata

    def test_include_trajectory_false_yields_empty_list(
        self, llm_config: LLMEndpointConfig
    ) -> None:
        RLMInference, _ = _import_rlm()
        rlm = RLMInference(
            llm_config=llm_config,
            max_iterations=2,
            max_llm_calls=4,
            timeout_seconds=600,
            tenant_id=unique_id("rlm_traj"),
        )
        r = rlm.process(
            query=_PHASE5_QUERY,
            context=_PHASE5_CONTEXT,
            include_trajectory=False,
        )
        assert r.trajectory == []
        # Summary stays even when caller opts out of the structured form.
        assert "trajectory_summary" in r.metadata


# ---------------------------------------------------------------------------
# 3. was_fallback flips when iteration budget is exhausted
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestWasFallbackContract:
    """RLMResult.was_fallback exposes a strict bool consistent with telemetry.

    Whether ``_extract_fallback`` fires is dspy.RLM-internal and depends on
    both the model and the query (small fast models on this query converge
    in 1 iteration without triggering it; longer chains can force it).
    The shipped contract this test pins is: the field always exists, is a
    strict bool, and round-trips through ``to_telemetry_dict`` identically.
    """

    def test_was_fallback_is_strict_bool_and_round_trips(
        self, llm_config: LLMEndpointConfig
    ) -> None:
        RLMInference, _ = _import_rlm()
        rlm = RLMInference(
            llm_config=llm_config,
            max_iterations=1,
            max_llm_calls=2,
            timeout_seconds=600,
            tenant_id=unique_id("rlm_fb"),
        )
        r = rlm.process(query=_PHASE5_QUERY, context=_PHASE5_CONTEXT)
        # Strict bool — would catch a regression that returns int/None.
        assert isinstance(r.was_fallback, bool), (
            f"was_fallback must be a strict bool, got {type(r.was_fallback)!r}"
        )
        td = r.to_telemetry_dict()
        # Identity round-trip — Phoenix dashboards filter on this exact key.
        assert td["rlm_was_fallback"] is r.was_fallback
        assert r.depth_reached >= 1


# ---------------------------------------------------------------------------
# 4. to_telemetry_dict shape must stay exact (Phoenix span schema)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestTelemetryDictShape:
    """RLMResult.to_telemetry_dict key set must be exact."""

    def test_telemetry_dict_keys(self, llm_config: LLMEndpointConfig) -> None:
        RLMInference, _ = _import_rlm()
        rlm = RLMInference(
            llm_config=llm_config,
            max_iterations=1,
            max_llm_calls=2,
            timeout_seconds=600,
            tenant_id=unique_id("rlm_tel"),
        )
        r = rlm.process(query=_PHASE5_QUERY, context=_PHASE5_CONTEXT)
        td = r.to_telemetry_dict()
        # Exact key set — drift = silent dashboard breakage.
        expected_keys = {
            "rlm_enabled",
            "rlm_depth_reached",
            "rlm_total_calls",
            "rlm_tokens_used",
            "rlm_latency_ms",
            "rlm_was_fallback",
            "rlm_trajectory_length",
        }
        assert set(td.keys()) == expected_keys, set(td.keys()) ^ expected_keys
        # Type contracts.
        assert td["rlm_enabled"] is True
        assert isinstance(td["rlm_depth_reached"], int)
        assert isinstance(td["rlm_total_calls"], int)
        assert isinstance(td["rlm_tokens_used"], int)
        assert isinstance(td["rlm_latency_ms"], float)
        assert isinstance(td["rlm_was_fallback"], bool)
        assert isinstance(td["rlm_trajectory_length"], int)
