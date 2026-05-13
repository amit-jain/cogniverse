"""Phase 5b — RLM A/B harness end-to-end.

Pins the shipped RLMABRunner output:

  * one ``ab_id`` (uuid4 hex, len 32) is stamped on both arms;
  * arm strings are exactly "without_rlm" and "with_rlm";
  * comparison deltas are pure arithmetic over the per-arm fields;
  * a deterministic judge function lands its scores on both arms and
    is reflected in comparison.judge_delta;
  * ``ABResult.to_telemetry_dict`` key set is exact.

Reuses the ``vllm_student_url`` / ``llm_config`` module fixtures from
``test_rlm_telemetry_e2e.py``; both files are collected together by
the e2e batched runner so the port-forward starts once per module.
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
from tests.e2e.conftest import skip_if_no_runtime


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def ab_vllm_student_url() -> Iterator[str]:
    """Module-scoped port-forward — independent from the telemetry module."""
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
                f"on {base} within 30s — RLM A/B tests can't run"
            )
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def ab_llm_config(ab_vllm_student_url: str) -> LLMEndpointConfig:
    return LLMEndpointConfig(
        model="openai/google/gemma-4-e4b-it",
        api_base=ab_vllm_student_url,
        api_key="not-required",
        temperature=0.1,
        max_tokens=512,
    )


_AB_QUERY = "Count how many planets are mentioned and list them by name."
_AB_CONTEXT = (
    "Our solar system has eight planets: Mercury, Venus, Earth, Mars, "
    "Jupiter, Saturn, Uranus, Neptune. Pluto was reclassified as a "
    "dwarf planet in 2006."
)


def _import_ab() -> tuple:
    from cogniverse_agents.inference.ab_harness import RLMABRunner

    return (RLMABRunner,)


# ---------------------------------------------------------------------------
# 1. ab_id length + shared identity across arms; arm strings exact
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestABRunReturnsBothArmsWithSharedAbId:
    """Both arm metadata blocks carry the same ab_id; arm strings pinned."""

    def test_ab_id_shape_and_arm_strings(
        self, ab_llm_config: LLMEndpointConfig
    ) -> None:
        (RLMABRunner,) = _import_ab()
        runner = RLMABRunner(
            llm_config=ab_llm_config,
            timeout_seconds=900,
            rlm_max_iterations=2,
            rlm_max_llm_calls=4,
        )
        result = runner.run(query=_AB_QUERY, context=_AB_CONTEXT)

        # uuid4 hex (no dashes) is exactly 32 hex chars.
        assert len(result.ab_id) == 32 and all(
            c in "0123456789abcdef" for c in result.ab_id
        )
        # Same ab_id stamped on both arm metadata blobs.
        assert result.without_rlm.metadata["ab_id"] == result.ab_id
        assert result.with_rlm.metadata["ab_id"] == result.ab_id
        # Arm strings are the canonical contract — UI / Phoenix queries
        # filter on these exact values.
        assert result.without_rlm.arm == "without_rlm"
        assert result.with_rlm.arm == "with_rlm"
        # context_size_chars is pinned to the input length, not approx.
        assert result.context_size_chars == len(_AB_CONTEXT)


# ---------------------------------------------------------------------------
# 2. Comparison deltas are pure arithmetic over the per-arm fields
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestABComparisonArithmeticIsPure:
    """comparison.tokens_delta / latency_delta_ms equal the per-arm subtraction."""

    def test_deltas_match_per_arm_subtraction(
        self, ab_llm_config: LLMEndpointConfig
    ) -> None:
        (RLMABRunner,) = _import_ab()
        runner = RLMABRunner(
            llm_config=ab_llm_config,
            timeout_seconds=900,
            rlm_max_iterations=2,
            rlm_max_llm_calls=4,
        )
        result = runner.run(query=_AB_QUERY, context=_AB_CONTEXT)

        # tokens_delta = with_rlm - without_rlm, exact integer.
        assert (
            result.comparison.tokens_delta
            == result.with_rlm.tokens_used - result.without_rlm.tokens_used
        )
        # latency_delta_ms = with - without (pytest.approx for float).
        assert result.comparison.latency_delta_ms == pytest.approx(
            result.with_rlm.latency_ms - result.without_rlm.latency_ms, rel=1e-9
        )
        # judge_delta is None when no judge supplied (the case here).
        assert result.comparison.judge_delta is None
        # rlm_was_fallback mirrors the with-RLM arm.
        assert result.comparison.rlm_was_fallback is result.with_rlm.was_fallback


# ---------------------------------------------------------------------------
# 3. Deterministic judge → judge_score on each arm + judge_delta arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestJudgeFnIntegration:
    """A pure-function judge lands its scores on both arms verbatim."""

    def test_deterministic_judge_threaded_through(
        self, ab_llm_config: LLMEndpointConfig
    ) -> None:
        (RLMABRunner,) = _import_ab()

        # Deterministic in {0.5, 0.9} so we can pin membership exactly.
        def _judge(query: str, context: str, answer: str) -> float:
            return 0.5 if "error" in answer.lower() else 0.9

        runner = RLMABRunner(
            llm_config=ab_llm_config,
            judge=_judge,
            timeout_seconds=900,
            rlm_max_iterations=2,
            rlm_max_llm_calls=4,
        )
        result = runner.run(query=_AB_QUERY, context=_AB_CONTEXT)

        assert result.without_rlm.judge_score in {0.5, 0.9}, (
            result.without_rlm.judge_score
        )
        assert result.with_rlm.judge_score in {0.5, 0.9}, result.with_rlm.judge_score
        # judge_delta is pure subtraction once both arms have scores.
        assert result.comparison.judge_delta == pytest.approx(
            result.with_rlm.judge_score - result.without_rlm.judge_score, rel=1e-9
        )


# ---------------------------------------------------------------------------
# 4. to_telemetry_dict shape is exact (Phoenix span schema)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestABTelemetryDictShape:
    """ABResult.to_telemetry_dict key set is the Phoenix span contract."""

    def test_telemetry_dict_keys(self, ab_llm_config: LLMEndpointConfig) -> None:
        (RLMABRunner,) = _import_ab()
        runner = RLMABRunner(
            llm_config=ab_llm_config,
            timeout_seconds=900,
            rlm_max_iterations=1,
            rlm_max_llm_calls=2,
        )
        result = runner.run(query=_AB_QUERY, context=_AB_CONTEXT)
        td = result.to_telemetry_dict()

        expected_keys = {
            "ab_id",
            "ab_query",
            "ab_context_chars",
            "ab_without_rlm_latency_ms",
            "ab_without_rlm_tokens",
            "ab_without_rlm_judge",
            "ab_with_rlm_latency_ms",
            "ab_with_rlm_tokens",
            "ab_with_rlm_judge",
            "ab_with_rlm_was_fallback",
            "ab_latency_delta_ms",
            "ab_tokens_delta",
            "ab_judge_delta",
        }
        assert set(td.keys()) == expected_keys, set(td.keys()) ^ expected_keys
        # ab_query is truncated to 120 chars.
        assert len(td["ab_query"]) <= 120
        assert td["ab_id"] == result.ab_id
        assert td["ab_context_chars"] == len(_AB_CONTEXT)
        assert td["ab_without_rlm_tokens"] == result.without_rlm.tokens_used
        assert td["ab_with_rlm_tokens"] == result.with_rlm.tokens_used
        assert td["ab_tokens_delta"] == result.comparison.tokens_delta
