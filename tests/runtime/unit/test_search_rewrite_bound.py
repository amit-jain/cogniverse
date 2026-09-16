"""Every dispatched search bounds its query rewrite.

The rewrite is one LM round trip on the live path. When the serving endpoint is
cold the round trip takes as long as the endpoint takes - 119.7s was measured
against a cold Modal container - so a search that does not bound it waits that
long and the caller has no ceiling to rely on. The answer-grounding path bounds
it; these pin that every other dispatched search does too, at the same budget,
and that a rewrite which overruns degrades to the original query instead of
hanging.
"""

from __future__ import annotations

import asyncio
import json
import select
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import dspy
import numpy as np
import pytest
from dspy.utils.dummies import DummyLM

from cogniverse_agents.search_agent import (
    QUERY_REWRITE_BUDGET_S,
    QUERY_REWRITE_FAILED,
    QUERY_REWRITE_TIMED_OUT,
    SearchAgent,
    SearchAgentDeps,
)
from cogniverse_foundation.config.lm_deadline import (
    LMCallDeadline,
    LMCallDeadlineExceeded,
    bound_lm_call_deadline,
)
from cogniverse_foundation.config.routed_lm import UpstreamUnavailable
from cogniverse_foundation.config.semantic_router import create_routed_lm
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
)
from cogniverse_foundation.telemetry.span_contract import (
    QUERY_ENHANCEMENT_PATH_ATTRIBUTE,
    QUERY_ENHANCEMENT_PATH_HEURISTIC_FALLBACK,
    QUERY_ENHANCEMENT_PATH_LM,
)
from cogniverse_runtime.agent_dispatcher import (
    GROUNDING_SEARCH_RESERVE_S,
    GROUNDING_SEARCH_TIMEOUT_KEY,
    AgentDispatcher,
    dispatched_query_rewrite_budget_s,
)
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_SHIPPED_CONFIG = json.loads(
    (Path(__file__).resolve().parents[3] / "configs" / "config.json").read_text()
)
_SHIPPED_ACTIVE_PROFILE = _SHIPPED_CONFIG["active_video_profile"]
_SHIPPED_GROUNDING_BUDGET_S = _SHIPPED_CONFIG[GROUNDING_SEARCH_TIMEOUT_KEY]

_TENANT = "acme:acme"
_QUERY = "people exercising"
_REWRITTEN = "workout routines for people exercising video"
_HIT = {"document_id": "v_001", "score": 0.9, "metadata": {"title": "Morning workout"}}

# A synthetic budget small enough to execute the overrun inside a unit test.
# The reserve is production's, so the bound under test stays derived.
_TEST_BUDGET_S = GROUNDING_SEARCH_RESERVE_S + 0.5
_EXPECTED_BOUND_S = _TEST_BUDGET_S - GROUNDING_SEARCH_RESERVE_S

# Longer than the bound by enough that a passing test cannot be the hang
# finishing early.
_HANG_S = 5.0


class _HangingLM(DummyLM):
    """An LM whose completion takes longer than the rewrite is allowed."""

    def __call__(self, *args, **kwargs):
        time.sleep(_HANG_S)
        return super().__call__(*args, **kwargs)


def _config_get(budget_s):
    def get(key, default=None):
        if key == GROUNDING_SEARCH_TIMEOUT_KEY:
            return budget_s
        if key == "active_video_profile":
            return _SHIPPED_ACTIVE_PROFILE
        return default

    return get


def _dispatcher(budget_s=_TEST_BUDGET_S):
    from cogniverse_foundation.config.manager import ConfigManager

    store = InMemoryConfigStore()
    store.initialize()
    config_manager = ConfigManager(store=store)

    with (
        patch("cogniverse_agents.search_agent.QueryEncoderFactory"),
        patch("cogniverse_agents.search_agent.get_backend_registry"),
    ):
        agent = SearchAgent(
            deps=SearchAgentDeps(),
            schema_loader=Mock(),
            config_manager=config_manager,
        )
    agent.query_encoder = SimpleNamespace(
        encode=lambda _query: np.zeros((2, 128), dtype=np.float32)
    )
    searched: list[str] = []

    def _by_text(query, **kwargs):
        searched.append(query)
        return [dict(_HIT)]

    agent._search_by_text = _by_text

    dispatcher = AgentDispatcher(
        agent_registry=MagicMock(),
        config_manager=config_manager,
        schema_loader=MagicMock(),
    )
    dispatcher._get_search_agent = lambda profile, tenant_id: agent

    captured: list[float | None] = []
    real_process = agent.process

    async def _record(search_input):
        captured.append(search_input.query_rewrite_timeout_s)
        return await real_process(search_input)

    agent.process = _record
    return dispatcher, captured, searched, _config_get(budget_s)


class TestEveryDispatchedSearchCarriesTheBound:
    async def test_direct_dispatch_bounds_the_rewrite_at_budget_minus_reserve(self):
        """A direct search names a bound; it does not leave the rewrite open."""
        dispatcher, captured, _, config_get = _dispatcher()

        with (
            patch("cogniverse_foundation.config.utils.get_config") as get_config,
            dspy.context(lm=DummyLM([{"enhanced_query": _REWRITTEN}])),
        ):
            get_config.return_value = SimpleNamespace(get=config_get)
            await dispatcher._execute_search_task(_QUERY, _TENANT, top_k=3)

        assert captured == [_EXPECTED_BOUND_S]

    async def test_the_bound_tracks_the_configured_budget(self):
        """Derived, not hardcoded: a different budget moves the bound with it."""
        budget = GROUNDING_SEARCH_RESERVE_S + 3.25
        dispatcher, captured, _, config_get = _dispatcher(budget)

        with (
            patch("cogniverse_foundation.config.utils.get_config") as get_config,
            dspy.context(lm=DummyLM([{"enhanced_query": _REWRITTEN}])),
        ):
            get_config.return_value = SimpleNamespace(get=config_get)
            await dispatcher._execute_search_task(_QUERY, _TENANT, top_k=3)

        assert captured == [3.25]

    async def test_an_explicit_bound_is_not_overridden(self):
        """The grounded path already computes its own; it stays as passed."""
        dispatcher, captured, _, config_get = _dispatcher()

        with (
            patch("cogniverse_foundation.config.utils.get_config") as get_config,
            dspy.context(lm=DummyLM([{"enhanced_query": _REWRITTEN}])),
        ):
            get_config.return_value = SimpleNamespace(get=config_get)
            await dispatcher._execute_search_task(
                _QUERY, _TENANT, top_k=3, query_rewrite_timeout_s=1.5
            )

        assert captured == [1.5]


class TestAnOverrunningRewriteDegradesInsteadOfHanging:
    async def test_a_hung_rewrite_returns_the_original_query_within_the_bound(self):
        """The fault contract: a cold endpoint costs the bound, not its own
        latency, and the search still runs on the query the caller sent."""
        dispatcher, _, searched, config_get = _dispatcher()

        started = time.perf_counter()
        with (
            patch("cogniverse_foundation.config.utils.get_config") as get_config,
            dspy.context(lm=_HangingLM([{"enhanced_query": _REWRITTEN}])),
        ):
            get_config.return_value = SimpleNamespace(get=config_get)
            response = await dispatcher._execute_search_task(_QUERY, _TENANT, top_k=3)
        elapsed = time.perf_counter() - started

        assert response["query_rewrite"] == {
            "enhanced_query": None,
            "degraded": QUERY_REWRITE_TIMED_OUT,
        }
        assert searched == [_QUERY]
        assert response["results"] == [_HIT]
        assert elapsed < _TEST_BUDGET_S, (
            f"the rewrite was not bounded: {elapsed:.2f}s elapsed against a "
            f"{_EXPECTED_BOUND_S}s bound and a {_HANG_S}s hang"
        )

    async def test_concurrent_searches_each_get_their_own_bound(self):
        """N first-touches share the dispatcher; one slow rewrite must not
        extend another's bound or leak its degradation."""
        dispatcher, _, searched, config_get = _dispatcher()

        started = time.perf_counter()
        with (
            patch("cogniverse_foundation.config.utils.get_config") as get_config,
            dspy.context(lm=_HangingLM([{"enhanced_query": _REWRITTEN}] * 4)),
        ):
            get_config.return_value = SimpleNamespace(get=config_get)
            responses = await asyncio.gather(
                *(
                    dispatcher._execute_search_task(_QUERY, _TENANT, top_k=3)
                    for _ in range(4)
                )
            )
        elapsed = time.perf_counter() - started

        assert [r["query_rewrite"]["degraded"] for r in responses] == [
            QUERY_REWRITE_TIMED_OUT
        ] * 4
        assert searched == [_QUERY] * 4
        assert elapsed < _HANG_S, (
            f"four bounded rewrites took {elapsed:.2f}s; they serialized on one "
            f"another instead of each bounding at {_EXPECTED_BOUND_S}s"
        )


class TestTheMeasuredBudgetCapsTheGroundingRemainder:
    """The remainder of the answer budget is what the caller can afford, not
    what a rewrite costs. Held to the remainder alone the rewrite waits ten
    seconds on a deployment whose measured p95 is under two, and every one of
    those seconds is spent before the search the caller asked for starts."""

    async def test_the_budget_is_the_sum_of_its_two_measurements(self):
        from cogniverse_agents.search_agent import (
            _MEASURED_ROUTED_REWRITE_P95_S,
            _MEASURED_ROUTING_DECISION_P95_S,
        )

        assert _MEASURED_ROUTED_REWRITE_P95_S == 1.798
        assert _MEASURED_ROUTING_DECISION_P95_S == 1.720
        assert QUERY_REWRITE_BUDGET_S == 3.5

    async def test_a_generous_remainder_is_capped_at_the_measured_budget(self):
        dispatcher, captured, _, config_get = _dispatcher(
            GROUNDING_SEARCH_RESERVE_S + 60.0
        )

        with (
            patch("cogniverse_foundation.config.utils.get_config") as get_config,
            dspy.context(lm=DummyLM([{"enhanced_query": _REWRITTEN}])),
        ):
            get_config.return_value = SimpleNamespace(get=config_get)
            await dispatcher._execute_search_task(_QUERY, _TENANT, top_k=3)

        assert captured == [QUERY_REWRITE_BUDGET_S]

    async def test_a_remainder_smaller_than_the_budget_still_wins(self):
        """The caller's ceiling is never exceeded to fit the rewrite in."""
        dispatcher, captured, _, config_get = _dispatcher(
            GROUNDING_SEARCH_RESERVE_S + 1.0
        )

        with (
            patch("cogniverse_foundation.config.utils.get_config") as get_config,
            dspy.context(lm=DummyLM([{"enhanced_query": _REWRITTEN}])),
        ):
            get_config.return_value = SimpleNamespace(get=config_get)
            await dispatcher._execute_search_task(_QUERY, _TENANT, top_k=3)

        assert captured == [1.0]


def _span_recorder(name: str):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer(name), exporter


def _rewrite_path(exporter) -> list[str]:
    return [
        span.attributes[QUERY_ENHANCEMENT_PATH_ATTRIBUTE]
        for span in exporter.get_finished_spans()
        if QUERY_ENHANCEMENT_PATH_ATTRIBUTE in (span.attributes or {})
    ]


class TestTheServedRewritePathIsOnTheSpan:
    """Which path served the rewrite has to be readable off a trace. A search
    that fell back logs one warning line and returns results either way, so
    without the marker a rewrite that is degraded for every request on the
    cluster is indistinguishable from one that is working."""

    async def test_a_completed_rewrite_marks_the_lm_path(self):
        dispatcher, _, _, config_get = _dispatcher()
        tracer, exporter = _span_recorder("rewrite-lm")

        with (
            patch("cogniverse_foundation.config.utils.get_config") as get_config,
            dspy.context(lm=DummyLM([{"enhanced_query": _REWRITTEN}])),
        ):
            get_config.return_value = SimpleNamespace(get=config_get)
            with tracer.start_as_current_span("dispatch"):
                await dispatcher._execute_search_task(_QUERY, _TENANT, top_k=3)

        assert _rewrite_path(exporter) == [QUERY_ENHANCEMENT_PATH_LM]

    async def test_an_overrunning_rewrite_marks_the_heuristic_fallback(self):
        dispatcher, _, searched, config_get = _dispatcher()
        tracer, exporter = _span_recorder("rewrite-timeout")

        with (
            patch("cogniverse_foundation.config.utils.get_config") as get_config,
            dspy.context(lm=_HangingLM([{"enhanced_query": _REWRITTEN}])),
        ):
            get_config.return_value = SimpleNamespace(get=config_get)
            with tracer.start_as_current_span("dispatch"):
                await dispatcher._execute_search_task(_QUERY, _TENANT, top_k=3)

        assert _rewrite_path(exporter) == [QUERY_ENHANCEMENT_PATH_HEURISTIC_FALLBACK]
        assert searched == [_QUERY]


class TestAnUnreachableRouterDegradesTheRewrite:
    """The fault contract for the router itself. The rewrite's LM is the router
    when semantic routing is on, so the router being down is an LM that refuses
    the connection - and the search must still answer, on the original query,
    naming the router entry it could not reach."""

    async def test_a_refused_router_falls_back_and_names_the_entry(self, caplog):
        import socket

        from cogniverse_foundation.config.semantic_router import create_routed_lm
        from cogniverse_foundation.config.unified_config import (
            LLMEndpointConfig,
            SemanticRouterConfig,
        )

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            dead_port = probe.getsockname()[1]

        routed_lm = create_routed_lm(
            LLMEndpointConfig(
                model="openai/google/gemma-4-e4b-it",
                api_base="http://127.0.0.1:1/v1",
                request_timeout=2.0,
                num_retries=0,
            ),
            SemanticRouterConfig(
                enabled=True,
                semantic_router_url=f"http://127.0.0.1:{dead_port}/v1",
            ),
            _TENANT,
            "default",
            call_site="search_agent",
        )
        routed_lm.cache = False
        assert routed_lm.model == "openai/cogniverse-classification"

        dispatcher, _, searched, config_get = _dispatcher()
        tracer, exporter = _span_recorder("rewrite-router-down")

        with (
            patch("cogniverse_foundation.config.utils.get_config") as get_config,
            dspy.context(lm=routed_lm),
            caplog.at_level("WARNING", logger="cogniverse_agents.search_agent"),
        ):
            get_config.return_value = SimpleNamespace(get=config_get)
            with tracer.start_as_current_span("dispatch"):
                response = await dispatcher._execute_search_task(
                    _QUERY, _TENANT, top_k=3
                )

        assert _rewrite_path(exporter) == [QUERY_ENHANCEMENT_PATH_HEURISTIC_FALLBACK]
        assert searched == [_QUERY]
        assert response["results"] == [_HIT]
        assert response["query_rewrite"]["enhanced_query"] is None
        assert response["query_rewrite"]["degraded"] in {
            QUERY_REWRITE_FAILED,
            QUERY_REWRITE_TIMED_OUT,
        }
        rewrite_warnings = [
            record.getMessage()
            for record in caplog.records
            if record.name == "cogniverse_agents.search_agent"
            and "Query rewrite on" in record.getMessage()
        ]
        assert len(rewrite_warnings) == 1
        assert "openai/cogniverse-classification" in rewrite_warnings[0]


# The bound the tests below run under: what a dispatched search gets out of the
# shipped search budget, through the dispatcher's own derivation.
_SHIPPED_REWRITE_BOUND_S = dispatched_query_rewrite_budget_s(
    _SHIPPED_GROUNDING_BUDGET_S
)

# How far past the bound a bounded call may return, and by when the upstream
# must have seen the client hang up: the worker hand-off and the fallback's
# bookkeeping.
_RETURN_MARGIN_S = 0.5

# One retry: enough to observe whether a second attempt starts.
_ENDPOINT_RETRIES = 1

# A 408 the upstream sends three quarters of the way to the deadline, followed
# by an answer: a retry fits inside the bound.
_INSIDE_408 = [(0.75 * _SHIPPED_REWRITE_BOUND_S, 408), (0.0, 200)]

_REWRITE_MODEL = "openai/cogniverse-classification"
_REWRITE_REPLY = f"[[ ## enhanced_query ## ]]\n{_REWRITTEN}\n\n[[ ## completed ## ]]"

_CONCURRENT_REWRITES = 16
_HEARTBEAT_S = 0.01
# Longest serving-loop stall while sixteen bounded rewrites run. Measured
# 0.042-0.045s warm in a fresh interpreter, up to 0.116s warm after the rest of
# this file, 0.265-0.293s on a cold first call. An LM call made on the loop
# stalls it for the whole rewrite bound instead.
_LOOP_GAP_BOUND_S = 0.5


def _hung_up_within(connection, wait_s: float) -> bool:
    """Wait ``wait_s`` on a served connection; True once the client closes it."""
    until = time.monotonic() + wait_s
    while (left := until - time.monotonic()) > 0:
        readable, _, _ = select.select([connection], [], [], left)
        if not readable:
            return False
        try:
            if connection.recv(1, socket.MSG_PEEK) == b"":
                return True
        except OSError:
            return True
        time.sleep(left)
    return False


class _ScriptedUpstream:
    """A real chat-completions endpoint scripted per request.

    Request ``n`` waits ``script[n][0]`` seconds and then answers with status
    ``script[n][1]`` (the last entry repeats), unless its client hangs up
    first. Every arrival and every hang-up is stamped on the test's monotonic
    clock.
    """

    def __init__(self, script: list[tuple[float, int]]):
        self.arrivals: list[float] = []
        self.hangups: list[float] = []
        lock = threading.Lock()
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with lock:
                    index = len(upstream.arrivals)
                    upstream.arrivals.append(time.monotonic())
                delay_s, status = script[min(index, len(script) - 1)]
                if _hung_up_within(self.connection, delay_s):
                    with lock:
                        upstream.hangups.append(time.monotonic())
                    self.close_connection = True
                    return
                if status == 200:
                    payload = {
                        "id": "stub",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": _REWRITE_REPLY,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                else:
                    payload = {"error": {"message": f"stub status {status}"}}
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            block_on_close = False

        self._server = Server(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"


def _routed_rewrite_lm(api_base: str):
    return create_routed_lm(
        LLMEndpointConfig(
            model="openai/google/gemma-4-e4b-it",
            api_base="http://127.0.0.1:1/v1",
            api_key="stub-key",
            num_retries=_ENDPOINT_RETRIES,
        ),
        SemanticRouterConfig(enabled=True, semantic_router_url=api_base),
        _TENANT,
        "default",
        call_site="search_agent",
    )


def _unique(label: str) -> str:
    return f"{_QUERY} {label} {uuid.uuid4().hex[:8]}"


def _dead_api_base() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{probe.getsockname()[1]}/v1"


@contextmanager
def _shipped_dispatch(lm):
    dispatcher, captured, searched, config_get = _dispatcher(
        _SHIPPED_GROUNDING_BUDGET_S
    )
    with (
        patch("cogniverse_foundation.config.utils.get_config") as get_config,
        dspy.context(lm=lm),
    ):
        get_config.return_value = SimpleNamespace(get=config_get)
        yield dispatcher, captured, searched


def _timed_out():
    return {"enhanced_query": None, "degraded": QUERY_REWRITE_TIMED_OUT}


class TestTheRewriteDeadlineReachesTheUpstream:
    """The rewrite's bound travels with its LM call onto the wire: the request
    carries the time that is left as its timeout, no attempt starts once the
    bound has passed or the search stopped waiting, and nothing is still
    talking to the upstream after the search fell back to the original query."""

    async def test_a_prompt_upstream_serves_the_rewrite_in_one_request(self):
        query = _unique("prompt")
        with _ScriptedUpstream([(0.0, 200)]) as upstream:
            with _shipped_dispatch(_routed_rewrite_lm(upstream.api_base)) as (
                dispatcher,
                captured,
                searched,
            ):
                response = await dispatcher._execute_search_task(
                    query, _TENANT, top_k=3
                )

        assert captured == [_SHIPPED_REWRITE_BOUND_S]
        assert response["query_rewrite"] == {
            "enhanced_query": _REWRITTEN,
            "degraded": None,
        }
        assert searched == [_REWRITTEN]
        assert len(upstream.arrivals) == 1

    async def test_a_hung_upstream_gets_one_request_and_is_hung_up_on_at_the_deadline(
        self,
    ):
        query = _unique("hung")
        with _ScriptedUpstream([(3 * _SHIPPED_REWRITE_BOUND_S, 200)]) as upstream:
            with _shipped_dispatch(_routed_rewrite_lm(upstream.api_base)) as (
                dispatcher,
                _,
                searched,
            ):
                started = time.monotonic()
                response = await dispatcher._execute_search_task(
                    query, _TENANT, top_k=3
                )
                returned = time.monotonic()
                await asyncio.sleep(_SHIPPED_REWRITE_BOUND_S)

        deadline = started + _SHIPPED_REWRITE_BOUND_S
        assert response["query_rewrite"] == _timed_out()
        assert searched == [query]
        assert returned - started < _SHIPPED_REWRITE_BOUND_S + _RETURN_MARGIN_S
        assert len(upstream.arrivals) == 1
        assert upstream.arrivals[0] < deadline
        assert len(upstream.hangups) == 1
        assert upstream.hangups[0] < deadline + _RETURN_MARGIN_S

    async def test_a_408_arriving_after_the_deadline_is_never_retried(self):
        query = _unique("late-408")
        late_s = _SHIPPED_REWRITE_BOUND_S + _RETURN_MARGIN_S
        with _ScriptedUpstream([(late_s, 408), (0.0, 200)]) as upstream:
            with _shipped_dispatch(_routed_rewrite_lm(upstream.api_base)) as (
                dispatcher,
                _,
                _,
            ):
                started = time.monotonic()
                response = await dispatcher._execute_search_task(
                    query, _TENANT, top_k=3
                )
                await asyncio.sleep(
                    started + late_s + _SHIPPED_REWRITE_BOUND_S - time.monotonic()
                )

        assert response["query_rewrite"] == _timed_out()
        assert len(upstream.arrivals) == 1
        assert upstream.arrivals[0] < started + _SHIPPED_REWRITE_BOUND_S

    async def test_a_408_inside_the_deadline_is_retried_once(self):
        query = _unique("inside-408")
        with _ScriptedUpstream(_INSIDE_408) as upstream:
            with _shipped_dispatch(_routed_rewrite_lm(upstream.api_base)) as (
                dispatcher,
                _,
                searched,
            ):
                started = time.monotonic()
                response = await dispatcher._execute_search_task(
                    query, _TENANT, top_k=3
                )
                await asyncio.sleep(_SHIPPED_REWRITE_BOUND_S)

        assert response["query_rewrite"] == {
            "enhanced_query": _REWRITTEN,
            "degraded": None,
        }
        assert searched == [_REWRITTEN]
        assert len(upstream.arrivals) == 2
        assert upstream.arrivals[1] < started + _SHIPPED_REWRITE_BOUND_S

    async def test_a_search_cancelled_mid_call_starts_no_further_request(self):
        query = _unique("cancelled")
        with _ScriptedUpstream(_INSIDE_408) as upstream:
            with _shipped_dispatch(_routed_rewrite_lm(upstream.api_base)) as (
                dispatcher,
                _,
                _,
            ):
                task = asyncio.create_task(
                    dispatcher._execute_search_task(query, _TENANT, top_k=3)
                )
                await asyncio.sleep(_SHIPPED_REWRITE_BOUND_S / 2)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                cancelled_at = time.monotonic()
                await asyncio.sleep(_SHIPPED_REWRITE_BOUND_S + _RETURN_MARGIN_S)

        assert len(upstream.arrivals) == 1
        assert upstream.arrivals[0] < cancelled_at

    async def test_sixteen_rewrites_released_together_leave_no_request_after_the_deadline(
        self,
    ):
        """Sixteen simultaneous connects overrun the stub's listen backlog, so
        TCP retries some of them a second later: those requests reach the
        wire late, and still end at their deadline."""
        queries = [_unique(f"concurrent-{i}") for i in range(_CONCURRENT_REWRITES)]
        barrier = asyncio.Barrier(_CONCURRENT_REWRITES)
        released: list[float] = []
        gaps: list[float] = []
        stop = asyncio.Event()

        async def heartbeat():
            last = time.monotonic()
            while not stop.is_set():
                await asyncio.sleep(_HEARTBEAT_S)
                now = time.monotonic()
                gaps.append(now - last - _HEARTBEAT_S)
                last = now

        with _ScriptedUpstream([(3 * _SHIPPED_REWRITE_BOUND_S, 200)]) as upstream:
            with _shipped_dispatch(_routed_rewrite_lm(upstream.api_base)) as (
                dispatcher,
                _,
                searched,
            ):

                async def one(query):
                    await barrier.wait()
                    released.append(time.monotonic())
                    return await dispatcher._execute_search_task(
                        query, _TENANT, top_k=3
                    )

                warmed = await dispatcher._execute_search_task(
                    _unique("concurrent-warm-up"), _TENANT, top_k=3
                )
                warm_hung_up_by = time.monotonic() + _RETURN_MARGIN_S
                while len(upstream.hangups) == 0 and time.monotonic() < warm_hung_up_by:
                    await asyncio.sleep(_HEARTBEAT_S)
                warm_arrivals = len(upstream.arrivals)
                warm_hangups = len(upstream.hangups)
                beat = asyncio.create_task(heartbeat())
                responses = await asyncio.gather(*(one(q) for q in queries))
                returned = time.monotonic()
                await asyncio.sleep(_SHIPPED_REWRITE_BOUND_S)
                stop.set()
                await beat

        assert warmed["query_rewrite"] == _timed_out()
        assert (warm_arrivals, warm_hangups) == (1, 1)
        released_arrivals = upstream.arrivals[warm_arrivals:]
        released_hangups = upstream.hangups[warm_hangups:]
        assert [r["query_rewrite"] for r in responses] == [
            _timed_out()
        ] * _CONCURRENT_REWRITES
        assert sorted(searched[1:]) == sorted(queries)
        assert returned - min(released) < _SHIPPED_REWRITE_BOUND_S + _RETURN_MARGIN_S
        assert len(released_arrivals) == _CONCURRENT_REWRITES
        assert max(released_arrivals) < min(released) + _SHIPPED_REWRITE_BOUND_S
        assert len(released_hangups) == _CONCURRENT_REWRITES
        assert (
            max(released_hangups)
            < max(released) + _SHIPPED_REWRITE_BOUND_S + _RETURN_MARGIN_S
        )
        assert max(gaps) < _LOOP_GAP_BOUND_S, (
            f"the serving loop stalled {max(gaps):.3f}s under "
            f"{_CONCURRENT_REWRITES} bounded rewrites"
        )


class TestABoundedCallThatCannotFinishNamesItsEndpointAndDeadline:
    """The fault contract of a bounded LM call: an upstream that hangs or is
    not there raises an error naming the endpoint and the deadline."""

    async def test_a_prompt_upstream_answers_the_bounded_call(self):
        agent = _dispatcher()[0]._get_search_agent(_SHIPPED_ACTIVE_PROFILE, _TENANT)
        with _ScriptedUpstream([(0.0, 200)]) as upstream:
            with dspy.context(lm=_routed_rewrite_lm(upstream.api_base)):
                prediction = await agent.call_dspy(
                    agent.search_module,
                    output_field="enhanced_query",
                    deadline=LMCallDeadline.after(_SHIPPED_REWRITE_BOUND_S),
                    query=_unique("bounded"),
                    modality="video",
                    top_k=3,
                )

        assert prediction.enhanced_query == _REWRITTEN
        assert len(upstream.arrivals) == 1

    async def test_a_hung_upstream_raises_naming_the_endpoint_and_the_deadline(self):
        agent = _dispatcher()[0]._get_search_agent(_SHIPPED_ACTIVE_PROFILE, _TENANT)
        with _ScriptedUpstream([(3 * _SHIPPED_REWRITE_BOUND_S, 200)]) as upstream:
            started = time.monotonic()
            with (
                dspy.context(lm=_routed_rewrite_lm(upstream.api_base)),
                pytest.raises(LMCallDeadlineExceeded) as raised,
            ):
                await agent.call_dspy(
                    agent.search_module,
                    output_field="enhanced_query",
                    deadline=LMCallDeadline.after(_SHIPPED_REWRITE_BOUND_S),
                    query=_unique("bounded-hung"),
                    modality="video",
                    top_k=3,
                )
            elapsed = time.monotonic() - started

        assert str(raised.value) == (
            f"LM call to {upstream.api_base} for {_REWRITE_MODEL} stopped, its "
            f"deadline passed: deadline {_SHIPPED_REWRITE_BOUND_S:.2f}s"
        )
        assert elapsed < _SHIPPED_REWRITE_BOUND_S + _RETURN_MARGIN_S
        assert len(upstream.arrivals) == 1

    async def test_an_unreachable_upstream_raises_naming_the_endpoint_and_the_deadline(
        self,
    ):
        agent = _dispatcher()[0]._get_search_agent(_SHIPPED_ACTIVE_PROFILE, _TENANT)
        dead = _dead_api_base()
        started = time.monotonic()
        with (
            dspy.context(lm=_routed_rewrite_lm(dead)),
            pytest.raises(UpstreamUnavailable) as raised,
        ):
            await agent.call_dspy(
                agent.search_module,
                output_field="enhanced_query",
                deadline=LMCallDeadline.after(_SHIPPED_REWRITE_BOUND_S),
                query=_unique("bounded-dead"),
                modality="video",
                top_k=3,
            )
        elapsed = time.monotonic() - started

        assert str(raised.value) == (
            f"the model endpoint did not answer: tenant={_TENANT} tier=default "
            f"routed_model={_REWRITE_MODEL} status=500 router_code=None "
            f"endpoint={dead} deadline_s={_SHIPPED_REWRITE_BOUND_S:.2f}"
        )
        assert elapsed < _SHIPPED_REWRITE_BOUND_S

    async def test_an_lm_that_ignores_the_deadline_still_releases_the_caller_at_it(
        self,
    ):
        """An LM with no endpoint to bound hangs on its own; the caller waiting
        on it is released at the deadline all the same."""
        agent = _dispatcher()[0]._get_search_agent(_SHIPPED_ACTIVE_PROFILE, _TENANT)
        lm = _HangingLM([{"enhanced_query": _REWRITTEN}])
        started = time.monotonic()
        with dspy.context(lm=lm), pytest.raises(LMCallDeadlineExceeded) as raised:
            await agent.call_dspy(
                agent.search_module,
                output_field="enhanced_query",
                deadline=LMCallDeadline.after(_SHIPPED_REWRITE_BOUND_S),
                query=_unique("endpointless"),
                modality="video",
                top_k=3,
            )
        elapsed = time.monotonic() - started

        assert str(raised.value) == (
            f"LM call for {lm.model} stopped, its deadline passed: "
            f"deadline {_SHIPPED_REWRITE_BOUND_S:.2f}s"
        )
        assert elapsed < _SHIPPED_REWRITE_BOUND_S + _RETURN_MARGIN_S

    async def test_a_caller_joining_an_identical_in_flight_call_stops_at_its_own_deadline(
        self,
    ):
        """Two callers of one tenant's identical request share one upstream
        call; the one with the nearer deadline stops at it instead of waiting
        on the other's."""
        messages = [{"role": "user", "content": _unique("joined")}]
        outcomes: dict[str, tuple[object, float]] = {}

        def call(lm, name: str, budget_s: float) -> None:
            started = time.monotonic()
            try:
                with bound_lm_call_deadline(LMCallDeadline.after(budget_s)):
                    lm.forward(messages=messages)
                outcomes[name] = ("returned", time.monotonic() - started)
            except BaseException as exc:  # noqa: BLE001 - the outcome under test
                outcomes[name] = (exc, time.monotonic() - started)

        with _ScriptedUpstream([(3 * _SHIPPED_REWRITE_BOUND_S, 200)]) as upstream:
            lm = _routed_rewrite_lm(upstream.api_base)
            owner = threading.Thread(
                target=call, args=(lm, "owner", 2 * _SHIPPED_REWRITE_BOUND_S)
            )
            owner.start()
            owner_sent_by = time.monotonic() + _SHIPPED_REWRITE_BOUND_S
            while len(upstream.arrivals) == 0 and time.monotonic() < owner_sent_by:
                await asyncio.sleep(_HEARTBEAT_S)
            assert len(upstream.arrivals) == 1
            await asyncio.to_thread(call, lm, "joined", _SHIPPED_REWRITE_BOUND_S)
            arrivals_when_joined_stopped = len(upstream.arrivals)
            await asyncio.to_thread(owner.join)

        joined, joined_s = outcomes["joined"]
        owner_outcome, _ = outcomes["owner"]
        assert type(joined) is LMCallDeadlineExceeded
        assert str(joined) == (
            f"LM call to {upstream.api_base} for {_REWRITE_MODEL} stopped, its "
            f"deadline passed: deadline {_SHIPPED_REWRITE_BOUND_S:.2f}s"
        )
        assert joined_s < _SHIPPED_REWRITE_BOUND_S + _RETURN_MARGIN_S
        assert type(owner_outcome) is LMCallDeadlineExceeded
        assert str(owner_outcome) == (
            f"LM call to {upstream.api_base} for {_REWRITE_MODEL} stopped, its "
            f"deadline passed: deadline {2 * _SHIPPED_REWRITE_BOUND_S:.2f}s"
        )
        assert arrivals_when_joined_stopped == 1
        assert len(upstream.arrivals) == 1
