"""Round-trip integration test for AgentBase telemetry spans + SPAN_NAME_BY_AGENT.

Audit fixes #2 + #10 — before #10, 5 of 6 main agents emitted ZERO spans
during processing because AgentBase.process() never wrapped _process_impl
in a telemetry span. Before #2, the QualityMonitor's lookup table had
hardcoded names (search_service.search, summarizer_agent.process, etc.)
that didn't match what agents actually emitted, so live-traffic eval
queried Phoenix for span names that didn't exist.

This test exercises the full chain end-to-end against real Phoenix:
1. Construct an agent inheriting AgentBase
2. Inject a real PhoenixProvider-backed TelemetryManager
3. Call agent.process(input)
4. Query real Phoenix for spans with name f"{ClassName}.process"
5. Verify the span was actually exported and queryable

If the SPAN_NAME_BY_AGENT lookup ever drifts from what AgentBase emits,
this test catches it.
"""

import asyncio
import time

import pytest

from cogniverse_core.agents.base import (
    AgentBase,
    AgentDeps,
    AgentInput,
    AgentOutput,
)
from cogniverse_evaluation.quality_monitor import SPAN_NAME_BY_AGENT, AgentType


class _TelemetryTestInput(AgentInput):
    query: str
    tenant_id: str = "telemetry_real_test"


class _TelemetryTestOutput(AgentOutput):
    result: str


class _TelemetryTestDeps(AgentDeps):
    pass


# Subclasses named to match real agents — the names matter because
# AgentBase emits spans as f"{ClassName}.process" and SPAN_NAME_BY_AGENT
# expects exactly those names.
class SearchAgent(
    AgentBase[_TelemetryTestInput, _TelemetryTestOutput, _TelemetryTestDeps]
):
    async def _process_impl(self, input):
        return _TelemetryTestOutput(result=f"searched: {input.query}")


class SummarizerAgent(
    AgentBase[_TelemetryTestInput, _TelemetryTestOutput, _TelemetryTestDeps]
):
    async def _process_impl(self, input):
        return _TelemetryTestOutput(result=f"summarized: {input.query}")


class DetailedReportAgent(
    AgentBase[_TelemetryTestInput, _TelemetryTestOutput, _TelemetryTestDeps]
):
    async def _process_impl(self, input):
        return _TelemetryTestOutput(result=f"reported: {input.query}")


class RoutingAgent(
    AgentBase[_TelemetryTestInput, _TelemetryTestOutput, _TelemetryTestDeps]
):
    async def _process_impl(self, input):
        return _TelemetryTestOutput(result=f"routed: {input.query}")


def _query_phoenix_for_span(
    span_name: str,
    project_name: str,
    phoenix_http_url: str,
    max_wait: int = 30,
):
    """Query the real Phoenix instance for spans with the given name.

    Returns the matched span row (or None) by polling for up to max_wait
    seconds — Phoenix has a small ingestion delay even with sync export.

    phoenix_http_url comes from real_telemetry.config.provider_config["http_endpoint"]
    so it's never hardcoded here.
    """
    from phoenix.client import Client

    client = Client(base_url=phoenix_http_url)

    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            spans_df = client.spans.get_spans_dataframe(
                project_identifier=project_name,
                limit=200,
            )
            if spans_df is not None and not spans_df.empty:
                matches = spans_df[spans_df["name"] == span_name]
                if not matches.empty:
                    return matches.iloc[0]
        except Exception:
            pass
        time.sleep(1)
    return None


@pytest.mark.integration
class TestAgentTelemetrySpansRealPhoenix:
    @pytest.mark.asyncio
    async def test_agent_emits_span_observable_in_real_phoenix(
        self, real_telemetry
    ):
        """Calling agent.process() must emit a span that's queryable via
        the real Phoenix HTTP API. Before fix #10, this would fail because
        AgentBase didn't wrap _process_impl in a span at all."""
        agent = SearchAgent(deps=_TelemetryTestDeps())
        agent.set_telemetry_manager(real_telemetry)

        await agent.process(
            _TelemetryTestInput(
                query="test span emission", tenant_id="telemetry_real_test"
            )
        )

        # Force span export — sync export should already have flushed,
        # but allow Phoenix a moment to ingest.
        await asyncio.sleep(2)

        project_name = real_telemetry.config.get_project_name(
            "telemetry_real_test"
        )
        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]
        span = _query_phoenix_for_span("SearchAgent.process", project_name, phoenix_url)

        assert span is not None, (
            f"SearchAgent.process span not found in Phoenix project "
            f"{project_name}. Either AgentBase isn't wrapping _process_impl "
            f"in a span (audit fix #10 regressed) or the span export pipeline "
            f"is broken."
        )

    @pytest.mark.asyncio
    async def test_span_name_matches_lookup_table_for_each_agent_type(
        self, real_telemetry
    ):
        """For every AgentType in SPAN_NAME_BY_AGENT, instantiate the
        matching agent class, emit a span, and verify Phoenix returns
        a match for the lookup table's name. This is the integration
        test that would have caught audit fix #2 (wrong span names)
        AND fix #10 (no spans emitted) at the same time."""
        agent_classes = {
            AgentType.SEARCH: SearchAgent,
            AgentType.SUMMARY: SummarizerAgent,
            AgentType.REPORT: DetailedReportAgent,
            AgentType.ROUTING: RoutingAgent,
        }

        project_name = real_telemetry.config.get_project_name(
            "telemetry_real_test"
        )

        for agent_type, expected_span_name in SPAN_NAME_BY_AGENT.items():
            agent_cls = agent_classes[agent_type]
            agent = agent_cls(deps=_TelemetryTestDeps())
            agent.set_telemetry_manager(real_telemetry)

            await agent.process(
                _TelemetryTestInput(
                    query=f"test {agent_type.value}",
                    tenant_id="telemetry_real_test",
                )
            )

        await asyncio.sleep(3)

        phoenix_url = real_telemetry.config.provider_config["http_endpoint"]
        for agent_type, expected_span_name in SPAN_NAME_BY_AGENT.items():
            span = _query_phoenix_for_span(
                expected_span_name, project_name, phoenix_url, max_wait=15
            )
            assert span is not None, (
                f"SPAN_NAME_BY_AGENT[{agent_type.value}] = "
                f"{expected_span_name!r} but Phoenix returned no spans "
                f"with that name. The lookup table is out of sync with "
                f"what AgentBase actually emits."
            )
