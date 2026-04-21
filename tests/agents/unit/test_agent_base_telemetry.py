"""Unit tests for AgentBase telemetry span wrapping (Audit fix #10).

Verifies that AgentBase wraps every ``_process_impl`` invocation in a
``telemetry_manager.span(...)`` context manager when a manager is attached,
and silently no-ops when none is set. This is the infrastructure fix that
makes SearchAgent, CodingAgent, TextAnalysisAgent, SummarizerAgent, and
DetailedReportAgent observable without each one having to roll its own
telemetry plumbing.
"""

from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import pytest

from cogniverse_core.agents.base import (
    AgentBase,
    AgentDeps,
    AgentInput,
    AgentOutput,
)


class _SpyTelemetryManager:
    """In-memory stand-in for TelemetryManager.

    Records every call to ``span()`` so tests can assert the wrap happened
    with the right name and tenant_id, without spinning up a real Phoenix
    backend.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    @contextmanager
    def span(
        self,
        name: str,
        tenant_id: str,
        project_name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        self.calls.append(
            {
                "name": name,
                "tenant_id": tenant_id,
                "project_name": project_name,
                "attributes": attributes,
            }
        )
        yield self


class _TelemetryInput(AgentInput):
    query: str
    tenant_id: Optional[str] = None


class _TelemetryOutput(AgentOutput):
    result: str


class _TelemetryDeps(AgentDeps):
    pass


class _TelemetryAgent(
    AgentBase[_TelemetryInput, _TelemetryOutput, _TelemetryDeps]
):
    async def _process_impl(self, input: _TelemetryInput) -> _TelemetryOutput:
        return _TelemetryOutput(result=f"processed: {input.query}")


@pytest.mark.unit
@pytest.mark.ci_fast
class TestAgentBaseTelemetrySpan:
    @pytest.mark.asyncio
    async def test_no_telemetry_manager_no_span_no_error(self):
        """Without a telemetry manager, process() must succeed and emit nothing."""
        agent = _TelemetryAgent(deps=_TelemetryDeps())
        assert agent.telemetry_manager is None

        result = await agent.process(_TelemetryInput(query="hi", tenant_id="t1"))
        assert result.result == "processed: hi"

    @pytest.mark.asyncio
    async def test_process_emits_span_with_class_name(self):
        """When a manager is attached, process() wraps _process_impl in a span
        named ``f"{ClassName}.process"`` and forwards the request tenant_id."""
        spy = _SpyTelemetryManager()
        agent = _TelemetryAgent(deps=_TelemetryDeps())
        agent.set_telemetry_manager(spy)

        await agent.process(_TelemetryInput(query="hello", tenant_id="acme"))

        assert len(spy.calls) == 1
        assert spy.calls[0]["name"] == "_TelemetryAgent.process"
        assert spy.calls[0]["tenant_id"] == "acme"

    @pytest.mark.asyncio
    async def test_run_emits_span_too(self):
        """The dict-based ``run()`` entry point must wrap the same way as
        ``process()`` so REST callers and direct callers get equivalent
        observability."""
        spy = _SpyTelemetryManager()
        agent = _TelemetryAgent(deps=_TelemetryDeps())
        agent.set_telemetry_manager(spy)

        await agent.run({"query": "hello", "tenant_id": "acme"})

        assert len(spy.calls) == 1
        assert spy.calls[0]["name"] == "_TelemetryAgent.process"
        assert spy.calls[0]["tenant_id"] == "acme"

    @pytest.mark.asyncio
    async def test_streaming_emits_span(self):
        """Streaming path (used by web UI) must also wrap _process_impl."""
        spy = _SpyTelemetryManager()
        agent = _TelemetryAgent(deps=_TelemetryDeps())
        agent.set_telemetry_manager(spy)

        events = []
        async for event in await agent.process(
            _TelemetryInput(query="hi", tenant_id="acme"), stream=True
        ):
            events.append(event)

        assert len(spy.calls) == 1
        assert spy.calls[0]["name"] == "_TelemetryAgent.process"
        assert spy.calls[0]["tenant_id"] == "acme"
        # Final event should still arrive after the span closes.
        assert events[-1]["type"] == "final"
        assert events[-1]["data"]["result"] == "processed: hi"

    @pytest.mark.asyncio
    async def test_missing_tenant_id_raises(self):
        """AgentBase MUST refuse to emit a span for an input that has no
        tenant_id.  The old silent ``or "default"`` fallback hid every
        missing-tenant plumbing bug in the runtime; now the ValueError
        surfaces to the dispatcher and the request returns 400."""
        spy = _SpyTelemetryManager()
        agent = _TelemetryAgent(deps=_TelemetryDeps())
        agent.set_telemetry_manager(spy)

        with pytest.raises(ValueError, match="tenant_id is required"):
            await agent.process(_TelemetryInput(query="hi"))

        assert spy.calls == [], "no span should have been emitted"

    @pytest.mark.asyncio
    async def test_subclass_set_telemetry_manager_pre_init_is_preserved(self):
        """Subclasses (e.g. RoutingAgent) initialize ``telemetry_manager`` BEFORE
        calling ``super().__init__()``. AgentBase must NOT clobber that value
        with its auto-init from deps."""
        spy = _SpyTelemetryManager()

        class _PreInitAgent(_TelemetryAgent):
            def __init__(self, deps: _TelemetryDeps, sentinel: Any) -> None:
                self.telemetry_manager = sentinel
                super().__init__(deps=deps)

        agent = _PreInitAgent(deps=_TelemetryDeps(), sentinel=spy)
        assert agent.telemetry_manager is spy

        await agent.process(_TelemetryInput(query="hi", tenant_id="acme"))
        assert len(spy.calls) == 1

    @pytest.mark.asyncio
    async def test_auto_init_from_deps_disabled_config(self):
        """When deps carries a ``telemetry_config`` with ``enabled=False``,
        AgentBase must skip auto-init and leave ``telemetry_manager`` as
        ``None``. The wrap should then no-op cleanly."""

        class _DisabledConfig:
            enabled = False

        deps = _TelemetryDeps()
        # Pydantic's extra="allow" lets us attach the config post-construction.
        deps.telemetry_config = _DisabledConfig()
        agent = _TelemetryAgent(deps=deps)
        assert agent.telemetry_manager is None

        result = await agent.process(
            _TelemetryInput(query="hi", tenant_id="acme")
        )
        assert result.result == "processed: hi"

    @pytest.mark.asyncio
    async def test_span_does_not_swallow_exceptions(self):
        """If _process_impl raises, the exception must propagate out of the
        span context, not get silently absorbed by the wrap."""

        class _BoomAgent(
            AgentBase[_TelemetryInput, _TelemetryOutput, _TelemetryDeps]
        ):
            async def _process_impl(
                self, input: _TelemetryInput
            ) -> _TelemetryOutput:
                raise RuntimeError("kaboom")

        spy = _SpyTelemetryManager()
        agent = _BoomAgent(deps=_TelemetryDeps())
        agent.set_telemetry_manager(spy)

        with pytest.raises(RuntimeError, match="kaboom"):
            await agent.process(_TelemetryInput(query="hi", tenant_id="acme"))

        # Span was still opened — observability captured the failed call.
        assert len(spy.calls) == 1
        assert spy.calls[0]["name"] == "_BoomAgent.process"
