"""The orchestrator honors each agent's configured call timeout.

AgentEndpoint.timeout is loaded from config, but the orchestrator's HTTP calls
to sub-agents ignored it and used a hardcoded 240s client default. So a
per-agent timeout override was dead, and the field's own 30s default (had it
been honored) would have cut off slow agents. These pin: the default now
matches the caller's timeout, and the call passes the endpoint's timeout.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cogniverse_core.common.agent_models import (
    DEFAULT_AGENT_CALL_TIMEOUT_SECONDS,
    AgentEndpoint,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def test_agent_endpoint_default_timeout_matches_agent_call():
    ep = AgentEndpoint(name="a", url="http://x", capabilities=[])
    assert ep.timeout == DEFAULT_AGENT_CALL_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_orchestrator_kg_call_uses_endpoint_timeout():
    from cogniverse_agents.orchestrator_agent import OrchestratorAgent

    endpoint = AgentEndpoint(
        name="kg_traversal_agent",
        url="http://kg:8000",
        capabilities=["kg"],
        process_endpoint="/agents/kg_traversal_agent/process",
        timeout=55,
    )

    captured = {}

    async def _post(url, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {"expanded_evidence": []},
        )

    orch = object.__new__(OrchestratorAgent)
    orch.registry = SimpleNamespace(get_agent=lambda name: endpoint)
    orch._http_client_override = SimpleNamespace(post=AsyncMock(side_effect=_post))
    orch._evidence_video_anchor = lambda evidence: {
        "source_doc_id": "doc1",
        "start": 0.0,
        "end": 5.0,
    }
    orch.emit_progress = lambda *a, **k: None

    await orch._expand_via_kg_traversal(
        evidence=[{"source_doc_id": "doc1"}],
        missing_aspects=["origin"],
        tenant_id="acme:acme",
        session_id=None,
    )

    assert "timeout" in captured, "agent call did not pass a per-request timeout"
    # httpx.Timeout stores the total read timeout; it must equal the endpoint's.
    assert captured["timeout"].read == 55
