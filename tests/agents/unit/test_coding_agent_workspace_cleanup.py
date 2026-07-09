"""CodingAgent must remove its temp workspace after a run.

The workspace is only a staging area for the file write (the sandbox is the
real execution environment). It was created per request with mkdtemp and never
removed, so the pod's disk grew unbounded under load.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from cogniverse_agents.coding_agent import CodingAgent, CodingInput


@pytest.mark.asyncio
async def test_workspace_removed_after_run():
    agent = object.__new__(CodingAgent)

    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def _tracking_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(path)
        return path

    with (
        patch.object(agent, "emit_progress", lambda *a, **k: None),
        patch.object(agent, "_search_code_context", AsyncMock(return_value="")),
        patch.object(agent, "_plan", AsyncMock(return_value="do the thing")),
        patch.object(
            agent, "_generate_code", AsyncMock(return_value=("print(1)", "python x"))
        ),
        patch.object(
            agent,
            "_execute_in_sandbox",
            AsyncMock(return_value={"stdout": "1", "stderr": "", "exit_code": 0}),
        ),
        patch.object(agent, "_evaluate_output", AsyncMock(return_value=(True, "ok"))),
        patch.object(agent, "should_use_rlm_for_query", lambda *a, **k: False),
        patch("cogniverse_agents.coding_agent.tempfile.mkdtemp", _tracking_mkdtemp),
    ):
        out = await agent._process_impl(
            CodingInput(task="add two numbers", tenant_id="acme:acme", max_iterations=1)
        )

    assert out is not None
    assert created, "a workspace must have been created"
    assert not Path(created[0]).exists(), "workspace temp dir must be cleaned up"
