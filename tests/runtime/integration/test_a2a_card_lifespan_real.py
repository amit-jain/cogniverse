"""The A2A agent card must advertise the real loaded agents.

Regression: the card was built from agent_registry.list_agents() BEFORE
config_loader.load_agents() ran, so the registry was empty and skills fell back
to a single 'default' skill. This drives the real main.py lifespan and reads the
mounted card over ASGI.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_a2a_card_advertises_loaded_agents(monkeypatch):
    # Keep the lifespan light: no sandbox connect, no memory lifecycle scan.
    monkeypatch.setenv("COGNIVERSE_SANDBOX_POLICY", "disabled")
    monkeypatch.setenv("COGNIVERSE_MEMORY_LIFECYCLE_DISABLED", "1")
    # dspy.configure is once-per-task; stub so a re-run in the module is safe.
    import dspy

    monkeypatch.setattr(dspy, "configure", lambda *a, **kw: None)

    from cogniverse_runtime.main import lifespan

    app = FastAPI()
    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            resp = await client.get("/a2a/.well-known/agent-card.json")

        assert resp.status_code == 200, resp.text[:300]
        body = resp.json()
        skill_ids = {s["id"] for s in body["skills"]}
        # The card must reflect the populated registry, not the 'default' stub.
        assert "search_agent" in skill_ids, skill_ids
        assert "default" not in skill_ids, skill_ids
        assert len(skill_ids) > 1, skill_ids
