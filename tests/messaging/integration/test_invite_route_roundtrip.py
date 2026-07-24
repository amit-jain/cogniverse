"""Invite tokens minted by the runtime admin route must validate at the gateway.

POST /admin/messaging/invite and InviteTokenManager share one
ConfigManager-backed store. This drives the real admin router over
httpx.ASGITransport through RuntimeClient (the same client the gateway
uses), then validates the returned token with the real InviteTokenManager
against the same store. Both sides canonicalize the system tenant through
the ConfigManager — a reader that hits the store directly with the raw id
looks up a key nobody writes, and every minted token comes back invalid.
"""

import httpx
import pytest
from cogniverse_core.messaging_auth import InviteTokenManager
from cogniverse_messaging.runtime_client import RuntimeClient
from fastapi import FastAPI

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.routers import admin as admin_router
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.integration]


@pytest.fixture
def config_manager():
    store = InMemoryConfigStore()
    store.initialize()
    return ConfigManager(store=store)


@pytest.fixture
def invite_app(config_manager):
    app = FastAPI()
    app.include_router(admin_router.router, prefix="/admin")
    app.dependency_overrides[admin_router.get_config_manager_dependency] = lambda: (
        config_manager
    )
    return app


@pytest.mark.asyncio
async def test_route_minted_token_validates_at_gateway(invite_app, config_manager):
    rc = RuntimeClient("http://runtime")
    rc._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=invite_app),
        base_url="http://runtime",
        timeout=30.0,
    )
    try:
        token = await rc.create_invite_token(tenant_id="acme:alice")
    finally:
        await rc._client.aclose()

    assert token is not None
    assert len(token) == 32

    assert InviteTokenManager(config_manager).validate_token(token) == "acme:alice"


@pytest.mark.asyncio
async def test_route_minted_token_is_single_use(invite_app, config_manager):
    rc = RuntimeClient("http://runtime")
    rc._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=invite_app),
        base_url="http://runtime",
        timeout=30.0,
    )
    try:
        token = await rc.create_invite_token(tenant_id="acme:alice")
    finally:
        await rc._client.aclose()

    manager = InviteTokenManager(config_manager)
    assert manager.validate_token(token) == "acme:alice"
    manager.mark_token_used(token, "acme:alice")
    assert manager.validate_token(token) is None
