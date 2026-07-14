"""The Telegram gateway /start invite-token handler.

test_auth.py covers InviteTokenManager against a MagicMock config store — it
asserts the payloads the manager builds, not the round-trip, and nothing tests
_handle_start at all. Here _handle_start runs against a real (in-memory)
ConfigStore, so the full invite flow is exercised: a valid token registers and
is CONSUMED (single-use), a used or unknown token is rejected, and a bare
/start prompts for a token.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from cogniverse_messaging.gateway import MessagingGateway

pytestmark = [pytest.mark.unit]


def _fake_update(text: str, user_id: int = 42):
    """Minimal stand-in for a PTB Update carrying a /start message.

    Returns (update, replies); replies accumulates every reply_text call.
    """
    replies: list[str] = []

    async def reply_text(msg, *a, **k):
        replies.append(msg)

    message = SimpleNamespace(text=text, reply_text=reply_text)
    update = SimpleNamespace(
        message=message, effective_user=SimpleNamespace(id=user_id)
    )
    return update, replies


def _gateway(config_manager):
    return MessagingGateway(
        bot_token="123:FAKE",
        runtime_url="http://runtime",
        config_manager=config_manager,
    )


@pytest.mark.asyncio
async def test_handle_start_consumes_valid_invite_token(config_manager_memory):
    gw = _gateway(config_manager_memory)
    gw._init_auth()
    token = gw._token_manager.generate_token("acme:alice")

    update, replies = _fake_update(f"/start {token}")
    await gw._handle_start(update, context=None)

    assert any("Registered as acme:alice" in r for r in replies), replies
    # The token round-tripped through the store and is now consumed.
    assert gw._token_manager.validate_token(token) is None

    update2, replies2 = _fake_update(f"/start {token}")
    await gw._handle_start(update2, context=None)
    assert any("Invalid or expired invite token" in r for r in replies2), replies2


@pytest.mark.asyncio
async def test_handle_start_rejects_unknown_token(config_manager_memory):
    gw = _gateway(config_manager_memory)
    update, replies = _fake_update("/start not-a-real-token-xyz")
    await gw._handle_start(update, context=None)
    assert any("Invalid or expired invite token" in r for r in replies), replies


@pytest.mark.asyncio
async def test_handle_start_without_token_prompts_registration(config_manager_memory):
    gw = _gateway(config_manager_memory)
    update, replies = _fake_update("/start")
    await gw._handle_start(update, context=None)
    assert any("/start <invite_token>" in r for r in replies), replies
