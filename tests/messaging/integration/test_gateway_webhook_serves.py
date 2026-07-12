"""Webhook mode must actually SERVE, not just register the webhook URL.

run_webhook previously called ``bot.set_webhook`` only — which registers the
URL with Telegram but binds no HTTP server, so the documented production mode
received zero messages. It must start the webhook server (``start_webhook``).

The external Telegram API (get_me / set_webhook / delete_webhook) is unreachable
in a test, so those bot methods are stubbed; the webhook HTTP server under test
runs for real, and a real POST to it must be served.
"""

from __future__ import annotations

import asyncio
import socket

import httpx
import pytest
import telegram
from cogniverse_messaging.gateway import MessagingGateway

pytestmark = [pytest.mark.integration]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.mark.asyncio
async def test_webhook_mode_binds_and_serves(monkeypatch):
    from telegram import User
    from telegram.ext import ExtBot

    fake_user = User(id=1, first_name="bot", is_bot=True, username="testbot")

    async def fake_get_me(self, *a, **k):
        self._bot_user = fake_user
        return fake_user

    async def fake_true(self, *a, **k):
        return True

    # Stub only the external Telegram HTTP calls; the webhook server is real.
    # PTB uses ExtBot (a Bot subclass), so patch there.
    for cls in (telegram.Bot, ExtBot):
        monkeypatch.setattr(cls, "get_me", fake_get_me, raising=False)
        monkeypatch.setattr(cls, "set_webhook", fake_true, raising=False)
        monkeypatch.setattr(cls, "delete_webhook", fake_true, raising=False)
        monkeypatch.setattr(cls, "get_webhook_info", fake_true, raising=False)

    port = _free_port()
    gw = MessagingGateway(
        bot_token="123:FAKE",
        runtime_url="http://runtime",
        mode="webhook",
        webhook_url=f"http://127.0.0.1:{port}/hook",
        webhook_listen="127.0.0.1",
        webhook_port=port,
        webhook_path="hook",
    )

    task = asyncio.create_task(gw.run_webhook())
    try:
        # Poll until the server binds. Pre-fix nothing ever binds this port.
        bound = False
        for _ in range(50):
            try:
                _r, w = await asyncio.open_connection("127.0.0.1", port)
                w.close()
                await w.wait_closed()
                bound = True
                break
            except (ConnectionRefusedError, OSError):
                await asyncio.sleep(0.1)
        assert bound, "webhook mode never bound an HTTP server on the port"

        # A real webhook POST is served (200), not connection-refused.
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://127.0.0.1:{port}/hook",
                json={"update_id": 1},
                timeout=5,
            )
        assert resp.status_code == 200
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
