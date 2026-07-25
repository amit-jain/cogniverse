"""Opt-in boundary test against Telegram's real Bot API.

Precondition: send the configured bot a photo from a real Telegram account.
The test deliberately does not acknowledge the update, so the same photo can
be reused while developing. ``TELEGRAM_CHAT_ID`` may be set to select one chat
when the bot has updates from multiple users.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest
from cogniverse_cli.secrets import read_secret
from cogniverse_messaging.gateway import MessagingGateway
from telegram import Bot

pytestmark = [pytest.mark.integration, pytest.mark.local_only]


@pytest.mark.asyncio
async def test_real_telegram_photo_reaches_runtime_unchanged():
    token = read_secret("TELEGRAM_BOT_TOKEN")
    if not token:
        pytest.skip(
            "TELEGRAM_BOT_TOKEN is not configured; export it or create "
            ".env/TELEGRAM_BOT_TOKEN.env"
        )

    selected_chat = read_secret("TELEGRAM_CHAT_ID")
    async with Bot(token=token) as bot:
        webhook = await bot.get_webhook_info()
        if webhook.url:
            pytest.skip(
                "The bot has a webhook configured, so Telegram disables getUpdates; "
                "run against a dedicated test bot or temporarily remove its webhook"
            )

        updates = await bot.get_updates(
            limit=100,
            timeout=0,
            allowed_updates=["message"],
        )
        photo_updates = [
            update
            for update in updates
            if update.message
            and update.message.photo
            and (
                selected_chat is None
                or str(update.effective_chat.id) == str(selected_chat)
            )
        ]
        if not photo_updates:
            pytest.skip(
                "No pending user-originated photo update; send the bot a photo "
                "and optionally set TELEGRAM_CHAT_ID, then rerun this test"
            )

        dispatched: dict = {}

        async def _dispatch(**kwargs):
            dispatched.update(kwargs)
            return {"message": "real Telegram photo received"}

        async def _resolve(_platform, _user_id):
            return {"status": "ok", "tenant_id": "telegram-live:test"}

        gateway = MessagingGateway(bot_token=token, runtime_url="http://runtime")
        gateway.runtime_client = SimpleNamespace(
            dispatch_agent=_dispatch,
            resolve_tenant=_resolve,
        )

        update = photo_updates[-1]
        await gateway._handle_message(update, SimpleNamespace(bot=bot))
        telegram_file = await bot.get_file(update.message.photo[-1].file_id)
        expected_payload = bytes(await telegram_file.download_as_bytearray())

    assert dispatched["agent_name"] == "image_search_agent"
    context = dispatched["context"]
    assert context["media_type"] == "photo"
    assert context["media_file_id"] == update.message.photo[-1].file_id
    payload = base64.b64decode(context["media_content_b64"], validate=True)
    assert payload == expected_payload
    assert context["media_mime"] == "image/jpeg"
