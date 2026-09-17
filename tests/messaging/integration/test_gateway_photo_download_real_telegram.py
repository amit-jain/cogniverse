"""The gateway's photo download against Telegram's real Bot API.

The test sends a generated photo to the test chat itself, wraps the message
Telegram returns (with its real, re-encoded ``file_id``s) in an ``Update`` and
hands it to the gateway, which downloads the photo through the real API. Bots
do not receive their own messages through ``getUpdates``, so the update is built
from the ``sendPhoto`` result rather than polled. The chat must be a private
chat or a group: the gateway reads ``effective_user``, which a channel post
does not carry.

Credentials: ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_TEST_CHAT_ID``, resolved
through ``read_secret`` (environment, then ``.env``).
"""

from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
from cogniverse_cli.secrets import read_secret
from cogniverse_messaging.gateway import MessagingGateway
from PIL import Image
from telegram import Bot, Update

pytestmark = [pytest.mark.integration, pytest.mark.local_only]

# Under Telegram's 1280 px photo bound, so the largest PhotoSize keeps it.
PHOTO_WIDTH = 320
PHOTO_HEIGHT = 240
CAPTION = "find frames that look like this"
RUNTIME_REPLY = "real Telegram photo received"
TENANT_ID = "telegram-live:test"
JPEG_MAGIC = b"\xff\xd8\xff"


def _credentials() -> tuple[str, int]:
    token = read_secret("TELEGRAM_BOT_TOKEN")
    chat_id = read_secret("TELEGRAM_TEST_CHAT_ID")
    missing = [
        name
        for name, value in (
            ("TELEGRAM_BOT_TOKEN", token),
            ("TELEGRAM_TEST_CHAT_ID", chat_id),
        )
        if not value
    ]
    if missing:
        pytest.fail(
            f"{' and '.join(missing)} must be set (environment or "
            ".env/<NAME>.env) to send the test photo through the real Bot API",
            pytrace=False,
        )
    return token, int(chat_id)


def _photo_png() -> bytes:
    """A two-tone PNG at the dimensions the downloaded photo must keep."""
    pixels = np.zeros((PHOTO_HEIGHT, PHOTO_WIDTH, 3), dtype=np.uint8)
    pixels[:, : PHOTO_WIDTH // 2] = (200, 40, 40)
    pixels[:, PHOTO_WIDTH // 2 :] = (40, 40, 200)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_real_telegram_photo_reaches_runtime_unchanged():
    token, chat_id = _credentials()
    dispatched: dict = {}
    replies: list = []

    class RecordingBot(Bot):
        """The real Bot, keeping the replies it sends so the test removes them."""

        async def send_message(self, *args, **kwargs):
            message = await super().send_message(*args, **kwargs)
            replies.append(message)
            return message

    async def _dispatch(**kwargs):
        dispatched.update(kwargs)
        return {"message": RUNTIME_REPLY}

    async def _resolve(_platform, _user_id):
        return {"status": "ok", "tenant_id": TENANT_ID}

    gateway = MessagingGateway(bot_token=token, runtime_url="http://runtime")
    gateway.runtime_client = SimpleNamespace(
        dispatch_agent=_dispatch,
        resolve_tenant=_resolve,
    )

    async with RecordingBot(token=token) as bot:
        sent = await bot.send_photo(
            chat_id=chat_id, photo=_photo_png(), caption=CAPTION
        )
        try:
            if sent.from_user is None:
                pytest.fail(
                    f"TELEGRAM_TEST_CHAT_ID {chat_id} is a {sent.chat.type}; the "
                    "gateway needs a private chat or group, whose messages "
                    "carry a sender",
                    pytrace=False,
                )
            largest = sent.photo[-1]
            await gateway._handle_message(
                Update(update_id=0, message=sent), SimpleNamespace(bot=bot)
            )
            telegram_file = await bot.get_file(largest.file_id)
            expected_payload = bytes(await telegram_file.download_as_bytearray())
        finally:
            for message in [sent, *replies]:
                await bot.delete_message(
                    chat_id=message.chat_id, message_id=message.message_id
                )

    assert set(dispatched) == {
        "agent_name",
        "query",
        "tenant_id",
        "context_id",
        "context",
    }
    assert dispatched["agent_name"] == "image_search_agent"
    assert dispatched["query"] == CAPTION
    assert dispatched["tenant_id"] == TENANT_ID
    assert dispatched["context_id"] == str(chat_id)
    context = dispatched["context"]
    assert set(context) == {
        "media_type",
        "media_file_id",
        "media_content_b64",
        "media_mime",
    }
    assert context["media_type"] == "photo"
    assert context["media_file_id"] == largest.file_id
    assert context["media_mime"] == "image/jpeg"

    payload = base64.b64decode(context["media_content_b64"], validate=True)
    assert payload == expected_payload
    assert len(payload) == largest.file_size
    assert payload[: len(JPEG_MAGIC)] == JPEG_MAGIC
    with Image.open(io.BytesIO(payload)) as image:
        assert (image.format, image.size) == ("JPEG", (PHOTO_WIDTH, PHOTO_HEIGHT))
    assert (largest.width, largest.height) == (PHOTO_WIDTH, PHOTO_HEIGHT)

    assert [(message.chat_id, message.text) for message in replies] == [
        (chat_id, RUNTIME_REPLY)
    ]
