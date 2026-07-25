"""The gateway's photo download, driven through real python-telegram-bot.

The gateway is the only component holding the bot token, so a photo's bytes must
be fetched here and carried to the runtime. This exercises that with a REAL
``telegram.Bot`` — real request building, real ``getFile`` round trip, real
``File.download_as_bytearray`` — pointed at a local server implementing the Bot
API endpoints the gateway uses (``base_url`` / ``base_file_url`` are supported
by PTB precisely for this).

Telegram's own servers need a bot token this environment does not have, so the
SERVER is local; every line of PTB client code that runs in production runs
here. A wrong method name, a missing await, a mishandled ``file_path``, or
bytes that never make it into the dispatch context all fail this test — none of
which a mocked bot object would catch.

The assertion is the outcome: the base64 payload the runtime receives decodes
byte-for-byte to the image the server served.
"""

from __future__ import annotations

import base64
import io
import json
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest
from cogniverse_messaging.gateway import MAX_PHOTO_BYTES, MessagingGateway
from PIL import Image
from telegram import Bot, Chat, Message, PhotoSize, Update

pytestmark = [pytest.mark.integration]

TOKEN = "123456:TESTTOKEN"
FILE_PATH = "photos/file_1.png"


def _png(colour: tuple[int, int, int], size: int = 96) -> bytes:
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :] = colour
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


class _BotApiHandler(BaseHTTPRequestHandler):
    """The handful of Bot API endpoints the gateway actually calls."""

    payload: bytes = b""
    declared_size: int | None = None
    serve_file_error: int | None = None

    def log_message(self, *args):  # silence per-request stderr logging
        pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)

        if self.path.endswith("/getMe"):
            self._json(
                {
                    "ok": True,
                    "result": {
                        "id": 123456,
                        "is_bot": True,
                        "first_name": "Test",
                        "username": "test_bot",
                    },
                }
            )
        elif self.path.endswith("/getFile"):
            size = (
                self.declared_size
                if self.declared_size is not None
                else len(type(self).payload)
            )
            self._json(
                {
                    "ok": True,
                    "result": {
                        "file_id": "AgACfile",
                        "file_unique_id": "uniq",
                        "file_size": size,
                        "file_path": FILE_PATH,
                    },
                }
            )
        elif self.path.endswith("/sendMessage"):
            self._json(
                {
                    "ok": True,
                    "result": {
                        "message_id": 2,
                        "date": 0,
                        "chat": {"id": 99, "type": "private"},
                        "text": "ok",
                    },
                }
            )
        elif self.path.endswith("/sendChatAction"):
            self._json({"ok": True, "result": True})
        else:
            self._json({"ok": False, "description": "unknown method"}, status=404)

    def do_GET(self):
        if self.path.endswith(FILE_PATH):
            if type(self).serve_file_error is not None:
                self.send_response(type(self).serve_file_error)
                self.end_headers()
                return
            body = type(self).payload
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture
def bot_api():
    """A live local Bot API server; tests set what it serves."""
    _BotApiHandler.payload = b""
    _BotApiHandler.declared_size = None
    _BotApiHandler.serve_file_error = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), _BotApiHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, _BotApiHandler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
async def real_bot(bot_api):
    server, _ = bot_api
    port = server.server_address[1]
    bot = Bot(
        token=TOKEN,
        base_url=f"http://127.0.0.1:{port}/bot",
        base_file_url=f"http://127.0.0.1:{port}/file/bot",
    )
    await bot.initialize()
    try:
        yield bot
    finally:
        await bot.shutdown()


def _photo_update(bot: Bot, *, caption: str | None, file_size: int) -> Update:
    """A real PTB Update carrying a real PhotoSize."""
    user = SimpleNamespace(id=7)
    chat = Chat(id=99, type=Chat.PRIVATE)
    photo = PhotoSize(
        file_id="AgACfile",
        file_unique_id="uniq",
        width=96,
        height=96,
        file_size=file_size,
    )
    message = Message(
        message_id=1,
        date=datetime.now(timezone.utc),
        chat=chat,
        caption=caption,
        photo=(photo,),
    )
    message.set_bot(bot)
    chat.set_bot(bot)
    update = Update(update_id=1, message=message)
    update.set_bot(bot)
    # effective_user is derived from from_user, which Message treats as
    # immutable; the gateway only reads its id.
    object.__setattr__(update, "_effective_user", user)
    return update


def _gateway(dispatched: dict, tenant_id="acme:acme"):
    g = MessagingGateway(bot_token=TOKEN, runtime_url="http://runtime")

    async def _dispatch(**kwargs):
        dispatched.update(kwargs)
        return {"message": "the answer"}

    async def _resolve(_platform, _user_id):
        return {"status": "ok", "tenant_id": tenant_id}

    g.runtime_client = SimpleNamespace(
        dispatch_agent=_dispatch, resolve_tenant=_resolve
    )
    return g


@pytest.mark.asyncio
async def test_photo_bytes_reach_the_runtime_unchanged(real_bot, bot_api):
    """The real PTB download path lands the exact served bytes in the context."""
    _, handler = bot_api
    payload = _png((10, 120, 200))
    handler.payload = payload

    dispatched: dict = {}
    gateway = _gateway(dispatched)
    update = _photo_update(real_bot, caption="what is this", file_size=len(payload))

    await gateway._handle_message(update, SimpleNamespace(bot=real_bot))

    assert dispatched["agent_name"] == "image_search_agent"
    assert dispatched["query"] == "what is this"
    ctx = dispatched["context"]
    assert ctx["media_type"] == "photo"
    assert ctx["media_file_id"] == "AgACfile"
    assert ctx["media_mime"] == "image/jpeg"
    # Byte-for-byte: what the server served is what the runtime receives.
    assert base64.b64decode(ctx["media_content_b64"]) == payload
    # And it is a real decodable image, not a truncated transfer.
    assert Image.open(io.BytesIO(base64.b64decode(ctx["media_content_b64"]))).size == (
        96,
        96,
    )


@pytest.mark.asyncio
async def test_oversized_photo_is_refused_and_never_dispatched(real_bot, bot_api):
    _, handler = bot_api
    handler.payload = _png((5, 5, 5))
    handler.declared_size = MAX_PHOTO_BYTES + 1

    dispatched: dict = {}
    gateway = _gateway(dispatched)
    update = _photo_update(real_bot, caption=None, file_size=MAX_PHOTO_BYTES + 1)

    await gateway._handle_message(update, SimpleNamespace(bot=real_bot))

    assert dispatched == {}


@pytest.mark.asyncio
async def test_download_failure_does_not_dispatch(real_bot, bot_api):
    """A real HTTP failure on the file fetch must not send a half-formed query."""
    _, handler = bot_api
    handler.payload = _png((9, 9, 9))
    handler.serve_file_error = 500

    dispatched: dict = {}
    gateway = _gateway(dispatched)
    update = _photo_update(real_bot, caption=None, file_size=1234)

    await gateway._handle_message(update, SimpleNamespace(bot=real_bot))

    assert dispatched == {}
