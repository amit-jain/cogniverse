"""The Telegram gateway /start invite-token handler plus run/main lifecycle.

test_auth.py covers InviteTokenManager against a MagicMock config store — it
asserts the payloads the manager builds, not the round-trip, and nothing tests
_handle_start at all. Here _handle_start runs against a real (in-memory)
ConfigStore, so the full invite flow is exercised: a valid token registers and
is CONSUMED (single-use), a used or unknown token is rejected, and a bare
/start prompts for a token.

The run()/run_polling()/main() entry points were also untested (only the
webhook HTTP path had coverage): run() must dispatch by mode, run_polling()
must start the updater and clean up on cancellation, and main() must validate
its required environment before constructing the gateway.
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


@pytest.mark.asyncio
async def test_run_dispatches_by_mode():
    from unittest.mock import AsyncMock, patch

    webhook_gw = MessagingGateway(
        bot_token="123:FAKE", runtime_url="http://runtime", mode="webhook"
    )
    with (
        patch.object(webhook_gw, "run_webhook", new=AsyncMock()) as wh,
        patch.object(webhook_gw, "run_polling", new=AsyncMock()) as pl,
    ):
        await webhook_gw.run()
        wh.assert_awaited_once()
        pl.assert_not_awaited()

    polling_gw = MessagingGateway(
        bot_token="123:FAKE", runtime_url="http://runtime", mode="polling"
    )
    with (
        patch.object(polling_gw, "run_webhook", new=AsyncMock()) as wh2,
        patch.object(polling_gw, "run_polling", new=AsyncMock()) as pl2,
    ):
        await polling_gw.run()
        pl2.assert_awaited_once()
        wh2.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_polling_starts_updater_and_cleans_up_on_cancel():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    gw = MessagingGateway(bot_token="123:FAKE", runtime_url="http://runtime")

    app = MagicMock()
    app.initialize = AsyncMock()
    app.start = AsyncMock()
    app.stop = AsyncMock()
    app.shutdown = AsyncMock()
    app.updater.start_polling = AsyncMock()
    app.updater.stop = AsyncMock()
    gw.runtime_client = MagicMock()
    gw.runtime_client.close = AsyncMock()

    with (
        patch.object(gw, "build_app", return_value=app),
        patch(
            "cogniverse_messaging.gateway.asyncio.sleep",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ),
    ):
        await gw.run_polling()

    app.initialize.assert_awaited_once()
    app.start.assert_awaited_once()
    app.updater.start_polling.assert_awaited_once()
    # The finally block must run the full teardown even though the poll loop
    # was cancelled — otherwise the updater and HTTP client leak.
    app.updater.stop.assert_awaited_once()
    app.stop.assert_awaited_once()
    app.shutdown.assert_awaited_once()
    gw.runtime_client.close.assert_awaited_once()


def test_main_exits_without_bot_token(monkeypatch):
    from cogniverse_messaging import gateway as gw_mod

    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exc:
        gw_mod.main()
    assert exc.value.code == 1


def test_main_webhook_mode_requires_url(monkeypatch):
    from cogniverse_messaging import gateway as gw_mod

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:FAKE")
    monkeypatch.setenv("GATEWAY_MODE", "webhook")
    monkeypatch.delenv("TELEGRAM_WEBHOOK_URL", raising=False)
    with pytest.raises(SystemExit) as exc:
        gw_mod.main()
    assert exc.value.code == 1


def test_main_constructs_gateway_from_env_and_runs(monkeypatch):
    from unittest.mock import MagicMock, patch

    from cogniverse_messaging import gateway as gw_mod

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:FAKE")
    monkeypatch.setenv("GATEWAY_MODE", "polling")
    monkeypatch.setenv("RUNTIME_URL", "http://rt:1234")

    fake_gw = MagicMock()
    with (
        patch.object(gw_mod, "MessagingGateway", return_value=fake_gw) as ctor,
        patch.object(gw_mod.asyncio, "run") as run,
    ):
        gw_mod.main()

    ctor.assert_called_once()
    kwargs = ctor.call_args.kwargs
    assert kwargs["bot_token"] == "123:FAKE"
    assert kwargs["runtime_url"] == "http://rt:1234"
    assert kwargs["mode"] == "polling"
    run.assert_called_once()
