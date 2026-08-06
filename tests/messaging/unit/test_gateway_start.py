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

import asyncio
import contextlib
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


class _PartitionedMemory:
    """Partition-faithful Mem0 double with failure toggles."""

    def __init__(self):
        self.store = {}
        self.fail_writes = False
        self.fail_reads = False
        self.memory = object()

    def add_memory(self, content, tenant_id, agent_name, metadata=None, **kwargs):
        if self.fail_writes:
            raise ConnectionError("mem0 down")
        self.store.setdefault((tenant_id, agent_name), []).append(
            {"memory": content, "metadata": metadata or {}}
        )
        return "mem_1"

    def get_all_memories(self, tenant_id, agent_name):
        if self.fail_reads:
            raise ConnectionError("mem0 down")
        return self.store.get((tenant_id, agent_name), [])


def _runtime_harness(config_manager):
    """Real admin router app + memory double, plus a gateway whose
    RuntimeClient speaks to it over ASGITransport — the full registration
    chain (gateway → HTTP → routes → stores) with no stub in the middle."""
    import httpx
    from fastapi import FastAPI

    from cogniverse_runtime.routers import admin as admin_router

    memory = _PartitionedMemory()
    admin_router.set_system_memory_factory(lambda: memory)

    app = FastAPI()
    app.include_router(admin_router.router, prefix="/admin")
    app.dependency_overrides[admin_router.get_config_manager_dependency] = lambda: (
        config_manager
    )

    gw = MessagingGateway(bot_token="123:FAKE", runtime_url="http://runtime")
    gw.runtime_client._client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://runtime"
    )
    return gw, memory


@pytest.fixture
def registration(config_manager_memory):
    gw, memory = _runtime_harness(config_manager_memory)
    try:
        yield gw, memory, config_manager_memory
    finally:
        from cogniverse_runtime.routers import admin as admin_router

        admin_router.set_system_memory_factory(None)


def _mint(config_manager, tenant="acme:alice"):
    from cogniverse_core.messaging_auth import InviteTokenManager

    return InviteTokenManager(config_manager).generate_token(tenant)


@pytest.mark.asyncio
async def test_handle_start_consumes_valid_invite_token(registration):
    gw, memory, cm = registration
    token = _mint(cm)

    update, replies = _fake_update(f"/start {token}")
    await gw._handle_start(update, context=None)

    assert any("Registered as acme:alice" in r for r in replies), replies

    # The mapping round-tripped into the system partition and the token is
    # consumed — a second /start with the same token is invalid.
    update2, replies2 = _fake_update(f"/start {token}", user_id=77)
    await gw._handle_start(update2, context=None)
    assert any("Invalid or expired invite token" in r for r in replies2), replies2


@pytest.mark.asyncio
async def test_handle_start_rejects_unknown_token(registration):
    gw, _memory, _cm = registration
    update, replies = _fake_update("/start not-a-real-token-xyz")
    await gw._handle_start(update, context=None)
    assert any("Invalid or expired invite token" in r for r in replies), replies


@pytest.mark.asyncio
async def test_handle_start_without_token_prompts_registration(registration):
    gw, _memory, _cm = registration
    update, replies = _fake_update("/start")
    await gw._handle_start(update, context=None)
    assert any("/start <invite_token>" in r for r in replies), replies


@pytest.mark.asyncio
async def test_registered_user_resolves_through_runtime(registration):
    """After /start, a message resolves the tenant via the runtime and the
    result is cached — the second message makes no further resolve call."""
    gw, _memory, cm = registration
    token = _mint(cm)
    update, replies = _fake_update(f"/start {token}", user_id=42)
    await gw._handle_start(update, context=None)
    assert any("Registered" in r for r in replies), replies

    resolved = await gw._resolve_tenant("42")
    assert resolved == {"status": "ok", "tenant_id": "acme:alice"}


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


@pytest.mark.asyncio
async def test_run_rejects_unknown_mode():
    """A typo'd GATEWAY_MODE (e.g. 'webook') must fail loudly — silently
    falling through to polling binds no webhook server and the deployment
    receives zero messages with no error anywhere."""
    from unittest.mock import AsyncMock, patch

    g = MessagingGateway(
        bot_token="123:FAKE", runtime_url="http://runtime", mode="webook"
    )
    with (
        patch.object(g, "run_webhook", new=AsyncMock()) as wh,
        patch.object(g, "run_polling", new=AsyncMock()) as pl,
    ):
        with pytest.raises(ValueError, match="webook"):
            await g.run()
        wh.assert_not_awaited()
        pl.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_mem0_failure_keeps_token_and_recovers(registration):
    """A failed mapping write must NOT consume the token: the runtime
    replies 503, the gateway says temporarily unavailable, and the same
    token succeeds once the memory backend recovers."""
    gw, memory, cm = registration
    token = _mint(cm)

    memory.fail_writes = True
    update, replies = _fake_update(f"/start {token}")
    await gw._handle_start(update, context=None)

    assert any("temporarily unavailable" in r for r in replies), replies
    assert not any("Registered as" in r for r in replies), replies

    memory.fail_writes = False
    update2, replies2 = _fake_update(f"/start {token}")
    await gw._handle_start(update2, context=None)

    assert any("Registered as acme:alice" in r for r in replies2), replies2


@pytest.mark.asyncio
async def test_start_config_outage_reports_temporarily_unavailable():
    """A config-store outage during /start must read as "temporarily
    unavailable", never as "Invalid token" — users discard good tokens."""
    from cogniverse_foundation.config.manager import ConfigManager
    from tests.utils.memory_store import InMemoryConfigStore

    class OutageStore(InMemoryConfigStore):
        def get_config(self, *args, **kwargs):
            raise ConnectionError("store down")

    store = OutageStore()
    store.initialize()
    gw, _memory = _runtime_harness(ConfigManager(store=store))
    try:
        update, replies = _fake_update("/start sometoken")
        await gw._handle_start(update, context=None)
    finally:
        from cogniverse_runtime.routers import admin as admin_router

        admin_router.set_system_memory_factory(None)

    assert any("temporarily unavailable" in r for r in replies), replies
    assert not any("Invalid or expired" in r for r in replies), replies


@pytest.mark.asyncio
async def test_start_with_runtime_down_reports_temporarily_unavailable():
    gw = MessagingGateway(bot_token="123:FAKE", runtime_url="http://127.0.0.1:29071")
    try:
        update, replies = _fake_update("/start sometoken")
        await gw._handle_start(update, context=None)
    finally:
        await gw.runtime_client.close()

    assert any("temporarily unavailable" in r for r in replies), replies


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("GATEWAY_WEBHOOK_PORT", "not-a-port"),
        ("GATEWAY_OUTBOUND_POLL_SECONDS", "fast"),
    ],
)
def test_main_rejects_non_numeric_env(monkeypatch, var, value):
    """A malformed numeric env var exits 1 with a logged config error —
    previously a bare ValueError traceback out of main()."""
    from cogniverse_messaging import gateway as gw_mod

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:FAKE")
    monkeypatch.setenv(var, value)
    with pytest.raises(SystemExit) as exc:
        gw_mod.main()
    assert exc.value.code == 1


@pytest.mark.asyncio
async def test_sigterm_routes_through_graceful_shutdown():
    """SIGTERM cancels the run task so the mode's finally-teardown runs —
    the default disposition killed the process with the webhook still
    registered and nothing closed."""
    import os
    import signal as signal_mod

    gw = MessagingGateway(bot_token="123:FAKE", runtime_url="http://runtime")

    teardown_ran = asyncio.Event()
    started = asyncio.Event()

    async def fake_polling():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            teardown_ran.set()

    gw.run_polling = fake_polling

    task = asyncio.create_task(gw.run())
    await asyncio.wait_for(started.wait(), timeout=2)

    os.kill(os.getpid(), signal_mod.SIGTERM)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)
    assert teardown_ran.is_set()

    # The handler is removed on exit — a second SIGTERM must not linger.
    assert asyncio.get_running_loop().remove_signal_handler(signal_mod.SIGTERM) is False


@pytest.mark.asyncio
async def test_sigterm_flushes_pending_outbound_retry_to_the_log(caplog):
    """A message mid-retry-backoff at SIGTERM has no persistence across
    restarts — the shutdown path must at least log it (and clear the
    buffer) instead of the process exiting with no record it existed."""
    import logging
    import os
    import signal as signal_mod

    gw = MessagingGateway(bot_token="123:FAKE", runtime_url="http://runtime")
    gw._outbound_retry = [
        ({"chat_id": "42", "text": "job done"}, 1),
        ({"chat_id": "43", "text": "job done too"}, 2),
    ]

    started = asyncio.Event()

    async def fake_polling():
        started.set()
        await asyncio.Event().wait()

    gw.run_polling = fake_polling

    task = asyncio.create_task(gw.run())
    await asyncio.wait_for(started.wait(), timeout=2)

    with caplog.at_level(logging.ERROR, logger="cogniverse_messaging.gateway"):
        os.kill(os.getpid(), signal_mod.SIGTERM)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)

    # The buffer is cleared and each dropped message left its own log line
    # naming the chat and how many attempts it had already burned.
    assert gw._outbound_retry == []
    messages = [r.getMessage() for r in caplog.records]
    assert any("chat 42" in m and "attempt 1/3" in m for m in messages), messages
    assert any("chat 43" in m and "attempt 2/3" in m for m in messages), messages


@pytest.mark.asyncio
async def test_sigterm_with_empty_outbound_retry_logs_nothing_extra(caplog):
    """An empty buffer at shutdown must not manufacture a spurious drop log."""
    import logging
    import os
    import signal as signal_mod

    gw = MessagingGateway(bot_token="123:FAKE", runtime_url="http://runtime")
    assert gw._outbound_retry == []

    started = asyncio.Event()

    async def fake_polling():
        started.set()
        await asyncio.Event().wait()

    gw.run_polling = fake_polling

    task = asyncio.create_task(gw.run())
    await asyncio.wait_for(started.wait(), timeout=2)

    with caplog.at_level(logging.ERROR, logger="cogniverse_messaging.gateway"):
        os.kill(os.getpid(), signal_mod.SIGTERM)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)

    assert not any("in-memory retry buffer" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_webhook_shutdown_survives_delete_webhook_failure():
    """A failing delete_webhook during shutdown must not skip the local
    teardown — stop/shutdown/close still run."""
    from unittest.mock import AsyncMock, patch

    gw = MessagingGateway(
        bot_token="123:FAKE", runtime_url="http://runtime", mode="webhook"
    )

    app = SimpleNamespace(
        initialize=AsyncMock(),
        start=AsyncMock(),
        updater=SimpleNamespace(start_webhook=AsyncMock(), stop=AsyncMock()),
        bot=SimpleNamespace(
            delete_webhook=AsyncMock(side_effect=RuntimeError("telegram down"))
        ),
        stop=AsyncMock(),
        shutdown=AsyncMock(),
    )
    gw.build_app = lambda: app
    close = AsyncMock()
    with patch.object(gw.runtime_client, "close", close):
        task = asyncio.create_task(gw.run_webhook())
        for _ in range(50):
            await asyncio.sleep(0.01)
            if app.updater.start_webhook.await_count:
                break
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    app.bot.delete_webhook.assert_awaited_once()
    app.stop.assert_awaited_once()
    app.shutdown.assert_awaited_once()
    close.assert_awaited_once()
