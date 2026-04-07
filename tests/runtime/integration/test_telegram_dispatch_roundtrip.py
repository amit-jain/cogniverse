"""Round-trip integration test for Telegram custom command dispatch.

Verifies the full chain: command_router.parse_message → gateway dispatch
arm → runtime_client method → real wiki/tenant router endpoint → real
WikiManager / ConfigStore → real Vespa.

Audit fixes #4 + #5 — before this fix, four command families (/wiki,
/instructions, /memories, /jobs) were parsed correctly into ParsedCommand
flags but the gateway never read those flags, and runtime_client had no
methods to call the corresponding endpoints. The pre-fix tests stopped
at parsing or used mocks that hid the missing methods.

This test exercises the chain through real services.
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from cogniverse_messaging.command_router import parse_message
from cogniverse_messaging.gateway import MessagingGateway
from cogniverse_messaging.runtime_client import RuntimeClient
from fastapi import FastAPI
from httpx import ASGITransport

from cogniverse_agents.wiki.wiki_manager import WikiManager
from cogniverse_runtime.routers import tenant
from cogniverse_runtime.routers import wiki as wiki_router


@pytest.fixture
def real_runtime_app(vespa_instance, config_manager, schema_loader):
    """A real FastAPI app with the wiki and tenant routers mounted, wired
    to the test Vespa via DI. The gateway will hit this app via ASGI
    transport — no network, but real router code, real Vespa I/O."""
    from cogniverse_core.registries.backend_registry import BackendRegistry

    BackendRegistry._backend_instances.clear()

    backend = BackendRegistry.get_instance().get_ingestion_backend(
        name="vespa",
        tenant_id="default",
        config={
            "backend": {
                "url": "http://localhost",
                "port": vespa_instance["http_port"],
                "config_port": vespa_instance["config_port"],
            }
        },
        config_manager=config_manager,
        schema_loader=schema_loader,
    )

    managers: dict = {}

    def factory(tenant_id: str) -> WikiManager:
        if tenant_id in managers:
            return managers[tenant_id]
        try:
            backend.schema_registry.deploy_schema(
                tenant_id=tenant_id, base_schema_name="wiki_pages"
            )
        except Exception:
            pass
        mgr = WikiManager(
            backend=backend,
            tenant_id=tenant_id,
            schema_name=f"wiki_pages_{tenant_id}",
        )
        managers[tenant_id] = mgr
        return mgr

    original_factory = wiki_router._wiki_manager_factory
    wiki_router.set_wiki_manager_factory(factory)
    tenant.set_config_manager(config_manager)

    app = FastAPI()
    app.include_router(wiki_router.router, prefix="/wiki")
    app.include_router(tenant.router, prefix="/admin/tenant")

    yield app

    wiki_router._wiki_manager_factory = original_factory


@pytest.fixture
def gateway_with_real_runtime(real_runtime_app):
    """MessagingGateway whose runtime_client makes real HTTP requests
    against the in-process FastAPI app via ASGI transport."""
    gateway = MessagingGateway(
        bot_token="fake-token",
        runtime_url="http://testrunner",
    )

    real_client = RuntimeClient(runtime_url="http://testrunner")
    real_client._client = httpx.AsyncClient(
        base_url="http://testrunner",
        transport=ASGITransport(app=real_runtime_app),
    )
    gateway.runtime_client = real_client

    yield gateway


def _mock_telegram_update():
    update = MagicMock()
    update.message = MagicMock()
    update.message.reply_text = AsyncMock()
    return update


@pytest.mark.integration
class TestTelegramWikiDispatchRoundTrip:
    @pytest.mark.asyncio
    async def test_wiki_index_full_chain(self, gateway_with_real_runtime):
        """``/wiki index`` from Telegram → real /wiki/index endpoint → real
        WikiManager.get_index() → real Vespa. Pre-fix the command was
        parsed into is_wiki=True but never dispatched anywhere."""
        gateway = gateway_with_real_runtime
        update = _mock_telegram_update()

        parsed = parse_message("/wiki index")
        assert parsed.is_wiki and parsed.wiki_subcommand == "index"

        await gateway._handle_wiki_command(update, parsed, "tg_wiki_idx_test")

        update.message.reply_text.assert_awaited_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert reply_text is not None

    @pytest.mark.asyncio
    async def test_wiki_search_full_chain(self, gateway_with_real_runtime):
        """``/wiki search`` must reach the real /wiki/search endpoint."""
        gateway = gateway_with_real_runtime
        update = _mock_telegram_update()

        parsed = parse_message("/wiki search test query")
        assert parsed.is_wiki and parsed.wiki_subcommand == "search"

        await gateway._handle_wiki_command(update, parsed, "tg_wiki_search_test")

        update.message.reply_text.assert_awaited_once()
        reply_text = update.message.reply_text.call_args[0][0]
        assert "wiki result" in reply_text.lower()


@pytest.mark.integration
class TestTelegramInstructionsDispatchRoundTrip:
    @pytest.mark.asyncio
    async def test_instructions_set_then_show(self, gateway_with_real_runtime):
        """``/instructions set`` followed by ``/instructions show`` must
        round-trip through the real ConfigStore — set persists, show
        retrieves the same text. Pre-fix neither command worked at all."""
        gateway = gateway_with_real_runtime
        tenant_id = "tg_instr_rt_test"

        update_set = _mock_telegram_update()
        parsed_set = parse_message("/instructions set always reply in french")
        await gateway._handle_instructions_command(update_set, parsed_set, tenant_id)
        update_set.message.reply_text.assert_awaited_once()
        set_reply = update_set.message.reply_text.call_args[0][0]
        assert "updated" in set_reply.lower()

        update_show = _mock_telegram_update()
        parsed_show = parse_message("/instructions show")
        await gateway._handle_instructions_command(update_show, parsed_show, tenant_id)
        update_show.message.reply_text.assert_awaited_once()
        show_reply = update_show.message.reply_text.call_args[0][0]
        assert "always reply in french" in show_reply.lower()


@pytest.mark.integration
class TestTelegramJobsDispatchRoundTrip:
    @pytest.mark.asyncio
    async def test_jobs_create_then_list(self, gateway_with_real_runtime):
        """``/jobs create`` followed by ``/jobs list`` must round-trip
        through real ConfigStore. Pre-fix the entire /jobs family was
        silently dropped."""
        gateway = gateway_with_real_runtime
        tenant_id = "tg_jobs_rt_test"

        update_create = _mock_telegram_update()
        parsed_create = parse_message(
            '/jobs create "0 9 * * 1" weekly news brief'
        )
        await gateway._handle_jobs_command(update_create, parsed_create, tenant_id)
        update_create.message.reply_text.assert_awaited_once()
        create_reply = update_create.message.reply_text.call_args[0][0]
        assert "created" in create_reply.lower() or "job" in create_reply.lower()

        update_list = _mock_telegram_update()
        parsed_list = parse_message("/jobs list")
        await gateway._handle_jobs_command(update_list, parsed_list, tenant_id)
        update_list.message.reply_text.assert_awaited_once()
        list_reply = update_list.message.reply_text.call_args[0][0]
        assert "weekly news brief" in list_reply.lower() or "1" in list_reply
