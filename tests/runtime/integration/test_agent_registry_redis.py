"""Agent registrations shared through a real Redis.

Every registry under test shares one key prefix with the others of its test
and nothing else: each has its own Redis client, standing in for its own
runtime process. Configured agents are registered on each the way every
process registers them from configuration at startup.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from redis.asyncio import Redis

from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import (
    AgentRegistry,
    AgentRegistryUnavailableError,
    endpoint_data,
    endpoint_from_data,
)
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_runtime.agent_registry_store import RedisAgentRegistryStore
from cogniverse_runtime.shared_state import connect_shared_state_redis

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
    pytest.mark.asyncio,
]

CONFIGURED = AgentEndpoint(
    name="search_agent",
    url="http://localhost:8000",
    capabilities=["search", "video_search"],
    process_endpoint="/agents/search_agent/process",
)
EXTERNAL = AgentEndpoint(
    name="external_agent",
    url="http://external:9000",
    capabilities=["summarization", "search"],
    process_endpoint="/tasks/send",
    timeout=30,
)


@pytest.fixture
def config_manager():
    from cogniverse_foundation.config.manager import ConfigManager
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    store.initialize()
    return ConfigManager(store=store)


@pytest.fixture
def prefix():
    return f"test:agent-registry:{uuid.uuid4().hex}"


@pytest.fixture
async def registries(shared_state_redis_url, prefix, config_manager):
    """Builds registries that each own a client; closes them all."""
    clients: list[Redis] = []

    async def build(count: int = 1, url: str | None = None, timeout: float = 5.0):
        built = []
        for _ in range(count):
            client = await connect_shared_state_redis(
                url or shared_state_redis_url, timeout_seconds=timeout
            )
            clients.append(client)
            registry = AgentRegistry(
                tenant_id="test:unit",
                config_manager=config_manager,
                store=RedisAgentRegistryStore(client, key_prefix=prefix),
            )
            registry.register_agent(endpoint_from_data(endpoint_data(CONFIGURED)))
            built.append(registry)
        return built

    yield build
    for client in clients:
        await client.aclose()


def _served(registry: AgentRegistry) -> dict:
    return {name: endpoint_data(agent) for name, agent in registry.agents.items()}


class TestSharedView:
    async def test_a_registration_is_served_by_another_process_on_refresh(
        self, registries
    ):
        first, second = await registries(2)
        await second.refresh()

        await first.add_registration(EXTERNAL)

        assert second.get_agent("external_agent") is None
        await second.refresh()
        assert _served(second) == {
            "search_agent": endpoint_data(CONFIGURED),
            "external_agent": endpoint_data(EXTERNAL),
        }
        assert second.capabilities == {
            "search": ["search_agent", "external_agent"],
            "video_search": ["search_agent"],
            "summarization": ["external_agent"],
        }
        assert second.list_agents() == ["search_agent", "external_agent"]

    async def test_a_removal_is_served_by_another_process(self, registries):
        first, second = await registries(2)
        await first.add_registration(EXTERNAL)

        assert await second.remove_registration("external_agent") is True
        await first.refresh()

        assert _served(first) == {"search_agent": endpoint_data(CONFIGURED)}
        assert first.capabilities == {
            "search": ["search_agent"],
            "video_search": ["search_agent"],
        }
        assert await first.remove_registration("external_agent") is False

    async def test_a_removed_configured_agent_stays_removed_for_a_new_process(
        self, registries
    ):
        first, second = await registries(2)

        assert await first.remove_registration("search_agent") is True
        await second.refresh()
        (restarted,) = await registries(1)
        await restarted.refresh()

        assert _served(second) == {}
        assert _served(restarted) == {}
        assert await restarted.remove_registration("search_agent") is False
        moved = AgentEndpoint(
            name="search_agent", url="http://moved:8000", capabilities=["search"]
        )
        await restarted.add_registration(moved)
        await second.refresh()
        assert _served(second) == {"search_agent": endpoint_data(moved)}

    async def test_a_registration_overrides_a_configured_agent(self, registries):
        first, second = await registries(2)
        override = AgentEndpoint(
            name="search_agent",
            url="http://override:8001",
            capabilities=["image_search"],
            streams_answer_tokens=True,
        )

        await first.add_registration(override)
        await second.refresh()

        assert _served(second) == {"search_agent": endpoint_data(override)}
        assert second.capabilities == {"image_search": ["search_agent"]}
        assert await second.remove_registration("search_agent") is True
        await first.refresh()
        assert _served(first) == {}

    async def test_an_unchanged_registration_keeps_its_observed_health(
        self, registries
    ):
        first, second = await registries(2)
        await first.add_registration(EXTERNAL)
        await second.refresh()
        observed = second.get_agent("external_agent")
        observed.health_status = "healthy"

        await first.add_registration(
            AgentEndpoint(name="other_agent", url="http://other:1", capabilities=[])
        )
        await second.refresh()

        assert second.get_agent("external_agent") is observed
        assert second.get_agent("external_agent").health_status == "healthy"
        assert second.list_agents() == ["search_agent", "external_agent", "other_agent"]

    async def test_a_wiped_store_is_served_as_a_new_epoch(
        self, registries, shared_state_redis, prefix
    ):
        first, second = await registries(2)
        await first.add_registration(EXTERNAL)
        await second.refresh()
        assert second.list_agents() == ["search_agent", "external_agent"]

        await shared_state_redis.delete(f"{prefix}:entries", f"{prefix}:meta")
        await first.add_registration(
            AgentEndpoint(name="other_agent", url="http://other:1", capabilities=[])
        )
        await second.refresh()

        assert second.list_agents() == ["search_agent", "other_agent"]

    async def test_a_registry_without_a_store_refuses_registrations(
        self, config_manager
    ):
        registry = AgentRegistry(tenant_id="test:unit", config_manager=config_manager)

        with pytest.raises(AgentRegistryUnavailableError) as refused:
            await registry.add_registration(EXTERNAL)
        await registry.refresh()

        assert str(refused.value) == (
            "AgentRegistry has no shared store; registrations need one"
        )
        assert registry.list_agents() == []

    async def test_a_registration_without_url_is_refused(self, registries):
        (registry,) = await registries(1)

        with pytest.raises(ValueError) as refused:
            await registry.add_registration(
                AgentEndpoint(name="no_url", url="", capabilities=[])
            )

        assert str(refused.value) == "Agent must have name and URL"
        await registry.refresh()
        assert registry.list_agents() == ["search_agent"]


class TestConcurrentProcesses:
    PROCESSES = 16

    async def test_concurrent_registrations_are_all_served(
        self, registries, shared_state_redis, prefix
    ):
        built = await registries(self.PROCESSES)
        barrier = asyncio.Barrier(self.PROCESSES)

        async def register(index, registry):
            await barrier.wait()
            await registry.add_registration(
                AgentEndpoint(
                    name=f"agent_{index:02d}",
                    url=f"http://agent-{index}:9000",
                    capabilities=["search"],
                )
            )

        await asyncio.gather(*(register(i, r) for i, r in enumerate(built)))
        for registry in built:
            await registry.refresh()

        expected = ["search_agent"] + [f"agent_{i:02d}" for i in range(self.PROCESSES)]
        assert [sorted(r.list_agents()) for r in built] == [sorted(expected)] * len(
            built
        )
        assert await shared_state_redis.hget(f"{prefix}:meta", "counter") == str(
            self.PROCESSES
        )

    async def test_concurrent_removals_report_exactly_one(self, registries):
        built = await registries(self.PROCESSES)
        await built[0].add_registration(EXTERNAL)
        barrier = asyncio.Barrier(self.PROCESSES)

        async def remove(registry):
            await barrier.wait()
            return await registry.remove_registration("external_agent")

        answers = await asyncio.gather(*(remove(r) for r in built))

        assert sorted(answers) == [False] * (self.PROCESSES - 1) + [True]
        assert {tuple(r.list_agents()) for r in built} == {("search_agent",)}

    async def test_concurrent_registrations_of_one_name_converge(self, registries):
        built = await registries(self.PROCESSES)
        barrier = asyncio.Barrier(self.PROCESSES)

        async def register(index, registry):
            await barrier.wait()
            await registry.add_registration(
                AgentEndpoint(
                    name="contended", url=f"http://agent-{index}:9000", capabilities=[]
                )
            )

        await asyncio.gather(*(register(i, r) for i, r in enumerate(built)))
        for registry in built:
            await registry.refresh()

        urls = {r.get_agent("contended").url for r in built}
        assert len(urls) == 1
        assert urls <= {f"http://agent-{i}:9000" for i in range(self.PROCESSES)}


class TestStoreFailures:
    async def test_every_store_operation_raises_when_redis_is_unreachable(
        self, dead_redis_url, prefix, config_manager
    ):
        client = Redis.from_url(
            dead_redis_url, decode_responses=True, socket_connect_timeout=1
        )
        registry = AgentRegistry(
            tenant_id="test:unit",
            config_manager=config_manager,
            store=RedisAgentRegistryStore(client, key_prefix=prefix),
        )
        registry.register_agent(CONFIGURED)
        operations = {
            "read version": registry.refresh,
            "register agent external_agent": lambda: registry.add_registration(
                EXTERNAL
            ),
            "unregister agent search_agent": lambda: registry.remove_registration(
                "search_agent"
            ),
        }
        try:
            for operation, call in operations.items():
                with pytest.raises(AgentRegistryUnavailableError) as failed:
                    await call()
                assert str(failed.value) == (
                    f"shared agent registry unavailable: {operation}"
                )
        finally:
            await client.aclose()
        assert _served(registry) == {"search_agent": endpoint_data(CONFIGURED)}

    async def test_a_hung_store_fails_the_refresh_and_keeps_the_served_view(
        self, own_redis, registries
    ):
        url, pause, _ = own_redis
        (registry,) = await registries(1, url=url, timeout=1.5)
        await registry.add_registration(EXTERNAL)
        pause()
        started = time.monotonic()

        with pytest.raises(AgentRegistryUnavailableError) as failed:
            await registry.refresh()
        elapsed = time.monotonic() - started

        assert str(failed.value) == "shared agent registry unavailable: read version"
        assert 1.5 <= elapsed < 4.0, elapsed
        assert registry.list_agents() == ["search_agent", "external_agent"]


def _app(registry: AgentRegistry) -> httpx.AsyncClient:
    """The agents router of one process, serving ``registry``."""
    from cogniverse_runtime.routers import agents as agents_router

    app = FastAPI()
    app.include_router(agents_router.router, prefix="/agents")

    @app.middleware("http")
    async def bind_registry(request, call_next):
        agents_router.set_agent_registry(registry)
        return await call_next(request)

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://runtime"
    )


@pytest.fixture
def restore_agents_router():
    from cogniverse_runtime.routers import agents as agents_router

    saved = (agents_router._agent_registry, agents_router._dispatcher)
    yield
    agents_router._agent_registry, agents_router._dispatcher = saved


@pytest.mark.usefixtures("restore_agents_router")
class TestRoutes:
    """Two routers, each bound to its own process's registry."""

    @pytest.fixture
    async def processes(self, registries):
        first, second = await registries(2)
        async with _app(first) as one, _app(second) as other:
            yield one, other

    async def test_a_registration_through_one_process_is_served_by_the_other(
        self, processes
    ):
        first, second = processes

        registered = await first.post(
            "/agents/register",
            json={
                "name": "external_agent",
                "url": "http://external:9000",
                "capabilities": ["summarization", "search"],
                "health_endpoint": "/health",
                "process_endpoint": "/tasks/send",
                "timeout": 30,
            },
        )
        info = await second.get("/agents/external_agent")
        card = await second.get("/agents/external_agent/card")
        by_capability = await second.get("/agents/by-capability/search")
        listed = await second.get("/agents/")

        assert (registered.status_code, registered.json()) == (
            201,
            {
                "status": "registered",
                "agent": "external_agent",
                "url": "http://external:9000",
                "capabilities": ["summarization", "search"],
            },
        )
        assert (info.status_code, info.json()) == (
            200,
            {
                "name": "external_agent",
                "url": "http://external:9000",
                "capabilities": ["summarization", "search"],
                "health_status": "unknown",
                "health_endpoint": "/health",
                "process_endpoint": "/tasks/send",
            },
        )
        assert card.json()["endpoints"] == {
            "health": "/health",
            "process": "/tasks/send",
            "info": "/agents/external_agent",
        }
        assert [a["name"] for a in by_capability.json()["agents"]] == [
            "search_agent",
            "external_agent",
        ]
        assert listed.json() == {
            "count": 2,
            "agents": ["search_agent", "external_agent"],
        }

    async def test_a_second_registration_updates_the_first(self, processes):
        first, second = processes
        payload = {"name": "external_agent", "url": "http://external:9000"}

        await first.post(
            "/agents/register", json={**payload, "capabilities": ["search"]}
        )
        await second.post(
            "/agents/register",
            json={**payload, "capabilities": ["search", "summarize"]},
        )
        info = await first.get("/agents/external_agent")

        assert info.json()["capabilities"] == ["search", "summarize"]

    async def test_an_unregistration_through_one_process_holds_on_the_other(
        self, processes
    ):
        first, second = processes
        await first.post(
            "/agents/register",
            json={"name": "external_agent", "url": "http://external:9000"},
        )

        removed = await second.delete("/agents/external_agent")
        again = await first.delete("/agents/external_agent")
        configured = await first.delete("/agents/search_agent")
        listed = await second.get("/agents/")
        configured_info = await second.get("/agents/search_agent")

        assert (removed.status_code, removed.json()) == (
            200,
            {"status": "unregistered", "agent": "external_agent"},
        )
        assert (again.status_code, again.json()) == (
            404,
            {"detail": "Agent 'external_agent' not found"},
        )
        assert (configured.status_code, configured.json()) == (
            200,
            {"status": "unregistered", "agent": "search_agent"},
        )
        assert listed.json() == {"count": 0, "agents": []}
        assert configured_info.status_code == 404

    async def test_an_unreachable_store_answers_503_on_every_registry_route(
        self, dead_redis_url, prefix, config_manager
    ):
        client = Redis.from_url(
            dead_redis_url, decode_responses=True, socket_connect_timeout=1
        )
        registry = AgentRegistry(
            tenant_id="test:unit",
            config_manager=config_manager,
            store=RedisAgentRegistryStore(client, key_prefix=prefix),
        )
        registry.register_agent(CONFIGURED)
        try:
            async with _app(registry) as app:
                answers = {
                    "register": await app.post(
                        "/agents/register",
                        json={"name": "external_agent", "url": "http://external:9000"},
                    ),
                    "list": await app.get("/agents/"),
                    "stats": await app.get("/agents/stats"),
                    "capability": await app.get("/agents/by-capability/search"),
                    "info": await app.get("/agents/search_agent"),
                    "card": await app.get("/agents/search_agent/card"),
                    "unregister": await app.delete("/agents/search_agent"),
                }
        finally:
            await client.aclose()

        detail = {
            "register": "register agent external_agent",
            "unregister": "unregister agent search_agent",
        }
        assert {name: (r.status_code, r.json()) for name, r in answers.items()} == {
            name: (
                503,
                {
                    "detail": "shared agent registry unavailable: "
                    + detail.get(name, "read version")
                },
            )
            for name in answers
        }


class TestRequestPathsReadTheSharedRegistry:
    """Each request that reads the registry applies the store first."""

    async def test_dispatch_finds_an_agent_another_process_registered(
        self, registries, config_manager
    ):
        from cogniverse_runtime.agent_dispatcher import AgentDispatcher

        first, second = await registries(2)
        dispatcher = AgentDispatcher(
            agent_registry=second, config_manager=config_manager, schema_loader=None
        )
        context = {"tenant_id": "acme:acme"}

        with pytest.raises(ValueError) as before:
            await dispatcher.dispatch("external_agent", "find clips", dict(context))
        await first.add_registration(
            AgentEndpoint(
                name="external_agent",
                url="http://external:9000",
                capabilities=["no_execution_path"],
            )
        )
        with pytest.raises(ValueError) as after:
            await dispatcher.dispatch("external_agent", "find clips", dict(context))

        assert str(before.value) == "Agent 'external_agent' not found in registry"
        assert str(after.value) == (
            "Agent 'external_agent' has no supported execution path "
            "(not in AGENT_CLASSES)"
        )

    async def test_process_on_an_unreachable_store_is_503(
        self, dead_redis_url, prefix, config_manager, restore_agents_router
    ):
        from cogniverse_runtime.routers import agents as agents_router

        client = Redis.from_url(
            dead_redis_url, decode_responses=True, socket_connect_timeout=1
        )
        registry = AgentRegistry(
            tenant_id="test:unit",
            config_manager=config_manager,
            store=RedisAgentRegistryStore(client, key_prefix=prefix),
        )
        registry.register_agent(CONFIGURED)
        saved = (agents_router._config_manager, agents_router._schema_loader)
        agents_router.set_agent_dependencies(
            config_manager, FilesystemSchemaLoader(Path("configs/schemas"))
        )
        try:
            async with _app(registry) as app:
                answer = await app.post(
                    "/agents/search_agent/process",
                    json={
                        "agent_name": "search_agent",
                        "query": "find clips",
                        "context": {"tenant_id": "acme:acme"},
                    },
                )
        finally:
            agents_router._config_manager, agents_router._schema_loader = saved
            await client.aclose()

        assert (answer.status_code, answer.json()) == (
            503,
            {"detail": "shared agent registry unavailable: read version"},
        )

    async def test_v1_chat_on_an_unreachable_store_is_503(
        self, dead_redis_url, prefix, config_manager
    ):
        from cogniverse_runtime.agent_dispatcher import AgentDispatcher
        from cogniverse_runtime.routers import openai_compat

        client = Redis.from_url(
            dead_redis_url, decode_responses=True, socket_connect_timeout=1
        )
        registry = AgentRegistry(
            tenant_id="test:unit",
            config_manager=config_manager,
            store=RedisAgentRegistryStore(client, key_prefix=prefix),
        )
        registry.register_agent(CONFIGURED)
        dispatcher = AgentDispatcher(
            agent_registry=registry, config_manager=config_manager, schema_loader=None
        )
        openai_compat.set_dispatcher_provider(lambda: dispatcher)
        openai_compat.set_api_keys({"sk-test": "acme:acme"})
        openai_compat.set_model_map({"cogniverse": "search_agent"})
        app = FastAPI()
        app.include_router(openai_compat.router, prefix="/v1")
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://runtime"
            ) as http:
                answer = await http.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer sk-test"},
                    json={
                        "model": "cogniverse",
                        "messages": [{"role": "user", "content": "find clips"}],
                    },
                )
        finally:
            openai_compat.set_dispatcher_provider(None)
            openai_compat.set_api_keys({})
            openai_compat.set_model_map({})
            await client.aclose()

        assert (answer.status_code, answer.json()) == (
            503,
            {
                "error": {
                    "message": "The agent registry is unavailable "
                    "(AgentRegistryUnavailableError). See server logs for detail.",
                    "type": "server_error",
                    "code": "service_unavailable",
                    "error_type": "AgentRegistryUnavailableError",
                }
            },
        )

    async def test_an_a2a_task_on_an_unreachable_store_fails_naming_it(
        self, dead_redis_url, prefix, config_manager
    ):
        import json
        from types import SimpleNamespace

        from a2a.server.events import EventQueue
        from a2a.types import TaskState
        from a2a.utils import get_message_text

        from cogniverse_runtime.a2a_executor import CogniverseAgentExecutor
        from cogniverse_runtime.agent_dispatcher import AgentDispatcher

        client = Redis.from_url(
            dead_redis_url, decode_responses=True, socket_connect_timeout=1
        )
        registry = AgentRegistry(
            tenant_id="test:unit",
            config_manager=config_manager,
            store=RedisAgentRegistryStore(client, key_prefix=prefix),
        )
        registry.register_agent(CONFIGURED)
        executor = CogniverseAgentExecutor(
            dispatcher=AgentDispatcher(
                agent_registry=registry,
                config_manager=config_manager,
                schema_loader=None,
            )
        )
        context = SimpleNamespace(
            get_user_input=lambda: "find clips",
            metadata={
                "agent_name": "search_agent",
                "tenant_id": "acme:acme",
                "stream": True,
            },
            message=None,
            task_id="task-1",
            context_id="ctx-1",
            current_task=None,
        )
        queue = EventQueue()
        try:
            await executor.execute(context, queue)
            event = await queue.dequeue_event(no_wait=True)
        finally:
            await client.aclose()

        assert (event.task_id, event.context_id, event.final, event.status.state) == (
            "task-1",
            "ctx-1",
            True,
            TaskState.failed,
        )
        assert json.loads(get_message_text(event.status.message)) == {
            "type": "error",
            "agent": "search_agent",
            "error_type": "AgentRegistryUnavailableError",
            "message": "Agent 'search_agent' failed with "
            "AgentRegistryUnavailableError. See runtime logs for detail.",
        }
