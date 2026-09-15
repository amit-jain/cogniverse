"""Schema operations keep unrelated HTTP requests responsive."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.registries.exceptions import BackendDeploymentError
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
    BackendProfileConfig,
    LLMConfig,
    SystemConfig,
)
from cogniverse_runtime.main import build_wiki_manager_factory
from cogniverse_runtime.routers import admin, wiki
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.http_fault_proxy import HTTPFaultProxy

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]


@pytest.fixture
def schema_env(shared_vespa, monkeypatch):
    def upstream(path):
        port = (
            shared_vespa["config_port"]
            if path.startswith(("/application/", "/config/"))
            else shared_vespa["http_port"]
        )
        return f"http://127.0.0.1:{port}"

    with HTTPFaultProxy(upstream) as proxy:
        cm = ConfigManager(
            store=VespaConfigStore(
                backend_url="http://127.0.0.1", backend_port=proxy.port
            )
        )
        cm.set_system_config(
            SystemConfig(backend_url="http://127.0.0.1", backend_port=proxy.port)
        )
        loader = FilesystemSchemaLoader(Path("configs/schemas"))
        tenant = f"prodfixruntime:t{uuid.uuid4().hex[:8]}"
        backend = BackendRegistry.get_instance().get_ingestion_backend(
            "vespa",
            tenant_id=tenant,
            config={
                "backend": {
                    "url": "http://127.0.0.1",
                    "port": proxy.port,
                    "config_port": proxy.port,
                }
            },
            config_manager=cm,
            schema_loader=loader,
        )
        factory = build_wiki_manager_factory(
            lambda: backend,
            SimpleNamespace(
                get_llm_config=lambda: LLMConfig.from_dict(
                    json.loads(Path("configs/config.json").read_text())["llm_config"]
                )
            ),
            cm,
        )
        monkeypatch.setattr(wiki, "_wiki_manager_factory", factory)
        app = FastAPI()
        app.include_router(wiki.router, prefix="/wiki")
        app.include_router(admin.router, prefix="/admin")
        app.dependency_overrides[admin.get_config_manager_dependency] = lambda: cm
        app.dependency_overrides[admin.get_schema_loader_dependency] = lambda: loader

        @app.get("/heartbeat")
        async def heartbeat():
            return {"status": "responsive"}

        yield SimpleNamespace(
            proxy=proxy,
            cm=cm,
            loader=loader,
            tenant=tenant,
            backend=backend,
            factory=factory,
            app=app,
        )
        BackendRegistry._backend_instances.clear()
        BackendRegistry._shared_schema_registry = None


async def assert_responsive(client, task, proxy):
    try:
        assert await asyncio.to_thread(proxy.entered.wait, 10) is True
        heartbeat = await client.get("/heartbeat")
        assert heartbeat.json() == {"status": "responsive"}
        assert proxy.expired.is_set() is False
        assert task.done() is False
    finally:
        proxy.release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_profile_delete_keeps_loop_responsive_and_preserves_config_on_failure(
    schema_env, failure
):
    env = schema_env
    env.backend.schema_registry.deploy_schema(
        tenant_id=env.tenant, base_schema_name="wiki_pages"
    )
    profile = BackendProfileConfig(
        profile_name="wiki_delete",
        type="text",
        schema_name="wiki_pages",
        embedding_model="lightonai/DenseOn",
    )
    env.cm.add_backend_profile(profile, tenant_id=env.tenant)
    env.proxy.arm(
        lambda method, path, body: method == "POST" and "prepareandactivate" in path,
        failure=failure,
    )
    async with AsyncClient(
        transport=ASGITransport(app=env.app), base_url="http://test"
    ) as client:
        task = asyncio.create_task(
            client.delete(
                f"/admin/profiles/wiki_delete?tenant_id={env.tenant}&delete_schema=true"
            )
        )
        try:
            await assert_responsive(client, task, env.proxy)
        finally:
            response = await task
    if failure:
        assert response.status_code == 500
        assert "injected storage refusal" in response.json()["detail"]
        assert (
            env.cm.get_backend_profile("wiki_delete", tenant_id=env.tenant).to_dict()
            == profile.to_dict()
        )
        assert env.backend.schema_exists("wiki_pages", env.tenant) is True
    else:
        assert response.status_code == 200
        assert {
            key: value for key, value in response.json().items() if key != "deleted_at"
        } == {
            "profile_name": "wiki_delete",
            "tenant_id": env.tenant,
            "schema_deleted": True,
        }
        assert env.cm.get_backend_profile("wiki_delete", tenant_id=env.tenant) is None
        assert env.backend.schema_exists("wiki_pages", env.tenant) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_cold_wiki_route_keeps_loop_responsive_and_retries_failed_setup(
    schema_env, failure
):
    env = schema_env
    env.proxy.arm(
        lambda method, path, body: method == "POST" and "prepareandactivate" in path,
        failure=failure,
    )
    async with AsyncClient(
        transport=ASGITransport(app=env.app, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        task = asyncio.create_task(
            client.get(f"/wiki/topic/absent?tenant_id={env.tenant}")
        )
        try:
            await assert_responsive(client, task, env.proxy)
        finally:
            response = await task
        assert response.status_code == (500 if failure else 404)
        if failure:
            retried = await client.get(f"/wiki/topic/absent?tenant_id={env.tenant}")
            assert retried.status_code == 404
            assert retried.json() == {"detail": "Topic 'absent' not found"}
        assert env.backend.schema_exists("wiki_pages", env.tenant) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_cold_wiki_factory_shares_one_manager_across_concurrent_threads(
    schema_env,
    failure,
):
    env = schema_env
    barrier = threading.Barrier(4)
    env.proxy.arm(
        lambda method, path, body: method == "POST" and "prepareandactivate" in path,
        failure=failure,
    )

    def build():
        barrier.wait(timeout=10)
        return env.factory(env.tenant)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(build) for _ in range(4)]
        assert await asyncio.to_thread(env.proxy.entered.wait, 10) is True
        await asyncio.sleep(0.1)
        env.proxy.release.set()
        results = await asyncio.gather(
            *(asyncio.wrap_future(future) for future in futures),
            return_exceptions=True,
        )
    if failure:
        assert [type(result) for result in results] == [BackendDeploymentError] * 4
        # One failed build, shared: every waiter received the owner's exception
        # object, and nothing was cached, so the next call rebuilds.
        assert len({id(result) for result in results}) == 1
        schema = f"wiki_pages_{env.tenant.replace(':', '_')}"
        assert str(results[0]) == (
            f"Backend deployment failed for schema '{schema}': Backend failed to "
            f"deploy schema '{schema}'. The durable definition is retained for "
            "late activation."
        )
        managers = [await asyncio.to_thread(env.factory, env.tenant)]
    else:
        managers = results
    assert len({id(manager) for manager in managers}) == 1
    assert {manager._schema_name for manager in managers} == {
        f"wiki_pages_{env.tenant.replace(':', '_')}"
    }
    assert env.backend.schema_exists("wiki_pages", env.tenant) is True
