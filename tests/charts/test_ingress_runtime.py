"""Exercise rendered ingress paths through nginx and Traefik to the runtime."""

from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import httpx
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CHART = REPO / "charts/cogniverse"
TENANT = "prodfixclients:ingress"


def _documents(profile, *settings):
    command = ["helm", "template", "cogniverse", str(CHART), "-f", str(CHART / profile)]
    for setting in (
        "runtime.qualityMonitor.tenantId=ingress-test",
        "minio.rootPassword=ingress-test-password",
        "phoenix.postgres.auth.password=ingress-test-password",
        "redis.auth.password=ingress-test-password",
        "openshell.server.sshHandshakeSecret=ingress-test-secret",
        *settings,
    ):
        command.extend(["--set", setting])
    rendered = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert rendered.returncode == 0, rendered.stderr
    return list(yaml.safe_load_all(rendered.stdout))


def _render(profile, *settings):
    documents = _documents(profile, *settings)
    ingress = next(doc for doc in documents if doc and doc["kind"] == "Ingress")
    return ingress, _container_env(documents, "runtime")


def _container_env(documents, name):
    deployment = next(
        doc
        for doc in documents
        if doc
        and doc["kind"] == "Deployment"
        and doc["metadata"]["name"] == f"cogniverse-{name}"
    )
    container = next(
        item
        for item in deployment["spec"]["template"]["spec"]["containers"]
        if item["name"] == name
    )
    return {item["name"]: item.get("value") for item in container["env"]}


@pytest.mark.parametrize("profile", ["values.prod.yaml", "values.k3s.yaml"])
def test_runtime_mount_matches_ingress_prefix(profile):
    ingress, env = _render(profile)
    paths = ingress["spec"]["rules"][0]["http"]["paths"]
    assert [(item["path"], item["pathType"]) for item in paths] == [
        ("/api", "Prefix"),
        ("/", "Prefix"),
    ]
    assert env.get("COGNIVERSE_ROOT_PATH") == "/api"


def test_runtime_mount_follows_configured_ingress_prefix():
    _, env = _render("values.k3s.yaml", "ingress.hosts[0].paths[0].path=/service")
    assert env.get("COGNIVERSE_ROOT_PATH") == "/service"


def test_runtime_mount_rejects_multiple_public_prefixes():
    with pytest.raises(
        AssertionError, match="runtime ingress paths must share one prefix"
    ):
        _render(
            "values.k3s.yaml",
            "ingress.hosts[1].host=another.local",
            "ingress.hosts[1].paths[0].path=/different",
            "ingress.hosts[1].paths[0].pathType=Prefix",
            "ingress.hosts[1].paths[0].service=runtime",
            "ingress.hosts[1].paths[0].port=8000",
        )


def test_runtime_mount_rejects_environment_override():
    with pytest.raises(
        AssertionError, match="COGNIVERSE_ROOT_PATH is derived from ingress paths"
    ):
        _render("values.k3s.yaml", "runtime.env.COGNIVERSE_ROOT_PATH=/different")


@pytest.mark.parametrize("root_path", ["/api", ""])
def test_runtime_mount_routes_with_asgi_transport(root_path):
    env = dict(os.environ, COGNIVERSE_ROOT_PATH=root_path)
    result = subprocess.run(
        [sys.executable, "-m", "tests.charts.test_ingress_runtime", "asgi"],
        env=env,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("profile", ["values.prod.yaml", "values.k3s.yaml"])
def test_in_cluster_callers_address_the_runtime_without_the_public_prefix(profile):
    documents = _documents(profile)
    assert _container_env(documents, "runtime")["COGNIVERSE_ROOT_PATH"] == "/api"
    assert (
        _container_env(documents, "dashboard")["RUNTIME_URL"]
        == "http://cogniverse-runtime:8000"
    )


def test_one_runtime_process_serves_the_public_prefix_and_the_bare_path(tmp_path):
    """Both entry points resolve every router and every mounted sub-app.

    The ingress forwards ``/api/...`` unrewritten while the dashboard, the CLI
    and the e2e suite reach the Service on the bare path, so the same process
    answers both shapes of every route.
    """
    port = _free_port()
    env = dict(os.environ, COGNIVERSE_ROOT_PATH="/api")
    env.pop("REDIS_URL", None)
    with _process(
        [sys.executable, "-m", "tests.charts.test_ingress_runtime", str(port)],
        tmp_path / "runtime.log",
        env,
    ) as runtime:
        _wait_http(f"http://127.0.0.1:{port}/api/health/live", runtime)
        base = f"http://127.0.0.1:{port}"
        headers = {"Authorization": "Bearer ingress-test-key"}
        for prefix in ("/api", ""):
            live = httpx.get(f"{base}{prefix}/health/live", timeout=10)
            assert (prefix, live.status_code) == (prefix, 200)
            assert live.json() == {"status": "alive"}

            models = httpx.get(f"{base}{prefix}/v1/models", headers=headers, timeout=10)
            assert (prefix, models.status_code) == (prefix, 200)
            assert [item["id"] for item in models.json()["data"]] == [
                "cogniverse/search"
            ]

            card = httpx.get(
                f"{base}{prefix}/a2a/.well-known/agent-card.json", timeout=10
            )
            assert (prefix, card.status_code) == (prefix, 200)
            assert card.json()["name"] == "Cogniverse Runtime"
            assert [skill["id"] for skill in card.json()["skills"]] == ["search_agent"]

            rpc = httpx.post(
                f"{base}{prefix}/a2a/",
                json={
                    "jsonrpc": "2.0",
                    "id": "1",
                    "method": "nosuch/method",
                    "params": {},
                },
                timeout=10,
            )
            assert (prefix, rpc.status_code) == (prefix, 200)
            assert rpc.json() == {
                "jsonrpc": "2.0",
                "id": "1",
                "error": {"code": -32601, "message": "Method not found"},
            }

            index = httpx.get(f"{base}{prefix}/", timeout=10)
            assert index.json()["docs"] == f"{prefix}/docs"
            docs = httpx.get(f"{base}{prefix}/docs", timeout=10)
            assert (prefix, docs.status_code) == (prefix, 200)
            assert f"url: '{prefix}/openapi.json'," in docs.text


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def _process(command, log_path, env=None):
    with log_path.open("w") as output:
        process = subprocess.Popen(
            command, cwd=REPO, env=env, stdout=output, stderr=subprocess.STDOUT
        )
        try:
            yield process
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            print(f"\n{log_path.name}:\n{log_path.read_text()}")


def _wait_http(url, process, headers=None):
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(f"Server exited with {process.returncode}: {url}")
        try:
            response = httpx.get(url, headers=headers, timeout=1)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise AssertionError(f"Server did not become ready: {url}")


def _proxy_config(ingress, ports, listen_port):
    rule = ingress["spec"]["rules"][0]
    paths = rule["http"]["paths"]
    if ingress["spec"]["ingressClassName"] == "nginx":
        locations = []
        for path in paths:
            service = path["backend"]["service"]["name"]
            # Kubernetes Prefix matches a complete path element.
            prefix = path["path"].rstrip("/")
            match = f"~ ^{re.escape(prefix)}(?:/|$)" if prefix else "/"
            locations.append(
                f"location {match} {{ proxy_pass http://127.0.0.1:{ports[service]}; "
                "proxy_http_version 1.1; proxy_set_header Host $host; "
                "proxy_set_header Upgrade $http_upgrade; "
                'proxy_set_header Connection "upgrade"; }'
            )
        return (
            "events {}\nhttp { server { "
            f"listen {listen_port}; server_name {rule['host']}; "
            + "\n".join(locations)
            + " } }\n"
        )
    routers = {}
    services = {}
    for index, path in enumerate(paths):
        name = path["backend"]["service"]["name"]
        prefix = path["path"].rstrip("/")
        path_rule = (
            f"(Path(`{prefix}`) || PathPrefix(`{prefix}/`))"
            if prefix
            else "PathPrefix(`/`)"
        )
        routers[str(index)] = {
            "rule": f"Host(`{rule['host']}`) && {path_rule}",
            "service": name,
            "entryPoints": ["web"],
            "priority": len(prefix) + 1,
        }
        services[name] = {
            "loadBalancer": {"servers": [{"url": f"http://127.0.0.1:{ports[name]}"}]}
        }
    return yaml.safe_dump({"http": {"routers": routers, "services": services}})


@pytest.fixture(params=["values.prod.yaml", "values.k3s.yaml"])
def ingress_stack(request, tmp_path):
    ingress, env = _render(request.param)
    runtime_port, dashboard_port, proxy_port = (_free_port() for _ in range(3))
    rule = ingress["spec"]["rules"][0]
    controller = ingress["spec"]["ingressClassName"]
    config = tmp_path / ("nginx.conf" if controller == "nginx" else "traefik.yaml")
    config.write_text(
        _proxy_config(
            ingress,
            {
                "cogniverse-runtime": runtime_port,
                "cogniverse-dashboard": dashboard_port,
            },
            proxy_port,
        )
    )
    with ExitStack() as stack:
        runtime_env = dict(os.environ)
        runtime_env["COGNIVERSE_ROOT_PATH"] = env.get("COGNIVERSE_ROOT_PATH", "")
        runtime_env.pop("REDIS_URL", None)
        runtime = stack.enter_context(
            _process(
                [
                    sys.executable,
                    "-m",
                    "tests.charts.test_ingress_runtime",
                    str(runtime_port),
                ],
                tmp_path / "runtime.log",
                runtime_env,
            )
        )
        _wait_http(f"http://127.0.0.1:{runtime_port}/health/live", runtime)
        dashboard = stack.enter_context(
            _process(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    "libs/dashboard/cogniverse_dashboard/app.py",
                    f"--server.port={dashboard_port}",
                    "--server.address=127.0.0.1",
                    "--server.headless=true",
                    "--browser.gatherUsageStats=false",
                ],
                tmp_path / "dashboard.log",
            )
        )
        _wait_http(f"http://127.0.0.1:{dashboard_port}/", dashboard)
        name = f"cogniverse-ingress-test-{os.getpid()}-{proxy_port}"
        container_command = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "--network=host",
            "--memory=128m",
        ]
        if controller == "nginx":
            container_command.extend(
                ["-v", f"{config}:/etc/nginx/nginx.conf:ro", "nginx:1.27-alpine"]
            )
        else:
            container_command.extend(
                [
                    "-v",
                    f"{config}:/etc/traefik/dynamic.yaml:ro",
                    "traefik:v3.5",
                    f"--entrypoints.web.address=127.0.0.1:{proxy_port}",
                    "--providers.file.filename=/etc/traefik/dynamic.yaml",
                ]
            )
        proxy = stack.enter_context(_process(container_command, tmp_path / "proxy.log"))
        try:
            url = f"http://127.0.0.1:{proxy_port}"
            headers = {"Host": rule["host"], "Authorization": "Bearer ingress-test-key"}
            _wait_http(url + "/", proxy, headers)
            yield url, headers, runtime
        finally:
            removed = subprocess.run(
                ["docker", "rm", "-f", name], capture_output=True, text=True, timeout=30
            )
            assert removed.returncode == 0, removed.stderr


@pytest.mark.asyncio
async def test_ingress_serves_runtime_models_docs_and_dashboard(ingress_stack):
    url, headers, _ = ingress_stack
    async with httpx.AsyncClient(base_url=url, headers=headers) as client:
        response = await client.get("/api/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}
        before = int(time.time())
        models = await client.get("/api/v1/models")
        after = int(time.time())
        assert models.status_code == 200
        created = models.json()["data"][0]["created"]
        assert before <= created <= after
        assert models.json() == {
            "object": "list",
            "data": [
                {
                    "id": "cogniverse/search",
                    "object": "model",
                    "created": created,
                    "owned_by": "cogniverse",
                }
            ],
        }
        cancelled = await client.post(
            "/api/events/workflows/roundtrip/cancel",
            json={"reason": "operator request"},
        )
        assert cancelled.status_code == 200
        assert cancelled.json() == {
            "task_id": "roundtrip",
            "cancelled": True,
            "message": "Workflow roundtrip cancellation requested",
        }
        queue = await client.get("/api/events/queues/roundtrip")
        assert queue.json()["is_cancelled"] is True
        docs = await client.get("/api/docs")
        assert docs.status_code == 200
        assert "url: '/api/openapi.json'," in docs.text
        openapi = await client.get("/api/openapi.json")
        assert openapi.status_code == 200
        assert openapi.json()["servers"] == [{"url": "/api"}]
        dashboard = await client.get("/")
        assert dashboard.status_code == 200
        assert "<title>Streamlit</title>" in dashboard.text
        assert '<div id="root"></div>' in dashboard.text


async def _subscriber_count(client, task, expected):
    deadline = time.monotonic() + 10
    count = None
    while time.monotonic() < deadline:
        response = await client.get(f"/api/events/queues/{task}")
        assert response.status_code == 200
        count = response.json()["subscriber_count"]
        if count == expected:
            break
        await asyncio.sleep(0.02)
    assert count == expected


@pytest.mark.asyncio
async def test_ingress_disconnect_cancels_only_its_concurrent_subscription(
    ingress_stack,
):
    url, headers, _ = ingress_stack
    ready = asyncio.Barrier(3)
    close_left, close_right = asyncio.Event(), asyncio.Event()

    async def subscribe(task, close):
        async with httpx.AsyncClient(
            base_url=url, headers=headers, timeout=10
        ) as client:
            async with client.stream(
                "GET", f"/api/events/workflows/{task}"
            ) as response:
                assert response.status_code == 200
                lines = response.aiter_lines()
                line = await anext(lines)
                connected = json.loads(line.removeprefix("data: "))
                assert {
                    key: connected[key] for key in ("type", "task_id", "offset")
                } == {
                    "type": "connected",
                    "task_id": task,
                    "offset": 0,
                }
                await ready.wait()
                await close.wait()

    left = asyncio.create_task(subscribe("left", close_left))
    right = asyncio.create_task(subscribe("right", close_right))
    try:
        await asyncio.wait_for(ready.wait(), timeout=15)
        async with httpx.AsyncClient(base_url=url, headers=headers) as client:
            await _subscriber_count(client, "left", 1)
            await _subscriber_count(client, "right", 1)
            close_left.set()
            await left
            await _subscriber_count(client, "left", 0)
            await _subscriber_count(client, "right", 1)
            response = await client.get("/api/health/live")
            assert response.json() == {"status": "alive"}
            close_right.set()
            await right
            await _subscriber_count(client, "right", 0)
    finally:
        close_left.set()
        close_right.set()
        await asyncio.gather(left, right, return_exceptions=True)


@pytest.mark.asyncio
async def test_ingress_runtime_failure_breaks_stream_and_returns_gateway_error(
    ingress_stack,
):
    url, headers, runtime = ingress_stack
    async with httpx.AsyncClient(base_url=url, headers=headers, timeout=10) as client:
        async with client.stream("GET", "/api/events/workflows/fault") as response:
            assert response.status_code == 200
            lines = response.aiter_lines()
            connected = json.loads((await anext(lines)).removeprefix("data: "))
            assert connected["task_id"] == "fault"
            runtime.kill()
            runtime.wait(timeout=10)
            with pytest.raises(
                httpx.RemoteProtocolError, match="incomplete chunked read"
            ):
                async for _ in lines:
                    pass
        health = await client.get("/api/health/live")
        assert health.status_code == 502
        dashboard = await client.get("/")
        assert dashboard.status_code == 200
        assert "<title>Streamlit</title>" in dashboard.text


async def _serve_runtime(port):
    import uvicorn
    from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill

    from cogniverse_core.events import get_queue_manager
    from cogniverse_runtime.a2a_executor import (
        BoundedInMemoryTaskStore,
        CogniverseAgentExecutor,
    )
    from cogniverse_runtime.main import app
    from cogniverse_runtime.routers.openai_compat import set_api_keys, set_model_map

    set_api_keys({"ingress-test-key": TENANT})
    set_model_map({"cogniverse/search": "search_agent"})
    agent_card = AgentCard(
        name="Cogniverse Runtime",
        description="Multi-agent AI platform for content intelligence",
        url="http://localhost:8000/a2a",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="search_agent",
                name="search_agent",
                description="Agent: search_agent (search)",
                tags=["search"],
            )
        ],
    )
    app.mount(
        "/a2a",
        A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=DefaultRequestHandler(
                agent_executor=CogniverseAgentExecutor(dispatcher=None),
                task_store=BoundedInMemoryTaskStore(),
            ),
        ).build(),
    )
    for task in ("roundtrip", "left", "right", "fault"):
        await get_queue_manager().create_queue(task_id=task, tenant_id=TENANT)
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=port, lifespan="off", log_level="info"
        )
    )
    await server.serve()


async def _asgi_probe():
    from cogniverse_runtime.main import app

    prefix = os.environ["COGNIVERSE_ROOT_PATH"]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://ingress"
    ) as client:
        response = await client.get(f"{prefix}/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}
        root = await client.get(f"{prefix}/")
        assert root.status_code == 200
        assert root.json() == {
            "service": "Cogniverse Runtime",
            "version": "1.0.0",
            "description": "Multi-agent AI platform for content intelligence",
            "docs": f"{prefix}/docs",
            "health": f"{prefix}/health",
        }


if __name__ == "__main__":
    asyncio.run(
        _asgi_probe() if sys.argv[1] == "asgi" else _serve_runtime(int(sys.argv[1]))
    )
