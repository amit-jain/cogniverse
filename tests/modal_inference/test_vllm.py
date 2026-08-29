from __future__ import annotations

import asyncio
import socket
import subprocess
import sys
import time
from collections.abc import Sequence

import httpx
import modal
import pytest
from cogniverse_cli.modal_inference.serving import build_authenticated_asgi_app
from cogniverse_cli.modal_inference.vllm import (
    _VLLM_VERSION,
    _build_process_proxy_app,
    _launch_process,
    _ServingProcess,
    _vllm_command,
    _vllm_environment,
    _vllm_image,
    build_vllm_app,
)
from cogniverse_cli.modal_inference_config import get_inference_service_spec
from fastapi.testclient import TestClient
from starlette.requests import Request

API_KEY = "vllm-serving-key"


def _unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _server_command(port: int) -> tuple[str, ...]:
    script = f"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        payload = json.loads(self.rfile.read(length))
        body = json.dumps({{'input': payload['input'], 'vector': [0.125, -0.75]}}, separators=(',', ':')).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('X-Serving-Process', 'loopback')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return

ThreadingHTTPServer(('127.0.0.1', {port}), Handler).serve_forever()
"""
    return (sys.executable, "-c", script)


def _hung_server_command(port: int) -> tuple[str, ...]:
    script = f"""
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        time.sleep(5)

    def log_message(self, format, *args):
        return

ThreadingHTTPServer(('127.0.0.1', {port}), Handler).serve_forever()
"""
    return (sys.executable, "-c", script)


def _one_connection_then_hang_command(port: int) -> tuple[str, ...]:
    script = f"""
import socket
import time

listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(('127.0.0.1', {port}))
listener.listen()
connection, _ = listener.accept()
connection.close()
listener.close()
time.sleep(30)
"""
    return (sys.executable, "-c", script)


async def _concurrent_proxy_requests(app, count: int) -> list[httpx.Response]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="https://modal.test",
        ) as client:
            return await asyncio.gather(
                *(
                    client.post(
                        "/v1/embeddings",
                        headers={"Authorization": f"Bearer {API_KEY}"},
                        json={"input": f"query-{index}"},
                    )
                    for index in range(count)
                )
            )


async def _measure_event_loop_resume(process: _ServingProcess) -> float:
    started = time.monotonic()
    task = asyncio.create_task(process.ensure_started())
    await asyncio.sleep(0.01)
    elapsed = time.monotonic() - started
    with pytest.raises(
        RuntimeError,
        match="vllm_colpali: failed to start vLLM serving process",
    ):
        await task
    return elapsed


def test_tomoro_launch_contract_matches_the_production_engine():
    spec = get_inference_service_spec("vllm_colpali")

    assert _vllm_command(spec, port=8001) == (
        "vllm",
        "serve",
        "TomoroAI/tomoro-colqwen3-embed-4b",
        "--revision",
        "bf790bd8780b098b86453444632a184bb770be1a",
        "--served-model-name",
        "TomoroAI/tomoro-colqwen3-embed-4b",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
        "--max-model-len",
        "4096",
        "--runner",
        "pooling",
        "--convert",
        "embed",
        "--limit-mm-per-prompt",
        '{"video":0,"image":1}',
    )
    assert _vllm_environment(spec) == {
        "HF_HOME": "/root/.cache/huggingface",
        "MALLOC_ARENA_MAX": "2",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        "VLLM_CACHE_ROOT": "/root/.cache/huggingface/.vllm",
        "VLLM_CPU_KVCACHE_SPACE": "2",
    }


def test_denseon_launch_contract_uses_production_pooling():
    spec = get_inference_service_spec("denseon")

    assert _vllm_command(spec, port=8001) == (
        "vllm",
        "serve",
        "lightonai/DenseOn",
        "--revision",
        "cb9947ebccb33862d24e3c7ca2edb25e51acd887",
        "--served-model-name",
        "lightonai/DenseOn",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
        "--runner",
        "pooling",
        "--convert",
        "embed",
        "--dtype",
        "float32",
    )


def test_loopback_process_receives_model_token_but_not_public_api_key(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "must-not-cross")
    monkeypatch.setenv("HF_TOKEN", "required-model-token")
    command = (
        sys.executable,
        "-c",
        "import os; "
        "assert 'COGNIVERSE_INFERENCE_API_KEY' not in os.environ; "
        "assert os.environ['HF_TOKEN'] == 'required-model-token'",
    )

    process = _launch_process(command)

    assert process.wait(timeout=5) == 0


def test_modal_app_registers_the_canonical_authenticated_inference_function():
    spec = get_inference_service_spec("vllm_colpali")

    app = build_vllm_app(spec)
    function = app.registered_functions["Inference"]

    assert app.name == "cogniverse-vllm-colpali"
    assert app.registered_web_endpoints == ["Inference"]
    assert function.tag == spec.modal_object
    assert function.spec.gpus == ["L4", "A10", "L40S"]
    assert list(function.spec.volumes) == ["/root/.cache/huggingface"]
    assert repr(function.spec.volumes["/root/.cache/huggingface"]) == (
        "modal.Volume.from_name('cogniverse-huggingface-cache')"
    )
    assert [repr(secret) for secret in function.spec.secrets] == [
        "modal.Secret.from_name('cogniverse-inference-api-key')"
    ]


@pytest.mark.parametrize(
    ("service", "expected_secrets"),
    [
        (
            "vllm_llm_student",
            [
                "modal.Secret.from_name('cogniverse-inference-api-key')",
                "modal.Secret.from_name('hf-token')",
            ],
        ),
        (
            "vllm_llm_teacher",
            [
                "modal.Secret.from_name('cogniverse-inference-api-key')",
                "modal.Secret.from_name('hf-token')",
            ],
        ),
        (
            "vllm_colpali",
            ["modal.Secret.from_name('cogniverse-inference-api-key')"],
        ),
    ],
)
def test_vllm_secret_selection_follows_requires_hf_token(
    service: str,
    expected_secrets: list[str],
):
    spec = get_inference_service_spec(service)
    app = build_vllm_app(spec)
    function = app.registered_functions["Inference"]

    assert [repr(secret) for secret in function.spec.secrets] == expected_secrets


def test_registered_modal_function_builds_the_authenticated_asgi_app(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
    spec = get_inference_service_spec("vllm_colpali")
    app = build_vllm_app(spec)

    inference_app = app.registered_functions["Inference"].get_raw_f()()

    with TestClient(inference_app) as client:
        unauthorized = client.get("/v1/models")
        identity = client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
    assert unauthorized.status_code == 401
    assert unauthorized.json() == {"detail": "Bearer authorization required"}
    assert identity.status_code == 200
    assert identity.json() == {
        "data": [
            {
                "created": 0,
                "id": spec.model_id,
                "object": "model",
                "owned_by": "cogniverse",
                "revision": spec.model_revision,
            }
        ],
        "object": "list",
    }


def test_startup_timeout_terminates_unreachable_process_before_retry():
    port = _unused_port()
    launches: list[subprocess.Popen[bytes]] = []

    def launch(_: Sequence[str]) -> subprocess.Popen[bytes]:
        command = (
            (sys.executable, "-c", "import time; time.sleep(30)")
            if not launches
            else _server_command(port)
        )
        child = subprocess.Popen(command)
        launches.append(child)
        return child

    process = _ServingProcess(
        service="vllm_colpali",
        command=("unused",),
        host="127.0.0.1",
        port=port,
        startup_timeout=0.1,
        launcher=launch,
    )

    async def timeout_then_retry():
        try:
            with pytest.raises(
                RuntimeError,
                match="did not listen .* within 0.1 seconds",
            ):
                await process.ensure_started()
            assert launches[0].poll() is not None
            # The 0.1s budget exists to force the timeout above. The relaunch
            # has to spawn an interpreter and bind a socket, which does not
            # fit that budget on a loaded host, so give the retry a real one.
            process.startup_timeout = 30
            await process.ensure_started()
        finally:
            await process.close()

    asyncio.run(timeout_then_retry())

    assert len(launches) == 2
    assert all(child.poll() is not None for child in launches)


def test_alive_but_unreachable_process_is_replaced_once(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
    spec = get_inference_service_spec("vllm_colpali")
    port = _unused_port()
    launches: list[subprocess.Popen[bytes]] = []

    def launch(_: Sequence[str]) -> subprocess.Popen[bytes]:
        command = (
            _one_connection_then_hang_command(port)
            if not launches
            else _server_command(port)
        )
        child = subprocess.Popen(command)
        launches.append(child)
        return child

    process = _ServingProcess(
        service=spec.name,
        command=("unused",),
        host="127.0.0.1",
        port=port,
        startup_timeout=2,
        launcher=launch,
    )
    app = build_authenticated_asgi_app(
        _build_process_proxy_app(process),
        model_id=spec.model_id,
        model_revision=spec.model_revision,
    )

    async def fail_then_recover():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="https://modal.test",
            ) as client:
                headers = {"Authorization": f"Bearer {API_KEY}"}
                failed = await asyncio.gather(
                    *(
                        client.post(
                            "/v1/embeddings",
                            headers=headers,
                            json={"input": f"failed-{index}"},
                        )
                        for index in range(8)
                    )
                )
                recovered = await asyncio.gather(
                    *(
                        client.post(
                            "/v1/embeddings",
                            headers=headers,
                            json={"input": f"recovered-{index}"},
                        )
                        for index in range(8)
                    )
                )
                return failed, recovered

    failed, recovered = asyncio.run(fail_then_recover())

    assert tuple(response.status_code for response in failed) == (503,) * 8
    assert (
        tuple(response.json() for response in failed)
        == ({"detail": "vllm_colpali: vLLM serving request failed (ConnectError)"},) * 8
    )
    assert tuple(response.status_code for response in recovered) == (200,) * 8
    assert tuple(response.json() for response in recovered) == tuple(
        {
            "input": f"recovered-{index}",
            "vector": [0.125, -0.75],
        }
        for index in range(8)
    )
    assert len(launches) == 2
    assert all(child.poll() is not None for child in launches)


def test_concurrent_cold_requests_start_one_real_serving_process(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
    spec = get_inference_service_spec("vllm_colpali")
    port = _unused_port()
    launches: list[tuple[str, ...]] = []

    def launch(command: Sequence[str]) -> subprocess.Popen[bytes]:
        launches.append(tuple(command))
        return subprocess.Popen(command)

    process = _ServingProcess(
        service=spec.name,
        command=_server_command(port),
        host="127.0.0.1",
        port=port,
        startup_timeout=5,
        launcher=launch,
    )
    app = build_authenticated_asgi_app(
        _build_process_proxy_app(process),
        model_id=spec.model_id,
        model_revision=spec.model_revision,
    )

    responses = asyncio.run(_concurrent_proxy_requests(app, 12))

    assert launches == [_server_command(port)]
    assert [response.status_code for response in responses] == [200] * 12
    assert [response.json() for response in responses] == [
        {"input": f"query-{index}", "vector": [0.125, -0.75]} for index in range(12)
    ]
    assert [response.headers["x-serving-process"] for response in responses] == [
        "loopback"
    ] * 12


def test_cold_process_launch_does_not_block_the_event_loop():
    def blocking_launcher(_: Sequence[str]) -> subprocess.Popen[bytes]:
        time.sleep(0.35)
        raise OSError("controlled launch failure")

    process = _ServingProcess(
        service="vllm_colpali",
        command=("unused",),
        host="127.0.0.1",
        port=_unused_port(),
        startup_timeout=1,
        launcher=blocking_launcher,
    )

    elapsed = asyncio.run(_measure_event_loop_resume(process))

    assert elapsed < 0.2


def test_proxy_reports_request_body_read_failure_before_opening_upstream_client(
    monkeypatch,
):
    class ReadyProcess:
        service = "vllm_colpali"
        host = "127.0.0.1"
        port = 39001

        async def ensure_started(self) -> None:
            return None

        async def close(self) -> None:
            return None

    client_constructions = 0

    def unexpected_client(*args, **kwargs):
        nonlocal client_constructions
        client_constructions += 1
        raise AssertionError("upstream client opened before request body was read")

    async def failed_receive():
        raise OSError("controlled request body failure")

    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/v1/embeddings",
        "raw_path": b"/v1/embeddings",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("modal.test", 443),
    }
    monkeypatch.setattr(
        "cogniverse_cli.modal_inference.vllm.httpx.AsyncClient", unexpected_client
    )
    app = _build_process_proxy_app(ReadyProcess())
    endpoint = next(
        route.endpoint for route in app.routes if route.path == "/{path:path}"
    )

    response = asyncio.run(
        endpoint("v1/embeddings", Request(scope, receive=failed_receive))
    )

    assert response.status_code == 503
    assert response.body == (
        b'{"detail":"vllm_colpali: request body read failed (OSError)"}'
    )
    assert client_constructions == 0


def test_proxy_bounds_hung_real_upstream_request(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
    spec = get_inference_service_spec("vllm_colpali")
    port = _unused_port()
    process = _ServingProcess(
        service=spec.name,
        command=_hung_server_command(port),
        host="127.0.0.1",
        port=port,
        startup_timeout=2,
    )
    app = build_authenticated_asgi_app(
        _build_process_proxy_app(process, request_timeout=0.1),
        model_id=spec.model_id,
        model_revision=spec.model_revision,
    )

    started = time.monotonic()
    response = asyncio.run(_concurrent_proxy_requests(app, 1))[0]
    elapsed = time.monotonic() - started

    assert response.status_code == 503
    assert response.content == (
        b'{"detail":"vllm_colpali: vLLM serving request failed (ReadTimeout)"}'
    )
    assert elapsed < 2


def test_serving_process_crash_is_contextual_then_recovers_once_for_concurrent_calls(
    monkeypatch,
):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
    spec = get_inference_service_spec("vllm_colpali")
    port = _unused_port()
    launches: list[tuple[str, ...]] = []

    def launch(command: Sequence[str]) -> subprocess.Popen[bytes]:
        launches.append(tuple(command))
        if len(launches) == 1:
            return subprocess.Popen((sys.executable, "-c", "raise SystemExit(23)"))
        return subprocess.Popen(command)

    process = _ServingProcess(
        service=spec.name,
        command=_server_command(port),
        host="127.0.0.1",
        port=port,
        startup_timeout=2,
        launcher=launch,
    )
    app = build_authenticated_asgi_app(
        _build_process_proxy_app(process),
        model_id=spec.model_id,
        model_revision=spec.model_revision,
    )

    async def crash_then_recover():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="https://modal.test",
            ) as client:
                headers = {"Authorization": f"Bearer {API_KEY}"}
                first = await client.post(
                    "/v1/embeddings",
                    headers=headers,
                    json={"input": "first"},
                )
                recovered = await asyncio.gather(
                    *(
                        client.post(
                            "/v1/embeddings",
                            headers=headers,
                            json={"input": f"recovered-{index}"},
                        )
                        for index in range(8)
                    )
                )
                return first, recovered

    response, recovered = asyncio.run(crash_then_recover())

    assert response.status_code == 503
    assert response.content == (
        b'{"detail":"vllm_colpali: vLLM serving process exited with status 23"}'
    )
    assert API_KEY not in response.text
    assert launches == [_server_command(port), _server_command(port)]
    assert tuple(result.status_code for result in recovered) == (200,) * 8
    assert tuple(result.json() for result in recovered) == tuple(
        {"input": f"recovered-{index}", "vector": [0.125, -0.75]} for index in range(8)
    )


class TestVllmImagePythonShim:
    """The vllm/vllm-openai base ships python3 but no `python` on PATH.

    Modal requires `python` for its own registry preamble and for every
    ``pip_install`` layer, so the shim has to run before either of them.
    ``setup_dockerfile_commands`` is the only hook that runs that early.
    """

    def _record_from_registry(self, monkeypatch) -> dict:
        recorded: dict = {}
        original = modal.Image.from_registry

        def _record(tag, **kwargs):
            recorded["tag"] = tag
            recorded["setup_dockerfile_commands"] = tuple(
                kwargs.get("setup_dockerfile_commands") or ()
            )
            recorded["add_python"] = kwargs.get("add_python")
            return original(tag, **kwargs)

        monkeypatch.setattr(modal.Image, "from_registry", staticmethod(_record))
        return recorded

    def test_base_image_gets_a_python_shim_before_any_python_command(self, monkeypatch):
        recorded = self._record_from_registry(monkeypatch)

        _vllm_image(get_inference_service_spec("vllm_llm_student"))

        assert recorded["setup_dockerfile_commands"] == (
            'RUN ln -sf "$(command -v python3)" /usr/local/bin/python',
        )

    def test_shim_is_used_instead_of_a_second_python_installation(self, monkeypatch):
        recorded = self._record_from_registry(monkeypatch)

        _vllm_image(get_inference_service_spec("vllm_llm_student"))

        assert recorded["add_python"] is None
        assert recorded["tag"] == f"vllm/vllm-openai:v{_VLLM_VERSION}"

    def test_every_vllm_service_gets_the_same_shim(self, monkeypatch):
        for service in ("vllm_llm_student", "vllm_llm_teacher", "vllm_asr"):
            recorded = self._record_from_registry(monkeypatch)
            _vllm_image(get_inference_service_spec(service))
            assert recorded["setup_dockerfile_commands"] == (
                'RUN ln -sf "$(command -v python3)" /usr/local/bin/python',
            ), service
