"""Modal App factory for exact authenticated vLLM services."""

from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncIterator

import httpx
import modal
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from cogniverse_cli.modal_inference.serving import (
    DEFAULT_API_KEY_ENV,
    build_authenticated_asgi_app,
)

if TYPE_CHECKING:
    from cogniverse_foundation.inference_specs import InferenceServiceSpec

_API_KEY_SECRET = "cogniverse-inference-api-key"
_HF_TOKEN_SECRET = "hf-token"
_HF_CACHE_NAME = "cogniverse-huggingface-cache"
_HF_CACHE_PATH = "/root/.cache/huggingface"
_VLLM_PORT = 8001
_VLLM_VERSION = "0.23.0"

_SERVICE_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "vllm_colpali": (
        "--max-model-len",
        "4096",
        "--runner",
        "pooling",
        "--convert",
        "embed",
        "--limit-mm-per-prompt",
        '{"video":0,"image":1}',
    ),
    "denseon": (
        "--runner",
        "pooling",
        "--convert",
        "embed",
        "--dtype",
        "float32",
    ),
    # Modal serves these on NVIDIA. --enforce-eager and --max-num-seqs 1 are
    # gfx1151 APU constraints (no CUDA graphs, no room to batch in the unified
    # pool) and cost 2-4x at batch-1 decode while stranding the KV cache here.
    "vllm_llm_student": (
        "--max-model-len",
        "8192",
        "--limit-mm-per-prompt",
        '{"video":0,"image":4}',
    ),
    "vllm_llm_teacher": (
        "--max-model-len",
        "4096",
    ),
    "vllm_asr": ("--runner", "generate", "--max-model-len", "448"),
}

_KV_CACHE_GIB = {
    "vllm_colpali": "2",
    "vllm_llm_student": "4",
    "vllm_llm_teacher": "4",
    "vllm_asr": "1",
}


def _vllm_command(
    spec: InferenceServiceSpec,
    *,
    port: int = _VLLM_PORT,
) -> tuple[str, ...]:
    try:
        engine_arguments = _SERVICE_ARGUMENTS[spec.name]
    except KeyError:
        raise ValueError(f"{spec.name}: no canonical vLLM launch contract") from None
    return (
        "vllm",
        "serve",
        spec.model_id,
        "--revision",
        spec.model_revision,
        "--served-model-name",
        spec.model_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        *engine_arguments,
    )


def _vllm_environment(spec: InferenceServiceSpec) -> dict[str, str]:
    if spec.name not in _SERVICE_ARGUMENTS:
        raise ValueError(f"{spec.name}: no canonical vLLM launch contract")
    environment = {
        "HF_HOME": _HF_CACHE_PATH,
        "MALLOC_ARENA_MAX": "2",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        "VLLM_CACHE_ROOT": f"{_HF_CACHE_PATH}/.vllm",
    }
    if cache_size := _KV_CACHE_GIB.get(spec.name):
        environment["VLLM_CPU_KVCACHE_SPACE"] = cache_size
    return environment


def _vllm_secrets(spec: InferenceServiceSpec) -> list[modal.Secret]:
    secrets = [
        modal.Secret.from_name(
            _API_KEY_SECRET,
            required_keys=["COGNIVERSE_INFERENCE_API_KEY"],
        )
    ]
    if spec.requires_hf_token:
        secrets.append(
            modal.Secret.from_name(_HF_TOKEN_SECRET, required_keys=["HF_TOKEN"])
        )
    return secrets


class _ServingProcessError(RuntimeError):
    pass


ProcessLauncher = Callable[[Sequence[str]], subprocess.Popen[bytes]]


def _launch_process(command: Sequence[str]) -> subprocess.Popen[bytes]:
    environment = os.environ.copy()
    environment.pop(DEFAULT_API_KEY_ENV, None)
    return subprocess.Popen(command, env=environment, start_new_session=True)


class _ServingProcess:
    """Start one loopback server per container and retain its crash state."""

    def __init__(
        self,
        *,
        service: str,
        command: Sequence[str],
        host: str,
        port: int,
        startup_timeout: float,
        launcher: ProcessLauncher = _launch_process,
    ) -> None:
        self.service = service
        self.command = tuple(command)
        self.host = host
        self.port = port
        self.startup_timeout = startup_timeout
        self._launcher = launcher
        self._lock = asyncio.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._ready = False

    async def ensure_started(self) -> subprocess.Popen[bytes]:
        async with self._lock:
            if self._ready:
                self._raise_if_exited()
                assert self._process is not None
                return self._process
            if self._process is None:
                try:
                    self._process = await asyncio.to_thread(
                        self._launcher,
                        self.command,
                    )
                except OSError as exc:
                    raise _ServingProcessError(
                        f"{self.service}: failed to start vLLM serving process "
                        f"({type(exc).__name__})"
                    ) from exc

            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.startup_timeout
            while loop.time() < deadline:
                self._raise_if_exited()
                try:
                    _, writer = await asyncio.wait_for(
                        asyncio.open_connection(self.host, self.port),
                        timeout=0.1,
                    )
                except (OSError, TimeoutError):
                    await asyncio.sleep(0.05)
                    continue
                writer.close()
                await writer.wait_closed()
                self._ready = True
                assert self._process is not None
                return self._process

            await self._stop_locked()
            raise _ServingProcessError(
                f"{self.service}: vLLM serving process did not listen on "
                f"{self.host}:{self.port} within {self.startup_timeout:g} seconds"
            )

    def _raise_if_exited(self) -> None:
        if self._process is None:
            return
        status = self._process.poll()
        if status is not None:
            self._process = None
            self._ready = False
            raise _ServingProcessError(
                f"{self.service}: vLLM serving process exited with status {status}"
            )

    async def invalidate(self, expected: subprocess.Popen[bytes]) -> None:
        async with self._lock:
            if self._process is not expected:
                return
            status = expected.poll()
            await self._stop_locked()
            if status is not None:
                raise _ServingProcessError(
                    f"{self.service}: vLLM serving process exited with status {status}"
                )

    async def _stop_locked(self) -> None:
        process = self._process
        self._process = None
        self._ready = False
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            await asyncio.to_thread(process.wait, 5)
        except subprocess.TimeoutExpired:
            process.kill()
            await asyncio.to_thread(process.wait, 5)

    async def close(self) -> None:
        async with self._lock:
            await self._stop_locked()


def _service_unavailable(detail: str) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": detail})


def _build_process_proxy_app(
    process: _ServingProcess,
    *,
    request_timeout: float = 300.0,
) -> FastAPI:
    if request_timeout <= 0:
        raise ValueError("request_timeout must be positive")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await process.close()

    app = FastAPI(lifespan=lifespan)

    @app.api_route(
        "/{path:path}",
        methods=["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"],
    )
    async def proxy(path: str, request: Request):
        try:
            serving_process = await process.ensure_started()
        except _ServingProcessError as exc:
            return _service_unavailable(str(exc))

        url = f"http://{process.host}:{process.port}/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        headers = [
            (name, value)
            for name, value in request.scope["headers"]
            if name.lower()
            not in {b"authorization", b"connection", b"host", b"transfer-encoding"}
        ]
        try:
            body = await request.body()
        except Exception as exc:
            return _service_unavailable(
                f"{process.service}: request body read failed ({type(exc).__name__})"
            )

        client = httpx.AsyncClient(timeout=request_timeout)
        try:
            upstream_request = client.build_request(
                request.method,
                url,
                headers=headers,
                content=body,
            )
            upstream = await client.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            await client.aclose()
            try:
                await process.invalidate(serving_process)
            except _ServingProcessError as process_exc:
                return _service_unavailable(str(process_exc))
            return _service_unavailable(
                f"{process.service}: vLLM serving request failed ({type(exc).__name__})"
            )
        except BaseException:
            await client.aclose()
            raise

        async def response_body():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()

        response = StreamingResponse(
            response_body(),
            status_code=upstream.status_code,
        )
        response.raw_headers = [
            (name, value)
            for name, value in upstream.headers.raw
            if name.lower() not in {b"connection", b"transfer-encoding"}
        ]
        return response

    return app


# The vllm/vllm-openai base ships python3 but no `python` on PATH. Modal's registry
# preamble and every pip_install layer invoke `python`, so the link has to exist before
# either runs; setup_dockerfile_commands is the only hook that early.
_PYTHON_SHIM_COMMAND = 'RUN ln -sf "$(command -v python3)" /usr/local/bin/python'


def _vllm_image(spec: InferenceServiceSpec) -> modal.Image:
    packages = ["fastapi==0.135.3", "httpx==0.28.1"]
    if spec.name == "vllm_asr":
        packages.extend(["librosa==0.11.0", "soundfile==0.13.1"])
    return (
        modal.Image.from_registry(
            f"vllm/vllm-openai:v{_VLLM_VERSION}",
            setup_dockerfile_commands=[_PYTHON_SHIM_COMMAND],
        )
        .entrypoint([])
        .pip_install(*packages)
        .add_local_python_source(
            "cogniverse_cli.modal_inference",
            "cogniverse_foundation.inference_specs",
            copy=True,
        )
        .env(_vllm_environment(spec))
    )


def build_vllm_app(spec: InferenceServiceSpec) -> modal.App:
    """Declare one scale-to-zero Modal App for a pinned vLLM service."""

    command = _vllm_command(spec)
    image = _vllm_image(spec)
    volume = modal.Volume.from_name(_HF_CACHE_NAME, create_if_missing=True)

    app = modal.App(spec.modal_app)

    @app.function(
        image=image,
        gpu=list(spec.gpu_candidates),
        volumes={_HF_CACHE_PATH: volume},
        secrets=_vllm_secrets(spec),
        min_containers=spec.min_containers,
        scaledown_window=spec.scaledown_window,
        timeout=900,
        startup_timeout=900,
        serialized=True,
        name=spec.modal_object,
    )
    @modal.concurrent(max_inputs=100)
    @modal.asgi_app()
    def inference() -> FastAPI:
        process = _ServingProcess(
            service=spec.name,
            command=command,
            host="127.0.0.1",
            port=_VLLM_PORT,
            startup_timeout=spec.boot_deadline_seconds,
        )
        return build_authenticated_asgi_app(
            _build_process_proxy_app(process),
            model_id=spec.model_id,
            model_revision=spec.model_revision,
        )

    return app
