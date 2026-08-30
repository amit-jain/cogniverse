from __future__ import annotations

import asyncio
import socket
import threading
import time
from contextlib import asynccontextmanager, contextmanager

import httpx
import pytest
import uvicorn
from cogniverse_cli.inference_endpoints import (
    CandidateEndpoint,
    EndpointCredentials,
    EndpointResolver,
    ResolvedInferenceEndpoint,
)
from cogniverse_cli.modal_inference.serving import build_authenticated_asgi_app
from fastapi import FastAPI, Request

from cogniverse_foundation.inference_specs import get_inference_service_spec

API_KEY = "serving-test-key"
MODEL_ID = "TomoroAI/tomoro-colqwen3-embed-4b"
MODEL_REVISION = "bf790bd8780b098b86453444632a184bb770be1a"


def _production_app(observed: dict[str, object]) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        observed["lifespan"] = "started"
        yield
        observed["lifespan"] = "stopped"

    app = FastAPI(lifespan=lifespan)

    @app.post("/embed")
    async def embed(request: Request) -> dict[str, object]:
        observed["authorization"] = request.headers.get("authorization")
        observed["payload"] = await request.json()
        return {
            "data": [{"embedding": [0.25, -0.5], "index": 0}],
            "model": MODEL_ID,
            "object": "list",
            "production_lifespan": observed["lifespan"],
        }

    return app


async def _request(
    app: FastAPI,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | list[tuple[str, str]] | None = None,
    json: object | None = None,
) -> httpx.Response:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="https://inference.test",
        ) as client:
            return await client.request(method, path, headers=headers, json=json)


def _authenticated_app(monkeypatch, observed: dict[str, object]) -> FastAPI:
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
    return build_authenticated_asgi_app(
        _production_app(observed),
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
    )


@contextmanager
def _live_server(app: FastAPI):
    with socket.socket() as port_reservation:
        port_reservation.bind(("127.0.0.1", 0))
        port = port_reservation.getsockname()[1]

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="error",
            ws="none",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not server.started:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("authenticated inference server did not start")

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("authenticated inference server did not stop")


def test_missing_bearer_key_returns_exact_unauthorized_contract(monkeypatch):
    observed: dict[str, object] = {}
    app = _authenticated_app(monkeypatch, observed)

    response = asyncio.run(_request(app, "POST", "/embed", json={"input": "frame"}))

    assert response.status_code == 401
    assert response.content == b'{"detail":"Bearer authorization required"}'
    assert response.headers["www-authenticate"] == "Bearer"
    assert "payload" not in observed


def test_missing_server_key_prevents_app_construction(monkeypatch):
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)

    with pytest.raises(
        RuntimeError,
        match=(
            "Inference authentication is not configured in COGNIVERSE_INFERENCE_API_KEY"
        ),
    ):
        build_authenticated_asgi_app(
            _production_app({}),
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
        )


def test_wrong_bearer_key_never_reaches_or_leaks_into_production(monkeypatch):
    observed: dict[str, object] = {}
    app = _authenticated_app(monkeypatch, observed)
    wrong_key = "do-not-echo-this-key"

    response = asyncio.run(
        _request(
            app,
            "POST",
            "/embed",
            headers={"Authorization": f"Bearer {wrong_key}"},
            json={"input": "frame"},
        )
    )

    assert response.status_code == 401
    assert response.content == b'{"detail":"Invalid bearer token"}'
    assert wrong_key not in response.text
    assert API_KEY not in response.text
    assert "payload" not in observed


@pytest.mark.parametrize(
    "headers",
    (
        [
            ("Authorization", f"Bearer {API_KEY}"),
            ("Authorization", "Bearer attacker-key"),
        ],
        [
            ("Authorization", "Bearer attacker-key"),
            ("Authorization", f"Bearer {API_KEY}"),
        ],
    ),
)
def test_duplicate_authorization_headers_are_rejected_before_production(
    monkeypatch,
    headers,
):
    observed: dict[str, object] = {}
    app = _authenticated_app(monkeypatch, observed)

    async def send_concurrently() -> tuple[httpx.Response, ...]:
        return tuple(
            await asyncio.gather(
                *(
                    _request(
                        app,
                        "POST",
                        "/embed",
                        headers=headers,
                        json={"input": "frame"},
                    )
                    for _ in range(4)
                )
            )
        )

    responses = asyncio.run(send_concurrently())

    assert tuple(response.status_code for response in responses) == (401,) * 4
    assert (
        tuple(response.content for response in responses)
        == (b'{"detail":"Exactly one Bearer authorization header is required"}',) * 4
    )
    assert (
        tuple(response.headers["www-authenticate"] for response in responses)
        == ("Bearer",) * 4
    )
    assert "payload" not in observed


def test_correct_bearer_key_preserves_the_production_route_contract(monkeypatch):
    observed: dict[str, object] = {}
    app = _authenticated_app(monkeypatch, observed)
    payload = {"input": [{"type": "image_url", "url": "data:image/png;base64,AA=="}]}

    response = asyncio.run(
        _request(
            app,
            "POST",
            "/embed",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json=payload,
        )
    )

    assert response.status_code == 200
    assert response.json() == {
        "data": [{"embedding": [0.25, -0.5], "index": 0}],
        "model": MODEL_ID,
        "object": "list",
        "production_lifespan": "started",
    }
    assert observed == {
        "authorization": None,
        "lifespan": "stopped",
        "payload": payload,
    }


def test_models_route_returns_the_exact_pinned_identity(monkeypatch):
    observed: dict[str, object] = {}
    app = _authenticated_app(monkeypatch, observed)

    response = asyncio.run(
        _request(
            app,
            "GET",
            "/v1/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
    )

    assert response.status_code == 200
    assert response.json() == {
        "data": [
            {
                "created": 0,
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "cogniverse",
                "revision": MODEL_REVISION,
            }
        ],
        "object": "list",
    }
    assert observed == {"lifespan": "stopped"}


def test_live_identity_round_trip_is_accepted_by_the_canonical_resolver(monkeypatch):
    observed: dict[str, object] = {}
    app = _authenticated_app(monkeypatch, observed)
    spec = get_inference_service_spec("vllm_colpali")

    with _live_server(app) as base_url:
        candidate = CandidateEndpoint(
            provider="local",
            base_url=base_url,
            credentials=EndpointCredentials(bearer_token=API_KEY),
        )
        with httpx.Client(timeout=2) as client:
            resolved = EndpointResolver(client).resolve(spec, explicit=candidate)

    assert resolved == ResolvedInferenceEndpoint(
        service="vllm_colpali",
        provider="local",
        base_url=base_url,
        headers={"Authorization": f"Bearer {API_KEY}"},
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
    )
    assert observed == {"lifespan": "stopped"}
