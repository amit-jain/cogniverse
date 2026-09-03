from __future__ import annotations

import asyncio
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from cogniverse_cli.modal_inference.gliner import app as gliner_app
from cogniverse_cli.modal_inference.servers import gliner as gliner_server

from cogniverse_core.common.models.model_loaders import (
    RemoteGlinerClient,
)
from cogniverse_foundation.inference_specs import get_inference_service_spec

API_KEY = "language-video-test-key"
VIDEO_PATH = Path("tests/system/resources/videos/v_-6dz6tBH77I.mp4")
GLINER_DOCKERFILE = Path("deploy/gliner/Dockerfile")


@pytest.fixture(autouse=True)
def _reset_model_state():
    gliner_server._models.clear()
    yield
    gliner_server._models.clear()


async def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="https://inference.test",
        ) as client:
            return await client.request(method, path, **kwargs)


def _modal_asgi_app(modal_app):
    return modal_app.registered_functions["Inference"].get_raw_f()()


def _authorization() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}


@contextmanager
def _live_server(app):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", ws="none")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not server.started:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("language/video inference server did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("language/video inference server did not stop")


class _ExactEntityModel:
    def predict_entities(self, text, labels, threshold):
        assert text == "Marie Curie founded institutes in Paris and Warsaw."
        assert labels == ["person", "city"]
        assert threshold == 0.4
        return [
            {
                "text": "Marie Curie",
                "label": "person",
                "score": 0.998,
                "start": 0,
                "end": 11,
            },
            {
                "text": "Paris",
                "label": "city",
                "score": 0.991,
                "start": 34,
                "end": 39,
            },
            {
                "text": "Warsaw",
                "label": "city",
                "score": 0.989,
                "start": 44,
                "end": 50,
            },
        ]


def test_modal_apps_pin_identity_gpu_cache_auth_and_scale_to_zero():
    expected = {
        "gliner": gliner_app,
    }

    for service, modal_app in expected.items():
        spec = get_inference_service_spec(service)
        function = modal_app.registered_functions["Inference"]

        assert modal_app.name == spec.modal_app
        assert modal_app.registered_web_endpoints == ["Inference"]
        assert function.tag == spec.modal_object
        assert function.spec.gpus == list(spec.gpu_candidates)
        assert spec.min_containers == 0
        assert spec.scaledown_window == 300
        assert list(function.spec.volumes) == ["/root/.cache/huggingface"]
        assert repr(function.spec.volumes["/root/.cache/huggingface"]) == (
            "modal.Volume.from_name('cogniverse-huggingface-cache')"
        )
        assert [repr(secret) for secret in function.spec.secrets] == [
            "modal.Secret.from_name('cogniverse-inference-api-key')"
        ]


def test_gliner_container_pins_the_canonical_model_identifier():
    dockerfile = GLINER_DOCKERFILE.read_text()

    assert "ENV MODEL_NAME=urchade/gliner_large-v2.1" in dockerfile
    assert "MODEL_REVISION=abd49a1f1ebc12af1be84d06f6848221cf96dcad" in dockerfile
    assert "gliner_medium" not in dockerfile


def test_gliner_modal_wrapper_preserves_exact_production_entities(monkeypatch):
    spec = get_inference_service_spec("gliner")
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
    gliner_server._models[spec.model_id] = _ExactEntityModel()
    app = _modal_asgi_app(gliner_app)

    identity = asyncio.run(_request(app, "GET", "/v1/models", headers=_authorization()))
    health = asyncio.run(_request(app, "GET", "/health", headers=_authorization()))
    response = asyncio.run(
        _request(
            app,
            "POST",
            "/predict_entities",
            headers=_authorization(),
            json={
                "text": "Marie Curie founded institutes in Paris and Warsaw.",
                "labels": ["person", "city"],
                "threshold": 0.4,
                "model": spec.model_id,
            },
        )
    )

    assert identity.status_code == 200
    assert identity.json() == {
        "data": [
            {
                "created": 0,
                "id": "urchade/gliner_large-v2.1",
                "object": "model",
                "owned_by": "cogniverse",
                "revision": "abd49a1f1ebc12af1be84d06f6848221cf96dcad",
            }
        ],
        "object": "list",
    }
    assert health.status_code == 200
    assert health.json() == {
        "status": "ready",
        "model": "urchade/gliner_large-v2.1",
        "model_revision": "abd49a1f1ebc12af1be84d06f6848221cf96dcad",
        "loaded_models": ["urchade/gliner_large-v2.1"],
    }
    assert response.status_code == 200
    assert response.json() == {
        "entities": [
            {
                "text": "Marie Curie",
                "label": "person",
                "score": 0.998,
                "start": 0,
                "end": 11,
            },
            {
                "text": "Paris",
                "label": "city",
                "score": 0.991,
                "start": 34,
                "end": 39,
            },
            {
                "text": "Warsaw",
                "label": "city",
                "score": 0.989,
                "start": 44,
                "end": 50,
            },
        ],
        "model": "urchade/gliner_large-v2.1",
    }


def test_remote_gliner_client_reaches_authenticated_production_route(monkeypatch):
    spec = get_inference_service_spec("gliner")
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
    gliner_server._models[spec.model_id] = _ExactEntityModel()

    with _live_server(_modal_asgi_app(gliner_app)) as endpoint:
        client = RemoteGlinerClient(
            endpoint,
            spec.model_id,
            api_key=API_KEY,
        )
        entities = client.predict_entities(
            "Marie Curie founded institutes in Paris and Warsaw.",
            ["person", "city"],
            threshold=0.4,
        )

    assert entities == [
        {
            "text": "Marie Curie",
            "label": "person",
            "score": 0.998,
            "start": 0,
            "end": 11,
        },
        {
            "text": "Paris",
            "label": "city",
            "score": 0.991,
            "start": 34,
            "end": 39,
        },
        {
            "text": "Warsaw",
            "label": "city",
            "score": 0.989,
            "start": 44,
            "end": 50,
        },
    ]


def test_gliner_concurrent_cold_health_requests_load_one_model(monkeypatch):
    spec = get_inference_service_spec("gliner")
    loaded_model = object()
    loads: list[str] = []

    class _Gliner:
        @staticmethod
        def from_pretrained(name: str, *, revision: str, map_location: str):
            loads.append(f"{name}@{revision}:{map_location}")
            time.sleep(0.05)
            return loaded_model

    monkeypatch.setitem(sys.modules, "gliner", SimpleNamespace(GLiNER=_Gliner))

    with _live_server(gliner_server.app) as endpoint:
        with ThreadPoolExecutor(max_workers=12) as executor:
            responses = list(
                executor.map(
                    lambda _: httpx.get(f"{endpoint}/health", timeout=5),
                    range(12),
                )
            )

    assert loads == [
        "urchade/gliner_large-v2.1@abd49a1f1ebc12af1be84d06f6848221cf96dcad:cpu"
    ]
    assert [response.status_code for response in responses] == [200] * 12
    assert [response.json() for response in responses] == [
        {
            "status": "ready",
            "model": spec.model_id,
            "model_revision": spec.model_revision,
            "loaded_models": [spec.model_id],
        }
    ] * 12


def test_gliner_health_load_failure_is_not_ready_and_next_request_retries(
    monkeypatch,
):
    spec = get_inference_service_spec("gliner")
    attempts = 0

    class _Gliner:
        @staticmethod
        def from_pretrained(name: str, *, revision: str, map_location: str):
            nonlocal attempts
            attempts += 1
            assert (name, revision, map_location) == (
                spec.model_id,
                spec.model_revision,
                "cpu",
            )
            if attempts == 1:
                raise OSError("checkpoint index is unreadable")
            return object()

    monkeypatch.setitem(sys.modules, "gliner", SimpleNamespace(GLiNER=_Gliner))

    failed = asyncio.run(_request(gliner_server.app, "GET", "/health"))
    recovered = asyncio.run(_request(gliner_server.app, "GET", "/health"))

    assert failed.status_code == 503
    assert failed.json() == {
        "detail": (
            "gliner: model urchade/gliner_large-v2.1 load failed (OSError): "
            "checkpoint index is unreadable"
        )
    }
    assert recovered.status_code == 200
    assert recovered.json() == {
        "status": "ready",
        "model": spec.model_id,
        "model_revision": spec.model_revision,
        "loaded_models": [spec.model_id],
    }
    assert attempts == 2


def test_gliner_model_load_failure_has_service_model_and_cause(monkeypatch):
    spec = get_inference_service_spec("gliner")

    class _Gliner:
        @staticmethod
        def from_pretrained(name: str, *, revision: str, map_location: str):
            assert revision == "abd49a1f1ebc12af1be84d06f6848221cf96dcad"
            assert map_location == "cpu"
            raise OSError(f"weights for {name} are corrupt")

    monkeypatch.setitem(sys.modules, "gliner", SimpleNamespace(GLiNER=_Gliner))

    response = asyncio.run(
        _request(
            gliner_server.app,
            "POST",
            "/predict_entities",
            json={"text": "Ada Lovelace", "labels": ["person"], "model": spec.model_id},
        )
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": (
            "gliner: model urchade/gliner_large-v2.1 load failed (OSError): "
            "weights for urchade/gliner_large-v2.1 are corrupt"
        )
    }


def test_gliner_inference_failure_has_service_model_and_cause():
    spec = get_inference_service_spec("gliner")

    class _FailedModel:
        def predict_entities(self, text, labels, threshold):
            raise RuntimeError("tensor allocation failed")

    gliner_server._models[spec.model_id] = _FailedModel()

    response = asyncio.run(
        _request(
            gliner_server.app,
            "POST",
            "/predict_entities",
            json={"text": "Ada Lovelace", "labels": ["person"], "model": spec.model_id},
        )
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": (
            "gliner: model urchade/gliner_large-v2.1 inference failed "
            "(RuntimeError): tensor allocation failed"
        )
    }


def test_gliner_rejects_an_unpinned_request_model():
    response = asyncio.run(
        _request(
            gliner_server.app,
            "POST",
            "/predict_entities",
            json={
                "text": "Ada Lovelace",
                "labels": ["person"],
                "model": "urchade/gliner_medium-v2.1",
            },
        )
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "model"]
