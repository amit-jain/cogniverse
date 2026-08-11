"""PyLate ColBERT server contract — exact shapes, faults, and cold-load.

Drives the real ``servers/pylate.py`` FastAPI app over ASGI and live HTTP,
pinning the boundary contract ``RemoteColBERTLoader`` and the deployment
probes rely on: the ``/pooling`` request and response shapes, ``is_query``
forwarding, the ``/health`` identity payload, load/inference failure
surfacing, and single cold-load under concurrency.

The encoder is a deterministic stand-in here because these tests pin the
HTTP mechanics — failure injection and load-timing control are impossible
with the real checkpoint. The real ``lightonai/LateOn`` model runs through
this same server in the ingestion parity suite, which asserts per-token
equality against the in-process PyLate oracle.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import httpx
import numpy as np
import pytest
import uvicorn
from cogniverse_cli.modal_inference.code_colbert_pylate import app as code_lateon_app
from cogniverse_cli.modal_inference.lateon import app as lateon_app
from cogniverse_cli.modal_inference.servers import pylate as pylate_server
from cogniverse_cli.modal_inference_config import get_inference_service_spec

SPEC = get_inference_service_spec("colbert_pylate")
CODE_SPEC = get_inference_service_spec("code_colbert_pylate")
API_KEY = "pylate-test-key"


def _authorization() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}


def _modal_asgi_app(modal_app):
    return modal_app.registered_functions["Inference"].get_raw_f()()


async def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://inference.test",
    ) as client:
        return await client.request(method, path, **kwargs)


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
        raise RuntimeError("pylate inference server did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("pylate inference server did not stop")


class _MatrixModel:
    """Per-token encoder with distinct values per input index and direction."""

    def __init__(self):
        self.calls = []

    def encode(self, texts, is_query=False, show_progress_bar=None, batch_size=32):
        self.calls.append((tuple(texts), is_query, show_progress_bar, batch_size))
        matrices = []
        for index in range(len(texts)):
            rows = 32 if is_query else 3 + index
            value = (0.5 if is_query else -0.25) + index
            matrices.append(np.full((rows, 4), value, dtype=np.float32))
        return matrices


def _app_with_model(monkeypatch, model=None):
    model = model if model is not None else _MatrixModel()
    loads: list[tuple[str, str, str]] = []

    def load(model_name, model_revision, device):
        loads.append((model_name, model_revision, device))
        return model

    monkeypatch.setattr(pylate_server, "_load_colbert", load)
    app = pylate_server.build_app(SPEC.model_id, SPEC.model_revision, "cpu")
    return app, model, loads


def test_pooling_returns_per_token_matrices_in_input_order(monkeypatch):
    app, model, loads = _app_with_model(monkeypatch)

    response = asyncio.run(
        _request(
            app,
            "POST",
            "/pooling",
            json={
                "input": ["alpha", "beta"],
                "model": SPEC.model_id,
                "is_query": False,
            },
        )
    )

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "model": SPEC.model_id,
        "data": [
            {"object": "pooling", "index": 0, "data": [[-0.25] * 4] * 3},
            {"object": "pooling", "index": 1, "data": [[0.75] * 4] * 4},
        ],
    }
    assert model.calls == [(("alpha", "beta"), False, False, 32)]
    assert loads == [(SPEC.model_id, SPEC.model_revision, "cpu")]


def test_pooling_forwards_query_direction_to_the_encoder(monkeypatch):
    app, model, _ = _app_with_model(monkeypatch)

    response = asyncio.run(
        _request(
            app,
            "POST",
            "/pooling",
            json={"input": ["what is a vector database"], "is_query": True},
        )
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["data"] == [
        {"object": "pooling", "index": 0, "data": [[0.5] * 4] * 32}
    ]
    assert model.calls == [(("what is a vector database",), True, False, 32)]


def test_health_reports_pinned_identity(monkeypatch):
    app, _, loads = _app_with_model(monkeypatch)

    response = asyncio.run(_request(app, "GET", "/health"))

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model": SPEC.model_id,
        "model_revision": SPEC.model_revision,
        "loaded_models": [SPEC.model_id],
    }
    assert loads == [(SPEC.model_id, SPEC.model_revision, "cpu")]


def test_runtime_boot_probe_identifies_the_served_model(monkeypatch):
    """The runtime's startup probe must read this server's real /health body.

    ``validate_inference_services`` identifies a service by feeding the
    ``/health`` body to ``_extract_model_from_health``; when that returns
    nothing it retries until the boot deadline and then refuses to start,
    failing every profile bound to the service. Asserting the payload shape
    alone does not catch a key the extractor cannot read, so drive the
    extractor itself with the body this server actually returns.
    """
    from cogniverse_runtime.inference_health_check import (
        _extract_model_from_health,
    )

    app, _, _ = _app_with_model(monkeypatch)

    response = asyncio.run(_request(app, "GET", "/health"))

    assert response.status_code == 200
    assert _extract_model_from_health(response.json()) == SPEC.model_id


def test_empty_input_is_rejected_before_model_load(monkeypatch):
    app, model, loads = _app_with_model(monkeypatch)

    response = asyncio.run(_request(app, "POST", "/pooling", json={"input": []}))

    assert response.status_code == 400
    assert response.json() == {"detail": "`input` must be a non-empty list"}
    assert model.calls == []
    assert loads == []


def test_unpinned_request_model_is_rejected_before_model_load(monkeypatch):
    app, model, loads = _app_with_model(monkeypatch)

    response = asyncio.run(
        _request(
            app,
            "POST",
            "/pooling",
            json={"input": ["alpha"], "model": "other/model"},
        )
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": f"model must equal pinned model {SPEC.model_id}"
    }
    assert model.calls == []
    assert loads == []


def test_concurrent_cold_pooling_requests_load_one_model(monkeypatch):
    loads: list[tuple[str, str, str]] = []
    model = _MatrixModel()

    def slow_load(model_name, model_revision, device):
        loads.append((model_name, model_revision, device))
        time.sleep(0.05)
        return model

    monkeypatch.setattr(pylate_server, "_load_colbert", slow_load)
    app = pylate_server.build_app(SPEC.model_id, SPEC.model_revision, "cpu")

    with _live_server(app) as endpoint:
        with ThreadPoolExecutor(max_workers=12) as executor:
            responses = list(
                executor.map(
                    lambda index: httpx.post(
                        f"{endpoint}/pooling",
                        json={"input": [f"text {index}"], "is_query": False},
                        timeout=10,
                    ),
                    range(12),
                )
            )

    assert loads == [(SPEC.model_id, SPEC.model_revision, "cpu")]
    assert [response.status_code for response in responses] == [200] * 12
    for response in responses:
        payload = response.json()
        assert payload["model"] == SPEC.model_id
        assert payload["data"][0]["data"] == [[-0.25] * 4] * 3


def test_health_load_failure_is_not_ready_and_next_request_retries(monkeypatch):
    attempts = 0
    model = _MatrixModel()

    def flaky_load(model_name, model_revision, device):
        nonlocal attempts
        attempts += 1
        assert (model_name, model_revision, device) == (
            SPEC.model_id,
            SPEC.model_revision,
            "cpu",
        )
        if attempts == 1:
            raise OSError("checkpoint index is unreadable")
        return model

    monkeypatch.setattr(pylate_server, "_load_colbert", flaky_load)
    app = pylate_server.build_app(SPEC.model_id, SPEC.model_revision, "cpu")

    failed = asyncio.run(_request(app, "GET", "/health"))
    recovered = asyncio.run(_request(app, "GET", "/health"))

    assert failed.status_code == 503
    assert failed.json() == {
        "detail": (
            f"pylate: model {SPEC.model_id} load failed (OSError): "
            "checkpoint index is unreadable"
        )
    }
    assert recovered.status_code == 200
    assert recovered.json()["status"] == "ready"
    assert attempts == 2


def test_pooling_load_failure_has_service_model_and_cause(monkeypatch):
    def broken_load(model_name, model_revision, device):
        raise OSError(f"weights for {model_name} are corrupt")

    monkeypatch.setattr(pylate_server, "_load_colbert", broken_load)
    app = pylate_server.build_app(SPEC.model_id, SPEC.model_revision, "cpu")

    response = asyncio.run(_request(app, "POST", "/pooling", json={"input": ["alpha"]}))

    assert response.status_code == 503
    assert response.json() == {
        "detail": (
            f"pylate: model {SPEC.model_id} load failed (OSError): "
            f"weights for {SPEC.model_id} are corrupt"
        )
    }


def test_inference_failure_has_service_model_and_cause(monkeypatch):
    class _FailedModel:
        def encode(self, texts, is_query=False, show_progress_bar=None, batch_size=32):
            raise RuntimeError("tensor allocation failed")

    app, _, _ = _app_with_model(monkeypatch, model=_FailedModel())

    response = asyncio.run(_request(app, "POST", "/pooling", json={"input": ["alpha"]}))

    assert response.status_code == 500
    assert response.json() == {
        "detail": (
            f"pylate: model {SPEC.model_id} inference failed "
            "(RuntimeError): tensor allocation failed"
        )
    }


@pytest.mark.parametrize(
    ("model_name", "model_revision", "device", "match"),
    [
        ("", CODE_SPEC.model_revision, "cpu", "MODEL_NAME"),
        (" lightonai/LateOn", CODE_SPEC.model_revision, "cpu", "MODEL_NAME"),
        (CODE_SPEC.model_id, "", "cpu", "MODEL_REVISION"),
        (CODE_SPEC.model_id, "main", "cpu", "MODEL_REVISION"),
        (CODE_SPEC.model_id, "master", "cpu", "MODEL_REVISION"),
        (CODE_SPEC.model_id, "latest", "cpu", "MODEL_REVISION"),
        (CODE_SPEC.model_id, CODE_SPEC.model_revision, "rocm", "DEVICE"),
    ],
)
def test_build_app_rejects_invalid_identity(model_name, model_revision, device, match):
    with pytest.raises(ValueError, match=match):
        pylate_server.build_app(model_name, model_revision, device)


def test_modal_apps_pin_identity_gpu_cache_auth_and_scale_to_zero():
    expected = {
        "colbert_pylate": lateon_app,
        "code_colbert_pylate": code_lateon_app,
    }

    for service, modal_app in expected.items():
        spec = get_inference_service_spec(service)
        function = modal_app.registered_functions["Inference"]

        assert modal_app.name == spec.modal_app
        assert modal_app.registered_web_endpoints == ["Inference"]
        assert set(modal_app.registered_functions) == {"Inference"}
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

    assert lateon_app is not code_lateon_app
    assert {SPEC.output_dimension, CODE_SPEC.output_dimension} == {128, 48}


def test_remote_colbert_loader_round_trips_via_live_pylate_server(monkeypatch):
    """RemoteColBERTLoader ↔ the real server module over live HTTP: raw text
    plus ``is_query`` goes over the wire, and the exact per-token matrices
    come back in input order for both encode directions."""
    from cogniverse_core.common.models.model_loaders import RemoteColBERTLoader

    app, model, _ = _app_with_model(monkeypatch)

    with _live_server(app) as endpoint:
        wrapper, processor = RemoteColBERTLoader(
            model_name=SPEC.model_id,
            config={"remote_inference_url": endpoint},
        ).load_model()
        assert processor is None
        try:
            documents = wrapper.encode(["alpha", "beta"], is_query=False)
            queries = wrapper.encode(["what is a vector database"], is_query=True)
        finally:
            wrapper._close()

    assert documents == [[[-0.25] * 4] * 3, [[0.75] * 4] * 4]
    assert queries == [[[0.5] * 4] * 32]
    assert model.calls == [
        (("alpha", "beta"), False, False, 32),
        (("what is a vector database",), True, False, 32),
    ]


def test_remote_colbert_loader_reaches_authenticated_modal_route(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
    model = _MatrixModel()
    monkeypatch.setattr(
        pylate_server,
        "_load_colbert",
        lambda model_name, model_revision, device: model,
    )
    from cogniverse_core.common.models.model_loaders import RemoteColBERTLoader

    with _live_server(_modal_asgi_app(lateon_app)) as endpoint:
        unauthenticated_wrapper, _ = RemoteColBERTLoader(
            model_name=SPEC.model_id,
            config={"remote_inference_url": endpoint},
        ).load_model()
        try:
            with pytest.raises(
                RuntimeError,
                match=f"remote ColBERT pooling failed for model '{SPEC.model_id}'",
            ):
                unauthenticated_wrapper.encode(["alpha"], is_query=True)
        finally:
            unauthenticated_wrapper._close()

        wrapper, _ = RemoteColBERTLoader(
            model_name=SPEC.model_id,
            config={"remote_inference_url": endpoint},
            _resolved_headers={"Authorization": f"Bearer {API_KEY}"},
        ).load_model()
        try:
            queries = wrapper.encode(["alpha"], is_query=True)
        finally:
            wrapper._close()

    assert queries == [[[0.5] * 4] * 32]
    assert model.calls == [(("alpha",), True, False, 32)]


def test_remote_colbert_loader_names_model_and_endpoint_on_dead_port():
    from cogniverse_core.common.models.model_loaders import RemoteColBERTLoader

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    dead_endpoint = f"http://127.0.0.1:{port}"

    wrapper, _ = RemoteColBERTLoader(
        model_name=SPEC.model_id,
        config={"remote_inference_url": dead_endpoint},
    ).load_model()
    try:
        with pytest.raises(
            RuntimeError,
            match=(
                f"remote ColBERT pooling failed for model '{SPEC.model_id}' "
                f"at {dead_endpoint}"
            ),
        ):
            wrapper.encode(["alpha"], is_query=False)
    finally:
        wrapper._close()


def test_modal_wrapper_requires_bearer_and_serves_exact_pooling(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
    model = _MatrixModel()
    loads: list[tuple[str, str, str]] = []

    def load(model_name, model_revision, device):
        loads.append((model_name, model_revision, device))
        return model

    monkeypatch.setattr(pylate_server, "_load_colbert", load)
    app = _modal_asgi_app(lateon_app)

    unauthenticated = asyncio.run(
        _request(app, "POST", "/pooling", json={"input": ["alpha"]})
    )
    identity = asyncio.run(_request(app, "GET", "/v1/models", headers=_authorization()))
    health = asyncio.run(_request(app, "GET", "/health", headers=_authorization()))
    pooling = asyncio.run(
        _request(
            app,
            "POST",
            "/pooling",
            headers=_authorization(),
            json={"input": ["alpha"], "model": SPEC.model_id, "is_query": True},
        )
    )

    assert unauthenticated.status_code == 401
    assert loads == [(SPEC.model_id, SPEC.model_revision, "cuda")]
    assert identity.status_code == 200
    assert identity.json() == {
        "data": [
            {
                "created": 0,
                "id": SPEC.model_id,
                "object": "model",
                "owned_by": "cogniverse",
                "revision": SPEC.model_revision,
            }
        ],
        "object": "list",
    }
    assert health.status_code == 200
    assert health.json() == {
        "status": "ready",
        "model": SPEC.model_id,
        "model_revision": SPEC.model_revision,
        "loaded_models": [SPEC.model_id],
    }
    assert pooling.status_code == 200
    assert pooling.json() == {
        "object": "list",
        "model": SPEC.model_id,
        "data": [{"object": "pooling", "index": 0, "data": [[0.5] * 4] * 32}],
    }
    assert model.calls == [(("alpha",), True, False, 32)]


def test_oversized_input_list_is_rejected_before_model_load(monkeypatch):
    app, model, loads = _app_with_model(monkeypatch)

    response = asyncio.run(
        _request(
            app,
            "POST",
            "/pooling",
            json={"input": ["t"] * 257, "is_query": False},
        )
    )

    assert response.status_code == 413
    assert response.json() == {
        "detail": "pylate: `input` holds 257 texts, limit is 256"
    }
    assert model.calls == []
    assert loads == []


def test_input_list_at_the_limit_is_encoded(monkeypatch):
    app, model, _ = _app_with_model(monkeypatch)

    response = asyncio.run(
        _request(
            app,
            "POST",
            "/pooling",
            json={"input": ["t"] * 256, "is_query": False},
        )
    )

    assert response.status_code == 200
    assert len(response.json()["data"]) == 256
    assert len(model.calls) == 1


def test_oversized_input_characters_are_rejected_before_model_load(monkeypatch):
    app, model, loads = _app_with_model(monkeypatch)

    response = asyncio.run(
        _request(
            app,
            "POST",
            "/pooling",
            json={"input": ["x" * 1_000_001, "y" * 1_000_000], "is_query": False},
        )
    )

    assert response.status_code == 413
    assert response.json() == {
        "detail": "pylate: `input` holds 2000001 characters, limit is 2000000"
    }
    assert model.calls == []
    assert loads == []


def test_encode_receives_the_configured_batch_size(monkeypatch):
    app, model, _ = _app_with_model(monkeypatch)

    asyncio.run(
        _request(app, "POST", "/pooling", json={"input": ["a", "b"], "is_query": True})
    )

    assert model.calls == [(("a", "b"), True, False, 32)]


def test_bounds_are_configurable_per_service(monkeypatch):
    model = _MatrixModel()
    monkeypatch.setattr(pylate_server, "_load_colbert", lambda *args: model)
    app = pylate_server.build_app(
        SPEC.model_id,
        SPEC.model_revision,
        "cpu",
        max_input_items=2,
        max_input_chars=10,
        encode_batch_size=1,
    )

    rejected = asyncio.run(
        _request(app, "POST", "/pooling", json={"input": ["a", "b", "c"]})
    )
    assert rejected.status_code == 413
    assert rejected.json() == {"detail": "pylate: `input` holds 3 texts, limit is 2"}

    accepted = asyncio.run(_request(app, "POST", "/pooling", json={"input": ["a"]}))
    assert accepted.status_code == 200
    assert model.calls == [(("a",), False, False, 1)]


@pytest.mark.parametrize(
    "bounds, match",
    [
        ({"max_input_items": 0}, "MAX_INPUT_ITEMS must be a positive integer"),
        ({"max_input_chars": -1}, "MAX_INPUT_CHARS must be a positive integer"),
        ({"encode_batch_size": 0}, "ENCODE_BATCH_SIZE must be a positive integer"),
        ({"max_input_items": True}, "MAX_INPUT_ITEMS must be a positive integer"),
    ],
)
def test_build_app_rejects_unusable_bounds(bounds, match):
    with pytest.raises(ValueError, match=match):
        pylate_server.build_app(SPEC.model_id, SPEC.model_revision, "cpu", **bounds)
