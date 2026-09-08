"""What a Modal inference app launches with and what it publishes as its window.

The wrapper answers ``/v1/models`` itself so discovery never wakes a
scale-to-zero GPU, which makes that listing the only place a client can read
the window. The flag the engine launches with and the ``max_model_len`` the
listing carries therefore both come from one spec field, and a service that
declares none publishes none.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from cogniverse_cli.modal_inference.serving import (
    build_authenticated_asgi_app,
    models_response,
)
from cogniverse_cli.modal_inference.vllm import _SERVICE_ARGUMENTS, _vllm_command
from fastapi import FastAPI

from cogniverse_foundation.config.token_budget import extract_context_window
from cogniverse_foundation.inference_specs import get_inference_service_spec

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

API_KEY = "context-window-test-key"

# Served by the wrapper's own ASGI app with no vLLM engine behind it: no
# ``--max-model-len`` to carry, so no ``max_model_len`` to publish.
NON_VLLM_SERVICES = (
    "colbert_pylate",
    "gliner",
    "video_embed",
    "clap_embed",
    "face_embed",
)


def _window_arguments(command: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        command[index : index + 2]
        for index, token in enumerate(command)
        if token == "--max-model-len"
    )


def _get(app: FastAPI, path: str) -> httpx.Response:
    async def send() -> httpx.Response:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="https://inference.test",
            ) as client:
                return await client.get(
                    path,
                    headers={"Authorization": f"Bearer {API_KEY}"},
                )

    return asyncio.run(send())


def test_the_vllm_services_declare_the_windows_they_are_launched_with():
    assert {
        name: get_inference_service_spec(name).context_window
        for name in _SERVICE_ARGUMENTS
    } == {
        "vllm_colpali": 4096,
        "denseon": None,
        "vllm_llm_student": 8192,
        "vllm_llm_teacher": 4096,
        "vllm_asr": 448,
    }


def test_every_launch_command_carries_exactly_the_window_its_spec_declares():
    specs = {name: get_inference_service_spec(name) for name in _SERVICE_ARGUMENTS}

    emitted = {
        name: _window_arguments(_vllm_command(spec)) for name, spec in specs.items()
    }

    assert emitted == {
        name: ()
        if spec.context_window is None
        else (("--max-model-len", str(spec.context_window)),)
        for name, spec in specs.items()
    }


def test_a_service_declaring_no_window_launches_without_the_flag():
    spec = get_inference_service_spec("denseon")

    command = _vllm_command(spec, port=8001)

    assert "--max-model-len" not in command
    assert command == (
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


def test_the_teacher_launch_command_carries_the_window_and_nothing_else():
    spec = get_inference_service_spec("vllm_llm_teacher")

    assert _vllm_command(spec, port=8001) == (
        "vllm",
        "serve",
        "Qwen/Qwen3-14B-AWQ",
        "--revision",
        "31c69efc29464b6bb0aee1398b5a7b50a99340c3",
        "--served-model-name",
        "Qwen/Qwen3-14B-AWQ",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
        "--max-model-len",
        str(spec.context_window),
    )


def test_the_asr_launch_command_keeps_its_engine_arguments_beside_the_window():
    spec = get_inference_service_spec("vllm_asr")

    assert _vllm_command(spec, port=8001) == (
        "vllm",
        "serve",
        "openai/whisper-large-v3-turbo",
        "--revision",
        "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
        "--served-model-name",
        "openai/whisper-large-v3-turbo",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
        "--max-model-len",
        str(spec.context_window),
        "--runner",
        "generate",
    )


def test_the_models_listing_publishes_the_window_the_spec_declares():
    spec = get_inference_service_spec("vllm_llm_teacher")

    assert models_response(
        spec.model_id,
        spec.model_revision,
        context_window=spec.context_window,
    ) == {
        "data": [
            {
                "created": 0,
                "id": spec.model_id,
                "object": "model",
                "owned_by": "cogniverse",
                "revision": spec.model_revision,
                "max_model_len": spec.context_window,
            }
        ],
        "object": "list",
    }


def test_the_published_window_is_the_one_the_budget_client_reads_back():
    spec = get_inference_service_spec("vllm_llm_teacher")

    listing = models_response(
        spec.model_id,
        spec.model_revision,
        context_window=spec.context_window,
    )

    assert extract_context_window(listing) == spec.context_window


def test_the_authenticated_route_serves_the_window_to_a_real_client(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", API_KEY)
    spec = get_inference_service_spec("vllm_llm_teacher")
    app = build_authenticated_asgi_app(
        FastAPI(),
        model_id=spec.model_id,
        model_revision=spec.model_revision,
        context_window=spec.context_window,
    )

    response = _get(app, "/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "data": [
            {
                "created": 0,
                "id": spec.model_id,
                "object": "model",
                "owned_by": "cogniverse",
                "revision": spec.model_revision,
                "max_model_len": spec.context_window,
            }
        ],
        "object": "list",
    }
    assert extract_context_window(response.json()) == spec.context_window


def test_the_services_without_an_engine_publish_no_window():
    specs = {name: get_inference_service_spec(name) for name in NON_VLLM_SERVICES}

    assert {name: spec.context_window for name, spec in specs.items()} == dict.fromkeys(
        NON_VLLM_SERVICES
    )
    assert {
        name: tuple(
            sorted(models_response(spec.model_id, spec.model_revision)["data"][0])
        )
        for name, spec in specs.items()
    } == dict.fromkeys(
        NON_VLLM_SERVICES,
        ("created", "id", "object", "owned_by", "revision"),
    )


def test_a_listing_without_a_window_is_byte_identical_to_the_shipped_payload():
    spec = get_inference_service_spec("gliner")

    encoded = json.dumps(models_response(spec.model_id, spec.model_revision)).encode()

    assert encoded == (
        b'{"data": [{"created": 0, "id": "urchade/gliner_large-v2.1", '
        b'"object": "model", "owned_by": "cogniverse", '
        b'"revision": "abd49a1f1ebc12af1be84d06f6848221cf96dcad"}], "object": "list"}'
    )
    assert extract_context_window(json.loads(encoded)) is None


@pytest.mark.parametrize("window", (0, -1))
def test_a_non_positive_published_window_is_refused(window: int):
    spec = get_inference_service_spec("vllm_llm_teacher")

    with pytest.raises(ValueError) as exc:
        models_response(
            spec.model_id,
            spec.model_revision,
            context_window=window,
        )

    assert str(exc.value) == (
        f"context_window must be a positive token count, got {window}"
    )
