from __future__ import annotations

import asyncio
import hashlib
import json
import re
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from cogniverse_cli.inference_endpoints import ModelIdentityError
from cogniverse_cli.modal_inference.serving import build_authenticated_asgi_app
from cogniverse_cli.modal_inference.vllm import (
    _build_process_proxy_app,
    _ServingProcess,
    _vllm_command,
    _vllm_environment,
)
from cogniverse_cli.modal_inference_config import get_inference_service_spec

from cogniverse_agents.audio_analysis_agent import AudioAnalysisAgent, AudioAnalysisDeps
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from tests.agents.integration import conftest as agents_conftest


def test_gemma_service_pins_the_production_chat_contract():
    from cogniverse_cli.modal_inference.gemma import app

    spec = get_inference_service_spec("vllm_llm_student")

    assert _vllm_command(spec) == (
        "vllm",
        "serve",
        "google/gemma-4-e4b-it",
        "--revision",
        "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "--served-model-name",
        "google/gemma-4-e4b-it",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
        "--max-model-len",
        "8192",
        "--enforce-eager",
        "--max-num-seqs",
        "1",
        "--limit-mm-per-prompt",
        '{"video":0,"image":4}',
    )
    assert _vllm_environment(spec) == {
        "HF_HOME": "/root/.cache/huggingface",
        "MALLOC_ARENA_MAX": "2",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        "VLLM_CACHE_ROOT": "/root/.cache/huggingface/.vllm",
        "VLLM_CPU_KVCACHE_SPACE": "4",
    }
    function = app.registered_functions["Inference"]
    assert app.name == "cogniverse-vllm-llm-student"
    assert app.registered_web_endpoints == ["Inference"]
    assert function.spec.gpus == ["L4", "A10", "L40S"]
    assert spec.min_containers == 0
    assert [repr(secret) for secret in function.spec.secrets] == [
        "modal.Secret.from_name('cogniverse-inference-api-key')",
        "modal.Secret.from_name('hf-token')",
    ]


def test_whisper_service_pins_the_production_transcription_contract():
    from cogniverse_cli.modal_inference.whisper import app

    spec = get_inference_service_spec("vllm_asr")

    assert _vllm_command(spec) == (
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
        "--runner",
        "generate",
        "--max-model-len",
        "448",
    )
    assert _vllm_environment(spec) == {
        "HF_HOME": "/root/.cache/huggingface",
        "MALLOC_ARENA_MAX": "2",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        "VLLM_CACHE_ROOT": "/root/.cache/huggingface/.vllm",
        "VLLM_CPU_KVCACHE_SPACE": "1",
    }
    function = app.registered_functions["Inference"]
    assert app.name == "cogniverse-vllm-asr"
    assert app.registered_web_endpoints == ["Inference"]
    assert function.spec.gpus == ["T4", "L4"]
    assert spec.min_containers == 0
    assert [repr(secret) for secret in function.spec.secrets] == [
        "modal.Secret.from_name('cogniverse-inference-api-key')"
    ]


def test_face_service_revision_is_the_official_artifact_digest():
    spec = get_inference_service_spec("face_embed")

    assert (spec.model_id, spec.model_revision) == (
        "buffalo_l",
        "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
    )


def test_generation_endpoint_uses_authenticated_canonical_resolution(monkeypatch):
    spec = get_inference_service_spec("vllm_llm_student")
    monkeypatch.setenv(
        "INFERENCE_SERVICE_URLS",
        json.dumps({spec.name: "https://gemma.modal.run"}),
    )
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "generation-secret")
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((str(request.url), request.headers["Authorization"]))
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": spec.model_id,
                        "revision": spec.model_revision,
                    }
                ]
            },
            request=request,
        )

    endpoint = agents_conftest._resolve_modal_generation_endpoint(
        spec.name,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert requests == [
        (
            "https://gemma.modal.run/v1/models",
            "Bearer generation-secret",
        )
    ]
    assert endpoint is not None
    assert (
        endpoint.service,
        endpoint.provider,
        endpoint.base_url,
        endpoint.model_id,
        endpoint.model_revision,
        dict(endpoint.headers),
    ) == (
        "vllm_llm_student",
        "modal",
        "https://gemma.modal.run",
        "google/gemma-4-e4b-it",
        "ee0ef6023621cff504d758262d4e04895a5af4a2",
        {"Authorization": "Bearer generation-secret"},
    )


def test_wrong_modal_identity_propagates_without_local_fallback(monkeypatch):
    spec = get_inference_service_spec("vllm_asr")
    monkeypatch.setenv(
        "INFERENCE_SERVICE_URLS",
        json.dumps({spec.name: "https://whisper.modal.run"}),
    )
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "generation-secret")

    def wrong_model(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "openai/whisper-tiny",
                        "revision": spec.model_revision,
                    }
                ]
            },
            request=request,
        )

    with pytest.raises(
        ModelIdentityError,
        match="vllm_asr: expected model 'openai/whisper-large-v3-turbo'",
    ):
        agents_conftest._resolve_modal_generation_endpoint(
            spec.name,
            client=httpx.Client(transport=httpx.MockTransport(wrong_model)),
        )


def test_generation_endpoint_configuration_is_strict(monkeypatch):
    monkeypatch.setenv("INFERENCE_SERVICE_URLS", "not-json")

    with pytest.raises(
        RuntimeError,
        match="INFERENCE_SERVICE_URLS must be a JSON object of service URLs",
    ):
        agents_conftest._resolve_modal_generation_endpoint("vllm_asr")


def test_insecure_modal_generation_url_is_rejected_before_request(monkeypatch):
    spec = get_inference_service_spec("vllm_llm_student")
    monkeypatch.setenv(
        "INFERENCE_SERVICE_URLS",
        json.dumps({spec.name: "http://gemma.modal.run"}),
    )
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "must-not-be-transmitted")
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500, request=request)

    with pytest.raises(ValueError, match="Modal inference endpoints require HTTPS"):
        agents_conftest._resolve_modal_generation_endpoint(
            spec.name,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

    assert requests == []


def test_non_modal_generation_url_remains_available_to_cluster_discovery(monkeypatch):
    monkeypatch.setenv(
        "INFERENCE_SERVICE_URLS",
        json.dumps({"vllm_llm_student": "http://127.0.0.1:31846"}),
    )

    endpoint = agents_conftest._resolve_modal_generation_endpoint("vllm_llm_student")

    assert endpoint is None

    class Item:
        _cogniverse_lm_roles = frozenset({"primary"})
        fixturenames = ["ensure_host_ollama", "dspy_lm"]

    item = Item()
    agents_conftest.pytest_collection_modifyitems([item])

    assert item.fixturenames == ["ensure_host_ollama", "dspy_lm"]


def test_live_modal_selection_requires_an_explicit_opt_in(monkeypatch):
    from tests.e2e import conftest as e2e_conftest

    class Item:
        def iter_markers(self, name):
            if name == "requires_modal_inference":
                return iter((SimpleNamespace(args=("vllm_llm_student",)),))
            return iter(())

    config = SimpleNamespace(option=SimpleNamespace(markexpr=""))
    item = Item()
    monkeypatch.delenv("RUN_MODAL_INFERENCE_E2E", raising=False)

    assert e2e_conftest._modal_inference_deselections(config, [item]) == [item]

    config.option.markexpr = "requires_modal_inference"
    assert e2e_conftest._modal_inference_deselections(config, [item]) == []

    config.option.markexpr = ""
    monkeypatch.setenv("RUN_MODAL_INFERENCE_E2E", "1")
    assert e2e_conftest._modal_inference_deselections(config, [item]) == []


def test_modal_requirement_rejects_local_endpoint_before_stateful_stack(monkeypatch):
    from tests.e2e import conftest as e2e_conftest

    spec = get_inference_service_spec("vllm_llm_student")

    class Item:
        nodeid = "tests/e2e/test_modal_inference_e2e.py::test_live_modal_gemma"

        def iter_markers(self, name):
            if name == "requires_modal_inference":
                return iter((SimpleNamespace(args=(spec.name,)),))
            return iter(())

    endpoint = agents_conftest.ResolvedInferenceEndpoint(
        service=spec.name,
        provider="local",
        base_url="http://127.0.0.1:29110",
        headers={"Authorization": "Bearer local-test-key"},
        model_id=spec.model_id,
        model_revision=spec.model_revision,
    )
    request = SimpleNamespace(session=SimpleNamespace(items=[Item()]))
    stateful_calls = []

    def deploy_sha():
        stateful_calls.append("deploy_sha")
        return "ffffffffffffffffffffffffffffffffffffffff"

    monkeypatch.setattr(e2e_conftest, "_current_e2e_deploy_sha", deploy_sha)
    fixture = e2e_conftest.e2e_stack.__wrapped__(
        request,
        {spec.name: endpoint},
    )

    with pytest.raises(
        pytest.fail.Exception,
        match="requires Modal provider.*resolved 'local'",
    ):
        next(fixture)

    assert stateful_calls == []


def test_gemma_fixture_injects_the_exact_authenticated_dspy_contract():
    spec = get_inference_service_spec("vllm_llm_student")
    endpoint = agents_conftest.ResolvedInferenceEndpoint(
        service=spec.name,
        provider="modal",
        base_url="https://gemma.modal.run",
        headers={"Authorization": "Bearer generation-secret"},
        model_id=spec.model_id,
        model_revision=spec.model_revision,
    )

    config = agents_conftest._gemma_llm_config(endpoint)

    assert config.model == "openai/google/gemma-4-e4b-it"
    assert config.api_base == "https://gemma.modal.run/v1"
    assert config.api_key == "generation-secret"
    assert config.extra_headers == {}
    assert config.temperature == 0.1
    assert config.max_tokens == 800


def test_modal_collection_injects_gemma_without_local_provisioning(monkeypatch):
    spec = get_inference_service_spec("vllm_llm_student")
    monkeypatch.setenv(
        "INFERENCE_SERVICE_URLS",
        json.dumps({spec.name: "https://gemma.modal.run"}),
    )

    class Item:
        _cogniverse_lm_roles = frozenset({"primary"})
        fixturenames = ["ensure_host_ollama", "dspy_lm"]

    item = Item()
    agents_conftest.pytest_collection_modifyitems([item])

    assert item.fixturenames == ["dspy_lm", "gemma_inference_endpoint"]


def test_gemma_fixture_publishes_and_restores_the_resolved_endpoint(monkeypatch):
    spec = get_inference_service_spec("vllm_llm_student")
    endpoint = agents_conftest.ResolvedInferenceEndpoint(
        service=spec.name,
        provider="modal",
        base_url="https://gemma.modal.run",
        headers={"Authorization": "Bearer generation-secret"},
        model_id=spec.model_id,
        model_revision=spec.model_revision,
    )
    monkeypatch.setattr(
        agents_conftest,
        "_resolve_modal_generation_endpoint",
        lambda service: endpoint,
    )
    original_environment = {
        "TEST_LLM_API_BASE": "https://original.test/v1",
        "TEST_LLM_MODEL": "original-model",
        "TEST_LLM_PROVIDER": "original-provider",
        "TEST_LLM_API_KEY": "original-test-key",
        "OPENAI_API_KEY": "original-openai-key",
    }
    for name, value in original_environment.items():
        monkeypatch.setenv(name, value)
    fixture = agents_conftest.gemma_inference_endpoint.__wrapped__(object())

    assert next(fixture) == endpoint
    assert {
        name: agents_conftest.os.environ.get(name)
        for name in (
            "TEST_LLM_API_BASE",
            "TEST_LLM_MODEL",
            "TEST_LLM_PROVIDER",
            "TEST_LLM_API_KEY",
            "OPENAI_API_KEY",
        )
    } == {
        "TEST_LLM_API_BASE": "https://gemma.modal.run/v1",
        "TEST_LLM_MODEL": "google/gemma-4-e4b-it",
        "TEST_LLM_PROVIDER": "openai",
        "TEST_LLM_API_KEY": "generation-secret",
        "OPENAI_API_KEY": "generation-secret",
    }
    fixture.close()
    assert {
        name: agents_conftest.os.environ.get(name) for name in original_environment
    } == original_environment


def test_whisper_fixture_fallback_uses_the_exact_production_arguments(monkeypatch):
    monkeypatch.delenv("INFERENCE_SERVICE_URLS", raising=False)

    class Sidecar:
        def __init__(self):
            self.calls: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = []

        def spawn(
            self,
            *,
            model,
            model_revision,
            required_snapshot_files,
            extra_args,
        ):
            self.calls.append(
                (
                    model,
                    model_revision,
                    tuple(required_snapshot_files),
                    tuple(extra_args),
                )
            )
            return "http://127.0.0.1:31845/"

    sidecar = Sidecar()

    endpoint = agents_conftest._resolve_whisper_inference_endpoint(sidecar)

    assert sidecar.calls == [
        (
            "openai/whisper-large-v3-turbo",
            "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
            (
                "added_tokens.json",
                "config.json",
                "generation_config.json",
                "merges.txt",
                "model.safetensors",
                "normalizer.json",
                "preprocessor_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            ),
            ("--runner", "generate", "--max-model-len", "448"),
        )
    ]
    assert (
        endpoint.service,
        endpoint.provider,
        endpoint.base_url,
        dict(endpoint.headers),
        endpoint.model_id,
        endpoint.model_revision,
    ) == (
        "vllm_asr",
        "local",
        "http://127.0.0.1:31845",
        {},
        "openai/whisper-large-v3-turbo",
        "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
    )


@pytest.mark.integration
def test_real_gemma_factory_preserves_concurrent_exact_answers():
    from tests.utils.hermetic_llm import ensure_llm

    endpoint = agents_conftest._resolve_verified_local_endpoint(
        "vllm_llm_student",
        base_url=ensure_llm(model="google/gemma-4-e4b-it"),
        api_key="not-required",
    )
    config = replace(
        agents_conftest._gemma_llm_config(endpoint),
        temperature=0.0,
        max_tokens=20,
        seed=0,
        request_timeout=30,
        num_retries=0,
    )
    lm = create_dspy_lm(config)

    def answer(index: int) -> str:
        expected = f"radium=Ra;request={index}"
        prompt = (
            "The chemical symbol for radium is Ra. Reply with exactly "
            f"{expected} and no other characters."
        )
        return lm(prompt, cache=False)[0]

    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = tuple(pool.map(answer, range(4)))

    assert answers == (
        "radium=Ra;request=0",
        "radium=Ra;request=1",
        "radium=Ra;request=2",
        "radium=Ra;request=3",
    )


def test_gemma_serving_process_failure_is_contextual(monkeypatch):
    spec = get_inference_service_spec("vllm_llm_student")
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "generation-secret")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    process = _ServingProcess(
        service=spec.name,
        command=(sys.executable, "-c", "raise SystemExit(37)"),
        host="127.0.0.1",
        port=port,
        startup_timeout=2,
    )
    app = build_authenticated_asgi_app(
        _build_process_proxy_app(process),
        model_id=spec.model_id,
        model_revision=spec.model_revision,
    )

    async def request() -> httpx.Response:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="https://gemma.modal.test",
            ) as client:
                return await client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer generation-secret"},
                    json={
                        "model": spec.model_id,
                        "messages": [{"role": "user", "content": "radium"}],
                    },
                )

    response = asyncio.run(request())

    assert response.status_code == 503
    assert response.content == (
        b'{"detail":"vllm_llm_student: vLLM serving process exited with status 37"}'
    )
    assert "generation-secret" not in response.text


@pytest.mark.integration
def test_real_whisper_agent_returns_the_exact_normalized_transcript(
    vllm_sidecar,
    tmp_path,
):
    source = httpx.get(
        "https://raw.githubusercontent.com/openai/whisper/main/tests/jfk.flac",
        timeout=60,
        follow_redirects=True,
    )
    source.raise_for_status()
    assert hashlib.sha256(source.content).hexdigest() == (
        "63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715"
    )
    flac_path = tmp_path / "jfk.flac"
    wav_path = tmp_path / "jfk.wav"
    flac_path.write_bytes(source.content)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(flac_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(wav_path),
        ],
        check=True,
        timeout=30,
    )
    endpoint = agents_conftest._resolve_whisper_inference_endpoint(vllm_sidecar)
    deps = AudioAnalysisDeps(
        tenant_id="modal:whisper-test",
        whisper_endpoint=endpoint.base_url,
        whisper_headers=dict(endpoint.headers),
        whisper_model=endpoint.model_id,
    )
    agent = AudioAnalysisAgent(deps=deps)

    async def transcribe_concurrently():
        return await asyncio.gather(
            *(agent.transcribe_audio(str(wav_path), language="en") for _ in range(3))
        )

    results = asyncio.run(transcribe_concurrently())

    def normalized(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", value.lower()))

    expected = (
        "and so my fellow americans ask not what your country can do for you "
        "ask what you can do for your country"
    )
    assert tuple(normalized(result.text) for result in results) == (expected,) * 3
    assert tuple(result.language for result in results) == ("en",) * 3
    assert tuple(result.confidence for result in results) == (1.0,) * 3
    assert (
        tuple(
            normalized(" ".join(segment["text"] for segment in result.segments))
            for result in results
        )
        == (expected,) * 3
    )
