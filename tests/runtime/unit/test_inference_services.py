"""Strict parsing tests for configured inference service endpoints."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import pytest

from cogniverse_runtime.inference_services import parse_inference_service_urls


def test_absent_configuration_has_no_override() -> None:
    assert parse_inference_service_urls(None) is None


def test_valid_configuration_preserves_exact_service_names_and_urls() -> None:
    raw = (
        '{"tomoro_vllm":"http://127.0.0.1:31005/v1",'
        '"denseon":"https://models.example.test:8443/embed?pool=mean"}'
    )

    assert parse_inference_service_urls(raw) == {
        "tomoro_vllm": "http://127.0.0.1:31005/v1",
        "denseon": "https://models.example.test:8443/embed?pool=mean",
    }


def test_explicit_empty_object_is_valid() -> None:
    assert parse_inference_service_urls("{}") == {}


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("", "valid JSON object"),
        ("not-json", "valid JSON object"),
        ("[]", "JSON object"),
        ('{"denseon": null}', "URL must be a string"),
        ('{"denseon": ""}', "URL must not be empty"),
        ('{"denseon": "relative/path"}', "absolute HTTP or HTTPS URL"),
        ('{"denseon": "ftp://models.example.test"}', "absolute HTTP or HTTPS URL"),
        ('{"denseon": "http:///missing-host"}', "absolute HTTP or HTTPS URL"),
        ('{"denseon": "http://user:secret@models.example.test"}', "credentials"),
        ('{"denseon": "http://models.example.test/v1#models"}', "fragment"),
        ('{"denseon": " http://models.example.test"}', "whitespace"),
        ('{"denseon": "http://models.example.test/bad path"}', "whitespace"),
        ('{"denseon": "http://models.example.test:invalid"}', "valid port"),
    ],
)
def test_invalid_url_configuration_is_rejected(raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_inference_service_urls(raw)


@pytest.mark.parametrize(
    "raw",
    [
        '{"": "http://models.example.test"}',
        '{"   ": "http://models.example.test"}',
        '{" denseon": "http://models.example.test"}',
        '{"denseon ": "http://models.example.test"}',
    ],
)
def test_invalid_service_name_is_rejected(raw: str) -> None:
    with pytest.raises(ValueError, match="service name"):
        parse_inference_service_urls(raw)


def test_duplicate_service_name_is_rejected() -> None:
    raw = '{"denseon":"http://one.example.test","denseon":"http://two.example.test"}'

    with pytest.raises(ValueError, match="duplicate service name 'denseon'"):
        parse_inference_service_urls(raw)


@pytest.mark.parametrize("port", [0, 65536])
def test_explicit_port_outside_tcp_range_is_rejected(port: int) -> None:
    raw = f'{{"denseon":"http://models.example.test:{port}/v1"}}'

    with pytest.raises(ValueError) as error:
        parse_inference_service_urls(raw)

    assert str(error.value) == (
        "INFERENCE_SERVICE_URLS['denseon'] URL must use a valid port"
    )


@pytest.mark.parametrize("port", [1, 65535])
def test_explicit_port_accepts_tcp_range_boundaries(port: int) -> None:
    url = f"http://models.example.test:{port}/v1"

    assert parse_inference_service_urls(f'{{"denseon":"{url}"}}') == {"denseon": url}


@pytest.mark.asyncio
async def test_runtime_rejects_invalid_configuration_before_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fastapi import FastAPI

    import cogniverse_runtime.main as runtime_main

    async def fail_if_backend_is_consulted(*args, **kwargs):
        raise AssertionError("backend startup was consulted")

    monkeypatch.setenv("INFERENCE_SERVICE_URLS", "not-json")
    monkeypatch.setattr(
        runtime_main,
        "_wait_for_backend_startup",
        fail_if_backend_is_consulted,
    )

    with pytest.raises(ValueError, match="valid JSON object"):
        async with runtime_main.lifespan(FastAPI()):
            pass


@pytest.mark.asyncio
async def test_worker_rejects_invalid_configuration_before_redis_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cogniverse_runtime.ingestion_worker import worker

    async def fail_if_redis_is_consulted(url):
        raise AssertionError(f"Redis was consulted at {url}")

    monkeypatch.setenv("INFERENCE_SERVICE_URLS", "[]")
    monkeypatch.setenv("REDIS_URL", "redis://redis.test:6379/0")
    monkeypatch.setattr(worker, "get_redis", fail_if_redis_is_consulted)

    with pytest.raises(ValueError, match="JSON object"):
        await worker.run(stop=asyncio.Event(), processor=object())


def test_worker_process_rejects_invalid_configuration_before_redis() -> None:
    env = os.environ.copy()
    env.update(
        {
            "INFERENCE_SERVICE_URLS": "not-json",
            "REDIS_URL": "redis://127.0.0.1:1/0",
        }
    )

    result = subprocess.run(
        [sys.executable, "-m", "cogniverse_runtime.ingestion_worker.worker"],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "INFERENCE_SERVICE_URLS must be a valid JSON object" in combined
    assert "Connection refused" not in combined


def test_model_discovery_rejects_invalid_configuration_before_cluster_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.utils import vllm_sidecar

    def fail_if_cluster_is_consulted(model: str) -> tuple[str, ...]:
        raise AssertionError(f"cluster discovery was consulted for {model}")

    monkeypatch.setenv("INFERENCE_SERVICE_URLS", '{"denseon":"relative/path"}')
    monkeypatch.delenv("TEST_LLM_MODEL", raising=False)
    monkeypatch.delenv("TEST_LLM_API_BASE", raising=False)
    monkeypatch.setattr(
        vllm_sidecar,
        "_discover_e2e_model_urls",
        fail_if_cluster_is_consulted,
    )
    monkeypatch.setattr(
        vllm_sidecar,
        "_discover_dev_model_urls",
        fail_if_cluster_is_consulted,
    )

    with pytest.raises(ValueError, match="absolute HTTP or HTTPS URL"):
        vllm_sidecar._configured_model_urls("TomoroAI/tomoro-colqwen3-embed-4b")
