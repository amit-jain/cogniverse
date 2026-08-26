from concurrent.futures import ThreadPoolExecutor

import pytest

from cogniverse_foundation.config.bootstrap import (
    inference_api_key_from_environment,
)
from cogniverse_foundation.config.inference_auth import (
    inference_headers,
    is_modal_inference_url,
)


def test_bootstrap_reads_one_canonical_inference_key(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "shared-production-key")

    assert inference_api_key_from_environment() == "shared-production-key"

    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", " key-with-whitespace ")
    with pytest.raises(
        RuntimeError,
        match="Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY",
    ):
        inference_api_key_from_environment()


def test_modal_endpoint_requires_https_before_reading_the_bearer_key(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "must-not-be-transmitted")

    with pytest.raises(
        ValueError,
        match="Modal inference endpoints require HTTPS",
    ):
        inference_headers("http://service.modal.run")


@pytest.mark.parametrize(
    "endpoint_url, expected",
    (
        ("https://service.modal.run", True),
        ("http://service.modal.run", False),
        ("https://service.example.com", False),
        ("https://service.modal.run/v1", False),
    ),
)
def test_modal_inference_url_classifies_only_https_modal_roots(endpoint_url, expected):
    assert is_modal_inference_url(endpoint_url) is expected


def test_modal_endpoint_requires_one_canonical_bearer_key(monkeypatch):
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)

    with pytest.raises(
        RuntimeError,
        match=("Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY"),
    ):
        inference_headers("https://service.modal.run")

    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", " key-with-whitespace ")
    with pytest.raises(
        RuntimeError,
        match=("Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY"),
    ):
        inference_headers("https://service.modal.run")


def test_concurrent_modal_header_resolution_is_exact_and_immutable(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "shared-production-key")

    with ThreadPoolExecutor(max_workers=16) as pool:
        resolved = tuple(
            pool.map(
                lambda _: inference_headers("https://service.modal.run"),
                range(32),
            )
        )

    assert (
        tuple(dict(headers) for headers in resolved)
        == ({"Authorization": "Bearer shared-production-key"},) * 32
    )
    with pytest.raises(TypeError):
        resolved[0]["Authorization"] = "Bearer replacement"


@pytest.mark.parametrize(
    "endpoint_url",
    (
        "http://gliner:8080",
        "http://127.0.0.1:29007",
        "https://modal.run.evil.example",
    ),
)
def test_non_modal_endpoints_receive_no_modal_credential(monkeypatch, endpoint_url):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "shared-production-key")

    assert dict(inference_headers(endpoint_url)) == {}


@pytest.mark.parametrize(
    "endpoint_url",
    (
        "service.modal.run",
        "https://user:password@service.modal.run",
        "https://service.modal.run/v1",
        "https://service.modal.run?tenant=one",
        "https://service.modal.run#fragment",
    ),
)
def test_inference_endpoint_must_be_a_root_http_url(endpoint_url):
    with pytest.raises(ValueError, match=r"root HTTP\(S\) URL"):
        inference_headers(endpoint_url)
