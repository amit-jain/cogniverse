"""Strict parsing tests for configured inference service endpoints."""

from __future__ import annotations

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
