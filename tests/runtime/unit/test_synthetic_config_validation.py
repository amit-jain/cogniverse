"""Boundary validation for strict synthetic runtime configuration."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config

pytestmark = pytest.mark.unit


def _deployable_config() -> dict:
    root = Path(__file__).resolve().parents[3]
    return json.loads((root / "configs/config.json").read_text())


@pytest.mark.parametrize(
    "timeout",
    [math.nan, math.inf, -math.inf],
    ids=["nan", "positive-infinity", "negative-infinity"],
)
def test_agent_timeout_rejects_non_finite_numbers(timeout: float) -> None:
    config = _deployable_config()
    config["agents"]["search_agent"]["timeout"] = timeout

    with pytest.raises(ValueError) as error:
        parse_synthetic_runtime_config(config, tenant_id="acme:strict")

    assert str(error.value) == (
        "Invalid synthetic runtime configuration for tenant='acme:strict': "
        "agents.search_agent.timeout must be a finite positive number"
    )


@pytest.mark.parametrize("port", [-1, 0, 65536])
def test_backend_port_rejects_values_outside_tcp_range(port: int) -> None:
    config = _deployable_config()
    config["backend"]["port"] = port

    with pytest.raises(ValueError) as error:
        parse_synthetic_runtime_config(config, tenant_id="acme:strict")

    assert str(error.value) == (
        "Invalid synthetic runtime configuration for tenant='acme:strict': "
        "backend.port must be an integer between 1 and 65535"
    )


@pytest.mark.parametrize("port", [1, 65535])
def test_backend_port_accepts_tcp_range_boundaries(port: int) -> None:
    config = _deployable_config()
    config["backend"]["port"] = port

    parsed = parse_synthetic_runtime_config(config, tenant_id="acme:strict")

    assert parsed.backend_config.port == port


def test_backend_accepts_matching_hydrated_tenant_context() -> None:
    config = _deployable_config()
    config["backend"]["tenant_id"] = "acme:strict"

    parsed = parse_synthetic_runtime_config(config, tenant_id="acme:strict")

    assert parsed.backend_config.tenant_id == "acme:strict"
    assert (
        parsed.backend_config.backend_type,
        parsed.backend_config.url,
        parsed.backend_config.port,
    ) == ("vespa", "http://localhost", 8080)


def test_backend_rejects_mismatched_hydrated_tenant_context() -> None:
    config = _deployable_config()
    config["backend"]["tenant_id"] = "acme:other"

    with pytest.raises(ValueError) as error:
        parse_synthetic_runtime_config(config, tenant_id="acme:strict")

    assert str(error.value) == (
        "Invalid synthetic runtime configuration for tenant='acme:strict': "
        "backend.tenant_id must equal 'acme:strict', got 'acme:other'"
    )


def test_deployable_field_mappings_cover_code_and_wiki_content() -> None:
    parsed = parse_synthetic_runtime_config(
        _deployable_config(), tenant_id="acme:strict"
    )

    assert parsed.generator_config.synthetic_generation_timeout_seconds == 300.0
    assert parsed.generator_config.synthetic_generation_floor_count == 1
    assert parsed.generator_config.field_mappings.topic_fields == [
        "video_title",
        "audio_title",
        "image_title",
        "document_title",
        "chunk_name",
        "title",
    ]
    assert parsed.generator_config.field_mappings.description_fields == [
        "segment_description",
        "image_description",
        "full_text",
        "source_code",
        "content",
        "description",
    ]
