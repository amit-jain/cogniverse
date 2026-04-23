"""Tests for the ColBERT query encoder factory path."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cogniverse_core.query.encoders import (
    ColBERTQueryEncoder,
    QueryEncoderFactory,
)


@pytest.fixture(autouse=True)
def _reset_encoder_cache():
    QueryEncoderFactory._encoder_cache.clear()
    yield
    QueryEncoderFactory._encoder_cache.clear()


def _build_system_config(
    profile_name: str,
    profile_body: dict,
    inference_service_urls: dict | None = None,
) -> MagicMock:
    """Minimal SystemConfig stub returning the provided profile + service URLs."""
    config = MagicMock()
    config.get.return_value = {"profiles": {profile_name: profile_body}}
    config.inference_service_urls = inference_service_urls or {}
    return config


@patch("cogniverse_core.query.encoders.get_or_load_model")
def test_factory_passes_embedding_dim_from_schema_config(mock_get_model):
    mock_get_model.return_value = (MagicMock(), None)
    profile_body = {
        "embedding_model": "lightonai/Reason-ModernColBERT",
        "model_loader": "colbert",
        "schema_config": {"embedding_dim": 128},
    }
    config = _build_system_config("document_text_semantic", profile_body)

    encoder = QueryEncoderFactory.create_encoder(
        profile="document_text_semantic", config=config
    )

    assert isinstance(encoder, ColBERTQueryEncoder)
    assert encoder.get_embedding_dim() == 128


@patch("cogniverse_core.query.encoders.get_or_load_model")
def test_factory_respects_non_128_dim(mock_get_model):
    """LateOn-Code-edge uses 48-dim output; hardcoded 128 would lose data."""
    mock_get_model.return_value = (MagicMock(), None)
    profile_body = {
        "embedding_model": "lightonai/LateOn-Code-edge",
        "model_loader": "colbert",
        "schema_config": {"embedding_dim": 48},
    }
    config = _build_system_config("code_lateon_mv", profile_body)

    encoder = QueryEncoderFactory.create_encoder(
        profile="code_lateon_mv", config=config
    )

    assert encoder.get_embedding_dim() == 48


@patch("cogniverse_core.query.encoders.get_or_load_model")
def test_factory_recognizes_lateon_model_name_without_model_loader(mock_get_model):
    """Profiles with no explicit model_loader must still route LateOn to ColBERT path."""
    mock_get_model.return_value = (MagicMock(), None)
    profile_body = {
        "embedding_model": "lightonai/LateOn",
        "schema_config": {"embedding_dim": 128},
    }
    config = _build_system_config("lateon_mv", profile_body)

    encoder = QueryEncoderFactory.create_encoder(profile="lateon_mv", config=config)

    assert isinstance(encoder, ColBERTQueryEncoder)
    assert encoder.get_embedding_dim() == 128


@patch("cogniverse_core.query.encoders.get_or_load_model")
def test_factory_raises_when_schema_config_missing_embedding_dim(mock_get_model):
    mock_get_model.return_value = (MagicMock(), None)
    profile_body = {
        "embedding_model": "lightonai/Reason-ModernColBERT",
        "model_loader": "colbert",
        "schema_config": {},
    }
    config = _build_system_config("broken_profile", profile_body)

    with pytest.raises(ValueError, match="schema_config.embedding_dim"):
        QueryEncoderFactory.create_encoder(profile="broken_profile", config=config)


@patch("cogniverse_core.query.encoders.get_or_load_model")
def test_factory_raises_when_schema_config_missing_entirely(mock_get_model):
    mock_get_model.return_value = (MagicMock(), None)
    profile_body = {
        "embedding_model": "lightonai/Reason-ModernColBERT",
        "model_loader": "colbert",
    }
    config = _build_system_config("broken_profile", profile_body)

    with pytest.raises(ValueError, match="schema_config.embedding_dim"):
        QueryEncoderFactory.create_encoder(profile="broken_profile", config=config)


def test_colbert_encoder_requires_embedding_dim_kwarg():
    """Direct callers must pass embedding_dim; no silent 128 default."""
    with pytest.raises(TypeError):
        ColBERTQueryEncoder("lightonai/LateOn")  # type: ignore[call-arg]


@patch("cogniverse_core.query.encoders.get_or_load_model")
def test_factory_wires_remote_url_from_inference_service(mock_get_model):
    """Profile.inference_service -> system_config.inference_service_urls -> loader."""
    mock_get_model.return_value = (MagicMock(), None)
    profile_body = {
        "embedding_model": "lightonai/LateOn",
        "model_loader": "colbert",
        "inference_service": "general",
        "schema_config": {"embedding_dim": 128},
    }
    config = _build_system_config(
        "lateon_mv",
        profile_body,
        inference_service_urls={"general": "http://cogniverse-general:8000"},
    )

    QueryEncoderFactory.create_encoder(profile="lateon_mv", config=config)

    passed_config = mock_get_model.call_args[0][1]
    assert passed_config["remote_inference_url"] == "http://cogniverse-general:8000"


@patch("cogniverse_core.query.encoders.get_or_load_model")
def test_factory_routes_code_profile_to_code_service(mock_get_model):
    """Code profile's inference_service=code must hit the code service URL."""
    mock_get_model.return_value = (MagicMock(), None)
    profile_body = {
        "embedding_model": "lightonai/LateOn-Code-edge",
        "model_loader": "colbert",
        "inference_service": "code",
        "schema_config": {"embedding_dim": 48},
    }
    config = _build_system_config(
        "code_lateon_mv",
        profile_body,
        inference_service_urls={
            "general": "http://cogniverse-general:8000",
            "code": "http://cogniverse-code:8000",
        },
    )

    QueryEncoderFactory.create_encoder(profile="code_lateon_mv", config=config)

    passed_config = mock_get_model.call_args[0][1]
    assert passed_config["remote_inference_url"] == "http://cogniverse-code:8000"


@patch("cogniverse_core.query.encoders.get_or_load_model")
def test_factory_raises_when_inference_service_not_deployed(mock_get_model):
    """Profile names a service that isn't in the URL map → loud error."""
    mock_get_model.return_value = (MagicMock(), None)
    profile_body = {
        "embedding_model": "lightonai/LateOn-Code-edge",
        "model_loader": "colbert",
        "inference_service": "code",
        "schema_config": {"embedding_dim": 48},
    }
    config = _build_system_config(
        "code_lateon_mv",
        profile_body,
        inference_service_urls={"general": "http://cogniverse-general:8000"},
    )

    with pytest.raises(ValueError, match="inference_service='code'"):
        QueryEncoderFactory.create_encoder(profile="code_lateon_mv", config=config)


@patch("cogniverse_core.query.encoders.get_or_load_model")
def test_factory_leaves_remote_url_unset_when_inference_service_absent(mock_get_model):
    """Profiles without inference_service fall back to local loading."""
    mock_get_model.return_value = (MagicMock(), None)
    profile_body = {
        "embedding_model": "lightonai/Reason-ModernColBERT",
        "model_loader": "colbert",
        "schema_config": {"embedding_dim": 128},
    }
    config = _build_system_config(
        "document_text_semantic",
        profile_body,
        inference_service_urls={"general": "http://cogniverse-general:8000"},
    )

    QueryEncoderFactory.create_encoder(profile="document_text_semantic", config=config)

    passed_config = mock_get_model.call_args[0][1]
    assert "remote_inference_url" not in passed_config


@patch("cogniverse_core.query.encoders.get_or_load_model")
def test_cache_key_separates_profiles_with_same_model_different_routing(mock_get_model):
    """Profiles sharing a model but differing on inference_service or
    embedding_dim must not collapse onto one cached encoder."""
    mock_get_model.return_value = (MagicMock(), None)
    sys_config = MagicMock()
    sys_config.get.return_value = {
        "profiles": {
            "profile_128": {
                "embedding_model": "lightonai/LateOn",
                "model_loader": "colbert",
                "inference_service": "general",
                "schema_config": {"embedding_dim": 128},
            },
            "profile_64": {
                "embedding_model": "lightonai/LateOn",
                "model_loader": "colbert",
                "inference_service": "general",
                "schema_config": {"embedding_dim": 64},
            },
        }
    }
    sys_config.inference_service_urls = {"general": "http://cogniverse-general:8000"}

    encoder_a = QueryEncoderFactory.create_encoder(
        profile="profile_128", config=sys_config
    )
    encoder_b = QueryEncoderFactory.create_encoder(
        profile="profile_64", config=sys_config
    )

    assert encoder_a is not encoder_b
    assert encoder_a.get_embedding_dim() == 128
    assert encoder_b.get_embedding_dim() == 64
