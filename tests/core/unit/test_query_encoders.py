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
    QueryEncoderFactory._encoder_key_locks.clear()
    yield
    QueryEncoderFactory._encoder_cache.clear()
    QueryEncoderFactory._encoder_key_locks.clear()


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


@pytest.mark.unit
@pytest.mark.ci_fast
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


@pytest.mark.unit
@pytest.mark.ci_fast
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


@pytest.mark.unit
@pytest.mark.ci_fast
def test_colbert_encoder_requires_embedding_dim_kwarg():
    """Direct callers must pass embedding_dim; no silent 128 default."""
    with pytest.raises(TypeError):
        ColBERTQueryEncoder("lightonai/LateOn")  # type: ignore[call-arg]


@pytest.mark.unit
@pytest.mark.ci_fast
@patch("cogniverse_core.query.encoders.get_or_load_model")
def test_factory_wires_remote_url_from_inference_service(mock_get_model):
    """Profile.inference_services.embedding -> system_config.inference_service_urls -> loader."""
    mock_get_model.return_value = (MagicMock(), None)
    profile_body = {
        "embedding_model": "lightonai/LateOn",
        "model_loader": "colbert",
        "inference_services": {"embedding": "colbert_pylate"},
        "schema_config": {"embedding_dim": 128},
    }
    config = _build_system_config(
        "lateon_mv",
        profile_body,
        inference_service_urls={"colbert_pylate": "COLBERT_REMOTE_URL"},
    )

    QueryEncoderFactory.create_encoder(profile="lateon_mv", config=config)

    passed_config = mock_get_model.call_args[0][1]
    assert passed_config["remote_inference_url"] == "COLBERT_REMOTE_URL"


@patch("cogniverse_core.query.encoders.get_or_load_model")
def test_factory_routes_shipped_audio_profile_to_colbert(mock_get_model):
    """audio_clap_semantic searches its transcript embedding with ColBERT; the
    acoustic CLAP embedding is a second field the audio agent queries itself.
    The shipped profile must therefore resolve to ColBERT at the transcript
    dimension the schema declares (semantic_embedding v[128])."""
    import json
    from pathlib import Path as _Path

    mock_get_model.return_value = (MagicMock(), None)
    shipped = json.loads(_Path("configs/config.json").read_text())
    profile_body = shipped["backend"]["profiles"]["audio_clap_semantic"]
    assert profile_body["model_loader"] == "colbert"
    config = _build_system_config(
        "audio_clap_semantic",
        profile_body,
        inference_service_urls={"colbert_pylate": "COLBERT_REMOTE_URL"},
    )

    encoder = QueryEncoderFactory.create_encoder(
        profile="audio_clap_semantic", config=config
    )

    assert type(encoder).__name__ == "ColBERTQueryEncoder"
    assert encoder.get_embedding_dim() == 128
    loaded_model_name, passed_config = mock_get_model.call_args[0][:2]
    # The transcript embedding is LateOn (semantic_model); embedding_model on
    # this profile names the CLAP acoustic model, which the pylate sidecar
    # would reject by name.
    assert loaded_model_name == "lightonai/LateOn"
    assert profile_body["embedding_model"] == "laion/clap-htsat-unfused"
    assert passed_config["remote_inference_url"] == "COLBERT_REMOTE_URL"


@patch("cogniverse_core.query.encoders.get_or_load_model")
def test_factory_routes_code_profile_to_code_service(mock_get_model):
    """Code profile's inference_services.embedding=code must hit the code service URL."""
    mock_get_model.return_value = (MagicMock(), None)
    profile_body = {
        "embedding_model": "lightonai/LateOn-Code-edge",
        "model_loader": "colbert",
        "inference_services": {"embedding": "code_colbert_pylate"},
        "schema_config": {"embedding_dim": 48},
    }
    config = _build_system_config(
        "code_lateon_mv",
        profile_body,
        inference_service_urls={
            "colbert_pylate": "COLBERT_REMOTE_URL",
            "code_colbert_pylate": "CODE_COLBERT_REMOTE_URL",
        },
    )

    QueryEncoderFactory.create_encoder(profile="code_lateon_mv", config=config)

    passed_config = mock_get_model.call_args[0][1]
    assert passed_config["remote_inference_url"] == "CODE_COLBERT_REMOTE_URL"


@patch("cogniverse_core.query.encoders.get_or_load_model")
def test_factory_raises_when_inference_service_not_deployed(mock_get_model):
    """Profile names a service that isn't in the URL map → loud error."""
    mock_get_model.return_value = (MagicMock(), None)
    profile_body = {
        "embedding_model": "lightonai/LateOn-Code-edge",
        "model_loader": "colbert",
        "inference_services": {"embedding": "code_colbert_pylate"},
        "schema_config": {"embedding_dim": 48},
    }
    config = _build_system_config(
        "code_lateon_mv",
        profile_body,
        inference_service_urls={"colbert_pylate": "COLBERT_REMOTE_URL"},
    )

    with pytest.raises(
        ValueError, match="inference_services.embedding='code_colbert_pylate'"
    ):
        QueryEncoderFactory.create_encoder(profile="code_lateon_mv", config=config)


@patch("cogniverse_core.query.encoders.get_or_load_model")
def test_factory_leaves_remote_url_unset_when_inference_service_absent(mock_get_model):
    """Profiles without inference_services.embedding fall back to local loading."""
    mock_get_model.return_value = (MagicMock(), None)
    profile_body = {
        "embedding_model": "lightonai/Reason-ModernColBERT",
        "model_loader": "colbert",
        "schema_config": {"embedding_dim": 128},
    }
    config = _build_system_config(
        "document_text_semantic",
        profile_body,
        inference_service_urls={"colbert_pylate": "COLBERT_REMOTE_URL"},
    )

    QueryEncoderFactory.create_encoder(profile="document_text_semantic", config=config)

    passed_config = mock_get_model.call_args[0][1]
    assert "remote_inference_url" not in passed_config


@patch("cogniverse_core.query.encoders.get_or_load_model")
def test_cache_key_separates_profiles_with_same_model_different_routing(mock_get_model):
    """Profiles sharing a model but differing on inference_services or
    embedding_dim must not collapse onto one cached encoder."""
    mock_get_model.return_value = (MagicMock(), None)
    sys_config = MagicMock()
    sys_config.get.return_value = {
        "profiles": {
            "profile_128": {
                "embedding_model": "lightonai/LateOn",
                "model_loader": "colbert",
                "inference_services": {"embedding": "colbert_pylate"},
                "schema_config": {"embedding_dim": 128},
            },
            "profile_64": {
                "embedding_model": "lightonai/LateOn",
                "model_loader": "colbert",
                "inference_services": {"embedding": "colbert_pylate"},
                "schema_config": {"embedding_dim": 64},
            },
        }
    }
    sys_config.inference_service_urls = {"colbert_pylate": "COLBERT_REMOTE_URL"}

    encoder_a = QueryEncoderFactory.create_encoder(
        profile="profile_128", config=sys_config
    )
    encoder_b = QueryEncoderFactory.create_encoder(
        profile="profile_64", config=sys_config
    )

    assert encoder_a is not encoder_b
    assert encoder_a.get_embedding_dim() == 128
    assert encoder_b.get_embedding_dim() == 64


def test_concurrent_cold_start_builds_one_encoder_per_key():
    """8 threads asking for the same key concurrently must trigger exactly ONE
    construction and all receive the same instance. Unlocked get-then-set spanned
    the multi-GB model build, so concurrent cold starts each built a duplicate."""
    import threading
    import time

    profile_body = {
        "embedding_model": "lightonai/Reason-ModernColBERT",
        "model_loader": "colbert",
        "schema_config": {"embedding_dim": 128},
    }
    config = _build_system_config("document_text_semantic", profile_body)

    build_count = 0
    count_lock = threading.Lock()

    def _slow_build(model_name, profile, profile_config, system_config):
        nonlocal build_count
        with count_lock:
            build_count += 1
        time.sleep(0.3)  # the model-load window the race spanned
        return MagicMock()

    results = []
    with patch.object(
        QueryEncoderFactory, "_create_encoder_instance", staticmethod(_slow_build)
    ):

        def _worker():
            results.append(
                QueryEncoderFactory.create_encoder(
                    profile="document_text_semantic", config=config
                )
            )

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert build_count == 1, f"built {build_count} encoders, expected exactly 1"
    assert len(results) == 8
    assert all(r is results[0] for r in results), "threads got different instances"


@patch("cogniverse_core.query.encoders.get_or_load_model")
def test_cache_key_separates_same_service_pointed_at_different_endpoints(
    mock_get_model,
):
    """One service name can resolve to different endpoints across configs (a
    test sidecar vs the deployed one, a dead port during an outage drill). The
    first caller's endpoint must not be handed to the second."""
    mock_get_model.return_value = (MagicMock(), None)
    profile_body = {
        "embedding_model": "lightonai/LateOn",
        "model_loader": "colbert",
        "inference_services": {"embedding": "colbert_pylate"},
        "schema_config": {"embedding_dim": 128},
    }
    live = _build_system_config(
        "audio_clap_semantic",
        profile_body,
        inference_service_urls={"colbert_pylate": "http://127.0.0.1:8111"},
    )
    dead = _build_system_config(
        "audio_clap_semantic",
        profile_body,
        inference_service_urls={"colbert_pylate": "http://127.0.0.1:29071"},
    )

    encoder_live = QueryEncoderFactory.create_encoder(
        profile="audio_clap_semantic", config=live
    )
    encoder_dead = QueryEncoderFactory.create_encoder(
        profile="audio_clap_semantic", config=dead
    )

    assert encoder_live is not encoder_dead
    urls = [
        call[0][1]["remote_inference_url"] for call in mock_get_model.call_args_list
    ]
    assert urls == ["http://127.0.0.1:8111", "http://127.0.0.1:29071"]


def test_concurrent_cold_start_keeps_endpoints_apart():
    """Two endpoints requested concurrently under the same model+service must
    build exactly one encoder each, bound to its own URL."""
    import threading
    import time

    profile_body = {
        "embedding_model": "lightonai/LateOn",
        "model_loader": "colbert",
        "inference_services": {"embedding": "colbert_pylate"},
        "schema_config": {"embedding_dim": 128},
    }
    urls = ["http://127.0.0.1:8111", "http://127.0.0.1:29071"]
    configs = {
        url: _build_system_config(
            "audio_clap_semantic",
            profile_body,
            inference_service_urls={"colbert_pylate": url},
        )
        for url in urls
    }

    builds: list[str] = []
    builds_lock = threading.Lock()
    start = threading.Barrier(8)

    def _slow_build(model_name, profile, profile_config, system_config):
        url = system_config.inference_service_urls["colbert_pylate"]
        with builds_lock:
            builds.append(url)
        time.sleep(0.3)
        encoder = MagicMock()
        encoder.url = url
        return encoder

    results: dict[str, list] = {url: [] for url in urls}
    with patch.object(
        QueryEncoderFactory, "_create_encoder_instance", staticmethod(_slow_build)
    ):

        def _worker(url):
            start.wait()
            results[url].append(
                QueryEncoderFactory.create_encoder(
                    profile="audio_clap_semantic", config=configs[url]
                )
            )

        threads = [
            threading.Thread(target=_worker, args=(urls[i % 2],)) for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert sorted(builds) == sorted(urls), f"built {builds}"
    for url in urls:
        assert len(results[url]) == 4
        assert all(r is results[url][0] for r in results[url])
        assert results[url][0].url == url
    assert results[urls[0]][0] is not results[urls[1]][0]
