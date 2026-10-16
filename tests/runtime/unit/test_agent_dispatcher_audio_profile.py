"""The dispatcher searches the tenant's deployed audio profile through its own
schema and acoustic service: the only one deployed, else the configured
default among several."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cogniverse_runtime.agent_dispatcher import audio_search_target

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

SHIPPED = json.loads(Path("configs/config.json").read_text())["backend"]
URLS = {
    "clap_embed": "http://cogniverse-clap-embed:8000",
    "vllm_omni_embed": "http://cogniverse-vllm-omni-embed:8000",
}
CLAP_TARGET = {
    "audio_profile": "audio_clap_semantic",
    "audio_schema": "audio_content",
    "acoustic_service": "clap_embed",
    "acoustic_endpoint": "http://cogniverse-clap-embed:8000",
}
OMNI_TARGET = {
    "audio_profile": "audio_embeddinggemma2",
    "audio_schema": "audio_embeddinggemma2",
    "acoustic_service": "vllm_omni_embed",
    "acoustic_endpoint": "http://cogniverse-vllm-omni-embed:8000",
}


def _deployed(*schemas: str):
    return lambda schema: schema in schemas


def test_both_deployed_searches_the_shipped_default_clap_profile():
    target = audio_search_target(
        SHIPPED, URLS, _deployed("audio_content", "audio_embeddinggemma2")
    )

    assert target == CLAP_TARGET


def test_a_tenant_defaulting_to_the_omni_profile_searches_it():
    backend = {
        **SHIPPED,
        "default_profiles": {"audio": {"profile": "audio_embeddinggemma2"}},
    }

    target = audio_search_target(
        backend, URLS, _deployed("audio_content", "audio_embeddinggemma2")
    )

    assert target == OMNI_TARGET


def test_a_tenant_with_only_the_omni_schema_searches_it_despite_the_default():
    assert (
        audio_search_target(SHIPPED, URLS, _deployed("audio_embeddinggemma2"))
        == OMNI_TARGET
    )


def test_a_tenant_with_only_the_clap_schema_searches_it_despite_an_omni_default():
    backend = {
        **SHIPPED,
        "default_profiles": {"audio": {"profile": "audio_embeddinggemma2"}},
    }

    assert audio_search_target(backend, URLS, _deployed("audio_content")) == (
        CLAP_TARGET
    )


def test_a_tenant_with_no_audio_schema_has_no_target():
    assert (
        audio_search_target(SHIPPED, URLS, _deployed("video_colpali_smol500_mv_frame"))
        is None
    )


def test_two_deployed_audio_profiles_and_no_default_are_refused():
    backend = {**SHIPPED, "default_profiles": {}}

    with pytest.raises(ValueError) as caught:
        audio_search_target(
            backend, URLS, _deployed("audio_content", "audio_embeddinggemma2")
        )

    assert str(caught.value) == (
        "Multiple profiles available for type 'audio' but no default configured. "
        "Available profiles: ['audio_clap_semantic', 'audio_embeddinggemma2']. "
        "Either specify 'profile' in query_dict or configure "
        "backend.default_profiles.audio.profile"
    )


def test_an_acoustic_service_with_no_url_leaves_the_endpoint_unset():
    target = audio_search_target(SHIPPED, {}, _deployed("audio_content"))

    assert target == {**CLAP_TARGET, "acoustic_endpoint": None}
