"""The dispatcher searches the tenant's audio profile through its own schema
and acoustic service, picked the way the search backend picks it."""

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


def test_the_shipped_default_audio_profile_searches_clap():
    assert audio_search_target(SHIPPED, URLS) == {
        "audio_profile": "audio_clap_semantic",
        "audio_schema": "audio_content",
        "acoustic_service": "clap_embed",
        "acoustic_endpoint": "http://cogniverse-clap-embed:8000",
    }


def test_a_tenant_defaulting_to_the_omni_profile_searches_it():
    backend = {
        **SHIPPED,
        "default_profiles": {"audio": {"profile": "audio_embeddinggemma2"}},
    }

    assert audio_search_target(backend, URLS) == {
        "audio_profile": "audio_embeddinggemma2",
        "audio_schema": "audio_embeddinggemma2",
        "acoustic_service": "vllm_omni_embed",
        "acoustic_endpoint": "http://cogniverse-vllm-omni-embed:8000",
    }


def test_two_audio_profiles_and_no_default_are_refused():
    backend = {**SHIPPED, "default_profiles": {}}

    with pytest.raises(ValueError) as caught:
        audio_search_target(backend, URLS)

    assert str(caught.value) == (
        "Multiple profiles available for type 'audio' but no default configured. "
        "Available profiles: ['audio_clap_semantic', 'audio_embeddinggemma2']. "
        "Either specify 'profile' in query_dict or configure "
        "backend.default_profiles.audio.profile"
    )


def test_an_acoustic_service_with_no_url_leaves_the_endpoint_unset():
    assert audio_search_target(SHIPPED, {}) == {
        "audio_profile": "audio_clap_semantic",
        "audio_schema": "audio_content",
        "acoustic_service": "clap_embed",
        "acoustic_endpoint": None,
    }
