"""Media agents must resolve query encoders through QueryEncoderFactory so the
deployed inference sidecar (SystemConfig.inference_service_urls) is used.

Before the fix, DocumentAgent and ImageSearchAgent constructed bare local
encoders -- ColBERTQueryEncoder(embedding_dim=128) and
ColPaliQueryEncoder(model_name=...) -- that never consult
inference_service_urls, so text hits a local pylate ImportError and visual a
"ColQwen3/Tomoro remote-only" RuntimeError in every real deployment while the
sibling /search route resolves the same profile's encoder correctly through the
factory. These pin the agents onto that same wiring, and pin the factory to
fail loud (naming the service) when a profile declares a sidecar but no URL is
configured, instead of silently attempting a local load.
"""

import json
from pathlib import Path

import pytest

from cogniverse_core.query.encoders import QueryEncoderFactory

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

_CONFIG = json.loads(Path("configs/config.json").read_text())
_COLBERT_URL = "http://sentinel-colbert:8000"
_COLPALI_URL = "http://sentinel-colpali:8000"


class _MergedConfig:
    """The ConfigUtils shape create_encoder consumes: ``.get("backend")`` for
    the profile table and ``.inference_service_urls`` for sidecar resolution.
    This is exactly the two-method contract the real merged config exposes."""

    def __init__(self, urls):
        self.inference_service_urls = dict(urls)

    def get(self, key, default=None):
        return _CONFIG.get(key, default)


@pytest.fixture(autouse=True)
def _clear_encoder_cache():
    # create_encoder keys the cache on (model, service, dim), not the URL, so a
    # stale entry from another test would mask the wiring under test.
    QueryEncoderFactory._encoder_cache.clear()
    yield
    QueryEncoderFactory._encoder_cache.clear()


@pytest.fixture
def wired():
    return _MergedConfig({"colbert_pylate": _COLBERT_URL, "vllm_colpali": _COLPALI_URL})


def _document_agent(encoder_config):
    from cogniverse_agents.document_agent import DocumentAgent, DocumentAgentDeps

    return DocumentAgent(
        deps=DocumentAgentDeps(
            vespa_endpoint="http://localhost:8080",
            tenant_id="acme:acme",
            encoder_config=encoder_config,
        )
    )


def _image_agent(encoder_config):
    from cogniverse_agents.image_search_agent import (
        ImageSearchAgent,
        ImageSearchDeps,
    )

    return ImageSearchAgent(
        deps=ImageSearchDeps(
            vespa_endpoint="http://localhost:8080",
            tenant_id="acme:acme",
            encoder_config=encoder_config,
        )
    )


class TestDocumentAgentEncoderWiring:
    def test_text_encoder_routes_through_deployed_colbert_sidecar(self, wired):
        enc = _document_agent(wired).text_query_encoder
        # Remote mode: the ColBERT model is the requests-backed wrapper pointed
        # at the configured colbert_pylate URL, not a local pylate load.
        assert enc.model.__class__.__name__ == "ColBERTRemoteWrapper"
        assert enc.model.endpoint_url == _COLBERT_URL

    def test_visual_encoder_routes_through_deployed_colpali_sidecar(self, wired):
        enc = _document_agent(wired).query_encoder
        assert enc._remote_client is not None
        assert enc._remote_client.endpoint_url == _COLPALI_URL

    def test_declared_sidecar_without_url_fails_loud(self):
        agent = _document_agent(_MergedConfig({}))
        with pytest.raises(ValueError, match="colbert_pylate"):
            _ = agent.text_query_encoder


class TestImageSearchAgentEncoderWiring:
    def test_encoder_routes_through_deployed_colpali_sidecar(self, wired):
        enc = _image_agent(wired).query_encoder
        assert enc._remote_client is not None
        assert enc._remote_client.endpoint_url == _COLPALI_URL

    def test_declared_sidecar_without_url_fails_loud(self):
        agent = _image_agent(_MergedConfig({}))
        with pytest.raises(ValueError, match="vllm_colpali"):
            _ = agent.query_encoder
