"""Session-scoped sidecar fixtures shared across all test subsuites.

Registered as ``pytest_plugins`` from each subsuite's conftest so the
fixtures are discoverable regardless of whether ``tests/ingestion/`` /
``tests/agents/`` etc. set their own pytest rootdir.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def vllm_sidecar():
    """Factory for spinning up real vLLM sidecars on demand. See
    tests/utils/vllm_sidecar.py for usage details."""
    from tests.utils.vllm_sidecar import VllmSidecarFactory

    factory = VllmSidecarFactory()
    try:
        yield factory
    finally:
        factory.teardown()


@pytest.fixture(scope="session")
def pylate_server():
    """LateOn served by the real PyLate sidecar container (deploy/pylate,
    the same engine the chart deploys) exposing the production ``/pooling``
    contract — session-scoped so LateOn loads once per run.

    The service owns PyLate's exact encode for both directions: query
    expansion over masked padding positions and the document punctuation
    skiplist. Generic vLLM ``/pooling`` cannot reproduce the query side
    because its request schema carries no attention mask. Integration tests
    provision their own inference; the cluster belongs to the e2e tier.
    """
    from cogniverse_foundation.inference_specs import get_inference_service_spec
    from tests.fixtures.inference import LocalEndpointProvider

    provider = LocalEndpointProvider()
    try:
        endpoint = provider.resolve(get_inference_service_spec("colbert_pylate"))
        yield endpoint.base_url
    finally:
        provider.close()


@pytest.fixture(scope="module")
def served_code_colbert():
    """Serve the pinned code encoder through the test-owned CPU PyLate service."""
    from cogniverse_core.common.models.model_loaders import RemoteColBERTLoader
    from cogniverse_foundation.inference_specs import get_inference_service_spec
    from tests.fixtures.inference import LocalEndpointProvider

    provider = LocalEndpointProvider()
    try:
        endpoint = provider.resolve(get_inference_service_spec("code_colbert_pylate"))
        loader = RemoteColBERTLoader(
            model_name=endpoint.model_id,
            config={"remote_inference_url": endpoint.base_url},
            _resolved_headers=dict(endpoint.headers),
        )
        model, _ = loader.load_model()
        try:
            yield model
        finally:
            model._close()
    finally:
        provider.close()
