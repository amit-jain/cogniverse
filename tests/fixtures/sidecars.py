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
