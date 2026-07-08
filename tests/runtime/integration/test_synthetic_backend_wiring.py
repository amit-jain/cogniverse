"""The runtime lifespan must configure the synthetic-data service with a real
search backend.

``/synthetic/generate`` serves the process-global service configured at
startup. When that service is built with ``backend=None`` the BackendQuerier
falls back to a hardcoded mock profile/topic list, so every tenant's generated
training data is fabricated instead of sampled from its Vespa corpus — and the
route still returns HTTP 200, so nothing surfaces the substitution.

This boots the real ``main.py`` lifespan and asserts the global service holds a
non-None backend.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI

pytestmark = pytest.mark.integration


class TestLifespanWiresSyntheticBackend:
    @pytest.mark.asyncio
    async def test_synthetic_service_configured_with_backend(self, monkeypatch):
        # Keep the boot light: skip the sandbox connect and the memory
        # lifecycle scheduler; neither is needed for the synthetic wiring.
        monkeypatch.setenv("COGNIVERSE_SANDBOX_POLICY", "disabled")
        monkeypatch.setenv("COGNIVERSE_MEMORY_LIFECYCLE_DISABLED", "1")

        from cogniverse_synthetic import api as synthetic_api

        app = FastAPI()
        from cogniverse_runtime.main import lifespan

        async with lifespan(app):
            service = synthetic_api._service
            assert service is not None, "lifespan did not configure the service"
            assert service.backend is not None, (
                "synthetic service was configured without a backend — "
                "/synthetic/generate would serve fabricated mock data"
            )
            # The backend config must carry the real backend kind, not the
            # mock default.
            assert service.backend_config is not None
            assert service.backend_config.backend_type == "vespa"
