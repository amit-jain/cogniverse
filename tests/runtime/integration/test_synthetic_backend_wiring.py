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

    @pytest.fixture
    def _dspy_ambient_state(self):
        """Snapshot dspy's ambient-config globals and restore them on exit.

        The test below claims the ambient ownership slot and binds a dead LM;
        without a restore, both would leak into every later test in the
        session (the exact leak class the production fix addresses).
        """
        import importlib

        # ``dspy.dsp.utils`` re-exports ``settings`` as the Settings
        # singleton; the ownership globals live on the module itself.
        dspy_settings = importlib.import_module("dspy.dsp.utils.settings")

        saved_config = dict(dspy_settings.main_thread_config)
        saved_thread = dspy_settings.config_owner_thread_id
        saved_task = dspy_settings.config_owner_async_task
        # Release any ownership an earlier test's lifespan claimed so the
        # claim task below becomes the first (and thus owning) configurer.
        dspy_settings.config_owner_async_task = None
        yield
        dspy_settings.main_thread_config.clear()
        dspy_settings.main_thread_config.update(saved_config)
        dspy_settings.config_owner_thread_id = saved_thread
        dspy_settings.config_owner_async_task = saved_task

    @pytest.mark.asyncio
    async def test_boot_completes_when_another_task_owns_dspy_ambient(
        self, monkeypatch, _dspy_ambient_state
    ):
        """The lifespan must boot when dspy's ambient slot is already claimed.

        ``dspy.configure`` grants ambient-binding ownership to the first async
        task that calls it. In a process that runs several event-loop tasks
        (multiple test lifespans, a worker job before the API), the lifespan's
        own configure call is not the first — it must fall back to
        first-writer-wins instead of aborting the boot, leaving the already
        bound ambient LM in place and still wiring the synthetic service.
        """
        import asyncio

        import dspy

        monkeypatch.setenv("COGNIVERSE_SANDBOX_POLICY", "disabled")
        monkeypatch.setenv("COGNIVERSE_MEMORY_LIFECYCLE_DISABLED", "1")

        import cogniverse_runtime.main as runtime_main
        from cogniverse_synthetic import api as synthetic_api

        # Claim dspy's async-task ownership from a sibling task, the state a
        # prior lifespan or worker job leaves behind in this process.
        claimed_lm = dspy.LM(
            "openai/ambient-owner", api_base="http://127.0.0.1:29071/v1"
        )

        async def claim_ambient() -> None:
            dspy.configure(lm=claimed_lm)

        await asyncio.create_task(claim_ambient())
        monkeypatch.setattr(runtime_main, "_DSPY_AMBIENT_CONFIGURED", False)

        app = FastAPI()
        async with runtime_main.lifespan(app):
            assert dspy.settings.lm is claimed_lm, (
                "the ambient LM bound by the owning task must survive the boot"
            )
            service = synthetic_api._service
            assert service is not None, "lifespan did not configure the service"
            assert service.backend is not None
            assert service.backend_config.backend_type == "vespa"
