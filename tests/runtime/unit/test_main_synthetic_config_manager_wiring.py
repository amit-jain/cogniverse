"""main.py wires the shared config manager into the synthetic data service.

``/synthetic/generate`` for the ``profile`` and ``cross_modal`` optimizers
resolves the tenant's usable profiles through the config manager, so the
runtime must hand its own manager to ``configure_service``. Without it the
service is built with ``config_manager=None`` and every profile or cross-modal
request fails with "ProfileGenerator requires backend config_manager for
tenant profile selection".
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime import main as runtime_main
from cogniverse_runtime.synthetic_config import SyntheticRuntimeConfig
from cogniverse_telemetry_phoenix.provider import PhoenixProvider
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _AbortStartup(RuntimeError):
    pass


class _FakeSchemaManager:
    def upload_metadata_schemas(self, *args, **kwargs) -> None:
        return None


class _FakeBackend:
    def __init__(self) -> None:
        self.schema_manager = _FakeSchemaManager()


class _FakeBackendRegistry:
    def __init__(self) -> None:
        self._backend_instances = {}

    def list_backends(self):
        return []

    def get_ingestion_backend(self, *args, **kwargs):
        return _FakeBackend()

    def get_search_backend(self, *args, **kwargs):
        return _FakeBackend()


class _FakeBackendConfig:
    backend_type = "vespa"


class _FakeConfigLoader:
    def load_backends(self) -> None:
        return None

    def load_agents(self, agent_registry=None) -> None:
        return None


@pytest.mark.asyncio
async def test_lifespan_passes_config_manager_to_synthetic_service(
    monkeypatch: pytest.MonkeyPatch,
):
    """The manager reaching configure_service is the runtime's own instance."""
    captured: dict = {}
    config_manager = ConfigManager(store=InMemoryConfigStore())

    def _spy_configure_synthetic(**kwargs):
        captured["kwargs"] = kwargs
        raise _AbortStartup("stop after synthetic wiring")

    async def _fake_wait_for_backend_startup(*args, **kwargs):
        return runtime_main.BackendStartupState.FEED_READY

    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        lambda: config_manager,
    )
    monkeypatch.setattr(
        runtime_main.BackendRegistry, "get_instance", lambda: _FakeBackendRegistry()
    )
    monkeypatch.setattr(runtime_main, "get_config_loader", lambda: _FakeConfigLoader())
    monkeypatch.setattr(PhoenixProvider, "initialize", lambda self, config: None)
    monkeypatch.setattr(
        runtime_main, "_wait_for_backend_startup", _fake_wait_for_backend_startup
    )
    monkeypatch.setattr(
        "cogniverse_synthetic.api.configure_service", _spy_configure_synthetic
    )
    # Synthetic config parsing has its own tests; this one pins the wiring that
    # follows it, so the parsed config is supplied directly.
    monkeypatch.setattr(
        runtime_main,
        "parse_synthetic_runtime_config",
        lambda *args, **kwargs: SyntheticRuntimeConfig(
            backend_config=_FakeBackendConfig(),
            backend_default_profiles={},
            generator_config=object(),
            agents_config={},
        ),
    )

    with pytest.raises(_AbortStartup, match="stop after synthetic wiring"):
        async with runtime_main.lifespan(FastAPI()):
            pass

    assert captured["kwargs"]["config_manager"] is config_manager


def test_runtime_and_cli_wire_the_same_synthetic_arguments():
    """main.py's comment claims it mirrors the CLI's wiring; pin that claim.

    The two call sites drifted once already: the CLI passed ``config_manager``
    and the runtime did not, so every ``/synthetic/generate`` request for the
    ``profile`` and ``cross_modal`` optimizers failed while the CLI worked.
    """
    import ast
    import pathlib

    def call_kwargs(relative_path: str, func_name: str) -> set[str]:
        root = pathlib.Path(__file__).resolve().parents[3]
        tree = ast.parse((root / relative_path).read_text())
        found: list[set[str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = node.func
            name = getattr(target, "id", None) or getattr(target, "attr", None)
            if name == func_name:
                found.append({kw.arg for kw in node.keywords if kw.arg})
        assert len(found) == 1, f"{func_name} in {relative_path}: {len(found)} calls"
        return found[0]

    runtime_args = call_kwargs(
        "libs/runtime/cogniverse_runtime/main.py", "configure_synthetic"
    )
    cli_args = call_kwargs(
        "libs/runtime/cogniverse_runtime/optimization_cli.py", "SyntheticDataService"
    )

    assert runtime_args == cli_args
    assert runtime_args == {
        "backend",
        "config_manager",
        "backend_config",
        "generator_config",
        "agents_config",
        "entity_extractor",
        "routing_decider",
        "query_enhancer",
        "profile_labeler",
    }
