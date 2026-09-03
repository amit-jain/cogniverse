"""Library modules take configuration as data, not from process env."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cogniverse_runtime import main as runtime_main

TARGET_ENV_VARS = {
    "MINIO_ENDPOINT",
    "MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY",
    "TELEMETRY_OTLP_ENDPOINT",
    "COGNIVERSE_SEMANTIC_EMBED_URL",
    "COGNIVERSE_SEMANTIC_EMBED_MODEL",
}

TARGET_FILES = [
    Path("libs/core/cogniverse_core/common/cache/backends/s3.py"),
    Path("libs/foundation/cogniverse_foundation/telemetry/manager.py"),
    Path("libs/core/cogniverse_core/common/models/semantic_embedder.py"),
    Path("libs/agents/cogniverse_agents/text_analysis_agent.py"),
    Path("libs/core/cogniverse_core/memory/manager.py"),
    Path("libs/core/cogniverse_core/registries/backend_registry.py"),
    Path("libs/foundation/cogniverse_foundation/registry/entry_point_registry.py"),
]

MODULES_SCANNED = 7

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _read_env_vars(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    env_vars: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "get"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "environ"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "os"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                env_vars.add(node.args[0].value)
        elif isinstance(node, ast.Subscript):
            target = node.value
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "environ"
                and isinstance(target.value, ast.Name)
                and target.value.id == "os"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                env_vars.add(node.slice.value)

    return env_vars


def test_target_library_modules_do_not_read_target_env_vars():
    scanned = 0
    offenders: dict[str, list[str]] = {}

    for path in TARGET_FILES:
        scanned += 1
        reads = sorted(_read_env_vars(path) & TARGET_ENV_VARS)
        if reads:
            offenders[str(path)] = reads

    assert scanned == MODULES_SCANNED
    assert offenders == {}


@pytest.fixture
def restore_library_module_defaults():
    """Restore the process-global defaults ``_configure_library_module_defaults`` sets.

    That call pushes resolved values into module-level state (semantic-embedder
    URL/model, S3 credentials, four tenant-cache capacities). ``monkeypatch.setenv``
    restores the environment but cannot undo those writes, so without this fixture
    the injected values leak into every test that runs later in the same process.
    The ``_CONFIGURED_*`` globals have no public reader, so they are snapshotted
    directly and put back through the public ``configure_*`` functions, which also
    rebuild the cache objects the capacity setters replace.
    """
    from cogniverse_agents import _rlm_promotion as rlm_promotion_module
    from cogniverse_agents import text_analysis_agent as text_analysis_module
    from cogniverse_agents._rlm_promotion import configure_rlm_promotion
    from cogniverse_agents.inference import deno_check as deno_check_module
    from cogniverse_agents.inference.deno_check import configure_deno_check
    from cogniverse_agents.text_analysis_agent import (
        configure_tenant_cache_capacity as configure_text_analysis_capacity,
    )
    from cogniverse_core.common.cache.backends import s3 as s3_backend
    from cogniverse_core.common.cache.backends.s3 import configure_s3_backend_defaults
    from cogniverse_core.common.models import semantic_embedder as embedder_module
    from cogniverse_core.common.models.semantic_embedder import (
        configure_semantic_embedder_defaults,
    )
    from cogniverse_core.memory import manager as memory_manager_module
    from cogniverse_core.memory.manager import (
        configure_tenant_cache_capacity as configure_memory_capacity,
    )
    from cogniverse_core.registries import backend_registry as backend_registry_module
    from cogniverse_core.registries.backend_registry import (
        configure_tenant_cache_capacity as configure_backend_registry_capacity,
    )
    from cogniverse_foundation.registry import (
        entry_point_registry as entry_point_registry_module,
    )
    from cogniverse_foundation.registry.entry_point_registry import (
        configure_tenant_cache_capacity as configure_entry_point_capacity,
    )

    original_embedder_url = embedder_module._CONFIGURED_REMOTE_URL
    original_embedder_model = embedder_module._CONFIGURED_MODEL_NAME
    original_s3_endpoint = s3_backend._CONFIGURED_ENDPOINT
    original_s3_access_key = s3_backend._CONFIGURED_ACCESS_KEY
    original_s3_secret_key = s3_backend._CONFIGURED_SECRET_KEY
    original_text_analysis_capacity = text_analysis_module._agent_instances.capacity
    original_memory_capacity = memory_manager_module._CONFIGURED_TENANT_CACHE_CAPACITY
    original_backend_registry_capacity = (
        backend_registry_module._CONFIGURED_TENANT_CACHE_CAPACITY
    )
    original_entry_point_capacity = (
        entry_point_registry_module._CONFIGURED_TENANT_CACHE_CAPACITY
    )
    original_rlm_promotion_enabled = rlm_promotion_module._promotion_enabled
    original_rlm_promotion_fraction = rlm_promotion_module._promotion_fraction
    original_skip_deno_check = deno_check_module._skip_deno_check

    yield

    configure_semantic_embedder_defaults(
        remote_url=original_embedder_url, model_name=original_embedder_model
    )
    configure_s3_backend_defaults(
        endpoint=original_s3_endpoint,
        access_key=original_s3_access_key,
        secret_key=original_s3_secret_key,
    )
    configure_text_analysis_capacity(original_text_analysis_capacity)
    if original_memory_capacity is not None:
        configure_memory_capacity(original_memory_capacity)
    if original_backend_registry_capacity is not None:
        configure_backend_registry_capacity(original_backend_registry_capacity)
    if original_entry_point_capacity is not None:
        configure_entry_point_capacity(original_entry_point_capacity)
    configure_rlm_promotion(
        enabled=original_rlm_promotion_enabled,
        fraction=original_rlm_promotion_fraction,
    )
    configure_deno_check(skip=original_skip_deno_check)


def test_runtime_main_resolves_and_injects_target_env_vars(
    monkeypatch, restore_library_module_defaults
):
    monkeypatch.setenv("MINIO_ENDPOINT", "http://minio.internal:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minio-access")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minio-secret")
    monkeypatch.setenv("TELEMETRY_OTLP_ENDPOINT", "wired-phoenix:4317")
    monkeypatch.setenv("TELEMETRY_HTTP_ENDPOINT", "http://wired-phoenix:6006")
    monkeypatch.setenv("COGNIVERSE_SEMANTIC_EMBED_URL", "http://embed.internal:8000")
    monkeypatch.setenv("COGNIVERSE_SEMANTIC_EMBED_MODEL", "from-config")
    monkeypatch.setenv("COGNIVERSE_TENANT_CACHE_CAPACITY", "23")
    monkeypatch.setenv("COGNIVERSE_ORCH_RLM_PROMOTION", "disabled")
    monkeypatch.setenv("COGNIVERSE_ORCH_RLM_PROMOTION_FRACTION", "0.5")
    monkeypatch.setenv("COGNIVERSE_RLM_SKIP_DENO_CHECK", "true")

    resolved = runtime_main._resolve_library_env_defaults()

    assert resolved == {
        "minio_endpoint": "http://minio.internal:9000",
        "minio_access_key": "minio-access",
        "minio_secret_key": "minio-secret",
        "telemetry_otlp_endpoint": "wired-phoenix:4317",
        "telemetry_http_endpoint": "http://wired-phoenix:6006",
        "semantic_embed_url": "http://embed.internal:8000",
        "semantic_embed_model": "from-config",
        "tenant_cache_capacity": 23,
        "rlm_promotion_enabled": False,
        "rlm_promotion_fraction": 0.5,
        "rlm_skip_deno_check": True,
    }

    monkeypatch.setattr(
        runtime_main, "get_telemetry_manager", lambda *args, **kwargs: object()
    )

    runtime_main._configure_library_module_defaults(
        object(),
        **resolved,
    )

    from cogniverse_agents.text_analysis_agent import _agent_instances
    from cogniverse_core.memory.manager import Mem0MemoryManager
    from cogniverse_core.registries.backend_registry import BackendRegistry
    from cogniverse_foundation.registry.entry_point_registry import (
        EntryPointRegistry,
    )

    assert _agent_instances.capacity == 23
    assert Mem0MemoryManager._instances.capacity == 23
    assert BackendRegistry._backend_instances.capacity == 23
    assert EntryPointRegistry._instances.capacity == 23

    from cogniverse_agents import _rlm_promotion
    from cogniverse_agents.inference import deno_check

    assert _rlm_promotion._promotion_enabled is False
    assert _rlm_promotion._promotion_fraction == 0.5
    assert deno_check._skip_deno_check is True


def test_injected_library_defaults_do_not_leak_into_later_tests():
    """The injected sentinels must not survive the test that installs them.

    Defined immediately after the injection test so pytest's in-file definition
    order runs it second. Without ``restore_library_module_defaults`` the values
    written by ``_configure_library_module_defaults`` persist for the remainder of
    the process, and every later test resolving an embedder or an S3 client picks
    up ``from-config`` / ``http://embed.internal:8000`` instead of its own.
    """
    from cogniverse_agents import text_analysis_agent as text_analysis_module
    from cogniverse_core.common.cache.backends import s3 as s3_backend
    from cogniverse_core.common.models import semantic_embedder as embedder_module

    assert embedder_module._CONFIGURED_REMOTE_URL != "http://embed.internal:8000"
    assert embedder_module._CONFIGURED_MODEL_NAME != "from-config"
    assert s3_backend._CONFIGURED_ENDPOINT != "http://minio.internal:9000"
    assert s3_backend._CONFIGURED_ACCESS_KEY != "minio-access"
    assert s3_backend._CONFIGURED_SECRET_KEY != "minio-secret"
    assert text_analysis_module._agent_instances.capacity != 23

    from cogniverse_agents import _rlm_promotion
    from cogniverse_agents.inference import deno_check

    assert _rlm_promotion._promotion_enabled is True
    assert _rlm_promotion._promotion_fraction == 0.75
    assert deno_check._skip_deno_check is False
