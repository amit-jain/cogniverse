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


def test_runtime_main_resolves_and_injects_target_env_vars(monkeypatch):
    monkeypatch.setenv("MINIO_ENDPOINT", "http://minio.internal:9000")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "minio-access")
    monkeypatch.setenv("MINIO_SECRET_KEY", "minio-secret")
    monkeypatch.setenv("TELEMETRY_OTLP_ENDPOINT", "wired-phoenix:4317")
    monkeypatch.setenv("TELEMETRY_HTTP_ENDPOINT", "http://wired-phoenix:6006")
    monkeypatch.setenv("COGNIVERSE_SEMANTIC_EMBED_URL", "http://embed.internal:8000")
    monkeypatch.setenv("COGNIVERSE_SEMANTIC_EMBED_MODEL", "from-config")
    monkeypatch.setenv("COGNIVERSE_TENANT_CACHE_CAPACITY", "23")

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
