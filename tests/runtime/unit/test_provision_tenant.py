"""Coverage for the provision_tenant cold-bootstrap script.

The tenant-provisioning WorkflowTemplate previously embedded this logic as
inline Python that drifted against the real APIs (wrong imports, wrong
signatures) and was never executed by a test. These pin the current call
shapes so the script can't silently rot again.
"""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[3] / "scripts" / "provision_tenant.py"


def _load():
    spec = importlib.util.spec_from_file_location("provision_tenant", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_real_lazy_init_memory_accepts_the_kwargs_we_pass():
    from cogniverse_runtime.memory_init import lazy_init_memory

    params = inspect.signature(lazy_init_memory).parameters
    assert {"tenant_id", "config_manager", "auto_create_schema"} <= set(params)


def test_init_memory_dispatch(monkeypatch):
    pt = _load()
    fake_cm = object()
    fake_mgr = object()
    captured = {}

    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        lambda: fake_cm,
    )
    monkeypatch.setattr(
        "cogniverse_core.memory.manager.Mem0MemoryManager",
        lambda tenant_id: fake_mgr,
    )

    def _fake_lazy(mgr, tenant_id, config_manager, auto_create_schema=True):
        captured.update(
            mgr=mgr,
            tenant_id=tenant_id,
            config_manager=config_manager,
            auto_create_schema=auto_create_schema,
        )
        return True

    monkeypatch.setattr("cogniverse_runtime.memory_init.lazy_init_memory", _fake_lazy)

    pt.init_memory("acme")
    assert captured == {
        "mgr": fake_mgr,
        "tenant_id": "acme",
        "config_manager": fake_cm,
        "auto_create_schema": True,
    }


def test_init_memory_raises_when_init_fails(monkeypatch):
    pt = _load()
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        lambda: object(),
    )
    monkeypatch.setattr(
        "cogniverse_core.memory.manager.Mem0MemoryManager", lambda tenant_id: object()
    )
    monkeypatch.setattr(
        "cogniverse_runtime.memory_init.lazy_init_memory",
        lambda *a, **k: False,
    )
    with pytest.raises(RuntimeError, match="acme"):
        pt.init_memory("acme")


def test_init_telemetry_emits_probe_span(monkeypatch):
    pt = _load()
    spans = []

    class _Span:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _TM:
        def span(self, name, *, tenant_id, component):
            spans.append((name, tenant_id, component))
            return _Span()

    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager", lambda: _TM()
    )
    pt.init_telemetry("acme")
    assert spans == [("provision.probe", "acme", "search_service")]


def test_cli_help_loads():
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    assert "--tenant-id" in proc.stdout and "--step" in proc.stdout
