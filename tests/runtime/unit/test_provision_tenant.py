"""Coverage for the provision_tenant cold-bootstrap script.

The tenant-provisioning WorkflowTemplate previously embedded this logic as
inline Python that drifted against the real APIs (wrong imports, wrong
signatures) and was never executed by a test. These pin the current call
shapes so the script can't silently rot again.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
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
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("denseon missing")),
    )
    with pytest.raises(RuntimeError, match="denseon missing"):
        pt.init_memory("acme")


def _telemetry_manager(spans, *, fail=None):
    """A manager whose ``required_span`` records the exact probe it exported."""
    from contextlib import asynccontextmanager

    class _TM:
        @asynccontextmanager
        async def required_span(self, name, *, tenant_id):
            if fail is not None:
                raise fail
            spans.append((name, tenant_id))
            yield object()

    return _TM()


def test_init_telemetry_exports_the_probe_under_the_canonical_tenant(monkeypatch):
    """The step must export, not merely open, the probe: a project Phoenix
    never received is a project that does not exist."""
    pt = _load()
    spans = []
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        lambda: object(),
    )
    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        lambda *args, **kwargs: _telemetry_manager(spans),
    )
    pt.init_telemetry("acme")
    assert spans == [("provision.probe", "acme:acme")]


def test_init_telemetry_reports_a_collector_outage_as_a_failure(monkeypatch):
    pt = _load()
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        lambda: object(),
    )
    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        lambda *args, **kwargs: _telemetry_manager(
            [],
            fail=RuntimeError(
                "Required telemetry export failed: endpoint=1.2.3.4:4317"
            ),
        ),
    )
    with pytest.raises(RuntimeError) as raised:
        pt.init_telemetry("acme")
    assert str(raised.value) == (
        "Provisioning telemetry failed for tenant acme:acme: "
        "Required telemetry export failed: endpoint=1.2.3.4:4317"
    )


def _run_cli(*args, env=None):
    base = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("BACKEND_", "VESPA_", "TELEMETRY_"))
    }
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=base | (env or {}),
        cwd=_SCRIPT.parents[1],
    )


@pytest.mark.parametrize(
    "step,args",
    [
        ("schemas", ("--profiles", "video_colpali_smol500_mv_frame")),
        ("verify", ("--profiles", "video_colpali_smol500_mv_frame")),
        ("telemetry", ()),
        ("memory", ()),
        ("tier", ("--tier", "pro")),
    ],
)
def test_a_step_without_its_backend_endpoint_reports_the_cause_not_a_crash(step, args):
    """Every step builds a ConfigManager, so a step the workflow launched
    without ``BACKEND_URL`` must exit 1 naming the missing variable. A
    traceback is an unhandled failure the workflow reports as a crash."""
    proc = _run_cli("--tenant-id", "acme", "--step", step, *args)
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert "Traceback" not in proc.stderr
    assert proc.stderr.strip() == (
        "Provisioning failed for tenant acme:acme: BACKEND_URL environment "
        "variable is required. Set it to your backend server URL, e.g., "
        "BACKEND_URL=http://localhost"
    )


def test_an_unknown_tier_is_reported_without_touching_a_store():
    from cogniverse_foundation.config.unified_config import ROUTER_TIERS

    proc = _run_cli("--tenant-id", "acme", "--step", "tier", "--tier", "platinum")
    assert proc.returncode == 1
    assert proc.stdout == ""
    assert "Traceback" not in proc.stderr
    assert proc.stderr.strip() == (
        f"Unknown router tier 'platinum'. Valid tiers: {sorted(ROUTER_TIERS)}"
    )


def test_cli_help_loads():
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    assert "--tenant-id" in proc.stdout and "--step" in proc.stdout


class TestTheTierStep:
    """``--step tier`` stores the tenant's router tier through the same seam
    the admin route uses; the vocabulary is checked before any store exists."""

    def test_help_lists_the_tier_argument(self):
        proc = subprocess.run(
            [sys.executable, str(_SCRIPT), "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0
        assert "--tier" in proc.stdout
        assert "tier" in proc.stdout.split("--step")[1]

    def test_the_tier_step_without_a_tier_is_an_argument_error(self):
        proc = subprocess.run(
            [sys.executable, str(_SCRIPT), "--tenant-id", "acme", "--step", "tier"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 2
        assert "--step tier requires --tier" in proc.stderr

    def test_a_tier_outside_the_vocabulary_is_refused_before_any_store(
        self, monkeypatch
    ):
        from cogniverse_foundation.config.unified_config import ROUTER_TIERS

        pt = _load()

        def _no_store():
            raise AssertionError("the store must not be built for a refused tier")

        monkeypatch.setattr(
            "cogniverse_foundation.config.utils.create_default_config_manager",
            _no_store,
        )
        with pytest.raises(ValueError) as raised:
            pt.init_tier("acme", "platinum")
        assert str(raised.value) == (
            f"Unknown router tier 'platinum'. Valid tiers: {sorted(ROUTER_TIERS)}"
        )
