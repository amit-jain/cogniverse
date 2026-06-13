"""Read-only routes must never deploy/redeploy schemas.

Deploying a tenant schema triggers a Vespa global app-redeploy that
reconfigures the content cluster and can drop documents another process
just fed but Vespa hasn't flushed — so a read that redeploys silently
loses the very rows it was asked to return. The reconcile / audit /
citation / summarize / traverse / promote / restore / pin routes are
read paths; their lazy init MUST pass ``auto_create_schema=False`` (Mem0)
and ``deploy=False`` (graph). These tests pin that wiring so a future
change can't re-enable deploy-on-read.
"""

from __future__ import annotations

import pytest

from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_runtime import memory_init
from cogniverse_runtime.routers import graph as graph_router


class _FakeMgr:
    def __init__(self):
        self.memory = None
        self.init_kwargs = None

    def initialize(self, **kwargs):
        self.init_kwargs = kwargs
        self.memory = object()


class _FakeConfigManager:
    def get_system_config(self):
        return SystemConfig(
            backend_url="http://vespa",
            backend_port=8080,
            inference_service_urls={"denseon": "http://denseon:8000/v1"},
        )


@pytest.fixture
def _stub_get_config(monkeypatch):
    monkeypatch.setattr(
        memory_init,
        "get_config",
        lambda tenant_id, config_manager: {
            "llm_config": {"primary": {"model": "openai/m", "api_base": "http://lm/v1"}}
        },
    )


def test_lazy_init_read_path_disables_auto_create(_stub_get_config):
    mgr = _FakeMgr()
    ok = memory_init.lazy_init_memory(
        mgr, "acme:prod", _FakeConfigManager(), auto_create_schema=False
    )
    assert ok is True
    assert mgr.init_kwargs["auto_create_schema"] is False


def test_lazy_init_defaults_to_auto_create_for_write_paths(_stub_get_config):
    mgr = _FakeMgr()
    memory_init.lazy_init_memory(mgr, "acme:prod", _FakeConfigManager())
    assert mgr.init_kwargs["auto_create_schema"] is True


@pytest.fixture
def _restore_graph_factory():
    saved = graph_router._graph_manager_factory
    yield
    graph_router._graph_manager_factory = saved


def test_get_graph_manager_read_path_passes_deploy_false(_restore_graph_factory):
    captured = []
    graph_router.set_graph_manager_factory(
        lambda tid, deploy=True: captured.append(deploy) or object()
    )
    graph_router.get_graph_manager("acme:prod", deploy=False)
    assert captured == [False]


def test_get_graph_manager_defaults_to_deploy_true(_restore_graph_factory):
    captured = []
    graph_router.set_graph_manager_factory(
        lambda tid, deploy=True: captured.append(deploy) or object()
    )
    graph_router.get_graph_manager("acme:prod")
    assert captured == [True]
