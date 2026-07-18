"""get_active_adapter_path resolves the adapter URI through the cache dir.

Returning ``adapter.adapter_path`` verbatim only worked for locally trained
adapters; a cloud-backed adapter (s3://, modal://) has to be downloaded under
``SystemConfig.adapter_cache_dir`` first. These pin that the loader routes
through ``resolve_adapter_path`` with the cache dir — a local file:// adapter
resolves to its path, and a cloud URI downloads under the cache dir.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from cogniverse_agents.adapter_loader import get_active_adapter_path


def _adapter(effective_uri: str, adapter_path: str = "/stale/unused/path"):
    adapter = Mock()
    adapter.get_effective_uri.return_value = effective_uri
    adapter.adapter_path = adapter_path
    adapter.name = "routing"
    adapter.version = "2.0.0"
    return adapter


@pytest.mark.unit
def test_cloud_uri_downloads_under_cache_dir(tmp_path):
    """A cloud adapter is downloaded under adapter_cache_dir, NOT read from the
    stale local adapter_path."""
    registry = Mock()
    registry.get_active_adapter.return_value = _adapter(
        "s3://bucket/adapters/routing_v2", adapter_path="/stale/unused/path"
    )
    expected_local = str(tmp_path / "routing_v2")

    with (
        patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=registry),
        patch(
            "cogniverse_finetuning.registry.download_adapter",
            return_value=expected_local,
        ) as mock_download,
    ):
        path = get_active_adapter_path("t1", "routing", adapter_cache_dir=str(tmp_path))

    assert path == expected_local
    # Downloaded to a path under the configured cache dir (not adapter_path).
    called_uri, called_local = mock_download.call_args.args
    assert called_uri == "s3://bucket/adapters/routing_v2"
    assert called_local == expected_local


@pytest.mark.unit
def test_local_file_uri_resolves_to_its_path():
    """A file:// adapter resolves to the on-disk path without downloading."""
    registry = Mock()
    registry.get_active_adapter.return_value = _adapter("file:///models/routing_lora")

    with (
        patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=registry),
        patch("cogniverse_finetuning.registry.download_adapter") as mock_download,
    ):
        path = get_active_adapter_path("t1", "routing", adapter_cache_dir="/cache")

    assert path == "/models/routing_lora"
    mock_download.assert_not_called()


@pytest.mark.unit
def test_empty_cache_dir_is_rejected_for_a_cloud_adapter():
    """resolve_adapter_path requires a non-empty cache dir; the loader swallows
    the resulting error and returns None rather than crashing the agent."""
    registry = Mock()
    registry.get_active_adapter.return_value = _adapter("s3://bucket/adapters/x")

    with patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=registry):
        assert get_active_adapter_path("t1", "routing", adapter_cache_dir="") is None


@pytest.mark.unit
def test_no_active_adapter_returns_none():
    registry = Mock()
    registry.get_active_adapter.return_value = None
    with patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=registry):
        assert (
            get_active_adapter_path("t1", "routing", adapter_cache_dir="/cache") is None
        )


def _real_config_manager():
    from cogniverse_foundation.config.manager import ConfigManager
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    store.initialize()
    return ConfigManager(store=store)


def _config_json_primary_api_base() -> str:
    import json
    from pathlib import Path

    cfg = json.loads(Path("configs/config.json").read_text())
    return cfg["llm_config"]["primary"]["api_base"]


@pytest.mark.unit
def test_adapter_lm_context_binds_adapter_model_when_active():
    """With an active adapter and a real config manager, the context binds a
    REAL dspy.LM built from the tenant's endpoint (config.json llm_config
    .primary) with the adapter's registry name as the model — no config or LM
    boundary is mocked, so a break anywhere in get_config → get_llm_config →
    create_dspy_lm fails here instead of being patched away."""
    import dspy

    from cogniverse_agents.adapter_loader import adapter_lm_context

    adapter = Mock()
    adapter.name = "profile_sft_v2"
    registry = Mock()
    registry.get_active_adapter.return_value = adapter

    ambient = Mock()
    with patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=registry):
        with dspy.context(lm=ambient):
            with adapter_lm_context(
                "acme:acme", "profile_selection", config_manager=_real_config_manager()
            ):
                bound = dspy.settings.lm
            restored = dspy.settings.lm

    assert bound is not ambient, "adapter LM never bound — ran on the base model"
    assert bound.model == "openai/profile_sft_v2"
    assert bound.kwargs["api_base"] == _config_json_primary_api_base()
    assert restored is ambient  # context cleanly restored


@pytest.mark.unit
def test_adapter_lm_context_is_noop_without_active_adapter():
    """No active adapter → the ambient LM is left in place (nullcontext)."""
    import dspy

    from cogniverse_agents.adapter_loader import adapter_lm_context

    registry = Mock()
    registry.get_active_adapter.return_value = None
    ambient = Mock()

    with patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=registry):
        with dspy.context(lm=ambient):
            with adapter_lm_context("acme:acme", "profile_selection"):
                assert dspy.settings.lm is ambient


@pytest.mark.unit
def test_adapter_lm_context_without_manager_resolves_default(monkeypatch):
    """Standalone-agent path: with NO config_manager passed, the helper must
    resolve the process-default manager instead of silently degrading — the
    original wiring let get_config(config_manager=None) raise into the broad
    except and stranded every tenant on the base model."""
    import dspy

    from cogniverse_agents.adapter_loader import adapter_lm_context

    adapter = Mock()
    adapter.name = "profile_sft_v2"
    registry = Mock()
    registry.get_active_adapter.return_value = adapter

    cm = _real_config_manager()
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        lambda: cm,
    )

    with patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=registry):
        with adapter_lm_context("acme:acme", "profile_selection"):
            assert dspy.settings.lm.model == "openai/profile_sft_v2"


@pytest.mark.unit
def test_profile_selection_agent_routes_to_its_adapter(monkeypatch):
    """ProfileSelectionAgent._adapter_lm_context delegates to adapter_lm_context
    with its agent_type, the dispatcher-injected tenant AND the dispatcher-
    injected config_manager — without the manager the helper cannot build the
    tenant endpoint and the adapter is silently dropped."""
    from contextlib import nullcontext

    from cogniverse_agents.profile_selection_agent import ProfileSelectionAgent

    # Bypass the heavy __init__ (LM/registration); exercise only the override.
    agent = object.__new__(ProfileSelectionAgent)
    agent._artifact_tenant_id = "acme:acme"
    agent._config_manager = Mock(name="injected_config_manager")
    agent.deps = Mock()

    captured = {}

    def fake_ctx(tenant_id, agent_type, config_manager=None):
        captured["args"] = (tenant_id, agent_type, config_manager)
        return nullcontext()

    monkeypatch.setattr("cogniverse_agents.adapter_loader.adapter_lm_context", fake_ctx)

    with agent._adapter_lm_context():
        pass
    assert captured["args"] == (
        "acme:acme",
        "profile_selection",
        agent._config_manager,
    )


@pytest.mark.unit
def test_profile_selection_binds_adapter_with_injected_config_manager():
    """PRODUCTION SHAPE, end to end through the override: the dispatcher
    injects _config_manager + _artifact_tenant_id; with an active adapter the
    override must bind the adapter LM through the REAL helper + REAL config
    path. This is the regression the masked tests missed — the original
    override passed no config_manager, get_config raised, and every DSPy call
    silently ran on the base model."""
    import dspy

    from cogniverse_agents.profile_selection_agent import ProfileSelectionAgent

    adapter = Mock()
    adapter.name = "profile_sft_v2"
    registry = Mock()
    registry.get_active_adapter.return_value = adapter

    agent = object.__new__(ProfileSelectionAgent)
    agent._artifact_tenant_id = "acme:acme"
    agent._config_manager = _real_config_manager()  # dispatcher injection
    agent.deps = Mock()

    ambient = Mock()
    with patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=registry):
        with dspy.context(lm=ambient):
            with agent._adapter_lm_context():
                bound = dspy.settings.lm

    assert bound is not ambient, "adapter never bound — ran on the base model"
    assert bound.model == "openai/profile_sft_v2"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generic_dispatch_injects_config_manager(monkeypatch):
    """_execute_generic_agent injects the dispatcher's config_manager onto the
    agent (same pattern as telemetry_manager) so per-tenant adapter/LM
    resolution has a real config source instead of silently degrading."""
    import sys
    import types
    from typing import Optional

    from pydantic import BaseModel

    from cogniverse_runtime.agent_dispatcher import AgentDispatcher
    from cogniverse_runtime.config_loader import ConfigLoader

    built = {}

    class FakeGenericDeps(BaseModel):
        pass

    class FakeGenericInput(BaseModel):
        query: str
        tenant_id: Optional[str] = None

    class FakeGenericAgent:
        def __init__(self, deps):
            self.deps = deps
            built["agent"] = self

        async def process(self, typed_input):
            return types.SimpleNamespace()

    mod = types.ModuleType("fake_generic_mod")
    mod.FakeGenericDeps = FakeGenericDeps
    mod.FakeGenericInput = FakeGenericInput
    mod.FakeGenericAgent = FakeGenericAgent
    monkeypatch.setitem(sys.modules, "fake_generic_mod", mod)
    monkeypatch.setitem(
        ConfigLoader.AGENT_CLASSES, "fake_generic", "fake_generic_mod:FakeGenericAgent"
    )

    cm = _real_config_manager()
    dispatcher = AgentDispatcher(
        agent_registry=Mock(), config_manager=cm, schema_loader=Mock()
    )
    monkeypatch.setattr(dispatcher, "_resolve_gliner_url", lambda: None)
    monkeypatch.setattr(dispatcher, "_init_agent_memory", lambda *a, **k: None)
    monkeypatch.setattr(dispatcher, "_bind_graph_manager", lambda *a, **k: None)
    monkeypatch.setattr(dispatcher, "_apply_artefact_overlay", lambda *a, **k: None)

    result = await dispatcher._execute_generic_agent(
        "fake_generic", "some query", {}, "acme:acme"
    )

    assert result["status"] == "success"
    assert built["agent"]._config_manager is cm


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_dspy_carries_adapter_lm_into_worker_thread():
    """The adapter LM bound by _adapter_lm_context must reach module.forward
    inside call_dspy's asyncio.to_thread worker (dspy overrides are
    contextvar-backed and to_thread copies the context). A propagation
    regression would silently run every adapter-aware agent on the base
    model while staying green everywhere else."""
    import threading

    import dspy

    from cogniverse_core.agents.base import (
        AgentBase,
        AgentDeps,
        AgentInput,
        AgentOutput,
    )

    sentinel = dspy.LM("openai/adapter-sentinel", api_base="http://x/v1", api_key="k")
    seen = {}

    class _Mod:
        def __call__(self, **kwargs):
            seen["lm"] = dspy.settings.lm
            seen["thread"] = threading.get_ident()
            seen["via"] = "__call__"
            return dspy.Prediction(out="ok")

        def forward(self, **kwargs):
            seen["via"] = "forward"
            return dspy.Prediction(out="ok")

    class _Agent(AgentBase[AgentInput, AgentOutput, AgentDeps]):
        async def _process_impl(self, input):  # pragma: no cover — not driven
            raise NotImplementedError

        def _adapter_lm_context(self):
            return dspy.context(lm=sentinel)

    agent = object.__new__(_Agent)

    prediction = await agent.call_dspy(_Mod(), output_field="out")

    assert prediction.out == "ok"
    assert seen["lm"] is sentinel, "adapter LM did not reach the worker thread"
    assert seen["thread"] != threading.get_ident()
    # call_dspy must go through module(...) — calling module.forward(...)
    # directly bypasses DSPy's __call__ instrumentation (callbacks, usage
    # tracking, history) and emits a deprecation warning on every dispatch.
    assert seen["via"] == "__call__", f"invoked via {seen['via']}"


@pytest.mark.unit
def test_base_agent_adapter_lm_context_defaults_to_noop():
    """A non-overriding agent's _adapter_lm_context is a nullcontext, so
    call_dspy is unaffected for every agent that doesn't opt in."""
    from contextlib import nullcontext

    from cogniverse_core.agents.base import AgentBase

    ctx = AgentBase._adapter_lm_context(Mock())
    assert isinstance(ctx, nullcontext)


@pytest.mark.unit
def test_text_analysis_picks_up_adapter_activated_after_construction(monkeypatch):
    """An adapter activated AFTER the agent was built (and cached) must take
    effect within the re-check interval — the LM was previously baked once at
    construction, so a cached standalone agent served the base model until a
    process restart."""

    from cogniverse_agents.text_analysis_agent import TextAnalysisAgent

    agent = object.__new__(TextAnalysisAgent)
    agent.tenant_id = "acme:acme"
    agent.config = Mock()
    active = {"name": None}
    monkeypatch.setattr(
        agent.__class__,
        "_active_adapter_model",
        lambda self: active["name"],
        raising=True,
    )

    rebuilt = []

    def fake_configure(config, adapter_model=TextAnalysisAgent._UNSET_ADAPTER):
        # The refresh threads the resolved adapter in; record what it rebuilt to.
        if adapter_model is TextAnalysisAgent._UNSET_ADAPTER:
            adapter_model = agent._active_adapter_model()
        rebuilt.append(adapter_model)

    agent._configure_dspy_lm = fake_configure
    agent._active_adapter_model_name = None  # what construction resolved
    agent._adapter_checked_at = 0.0  # interval elapsed

    # No change → no rebuild.
    agent._refresh_adapter_lm_if_changed()
    assert rebuilt == []

    # Operator activates an adapter; interval elapsed → LM rebuilt to it.
    active["name"] = "entity_sft_v9"
    agent._adapter_checked_at = 0.0
    agent._refresh_adapter_lm_if_changed()
    assert rebuilt == ["entity_sft_v9"]

    # Within the interval → no extra registry lookups / rebuilds.
    active["name"] = "entity_sft_v10"
    agent._refresh_adapter_lm_if_changed()
    assert rebuilt == ["entity_sft_v9"]


@pytest.mark.unit
def test_adapter_refresh_does_not_double_query_the_registry(monkeypatch):
    """On an adapter change the refresh resolves the active adapter ONCE and
    threads it into the LM rebuild — not a second registry lookup."""
    from cogniverse_agents.text_analysis_agent import TextAnalysisAgent

    agent = object.__new__(TextAnalysisAgent)
    agent.tenant_id = "acme:acme"
    agent.config = Mock()
    agent._active_adapter_model_name = None  # what construction resolved
    agent._adapter_checked_at = 0.0  # interval elapsed

    lookups = []

    def counting_lookup(self):
        lookups.append(1)
        return "entity_sft_v9"

    rebuilt_with = {}

    def fake_configure(config, adapter_model=TextAnalysisAgent._UNSET_ADAPTER):
        rebuilt_with["adapter_model"] = adapter_model

    monkeypatch.setattr(
        TextAnalysisAgent, "_active_adapter_model", counting_lookup, raising=True
    )
    agent._configure_dspy_lm = fake_configure

    agent._refresh_adapter_lm_if_changed()

    assert len(lookups) == 1, f"registry queried {len(lookups)}x, expected 1"
    assert rebuilt_with["adapter_model"] == "entity_sft_v9", (
        "the resolved adapter must be threaded into the rebuild, not re-looked-up"
    )


@pytest.mark.unit
def test_adapter_lm_context_outage_reuses_last_known_adapter():
    """A registry outage must not silently revert a tenant with a known
    active adapter to the base model — the last successful lookup is reused."""
    import dspy

    from cogniverse_agents import adapter_loader
    from cogniverse_agents.adapter_loader import adapter_lm_context

    adapter_loader._LAST_KNOWN_ADAPTERS.clear()
    adapter = Mock()
    adapter.name = "profile_sft_v2"
    ok = Mock()
    ok.get_active_adapter.return_value = adapter
    cm = _real_config_manager()
    with patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=ok):
        with adapter_lm_context(
            "outage:tenant", "profile_selection", config_manager=cm
        ):
            pass

    dead = Mock()
    dead.get_active_adapter.side_effect = ConnectionError("vespa down")
    ambient = Mock()
    with patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=dead):
        with dspy.context(lm=ambient):
            with adapter_lm_context(
                "outage:tenant", "profile_selection", config_manager=cm
            ):
                bound = dspy.settings.lm

    assert bound is not ambient, "outage reverted a known-adapter tenant to base"
    assert bound.model == "openai/profile_sft_v2"


@pytest.mark.unit
def test_adapter_lm_context_outage_without_history_degrades_with_error(caplog):
    """First-ever lookup failing → base model, but at ERROR level so the
    degradation is operationally visible."""
    import logging

    import dspy

    from cogniverse_agents import adapter_loader
    from cogniverse_agents.adapter_loader import adapter_lm_context

    adapter_loader._LAST_KNOWN_ADAPTERS.clear()
    dead = Mock()
    dead.get_active_adapter.side_effect = ConnectionError("vespa down")
    ambient = Mock()
    with patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=dead):
        with dspy.context(lm=ambient):
            with caplog.at_level(
                logging.ERROR, logger="cogniverse_agents.adapter_loader"
            ):
                with adapter_lm_context("cold:tenant", "profile_selection"):
                    assert dspy.settings.lm is ambient
    assert any("vespa down" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
def test_adapter_lm_context_outage_after_known_absence_stays_base_quietly(caplog):
    """A successful "no active adapter" answer is remembered: a later outage
    keeps serving the base model without an error-level alarm."""
    import logging

    import dspy

    from cogniverse_agents import adapter_loader
    from cogniverse_agents.adapter_loader import adapter_lm_context

    adapter_loader._LAST_KNOWN_ADAPTERS.clear()
    none_reg = Mock()
    none_reg.get_active_adapter.return_value = None
    with patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=none_reg):
        with adapter_lm_context("absent:tenant", "profile_selection"):
            pass

    dead = Mock()
    dead.get_active_adapter.side_effect = ConnectionError("vespa down")
    ambient = Mock()
    with patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=dead):
        with dspy.context(lm=ambient):
            with caplog.at_level(
                logging.ERROR, logger="cogniverse_agents.adapter_loader"
            ):
                with adapter_lm_context("absent:tenant", "profile_selection"):
                    assert dspy.settings.lm is ambient
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


@pytest.mark.unit
def test_text_analysis_adapter_outage_reuses_last_known():
    """A registry outage during the periodic adapter re-check must not flip a
    finetuned tenant back to the base model — the last successful answer is
    reused until the registry recovers."""
    from cogniverse_agents.text_analysis_agent import TextAnalysisAgent

    agent = object.__new__(TextAnalysisAgent)
    agent.tenant_id = "acme:acme"

    adapter = Mock()
    adapter.name = "entity_sft_v9"
    ok = Mock()
    ok.get_active_adapter.return_value = adapter
    with patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=ok):
        assert agent._active_adapter_model() == "entity_sft_v9"

    dead = Mock()
    dead.get_active_adapter.side_effect = ConnectionError("vespa down")
    with patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=dead):
        assert agent._active_adapter_model() == "entity_sft_v9"


@pytest.mark.unit
def test_text_analysis_adapter_outage_without_history_returns_none():
    from cogniverse_agents.text_analysis_agent import TextAnalysisAgent

    agent = object.__new__(TextAnalysisAgent)
    agent.tenant_id = "acme:acme"

    dead = Mock()
    dead.get_active_adapter.side_effect = ConnectionError("vespa down")
    with patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=dead):
        assert agent._active_adapter_model() is None
