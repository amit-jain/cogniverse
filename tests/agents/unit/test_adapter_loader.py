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


@pytest.mark.unit
def test_adapter_lm_context_binds_adapter_model_when_active():
    """With an active adapter, the context binds an LM built from the tenant's
    endpoint with the adapter's registry name as the model (vLLM serves the
    LoRA by name)."""
    import dspy

    from cogniverse_agents.adapter_loader import adapter_lm_context
    from cogniverse_foundation.config.unified_config import LLMEndpointConfig

    adapter = Mock()
    adapter.name = "profile_sft_v2"
    registry = Mock()
    registry.get_active_adapter.return_value = adapter

    primary = LLMEndpointConfig(model="openai/base", api_base="http://x/v1")
    system_config = Mock()
    system_config.get_llm_config.return_value = Mock(primary=primary)
    sentinel_lm = Mock()

    with (
        patch("cogniverse_finetuning.registry.AdapterRegistry", return_value=registry),
        patch(
            "cogniverse_foundation.config.utils.get_config", return_value=system_config
        ),
        patch(
            "cogniverse_foundation.config.llm_factory.create_dspy_lm",
            return_value=sentinel_lm,
        ) as mock_factory,
    ):
        with adapter_lm_context("acme:acme", "profile_selection"):
            assert dspy.settings.lm is sentinel_lm

    # The LM was built from an endpoint whose model is the adapter's name.
    endpoint = mock_factory.call_args.args[0]
    assert endpoint.model == "openai/profile_sft_v2"
    assert endpoint.api_base == "http://x/v1"  # tenant endpoint preserved


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
def test_profile_selection_agent_routes_to_its_adapter(monkeypatch):
    """ProfileSelectionAgent._adapter_lm_context delegates to adapter_lm_context
    for its agent_type using the dispatcher-injected tenant — so call_dspy runs
    the profile-selection module against the tenant's fine-tuned adapter."""
    from contextlib import nullcontext

    from cogniverse_agents.profile_selection_agent import ProfileSelectionAgent

    # Bypass the heavy __init__ (LM/registration); exercise only the override.
    agent = object.__new__(ProfileSelectionAgent)
    agent._artifact_tenant_id = "acme:acme"
    agent.deps = Mock()

    captured = {}

    def fake_ctx(tenant_id, agent_type):
        captured["args"] = (tenant_id, agent_type)
        return nullcontext()

    monkeypatch.setattr("cogniverse_agents.adapter_loader.adapter_lm_context", fake_ctx)

    with agent._adapter_lm_context():
        pass
    assert captured["args"] == ("acme:acme", "profile_selection")


@pytest.mark.unit
def test_base_agent_adapter_lm_context_defaults_to_noop():
    """A non-overriding agent's _adapter_lm_context is a nullcontext, so
    call_dspy is unaffected for every agent that doesn't opt in."""
    from contextlib import nullcontext

    from cogniverse_core.agents.base import AgentBase

    ctx = AgentBase._adapter_lm_context(Mock())
    assert isinstance(ctx, nullcontext)
