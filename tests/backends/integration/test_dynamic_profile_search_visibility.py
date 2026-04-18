"""Integration test for dynamic profile visibility at search time.

Closes the coverage gap in `test_tenant_schema_lifecycle.py`: existing
backend integration tests deploy schemas via
`backend.schema_registry.deploy_schema(...)` and then assert against the
SchemaRegistry's own tracking — they never execute a search against the
just-deployed schema. That meant the "profile not found" error path in
VespaSearchBackend was never exercised after a dynamic deploy, letting
the whole visibility gap go undetected until the e2e suite ran long
enough to trigger retry-storm event-loop wedging.

This test runs the full chain: register a profile via
`BackendRegistry.add_profile_to_backends` (the fanout the
`profile_change_listener` fires), then invoke `backend.search(...)` and
assert the profile-resolution phase passes. We don't need documents in
the index — passing profile resolution alone proves the fix.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def temp_config_manager(vespa_instance):
    """ConfigManager wired to the module-scoped Vespa Docker instance."""
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_vespa.config.config_store import VespaConfigStore

    store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=vespa_instance["http_port"],
    )
    cm = ConfigManager(store=store)

    # Wire the same listener main.py wires at runtime startup.
    def listener(event, name, cfg):
        if event == "added" and cfg is not None:
            BackendRegistry.add_profile_to_backends(name, cfg)
        elif event == "removed":
            BackendRegistry.remove_profile_from_backends(name)

    cm.set_profile_change_listener(listener)
    return cm


@pytest.fixture(scope="module")
def schema_loader():
    return FilesystemSchemaLoader(Path("configs/schemas"))


@pytest.fixture
def clean_registry(vespa_instance, temp_config_manager, schema_loader):
    """Reset the shared backend cache before and after each test."""
    BackendRegistry._backend_instances.clear()
    yield
    BackendRegistry._backend_instances.clear()


@pytest.mark.integration
def test_search_against_dynamically_registered_profile_resolves(
    vespa_instance, temp_config_manager, schema_loader, clean_registry
):
    """Regression guard for the agent_memories retry-storm bug.

    Before the fix:
      `search(profile=X)` raised `ValueError: Requested profile 'X' not found`
      because `VespaSearchBackend.profiles` was a snapshot from pod boot.

    After the fix:
      `BackendRegistry.add_profile_to_backends(...)` updates the cached
      backend's in-memory dict, so the same search now passes the
      profile-resolution phase.

    We don't assert on actual Vespa query results — just that the error
    path the bug triggered does not fire. Any later failure (no schema,
    no strategies, etc.) is acceptable for this test: the visibility
    gap is what we're proving is closed.
    """
    registry = BackendRegistry.get_instance()

    # Construct the search backend via the normal cache path with NO
    # profiles — mirrors a cold pod that has no agent_memories profile.
    config = {
        "backend": {
            "url": "http://localhost",
            "config_port": vespa_instance["config_port"],
            "port": vespa_instance["http_port"],
        }
    }
    backend = registry.get_search_backend(
        name="vespa",
        config=config,
        config_manager=temp_config_manager,
        schema_loader=schema_loader,
    )
    assert "agent_memories" not in backend.profiles, (
        "backend must start without the profile so we're proving "
        "dynamic registration worked"
    )

    # Fanout a brand-new profile into the cached backend.
    registered = BackendRegistry.add_profile_to_backends(
        "agent_memories",
        {
            "type": "memory",
            "schema_name": "agent_memories",
            "embedding_model": "nomic-embed-text",
            "embedding_type": "dense",
            "schema_config": {"embedding_dims": 768},
        },
    )
    assert registered >= 1, "fanout must have hit at least the search backend"
    assert "agent_memories" in backend.profiles

    # Now the profile-resolution phase must pass. Whatever happens after
    # that (no index yet, etc.) is fine for this test.
    try:
        backend.search(
            query_dict={
                "query": "hello",
                "type": "memory",
                "profile": "agent_memories",
                "tenant_id": "dyn_profile_test",
                "top_k": 1,
            }
        )
    except ValueError as exc:
        if "not found" in str(exc) and "agent_memories" in str(exc):
            pytest.fail(
                "Profile-resolution still raises 'profile not found' — "
                "BackendRegistry fanout did not update the cached backend."
            )
        # Other ValueErrors (e.g., ranking strategies not loaded) are not
        # what this test guards against.
    except Exception:
        # Any other exception is downstream of profile resolution.
        pass


@pytest.mark.integration
def test_add_backend_profile_end_to_end_reaches_search_backend(
    vespa_instance, temp_config_manager, schema_loader, clean_registry
):
    """Full chain: ConfigManager.add_backend_profile → listener → fanout
    → VespaSearchBackend.profiles updated. No mocks."""
    from cogniverse_foundation.config.unified_config import BackendProfileConfig

    registry = BackendRegistry.get_instance()
    config = {
        "backend": {
            "url": "http://localhost",
            "config_port": vespa_instance["config_port"],
            "port": vespa_instance["http_port"],
        }
    }
    backend = registry.get_search_backend(
        name="vespa",
        config=config,
        config_manager=temp_config_manager,
        schema_loader=schema_loader,
    )
    assert "e2e_probe" not in backend.profiles

    profile = BackendProfileConfig(
        profile_name="e2e_probe",
        type="memory",
        schema_name="e2e_probe",
        embedding_model="nomic-embed-text",
        embedding_type="dense",
        schema_config={"embedding_dims": 768},
    )
    temp_config_manager.add_backend_profile(
        profile, tenant_id="default", service="backend"
    )

    assert "e2e_probe" in backend.profiles, (
        "profile added via ConfigManager must have reached the cached "
        "search backend through the listener chain"
    )
    assert backend.profiles["e2e_probe"]["schema_name"] == "e2e_probe"

    # And removal propagates too.
    assert temp_config_manager.delete_backend_profile(
        "e2e_probe", tenant_id="default", service="backend"
    ) is True
    assert "e2e_probe" not in backend.profiles
