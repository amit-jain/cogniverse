"""deploy_schemas must surface a data-loss refusal, not swallow it to False.

When the live cluster holds a schema that is unregistered and cannot be
reconstructed, redeploying the application package without it would remove the
document type and destroy its documents. deploy_schemas raises
BackendDeploymentError for that case; the broad "failed to deploy -> return
False" handler must not catch it, or the caller cannot tell a safety refusal
(never retry / never force) from a transient deploy failure.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cogniverse_core.registries.exceptions import BackendDeploymentError
from cogniverse_vespa import json_schema_parser
from cogniverse_vespa.backend import VespaBackend
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.fixture
def backend_with_orphan(monkeypatch):
    # Parse any schema JSON into a lightweight object carrying just its name.
    monkeypatch.setattr(
        json_schema_parser.JsonSchemaParser,
        "parse_schema",
        lambda self, d: SimpleNamespace(name=d.get("name", "new_schema")),
    )

    backend = VespaBackend.__new__(VespaBackend)
    registry = MagicMock()
    registry._get_all_schemas.return_value = []  # nothing to merge/reconstruct
    registry.reconcile_deployment_intents.return_value = []
    registry.reserved_schemas.return_value = {}
    backend.schema_registry = registry

    # The real manager resolves live schemas against the registry and the
    # intent journal; only its config-server probe is stubbed. An orphan lives
    # in Vespa that neither source can rebuild.
    manager = VespaSchemaManager(
        backend_endpoint="http://localhost",
        backend_port=19071,
        schema_registry=registry,
    )
    manager.list_deployed_document_types = MagicMock(
        return_value=["knowledge_graph_globex_globex"]
    )
    backend.schema_manager = manager

    backend._deploy_package = MagicMock()
    backend._wait_for_schema_convergence = MagicMock()
    return backend


def test_unreconstructable_orphan_raises_and_does_not_deploy(backend_with_orphan):
    schema_defs = [
        {"name": "video_acme_acme", "definition": {"name": "video_acme_acme"}}
    ]

    with pytest.raises(BackendDeploymentError, match="destroy their documents"):
        backend_with_orphan.deploy_schemas(schema_defs)

    # The destructive redeploy never ran.
    backend_with_orphan._deploy_package.assert_not_called()


def test_config_server_enumeration_failure_aborts_before_deploy(
    backend_with_orphan,
    monkeypatch,
):
    manager = backend_with_orphan.schema_manager

    def enumerate_schemas(*, raise_on_failure=False):
        if raise_on_failure:
            raise ConnectionError("config server unavailable")
        return []

    manager.list_deployed_document_types.side_effect = enumerate_schemas
    backend_with_orphan._config_manager_instance = MagicMock()
    backend_with_orphan._config_manager_instance.get_system_config.return_value = (
        SimpleNamespace(application_name="cogniverse")
    )
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    schema_defs = [
        {"name": "video_acme_acme", "definition": {"name": "video_acme_acme"}}
    ]

    with pytest.raises(BackendDeploymentError, match="enumerate"):
        backend_with_orphan.deploy_schemas(schema_defs)

    manager.list_deployed_document_types.assert_called_once_with(raise_on_failure=True)
    backend_with_orphan._deploy_package.assert_not_called()


def test_registry_enumeration_failure_aborts_before_deploy(
    backend_with_orphan,
):
    backend_with_orphan.schema_registry._get_all_schemas.side_effect = ConnectionError(
        "registry unavailable"
    )
    schema_defs = [
        {"name": "video_acme_acme", "definition": {"name": "video_acme_acme"}}
    ]

    with pytest.raises(BackendDeploymentError, match="registry"):
        backend_with_orphan.deploy_schemas(schema_defs)

    backend_with_orphan.schema_manager.list_deployed_document_types.assert_not_called()
    backend_with_orphan._deploy_package.assert_not_called()


def test_intent_recovery_writes_nothing_after_the_lease_is_taken_over(
    backend_with_orphan, monkeypatch
):
    """A deployer stuck in its enumeration past the heartbeat cap is taken
    over; resuming, it must not recover intents into the successor's registry."""
    from cogniverse_core.registries import schema_deploy_lease
    from cogniverse_core.registries.schema_deploy_lease import SchemaDeployLease
    from tests.utils.memory_store import InMemoryConfigStore

    monkeypatch.setattr(schema_deploy_lease, "MAX_TOTAL_HOLD_SECONDS", 0.6)
    store = InMemoryConfigStore()
    registry = backend_with_orphan.schema_registry
    registry.deployment_lease = lambda **kwargs: SchemaDeployLease(
        store, lease_seconds=0.3, **kwargs
    )
    successors = []

    def enumerate_until_taken_over(*, raise_on_failure=False):
        successor = SchemaDeployLease(store, wait_seconds=5)
        assert successor.acquire() is successor
        successors.append(successor)
        return ["knowledge_graph_globex_globex"]

    backend_with_orphan.schema_manager.list_deployed_document_types.side_effect = (
        enumerate_until_taken_over
    )
    recovered = []

    def reconcile(live_names, fence=None):
        if fence is not None:
            fence()
        recovered.append(set(live_names))
        return []

    registry.reconcile_deployment_intents = reconcile
    schema_defs = [
        {"name": "video_acme_acme", "definition": {"name": "video_acme_acme"}}
    ]

    from cogniverse_core.registries.schema_deploy_lease import DeploymentLeaseLost

    with pytest.raises(BackendDeploymentError) as failure:
        backend_with_orphan.deploy_schemas(schema_defs)

    assert str(failure.value) == (
        "Deployment lease was taken over during intent recovery; nothing was "
        "activated or registered. Retry the deploy: Vespa deployment lease "
        "expired or was replaced"
    )
    assert isinstance(failure.value.__cause__, DeploymentLeaseLost)

    assert len(successors) == 1
    assert recovered == []
    backend_with_orphan._deploy_package.assert_not_called()
    successors[0].release()


def test_a_lease_lost_before_activation_is_reported_as_retryable(monkeypatch):
    """A deployer taken over after building its package must not post it, and
    the lost lease is reported as a retryable takeover, not a Vespa refusal."""
    import json
    from pathlib import Path

    import requests

    from cogniverse_core.registries import schema_deploy_lease
    from cogniverse_core.registries.schema_deploy_lease import (
        DeploymentLeaseLost,
        SchemaDeployLease,
    )
    from tests.utils.memory_store import InMemoryConfigStore

    monkeypatch.setattr(schema_deploy_lease, "MAX_TOTAL_HOLD_SECONDS", 0.6)
    store = InMemoryConfigStore()
    registry = MagicMock()
    registry._get_all_schemas.return_value = []
    registry.reconcile_deployment_intents.return_value = []
    registry.reserved_schemas.return_value = {}
    registry.deployment_lease = lambda **kwargs: SchemaDeployLease(
        store, lease_seconds=0.3, **kwargs
    )
    backend = VespaBackend.__new__(VespaBackend)
    backend.schema_registry = registry
    manager = VespaSchemaManager(
        backend_endpoint="http://localhost",
        backend_port=19071,
        schema_registry=registry,
    )
    manager.list_deployed_document_types = MagicMock(return_value=[])
    backend.schema_manager = manager
    backend._url = "http://localhost"
    backend._config_port = 19071
    backend._wait_for_schema_convergence = MagicMock()
    successors = []

    def taken_over_while_building():
        successor = SchemaDeployLease(store, wait_seconds=5)
        assert successor.acquire() is successor
        successors.append(successor)
        return SimpleNamespace(application_name="cogniverse")

    backend._config_manager_instance = MagicMock()
    backend._config_manager_instance.get_system_config.side_effect = (
        taken_over_while_building
    )
    posts = []
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: posts.append(args))
    definition = json.loads(Path("configs/schemas/provenance_schema.json").read_text())
    definition["name"] = "provenance_acme_acme"

    with pytest.raises(BackendDeploymentError) as failure:
        backend.deploy_schemas(
            [{"name": "provenance_acme_acme", "definition": definition}]
        )

    assert str(failure.value) == (
        "Deployment lease was taken over before activation; nothing was "
        "activated or registered. Retry the deploy: Vespa deployment lease "
        "expired or was replaced"
    )
    assert isinstance(failure.value.__cause__, DeploymentLeaseLost)
    assert posts == []
    assert len(successors) == 1
    backend._wait_for_schema_convergence.assert_not_called()
    successors[0].release()
