"""A config-server enumeration failure must be fatal for redeploy paths.

delete_schema (and the other tenant-schema redeploy paths) replace the whole
application package with metadata + a computed survivor set. If
list_deployed_document_types swallows a config-server outage to an empty list,
the survivor computation reads "no other schemas deployed" and the redeploy
drops every peer-tenant schema and destroys its documents. The enumeration
failure must raise so the delete refuses instead of guessing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _manager(schema_registry=None) -> VespaSchemaManager:
    return VespaSchemaManager(
        backend_endpoint="http://localhost",
        backend_port=19071,
        schema_registry=schema_registry,
    )


def test_enumeration_failure_raises_when_requested(monkeypatch):
    manager = _manager()

    def _boom(*a, **k):
        raise requests.exceptions.ConnectionError("config server down")

    monkeypatch.setattr(requests, "get", _boom)

    # Default: swallow to [] (benign callers fall back to the registry).
    assert manager.list_deployed_document_types() == []
    # Redeploy callers demand the truth — a failed probe must raise, not read
    # as an authoritative empty deployment.
    with pytest.raises(requests.exceptions.ConnectionError):
        manager.list_deployed_document_types(raise_on_failure=True)


def test_delete_schema_refuses_and_does_not_deploy_on_outage(monkeypatch):
    manager = _manager(schema_registry=MagicMock())

    # Reach the enumeration guard: a target carrying the canonical suffix and
    # an empty registry survivor set.
    monkeypatch.setattr(
        manager, "get_tenant_schema_name", lambda t, b: f"{b}_acme_acme"
    )
    monkeypatch.setattr(manager, "_get_existing_tenant_schemas", lambda: [])

    deploy_spy = MagicMock()
    monkeypatch.setattr(manager, "_deploy_package", deploy_spy)

    # The config-server probe fails mid-delete.
    def _boom(*a, **k):
        raise requests.exceptions.ConnectionError("config server down")

    monkeypatch.setattr(requests, "get", _boom)

    with pytest.raises(RuntimeError, match="Cannot enumerate Vespa-deployed schemas"):
        manager.delete_schema("acme", "knowledge_graph")

    # The redeploy that would have dropped peer-tenant schemas never ran.
    deploy_spy.assert_not_called()


def test_upload_metadata_schemas_defaults_to_removal_disabled():
    """The metadata deploy must default to refusing a schema-dropping deploy:
    a merge that misses a peer tenant's not-yet-registered schema fails loudly
    rather than destroying its documents. Only a deliberate cleanup passes True.
    """
    import inspect

    sig = inspect.signature(VespaSchemaManager.upload_metadata_schemas)
    assert sig.parameters["allow_schema_removal"].default is False
