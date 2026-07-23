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
    backend.schema_registry = registry

    manager = MagicMock()
    # An orphan lives in Vespa that the registry does not know and cannot
    # reconstruct — a peer tenant's schema mid-registration, say.
    manager.list_deployed_document_types.return_value = [
        "knowledge_graph_globex_globex"
    ]
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
