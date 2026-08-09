"""delete_schema guards.

Cross-tenant suffix guard: the target name is built from the canonicalized
tenant (acme -> acme:acme -> suffix _acme_acme), but the guard compared
against the raw tenant_id's suffix (_acme), which is only a substring — so a
wrong-tenant target ending in _acme slipped past the defensive check.

Live-vs-registry guard: the removal redeploy's survivors come from the
registry, so any DEPLOYED-but-unregistered schema silently vanished from the
application package — deleting one profile destroyed sibling data (e.g. a
tenant's knowledge_graph schema deployed without registration). delete_schema
must refuse when the redeploy would drop live schemas the registry does not
know.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

METADATA_SCHEMAS = (
    "adapter_registry",
    "config_metadata",
    "organization_metadata",
    "tenant_metadata",
)


def _bare_manager() -> VespaSchemaManager:
    mgr = object.__new__(VespaSchemaManager)
    mgr._schema_registry = object()  # truthy — past the registry guard
    mgr._PROTECTED_SCHEMAS = frozenset()
    return mgr


def test_cross_tenant_target_rejected_by_canonical_suffix():
    mgr = _bare_manager()
    # A target for a DIFFERENT tenant that still ends in the raw "_acme".
    mgr.get_tenant_schema_name = lambda tenant_id, base: "video_other_acme"

    with pytest.raises(ValueError, match="does not carry the expected"):
        mgr.delete_schema("acme", "video")


class _RecordingRegistry:
    def __init__(self):
        self.unregistered: list = []

    def unregister_schema(self, tenant_id: str, base_schema_name: str) -> None:
        self.unregistered.append((tenant_id, base_schema_name))


def _guard_manager(survivor_names: list, deployed_names: list) -> VespaSchemaManager:
    from vespa.package import Document, Schema

    mgr = object.__new__(VespaSchemaManager)
    mgr._schema_registry = _RecordingRegistry()
    mgr._logger = logging.getLogger("test_delete_schema_guard")
    mgr._get_existing_tenant_schemas = lambda: [
        Schema(name=n, document=Document()) for n in survivor_names
    ]
    mgr.list_deployed_document_types = lambda **_: list(deployed_names)
    mgr.deployed_packages = []
    mgr._deploy_package = lambda pkg, allow_schema_removal=False: (
        mgr.deployed_packages.append(pkg)
    )
    return mgr


class TestDeleteSchemaLiveGuard:
    def test_unregistered_live_schema_blocks_delete(self):
        mgr = _guard_manager(
            survivor_names=["video_other_acme_acme"],
            deployed_names=[
                *METADATA_SCHEMAS,
                "video_colpali_acme_acme",
                "video_other_acme_acme",
                "knowledge_graph_acme_acme",  # deployed, not in registry
            ],
        )

        with pytest.raises(ValueError) as exc:
            mgr.delete_schema("acme", "video_colpali")

        assert "knowledge_graph_acme_acme" in str(exc.value)
        assert "registry" in str(exc.value)
        assert mgr.deployed_packages == []
        assert mgr._schema_registry.unregistered == []

    def test_live_matching_registry_plus_target_proceeds(self):
        mgr = _guard_manager(
            survivor_names=["video_other_acme_acme"],
            deployed_names=[
                *METADATA_SCHEMAS,
                "video_colpali_acme_acme",
                "video_other_acme_acme",
            ],
        )

        removed = mgr.delete_schema("acme", "video_colpali")

        assert removed == "video_colpali_acme_acme"
        assert len(mgr.deployed_packages) == 1
        deployed = {s.name for s in mgr.deployed_packages[0].schemas}
        assert deployed == {*METADATA_SCHEMAS, "video_other_acme_acme"}
        assert mgr._schema_registry.unregistered == [("acme", "video_colpali")]

    def test_live_listing_failure_propagates_without_deploy(self):
        mgr = _guard_manager(
            survivor_names=["video_other_acme_acme"],
            deployed_names=[],
        )

        def _raise(**_) -> list:
            raise ConnectionError("config server down")

        mgr.list_deployed_document_types = _raise

        with pytest.raises(RuntimeError, match="Cannot enumerate") as exc:
            mgr.delete_schema("acme", "video_colpali")

        assert isinstance(exc.value.__cause__, ConnectionError)
        assert mgr.deployed_packages == []

    def test_registry_tombstone_failure_raises_after_vespa_removal(self):
        mgr = _guard_manager(
            survivor_names=[],
            deployed_names=[*METADATA_SCHEMAS, "video_colpali_acme_acme"],
        )

        def fail_tombstone(*_args):
            raise RuntimeError("registry write unavailable")

        mgr._schema_registry.unregister_schema = fail_tombstone

        with pytest.raises(RuntimeError, match="registry tombstone"):
            mgr.delete_schema("acme", "video_colpali")

        assert len(mgr.deployed_packages) == 1
        deployed = {s.name for s in mgr.deployed_packages[0].schemas}
        assert deployed == set(METADATA_SCHEMAS)


class TestBackendDeleteSchemaWiring:
    """VespaBackend.delete_schema must delete THE NAMED schema through the
    guarded singular manager method and fail loud. It previously ignored
    schema_name, deleted every tenant schema via the plural path, and
    swallowed failures into an empty list — the admin profile-delete route
    then replied 200 with schema_deleted false while sibling schemas were
    gone."""

    def _backend(self):
        from unittest.mock import MagicMock

        from cogniverse_vespa.backend import VespaBackend

        backend = object.__new__(VespaBackend)
        backend.schema_manager = MagicMock()
        backend._tenant_id = None
        return backend

    def test_deletes_exactly_the_named_schema(self):
        backend = self._backend()
        backend.schema_manager.delete_schema.return_value = "video_x_acme_acme"

        out = backend.delete_schema(schema_name="video_x", tenant_id="acme:acme")

        backend.schema_manager.delete_schema.assert_called_once_with(
            "acme:acme", "video_x"
        )
        assert out == ["video_x_acme_acme"]

    def test_guard_refusal_propagates(self):
        backend = self._backend()
        backend.schema_manager.delete_schema.side_effect = ValueError(
            "Refusing to delete 'video_x_acme_acme': redeploying without it "
            "would also drop ['knowledge_graph_acme_acme']"
        )

        with pytest.raises(ValueError, match="Refusing to delete"):
            backend.delete_schema(schema_name="video_x", tenant_id="acme:acme")


class _BulkRegistry:
    """Registry stub for delete_tenant_schemas_bulk: exposes the registered
    full-name set (``_get_all_schemas``) and per-tenant bases."""

    def __init__(self, registered_full_names: list, tenant_bases: dict):
        self._registered = registered_full_names
        self._tenant_bases = tenant_bases
        self.unregistered: list = []

    def _get_all_schemas(self):
        return [
            SimpleNamespace(full_schema_name=n, schema_definition="{}")
            for n in self._registered
        ]

    def get_tenant_schemas(self, tid: str):
        return [
            SimpleNamespace(base_schema_name=b) for b in self._tenant_bases.get(tid, [])
        ]

    def unregister_schema(self, tid: str, base: str) -> None:
        self.unregistered.append((tid, base))


class TestBulkDeleteSuffixAndRefuseGuards:
    """delete_tenant_schemas_bulk must not sweep a registered peer in via a
    proper-suffix match, and must refuse an unconfirmable survivor — the two
    reconcile-orphans data-loss paths."""

    def _capture_manager(self, registered, deployed, tenant_bases):
        mgr = object.__new__(VespaSchemaManager)
        mgr._PROTECTED_SCHEMAS = frozenset(METADATA_SCHEMAS)
        mgr._schema_registry = _BulkRegistry(registered, tenant_bases)
        mgr._logger = logging.getLogger("test_bulk_guard")
        mgr.list_deployed_document_types = lambda **_: list(deployed)
        mgr.get_tenant_schema_name = lambda tid, base: f"{base}_{tid.replace(':', '_')}"
        captured: dict = {}

        def _capture(targets):
            captured["targets"] = set(targets)
            return sorted(set(targets) & set(deployed))

        mgr._redeploy_dropping = _capture
        return mgr, captured

    def test_canonical_orphan_is_deleted_without_matching_registered_peer(self):
        registered = ["knowledge_graph_other_acme_acme", *METADATA_SCHEMAS]
        deployed = [
            "knowledge_graph_acme_acme",
            "knowledge_graph_other_acme_acme",
            "knowledge_graph_acme",
            *METADATA_SCHEMAS,
        ]
        mgr, captured = self._capture_manager(
            registered, deployed, tenant_bases={"acme": []}
        )

        mgr.delete_tenant_schemas_bulk(["acme"])

        assert captured["targets"] == {"knowledge_graph_acme_acme"}
        assert "knowledge_graph_other_acme_acme" not in captured["targets"]
        assert "knowledge_graph_acme" not in captured["targets"]

    def test_bulk_refuses_on_unresolved_survivor(self):
        # A deployed schema with no registry record and outside the named
        # tenants is an unconfirmable survivor — the real redeploy must refuse
        # rather than drop it (it could be a peer tenant's live data).
        from cogniverse_core.registries.exceptions import BackendDeploymentError

        registered = [*METADATA_SCHEMAS]  # neither orphan nor survivor registered
        deployed = [
            "knowledge_graph_acme_acme",
            "video_other_globex_globex",  # unregistered survivor, NOT named
            *METADATA_SCHEMAS,
        ]
        mgr = object.__new__(VespaSchemaManager)
        mgr._PROTECTED_SCHEMAS = frozenset(METADATA_SCHEMAS)
        mgr._schema_registry = _BulkRegistry(registered, tenant_bases={"acme": []})
        mgr._logger = logging.getLogger("test_bulk_refuse")
        mgr.list_deployed_document_types = lambda **_: list(deployed)
        mgr.get_tenant_schema_name = lambda tid, base: f"{base}_{tid.replace(':', '_')}"
        deployed_packages: list = []
        mgr._deploy_package = lambda pkg, allow_schema_removal=False: (
            deployed_packages.append(pkg)
        )

        with pytest.raises(BackendDeploymentError, match="no registry record"):
            mgr.delete_tenant_schemas_bulk(["acme"])

        # Refused before any redeploy — nothing was deployed.
        assert deployed_packages == []


def test_single_tenant_delete_uses_canonical_suffix_only():
    registry = _BulkRegistry([*METADATA_SCHEMAS], tenant_bases={"acme": []})
    mgr = object.__new__(VespaSchemaManager)
    mgr._schema_registry = registry
    mgr._logger = logging.getLogger("test_single_tenant_suffix")
    mgr.list_deployed_document_types = lambda **_: [
        *METADATA_SCHEMAS,
        "knowledge_graph_acme_acme",
        "knowledge_graph_acme",
    ]
    captured = {}

    def capture_targets(targets):
        captured["targets"] = set(targets)
        return []

    mgr._redeploy_dropping = capture_targets

    mgr.delete_tenant_schemas("acme")

    assert captured["targets"] == {"knowledge_graph_acme_acme"}


def test_tenant_tombstone_retries_even_when_vespa_schema_is_already_gone():
    registry = _BulkRegistry(
        ["video_acme_acme", *METADATA_SCHEMAS],
        tenant_bases={"acme": ["video"]},
    )
    mgr = object.__new__(VespaSchemaManager)
    mgr._schema_registry = registry
    mgr._logger = logging.getLogger("test_tombstone_retry")
    mgr.list_deployed_document_types = lambda **_: [*METADATA_SCHEMAS]
    mgr._redeploy_dropping = lambda _targets: []

    assert mgr.delete_tenant_schemas("acme") == []
    assert registry.unregistered == [("acme", "video")]


class TestSchemaEnumerationRefusesPartial:
    """A registry entry that cannot be rebuilt into the deployment package is
    a schema the deploy would DROP (with its documents) — enumeration must
    abort, never skip the entry and hand back a partial package."""

    def _manager_with_rows(self, rows) -> VespaSchemaManager:
        from types import SimpleNamespace

        mgr = object.__new__(VespaSchemaManager)
        mgr._logger = logging.getLogger("test_schema_enumeration")
        mgr._schema_registry = SimpleNamespace(_get_all_schemas=lambda: rows)
        return mgr

    def test_empty_definition_aborts_enumeration(self):
        from types import SimpleNamespace

        rows = [
            SimpleNamespace(
                full_schema_name="video_x_acme_acme", schema_definition="   "
            )
        ]
        with pytest.raises(RuntimeError, match="drop the schema"):
            self._manager_with_rows(rows)._get_existing_tenant_schemas()

    def test_invalid_json_definition_aborts_enumeration(self):
        from types import SimpleNamespace

        rows = [
            SimpleNamespace(
                full_schema_name="video_x_acme_acme", schema_definition="{not json"
            )
        ]
        with pytest.raises(RuntimeError, match="Cannot rebuild"):
            self._manager_with_rows(rows)._get_existing_tenant_schemas()


def test_upload_metadata_schemas_defaults_to_removal_disabled():
    """Safe-by-default: only registry-aware callers that need deleted-tenant
    cleanup opt into allow_schema_removal=True explicitly."""
    import inspect

    signature = inspect.signature(VespaSchemaManager.upload_metadata_schemas)
    assert signature.parameters["allow_schema_removal"].default is False
