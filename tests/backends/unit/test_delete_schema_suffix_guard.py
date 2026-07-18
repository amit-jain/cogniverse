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
    mgr.list_deployed_document_types = lambda: list(deployed_names)
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

        def _raise() -> list:
            raise ConnectionError("config server down")

        mgr.list_deployed_document_types = _raise

        with pytest.raises(RuntimeError, match="Cannot enumerate") as exc:
            mgr.delete_schema("acme", "video_colpali")

        assert isinstance(exc.value.__cause__, ConnectionError)
        assert mgr.deployed_packages == []


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
