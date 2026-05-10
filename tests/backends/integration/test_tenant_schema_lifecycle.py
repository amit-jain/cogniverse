"""
Integration tests for SchemaRegistry with real Vespa Docker instance.

Tests actual schema deployment, deletion, and tenant isolation via SchemaRegistry.
Requires Docker to be running.
"""

import logging
from pathlib import Path

import pytest

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def temp_config_manager(vespa_instance, tmp_path_factory):
    """
    Provide a temporary ConfigManager with real VespaConfigStore.

    Uses VespaConfigStore connected to the test Vespa instance.
    The config_metadata schema is automatically deployed as part of
    VespaSchemaManager.upload_metadata_schemas() during backend initialization.
    """
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_vespa.config.config_store import VespaConfigStore

    http_port = vespa_instance["http_port"]
    logger.info(f"Creating VespaConfigStore with http_port={http_port}")

    store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=http_port,
    )
    logger.info(f"VespaConfigStore created, vespa_app URL: {store.vespa_app.url}")

    return ConfigManager(store=store)


@pytest.fixture(scope="module")
def schema_loader():
    """Provide FilesystemSchemaLoader for tests (module-scoped for reuse)."""
    return FilesystemSchemaLoader(Path("configs/schemas"))


@pytest.fixture(scope="module")
def get_backend(vespa_instance, temp_config_manager, schema_loader):
    """
    Factory function to get backend for a tenant.

    Returns a function that creates backend instances for different tenants.
    Module-scoped to reuse backend instances across tests.
    """

    def _get_backend(tenant_id: str):
        registry = BackendRegistry.get_instance()
        config = {
            "backend": {
                "url": "http://localhost",
                "config_port": vespa_instance["config_port"],
                "port": vespa_instance["http_port"],
            }
        }
        return registry.get_search_backend(
            name="vespa",
            config=config,
            config_manager=temp_config_manager,
            schema_loader=schema_loader,
        )

    return _get_backend


_KNOWN_BASE_SCHEMAS = (
    "video_colpali_smol500_mv_frame",
    "video_videoprism_base_mv_chunk_30s",
    "agent_memories",
    "knowledge_graph",
    "wiki",
)


@pytest.fixture
def wipe_non_protected_schemas(get_backend):
    """Per-test fixture: wipes non-protected, tenant-scoped schemas before yield.

    The conftest's ``vespa_instance`` hashes the conftest's ``__name__`` for
    its port, so every module sharing this conftest lands on the same
    container name. Schemas leaked by an earlier module (e.g.
    ``agent_memories_dyn_roundtrip_*`` from
    ``test_dynamic_profile_search_visibility.py``) survive into later
    modules. The 4 tests below then hit the deploy safety check because
    the registry singleton is fresh but Vespa still holds those schemas.
    Tests opt in by requesting this fixture.
    """
    backend = get_backend("__bootstrap_cleanup__")
    sm = backend.schema_manager
    try:
        deployed = sm.list_deployed_document_types()
    except Exception:
        return

    for full_name in deployed:
        if full_name in sm._PROTECTED_SCHEMAS:
            continue
        for base in _KNOWN_BASE_SCHEMAS:
            if full_name.startswith(base + "_"):
                tenant_id = full_name[len(base) + 1 :]
                try:
                    sm.delete_schema(tenant_id, base)
                except Exception as e:
                    logger.warning(f"per-test cleanup failed for {full_name}: {e}")
                break


@pytest.mark.integration
@pytest.mark.ci_fast
class TestSchemaRegistryDeployment:
    """Test schema deployment via SchemaRegistry"""

    def test_deploy_single_schema(self, get_backend):
        """Test deploying a single schema for a tenant"""
        backend = get_backend("acme")

        # Deploy schema
        backend.schema_registry.deploy_schema("acme", "video_colpali_smol500_mv_frame")

        # Verify registered
        schemas = backend.schema_registry.get_tenant_schemas("acme")
        assert len(schemas) == 1
        assert schemas[0].base_schema_name == "video_colpali_smol500_mv_frame"
        assert schemas[0].full_schema_name == "video_colpali_smol500_mv_frame_acme"

    def test_deploy_multiple_schemas_same_tenant(self, get_backend):
        """Test deploying multiple schemas for the same tenant"""
        backend = get_backend("startup")

        # Deploy two schemas
        backend.schema_registry.deploy_schema(
            "startup", "video_colpali_smol500_mv_frame"
        )
        backend.schema_registry.deploy_schema(
            "startup", "video_videoprism_base_mv_chunk_30s"
        )

        # Verify both registered
        schemas = backend.schema_registry.get_tenant_schemas("startup")
        assert len(schemas) == 2
        base_names = {s.base_schema_name for s in schemas}
        assert "video_colpali_smol500_mv_frame" in base_names
        assert "video_videoprism_base_mv_chunk_30s" in base_names

    def test_deploy_same_schema_multiple_tenants(self, get_backend):
        """Test deploying the same base schema for different tenants"""
        # Use same backend but deploy for different tenants
        # (SchemaRegistry is per-backend instance, so both deployments share registry)
        backend = get_backend("multi_tenant_test")

        # Deploy for both tenants via same SchemaRegistry
        backend.schema_registry.deploy_schema(
            "tenant_a", "video_colpali_smol500_mv_frame"
        )
        backend.schema_registry.deploy_schema(
            "tenant_b", "video_colpali_smol500_mv_frame"
        )

        # Verify isolation - each tenant has their own schema
        schemas_a = backend.schema_registry.get_tenant_schemas("tenant_a")
        schemas_b = backend.schema_registry.get_tenant_schemas("tenant_b")

        assert len(schemas_a) == 1
        assert len(schemas_b) == 1
        assert (
            schemas_a[0].full_schema_name == "video_colpali_smol500_mv_frame_tenant_a"
        )
        assert (
            schemas_b[0].full_schema_name == "video_colpali_smol500_mv_frame_tenant_b"
        )

    def test_idempotent_deployment(self, get_backend):
        """Test that deploying same schema twice is idempotent"""
        backend = get_backend("idempotent_test")

        # Deploy twice
        result1 = backend.schema_registry.deploy_schema(
            "idempotent_test", "video_colpali_smol500_mv_frame"
        )
        result2 = backend.schema_registry.deploy_schema(
            "idempotent_test", "video_colpali_smol500_mv_frame"
        )

        # Both should succeed and return same name
        assert result1 == result2
        assert result1 == "video_colpali_smol500_mv_frame_idempotent_test"

        # Should only have one schema registered
        schemas = backend.schema_registry.get_tenant_schemas("idempotent_test")
        assert len(schemas) == 1

    def test_invalid_tenant_id_rejected(self, get_backend):
        """Test that invalid tenant IDs are rejected"""
        backend = get_backend("valid_tenant")

        # Invalid characters
        with pytest.raises(
            ValueError, match="only alphanumeric, underscore, and colon allowed"
        ):
            backend.schema_registry.deploy_schema(
                "tenant-with-dash", "video_colpali_smol500_mv_frame"
            )

        # Empty tenant_id
        with pytest.raises(ValueError, match="tenant_id is required"):
            backend.schema_registry.deploy_schema("", "video_colpali_smol500_mv_frame")

    def test_invalid_schema_name_rejected(self, get_backend):
        """Test that invalid schema names are rejected"""
        backend = get_backend("test_tenant")

        # Empty schema name
        with pytest.raises(ValueError, match="schema_name is required"):
            backend.schema_registry.deploy_schema("test_tenant", "")

    def test_nonexistent_schema_fails(self, get_backend):
        """Test that deploying nonexistent schema raises exception"""
        backend = get_backend("test_tenant_nonexistent")

        with pytest.raises(Exception, match="Failed to load base schema"):
            backend.schema_registry.deploy_schema(
                "test_tenant_nonexistent", "nonexistent_schema_xyz"
            )


@pytest.mark.integration
@pytest.mark.ci_fast
class TestSchemaRegistryDeletion:
    """Test schema deletion paths — both the registry-tracked happy path
    and the orphan-recovery case (Vespa has the schema, registry doesn't).

    The orphan case is the bug from
    ``.claude/plans/i-would-like-you-melodic-sunbeam.md``: a SIGKILL or
    crash mid-cleanup can leave the registry's tombstone half-written
    while Vespa still has the schema. Without the union-of-sources
    discovery in ``delete_tenant_schemas``, a subsequent DELETE returned
    200 OK while the orphan stayed in Vespa, blocking the next deploy
    with ``Refusing to deploy: Vespa has schemas X that are not in
    SchemaRegistry``.
    """

    def test_delete_tenant_schema_round_trip(self, get_backend):
        """Deploy → DELETE → verify gone from BOTH registry and Vespa.

        Round-trip: registry write + Vespa deploy → registry tombstone +
        Vespa redeploy without the schema. After the cycle, the running
        application generation must no longer list the tenant-namespaced
        schema.
        """
        backend = get_backend("del_round_trip")

        backend.schema_registry.deploy_schema(
            "del_round_trip", "video_colpali_smol500_mv_frame"
        )
        full_name = "video_colpali_smol500_mv_frame_del_round_trip"

        # Sanity: schema is deployed in Vespa.
        deployed_before = backend.schema_manager.list_deployed_document_types()
        assert full_name in deployed_before, (
            f"setup failure — schema {full_name!r} not in Vespa after deploy"
        )

        # Delete via the bulk path (same code DELETE /admin/tenants takes).
        deleted = backend.schema_manager.delete_tenant_schemas("del_round_trip")
        assert full_name in deleted

        # Vespa-side: schema removed from running application.
        deployed_after = backend.schema_manager.list_deployed_document_types()
        assert full_name not in deployed_after, (
            f"orphan: {full_name!r} still in Vespa after delete_tenant_schemas"
        )

        # Registry-side: tombstoned.
        assert backend.schema_registry.get_tenant_schemas("del_round_trip") == [], (
            "registry still lists schemas for the deleted tenant"
        )

    def test_delete_tenant_schema_recovers_orphan(self, get_backend):
        """Simulate the kill-recovery case: registry empty, Vespa has it.

        Build the inconsistent state by deploying a schema, then directly
        clearing the registry entry without going through the redeploy
        path (mimics a SIGKILL between ``unregister_schema`` and the
        application-package redeploy). ``delete_tenant_schemas`` must
        still drop the schema from Vespa using the Vespa-side discovery
        leg, otherwise the next deploy fails with
        ``Refusing to deploy: Vespa has schemas X that are not in
        SchemaRegistry``.
        """
        backend = get_backend("del_orphan")
        tenant_id = "del_orphan"
        full_name = "video_colpali_smol500_mv_frame_del_orphan"

        backend.schema_registry.deploy_schema(
            tenant_id, "video_colpali_smol500_mv_frame"
        )
        deployed_before = backend.schema_manager.list_deployed_document_types()
        assert full_name in deployed_before, "setup failure — schema not deployed"

        # Force registry-only tombstone (no redeploy) → orphan in Vespa.
        # This is what an interrupted cleanup leaves behind.
        backend.schema_registry.unregister_schema(
            tenant_id, "video_colpali_smol500_mv_frame"
        )
        assert backend.schema_registry.get_tenant_schemas(tenant_id) == [], (
            "setup failure — registry still has the entry"
        )
        deployed_mid = backend.schema_manager.list_deployed_document_types()
        assert full_name in deployed_mid, (
            "setup failure — Vespa already dropped the schema; cannot test "
            "orphan recovery"
        )

        # Run the cleanup. Pre-fix this returned [] (registry empty → no
        # work, redeploy skipped). Post-fix it must include the orphan.
        deleted = backend.schema_manager.delete_tenant_schemas(tenant_id)
        assert full_name in deleted, (
            f"delete_tenant_schemas did not see the Vespa-side orphan "
            f"{full_name!r} (returned {deleted!r}). The union-of-sources "
            f"discovery is broken — this is the regression the plan fixed."
        )

        deployed_after = backend.schema_manager.list_deployed_document_types()
        assert full_name not in deployed_after, (
            f"orphan {full_name!r} still in Vespa after kill-recovery cleanup"
        )

    def test_delete_schema_only_tenant_succeeds(self, get_backend):
        """``DELETE /admin/tenants`` must clean up tenants that have a
        schema in Vespa + registry but no ``tenant_metadata`` record.

        ``/ingestion/upload`` auto-deploys the per-tenant ingestion schema
        without going through the tenant-create flow, so the tenant_metadata
        document is never created. Pre-fix the early 404 check in
        ``delete_tenant_internal`` rejected DELETE for these tenants — the
        e2e session-start orphan reconciliation could not clean them up
        and the suite kept piling on schemas every run.
        """
        backend = get_backend("schema_only_tenant")
        tenant_id = "schema_only_tenant"
        backend.schema_registry.deploy_schema(
            tenant_id, "video_colpali_smol500_mv_frame"
        )
        full_name = "video_colpali_smol500_mv_frame_schema_only_tenant"
        deployed = backend.schema_manager.list_deployed_document_types()
        assert full_name in deployed, "setup failure — schema not deployed"

        # No tenant_metadata exists for this tenant — delete must still
        # succeed and drop the schema. Calling the schema-side leg
        # directly mirrors what ``delete_tenant_internal`` now does
        # after the early-404 check was loosened.
        deleted = backend.schema_manager.delete_tenant_schemas(tenant_id)
        assert full_name in deleted, (
            f"delete_tenant_schemas missed {full_name!r}; returned {deleted!r}"
        )

        deployed_after = backend.schema_manager.list_deployed_document_types()
        assert full_name not in deployed_after, (
            f"schema {full_name!r} still in Vespa after schema-only tenant delete"
        )

    def test_delete_tenant_does_not_drop_peer_tenant_orphan(self, get_backend):
        """Deleting tenant A must not silently drop tenant B's Vespa-only orphan.

        ``delete_tenant_schemas`` previously built the redeploy survivor list
        from the registry only. A peer tenant's Vespa-only orphan (schema in
        Vespa but no registry entry — e.g. from an interrupted earlier
        cleanup) was excluded from survivors and silently dropped by the
        ``allow_schema_removal=True`` redeploy. Multi-tenant data loss.

        Setup builds the inconsistent peer state by deploying tenant B's
        schema, then tombstoning the registry entry without redeploy. Then
        delete tenant A. Tenant B's orphan must either survive the redeploy
        or the call must refuse with a clear error.
        """
        from cogniverse_core.registries.exceptions import BackendDeploymentError

        backend = get_backend("victim_a")
        peer_tenant = "victim_b"
        a_full = "video_colpali_smol500_mv_frame_victim_a"
        b_full = "video_colpali_smol500_mv_frame_victim_b"

        backend.schema_registry.deploy_schema(
            "victim_a", "video_colpali_smol500_mv_frame"
        )
        backend.schema_registry.deploy_schema(
            peer_tenant, "video_colpali_smol500_mv_frame"
        )

        # Capture the peer's registry record BEFORE we tombstone it so the
        # finally-block can re-register and clean up properly. Without
        # this the peer orphan blocks every later test in the module.
        peer_info = backend.schema_registry.get_tenant_schemas(peer_tenant)[0]
        backend.schema_registry.unregister_schema(
            peer_tenant, "video_colpali_smol500_mv_frame"
        )

        try:
            deployed_before = backend.schema_manager.list_deployed_document_types()
            assert a_full in deployed_before, "setup failure — tenant A schema missing"
            assert b_full in deployed_before, "setup failure — peer orphan missing"

            try:
                backend.schema_manager.delete_tenant_schemas("victim_a")
            except BackendDeploymentError:
                deployed_after = backend.schema_manager.list_deployed_document_types()
                assert b_full in deployed_after, (
                    f"peer orphan {b_full!r} dropped despite refused delete"
                )
                return

            deployed_after = backend.schema_manager.list_deployed_document_types()
            assert a_full not in deployed_after, (
                f"tenant A schema {a_full!r} still in Vespa"
            )
            assert b_full in deployed_after, (
                f"peer orphan {b_full!r} silently dropped when deleting tenant A"
            )
        finally:
            # Re-register peer_tenant in the registry, then delete it via
            # the bulk path so both Vespa and the registry are clean.
            try:
                backend.schema_registry.register_schema(
                    tenant_id=peer_tenant,
                    base_schema_name=peer_info.base_schema_name,
                    full_schema_name=peer_info.full_schema_name,
                    schema_definition=peer_info.schema_definition,
                    config=peer_info.config,
                )
                backend.schema_manager.delete_tenant_schemas(peer_tenant)
            except Exception as cleanup_exc:
                # Surface but don't mask the test result.
                logger.warning(f"peer orphan cleanup failed: {cleanup_exc}")


@pytest.mark.integration
@pytest.mark.ci_fast
class TestBulkTenantDelete:
    """Atomic bulk-delete path that drops schemas for multiple tenants in
    one Vespa redeploy. Single-tenant ``delete_tenant_schemas`` refuses
    when an unreconstructable peer-tenant orphan exists; the bulk variant
    accepts every orphan tenant in ``deletion_targets`` simultaneously
    so the survivor reconstruction succeeds.
    """

    def test_bulk_delete_two_orphan_tenants_in_one_redeploy(self, get_backend):
        backend = get_backend("bulk_a")

        backend.schema_registry.deploy_schema(
            "bulk_a", "video_colpali_smol500_mv_frame"
        )
        backend.schema_registry.deploy_schema(
            "bulk_b", "video_colpali_smol500_mv_frame"
        )
        a_full = "video_colpali_smol500_mv_frame_bulk_a"
        b_full = "video_colpali_smol500_mv_frame_bulk_b"

        backend.schema_registry.unregister_schema(
            "bulk_a", "video_colpali_smol500_mv_frame"
        )
        backend.schema_registry.unregister_schema(
            "bulk_b", "video_colpali_smol500_mv_frame"
        )

        deployed_before = backend.schema_manager.list_deployed_document_types()
        assert a_full in deployed_before
        assert b_full in deployed_before

        deleted = backend.schema_manager.delete_tenant_schemas_bulk(
            ["bulk_a", "bulk_b"]
        )
        assert a_full in deleted
        assert b_full in deleted

        deployed_after = backend.schema_manager.list_deployed_document_types()
        assert a_full not in deployed_after
        assert b_full not in deployed_after

    def test_bulk_delete_empty_list_is_noop(self, get_backend):
        backend = get_backend("bulk_noop")
        assert backend.schema_manager.delete_tenant_schemas_bulk([]) == []


@pytest.mark.integration
@pytest.mark.ci_fast
class TestSharedSchemaRegistry:
    """The process-wide ``_shared_schema_registry`` singleton is the
    coupling that lets two backends in one process see each other's
    ``register_schema`` writes. Without this round-trip test the wiring
    is constructor-acceptance only — backends accept the parameter but
    nothing proves they actually share state.
    """

    def test_two_ingestion_backends_share_registry(
        self,
        vespa_instance,
        temp_config_manager,
        schema_loader,
        wipe_non_protected_schemas,
    ):
        """Two ingestion backends for different tenants share one registry.

        Reset the singleton, build two ingestion backends keyed by
        different tenants, register a schema via backend A, and assert
        backend B's ``schema_registry`` reflects it without any
        cross-backend reload.
        """
        BackendRegistry._shared_schema_registry = None
        BackendRegistry._backend_instances.clear()

        registry = BackendRegistry.get_instance()
        config = {
            "backend": {
                "url": "http://localhost",
                "config_port": vespa_instance["config_port"],
                "port": vespa_instance["http_port"],
            }
        }
        backend_a = registry.get_ingestion_backend(
            name="vespa",
            tenant_id="shared_a",
            config=config,
            config_manager=temp_config_manager,
            schema_loader=schema_loader,
        )
        backend_b = registry.get_ingestion_backend(
            name="vespa",
            tenant_id="shared_b",
            config=config,
            config_manager=temp_config_manager,
            schema_loader=schema_loader,
        )

        assert backend_a is not backend_b, (
            "ingestion backends are tenant-keyed; same instance defeats the test"
        )
        assert backend_a.schema_registry is backend_b.schema_registry, (
            "shared singleton broken — backends got two distinct registries"
        )

        backend_a.schema_registry.deploy_schema(
            "shared_a", "video_colpali_smol500_mv_frame"
        )
        peer_view = backend_b.schema_registry.get_tenant_schemas("shared_a")
        assert peer_view, (
            "peer backend's schema_registry doesn't see schemas registered "
            "via backend_a — singleton sharing is broken or each "
            "registry holds its own in-memory cache."
        )
        assert peer_view[0].base_schema_name == "video_colpali_smol500_mv_frame"


@pytest.mark.integration
@pytest.mark.ci_fast
class TestDeleteSchemaDirect:
    """Direct coverage for the per-schema ``delete_schema`` entry point on
    VespaSchemaManager. The bulk path ``delete_tenant_schemas`` exercises it
    transitively, but the input-validation guards (system schema list,
    tenant suffix typo) need explicit coverage so a future refactor can't
    quietly weaken them.
    """

    def test_delete_schema_refuses_protected_metadata(self, get_backend):
        """``tenant_metadata`` and the other system schemas must not be
        droppable via the per-tenant entry point.
        """
        backend = get_backend("guard_protected")
        with pytest.raises(ValueError, match="Protected schemas"):
            backend.schema_manager.delete_schema("guard_protected", "tenant_metadata")

    def test_delete_schema_refuses_empty_inputs(self, get_backend):
        backend = get_backend("guard_empty")
        with pytest.raises(ValueError, match="tenant_id is required"):
            backend.schema_manager.delete_schema("", "video_colpali_smol500_mv_frame")
        with pytest.raises(ValueError, match="base_schema_name is required"):
            backend.schema_manager.delete_schema("guard_empty", "")

    def test_delete_schema_round_trip(self, get_backend, wipe_non_protected_schemas):
        """Deploy → delete_schema → schema gone from Vespa AND tombstoned."""
        backend = get_backend("single_delete")
        backend.schema_registry.deploy_schema(
            "single_delete", "video_colpali_smol500_mv_frame"
        )
        full_name = "video_colpali_smol500_mv_frame_single_delete"

        deployed = backend.schema_manager.list_deployed_document_types()
        assert full_name in deployed

        removed = backend.schema_manager.delete_schema(
            "single_delete", "video_colpali_smol500_mv_frame"
        )
        assert removed == full_name

        deployed_after = backend.schema_manager.list_deployed_document_types()
        assert full_name not in deployed_after
        assert backend.schema_registry.get_tenant_schemas("single_delete") == []


@pytest.mark.integration
@pytest.mark.ci_fast
class TestDeleteFailureSemantics:
    """The reordered delete path promises two invariants:

    1. ``Vespa redeploy fails → registry untouched.`` The whole point of
       deploying Vespa first and tombstoning the registry second.
    2. ``Vespa redeploy succeeds, registry tombstone fails → Vespa is
       authoritative; failure is logged.`` The asymmetric rollback
       trade-off the docstring acknowledges.

    Both paths are failure-injection tests; without them the docstring
    is aspirational.
    """

    def test_vespa_failure_leaves_registry_untouched(
        self, get_backend, monkeypatch, wipe_non_protected_schemas
    ):
        """Inject a ``_deploy_package`` failure during delete_tenant_schemas.

        After the failure the registry must STILL have the schema entry —
        if it doesn't, the next delete retry would observe an empty
        registry, redeploy successfully, and Vespa would still hold the
        schema as an orphan. That's the bug the reorder was supposed to
        eliminate.
        """
        backend = get_backend("vespa_fail")
        backend.schema_registry.deploy_schema(
            "vespa_fail", "video_colpali_smol500_mv_frame"
        )
        assert backend.schema_registry.get_tenant_schemas("vespa_fail")

        original = backend.schema_manager._deploy_package

        def failing_deploy(*args, **kwargs):
            raise RuntimeError("simulated Vespa redeploy failure")

        monkeypatch.setattr(backend.schema_manager, "_deploy_package", failing_deploy)

        with pytest.raises(RuntimeError, match="simulated Vespa redeploy failure"):
            backend.schema_manager.delete_tenant_schemas("vespa_fail")

        monkeypatch.setattr(backend.schema_manager, "_deploy_package", original)

        assert backend.schema_registry.get_tenant_schemas("vespa_fail"), (
            "registry was tombstoned despite Vespa redeploy failure — "
            "ordering invariant violated"
        )

        # Cleanup: properly delete now that the failure is uninjected so
        # the schema doesn't pollute later module-scoped tests.
        backend.schema_manager.delete_tenant_schemas("vespa_fail")

    def test_registry_tombstone_failure_does_not_block_vespa_removal(
        self, get_backend, monkeypatch, wipe_non_protected_schemas
    ):
        """When ``unregister_schema`` fails AFTER Vespa removal, the
        schema must still be gone from Vespa (Vespa is authoritative).

        The asymmetric failure mode the docstring acknowledges: Vespa
        is the source of truth; a registry-side write failure is logged
        and recovered later.
        """
        backend = get_backend("tombstone_fail")
        backend.schema_registry.deploy_schema(
            "tombstone_fail", "video_colpali_smol500_mv_frame"
        )
        full_name = "video_colpali_smol500_mv_frame_tombstone_fail"
        assert full_name in backend.schema_manager.list_deployed_document_types()

        original_unregister = backend.schema_registry.unregister_schema

        def failing_unregister(*args, **kwargs):
            raise RuntimeError("simulated registry tombstone failure")

        monkeypatch.setattr(
            backend.schema_registry, "unregister_schema", failing_unregister
        )

        backend.schema_manager.delete_tenant_schemas("tombstone_fail")

        monkeypatch.setattr(
            backend.schema_registry, "unregister_schema", original_unregister
        )

        deployed_after = backend.schema_manager.list_deployed_document_types()
        assert full_name not in deployed_after, (
            "Vespa redeploy must have removed the schema even though the "
            "registry tombstone failed; Vespa is authoritative."
        )

        # Reconcile so the lingering registry entry doesn't pollute
        # later tests. The tombstone failure was simulated; manually
        # clean it up.
        try:
            backend.schema_registry.unregister_schema(
                "tombstone_fail", "video_colpali_smol500_mv_frame"
            )
        except Exception:
            pass
