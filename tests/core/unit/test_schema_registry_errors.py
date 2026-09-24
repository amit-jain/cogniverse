"""SchemaRegistry raises typed, chained errors for schema-load failures."""

from unittest.mock import MagicMock

import pytest

from cogniverse_core.registries.exceptions import SchemaLoadError
from cogniverse_core.registries.schema_registry import SchemaRegistry

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def test_deploy_schema_load_failure_raises_typed_chained_error():
    """A caller must be able to tell a missing schema file (permanent, never
    retry) from storage being down (transient) - a bare Exception without a
    cause destroyed the type and left only the message text."""
    loader = MagicMock()
    cause = FileNotFoundError("no such schema: video_missing")
    loader.load_schema.side_effect = cause
    registry = SchemaRegistry(
        config_manager=MagicMock(), backend=MagicMock(), schema_loader=loader
    )
    registry.schema_exists = MagicMock(return_value=False)

    with pytest.raises(
        SchemaLoadError, match="Failed to load base schema 'video_missing'"
    ) as exc_info:
        registry.deploy_schema("acme:acme", "video_missing")

    assert exc_info.value.__cause__ is cause


def test_failed_intent_retirement_preserves_deployment_error_and_pending_record():
    from types import SimpleNamespace

    from cogniverse_core.registries.exceptions import BackendDeploymentError
    from tests.core.unit.test_schema_deployment_intents import Store

    store = Store()
    persist = store.compare_and_set_config

    def fail_retirement(**kwargs):
        if kwargs["config_value"].get("state") == "absent":
            raise ConnectionError("metadata service unavailable during retirement")
        return persist(**kwargs)

    store.compare_and_set_config = fail_retirement
    primary = BackendDeploymentError("configuration generation did not converge")

    def deploy(_schemas):
        raise primary

    registry = SchemaRegistry(
        config_manager=SimpleNamespace(store=store),
        backend=SimpleNamespace(deploy_schemas=deploy),
        schema_loader=SimpleNamespace(load_schema=lambda _: {"name": "wiki_pages"}),
    )
    with pytest.raises(BackendDeploymentError) as failure:
        registry.deploy_schema("acme:prod", "wiki_pages")
    assert str(failure.value) == (
        "Backend deployment failed for schema 'wiki_pages_acme_prod': "
        "configuration generation did not converge. "
        "The durable definition is retained for late activation."
    )
    assert failure.value.__cause__ is primary
    assert failure.value.__notes__ == [
        "Intent retirement failed: Cannot persist deployment intent for "
        "'wiki_pages_acme_prod': metadata service unavailable during retirement. "
        "The durable record is retained for recovery."
    ]
    assert [
        (record["state"], record["registration"]["full_schema_name"])
        for record in registry._deployment_intents.pending()
    ] == [("pending", "wiki_pages_acme_prod")]


@pytest.mark.parametrize(
    "names, message",
    [
        ([], "base_schema_names is required"),
        (["wiki_pages", "wiki_pages"], "Duplicate base schema names: ['wiki_pages']"),
    ],
)
def test_batch_rejects_invalid_names_before_activation(names, message):
    import re
    from types import SimpleNamespace

    from tests.core.unit.test_schema_deployment_intents import Store

    packages = []
    registry = SchemaRegistry(
        config_manager=SimpleNamespace(store=Store()),
        backend=SimpleNamespace(
            deploy_schemas=lambda schemas: packages.append(schemas)
        ),
        schema_loader=SimpleNamespace(load_schema=lambda base: {"name": base}),
    )
    with pytest.raises(ValueError, match=re.escape(message)):
        registry.deploy_schemas("acme:prod", names)
    assert packages == []


def test_single_schema_uses_batch_registration():
    from types import SimpleNamespace

    from tests.core.unit.test_schema_deployment_intents import Store

    registry = SchemaRegistry(
        config_manager=SimpleNamespace(store=Store()),
        backend=SimpleNamespace(),
        schema_loader=SimpleNamespace(),
    )
    calls = []

    def batch(tenant_id, base_schema_names, config=None, force=False):
        calls.append((tenant_id, base_schema_names, config, force))
        return ["wiki_pages_acme_prod"]

    registry.deploy_schemas = batch
    assert (
        registry.deploy_schema("acme:prod", "wiki_pages", {"a": 1}, True)
        == "wiki_pages_acme_prod"
    )
    assert calls == [("acme:prod", ["wiki_pages"], {"a": 1}, True)]


def test_batch_reloads_definition_when_peer_deletes_cached_schema():
    from types import SimpleNamespace

    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    packages = []

    def deploy(schemas):
        packages.append(schemas)
        return True

    registry = SchemaRegistry(
        config_manager=SimpleNamespace(store=store),
        backend=SimpleNamespace(deploy_schemas=deploy),
        schema_loader=SimpleNamespace(load_schema=lambda base: {"name": base}),
    )
    assert registry.deploy_schema("acme:prod", "wiki_pages") == "wiki_pages_acme_prod"
    refresh = registry._get_all_schemas

    def peer_deletes_before_refresh():
        registry.unregister_schema("acme:prod", "wiki_pages")
        return refresh()

    registry._get_all_schemas = peer_deletes_before_refresh
    assert registry.deploy_schemas("acme:prod", ["wiki_pages", "provenance"]) == [
        "wiki_pages_acme_prod",
        "provenance_acme_prod",
    ]
    assert [[row["name"] for row in package] for package in packages] == [
        ["wiki_pages_acme_prod"],
        ["wiki_pages_acme_prod", "provenance_acme_prod"],
    ]
    assert {
        info.full_schema_name for info in registry.get_tenant_schemas("acme:prod")
    } == {"wiki_pages_acme_prod", "provenance_acme_prod"}


def _registry_with_peer(peer_action):
    """A registry whose backend activation is followed by ``peer_action``
    from another registry over the same store, before registration."""
    import json
    from types import SimpleNamespace

    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    tenant, base, name = "acme:prod", "wiki_pages", "wiki_pages_acme_prod"
    loader = SimpleNamespace(
        load_schema=lambda _base: {"name": base, "document": {"fields": ["v2"]}}
    )
    packages = []

    def deploy_then_peer_acts(schemas):
        packages.append([schema["name"] for schema in schemas])
        peer = SchemaRegistry(
            config_manager=SimpleNamespace(store=store),
            backend=SimpleNamespace(deploy_schemas=lambda _schemas: True),
            schema_loader=loader,
        )
        peer_action(peer, tenant, base, name)
        return True

    registry = SchemaRegistry(
        config_manager=SimpleNamespace(store=store),
        backend=SimpleNamespace(deploy_schemas=deploy_then_peer_acts),
        schema_loader=loader,
    )
    registry.register_schema(
        tenant_id=tenant,
        base_schema_name=base,
        full_schema_name=name,
        schema_definition=json.dumps({"name": name, "document": {"fields": ["v1"]}}),
    )
    return registry, store, packages, (tenant, base, name)


def _row(store, tenant, base):
    from cogniverse_sdk.interfaces.config_store import ConfigScope

    return store.get_config(
        tenant_id=tenant,
        scope=ConfigScope.SCHEMA,
        service="schema_registry",
        config_key=f"schema_{base}",
    )


def test_a_peer_tombstone_after_activation_is_never_overwritten():
    """An existing schema's re-registration after activation is conditional on
    the row the deploy was decided from: a peer that deleted the schema in
    between owns the row, nothing is rolled back over it, and the conflict is
    reported as a retryable tombstone conflict."""
    from cogniverse_core.registries.exceptions import SchemaRevisionConflictError

    registry, store, packages, (tenant, base, name) = _registry_with_peer(
        lambda peer, tenant, base, _name: peer.unregister_schema(tenant, base)
    )

    with pytest.raises(SchemaRevisionConflictError) as caught:
        registry.deploy_schema(tenant, base)

    assert (caught.value.schema_name, caught.value.peer_revision) == (
        name,
        "tombstone",
    )
    assert caught.value.activated is True
    assert caught.value.retryable is True
    assert str(caught.value) == (
        f"Schema {name!r} was deleted by another process after this deploy read "
        f"its registry row; the activation stands and that revision was not "
        f"overwritten. Retry the deploy."
    )
    assert _row(store, tenant, base).config_value["deleted"] is True
    assert packages == [[name]]


def test_a_peer_registration_after_activation_is_reported_as_one():
    import json

    from cogniverse_core.registries.exceptions import SchemaRevisionConflictError

    def peer_registers(peer, tenant, base, name):
        peer.register_schema(
            tenant_id=tenant,
            base_schema_name=base,
            full_schema_name=name,
            schema_definition=json.dumps(
                {"name": name, "document": {"fields": ["v3"]}}
            ),
        )

    registry, store, packages, (tenant, base, name) = _registry_with_peer(
        peer_registers
    )

    with pytest.raises(SchemaRevisionConflictError) as caught:
        registry.deploy_schema(tenant, base)

    assert caught.value.peer_revision == "registration"
    assert str(caught.value) == (
        f"Schema {name!r} was re-registered by another process after this deploy "
        f"read its registry row; the activation stands and that revision was not "
        f"overwritten. Retry the deploy."
    )
    assert json.loads(_row(store, tenant, base).config_value["schema_definition"])[
        "document"
    ] == {"fields": ["v3"]}
    assert packages == [[name]]
