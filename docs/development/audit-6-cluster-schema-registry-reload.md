# Audit Cycle 6 — Cluster: SchemaRegistry reload drops stale entries

Review summary for the Class-E "reload from storage rebuilt the in-memory map
additively" finding.

## Finding & fix

| Site | Class | Failure on happy path | Fix |
|------|-------|-----------------------|-----|
| `core/registries/schema_registry.py:_load_schemas_from_storage` | E | `get_tenant_schemas` and `_get_all_schemas` re-read storage to pick up a peer instance's changes, but the load assigned `self._schemas[key] = ...` without clearing — so a schema another instance **deleted** lingered in memory after reload, and `_get_all_schemas` (which feeds the deploy package) could re-collect and resurrect it | build a fresh dict and swap it in (`self._schemas = loaded`) only after a successful load — deletions are reflected; a failed load still falls back to the existing cache (preserving the documented behaviour) |

## Test-fidelity fix (Class A)

The `mock_config_manager` fixture's store was unfaithful: `set_config` was a
no-op and `list_all_configs` returned `[]`. Two tracking tests passed only
because the additive reload never cleared the unpersisted in-memory entry —
i.e. they relied on the bug. The fixture now uses a backing dict so
`set_config` persists and `list_all_configs`/`get_config` read it back; the
register/deploy → reload round-trip behaves like the real store. Two tests
that injected rows via `list_all_configs.return_value` now seed the store via
`config_manager.seed_schema(...)`.

## Tests

| Test | Assertion |
|------|-----------|
| `TestReloadReflectsDeletions::test_reload_drops_schema_deleted_from_storage` | store returns `{video, audio}` then `{video}`; after reload `_get_all_schemas()` is `{video}` (fails on pre-fix — `audio` lingers) |
| `test_register_schema_adds_to_tracking`, `test_get_tenant_schemas_returns_only_tenant_schemas` | now pass via the faithful store (register/deploy persist → reload reads back) instead of via the additive bug |

29 tests in the file pass; `test_admin_reconcile_orphans` (consumer) green.
