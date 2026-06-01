# Audit Cycle 6 — Cluster: delete_schema cross-tenant guard too lenient

| Site | Class | Failure | Fix |
|------|-------|---------|-----|
| `vespa/vespa_schema_manager.py:724` `delete_schema` | D | `target` is built from the canonicalized tenant (`acme`→`acme:acme`→suffix `_acme_acme`), but the defensive cross-tenant guard compared `target.endswith("_" + tenant_id.replace(":","_"))` using the RAW tenant_id (`_acme`) — a substring of the canonical suffix, so a wrong-tenant target ending in `_acme` passed the guard | compute the guard suffix from `canonical_tenant_id(tenant_id)` so it matches how `target` was built |

## Test (`tests/backends/unit/test_delete_schema_suffix_guard.py`, fails on pre-fix)

`get_tenant_schema_name` is stubbed to return a wrong-tenant `video_other_acme`
(ends in raw `_acme`); with the fix the guard raises ("does not carry the
expected ... tenant suffix"). Pre-fix the raw `_acme` check passed and the guard
never fired.
