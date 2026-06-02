# Audit Cycle 6 — Cluster: spurious variant warning on the default path

| Site | Class | Failure | Fix |
|------|-------|---------|-----|
| `agents/optimizer/signature_variants.py` `selected_for_tenant` | C | when a tenant had no override, `chosen` fell back to `"default"`, then `is_registered(agent_type, "default")` returned False (no variants registered) → logged a misleading WARNING that the tenant "selected variant 'default' ... not registered" on the normal path | only warn for an EXPLICIT non-default variant that is unregistered (operator typo); the no-override/default path returns the baseline silently |

## Test (`tests/agents/unit/test_signature_variants.py::TestSelectedForTenantWarning`)

- no override → returns default, no "not registered" warning (fails on pre-fix).
- explicit unregistered variant → returns default + warning.
