# Audit Cycle 6 — Cluster: RLM A/B tab read the wrong session key

| Site | Class | Failure | Fix |
|------|-------|---------|-----|
| `dashboard/tabs/rlm_ab_compare.py:264` | E | read `st.session_state.get("tenant_id")`, but the app shell stores the gate-selected tenant under `"current_tenant"` (app.py:344). `"tenant_id"` was never set, so the tab always fell back to a manual text input and ignored the active tenant | read `"current_tenant"` to match the shell (and every other tab) |

## Test (`tests/dashboard/unit/test_rlm_ab_tenant.py`, fails on pre-fix)

AppTest renders the tab with `current_tenant` set and asserts the manual
"Tenant id" fallback input is NOT shown (it used the session tenant). Pre-fix
the manual input appeared. A complementary test confirms the fallback shows
when no tenant is in session.
