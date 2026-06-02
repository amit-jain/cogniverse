# Audit Cycle 6 — Cluster: hardcoded Phoenix GraphQL endpoint

| Site | Class | Failure | Fix |
|------|-------|---------|-----|
| `dashboard/tabs/optimization.py:896` | C | the golden-dataset query POSTed to a hardcoded `http://localhost:6006/graphql`, ignoring the configured Phoenix endpoint | use `_phoenix_base_url()` (the evaluation tab's helper that reads `session_state["phoenix_url"]`, localhost fallback) — `f"{_phoenix_base_url()}/graphql"` |

Coverage: `_phoenix_base_url` is already unit-tested in
`tests/dashboard/unit/test_evaluation_tab.py`; this change reuses that tested,
config-driven helper instead of a hardcoded constant.
