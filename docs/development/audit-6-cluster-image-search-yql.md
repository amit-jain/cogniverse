# Audit Cycle 6 — Cluster: image_search YQL filter injection

| Site | Class | Failure | Fix |
|------|-------|---------|-----|
| `agents/image_search_agent.py:300,304` `_search_vespa` | C | filter values interpolated raw into `contains(field, '{value}')` — a quote in a filter value breaks/injects the YQL | wrap with `yql_quote(...)` (same helper graph_manager already uses) |

## Test (`tests/agents/unit/test_image_search_agent.py::TestImageSearchFilterEscaping`)

Patches `requests.post`, captures the built `yql` for `detected_objects="cat's toy"`,
asserts the escaped `yql_quote` form is present and the raw single-quoted form
is not. Fails on pre-fix.
