# Audit Cycle 6 — Cluster: per-call pyvespa app in metadata ops

| Site | Class | Failure | Fix |
|------|-------|---------|-----|
| `vespa/backend.py:1302,1340,1377,1444` (create/get/query/delete_metadata_document) | PERF | each metadata op called `make_vespa_app(url, port)` → a fresh `Vespa(url=...)` (new connection pool) per call | `_metadata_vespa_app()` caches the app, keyed on `(url, port)` so it rebuilds only after a deploy-time url/port override; the 4 methods route through it |

## Test (`tests/backends/unit/test_metadata_app_cache.py`, fails on pre-fix)

- `test_metadata_app_is_cached_across_calls`: two calls return the same object.
- `test_metadata_app_rebuilt_when_url_changes`: a url change yields a new app.
