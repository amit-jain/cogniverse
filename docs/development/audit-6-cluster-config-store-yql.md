# Audit Cycle 6 — Cluster: config_store versioned-get YQL injection

| Site | Class | Failure | Fix |
|------|-------|---------|-----|
| `vespa/config/config_store.py:341` `get_config(version=N)` | C | `doc_id` embeds `config_id` (built raw from tenant_id/service/config_key via `_create_document_id`) and was interpolated into `documentid = "{doc_id}"` unescaped — a quote in `config_key` breaks/injects the YQL. The latest-version branch already used `yql_quote`; the versioned branch did not | wrap the documentid value in `yql_quote(doc_id)` (consistent with the sibling branch) |

## Test (`tests/backends/unit/test_config_store_yql_escape.py`, fails on pre-fix)

Captures the built YQL for a `config_key='key"; bad'` and asserts the
documentid value equals `yql_quote(doc_id)` (inner quote escaped). Pre-fix the
raw interpolation left the quote unescaped.
