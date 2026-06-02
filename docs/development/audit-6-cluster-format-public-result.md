# Audit Cycle 6 — Cluster: _format_public_result short-circuit misfire

| Site | Class | Failure | Fix |
|------|-------|---------|-----|
| `agents/search_agent.py:170` `_format_public_result` | D | short-circuited on `if "metadata" in result`. Backend results are `{**sr.document.metadata, ...}`, so a Vespa `metadata` sub-field made the function return the RAW Vespa-shaped result (leaking native field names, missing the public `document_id`/`temporal_info` shape) | only short-circuit when already public-shaped: `"document_id" in result and "metadata" in result` (raw results carry `id`/`documentid`, never the public `document_id`) |

## Test (`tests/agents/unit/test_format_public_result.py`, fails on pre-fix)

A raw result with a `metadata` key is still reshaped (gets `document_id`,
nested raw fields under `metadata`); an already-public result passes through.
