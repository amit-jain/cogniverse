# Audit Cycle 6 — Cluster: /analyze used query params, not a body

| Site | Class | Failure | Fix |
|------|-------|---------|-----|
| `agents/text_analysis_agent.py:300` `POST /analyze` | B | declared `text/tenant_id/analysis_type` as bare scalar params → FastAPI treats them as QUERY params; a JSON-body POST (the real client shape) 422s | introduced `AnalyzeRequest(BaseModel)` and accept it as the request body |

## Tests (`tests/agents/unit/test_text_analysis_agent.py`)

- `test_analyze_endpoint` / `test_analyze_endpoint_error_handling` now POST a
  JSON body and assert 200/500 + the response shape.
- Also fixed a **pre-existing** failure in these tests: they used a bare
  `TestClient(app)` while the app has a lifespan (sets the module
  config_manager); switched to `with TestClient(app) as client:` (matching the
  passing health test). Both were failing before this change.
- `test_analyze_endpoint_without_tenant_id_fails` asserts the 422 loc is
  `["body", "tenant_id"]`.
