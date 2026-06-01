# Audit Cycle 6 — Cluster: /health rebuilds config stack per probe

Review summary for the PERF finding on the runtime health endpoint (plus the
related `AgentRegistry` httpx-client-per-init leak it triggered).

## Finding & fix

| Site | Class | Failure on happy path | Fix |
|------|-------|-----------------------|-----|
| `runtime/routers/health.py` `health_check` | PERF | every `GET /health` called `create_default_config_manager()` (re-parsing config.json + rebuilding the config stack) and constructed a fresh `AgentRegistry`, which opens an `httpx.AsyncClient(timeout=10.0)` in `__init__`. A k8s probe loop (every few seconds) re-parsed config and leaked a client per hit. | cache the AgentRegistry behind `@lru_cache(maxsize=1)` (`_get_agent_registry`) and reuse it; `backends`/`agents` are still queried live each probe. `lru_cache` does not cache exceptions, so a failed build still retries and surfaces as 503 on the next probe. |

This also fixes the related LOW finding (`AgentRegistry.__init__` opens an httpx
client unconditionally) **as it manifests on the health path** — only one
registry/client now exists for probes. The unconditional-client construction
for other callers is unchanged and tracked separately.

## Tests (`tests/runtime/unit/test_health_endpoints.py`)

Real FastAPI app via `TestClient`. An autouse fixture clears
`_get_agent_registry.cache_clear()` between tests so each test's patched
dependencies are re-read.

| Test | Assertion |
|------|-----------|
| `test_config_stack_built_once_across_probes` | 3 × `GET /health` → `create_default_config_manager` called once and `AgentRegistry` constructed once (pre-fix: 3 each) |
| existing `test_health_check_full` / `test_health_response_structure` / `test_health_returns_503_not_500_on_config_error` | still pass — 503 retry path preserved because lru_cache doesn't cache the raising build |
