# 2026-05 Audit — Deferred fixes

**Status:** Active. Each section ships independently. Tracked from this file via inline pointer-TODOs at the affected sites (`# TODO(audit-2026-05): see docs/development/audit-2026-05-deferred-fixes.md#<anchor>`). Approved scope: ship in priority order.

The 2026-05 audit (single-pass, five orthogonal bug classes per `.claude/rules/audit.md`) surfaced 26 HIGH-confidence findings. Eight landed in-session (Batches A–E); the rest are deferred here per audit Done criterion #3 — every deferred item has a written plan, not "we'll catch it next audit." Commits already landed: `b5bb9a47` (hunt list), `995e0279` (1D embedding reshape), `94a7dae6` (canonical tenant in search_backend), `70073ec9` (YQL escape + ISO→int in memory layer), `abba096f` (Vespa status_code check), `bcdb14ef` (DSPy LM context bind), `4e86d631` (naive analytics timestamp).

## Priority key

| Tier | Meaning |
|---|---|
| P0 | Silent-wrong output on a production happy path. Ship before the next ingestion change. |
| P1 | Crashes on real input the route advertises as valid; ops-grade pain. Ship this cycle. |
| P2 | Untested surface with non-trivial blast radius. Ship as the surface gets touched (or in a dedicated coverage push). |
| P3 | Hygiene / lurking footgun without a current trigger. Ship when the area is next refactored. |

---

## P0 — Silent-wrong output

### A. Float-coercion of LM outputs (#9) — agents/{text_analysis, deep_research, profile_selection, query_enhancement}_agent.py

**The bug.** Four agents call `float(result.confidence)` (or `float(result.<field>)`) on a DSPy module output without try/except. Real DSPy LMs return strings like `"high"`, `"low"`, or `"85%"` — `float()` raises `ValueError`, which bubbles up the FastAPI route as a 500. The advertised contract is "returns a confidence float in [0,1]"; the 500 is on the LM, not the route.

**Affected sites** (after Phase 7 review gate):
- `libs/agents/cogniverse_agents/text_analysis_agent.py:176` — `float(result.confidence) if result.confidence else 0.0` (no try/except)
- `libs/agents/cogniverse_agents/deep_research_agent.py:290` — same pattern, no guard
- ~~`libs/agents/cogniverse_agents/profile_selection_agent.py:309`~~ — already wrapped in try/except (ValueError, AttributeError, TypeError). False positive from the initial sweep.
- ~~`libs/agents/cogniverse_agents/query_enhancement_agent.py:326`~~ — same, already guarded. False positive.

**Fix.** Add a small shared helper in `libs/agents/cogniverse_agents/_confidence.py`:

```python
def parse_confidence(raw: object, default: float = 0.0) -> float:
    """Best-effort parse of an LM confidence output.

    Real LMs return floats, percent strings ("85%"), label strings
    ("high"/"medium"/"low"), or empty strings. None of those should
    crash the route. Map labels to numeric bands; strip "%"; fall
    back to ``default`` on any other shape.
    """
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().rstrip("%").lower()
    if not s:
        return default
    try:
        v = float(s)
    except ValueError:
        v = {"high": 0.9, "medium": 0.5, "low": 0.1}.get(s, default)
    if v > 1.0:  # "85%" → 0.85
        v = v / 100.0
    return max(0.0, min(1.0, v))
```

**Tests** (`tests/agents/unit/test_confidence_parse.py`): assert exact mapping for "high"/"medium"/"low" labels, percent strings, plain floats, empty string, None, garbage. Real-LM round-trip in one of the four agents' existing integration tests — confirm a `"high"` confidence from a real LM produces `0.9` in the response, not a 500.

**Touched files:** 5 (helper + 4 agents) + 1 test. Within Phased Execution budget.

### B. claim_extractor LM context (#7) — agents/graph/claim_extractor.py:213

**The bug.** `ClaimExtractor._extract_segment` invokes `self._module(text_segment=..., entity_hints=..., modality_hint=...)` with NO `dspy.context(lm=...)` wrap. The class has no `_llm_config` attribute at all — relies entirely on `dspy.settings.lm` configured at ingestion-worker startup. Per-tenant LM config does not reach claim extraction.

**Design constraint.** `ClaimExtractor` runs inside the ingestion worker (`libs/runtime/cogniverse_runtime/ingestion_worker/worker.py:172`) which currently binds a single worker-startup LM. Multi-tenant LM dispatch at this layer requires (1) threading tenant_id through the ingestion pipeline to the extractor, (2) resolving each tenant's `llm_config` from the registry at extraction time, (3) wrapping the per-segment module call with `dspy.context(lm=create_dspy_lm(tenant_lm_config))`.

**Plan.**
1. Add `tenant_id: str` arg to `ClaimExtractor.__init__` (already partially threaded — verify the worker passes it).
2. Add `llm_config_resolver: Callable[[str], LLMEndpointConfig | None]` injectable dep; default to a module-level helper that reads from the registry the way the runtime does.
3. In `_extract_segment`, resolve the tenant LM config once per call and wrap the module call.
4. Add a test in `tests/agents/unit/test_claim_extractor_lm_context.py` mirroring the Batch D `test_dspy_lm_context_binding.py` pattern — sentinel LM captured at module invocation time.

**Touched files:** claim_extractor.py + worker.py wiring + 1 test. ~3 files.

---

## P1 — Crashes / ops-grade pain

### C. Vespa filter-condition quoting (#8) — vespa/search_backend.py:822-845

**The bug.** `_build_filter_conditions` has three independent quoting failures:
1. Range filter `{"gte": value}` interpolates `value` UNQUOTED. A string ISO timestamp `"2024-01-01"` becomes YQL `ts >= 2024-01-01` (malformed; Vespa 400).
2. `None` filter value falls through to `field contains "None"` — silently filters on the literal string `"None"` instead of treating null as "no filter" or raising.
3. NaN/Inf numeric values produce `field = nan` / `field = inf` — malformed YQL (no quoting around `nan`, Vespa rejects).

**Fix.** Replace the trailing `else` chain with a typed dispatch:

```python
def _safe_filter_value(value: object, field: str) -> str:
    if value is None:
        # explicit no-op marker; caller should not pass None — raise to surface
        raise ValueError(f"None filter value for field {field!r}; remove the key")
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"Non-finite filter value {value!r} for {field!r}")
        return repr(value)
    if isinstance(value, (int, bool)):
        return repr(int(value))
    return yql_quote(str(value))
```

Apply to all three call sites (range gte/lte, equality, contains).

**Tests** (`tests/backends/unit/test_filter_condition_quoting.py`): exact-YQL assertions for {plain str, str with embedded `"`, int, float, ISO date string in range, NaN, Inf, None, embedded backslash}.

**Touched files:** 1 production + 1 test.

### D. Naive-datetime sweep (#10) — 14+ sites across evaluation/dashboard

**The bug.** Multiple time-window queries to Phoenix (UTC) use `datetime.now()` (naive). On a non-UTC host (IST = UTC+5:30), the window is off by the local offset — dashboards miss recent traces or fetch wrong ones. This is the 4th-audit IST host bug regressed in 14+ sites.

**Affected sites** (verified):
- `libs/evaluation/cogniverse_evaluation/data/traces.py:41,136`
- `libs/evaluation/cogniverse_evaluation/core/solvers.py:229,331,353`
- `libs/dashboard/cogniverse_dashboard/app.py:366-385, 699-700`
- `libs/dashboard/cogniverse_dashboard/tabs/routing_evaluation.py:81-99`
- `libs/dashboard/cogniverse_dashboard/tabs/optimization.py:211, 504, 1224-25, 1254, 1557-58, 1603`
- `libs/dashboard/cogniverse_dashboard/tabs/profile_metrics.py:99`
- `libs/dashboard/cogniverse_dashboard/tabs/orchestration_annotation.py:104`

**Fix.** Mechanical replacement: every `datetime.now()` whose result feeds a Phoenix/Vespa query window or a stored-for-cross-pod-compare value becomes `datetime.now(timezone.utc)`. Streamlit's `st.date_input` / `st.time_input` return NAIVE objects; combine them with an explicit `tzinfo=timezone.utc` at the call site.

**Sequencing.** This touches 8 files. Per Phased Execution (≤5 files), split as:
- D1: evaluation/{data/traces.py, core/solvers.py} (2 files)
- D2: dashboard/app.py + dashboard/tabs/routing_evaluation.py + dashboard/tabs/profile_metrics.py (3 files)
- D3: dashboard/tabs/optimization.py + dashboard/tabs/orchestration_annotation.py (2 files)

**Tests** (`tests/dashboard/unit/test_time_window_tz.py` + similar in tests/evaluation/): a freeze-time test on a non-UTC TZ host that constructs the query window and asserts the start/end are aware UTC; a Phoenix-mock test that asserts the kwargs passed to `provider.telemetry.traces.get_spans` are aware.

### E. Typed 503 on dep-not-wired routes (#26) — runtime/routers/{agents,search,ingestion}.py

**The bug.** `_ensure_dispatcher()`, `get_config_manager_dependency()`, `get_schema_loader_dependency()` raise `RuntimeError("dependency not configured")` when FastAPI lifespan hasn't installed dep overrides yet. The route handler doesn't catch it; FastAPI surfaces it as **500 Internal Server Error**. The `wiki` router does this correctly with `503 Service Unavailable`.

**Fix.** Replace `raise RuntimeError(...)` in each dependency helper with `raise HTTPException(status_code=503, detail="Service initializing; retry")`. Three files:
- `libs/runtime/cogniverse_runtime/routers/agents.py:120` (`_ensure_dispatcher`)
- `libs/runtime/cogniverse_runtime/routers/search.py:34,47` (config_manager + schema_loader)
- `libs/runtime/cogniverse_runtime/routers/ingestion.py` (find equivalent)

**Tests** (`tests/runtime/unit/test_dep_not_wired_returns_503.py`): ASGITransport against an app with no `dependency_overrides` set. POST each route. Assert `status_code == 503`, response body matches `{"detail": "Service initializing; retry"}`, and the wiki router's existing 503 path remains green (regression guard).

**Touched files:** 3 routers + 1 test.

### F. _submit_cron_workflow swallows Argo errors (#16, F-class from Phase 1+6)

**The bug.** `libs/runtime/cogniverse_runtime/routers/tenant.py:470-490` posts to Argo, logs on non-2xx, returns `None`. The caller at `:877` does not check the return, so `POST /admin/tenant/{tid}/jobs` writes the ConfigStore row AND returns `{"status": "created"}` even when Argo rejected the manifest (cron expression invalid, namespace missing, etc.). The schedule never fires, the user sees "created", operators are blind.

**Fix.**
1. `_submit_cron_workflow` raises `HTTPException(503, ...)` on non-2xx instead of returning `None`.
2. `create_job` either (a) wraps in try/except + rolls back the ConfigStore write, or (b) writes the ConfigStore row AFTER the Argo POST succeeds (preferred — fewer compensating-action paths).
3. `delete_job` / `_delete_cron_workflow` same pattern — don't tombstone ConfigStore if Argo failed to delete.

**Tests** (`tests/runtime/integration/test_create_job_argo_failure.py`): use a `httpx.MockTransport` to simulate Argo 503 / 422 / 400; assert (a) `POST /admin/tenant/{tid}/jobs` returns the corresponding error, (b) no ConfigStore row was persisted. Cannot rely on the existing `test_argo_wiring_roundtrip.py` — it mocks `_submit_cron_workflow` itself, the exact path being fixed.

**Touched files:** tenant.py (2 helpers, 2 callers) + 1 test.

### G. Gateway artifact load silently swallows failures (#15)

**The bug.** `libs/agents/cogniverse_agents/gateway_agent.py:158-199` `_load_artifact` has a broad `except Exception: logger.debug(...)` that masks Mem0/Phoenix telemetry outages — tenants silently revert to default thresholds with no metric, no alert, no span attribute. Production traffic loses optimized thresholds during transient outages, and ops can't tell.

**Fix.** Distinguish three load outcomes:
1. **No artifact persisted** — legitimate first-time tenant; use defaults. Span attribute `gateway.artifact_load_status=no_artifact`.
2. **Artifact persisted, load succeeded** — use loaded thresholds. Span attribute `gateway.artifact_load_status=loaded`.
3. **Load failed (connection refused, deserialization error, telemetry provider down)** — use defaults BUT emit metric `gateway_artifact_load_failures_total{tenant=...}` and span attribute `gateway.artifact_load_status=failed`. Operators must distinguish (3) from (1).

**Tests** (`tests/agents/unit/test_gateway_artifact_load.py`): three tests — one per outcome — driving a real Phoenix Docker via the existing `phoenix_test_server` fixture. Assert the exact span attribute and metric on each path. The current test (`test_gateway_artifact_load_failure_uses_defaults`) is Class A — it locks in the broken silent-fallback contract; replace it.

**Touched files:** gateway_agent.py + test file (replaced).

---

## P2 — Untested surface

These are Class B findings (audit Phase 2). Each item below has zero (or 404-only) test coverage today. The blast radius matters because every untested entry point can ship a regression invisible to the test sweep.

### H. SSE event streams (#17) — runtime/routers/events.py:129, 155

**Surface.** `GET /events/workflows/{workflow_id}` and `GET /events/ingestion/{job_id}` (SSE). Zero tests for subscribe, replay-from-offset, heartbeat, complete/error early-break. Any regression in `_event_stream` silently breaks every dashboard subscriber.

**Plan.** Use `httpx.AsyncClient` against `ASGITransport(app)` with `stream=True`. Drive an end-to-end: POST a job → subscribe to `/events/ingestion/{id}` → assert the heartbeat event arrives within 5s, then a `status` event, then a `complete` event, then the stream closes. Cover replay: subscribe with `Last-Event-ID: 3` and assert the stream begins at offset 4. Cover early-break: emit an `error` event and assert the client receives it before disconnect.

### I. Event cancel routes — 404 only (#18) — events.py:181, 211

**Surface.** `POST /events/workflows/{id}/cancel` and `POST /events/ingestion/{id}/cancel`. Existing tests (`test_api_e2e.py:803, 812`) only cover the 404 path.

**Plan.** Build a queue → cancel it → assert response `{"cancelled": true, "message": "..."}`. Cover the race: cancel after a queue has completed — assert idempotent response. Two tests, ASGITransport-driven.

### J. Debug routes — memsnap / memreset (#19) — routers/debug.py:39, 103

**Surface.** Tracemalloc snapshot + diff routes. The env-gate `COGNIVERSE_DEBUG_MEM` is also untested. Only the bash sampler at `scripts/memsnap_sampler.sh` shells out to them.

**Plan.** Set `COGNIVERSE_DEBUG_MEM=1`, POST `/admin/debug/memsnap` twice, POST `/admin/debug/memreset`, POST `/admin/debug/memsnap` again. Assert (a) snapshot 1 → snapshot 2 diff shape (`size_diff`, top-N sort, file/line columns), (b) reset zeros the baseline, (c) without the env var, both routes return 403.

### K. Synthetic API routes (#20) — synthetic/api.py:59, 88, 99, 129, 146

**Surface.** `POST /synthetic/generate`, `POST /synthetic/batch/generate`, `GET /synthetic/optimizers`, `GET /synthetic/optimizers/{name}`, `GET /synthetic/health`. No HTTP-level tests; existing tests only construct `SyntheticDataRequest` Python objects.

**Plan.** Five tests, ASGITransport-driven, covering (a) the HTTPException ladder (400 on ValueError, 422 on ValidationError, 500 wrap), (b) the tenant_id required guard, (c) the per-batch loop semantics for `/batch/generate`, (d) the optimizer not-found 404, (e) health check returns 200.

### L. Wiki server-side routes (#21) — wiki.py:93, 115, 124

**Surface.** `GET /wiki/topic/{slug}`, `GET /wiki/lint`, `DELETE /wiki/topic/{slug}`. The existing tests in `tests/messaging/unit/test_runtime_client_crud.py` mock the underlying `httpx` call client-side; the server route handlers are never exercised.

**Plan.** Three tests, ASGITransport-driven. `DELETE`: assert `wm.delete_page` was called with the correct `doc_id` (`f"wiki_topic_{safe}_{slug}"`); RuntimeError → 500; OK → 200. `GET /lint`: assert response shape matches `wm.lint()`. `GET /topic/{slug}`: 200 vs 404 paths.

### M. Standalone agent FastAPI apps (#22) — 12 routes across {search, summarizer, detailed_report}_agent.py

**Surface.** Each of these three agents has a standalone `app = FastAPI(...)` and `if __name__ == "__main__"` block — they CAN be deployed standalone, but in production traffic flows through the runtime dispatcher (which bypasses the standalone HTTP wrappers). The wrappers' guards (503-on-uninit, 400-on-missing-tenant, response envelope) are untested.

**Plan.** Decide first:
- **Option α** — keep the standalone apps, write 4 tests per agent (= 12 tests total) covering `/health`, `/agent.json`, `/process`, and the agent-specific route(s).
- **Option β** — delete the standalone apps + their `__main__` blocks, since runtime dispatch is the production path. **Per the project hard rule, deletion of production code requires explicit user approval — propose, do not act.** Estimate: 3 file edits, ~60 LOC removed per file.

Recommend Option β to the user; if approved, ship as a single commit per agent with the deletion + a CHANGELOG entry.

### N. Dashboard render_*_tab functions (#23) — 12 tabs

**Surface.** 12 `render_*_tab` entry points in `libs/dashboard/cogniverse_dashboard/tabs/*.py`. Tests only cover helper functions (`_phoenix_base_url`, `_load_pending_items`, etc.); the render paths themselves are never invoked.

**Plan.** Use `streamlit.testing.v1.AppTest` (Streamlit 1.34+). One test per tab, asserting (a) no exceptions on render with a happy-path session_state, (b) the expected widgets render (using `at.dataframe`, `at.button`, etc.). 12 tests, ~one-to-two assertions each.

**Sequencing.** Land one cluster per commit:
- N1: approval_queue, evaluation, optimization (3 tabs)
- N2: backend_profile, tenant_management, memory_management (3 tabs)
- N3: config_management, routing_evaluation, profile_metrics, orchestration_annotation (4 tabs)
- N4: embedding_atlas, rlm_ab_compare (2 tabs)

### O. CLI subcommands (#24)

**Surface.** `libs/cli/cogniverse_cli/admin.py` (reconcile-orphans wrapper), `libs/cli/cogniverse_cli/graph.py` (stats/search/neighbors/path subcommands), `libs/cli/cogniverse_cli/sandbox.py` (sync/status).

**Plan.** Click's `CliRunner.invoke(cli, [...])` against each command. Assert exit code, stdout shape, error stdout. The graph subcommands call the runtime HTTP API — use a `httpx.MockTransport` to drive happy + error responses.

### P. Argo workflow YAMLs (#25)

**Surface.** `workflows/scheduled-maintenance.yaml`, `workflows/tenant-provisioning.yaml`, `workflows/video-ingestion.yaml`. Zero validation/lint. The literal string `# Add actual Vespa backup command` exists in `scheduled-maintenance.yaml` — a known-incomplete script in shipped infrastructure.

**Plan.**
1. Add `argo lint workflows/*.yaml` as a CI step (the Argo CLI is already in the deploy pipeline).
2. For each embedded bash script (Vespa backup, Phoenix backup, cleanup-old-backups), extract into `scripts/argo/*.sh` with `set -euo pipefail` + a unit test that runs the script with `--dry-run` or against a fixture container.
3. Replace the `# Add actual Vespa backup command` placeholder with a real `vespa-cli` (or the project's equivalent) invocation, gated by a `VESPA_BACKUP_DESTINATION` env var so it's a no-op in dev.

---

## P3 — Lurking footguns / hygiene

### Q. `"lvt" in schema_name.lower()` + case-inconsistent `"_sv_" in name` (#1)

**The bug.** `libs/vespa/cogniverse_vespa/embedding_processor.py:103, 131`. Substring matching on a model-family token (`lvt`) without word boundaries — any future schema name containing the substring (e.g. `audio_alvtree_index`, `multi_lvt_patch_index`) silently collapses an (N, dim) multi-vector ndarray to its first row. And `_sv_` is matched case-sensitively while `lvt` is case-insensitive — a schema named `..._SV_...` would take the multi-vector path despite being intent-as-single-vector.

**Status.** No current production schema triggers the data-loss path (every current `lvt`-named schema is also `_sv_`-named, so the truncation matches intent). Lurking; ship before adding any new schema family.

**Fix.** Replace substring matching with a typed predicate driven from schema metadata: `is_single_vector_schema(schema_name) -> bool` returns `True` only when the schema's deployed config declares `tensor<float>(d0[N])` (single-vector). Use the `cogniverse_vespa.config.config_store.VespaConfigStore` to look it up at processor construction; cache per schema name.

**Tests** (`tests/backends/unit/test_schema_name_matching.py`): a parameterised matrix — for each `(schema_name, expected_path)` pair, assert `process_embeddings` produces the expected shape. Include `audio_alvtree_index`, `video_VIDEOPRISM_SV_global`, `video_videoprism_sv_global` — the substring-collision and case-inconsistency tripwires.

---

## Test rewrites (Class A — locking in broken contracts)

### R. tests/evaluation/unit/test_solvers.py (#12, #13)

**The bug.** `TestBatchSolver::test_trace_loader_*` — three tests assert only `result is not None`. The autouse fixture `mock_provider_for_unit_tests` in `tests/evaluation/conftest.py:389` returns an empty `pd.DataFrame()`, so the solver short-circuits at `if df.empty` and never exercises ground-truth extraction, ground-truth strategy dispatch, ground-truth confidence wiring, reranking, or any of the trace_data field assembly. The tests would pass if every one of those code paths were deleted.

`TestLiveSolver::test_live_trace_solver_continuous` mocks `asyncio.sleep` to inject `KeyboardInterrupt`, swallows it with try/except, asserts nothing. The continuous test hangs the test sweep (timeout 90s, exit 143).

**Plan.** Replace with real-Phoenix-Docker tests using the `phoenix_test_server` fixture (`tests/evaluation/conftest.py:88-135`). For each test:
1. Seed N spans with known trace_ids into the test Phoenix via the live `provider.telemetry.spans` writer.
2. Invoke the solver against the real provider.
3. Assert (a) `len(state.metadata["loaded_traces"]) == N`, (b) the exact trace_ids match, (c) every key in `trace_data` exists (the 9 fields named at `solvers.py:246-255`), (d) `ground_truth_stats["total_traces"] == N` and `traces_with_ground_truth == X` where X is precisely seeded.

For `test_live_trace_solver_continuous`, run with `continuous=False, timeout=1s`, seed one span between polls, assert that span appears with its exact trace_id. The hang itself is a separate bug — the autouse fixture's `get_spans` mock collides with the `continuous=True` loop; switching to a real Phoenix removes both issues at once.

### S. tests/agents/unit/test_query_enhancement.py:1011-1068 (#14)

**The bug.** The test patches `RelationshipExtractorTool.extract_comprehensive_relationships`, then IGNORES the patched call and uses the literal dict it just authored as `phase2_result`. The downstream assertions check keys on the dict literal, not on any code-produced output. The downstream `ComposableQueryAnalysisModule` block only verifies `hasattr(module, "forward")` and inspects the signature.

**Plan.** Delete the literal-dict-roundtrip block (lines 1029-1051). Replace with a test that invokes `RelationshipExtractorTool.extract_comprehensive_relationships` against a real GLiNER+spaCy stack (or a real `ComposableQueryAnalysisModule(real_extractor, real_analyzer).forward("query", search_context)`) and asserts the shape of the live output. The signature-check at lines 1064-1068 stays as a separate test.

---

## Hunt-list additions already shipped

Per audit Done criterion #5, the 2026-05 cycle's new pattern regexes shipped in `.claude/rules/audit.md` via commit `b5bb9a47`. Future audits scale by adding regexes, not by running another audit. The list now covers:

- Substring-vs-token schema name match (Q above)
- Document v1 selection-expression interpolation (sibling of YQL `contains` injection)
- Naive datetime explicitly passed to a query window kwarg (D above)
- Fire-and-forget `asyncio.create_task` (LOW finding in Phase 3)
- Streamlit naive date/time input (D above, dashboard sites)
- `float()` coercion of LM output (A above)
- `isinstance(_, (int, float))` timestamp gate without magnitude check
- Standalone-agent FastAPI app routes (M above)
- Streamlit render-tab entry points (N above)
- Local `_escape` helpers that handle only `"` and miss `\` (closed by Batch B)

---

## Order of operations

1. **P0 first.** A (float-LM-output) and B (claim_extractor) are silent-wrong on production happy paths.
2. **P1 next, by ease.** Simplest wins: E (typed 503, 3 files), C (filter quoting, 1 file). Then F (Argo error propagation), G (gateway artifact load), D (datetime sweep, 3 sub-commits).
3. **P2 in dedicated coverage push.** H–P each ship as their own commit; expect a 1–2 week coverage burndown.
4. **P3 + test rewrites last.** Q lurks; ship before next schema family. R + S land as the evaluation/query_enhancement modules get next touched.

## What this is NOT

- Not a license to defer P0/P1 past the next code change in the touched area. Inline pointer-TODOs at the affected sites (added in a follow-up commit) keep each ticket visible.
- Not exhaustive — the 5 MED and 4 LOW findings live in the audit transcript, not here. They get hunt-list pattern entries (already shipped) rather than dedicated tickets.
- Not authorisation to delete production code. Items M-β and other "propose deletion" notes require explicit user approval per `[[feedback-never-delete-on-grep-alone]]`.

## Cross-references

- The audit protocol that produced these findings: `.claude/rules/audit.md`
- The strict-commit workflow each fix routes through: `.claude/rules/strict-commit.md`
- The hunt-list extensions this cycle shipped: `.claude/rules/audit.md` Class C section, commit `b5bb9a47`
- The plan-doc model this file mirrors: `docs/development/pipeline-cache-multi-pod-todo.md`
