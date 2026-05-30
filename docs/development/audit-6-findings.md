# Audit Cycle 6 — Findings & Remediation Program

Single-pass audit per `.claude/rules/audit.md` (89 agents, 5 bug classes A–E + PERF/SLOP/HOLLOW-TEST/UNTESTED). Every CRIT/HIGH was adversarially verified against live code before inclusion; 3 false positives were dropped at the review gate. This doc is the multi-session tracker — update the per-item `status` as each ships with its real-boundary regression test.

**Totals:** 6 CRIT · 46 HIGH · 83 MED · 83 LOW = 218.

## Systemic root-cause clusters (fix the cause, not each symptom)

1. **Phoenix span-DataFrame flattening** — `get_spans_dataframe` returns flattened dotted columns (`attributes.routing.*`), but several sites read `attributes.routing` as a nested dict. Sites: `annotation_agent`, `orchestration_evaluator`, `finetuning/embedding_extractor`, dashboard `:.2f` format, quality_monitor (C4). Fix once with a shared accessor, apply everywhere.
2. **Canonical `org:tenant` format** — the colon/hyphen form breaks: `schema_registry` regex (rejects hyphen), `tenant.py` k8s cron name (raw colon = invalid), `agent_dispatcher` signature-variant lookup, `cross_tenant`/`federated` `node_id` split.
3. **Standalone FastAPI agent apps** — `get_agent_skills()` missing (×3 `/agent.json`), `config_manager` not set (×2 + text_analysis `__main__`); the "coverage" test is hollow (Class A MagicMock singleton).
4. **Sync I/O on the async event loop** — `document_agent` blocking `requests.post`, sync DSPy on loop (×3 agents), unbatched dashboard HTTP, one-doc-per-segment Vespa feeding, leaked per-request httpx client, unbounded `search_latencies`.
5. **Request bleed** — per-request canary overlay mutates dispatcher-cached cross-request-shared agents.
6. **Hollow tests** — finetuning trainers (`try/except pass` + `if mock.called:`), query_enhancement (`patch.object` on SUT asserting the mock's own return, one for a nonexistent method), standalone-apps, worker-failure (raises where prod returns).

## Phase plan (≤5 files/phase, each ships a real-boundary regression test; pause + verify between phases)

- **Tier 1 — CRIT (6):** C1 orchestrator plan index · C2 approval infinite loop · C3 worker failure-as-success · C4 live-traffic eval shape · C5 dashboard config wipe · C6 messaging auth partition.
- **Tier 2 — HIGH (46):** by cluster (Phoenix-flatten → tenant-format → standalone-apps → async-blocking → request-bleed → remaining), then the 10 hollow tests.
- **Tier 3 — MED (83)** then **Tier 4 — LOW (83)**, including SLOP fixes + the dead-code deletions (user pre-approved; confirm zero callers each).
- Defer to a written TODO only what genuinely needs live infra unavailable locally; mark `NEEDS_LIVE_BOUNDARY` items and verify against real Vespa/Phoenix when reachable.


## Progress log

**Tier 1 — CRIT: ✅ DONE (6/6).** All fixed locally on `main`, each with a real-boundary regression test that fails on the pre-fix code:
- C1 — remap plan dependency/parallel-group indices to surviving-step space (`test_orchestrator_agent.py`: strengthened skip-unknown test + parallel-remap + readiness-simulation).
- C2 — add `RUNNING→APPROVED` transition + advance index on auto-approved step (`test_decision_orchestrator.py`: real state-machine, `wait_for` fail-fast).
- C3 — raise on `status=failed/cancelled` pipeline envelope (`test_ingestion_worker_e2e.py`: real Redis, failed-envelope → no `mark_done`).
- C4 — `require_search_shape=False` keeps non-search agent spans + read raw `value` (`test_incremental_span_eval.py`: real Phoenix round-trip).
- C5 — `dataclasses.replace` preserves unedited global-config fields (`test_system_config_save.py`: real ConfigManager round-trip).
- C6 — write user→tenant map to SYSTEM partition + `infer=False` (`test_auth.py` partition-faithful + `test_user_tenant_mapping_real_mem0.py` real Vespa-Mem0).

**Tier 2 — HIGH: in progress.**

✅ **Standalone-apps cluster DONE** (commits `Add get_agent_skills accessor…`, `Wire config manager via lifespan…`):
- `A2AAgent.get_agent_skills()` public accessor added → the 3 `/agent.json` routes (search/summarizer/detailed_report) no longer 500 on a real agent.
- summarizer + detailed_report lifespans now build `create_default_config_manager()` → standalone apps start instead of crashing on the required-config_manager raise.
- text_analysis app gained a `lifespan` that calls `set_config_manager` → `/analyze` no longer 500s with "ConfigManager not initialized".
- Hollow `test_standalone_agent_apps.py` upgraded: real-lifespan tests (TestClient context manager) construct the real agent and assert `/agent.json` serves real skills + `/health` healthy.

✅ **Phoenix-flatten cluster DONE** (commits `Read flattened Phoenix span columns…`, `Coerce orchestration execution time…`):
- `finetuning/embedding_extractor` — read flattened `attributes.*` columns (no bare `attributes` column on a real Phoenix df) + tolerate `document_id`/`video_id`/`source_id` result ids (matching only `id` produced zero triplets). The whole embedding-finetuning triplet path was dead; now extracts. Unit test feeds the **probe-verified** flattened shape.
- dashboard `orchestration_annotation` — `_format_seconds()` coerces a possibly-string Phoenix attribute before `:.2f` (a string crashed the tab). Unit-tested.
- ⚠️ **`annotation_agent.py:230` reading `attributes.routing` = FALSE POSITIVE** — a live-Phoenix probe + the passing `test_identify_enqueue_assign_complete` confirm the provider returns `attributes.routing` as a **nested dict** (namespaced multi-key attrs are grouped into a dict column; only leaf conventions like `input.value`/`output.value` stay flat). `orchestration_evaluator` reads the same dict → also correct. The audit agent reasoned from the raw phoenix-client source; the review gate (run the live test) refuted it. **Do not "fix" — it would break working code.**

✅ **Tenant-format cluster DONE** (commits `Allow hyphens in schema registry…`, `Recover KG node id…`, `Canonicalize tenant key for signature variant…`, `Sanitize tenant id in Argo CronWorkflow…`):
- `schema_registry._validate_tenant_id` now allows hyphens, matching `validate_tenant_name` (a hyphenated tenant could be created but never deploy a schema). Cross-validator agreement test.
- `node_id_from_doc_id()` helper in `graph_schema` strips the exact `kg_node_{safe_tenant}_` prefix; `cross_tenant`/`federated` use it (positional `split('_',3)` glued the colon-tenant's tail onto node_ids). Round-trip test through the real `Node.doc_id`.
- signature-variant overrides keyed by `canonical_tenant_id` on both admin store/read and dispatcher resolve (sibling admin routes already canonicalized; this one didn't). NOTE: the finding's claimed mechanism (`require_tenant_id` canonicalizes) was inaccurate — it returns unchanged; the real gap is the cross-endpoint key-normalization inconsistency. Cross-form contract test (PUT simple → resolve colon).
- Argo CronWorkflow `metadata.name` sanitized via `_cron_workflow_name()` (RFC-1123: lowercase + `-`) on both create and delete; the `tenant` label uses `_sanitize_label_value`. Colon/underscore tenants produced Argo-illegal names → CronWorkflow create 4xx'd → scheduled jobs never ran.

✅ **Async / resource cluster — sync-on-loop + leaks DONE** (commits `Bound search latency window…`, `Close per-request orchestrator http client…`, `Offload document search HTTP…`, `Offload synchronous RLM calls…`):
- `SearchMetrics.search_latencies` → bounded `deque(maxlen=1000)`; lifetime avg from running totals (was an unbounded per-query leak on a process-lifetime-cached backend). Unit-tested for the bound + lifetime average.
- `_execute_orchestration_task` closes the per-request policy `httpx.AsyncClient` in a `finally` (success + error). Was leaking a client/pool per orchestration. Tested both paths.
- `document_agent._search_visual`/`_search_text` offload the blocking `requests.post` via `asyncio.to_thread`; `_search_hybrid` runs both via `asyncio.gather` (was sequential despite the "parallel" comment). Deterministic offload + concurrency tests.
- `multi_document_synthesis`/`temporal_reasoning`/`federated_query` offload the blocking `rlm.process` via `asyncio.to_thread`; multi-doc also offloads the no-RLM `_synthesise_without_rlm` (dspy.context stays inside the sync method → lands on the worker thread). Parametrized deterministic offload test.

✅ **Finetuning trainer bug + hollow tests DONE** (commit `Fix trainer eval_strategy kwarg and unhollow trainer tests`): H23 — `evaluation_strategy=` → `eval_strategy=` in `sft_trainer`/`dpo_trainer` (transformers renamed it; local SFT/DPO training raised TypeError on the installed 4.56.2). H43/H44 — removed the `try/except Exception: pass` + `if mock.called:` guards in `test_sft_trainer`/`test_dpo_trainer` so `_train_local` runs to completion with UNCONDITIONAL assertions (also fixed the SFT test's `patch("datasets.Dataset")` → `patch("datasets.Dataset.from_list")` so `isinstance(x, Dataset)` doesn't break). 14 tests now genuinely exercise the path; they fail on the pre-fix code.

✅ **request-bleed DONE** (commit `Isolate per-request agent state with contextvars`): moved `_dispatched_artefact` + `_memory_session_id` from instance attrs to module-level `contextvars.ContextVar` in `MemoryAwareMixin` (per-asyncio-task isolation; copied into `asyncio.to_thread`). Added `get_dispatched_artefact()`/`get_session_id()` accessors; updated 4 test files off the instance-attr contract + a root-conftest autouse reset fixture. New deterministic concurrency test (`test_memory_mixin_request_isolation`): two interleaved requests on one shared agent each read back their own overlay/session (fails on the old instance-attr code). Verified: isolation (2) + session-propagation (6) + artefact-overlay (12) + a2a (22) + canary real-Phoenix (8) + memory units (35).

~~⏸ request-bleed — superseded by the entry above; original plan:~~ `_apply_artefact_overlay`/`_scoped_session` mutate instance attrs (`_dispatched_artefact`, `_memory_session_id`) on a dispatcher-CACHED, cross-request-shared agent (`_get_search_agent`, `_gateway_agent`) → concurrent requests race/bleed. Correct fix = move both to `contextvars.ContextVar` in `MemoryAwareMixin` (per-async-task isolation, API preserved). Blast radius: 4 test files + the a2a streaming stub assert the instance-attr contract directly (`agent._dispatched_artefact == ...`, `not hasattr(...)`), so they need updating too. Deferred pending approval (cross-cutting core-mixin concurrency change).

✅ **query_enhancement hollow tests DONE** (commit `Unhollow query enhancement relevance and entity-match tests`): H38 `test_relevance_score_calculation` + H39 `test_entity_matching_logic` rewritten to call the REAL `_calculate_relationship_relevance` / `_find_matching_entities` (they `patch.object`'d the SUT and asserted the mock's own return) with exact assertions (relevance == 1.0 and 0.5; matched == ['robots','soccer']). H40 `test_search_result_enhancement` DELETED — it `patch.object`'d a method (`_enhance_results_with_context`) that does not exist in production (phantom). 64 tests in the file pass.

✅ **All remaining hollow tests DONE:**
- H37 `test_orchestration_span_emitted` (commit `Assert orchestration span attributes…`) — was asserting nothing (fixture never set telemetry_manager → prod short-circuited). Now a recording telemetry manager + exact span-attribute assertions (query truncated to 200, agent_sequence joined) + no-op-without-telemetry + missing-tenant-guard cases.
- H41 `test_generate_report_empty_results` (commit `Actually invoke generate_report…`) — never called the agent (only inspected the ReportRequest). Now calls `_generate_report([])` and asserts the empty-case ReportResult shape (total_results==0, non-empty exec summary, list findings/recs, dict confidence).
- H42 `TestCliArgumentParser` (commit `Expose build_parser…`) — hand-built a parallel argparse parser that had drifted (phantom `routing` mode, 5 real modes missing, wrong tenant default). Extracted `build_parser()` from `optimization_cli.main()`; the test now drives the REAL parser: all 13 real modes accepted, `online-routing-eval` accepted, `routing` rejected, cleanup tenant default is None.

**HOLLOW-TEST / Class-A tier COMPLETE** (worker-failure, standalone-apps, trainers ×2, query_enhancement ×3, orchestration_span, detailed_report, optimization_cli).

Remaining (all NEEDS_LIVE_BOUNDARY — need real Vespa to verify, best in a fresh session): VLM-descriptions-dropped (HIGH/E); image_search multi-vector format (HIGH/D); architectural async (one-doc-per-segment batch, `load_for_request` cache, dashboard nested loop). Then MED (83) + LOW (83) tiers, incl. the user-pre-approved dead-code deletions. (`schema_registry` regex, `tenant.py` cron name, `agent_dispatcher` signature-variant, `node_id` split) → async-blocking cluster (`document_agent` sync POST, sync DSPy on loop, one-doc-per-segment feed, leaked httpx client, unbounded `search_latencies`) → request-bleed (shared cached agent overlay) → VLM-descriptions-dropped → remaining → 10 hollow tests.


## CRIT tier (6)

### [C1] `libs/agents/cogniverse_agents/orchestrator_agent.py:1210-1226 (_create_plan) and :1224 (_calculate_dependencies call)`
- **class** D · **diagnosis** feature-gone-haywire · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: When the DSPy planner proposes any agent NOT in the registry in the middle/start of the sequence (which the code explicitly anticipates via unavailable_agents), _create_plan computes each step's depends_on using the UNFILTERED enumerate index i, but appends surviving steps to a list that skips filtered agents. The depends_on index space and parallel_groups index space then no longer match the steps list. The last (execution) agent gets a depends_on pointing at itself or out of range. Concretely for the planner output 'query_enhancement_agent, nonexistent_agent, search_agent' (registry has the two real ones), the surviving search_agent step gets depends_on=[1] which is search_agent itself -> 
- **fix**: After filtering, remap indices: build old_index->new_step_index map as steps are appended, then recompute depends_on and parallel_groups against the surviving step indices (or compute depends_on AFTER the filter loop using the new positions). Add a regression test that drives _create_plan with a mid-sequence unknown agent AND runs _execute_plan, asserting the execution agent actually executes and 
- **status**: ☐ open

### [C2] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/approval/orchestrator.py:212-287`
- **class** B · **diagnosis** feature-gone-haywire · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: DecisionOrchestrator.execute() infinite-loops on the primary approval happy path. _execute_current_step only advances current_step_index for steps where requires_approval is False (lines 286-287); approval steps rely on apply_approvals() to advance. When an approval-required step yields pending_review_count==0 AND rejection_count==0 (all items auto-approved, an empty list, or a non-list result), no RUNNING transition matches (AWAITING_APPROVAL needs pending>0; RUNNING->COMPLETED needs NOT requires_approval), so the state stays RUNNING, the step index never advances, and the same step re-executes forever. The all-auto-approved case is the central intended use of an auto-approval threshold, so
- **fix**: In _execute_current_step, after processing an approval step that yields zero pending and zero rejections, advance current_step_index and set context so a RUNNING->RUNNING/COMPLETED transition can fire (e.g. add a transition for current_step_requires_approval AND pending==0, or unconditionally advance the index and only pause when pending_review_count>0). Add a guard against re-executing the same i
- **status**: ☐ open

### [C3] `/home/amitjain/source/cogniverse/libs/dashboard/cogniverse_dashboard/tabs/config_management.py:195-216`
- **class** D · **diagnosis** feature-gone-haywire · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: The System Config 'Save' handler rebuilds SystemConfig(...) passing only 11 of the 25+ dataclass fields. SystemConfig is GLOBAL (one per deployment, not per-tenant) and set_system_config persists system_config.to_dict() wholesale, so every field the form omits is reset to its dataclass default. Clicking Save wipes inference_service_urls -> {}, redis_url -> '', minio_endpoint -> '', and resets agent_registry_url to http://localhost:8000 (plus ingestion_api_url, application_name, llm_engine, agents, colpali_inference_url, iter_retrieval_*, adapter_cache_dir, video_processing_profiles). This silently breaks ingestion (MinIO), cross-pod messaging (Redis), inference (denseon/vllm URLs), and A2A d
- **fix**: Start from the loaded system_config and only override the fields the form edits: use dataclasses.replace(system_config, video_agent_url=..., backend_port=..., ...) instead of constructing a fresh SystemConfig(). Add a round-trip test: populate a config with non-default infra fields, render+submit the form via AppTest, reload via get_system_config, assert inference_service_urls/redis_url/minio_endp
- **status**: ☐ open

### [C4] `libs/evaluation/cogniverse_evaluation/quality_monitor.py:412-448 (evaluate_live_traffic) + libs/evaluation/cogniverse_evaluation/span_evaluator.py:171-179 (get_recent_spans)`
- **class** D · **diagnosis** feature-gone-haywire · **verdict** CONFIRMED · **conf** HIGH
- **broken**: Live-traffic quality evaluation silently produces ZERO samples for 3 of the 4 agent types (SUMMARY, REPORT, GATEWAY). evaluate_live_traffic() pulls spans for each agent via SpanEvaluator.get_recent_spans(operation_name=<AgentClass>.process). But get_recent_spans only keeps spans whose attributes.output.value is a list of dicts containing document_id/video_id/source_id (search-result shape); a summary string, report string, or routing-decision dict fails that filter and the span is dropped (continue at line 144/163/177). So summary/report/gateway agents always come back empty and are never scored, never compared to baseline, and never trigger optimization. Only SEARCH ever gets evaluated.
- **fix**: get_recent_spans must not assume search-result shape for all agents. Either parametrize the output-extraction/validation per operation_name (search -> results list; summary -> summary string; report -> report string; gateway -> routing_decision dict), or have evaluate_live_traffic call a shape-agnostic span fetch that preserves summary/report/routing outputs. Add a real-boundary test feeding summa
- **status**: ☐ open

### [C5] `libs/messaging/cogniverse_messaging/auth.py:146-151 (get_tenant_id) vs auth.py:131-136 (register_user)`
- **class** E · **diagnosis** feature-gone-haywire · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: Telegram user→tenant lookup reads from the wrong Mem0 partition. register_user() stores the mapping with user_id=tenant_id (e.g. "acme:alice"), but get_tenant_id() searches with tenant_id=SYSTEM_TENANT_ID ("__system__"). Mem0.search() filters hard on user_id, so the lookup partition never contains the mapping written under the real tenant. Every already-registered user is therefore treated as unregistered: gateway._handle_message() gets tenant_id=None and replies format_registration_required() on every message. Registration is effectively non-functional end-to-end.
- **fix**: In get_tenant_id, search the same partition register_user writes to. Since the gateway doesn't know the tenant at lookup time, store the user→tenant mapping under a fixed lookup partition for BOTH write and read (e.g. user_id=SYSTEM_TENANT_ID in register_user too), or store/read via a deterministic ConfigStore key like the invite tokens already do. Add an integration test that calls register_user 
- **status**: ☐ open

### [C6] `libs/runtime/cogniverse_runtime/ingestion_worker/worker.py:372-380 (also _default_processor 256-317)`
- **class** E · **diagnosis** valid-bug · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: The worker treats ANY return from the processor as success. `pipeline.process_video_async` catches all internal exceptions and RETURNS a dict `{"status": "failed", "error": ...}` instead of raising (pipeline.py:871-923). So when ingestion actually fails (model OOM, backend 400, missing source), `_process_job` sets `success=True`, writes the `ingest:done:<sha>` idempotency record, and publishes a terminal event `state="complete"` with NO error field. The SSE stream and `/ingestion/{id}/status` report a successful ingestion; a re-submit of the same source returns the failed run id as complete and never retries.
- **fix**: In `_default_processor` (or `_process_job`), inspect the returned envelope: if `pipeline_envelope.get('status') == 'failed'` (or `results['embeddings']['documents_fed'] == 0` with non-empty `errors`), raise so the worker publishes `state='failed'` and does NOT call `mark_done`. Alternatively make `process_video_async` re-raise. Round-trip regression test: real Redis + a processor returning a faile
- **status**: ☐ open


## HIGH tier (46)

### [H1] `libs/agents/cogniverse_agents/routing/annotation_agent.py:230-232 (_analyze_span_for_annotation)`
- **class** B · **diagnosis** feature-gone-haywire · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: identify_spans_needing_annotation reads each Phoenix span attribute via span_row.get('attributes.routing') and requires isinstance(routing_attrs, dict). But Phoenix's get_spans_dataframe flattens span attributes into fully-dotted COLUMN names (e.g. column 'attributes.routing.chosen_agent', 'attributes.routing.confidence') -- there is no column named 'attributes.routing' holding a nested dict (confirmed in phoenix/client/helpers/spans: col.startswith('attributes.'), attr_name = col[len('attributes.'):]). So routing_attrs is None for every real span, the isinstance gate fails, and the method returns None for every span. The annotation agent therefore identifies ZERO spans needing annotation ag
- **fix**: Mirror RoutingEvaluator's dual-format handling: first try a nested 'attributes.routing' dict column, then fall back to reading individual flattened columns span_row.get('attributes.routing.chosen_agent'), span_row.get('attributes.routing.confidence'), etc. Add an integration test that feeds a fully-flattened-column DataFrame (matching Phoenix's real output) and asserts at least one AnnotationReque
- **status**: ☐ open

### [H2] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/image_search_agent.py:316-323`
- **class** D · **diagnosis** feature-gone-haywire · **verdict** CONFIRMED · **conf** HIGH
- **broken**: ImageSearchAgent._search_vespa sends a ColPali multi-vector embedding to Vespa in the wrong tensor format. ColPaliQueryEncoder.encode()/_encode_image() return a 2D array of shape [1024,128] (encoders.py:155 squeeze(0); _encode_image pads to [1024,128]). The agent does query_embedding.flatten().tolist() then str(...) and posts it as input.query(q) for a field declared tensor<float>(x[1024],d[128]). The working backend (search_backend.py:1000-1004) encodes a 2D multi-vector tensor as a dict {str(index): vector.tolist()}, NOT a stringified flat 1D list. Vespa rejects/mis-parses the stringified flat list, so semantic image search and find_similar_images return zero results or HTTP 400 (swallowed
- **fix**: Mirror VespaSearchBackend: for ndim==2 embeddings send a dict {str(i): vector.tolist() for i,vector in enumerate(emb)} under input.query(qt) with the rank profile expecting tensor<float>(x[1024],d[128]); do not flatten and do not str() the value. Better: route image search through the shared SearchService/VespaSearchBackend instead of hand-rolling the YQL/tensor in the agent. Add a real-Vespa (or 
- **status**: ☐ open

### [H3] `libs/agents/cogniverse_agents/graph/claim_extractor.py:312`
- **class** D · **diagnosis** valid-bug · **verdict** CONFIRMED · **conf** HIGH
- **broken**: `confidence = float(claim.get("confidence", 1.0))` coerces an LM-produced field with a bare float(). The ClaimExtractionSignature.claims output is a List[dict] produced by a DSPy module backed by a real LM; the code's own comments (lines 60-64) note small models 'copy verb tense' and emit loose output. A real LM commonly returns confidence as 'high', 'medium', '85%', or '0.9 (estimated)'. Any non-numeric value raises ValueError, which is NOT caught in _claims_to_edges, extract(), or doc_extractor._extract_from_text (lines 425-433 call self._claim_extractor.extract with no try/except), so the ENTIRE segment's KG extraction crashes — every node and edge for that chunk is lost. The codebase alr
- **fix**: Replace line 312 with `from cogniverse_agents._confidence import parse_confidence` (module-level) and `confidence = parse_confidence(claim.get("confidence"), default=1.0)`. Add a unit test that drives a ClaimExtractor with a stubbed module returning confidence='high' and '85%' and asserts edges are produced with confidence 0.9 and 0.85 respectively (would have crashed on the old code).
- **status**: ☐ open

### [H4] `libs/agents/cogniverse_agents/cross_tenant_comparison_agent.py:48 and libs/agents/cogniverse_agents/federated_query_agent.py:213`
- **class** D · **diagnosis** feature-gone-haywire · **verdict** CONFIRMED · **conf** HIGH
- **broken**: node_id is recovered from a node doc_id with `doc_id.split("_", 3)[-1]`. The doc_id format is `kg_node_{safe_tenant}_{node_id}` where safe_tenant = tenant_id.replace(':','_'). Every real tenant is `org:tenant` form, so safe_tenant ALWAYS contains an underscore (e.g. `acme:cell_a` -> `acme_cell_a`). split('_',3) stops after 3 splits, so the tenant's second segment gets glued onto the node_id: `kg_node_acme_cell_a_marie_curie` -> `cell_a_marie_curie` instead of `marie_curie`. In CrossTenantComparisonAgent.compare(), two tenants holding the SAME logical node produce DIFFERENT corrupted ids (`cell_a_marie_curie` vs `cell_b_marie_curie`), so `shared = ids_a & ids_b` is empty and every node is rep
- **fix**: Strip the known prefix instead of positional split: node_id = doc_id[len('kg_node_' + _safe_tenant(tenant_id) + '_'):] for the per-source manager's own tenant (pass the manager's tenant_id in), or have GraphManager expose a method returning node_ids directly so consumers never parse doc_ids. Regenerate both golden JSONs after the fix and add a regression test that binds managers for ASYMMETRIC org
- **status**: ☐ open

### [H5] `libs/agents/cogniverse_agents/document_agent.py:415, 479, 519-520`
- **class** PERF · **diagnosis** feature-gone-haywire · **verdict** CONFIRMED · **conf** HIGH
- **broken**: _search_visual and _search_text are `async def` but issue blocking `requests.post(..., timeout=10)` directly on the event loop thread (lines 415, 479). On the document-search hot path this stalls every other coroutine on the worker for the full Vespa round-trip. Additionally _search_hybrid's docstring/comment says it runs both searches 'in parallel' (line 518 comment 'Execute both searches in parallel') but lines 519-520 await them sequentially, so hybrid latency is visual_time + text_time, double the intended.
- **fix**: Wrap each blocking requests.post in `await asyncio.to_thread(...)` (or switch to httpx.AsyncClient), and run visual+text concurrently with `visual_results, text_results = await asyncio.gather(self._search_visual(...), self._search_text(...))`. Add a test asserting hybrid issues both backend calls and that the blocking call is off-loaded (monkeypatch requests.post to record the calling thread != ev
- **status**: ☐ open

### [H6] `libs/agents/cogniverse_agents/multi_document_synthesis_agent.py:385/388/400, libs/agents/cogniverse_agents/temporal_reasoning_agent.py:425, libs/agents/cogniverse_agents/federated_query_agent.py:328`
- **class** PERF · **diagnosis** feature-gone-haywire · **verdict** CONFIRMED · **conf** HIGH
- **broken**: These agents call synchronous LM work directly inside `async def _process_impl`/helpers without offloading. multi_document_synthesis_agent._synthesise_without_rlm calls `self._dspy_module(query=..., documents=...)` synchronously (lines 385/388) and _synthesise_with_rlm calls `rlm.process(...)` (line 400, an awaited-from helper but the call itself is the sync `RLMInference.process` defined at rlm_inference.py:331). temporal_reasoning_agent._summarise_with_rlm (line 425) and federated_query_agent._summarise_with_rlm (line 328) likewise call sync `rlm.process` on the event loop. Each blocks the loop for the full multi-second LM round trip, starving concurrent requests on the same worker (includ
- **fix**: Wrap the sync LM calls in `await asyncio.to_thread(...)`: make _synthesise_without_rlm async and `return await asyncio.to_thread(self._dspy_module, query=..., documents=...)`; replace the three bare `rlm.process(...)` calls with `await asyncio.to_thread(rlm.process, ...)`. Mirror deep_synthesis_workflow.py:224.
- **status**: ☐ open

### [H7] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/optimizer/artifact_manager.py:785-855 (load_for_request)`
- **class** PERF · **diagnosis** valid-bug · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: load_for_request performs 2-3 uncached telemetry/Phoenix dataset reads on every agent dispatch: get_artefact_state (load_blob -> get_dataset) plus a get_dataset for the canary or active versioned prompts dataset. The runtime wires this on the hot path: agent_dispatcher.dispatch -> resolve_artefact_for_request -> load_for_request fires for every request when _artifact_manager_factory is set, and routers/agents.py:_build_artifact_manager_factory returns a live factory whenever a telemetry manager is configured (the production norm). The canary state blob and versioned prompt datasets change only on promote/retire, so re-fetching them per request adds multiple network round-trips to every dispa
- **fix**: Add a short-TTL per-(tenant,agent,variant) cache for the artefact state blob and the resolved prompts (invalidate on promote_to_canary/promote_canary_to_active/retire_canary). State changes are rare and operator-driven, so a few-seconds TTL or explicit invalidation removes the per-request reads entirely.
- **status**: ☐ open

### [H8] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/summarizer_agent.py:865 and /home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/detailed_report_agent.py:918`
- **class** B · **diagnosis** feature-gone-haywire · **verdict** CONFIRMED · **conf** HIGH
- **broken**: The standalone FastAPI lifespan handlers construct the agent with no config_manager: `SummarizerAgent(deps=deps)` / `DetailedReportAgent(deps=deps)`. But `__init__` (summarizer line 221-225, detailed_report line 238-242) raises `ValueError('config_manager is required...')` when config_manager is None. So launching either agent standalone crashes at startup — uvicorn fails the lifespan and the app never serves a request. Every standalone HTTP route is dead. This is a regression: commit e2f7b5bc replaced the previous `create_default_config_manager()` fallback with a hard raise, but the lifespan callers (commit 9a65fcdb) were never updated to pass one.
- **fix**: In each lifespan, build the config_manager at the startup boundary and pass it: `from cogniverse_foundation.config.utils import create_default_config_manager; agent = SummarizerAgent(deps=deps, config_manager=create_default_config_manager())`. Add a real-boundary test that drives the actual lifespan (e.g. asgi-lifespan LifespanManager, or `with TestClient(app):` so lifespan runs) and asserts the s
- **status**: ☐ open

### [H9] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/search_agent.py:2048, /home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/summarizer_agent.py:916, /home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/detailed_report_agent.py:970`
- **class** B · **diagnosis** valid-bug · **verdict** CONFIRMED · **conf** HIGH
- **broken**: The `GET /agent.json` route on all three agents calls `<agent>.get_agent_skills()` when the singleton is initialized. No such method exists anywhere in the class hierarchy. `A2AAgent` defines `_get_skills` (no agent-prefix), and other agents define `_get_agent_skills` (leading underscore) — but `get_agent_skills` (no underscore) is undefined. So once the agent is actually initialized, `GET /agent.json` raises AttributeError -> HTTP 500. The in-code comment 'Note: _dspy_to_a2a_output and _get_agent_skills handled by A2AAgent base class' is false: neither `_dspy_to_a2a_output` nor `_get_agent_skills` exists on these classes.
- **fix**: Change the three routes to call `agent._get_skills()` (the method that actually exists on A2AAgent) or add a public `get_agent_skills()` that delegates to `_get_skills()`. Then write a test that sets the singleton to a REAL agent instance (not MagicMock) and asserts `GET /agent.json` returns 200 with a non-empty skills list whose first entry has name=='process'. The existing tests only assert skil
- **status**: ☐ open

### [H10] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/text_analysis_agent.py:316-343 (__main__ block)`
- **class** B · **diagnosis** valid-bug · **verdict** CONFIRMED · **conf** HIGH
- **broken**: The standalone `__main__` launch builds `_capacity` and reads `config`/`port`, then calls `uvicorn.run(app,...)` — but it NEVER calls `set_config_manager(...)`. The module-level `_config_manager` therefore stays None. Every `POST /analyze` request goes through `get_agent(tenant_id)` which raises `RuntimeError('ConfigManager not initialized. Call set_config_manager() during app startup.')`. The route re-raises (bare `raise`, not HTTPException), so FastAPI returns HTTP 500 on every analyze call. The standalone text-analysis service is non-functional.
- **fix**: In the __main__ block, before uvicorn.run, call `set_config_manager(create_default_config_manager())` (the same manager used to build `config`). Better: do it in a lifespan handler so it also covers programmatic launches. Add a test that constructs the app, calls set_config_manager with a real ConfigManager, posts /analyze, and asserts status 200 with analysis.result populated. Also fix the route 
- **status**: ☐ open

### [H11] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/registries/schema_registry.py:259`
- **class** C · **diagnosis** valid-bug · **verdict** CONFIRMED · **conf** HIGH
- **broken**: SchemaRegistry._validate_tenant_id uses regex ^[a-zA-Z0-9_:]+$ which rejects hyphens, but tenant creation (libs/runtime/cogniverse_runtime/admin/tenant_manager.py:162 validate_tenant_name) explicitly ALLOWS hyphens. A tenant created via POST /admin/tenants with id 'my-team' succeeds, but when that tenant later deploys a schema, deploy_schema -> _validate_tenant_id raises ValueError. A validly-registered tenant can never deploy a schema. Cross-module contract divergence.
- **fix**: Align SchemaRegistry._validate_tenant_id with the tenant-creation contract: allow hyphens (regex ^[a-zA-Z0-9_:-]+$) so any tenant accepted by validate_tenant_name can deploy a schema. Centralize on one validator (tenant_utils.validate_tenant_id) instead of two divergent copies. Then fix the test below that locks in the wrong behavior. Add a round-trip test: create tenant 'my-team' then deploy a sc
- **status**: ☐ open

### [H12] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/query/encoders.py:405-406, 413-414, 425-426`
- **class** E · **diagnosis** not-wired · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: QueryEncoderFactory._create_encoder_instance resolves the deployed inference-service URL (inference_url) for the profile, then passes it to ColPali/ColQwen/ColBERT encoders but SILENTLY DROPS it for every videoprism branch: `return VideoPrismQueryEncoder(model_name)` never forwards inference_url, and VideoPrismQueryEncoder.__init__ does not even accept an inference_service_url parameter or set config['remote_inference_url']. All four VideoPrism profiles in configs/config.json declare inference_services.embedding='videoprism_jax', and the ingestion side (embedding_generator_factory.py:109-126) DOES resolve that service into remote_inference_url for every loader. So embeddings are INGESTED via
- **fix**: Add inference_service_url param to VideoPrismQueryEncoder.__init__; when set, put it in the config dict as config['remote_inference_url'] so get_or_load_model selects RemoteVideoPrismLoader. In the factory, forward inference_url to all three videoprism branches exactly like ColPali/ColQwen. Add a unit test mirroring test_factory_wires_remote_url_from_inference_service but for a videoprism profile 
- **status**: ☐ open

### [H13] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/validation/profile_validator.py:33`
- **class** D · **diagnosis** feature-gone-haywire · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: ProfileValidator.VALID_PROFILE_TYPES = ['video','image','audio','text','document'] has drifted from the actual profile types in production config. configs/config.json defines a live profile code_lateon_mv with type='code', but 'code' is NOT in the allowlist, while 'text' IS in the allowlist yet no profile uses it. _validate_profile_type rejects any create/validate of a code-type profile through the admin API with "Invalid profile type 'code'. Must be one of: [...]". The admin Pydantic model (profile_models.py) leaves type as a free str so the request reaches the validator and is rejected there — there is no other gate that would catch/allow it earlier.
- **fix**: Add 'code' to VALID_PROFILE_TYPES; drop the unused 'text' (or keep if intentionally reserved). Update the parallel description in libs/runtime/cogniverse_runtime/admin/profile_models.py:30 which lists '(video, image, audio, text)'. Add a validator test asserting _validate_profile_type('code') returns [] and that validate_profile accepts a code profile loaded from real config.
- **status**: ☐ open

### [H14] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/agents/base.py:535-544 and 561-616`
- **class** B · **diagnosis** feature-gone-haywire · **verdict** CONFIRMED · **conf** HIGH
- **broken**: AgentBase.process() with stream=True returns _stream_with_progress(), which NEVER runs self._output_rails. The non-streaming path checks output rails at line 541-542, but the streaming path delivers the agent's final output to the client completely unchecked. ContentSafetyRail (PII/prompt-injection blocking) and OutputFormatRail are silently bypassed for every streaming request. This is exercised in production via libs/runtime/cogniverse_runtime/a2a_executor.py:218 (agent.process(typed_input, stream=True)) and the search router SSE path. A test (test_rails.py::test_run_applies_output_rails_via_process_delegation) explicitly documents fixing the SAME bypass for run(), but the streaming path w
- **fix**: In _stream_with_progress, after _process_impl returns and before yielding the final event, run `if self._output_rails: self._output_rails.check(result_holder[0].model_dump())`. On RailBlockedError, yield an error event instead of the final data. Add a streaming test asserting the output rail fires (would have caught this).
- **status**: ☐ open

### [H15] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/events/backends/memory.py:56-74 (enqueue pop) and 76-122 (subscribe offset tracking)`
- **class** D · **diagnosis** feature-gone-haywire · **verdict** CONFIRMED · **conf** HIGH
- **broken**: InMemoryEventQueue.enqueue does self._events.pop(0) when the buffer is full, but subscribe() tracks position with an integer current_offset that indexes into self._events. After a pop, all element indices shift down by one, so the absolute offset no longer maps to the correct event. A live subscriber that has consumed N events ends up with current_offset >= len(self._events) and SILENTLY SKIPS all the newest retained events — the exact events still in the buffer. The 'drop oldest' intent inverts into 'drop the newest unconsumed events from the subscriber's view'. This is the production SSE queue used by every search/ingestion/mem0 operation (libs/runtime/.../main.py:843, routers/events.py).
- **fix**: Track a monotonic absolute base offset that increments on each pop(0) (e.g. self._dropped_count), and in subscribe() index as self._events[current_offset - self._dropped_count] with current_offset clamped to >= self._dropped_count. Or switch _events to collections.deque and store a running total. Add a test: slow subscriber + buffer overflow must receive every event that was never popped, in order
- **status**: ☐ open

### [H16] `/home/amitjain/source/cogniverse/libs/dashboard/cogniverse_dashboard/tabs/orchestration_annotation.py:164`
- **class** C · **diagnosis** valid-bug · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: st.metric('Execution Time', f"{attrs.get('orchestration.execution_time', 0):.2f}s") applies the :.2f format code directly to a Phoenix span attribute. Phoenix flattened columns / OTLP string attributes frequently return values as strings; the .get(...,0) default only guards a MISSING key, not a string VALUE. A string execution_time crashes the entire tab render with ValueError before the form is even shown. This render path runs after the user clicks 'Refresh Workflows' and selects a workflow, so it is unreachable by the smoke test (which never loads orch_spans).
- **fix**: Coerce defensively before formatting: `_et = pd.to_numeric(attrs.get('orchestration.execution_time'), errors='coerce'); st.metric('Execution Time', f'{_et:.2f}s' if pd.notna(_et) else 'N/A')`. Same hardening for orchestration.tasks_completed if numeric formatting is later added. Test: render with orch_spans whose attributes.orchestration.execution_time is the string '1.23' and assert no exception 
- **status**: ☐ open

### [H17] `/home/amitjain/source/cogniverse/libs/dashboard/cogniverse_dashboard/tabs/evaluation.py:26,84,134,160,167`
- **class** B · **diagnosis** valid-bug · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: Every requests.get/post to Phoenix in this tab omits a timeout. If Phoenix is reachable at the TCP layer but slow/stalled (not connection-refused), the dashboard worker thread blocks indefinitely with no recourse — the except clauses catch errors but not hangs. get_phoenix_datasets runs on the very first render of the Evaluation tab, so an unhealthy Phoenix freezes the tab on open.
- **fix**: Add timeout= to all 5 calls (e.g. timeout=10 for queries, 30 for experiment json). Test: point _phoenix_base_url at a non-routable host:port with a small timeout and assert get_phoenix_datasets returns [] within the timeout rather than hanging.
- **status**: ☐ open

### [H18] `/home/amitjain/source/cogniverse/libs/dashboard/cogniverse_dashboard/tabs/evaluation.py:122-270`
- **class** PERF · **diagnosis** feature-gone-haywire · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: get_all_experiment_data_for_dataset loops over every experiment id and issues TWO sequential blocking HTTP GETs per experiment (the /json run data at 160 and the metadata at 167) with no batching and no timeout. Worse, the aggregate-metrics recompute (lines 252-268) is nested INSIDE the per-experiment loop, so for E experiments it recomputes mean MRR/recall over all accumulated profile/strategy queries E times — O(E * total_queries) redundant work. This is the data-load hot path that runs on every dataset selection.
- **fix**: Move the aggregate-metrics recompute (252-268) out of the per-experiment loop to after it completes. Add timeouts. Optionally fetch experiment json+metadata concurrently. Test: feed two experiments and assert aggregates are computed once and are correct (exact mean values).
- **status**: ☐ open

### [H19] `/home/amitjain/source/cogniverse/libs/dashboard/cogniverse_dashboard/app.py:2797-2798`
- **class** D · **diagnosis** valid-bug · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: The '📝 Summarize Results (Streaming)' button is rendered unconditionally on the Interactive Search tab, but its handler accesses st.session_state.current_search_results (line 2798) with no hasattr guard. current_search_results is only ever set inside the search-success branch (line 2610) and is never initialized in session-state setup. Clicking Summarize before running any search raises AttributeError and shows a Streamlit error banner.
- **fix**: Gate the Summarize button (and the Export Annotations button) on `hasattr(st.session_state, 'current_search_results') and st.session_state.current_search_results`, or guard line 2798 and st.warning('Run a search first'). Test: AppTest the search tab, click Summarize with no prior search, assert no exception and a warning is shown.
- **status**: ☐ open

### [H20] `libs/evaluation/cogniverse_evaluation/quality_monitor.py:333-410 (evaluate_golden_set stores baseline at line 409) + 576-597 (check_thresholds reads _last_golden_baseline_mrr at 588) + 719-746 (_last_golden_baseline_mrr property)`
- **class** E · **diagnosis** feature-gone-haywire · **verdict** CONFIRMED · **conf** HIGH
- **broken**: The golden-MRR drop detection can never fire. run() calls evaluate_golden_set(), which at line 409 writes the just-computed MRR into the Phoenix dataset 'quality-baseline-{tenant}'. check_thresholds() then reads _last_golden_baseline_mrr, which returns the LAST row of that same dataset — i.e. the value just written. So baseline_mrr == golden.mean_mrr and drop = (baseline-current)/baseline = 0 every cycle, never reaching golden_mrr_drop_pct. The search agent is therefore never flagged for optimization on golden degradation.
- **fix**: Capture the prior baseline BEFORE storing the new eval (read _last_golden_baseline_mrr at the top of evaluate_golden_set or pass the pre-store baseline into check_thresholds), so the comparison is current-vs-previous, not current-vs-just-written. Add a real-Phoenix round-trip test: store baseline A, run eval producing lower MRR B, assert verdict==OPTIMIZE.
- **status**: ☐ open

### [H21] `libs/evaluation/cogniverse_evaluation/evaluators/llm_judge.py:138-185 (_call_llm) + 187-227 (_extract_score_from_response), consumed at quality_monitor.py:482-499`
- **class** D · **diagnosis** valid-bug · **verdict** CONFIRMED · **conf** HIGH
- **broken**: On LM endpoint failure (timeout/connection refused/bad JSON), _call_llm returns the string 'Evaluation failed: ...' instead of raising. _extract_score_from_response finds no score pattern and returns the DEFAULT 0.5. In quality_monitor._evaluate_agent_spans this 0.5 is appended to scores and averaged into the persisted live quality score. So an LM outage silently fabricates a neutral 0.5 quality signal for every span — exactly the 'fabricated quality scores get persisted as live quality' hazard the span_evaluator code comments warn against, re-introduced here. A 'no-score' real LM reply (e.g. 'The results look reasonable overall.') likewise scores 0.5.
- **fix**: Have _call_llm raise on transport/parse failure (or return a sentinel), and _extract_score_from_response signal 'no score found' distinctly from a real 0.5; in _evaluate_agent_spans, skip/raise on failed judgements rather than averaging a fabricated 0.5. Add a test with an unreachable endpoint asserting the span is skipped (not scored 0.5).
- **status**: ☐ open

### [H22] `/home/amitjain/source/cogniverse/libs/finetuning/cogniverse_finetuning/dataset/embedding_extractor.py:137 (and 170-174)`
- **class** D · **diagnosis** feature-gone-haywire · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: TripletExtractor crashes with KeyError: 'attributes' on the real Phoenix span DataFrame. Phoenix get_spans returns FLATTENED columns (attributes.input.query, attributes.output.results, ...) with NO dict-valued 'attributes' column (provider.py docstring line 73). _filter_search_spans does spans_df['attributes'].apply(...) and _extract_from_span does span.get('attributes', {}) — both assume an un-flattened dict column that never exists. The entire embedding fine-tuning path (orchestrator._run_embedding_finetuning -> TripletExtractor.extract) is dead on arrival against a real provider. It also uses provider._trace_store / provider._annotation_store private attrs instead of the public .traces/.a
- **fix**: Rewrite _filter_search_spans/_extract_from_span/_check_modality to read the flattened attributes.* columns (build a dict via {k.removeprefix('attributes.'): v for k,v in row.items() if k.startswith('attributes.')}, same pattern as trace_converter._extract_input). Switch provider._trace_store/_annotation_store to provider.traces/provider.annotations. Add a real-format round-trip test feeding a flat
- **status**: ☐ open

### [H23] `libs/finetuning/cogniverse_finetuning/training/sft_trainer.py:196 and libs/finetuning/cogniverse_finetuning/training/dpo_trainer.py:206`
- **class** UNTESTED · **diagnosis** valid-bug · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: Production bug masked by the two hollow trainer test files above: `TrainingArguments(..., evaluation_strategy="steps" if val_dataset else "no", ...)`. transformers 4.56.2 (the installed version) removed `evaluation_strategy` (renamed to `eval_strategy`), so any real `_train_local` call raises `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'`. Local SFT/DPO fine-tuning is completely broken on the current dependency set; no test catches it because both test suites swallow the exception. Effectively zero real-execution coverage of `_train_local`.
- **fix**: Rename kwarg `evaluation_strategy` -> `eval_strategy` in both files. Add a regression test that runs `_train_local` to completion against the mocked trainer and asserts `SFTTrainer/DPOTrainer` WAS constructed (no swallowed exception) — i.e. the unconditional-assertion fix above doubles as the regression test.
- **status**: ☐ open

### [H24] `libs/runtime/cogniverse_runtime/agent_dispatcher.py:468-489 (_resolve_signature_variant) vs libs/runtime/cogniverse_runtime/routers/admin.py:1470-1491 (set_signature_variant)`
- **class** E · **diagnosis** not-wired · **verdict** CONFIRMED · **conf** HIGH
- **broken**: Admin signature-variant selection is silently dropped for every simple-form tenant. PUT /admin/tenants/{tenant_id}/signature_variants/{agent_type} stores the override under the RAW path tenant_id (e.g. 'acme'). The dispatch path resolves it via _resolve_signature_variant(tenant_id, agent_name) where tenant_id has already been canonicalized by require_tenant_id (e.g. 'acme' -> 'acme:acme'). per_tenant = _admin_overrides.get('acme:acme') misses the 'acme' key, so the dispatcher always falls back to DEFAULT_VARIANT_ID. The operator's variant pick has zero effect on dispatch for simple-form tenants (colon-form 'acme:prod' happens to match and works).
- **fix**: Canonicalize the tenant key on both sides. Either store under canonical_tenant_id(tenant_id) in set_signature_variant/get_signature_variant (and pin_quotas) so the key matches the dispatcher's canonical form, or have _resolve_signature_variant try both the canonical and the simple form. Cleaner: canonicalize at the admin route boundary the same way every other tenant-scoped route does.
- **status**: ☐ open

### [H25] `libs/runtime/cogniverse_runtime/routers/tenant.py:423-467 (_build_cron_workflow) and 954-985 (delete_job)`
- **class** D · **diagnosis** valid-bug · **verdict** CONFIRMED · **conf** HIGH
- **broken**: Scheduled-job creation fails for every colon-form tenant. _build_cron_workflow sets metadata.name = f'tenant-job-{tenant_id}-{job_id}' with the raw tenant_id. For a registered tenant like 'acme:production' this yields 'tenant-job-acme:production-<id>', which is not a valid k8s/Argo resource name (colons and uppercase are illegal in RFC-1123 names). Argo rejects the CronWorkflow with a 4xx, and _submit_cron_workflow turns that into HTTPException(503), so POST /admin/tenant/{t}/jobs returns 503 and no job is persisted. The _sanitize_label_value helper exists but is applied only to the optimization-workflow LABELS, never to the CronWorkflow NAME. delete_job rebuilds the same invalid name, so it
- **fix**: Sanitize the tenant_id segment of the CronWorkflow name with the same RFC-1123 transform _sanitize_label_value uses (lowercase, replace illegal chars with '-', trim) and reuse the sanitized form in both _build_cron_workflow and delete_job so create/delete agree on the resource name. Keep the raw tenant_id only in the --tenant-id CLI arg / labels.
- **status**: ☐ open

### [H26] `libs/runtime/cogniverse_runtime/a2a_executor.py:104-129 (execute) and 168-237 (_execute_streaming) vs create_streaming_agent in agent_dispatcher.py:749-917`
- **class** E · **diagnosis** feature-gone-haywire · **verdict** CONFIRMED · **conf** HIGH
- **broken**: A2A streaming requests silently drop multi-turn conversation history and per-agent memory/graph context. execute() builds task_context containing conversation_history (via _extract_conversation_history) and the canary request_id, but the use_streaming branch calls _execute_streaming(agent_name, query, tenant_id, ...) without ever passing task_context. _execute_streaming builds the agent via create_streaming_agent, which constructs e.g. SearchInput(query=...) with no conversation_history/enrichment and never calls _init_agent_memory or _bind_graph_manager. So a client using /a2a message/stream gets unresolved anaphora (no query rewrite over history) and no mem0/KG enrichment, while the identi
- **fix**: Thread task_context into _execute_streaming and have create_streaming_agent accept conversation_history/enrichment and run the same _rewrite_query_with_history + _init_agent_memory + _bind_graph_manager steps the non-streaming path uses, or route streaming through dispatch() with a streaming sink so the two paths share one code path.
- **status**: ☐ open

### [H27] `libs/runtime/cogniverse_runtime/agent_dispatcher.py:1009-1013, 1058-1085 (_get_search_agent + overlay) and 1147-1199 (_execute_gateway_task shared _gateway_agent), consumed in libs/core/cogniverse_core/agents/base.py:129-189,400-454`
- **class** D · **diagnosis** valid-bug · **verdict** CONFIRMED · **conf** MEDIUM
- **broken**: Per-request canary/variant artefact overlay is applied to dispatcher-cached, cross-request-shared agent instances, causing request bleed under concurrency. _get_search_agent caches one SearchAgent per profile and _gateway_agent is a singleton; both are reused by every concurrent dispatch. _apply_artefact_overlay sets self._dispatched_artefact on that shared instance, and call_dspy() then enters _DispatchedPromptOverlayContext which MUTATES module.predictor.signature.instructions in place (then restores on exit) while module.forward runs in a worker thread. Two overlapping requests on the same cached agent race: request B's set_dispatched_artefact / instruction-swap / restore can clobber requ
- **fix**: Do not apply per-request mutable overlay to a shared cached instance. Either construct a fresh agent per request when an overlay is in scope, pass the overlay as a call-local argument into call_dspy instead of stashing on self, or guard the instruction-swap with a per-instance lock so concurrent dispatches serialize. Also set _gateway_agent._artifact_tenant_id per request, not once at construction
- **status**: ☐ open

### [H28] `libs/runtime/cogniverse_runtime/ingestion/processors/embedding_generator/embedding_generator_impl.py:227,260`
- **class** E · **diagnosis** feature-gone-haywire · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: VLM frame descriptions are generated (expensive GPU work) then dropped — never attached to any Document. `descriptions = video_data.get('descriptions', {})` is the VLM WRAPPER dict `{'video_id', 'descriptions': {<frame_number>: text}, 'total_descriptions', 'created_at'}` (vlm_descriptor.py:174-179), but the code does `descriptions.get(str(idx))` where `idx` is the segment enumeration index. Two mismatches: (1) wrong nesting level — per-frame text lives under `wrapper['descriptions']`, not the wrapper top level; (2) wrong key — wrapper sub-dict is keyed by source `frame_number` (e.g. 0,30,90), not the segment index. Result: every keyframe document gets `description=''`.
- **fix**: Unwrap one level: `descriptions = video_data.get('descriptions', {}).get('descriptions', {})` if the wrapper shape is passed, then look up by the segment's actual frame key — pass `segment.get('frame_number')`/`segment.get('frame_id')` into `_create_segment_document` and use `descriptions.get(str(frame_ref), '')`. Add a round-trip test: keyframes with sparse frame_numbers + VLM descriptions → each
- **status**: ☐ open

### [H29] `libs/runtime/cogniverse_runtime/ingestion/processors/embedding_generator/embedding_generator_impl.py:267,1075-1080 + libs/vespa/cogniverse_vespa/backend.py:311-340`
- **class** PERF · **diagnosis** feature-gone-haywire · **verdict** NEEDS_LIVE_BOUNDARY · **conf** MEDIUM
- **broken**: `_process_multi_documents` feeds documents to the backend ONE PER SEGMENT inside the loop (`self._feed_document(doc)` → `ingest_documents([document], schema)`). For a video with N keyframes this is N separate Vespa feeds, and with `wait_for_indexing` (default True, backend.py:317) each feed triggers a per-document Document-v1 visibility probe loop (up to `indexing_timeout`=30s each). `ingest_documents`/`feed_iterable` is built to batch many docs in one call; calling it per-segment defeats that and adds N synchronous index-wait round-trips on the ingestion hot path.
- **fix**: Accumulate Documents in a list across the segment loop and feed them in batches (e.g. every 50-100, plus a final flush), calling `backend_client.ingest_documents(batch, schema_name)` once per batch. Keep the per-frame `del`/gc memory hygiene. Add an assertion-based test counting backend feed calls << segment count.
- **status**: ☐ open

### [H30] `libs/runtime/cogniverse_runtime/job_executor.py:248-296 (run_job), 71-116 (_detect_deliveries/_is_pure_delivery), 214-245 (_execute_action)`
- **class** B · **diagnosis** not-wired · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: The entire job-executor logic — query dispatch, semantic delivery-destination detection, pure-delivery shortcut, wiki/Telegram delivery routing — has zero test coverage. The only test reference asserts the Argo container command string contains `cogniverse_runtime.job_executor` (test_argo_wiring_roundtrip.py:116); no test imports or exercises `run_job`, `_execute_action`, `_detect_deliveries`, `_is_pure_delivery`, or `_cosine_sim`. This is the Argo CronWorkflow entrypoint that runs scheduled tenant jobs in production.
- **fix**: Add a real-boundary integration test driving `run_job` against an ASGITransport stub of `/agents/.../process`, `/wiki/save`, `/messaging/send` and a real/stub DenseOn embedding endpoint, asserting: pure-delivery action skips the agent call and delivers the prior result verbatim; processing action routes through orchestrator then delivers; ambiguous action below threshold delivers nowhere. Cover `_
- **status**: ☐ open

### [H31] `libs/runtime/cogniverse_runtime/agent_dispatcher.py:1291 (created) + libs/agents/cogniverse_agents/orchestrator_agent.py:599 (stored, never closed)`
- **class** PERF · **diagnosis** not-wired · **verdict** CONFIRMED · **conf** HIGH
- **broken**: Per-orchestration-request resource leak: `_execute_orchestration_task` calls `self._sandbox_manager.make_http_client("orchestrator_agent")`, which always builds a brand-new `httpx.AsyncClient` (with its own connection pool + PolicyEnforcingTransport). It is passed to a per-request OrchestratorAgent as `_http_client_override` and is NEVER `aclose()`d — there is no finally block, and the orchestrator only manages the shared `_http_clients` pool, not the override. Every orchestration request (when a sandbox manager is present, i.e. the production unified-runtime path) leaks one AsyncClient + its open sockets/file descriptors. Under sustained traffic this exhausts FDs and connection pools.
- **fix**: Wrap usage in `async with` or add a `finally: if orch_http_client is not None: await orch_http_client.aclose()` in `_execute_orchestration_task` after the orchestrator finishes; alternatively cache one policy-enforcing client per (agent_type) on the SandboxManager and reuse it instead of building per call.
- **status**: ☐ open

### [H32] `/home/amitjain/source/cogniverse/libs/synthetic/cogniverse_synthetic/backend_querier.py:121-138`
- **class** D · **diagnosis** feature-gone-haywire · **verdict** CONFIRMED · **conf** HIGH
- **broken**: _query_profile builds query_params={'yql': yql, 'hits': sample_size, ...} then calls self.backend.query_metadata_documents(schema=schema_name, yql=yql, hits=sample_size, **query_params). Since query_params ALSO contains yql and hits, this raises TypeError: got multiple values for keyword argument 'yql' on every real backend call. The broad except Exception at line 143 swallows it and returns [], so query_profiles returns ZERO sampled content whenever a real backend is supplied. Synthetic generation then silently falls back to hardcoded mock patterns — synthetic data is NEVER grounded in real Vespa content (the default 'diverse' strategy hits this path).
- **fix**: Stop double-passing: call self.backend.query_metadata_documents(schema=schema_name, yql=yql, hits=sample_size, ranking=..., **strategy_only_params) where strategy params do NOT include yql/hits. Build query_params as the extra ranking kwargs only. Also narrow the except so a programming TypeError is not silently swallowed. Add a real-Backend round-trip test asserting non-empty samples and that yql
- **status**: ☐ open

### [H33] `/home/amitjain/source/cogniverse/libs/synthetic/cogniverse_synthetic/approval/confidence_extractor.py:80-88`
- **class** E · **diagnosis** not-wired · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: SyntheticDataConfidenceExtractor.extract reads data.get('_generation_metadata') at the TOP level, but RoutingGenerator stores it nested under the schema's metadata field (routing.py:156 metadata=generation_metadata -> model_dump() => {'metadata': {'_generation_metadata': {'retry_count': N}}}). So retry_count is always read as 0 — the documented 'most important' confidence signal is silently dropped. A 3-retry template-fallback query (worst quality) scores 0.7 instead of 0.39, so with the default 0.7 auto-approve threshold it gets AUTO-APPROVED instead of routed to human review. This is the live path used by the optimization dashboard tab (optimization.py:2054 extract(item_data) over result['
- **fix**: Make extract read metadata = data.get('metadata',{}).get('_generation_metadata') or data.get('_generation_metadata',{}) (support both shapes), OR have RoutingGenerator place _generation_metadata at the top level of the dumped dict. Fix the dashboard read sites in lockstep. Add a test that a fallback (retry_count==max_retries) example yields confidence below the default 0.7 threshold.
- **status**: ☐ open

### [H34] `libs/telemetry-phoenix/cogniverse_telemetry_phoenix/evaluation/monitoring.py:164`
- **class** D · **diagnosis** feature-gone-haywire · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: RetrievalMonitor.log_retrieval_event calls px.log_trace(...), but the installed Phoenix `phoenix` module has no `log_trace` attribute. The call raises AttributeError on every event, swallowed by the surrounding try/except (line 175), so every retrieval/experiment event silently fails to log to Phoenix. Reached on the real path via evaluation_provider.log_experiment_event (evaluation_provider.py:300-301), which instantiates RetrievalMonitor() and calls log_retrieval_event for every experiment event. The metrics_buffer also accumulates per event but is only drained by _monitoring_loop (started by start(), which this path never calls) — unbounded in-memory growth.
- **fix**: Replace px.log_trace with the supported span-export path (configure_span_export / OTLP tracer) used by PhoenixProvider, or emit via the OpenTelemetry tracer the TelemetryManager owns. Also bound metrics_buffer (deque(maxlen=...)) or drain it without requiring start(). Add a test that drives log_experiment_event through the real provider and asserts a span/annotation actually lands in a Phoenix Doc
- **status**: ☐ open

### [H35] `/home/amitjain/source/cogniverse/tests/agents/unit/test_standalone_agent_apps.py:76-103, 220-247, 250-284, 287-324 (and the whole file)`
- **class** A · **diagnosis** valid-bug · **verdict** CONFIRMED · **conf** HIGH
- **broken**: This test file claims 'HTTP-level coverage for the standalone agent FastAPI apps' but locks in the broken contract. Every happy-path test monkeypatches the singleton to `MagicMock()`/`AsyncMock()`. Consequences: (1) `/agent.json` happy path is never tested with a real agent, so `agent.get_agent_skills()` returns a MagicMock instead of raising AttributeError — the HIGH /agent.json bug is masked. (2) The real `lifespan()` is never run (every test monkeypatches the singleton directly), so the CRIT lifespan-crash bug is invisible. (3) `/process`, `/summarize`, `/generate_report` happy paths assert the code's own envelope shape against a mock return value — they prove the route copies fields, not
- **fix**: Add real-boundary tests: drive each app with the actual lifespan running (asgi-lifespan LifespanManager, or TestClient(app) as a context manager) so the real startup path executes and the config_manager-None crash surfaces. Add an /agent.json test with a real (non-mock) agent instance asserting skills is a non-empty list. Keep the 503/400 guard tests, but the happy-path envelope tests should const
- **status**: ☐ open

### [H36] `tests/runtime/integration/test_ingestion_worker_e2e.py:140 (test_failing_processor_publishes_failed_and_acks)`
- **class** A · **diagnosis** valid-bug · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: The only worker-failure test uses a processor that RAISES (`raise RuntimeError(...)`), which locks in a contract the production processor never satisfies. `_default_processor` → `process_video_async` never raises on pipeline failure (it returns a failed-status dict). So the test is green while the real failure path (CRIT finding above) publishes `state='complete'` for failed ingests. The test asserts the boundary behavior for an input shape the real code cannot produce.
- **fix**: Add a second test whose processor RETURNS `{'status':'failed','error':...,'results':{}}` (mirroring `process_video_async`'s real failure envelope) and assert the worker publishes `state='failed'` and does NOT write `ingest:done:<sha>`. Pair it with the CRIT fix so it fails on current code.
- **status**: ☐ open

### [H37] `tests/agents/unit/test_orchestrator_agent.py:951-963 (TestOrchestratorIntelligence.test_orchestration_span_emitted)`
- **class** HOLLOW-TEST · **diagnosis** valid-bug · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: The `orchestrator_agent` fixture never sets `telemetry_manager`, so `_emit_orchestration_span` short-circuits at the first line of prod (`if not (hasattr(self,'telemetry_manager') and self.telemetry_manager): return`, orchestrator_agent.py:2616). The test's only assertion is the comment `# No exception means success` — it asserts NOTHING. The actual span body (lines 2624-2641: opens a `telemetry_manager.span(name='cogniverse.orchestration', ...)` with attributes orchestration.workflow_id/query[:200]/agent_sequence/execution_time/success/tasks_completed) and the missing-tenant guard (line 2620 `raise RuntimeError('...called before _process_impl set self._current_tenant_id')`) are NEVER execut
- **fix**: Set a real (or in-memory recording) TelemetryManager on the agent and assert the emitted span's exact attributes. Strong assertion: after setting agent._current_tenant_id='acme:prod' and a recording telemetry_manager, call _emit_orchestration_span(workflow_id='wf_test', query='q'*300, agent_sequence=['search_agent','summarizer_agent'], execution_time=1.5, success=True, tasks_completed=2); then ass
- **status**: ☐ open

### [H38] `tests/agents/unit/test_query_enhancement.py:1299-1358 (test_relevance_score_calculation)`
- **class** HOLLOW-TEST · **diagnosis** valid-bug · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: patch.object on the SUT: builds a real SearchAgent, then does `agent._calculate_relationship_relevance = Mock(return_value=0.85)` and asserts `relevance > 0.0 and relevance <= 1.0`. It asserts the mock's own injected return value (0.85), so the real scoring function (entities +0.3 each, relationships +2-match → +0.5 / 1-match → +0.2, capped at 1.0) never executes. The test passes no matter what the production method computes.
- **fix**: Delete the `agent._calculate_relationship_relevance = Mock(...)` line and call the REAL method. With result={'title':'Robots playing soccer in championship','description':'Advanced robots demonstrate soccer skills'}, entities=[{'text':'robots'},{'text':'soccer'}] and relationships=[{'subject':'robots','relation':'playing','object':'soccer'}], assert the exact score: 0.3(robots)+0.3(soccer) + 0.5(s
- **status**: ☐ open

### [H39] `tests/agents/unit/test_query_enhancement.py:1362-1420 (test_entity_matching_logic)`
- **class** HOLLOW-TEST · **diagnosis** valid-bug · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: patch.object on the SUT: `agent._find_matching_entities = Mock(return_value=[robots, soccer])` then asserts the returned list has those two. Asserts the mock's own return value; the real method never runs. Additionally the test calls it with a raw string `result_text` while the real signature is `_find_matching_entities(self, result: Dict, entities)` (search_agent.py:1700) — the test's call contract doesn't even match prod, so it could not pass against the real implementation.
- **fix**: Drop the Mock; pass a real result dict, e.g. {'title':'autonomous robots learning to play soccer'}, with entities robots/soccer/basketball. Assert `[e['text'] for e in agent._find_matching_entities(result, entities)] == ['robots','soccer']` (basketball absent because it's not in the text).
- **status**: ☐ open

### [H40] `tests/agents/unit/test_query_enhancement.py:1424-1494 (test_search_result_enhancement)`
- **class** HOLLOW-TEST · **diagnosis** valid-bug · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: patch.object on the SUT for a method that does NOT exist in production: `agent._enhance_results_with_context = Mock(return_value=enhanced_results)`. grep of search_agent.py shows `_enhance_results_with_context` is referenced only in this test. The test then asserts `enhanced_results[0]['enhanced_score'] > enhanced_results[0]['score']` — comparing fields of the hand-written `enhanced_results` literal the test itself supplied to the Mock. Zero production code executes; the 'feature' is fabricated by the mock.
- **fix**: Either the enhancement feature exists under a different name (find it and test the real method with strong assertions on computed enhanced_score/entity_matches) or it does not exist — in which case delete this test rather than assert a fabricated API. If real, build SearchAgent, feed original_results + entities/relationships, assert the exact enhanced_score recomputed by the real scorer.
- **status**: ☐ open

### [H41] `tests/agents/unit/test_detailed_report_agent.py:307-328 (test_generate_report_empty_results)`
- **class** HOLLOW-TEST · **diagnosis** total-slop · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: Mocks get_config and VLMInterface, builds a ReportRequest, then NEVER calls `_generate_report` (or any agent method). The only assertions are `request.query == 'test query'` and `len(request.search_results) == 0` — i.e. it asserts the pydantic request object it just constructed. The docstring even hedges: 'In a real scenario, generate_report would return appropriate empty structures.' Proves nothing about the agent.
- **fix**: Build the agent (mock _initialize_vlm_client + set _dspy_lm + stub call_dspy), call `await agent._generate_report(ReportRequest(query='q', search_results=[], ...))`, and assert the exact empty-state shape: result.thinking_phase.content_analysis['total_results']==0, content_types=={}, executive_summary equals the stubbed call_dspy output. (test_process_a2a_task_no_messages at line 277 already does 
- **status**: ☐ open

### [H42] `tests/runtime/unit/test_optimization_cli_batch_modes.py:121-189 (TestCliArgumentParser)`
- **class** HOLLOW-TEST · **diagnosis** valid-bug · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: The file docstring claims it tests that 'the CLI argument parser recognizes all new modes', but the `parser` fixture (lines 124-147) HAND-BUILDS a parallel argparse parser instead of importing the real CLI's parser. The real parser lives inline in `optimization_cli.main()` (libs/runtime/cogniverse_runtime/optimization_cli.py:2243-2263). The two have diverged: the test's choices are `[cleanup, triggered, simba, workflow, gateway-thresholds, profile, entity-extraction, routing, synthetic]` and `--tenant-id` defaults to `'default'`; the REAL CLI's choices are `[cleanup, triggered, simba, workflow, gateway-thresholds, online-routing-eval, profile, entity-extraction, synthetic, rollback, ab-compa
- **fix**: Refactor the real CLI to expose a `build_parser()` (or `_build_parser()`) function that `main()` calls, then import THAT in the test fixture. Assert: (a) the real parser accepts each of the 13 actual modes; (b) `--mode routing` raises SystemExit (it is NOT a mode); (c) `--mode online-routing-eval` is accepted; (d) `parser.parse_args(['--mode','cleanup']).tenant_id is None` (real default). The test
- **status**: ☐ open

### [H43] `tests/finetuning/test_sft_trainer.py:56-105 (test_no_validation_split_for_small_dataset), :107-171 (test_validation_split_for_large_dataset), :179-235 (test_lora_success_path), :237-297 (test_lora_fallback_on_error), :300-349 (test_lora_disabled_via_config)`
- **class** HOLLOW-TEST · **diagnosis** valid-bug · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: Each test wraps `await finetuner._train_local(...)` in `try: ... except Exception: pass` ("Some mocking may be incomplete, but we can still verify the calls" — a docstring hedge) and then makes its key wiring assertions conditional: `if mock_sft_trainer.called:` before checking `trainer_call_kwargs.get('eval_dataset') is None/is not None` and `trainer_call_kwargs.get('model') == mock_model`. I executed the path with the test's exact mocks: `_train_local` raises `TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'` at sft_trainer.py:196, so `mock_sft_trainer.called == False`. Every `eval_dataset`/`model`-wiring assertion is therefore VACUOUSLY SKIP
- **fix**: Remove the `try/except Exception: pass` and the `if mock_sft_trainer.called:` guards so the trainer-arg assertions are UNCONDITIONAL (the test must FAIL when SFTTrainer is never reached). Then the masked prod bug surfaces and must be fixed: change `evaluation_strategy=` to `eval_strategy=` at sft_trainer.py:196. Strongest assertions: after _train_local runs to completion, `mock_sft_trainer.call_ar
- **status**: ☐ open

### [H44] `tests/finetuning/test_dpo_trainer.py:56-113 (test_no_validation_split_for_small_dataset), :116-188 (test_validation_split_for_large_dataset), :196-260 (test_lora_success_path), :262-331 (test_lora_fallback_on_error), :334-391 (test_lora_disabled_via_config)`
- **class** HOLLOW-TEST · **diagnosis** valid-bug · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: Identical hollow pattern to the SFT trainer tests: `try: await finetuner._train_local(...) except Exception: pass` plus `if mock_dpo_trainer.called:` guarding every eval_dataset/model/ref_model assertion. dpo_trainer.py:206 also passes `evaluation_strategy=` to `TrainingArguments`, which raises TypeError on transformers 4.56.2 before DPOTrainer is constructed, so `mock_dpo_trainer.called == False` and all the trainer-arg assertions (`eval_dataset is None/is not None`, `model == mock_model`, `ref_model == mock_model_ref`) are silently skipped. The except swallows the production crash.
- **fix**: Drop the except-pass and the `if mock_dpo_trainer.called:` guards (make assertions unconditional so the test fails when DPOTrainer is unreached), then fix prod: `evaluation_strategy` -> `eval_strategy` at dpo_trainer.py:206. Assert exact: `mock_dpo_trainer.call_args.kwargs['ref_model'] is mock_model_ref`, `['model'] is mock_peft_model` (LoRA on) / `is mock_model` (fallback), `['eval_dataset'] is m
- **status**: ☐ open

### [H45] `libs/vespa/cogniverse_vespa/search_backend.py:89,103,128-135`
- **class** PERF · **diagnosis** feature-gone-haywire · **verdict** CONFIRMED · **conf** HIGH
- **broken**: SearchMetrics.search_latencies is a plain list appended once per search (record_search line 103) and NEVER trimmed/bounded. VespaSearchBackend instances are cached for the process lifetime in BackendRegistry._backend_instances (TenantLRUCache, libs/core/.../backend_registry.py:54), so on a long-lived service this list grows by one float per query forever — unbounded memory leak on the search hot path. Compounding it, p95_latency_ms (line 133) does `sorted(self.search_latencies)` on the entire list on every get_metrics()/health_check() call — O(n log n) over an ever-growing list.
- **fix**: Make search_latencies a collections.deque(maxlen=N) (e.g. 1000) sized for the percentile window; this bounds memory and the p95 sort. Track a running sum/count separately for the lifetime average. Add a test that records >maxlen searches and asserts len(search_latencies)==maxlen.
- **status**: ☐ open

### [H46] `/home/amitjain/source/cogniverse/libs/vespa/cogniverse_vespa/search_backend.py:1251`
- **class** C · **diagnosis** valid-bug · **verdict** NEEDS_LIVE_BOUNDARY · **conf** HIGH
- **broken**: In export_documents_with_embeddings, string filter values are interpolated RAW into YQL with single quotes and no escaping: conditions.append(f"{key} contains '{value}'"). Sibling code in the same file (line 878 and _yql_scalar/yql_quote helpers, search_backend.py:60) escapes every other interpolated string via yql_quote. This is the convention-divergence YQL-injection / malformed-query footgun the hunt list targets. The negative-lookahead regex in audit.md missed it because it anchors on the double-quote form `contains "{...}"`; this site uses single quotes. Line 1253 likewise interpolates a numeric value and the field key raw (f"{key} = {value}").
- **fix**: Replace the raw f-string with the existing escaper: conditions.append(f"{key} contains {yql_quote(value)}") for the str branch, and conditions.append(f"{key} = {_yql_scalar(value, key)}") for the numeric branch (both already imported/defined in this module). Add a real-Vespa-Docker round-trip test that exports with a filter value containing a single quote and asserts the documents come back (not a
- **status**: ☐ open


## MED tier (83) — compact


**agents** (22)
- [D] `libs/agents/cogniverse_agents/routing/dspy_relationship_router.py:227-236 (_pars` — ComposableQueryAnalysisModule._parse_confidence mishandles the common LM confidence output shapes that the codebase already has a 
- [D] `libs/agents/cogniverse_agents/orchestrator_agent.py:2325-2336 (_aggregate_result` — _aggregate_results extracts confidence = result_data.get('confidence', 0.5) raw from each sub-agent result dict with no numeric co
- [C] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/image_search_agen` — ImageSearchAgent._search_vespa interpolates filter values raw into the YQL where-clause: contains(detected_objects, '{filters['det
- [D] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/audio_analysis_ag` — AudioAnalysisAgent._search_acoustic generates a 768-dim sentence-transformer SEMANTIC embedding (generate_semantic_embedding, line
- [PERF] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/search_agent.py:7` — SearchAgent._search_ensemble claims to 'Pre-compute query embeddings for each profile in parallel' (comments lines 799, 840), but 
- [C] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/search_agent.py:1` — In _process_impl text path, `if dspy_result.confidence > 0.7:` compares a DSPy OutputField value to a float. DSPy ChainOfThought o
- [PERF] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/image_search_agen` — All Vespa/ASR HTTP calls in the image and audio agents use synchronous requests.post inside async def handlers (_search_vespa, _se
- [PERF] `libs/agents/cogniverse_agents/citation_tracing_agent.py:177-199` — trace(claim_id) calls graph_manager._visit(doc_type='edge', top_k=2000) — pulls up to 2000 edge documents via the Document v1 visi
- [PERF] `libs/agents/cogniverse_agents/wiki/wiki_manager.py:119 (save_session → _rebuild_` — save_session (invoked from agent_dispatcher._maybe_auto_file_wiki on qualifying agent turns) does, per call: one _get_or_create_to
- [PERF] `libs/agents/cogniverse_agents/graph/graph_manager.py:201-218 (get_path) and 176-` — get_path runs a BFS where each frontier node triggers a separate _visit_edges HTTP GET to Vespa (line 204). For a path query of de
- [PERF] `libs/agents/cogniverse_agents/graph/face_extractor.py:108-121 (extract_faces_per` — Each keyframe is POSTed to the face-embed sidecar one at a time in a serial for-loop over a synchronous httpx.Client (_post_one at
- [PERF] `libs/agents/cogniverse_agents/audit_explanation_agent.py:213 and 253` — _process_impl fetches every source memory twice on the happy path. Line 213 calls self._fetch_memory(mm, tenant_id, node.memory_id
- [D] `libs/agents/cogniverse_agents/deep_research_agent.py:276-277` — _evaluate_evidence builds the evidence_summary fed to the LLM with `len(e.get('results', []))` to report a result count per sub-qu
- [D] `libs/agents/cogniverse_agents/deep_research_agent.py:265-266` — _search_parallel dispatches all sub-question searches with `asyncio.gather(*tasks, return_exceptions=False)`. If a single sub-sear
- [E] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/workflow/intellig` — _optimize_for_performance does `task.metadata = task.metadata or {}` (line 304), but WorkflowTask (workflow_types.py) has no `meta
- [D] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/workflow_types.py` — get_ready_tasks references TaskStatus.PENDING, which does not exist on the TaskStatus enum (members: WAITING, READY, RUNNING, COMP
- [C] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/inference/instrum` — Fire-and-forget asyncio task with a dropped reference: `loop.create_task(self._event_queue.enqueue(event))` does not retain the re
- [A] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/approval/interfac` — Contract drift between the ApprovalStorage ABC and its callers. The ABC declares `async def update_item(self, item: ReviewItem) ->
- [PERF] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/approval/approval` — Two compounding issues in the telemetry-backed approval reads. (1) Blocking time.sleep() inside async methods: get_batch (line 285
- [UNTESTED] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/optimizer/dspy_ag` — DSPyAgentOptimizerPipeline (load_training_data + _create_metric_for_module + optimize_all_modules + __main__) is untested surface 
- [B] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/summarizer_agent.` — The DSPy `SummarizationModule` is constructed and the agent advertises LLM summarization, but the LLM is only invoked on the `brie
- [PERF] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/search_agent.py:2` — All four standalone search routes are declared `async def` but call synchronous, CPU/IO-blocking agent methods directly on the eve

**core** (9)
- [E] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/registries/schema_reg` — _load_schemas_from_storage rebuilds self._schemas additively: it iterates storage entries and assigns self._schemas[key] = SchemaI
- [B] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/registries/backend_re` — The shared search backend (cache key 'search_{name}') and per-tenant ingestion/full backends (keys 'ingestion_{name}_{tenant}', 'b
- [C] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/query/encoders.py:194` — VideoPrism global-vs-patch routing and embedding dim are decided by raw substring matching: is_global = 'lvt' in model_name.lower(
- [E] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/query/encoders.py:62-` — ColBERTQueryEncoder.encode(self, query, trace='') implements the 'joint-trace AgentIR' feature (embedding `f'{query} {trace}'` to 
- [D] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/query/encoders.py:233` — For VideoPrism patch (mv) profiles (videoprism_public_v1_base_hf / _large_hf, type video, schema tensor(patch{}, v[768\|1024])), t
- [C] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/common/vlm_interface.` — float(result.relevance_score) coerces a raw DSPy LM output field with no try/except. The signature field is described as 'Relevanc
- [PERF] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/memory/provenance.py:` — ProvenanceStore.walk() already fetches every node's ProvenanceRecord while doing the BFS (one Vespa query per level) but returns o
- [PERF] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/memory/manager.py:637` — On every knowledge write that has a knowledge_registry wired AND metadata.subject_key set, add_memory triggers _detect_and_persist
- [E] `libs/core/cogniverse_core/common/cache/backends/structured_filesystem.py:63` — StructuredFilesystemBackend sets self._needs_cleanup = True (line 63) when config.cleanup_on_startup and config.enable_ttl, with t

**dashboard** (5)
- [E] `/home/amitjain/source/cogniverse/libs/dashboard/cogniverse_dashboard/tabs/rlm_ab` — render_rlm_ab_compare_tab reads the tenant from st.session_state.get('tenant_id'), but the dashboard app shell never sets a 'tenan
- [PERF] `/home/amitjain/source/cogniverse/libs/dashboard/cogniverse_dashboard/tabs/routin` — render_routing_evaluation_tab pulls the same Phoenix span window up to 4 times per render: a connectivity probe (line 80), query_r
- [C] `/home/amitjain/source/cogniverse/libs/dashboard/cogniverse_dashboard/tabs/optimi` — spans_df['name'].str.contains('search', case=False) at lines 232 and 518 omits na=False. Phoenix span DataFrames can contain rows 
- [SLOP] `/home/amitjain/source/cogniverse/libs/dashboard/cogniverse_dashboard/app.py:2907` — The Interactive Search tab's 'Search Analytics' section renders four st.metric widgets with hardcoded fabricated values and fake d
- [E] `/home/amitjain/source/cogniverse/libs/dashboard/cogniverse_dashboard/tabs/optimi` — The Optimization Overview tab's 'Optimization Runs' count, 'Last Optimization' age, and 'Recent Optimization History' table all re

**evaluation** (6)
- [E] `libs/evaluation/cogniverse_evaluation/quality_monitor.py:781-810 (_store_golden_` — Schema collision on the single Phoenix dataset 'quality-baseline-{tenant}'. _store_golden_eval_result writes columns {mean_mrr, me
- [E] `libs/evaluation/cogniverse_evaluation/evaluators/metadata_fetcher.py:79-82 (_fet` — Reference-based LLM judge never gets real video metadata. _fetch_from_vespa constructs SearchService(self.config, profile='video_c
- [E] `libs/evaluation/cogniverse_evaluation/data/traces.py:122-146 (get_traces_by_expe` — get_traces_by_experiment builds filter_condition="attributes.metadata.profile == '{profile}' AND ... strategy == '{strategy}'" and
- [B] `libs/evaluation/cogniverse_evaluation/plugins/visual_evaluator.py:192-211 (get_v` — The visual-judge and quality-evaluator scorers are never wired into the production experiment scorer set. task.py builds scorers o
- [E] `libs/evaluation/cogniverse_evaluation/core/experiment_tracker.py:96-104 (_regist` — When enable_quality_evaluators is set, _register_evaluator_plugins imports VideoAnalyzerPlugin from cogniverse_evaluation.plugins.
- [C] `libs/evaluation/cogniverse_evaluation/online_evaluator.py:173 (_eval_confidence_` — float(routing_attrs.get('confidence', 0.5)) is called on a routing-span attribute that originates from an LM/router output. A real

**finetuning** (4)
- [B] `/home/amitjain/source/cogniverse/libs/finetuning/cogniverse_finetuning/orchestra` — _log_experiment_to_phoenix generates its OWN run_id (line 373) and writes it into the Phoenix EXPERIMENT span attribute experiment
- [E] `/home/amitjain/source/cogniverse/libs/finetuning/cogniverse_finetuning/orchestra` — OrchestrationConfig.hf_token is set (line 180) but never read. _upload_adapter_to_storage calls upload_adapter(result.adapter_path
- [SLOP] `/home/amitjain/source/cogniverse/libs/finetuning/cogniverse_finetuning/evaluatio` — Fabricated statistics presented as real metrics, then logged to Phoenix as improvement.p_value / improvement.significant. improvem
- [C] `/home/amitjain/source/cogniverse/libs/finetuning/cogniverse_finetuning/evaluatio` — confidence = pred_json.get('confidence', 0.5) takes a model-generated JSON field and immediately does total_confidence += confiden

**foundation** (1)
- [E] `libs/foundation/cogniverse_foundation/telemetry/manager.py:450-458 (get_provider` — get_provider computes project-specific http_endpoint/grpc_endpoint overrides and puts them in provider_config, but the tenant-scop

**messaging** (1)
- [C] `libs/messaging/cogniverse_messaging/telegram_handler.py:57 (and the total>MAX co` — _format_results does float(score) with no try/except. score comes straight from the agent-dispatch JSON payload (response['results

**runtime** (12)
- [PERF] `libs/runtime/cogniverse_runtime/routers/health.py:27-35 (health_check)` — GET /health rebuilds the entire config stack on every probe: create_default_config_manager() (re-reads/parses config.json and reco
- [PERF] `libs/runtime/cogniverse_runtime/main.py:194-211 (lifespan backend-readiness loop` — The startup readiness loop uses the synchronous, blocking httpx.get(..., timeout=5) and _time.sleep(5) inside the async lifespan c
- [D] `libs/runtime/cogniverse_runtime/ingestion/processors/audio_embedding_generator.p` — `generate_acoustic_embedding` returns `np.zeros(512)` on ANY exception (librosa decode failure, CLAP load failure, corrupt audio).
- [B] `libs/runtime/cogniverse_runtime/ingestion/pipeline.py:1408,1429 (__main__ block)` — The module `__main__` is broken on the first line of real work. `config = PipelineConfig.from_config()` is called with no argument
- [E] `libs/runtime/cogniverse_runtime/ingestion/strategy_factory.py:142-151 (_create_s` — Contradicts its own contract. The class docstring (line 56) promises 'a typo like "models": "..." in the JSON raises TypeError at 
- [E] `libs/runtime/cogniverse_runtime/ingestion/pipeline.py:318-345 (_init_cache), 471` — PipelineArtifactCache is initialized in `_init_cache` (read from `pipeline_cache` config) but never read on the live ingestion pat
- [D] `libs/runtime/cogniverse_runtime/messaging_redis.py:264 (list_active_queues key p` — `session_id = key.split(":", 2)[1]` mis-parses any session_id that contains a colon. The active-marker key is `session:<session_id
- [E] `libs/runtime/cogniverse_runtime/messaging_redis.py:139-149 (enqueue/lpush) and 2` — Redis inbound message list (`inbound:<tenant>:<session>`) is created by LPUSH with NO TTL. The active-marker `session:<id>:tenant`
- [D] `libs/runtime/cogniverse_runtime/sandbox_pool.py:208-217 (_checkout race)` — Concurrent `with_session` calls for the SAME agent_type can orphan a live session. Pool is documented as called via `asyncio.to_th
- [D] `libs/runtime/cogniverse_runtime/admin/tenant_manager.py:769-782 (delete_tenant_i` — Simple-form tenant deletes use `full_name.endswith(original_suffix)` where original_suffix = `_` + tid. Deleting simple tenant `pr
- [B] `/home/amitjain/source/cogniverse/libs/runtime/cogniverse_runtime/routers/admin.p` — The profile UPDATE path validates only immutability (validate_update_fields, which checks the fixed immutable_fields set) but 'str
- [E] `libs/runtime/cogniverse_runtime/ingestion/pipeline_builder.py:106` — VideoIngestionPipelineBuilder.with_concurrency(max_concurrent) is a public fluent-API method that sets self._max_concurrent (defau

**synthetic** (2)
- [E] `/home/amitjain/source/cogniverse/libs/synthetic/cogniverse_synthetic/service.py:` — _get_generator computes the cache key as f"{optimizer_name.title().replace('_','')}Generator" — for 'unified' this is 'UnifiedGene
- [B] `libs/synthetic/cogniverse_synthetic/api.py:146 (generate_batch_synthetic_data)` — The POST /synthetic/batch/generate route's live body is never exercised. The only tests for it (tests/synthetic/unit/test_syntheti

**tests** (17)
- [HOLLOW-TEST] `tests/routing/unit/test_learned_reranker.py:80-104 (test_reranker_raises_on_heur` — Both tests claim to verify the reranker rejects a 'heuristic' model and an unknown model key, but they construct `LearnedReranker(
- [HOLLOW-TEST] `tests/runtime/unit/test_dispatcher_background_tasks.py:27-97 (test_background_ta` — These tests claim to guard the fire-and-forget `asyncio.create_task` strong-reference fix in AgentDispatcher (prod at agent_dispat
- [HOLLOW-TEST] `tests/agents/unit/test_detailed_report_agent.py:245-273 (test_process_a2a_task_s` — Sets `agent._dspy_lm = Mock()` but does NOT stub call_dspy or patch dspy.context, so `_generate_executive_summary` falls into its 
- [HOLLOW-TEST] `tests/agents/unit/test_detailed_report_agent.py:611-645 (test_process_impl_uses_` — Despite the name claiming it verifies enrichment (entities/relationships/enhanced_query) is consumed, the only assertion is `asser
- [HOLLOW-TEST] `tests/agents/unit/test_summarizer_agent.py:181-213 (test_process_a2a_task_succes` — `agent._dspy_lm = Mock()` and `summarization_module.forward = Mock(return_value=mock_prediction with summary='Brief summary of tes
- [HOLLOW-TEST] `tests/agents/unit/test_summarizer_agent.py:308-323 (test_extract_themes_function` — Calls the real `_extract_themes` but asserts only `len(themes) > 0` and `any('content' in t for t in themes)` — an unbounded subst
- [HOLLOW-TEST] `tests/agents/unit/test_search_agent.py:201-233 (test_search_by_text) and 239-332` — Class A self-confirming mock: the mock backend returns Mock results whose `.document.metadata = {'video_id':'video1',...}`. Prod `
- [HOLLOW-TEST] `tests/runtime/integration/test_a2a_multiturn.py:89-121 (test_multiturn_history_a` — These run the full real A2A->executor->dispatcher->DSPy->Vespa stack (good), but the core contract each docstring claims is never 
- [HOLLOW-TEST] `tests/runtime/integration/test_a2a_multiturn.py:200-220 (test_first_turn_no_rewr` — Both wrap their only real assertion in `try: result_data = json.loads(text) ... except json.JSONDecodeError: pass`. If the agent r
- [HOLLOW-TEST] `/home/amitjain/source/cogniverse/tests/backends/unit/test_partial_update.py:22-6` — test_feed_prepared_batch_forwards_operation_type and test_feed_defaults_to_full_feed set client.app = MagicMock() (the pyvespa bou
- [HOLLOW-TEST] `/home/amitjain/source/cogniverse/tests/e2e/test_tenant_extensibility_e2e.py:439-` — test_job_execute_with_wiki_delivery and test_job_execute_with_summarize_and_telegram patch cogniverse_runtime.job_executor.httpx.A
- [HOLLOW-TEST] `/home/amitjain/source/cogniverse/tests/evaluation/unit/test_quality_monitor.py:3` — Class A self-confirming mock. Each test sets `monitor._dataset_store = AsyncMock()` then the ONLY assertion is `mock_store.create_
- [HOLLOW-TEST] `/home/amitjain/source/cogniverse/tests/evaluation/unit/test_experiment_tracker.p` — Class A. The test mocks the heavy boundary (`patch("inspect_ai.eval", return_value=mock_result)`) and builds `mock_result[0].resul
- [HOLLOW-TEST] `tests/finetuning/test_trace_converter.py:32-45 (test_uses_public_traces_property` — The sole assertion is `assert converter.provider.traces is not None`. But the test itself sets `mock_provider.traces = Mock()` and
- [HOLLOW-TEST] `tests/finetuning/test_preference_extractor.py:192-251 (test_uses_public_traces_p` — Wraps `await extractor.extract(...)` in `try: ... except Exception: pass` ("May fail due to incomplete mocking, but we check the c
- [HOLLOW-TEST] `tests/events/unit/test_event_queue_integration.py:60-95 (test_pipeline_accepts_e` — Patches three SUT methods (`_init_backend`, `_resolve_strategy`, `_create_strategy_set_from_config`) then asserts only `pipeline.e
- [HOLLOW-TEST] `tests/telemetry/unit/test_session_tracking.py:320-341 (test_multiple_requests_gr` — Docstring promises 'multiple requests with same session_id are grouped', but the test only counts `spans_created` and calls `manag

**vespa** (4)
- [C] `libs/vespa/cogniverse_vespa/config/config_store.py:341-342` — get_config(..., version=N) builds `doc_id = f"{self.schema_name}::{config_id}::{version}"` then `yql = f'select * from {self.schem
- [E] `libs/vespa/cogniverse_vespa/ranking_strategy_extractor.py:126` — Inside _parse_ranking_profile: `schema_name = schema_json.get("schema", "")`. Every schema JSON in configs/schemas/*_schema.json u
- [C] `libs/vespa/cogniverse_vespa/ranking_strategy_extractor.py:58` — `is_single_vector = "_sv_" in schema_name.lower()` — substring (not token) matching of the schema-name discriminator, the exact `"
- [C] `/home/amitjain/source/cogniverse/libs/vespa/cogniverse_vespa/ranking_strategy_ex` — Single-vector schema detection uses only `is_single_vector = "_sv_" in schema_name.lower()`, diverging from the authoritative help

## LOW tier (83) — compact


**agents** (22)
- [B] `libs/agents/cogniverse_agents/routing/orchestration_evaluator.py:182-188 (_extra` — _extract_workflow_execution reads orch_attrs = span_row.get('attributes.orchestration', {}) and routing_attrs = span_row.get('attr
- [C] `libs/agents/cogniverse_agents/inference/instrumented_rlm.py:127 (_emit_sync)` — _emit_sync does loop.create_task(self._event_queue.enqueue(event)) without retaining the returned task reference. Per the Class C 
- [PERF] `libs/agents/cogniverse_agents/routing/orchestration_evaluator.py:60,150 (_proces` — _processed_span_ids is an in-memory set that grows monotonically for the lifetime of the OrchestrationEvaluator instance (one entr
- [SLOP] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/search/multi_moda` — analyze_ranking_quality computes Shannon entropy as `entropy += -p * (p if p == 0 else __import__('math').log2(p))`. The inner ter
- [E] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/search/multi_moda` — _calculate_temporal_score implements a full time-range / centrality / exponential-decay scoring path keyed on context['temporal'][
- [D] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/search_agent.py:1` — _format_public_result short-circuits with `if 'metadata' in result: return result`. Backend results are built by flattening **sr.d
- [D] `libs/agents/cogniverse_agents/contradiction_reconciliation_agent.py:375` — survivors=sorted(survivor_ids) where survivor_ids is a set built from m.get('id') for each reconciled member (line 344). None is f
- [SLOP] `libs/agents/cogniverse_agents/graph/graph_manager.py:457-463 (_extract_hits)` — _extract_hits is dead code — grep across libs/ and tests/ finds zero callers. It is a leftover helper that handled a response-obje
- [SLOP] `libs/agents/cogniverse_agents/graph/code_extractor.py:1-8 (module docstring) vs ` — Module docstring claims support for 'Python, TypeScript, JavaScript, Go, Rust, Java, C, C++, Ruby' and _DEFINITION_TYPES lists str
- [SLOP] `libs/agents/cogniverse_agents/entity_extraction_agent.py:1-10, 49-51, 476-488` — Decorative/AI-slop artifacts a senior reviewer would cut: (1) the banner comment block at lines 49-51 ('# ===...  # Type-Safe Inpu
- [B] `libs/agents/cogniverse_agents/text_analysis_agent.py:288-291` — The POST /analyze endpoint declares `text: str, tenant_id: str, analysis_type: str` as bare scalar parameters. In FastAPI, non-Pyd
- [E] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/inference/ab_harn` — RLMABRunner._run_with_rlm constructs RLMInference with event_queue=self._event_queue and tenant_id=self._tenant_id but never passe
- [C] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/optimizer/signatu` — selected_for_tenant logs a misleading WARNING for the normal default case. is_registered checks self._variants directly without th
- [E] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/mixins/rlm_aware_` — process_with_rlm always calls self._resolve_tenant_id_for_rlm(tenant_id) at line 200, which raises RuntimeError when neither an ex
- [SLOP] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/workflow_types.py` — WorkflowTemplate, WorkflowExecutionResult, and AgentPerformanceProfile in workflow_types.py are dead code — grep finds zero import
- [PERF] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/search_agent.py:1` — `process_with_rlm(...)` is a synchronous, network-heavy method (runs the full RLM recursion making multiple LM calls) but is calle
- [SLOP] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/summarizer_agent.` — `metadata={'processing_time': asyncio.get_event_loop().time(), ...}` stores the event-loop monotonic clock reference (seconds sinc
- [SLOP] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/search_agent.py:1` — Comment 'Note: _dspy_to_a2a_output and _get_agent_skills handled by A2AAgent base class' is factually wrong and actively misleadin
- [C] `/home/amitjain/source/cogniverse/libs/agents/cogniverse_agents/inference/instrum` — Fire-and-forget asyncio task with the reference discarded: `loop.create_task(self._event_queue.enqueue(event))` in InstrumentedRLM
- [E] `libs/agents/cogniverse_agents/federated_query_agent.py:169` — FederatedQueryAgent.__init__ stores self._config_manager = config_manager (line 169) but the attribute is never read again anywher
- [E] `libs/agents/cogniverse_agents/multi_document_synthesis_agent.py:199` — MultiDocumentSynthesisAgent.__init__ stores self._config_manager = config_manager (line 199), used only via the local parameter in
- [E] `libs/agents/cogniverse_agents/kg_traversal_agent.py:229` — KnowledgeGraphTraversalAgent.__init__ stores self._config_manager = config_manager (line 229), consumed only via the local paramet

**cli** (2)
- [SLOP] `libs/cli/cogniverse_cli/code.py:18, libs/cli/cogniverse_cli/streaming.py:14, lib` — Three CLI modules place `from cogniverse_cli.constants import RUNTIME_URL  # noqa: F401` after module-level executable code (Conso
- [SLOP] `libs/cli/cogniverse_cli/index.py:3-4 docstring` — Docstring claims indexing 'does two things in parallel' but index_files iterates files strictly sequentially (for file_path in fil

**core** (11)
- [E] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/registries/agent_regi` — AgentRegistry.__init__ unconditionally constructs httpx.AsyncClient(timeout=10.0) on every instantiation. The /health route (libs/
- [SLOP] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/factories/backend_fac` — Decorative narrative comments enumerate non-existent/mislabeled phases: '# Phase 1: Create backend instance', then jumps to '# Pha
- [SLOP] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/registries/schema_reg` — Debug/log lines use emoji and mislabeled levels in the production deploy path: 'logger.error(f"🔍 Backend deployment FAILED: {e}")'
- [D] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/schemas/filesystem_lo` — list_available_schemas derives the schema name with f.stem.replace('_schema', '') which replaces ALL occurrences, not just the tra
- [SLOP] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/query/encoders.py:209` — VideoPrismQueryEncoder.__init__ emits five INFO-level introspection logs on a hot construction path that merely restate hasattr() 
- [SLOP] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/config/__init__.py:1-` — Backward-compat shim that re-exports everything from cogniverse_foundation.config via `from ... import *` plus `import utils`. The
- [SLOP] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/agents/rlm_options.py` — Docstring cites a fabricated/hallucinated arxiv reference 'https://arxiv.org/abs/2512.24601' (arxiv IDs are YYMM-prefixed; 2512 = 
- [E] `/home/amitjain/source/cogniverse/libs/core/cogniverse_core/memory/pinning.py:247` — PinService.pin and FederationService.promote_to_org_trunk both call self._mm.add_memory(...) which is documented to return None wh
- [E] `libs/core/cogniverse_core/common/health_mixin.py:19 (HealthCheckMixin.setup_heal` — HealthCheckMixin.setup_health_endpoint (which registers a GET /health route returning get_health_status()) has zero production cal
- [E] `libs/core/cogniverse_core/agents/base.py:152` — _DispatchedPromptOverlayContext sets self._prompts (line 152 init {} and line 166 self._prompts = prompts) but never reads self._p
- [E] `libs/core/cogniverse_core/common/dynamic_dspy_mixin.py:180` — DynamicDSPyMixin.create_optimizer() builds an optimizer, RETURNS it to the caller, and also stores it on self._optimizer (line 180

**dashboard** (4)
- [C] `/home/amitjain/source/cogniverse/libs/dashboard/cogniverse_dashboard/tabs/optimi` — The golden-dataset-selection branch hardcodes the Phoenix GraphQL endpoint to 'http://localhost:6006/graphql' instead of using the
- [D] `/home/amitjain/source/cogniverse/libs/dashboard/cogniverse_dashboard/tabs/rlm_ab` — The Δ latency and Δ tokens metrics use truthiness to decide whether data is present: f"{agg.avg_latency_delta_ms:.1f}" if agg.avg_
- [SLOP] `/home/amitjain/source/cogniverse/libs/dashboard/cogniverse_dashboard/tabs/optimi` — In the Metrics Dashboard 'Search Quality Metrics' section, the metric 'Search Queries Evaluated' is hardcoded to the string 'N/A' 
- [C] `/home/amitjain/source/cogniverse/libs/dashboard/cogniverse_dashboard/app.py:2538` — The Interactive Search tab gates the Search button on agent_status.get('Search Agent') or agent_status.get('Video Search Agent'). 

**evaluation** (2)
- [SLOP] `libs/evaluation/cogniverse_evaluation/span_evaluator.py:197-229 (_create_mock_sp` — _create_mock_spans_df fabricates hardcoded fake search spans (video ids, scores) and has no caller in production or tests — dead. 
- [PERF] `libs/evaluation/cogniverse_evaluation/quality_monitor.py:333-386 (evaluate_golde` — Golden-set eval issues one sequential await client.post per query, and live-traffic eval issues one sequential await judge._call_l

**foundation** (1)
- [PERF] `libs/foundation/cogniverse_foundation/telemetry/manager.py:523-535 (_evict_old_t` — _evict_old_tracers only trims self._tenant_tracers; self._tenant_providers (keyed by tenant_id:project) is never evicted and grows

**misc** (1)
- [B] `scripts/auto_optimization_trigger.py:33 (AutoOptimizationTrigger.run / trigger_o` — Three operational CLI scripts with substantial production logic (302, 535, 462 LOC respectively) have zero test coverage — neither

**runtime** (9)
- [C] `libs/runtime/cogniverse_runtime/routers/events.py:98` — datetime.utcnow() is naive (no tzinfo) and deprecated in 3.12+. The emitted 'connected' SSE event timestamp lacks timezone, incons
- [SLOP] `libs/runtime/cogniverse_runtime/routers/admin.py:66-109 (get_config_manager_depe` — Two ~20-line docstrings with Args/Returns/Raises sections that restate the one-line behaviour ('return injected instance or raise 
- [SLOP] `libs/runtime/cogniverse_runtime/ingestion/processors/embedding_generator/embeddi` — Duplicate `raise` — an unreachable second `raise` immediately after the first inside `except Exception` of `_load_model`. Dead lin
- [D] `libs/runtime/cogniverse_runtime/ingestion/processors/single_vector_processor.py:` — `process(self, video_path, output_dir=None, **kwargs)` calls `self.process_video(video_path, output_dir)`, but `process_video`'s s
- [E] `libs/runtime/cogniverse_runtime/ingestion_worker/queue.py:121-138 + backpressure` — Per-tenant `ingest:active:<tenant>` counter is incremented at submit but only decremented in the worker's cleanup block (worker.py
- [SLOP] `libs/runtime/cogniverse_runtime/sandbox_manager.py:404-434 (exec_in_sandbox exec` — ~30 lines of near-identical span-emission + `_classify_exec_failure` + result-dict construction are copy-pasted between the pooled
- [C] `libs/runtime/cogniverse_runtime/messaging_redis.py:126 (RedisInboundQueue.is_clo` — The two interchangeable backends expose `is_closed` with different shapes: in-pod `InboundQueue.is_closed` is a synchronous `@prop
- [UNTESTED] `/home/amitjain/source/cogniverse/libs/runtime/cogniverse_runtime/job_executor.py` — _deliver_to_telegram swallows ALL exceptions (logger.error then return) and special-cases a 404 by logging-and-skipping. The only 
- [E] `libs/runtime/cogniverse_runtime/ingestion/processors/vlm_descriptor.py:36` — VLMDescriptor.__init__ sets self._modal_process = None (line 36) and never reassigns or reads it (grep -rn '_modal_process' return

**synthetic** (4)
- [SLOP] `/home/amitjain/source/cogniverse/libs/synthetic/cogniverse_synthetic/generators/` — _generate_entity_query docstring says ValidatedEntityQueryGenerator 'eliminat[es] the need for arbitrary fallbacks' and the code w
- [SLOP] `/home/amitjain/source/cogniverse/libs/synthetic/cogniverse_synthetic/approval/fe` — get_regeneration_stats computes 'successful' as items whose metadata['regeneration_attempt'] <= self.max_attempts. Every regenerat
- [C] `/home/amitjain/source/cogniverse/libs/synthetic/cogniverse_synthetic/schemas.py:` — RoutingExperienceSchema.timestamp and WorkflowExecutionSchema.timestamp use default_factory=datetime.now (naive local time), not d
- [B] `/home/amitjain/source/cogniverse/libs/synthetic/cogniverse_synthetic/api.py:146-` — POST /synthetic/batch/generate (the multi-batch endpoint) has no test reaching it (tests/synthetic/unit/test_synthetic_api_http.py

**telemetry-phoenix** (2)
- [B] `libs/telemetry-phoenix/cogniverse_telemetry_phoenix/provider.py:102-130 (Phoenix` — get_span_by_id fetches up to 10000 spans (get_spans(limit=10000)) into a DataFrame for every single-ID lookup, then filters in pan
- [E] `libs/telemetry-phoenix/cogniverse_telemetry_phoenix/evaluation/analytics.py:36` — PhoenixAnalytics.__init__ sets self._cache = {} (line 36) but the cache is never read or written again (grep '_cache' in the file 

**tests** (19)
- [HOLLOW-TEST] `tests/agents/unit/test_orchestrator_agent.py:1025-1063 (test_no_workflow_intelli` — Three `_load_artifact` tests whose only assertion is the comment `# Should not raise` (no assert statement at all in test_no_workf
- [HOLLOW-TEST] `tests/routing/unit/test_multi_modal_reranker.py:74-98 (test_temporal_reranking)` — The test sets up a temporal context that should prefer recent results and the docstring/comment asserts 'doc_1 is most recent (2 d
- [HOLLOW-TEST] `tests/runtime/integration/test_dispatcher_egress_policy_consult.py:110-158 (Test` — The test patches the SUT's own `consult_egress_policy` into a recording spy, wraps the entire dispatch call in `try: ... except Ex
- [HOLLOW-TEST] `tests/agents/unit/test_summarizer_agent.py:354-366 (test_identify_visual_element` — Calls real `_identify_visual_elements` but the ONLY assertion is `isinstance(visual_elements, list)` — it never checks the content
- [HOLLOW-TEST] `tests/agents/unit/test_text_analysis_agent.py:216-258 (test_analyze_endpoint)` — Class A: patches `TextAnalysisAgent.analyze_text` (a method on the SUT) to return a literal dict, then asserts the /analyze respon
- [HOLLOW-TEST] `tests/agents/unit/test_standalone_agent_apps.py:76-102 (test_summarizer_process_` — monkeypatch the module-level agent singleton with a MagicMock whose .summarize/.generate_report return MagicMock(summary='the answ
- [HOLLOW-TEST] `tests/runtime/unit/test_a2a_server.py:253-283 (test_message_send_error_returns_e` — The docstring/name says 'When dispatch raises, executor returns error as text message' and a comment says 'The error should be in 
- [HOLLOW-TEST] `tests/runtime/unit/test_a2a_server.py:285-316 (test_context_id_passed_to_dispatc` — test_context_id_passed_to_dispatcher's only meaningful assertion is `assert ctx['context_id'] is not None` with an inline comment 
- [HOLLOW-TEST] `tests/runtime/unit/test_policy_enforcing_transport.py:186-196 (test_error_lists_` — The assertions live inside an `except EgressDeniedError as exc:` block with no `else` / no `pytest.raises`. If the request to http
- [HOLLOW-TEST] `/home/amitjain/source/cogniverse/tests/e2e/test_a2a_multiturn_e2e.py:120` — In test_conversation_history_passed (downstream_result branch): `assert ds.get("agent") or ds.get("results") is not None`. Python 
- [HOLLOW-TEST] `/home/amitjain/source/cogniverse/tests/evaluation/unit/test_quality_monitor.py:4` — Weak echo assertion. The test patches SpanEvaluator, `_evaluate_agent_spans`, and `_store_live_eval_result` (mocking out everythin
- [HOLLOW-TEST] `/home/amitjain/source/cogniverse/tests/routing/unit/test_multi_modal_reranker.py` — Assertion does not match the stated contract. The docstring/comments claim 'More recent results should rank higher with temporal c
- [HOLLOW-TEST] `/home/amitjain/source/cogniverse/tests/evaluation/unit/test_multi_turn_llm_judge` — test_hybrid_multi_turn discards the evaluate() return (`_ = evaluator.evaluate(...)`) and asserts only `mock_free_llm.called` / `m
- [HOLLOW-TEST] `/home/amitjain/source/cogniverse/tests/memory/integration/test_contradiction_wri` — Self-admitted weak assertion. The docstring inside the test concedes 'We can't assert zero ... so just confirm the hook didn't err
- [HOLLOW-TEST] `/home/amitjain/source/cogniverse/tests/evaluation/unit/test_experiment_tracker.p` — Constructor-acceptance only. The test patches the plugin module and constructs ExperimentTracker, with NO assertion at all — the t
- [HOLLOW-TEST] `/home/amitjain/source/cogniverse/tests/evaluation/unit/test_task.py:36-53 (test_` — Class A partial. These mock MemoryDataset, the solver factory, and get_configured_scorers (every collaborator), then assert `task 
- [HOLLOW-TEST] `tests/telemetry/unit/test_session_tracking.py:109-126 (test_session_context_uses` — Patches the boundary `openinference.instrumentation.using_session` to a MagicMock and asserts `mock_using_session.assert_called_on
- [HOLLOW-TEST] `tests/messaging/unit/test_runtime_client_crud.py:44-258 (all Wiki/Instructions/M` — Class A: these mock the httpx transport (`rc._client`) and assert the URL + JSON/params the RuntimeClient BUILDS (e.g. `/wiki/lint
- [HOLLOW-TEST] `tests/messaging/unit/test_auth.py:107-114 (TestUserTenantMapper.test_get_tenant_` — Mocks the memory_manager boundary's `search_memory` to return a hardcoded string `'User 12345 on telegram is mapped to tenant acme

**vespa** (6)
- [D] `libs/vespa/cogniverse_vespa/vespa_schema_manager.py:724-730` — delete_schema computes `target = get_tenant_schema_name(tenant_id, base)` (which CANONICALIZES tenant_id, e.g. acme -> acme:acme -
- [PERF] `libs/vespa/cogniverse_vespa/backend.py:1302,1340,1377,1444` — create_metadata_document / get_metadata_document / query_metadata_documents / delete_metadata_document each call make_vespa_app(ur
- [D] `libs/vespa/cogniverse_vespa/embedding_processor.py:13-22,17` — _SINGLE_VECTOR_TOKENS = ("_sv_","_lvt_") classifies single-vector schemas by name token, but agent_memories (a tensor<float>(d0[76
- [SLOP] `libs/vespa/cogniverse_vespa/backend.py:1461-1463 (and similar banner at 948)` — Decorative section-banner comment block `# =====...` / `# Connection Management` / `# =====...`. Pure visual filler a senior revie
- [SLOP] `libs/vespa/cogniverse_vespa/ingestion_client.py:97-101,388,395` — Dead/stale comments: lines 388 `# Removed _feed_prepared - using only batch method` and 395 `# Removed duplicate conversion method
- [E] `libs/vespa/cogniverse_vespa/backend.py:69` — VespaBackend.__init__ stores self._backend_config = backend_config (line 69) but never reads self._backend_config afterward — the 