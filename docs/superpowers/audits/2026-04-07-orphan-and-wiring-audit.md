# Cogniverse Code Audit — Orphans, Duplicates, and Feature Wiring

**Date:** 2026-04-07
**Scope:** End-to-end audit of feature wiring, duplicate definitions, and orphan code across the cogniverse codebase.
**Method:** 9 parallel investigations — 4 semantic feature traces (StrategyLearner, QualityMonitor, Telemetry+RLM, Messaging+Wiki) + 2 mechanical scans (duplicates, orphan endpoints) + 3 additional feature traces (Jobs, Sandbox, Tenant Extensibility).
**Severity tiers:** **BROKEN** (feature does not work end-to-end) > **DEGRADED** (works partially or silently misroutes) > **ORPHANED** (defined but no caller) > **OK** (verified working).

---

## Critical Findings (BROKEN — feature does not work)

| # | Severity | File:Line | Name | Evidence | Recommended action |
|---|---|---|---|---|---|
| 1 | **BROKEN** | `charts/cogniverse/templates/all-resources.yaml:279-337` | QualityMonitor sidecar | `data/testset/` not mounted; sidecar passes `--golden-dataset-path data/testset/...` (relative path that doesn't exist in container). Sidecar crash-loops at startup. | Mount `data/testset/evaluation/` as ConfigMap or PVC into the sidecar. Pass absolute path. |
| 2 | **BROKEN** | `libs/evaluation/cogniverse_evaluation/quality_monitor.py:42-47` | `SPAN_NAME_BY_AGENT` lookup table | Hard-coded names (`search_service.search`, `summarizer_agent.process`, `routing_agent.route`) do NOT match what agents actually emit (e.g. RoutingAgent emits `cogniverse.routing`). Live-traffic eval queries return zero spans. | Replace lookup table with dynamic discovery from `AgentRegistry`, or update names to match real span names confirmed via Phoenix UI. |
| 3 | **BROKEN** | `libs/runtime/cogniverse_runtime/routers/tenant.py:30, 40, 474` | Jobs scheduler / `set_argo_config()` | `set_argo_config()` is defined but **never called from `main.py` lifespan**. `_argo_api_url` stays `None`. Job creation succeeds (persists to ConfigStore) but `if _argo_api_url:` at line 474 silently skips Argo CronWorkflow submission. **No fallback in-process scheduler exists.** | Add `set_argo_config(...)` call in `main.py` startup, reading endpoint from env var. Add startup-time validation that fails fast if Argo unreachable when jobs feature enabled. |
| 4 | **BROKEN** | `libs/messaging/cogniverse_messaging/gateway.py:124-193` | Telegram custom command dispatch | `_handle_message` only checks `parsed.is_help` and `parsed.query`. Parsed `is_wiki / is_instructions / is_memories / is_jobs` flags from `command_router.py:126-172` are silently dropped. Help text advertises the commands but they fall through to `routing_agent` instead of their respective endpoints. | Implement dispatch arms in `_handle_message` for each command family. Add corresponding methods on `runtime_client.py` (currently has none for these). |
| 5 | **BROKEN** | `libs/messaging/cogniverse_messaging/runtime_client.py` | runtime_client missing CRUD methods | No methods for `/wiki/save`, `/wiki/lint`, `/instructions/*`, `/memories/*`, `/jobs/*`. Even if gateway dispatched correctly there's no client to call. | Add `save_wiki`, `set_instructions`, `list_memories`, `create_job`, etc. — one method per advertised Telegram subcommand. |

---

## Degraded Features (work partially but route through dead/wrong paths)

| # | Severity | File:Line | Name | Evidence | Recommended action |
|---|---|---|---|---|---|
| 6 | **DEGRADED** | `libs/agents/cogniverse_agents/optimizer/strategy_learner.py:481, 499, 514` | StrategyLearner per-agent filtering | `get_strategies_for_agent` accepts `agent_name` but immediately overrides it to `STRATEGY_AGENT_NAME = "_strategy_store"`. All agents share one global namespace; per-agent filtering is dead code. | Either honor `agent_name` (filter by agent in Mem0 metadata) or remove the parameter. Pick one — current code lies about what it does. |
| 7 | **DEGRADED** | `libs/runtime/cogniverse_runtime/optimization_cli.py:195` | Strategy distillation trigger | `learn_from_trigger_dataset` only invoked from `run_triggered_optimization()`, which itself only fires when `QualityMonitor` (broken — see #1, #2) detects a quality drop. **No scheduled cron, no manual CLI for routine learning**. | Add a cron-style trigger in `main.py` lifespan or an Argo CronWorkflow that calls `optimization_cli` on a schedule independent of QualityMonitor. |
| 8 | **DEGRADED** | `libs/agents/cogniverse_agents/routing_agent.py:788, 1096, 1115` | RoutingAgent — no `inject_context_into_prompt` | RoutingAgent only calls `_get_tenant_instructions`, never the full `inject_context_into_prompt`. So learned strategies never reach the router. | Wire `inject_context_into_prompt(tenant_id=, agent_name="routing_agent", base_prompt=)` into `_process_impl`. |
| 9 | **DEGRADED** | `libs/agents/cogniverse_agents/coding_agent.py:126, 234, 264-265` | CodingAgent — no MemoryAwareMixin | Wired for RLM but does NOT inherit MemoryAwareMixin or call `inject_context_into_prompt`. Coding agent receives no learned strategies, no tenant memories, no tenant instructions. | Add `MemoryAwareMixin` to inheritance and wire `inject_context_into_prompt` into `_process_impl`. |
| 10 | **DEGRADED** | `libs/core/cogniverse_core/agents/base.py:345-491` | `AgentBase.process()` — no telemetry span | `process()` / `run()` does NOT wrap `_process_impl` in any span. SearchAgent, CodingAgent, TextAnalysisAgent, SummarizerAgent, DetailedReportAgent emit ZERO spans during processing. Only RoutingAgent emits its own span manually. Quality monitor and observability dashboards see nothing for 5/6 agents. | Wrap `_process_impl` in `telemetry_provider.span(f"{agent_name}.process", tenant_id=...)` inside `AgentBase`. One change fixes all 5 agents. |
| 11 | **DEGRADED** | `libs/foundation/cogniverse_foundation/telemetry/manager.py:226-230, 519-535` | TelemetryManager silent NoOp fallback | When Phoenix unavailable, `TelemetryManager` falls back to `NoOpSpan` without raising. Agents emit "spans" that go nowhere. Combined with #10, observability is silently broken. | Add startup-time Phoenix reachability probe in `main.py` lifespan. Fail fast if telemetry required but unreachable. |
| 12 | **DEGRADED** | `libs/runtime/cogniverse_runtime/routers/wiki.py:37, 90, 100` | `WikiSaveRequest.tenant_id` ignored | Singleton `wm = WikiManager(tenant_id="default")` bound at startup in `main.py:285-319`. The `tenant_id` field on `WikiSaveRequest` is parsed but never used to route to a per-tenant wiki. All tenants write to the "default" wiki. | Make `WikiManager` tenant-aware: either look up per-tenant managers from a registry, or pass `tenant_id` through to all reads/writes. |
| 13 | **DEGRADED** | `libs/runtime/cogniverse_runtime/routers/agents.py:337-358` | `POST /agents/{name}/upload` | Implemented as `upload_file_to_agent()` but unconditionally raises `HTTPException(501, "not supported")`. Test expects 501. | Either implement actual upload handling or remove the endpoint. Stubs that return 501 forever are noise. |
| 14 | **DEGRADED** | `libs/runtime/cogniverse_runtime/agent_dispatcher.py:67, 371` | Memory not auto-wired | Dispatcher passes `tenant_id` through all execution paths but never calls `MemoryAwareMixin.initialize_memory()`. Memory consumption requires per-agent `enable_memory: true` config flag (default `False`). Agents have memory capability but it's dormant in production. | Initialize memory in dispatcher when constructing agents, using the propagated `tenant_id`. Make memory enabled by default. |
| 15 | **DEGRADED** | `libs/agents/cogniverse_agents/optimizer/strategy_learner.py` (XGBoost gate) | XGBoost training-decision gate | Gating block requires `telemetry_provider` to fetch span data, but `telemetry_provider` is never injected into the StrategyLearner constructor at any call site. Gating block is dead 100% of the time. | Either inject `telemetry_provider` at construction sites, or remove the unreachable gating block. |

---

## Orphans (defined, no caller)

| # | Severity | File:Line | Name | Evidence | Recommended action |
|---|---|---|---|---|---|
| 16 | **ORPHANED** | `libs/core/cogniverse_core/agents/memory_aware_mixin.py` | `MemoryAwareMixin` (base copy) | Two copies exist. The **agents/** copy has the extended `inject_context_into_prompt` and `get_strategies`; the **core/** copy has only the base. **Only `text_analysis_agent.py` imports the agents/ copy**; the other 9 agents import the core copy and silently miss the extended features (strategies + instructions injection). | Delete `libs/core/cogniverse_core/agents/memory_aware_mixin.py`. Update all 9 agents to import from `cogniverse_agents.memory_aware_mixin`. (Already on the user's TODO list.) |
| 17 | **ORPHANED** | `libs/runtime/cogniverse_runtime/routers/wiki.py:86-90` | `GET /wiki/lint` | Endpoint defined; HELP_TEXT in `command_router.py` documents `/wiki lint` but gateway never dispatches it. No CLI, no UI, no test caller. | Either wire to Telegram via fix #4, or delete the endpoint. |
| 18 | **ORPHANED** | `libs/runtime/cogniverse_runtime/routers/wiki.py:93-103` | `DELETE /wiki/topic/{slug}` | Endpoint defined, never documented in help text, no CLI/UI/test caller. | Delete unless wiring planned. |
| 19 | **ORPHANED** | `libs/messaging/cogniverse_messaging/command_router.py:126-172` | 4 parsed-but-unhandled command families | `is_wiki`, `is_instructions`, `is_memories`, `is_jobs` flags set on `ParsedCommand` but never read in `gateway.py`. | Covered by fix #4 + #5. |

---

## Verified OK (audited and working)

| # | Severity | File:Line | Name | Evidence |
|---|---|---|---|---|
| 20 | **OK** | `libs/runtime/cogniverse_runtime/sandbox_manager.py:65-94, 147-183` | OpenShell sandbox / CodingAgent execution | Real OpenShell Python SDK, real gRPC connection, `OPENSHELL_GATEWAY_ENDPOINT` env var wired through Helm chart, mTLS certs synced via `cogniverse_cli/sandbox.py`. CodingAgent raises `RuntimeError` if sandbox unavailable (no silent fallback). Real e2e tests start a real gateway and assert real stdout. **Not vaporware.** |
| 21 | **OK** | `libs/agents/cogniverse_agents/{search_agent.py:1803, text_analysis_agent.py:164, summarizer_agent.py:825, detailed_report_agent.py:857}` | RLM integration | Fully wired in 4 of 4 advertised agents (SearchAgent, TextAnalysisAgent, SummarizerAgent, DetailedReportAgent). |
| 22 | **OK** | `libs/runtime/cogniverse_runtime/job_executor.py:129-148` | Job post-action delivery (`_deliver_to_wiki`, `_deliver_to_telegram`) | POSTs to real `/wiki/save` and `/messaging/send`. Tested in `test_tenant_extensibility_e2e.py`. Implementation is correct — only orphaned because trigger #3 is broken. |
| 23 | **OK** | `libs/runtime/cogniverse_runtime/routers/ingestion.py:240-359` + `tests/runtime/unit/test_multimodal_graph_extraction.py` + `tests/e2e/test_graph_cli_e2e.py` | Knowledge graph extraction | `_extract_text_for_graph` + `_extract_graph_from_multimodal` correctly harvest Whisper/VLM/OCR text and feed `DocExtractor`. Unit tests cover all branches. E2E tests verify `/graph/upsert`, `/graph/stats`, `/graph/neighbors`, `/graph/path` round-trips through real Vespa. Documentation in `docs/user/knowledge-graph.md` matches code. |
| 24 | **OK** | `libs/runtime/cogniverse_runtime/routers/tenant.py:107-155` + `MemoryAwareMixin._get_tenant_instructions` | Per-tenant Instructions | ConfigStore-backed, tenant-isolated, consumed by all 9 agents that import core MemoryAwareMixin via `get_prompt_context()`. |
| 25 | **OK** | `libs/runtime/cogniverse_runtime/routers/tenant.py:446-540` | Per-tenant Jobs CRUD storage | ConfigStore-backed with proper tenant_id filtering on POST/GET/DELETE. (Storage layer only — see #3 for the broken scheduling layer.) |
| 26 | **OK** | `libs/runtime/cogniverse_runtime/routers/tenant.py:248-381` + Mem0 manager | Per-tenant Memories storage | Each tenant gets a dedicated `agent_memories_{tenant_id}` Vespa schema. All Mem0 calls pass `user_id=tenant_id`. Storage isolation correct. (Consumer wiring is degraded — see #14.) |

---

## Prioritized Fix Order

### Wave 1 — Critical infra (one PR each, small, high leverage)
1. **Fix #10** (telemetry span in AgentBase) — one-line wrap in `base.py` makes 5 agents observable. Required for #2 to even matter.
2. **Fix #1** (mount dataset into QualityMonitor sidecar) — unblocks the entire monitoring loop.
3. **Fix #2** (correct span names in QualityMonitor lookup table) — finishes unblocking the monitoring loop.
4. **Fix #3** (call `set_argo_config()` in `main.py` lifespan) — unblocks all scheduled jobs.

### Wave 2 — Wiring fixes (each enables a degraded feature)
5. **Fix #16** (delete duplicate `MemoryAwareMixin`, redirect imports)
6. **Fix #14** (auto-init memory in dispatcher)
7. **Fix #8** (wire `inject_context_into_prompt` into RoutingAgent)
8. **Fix #9** (wire MemoryAwareMixin into CodingAgent)
9. **Fix #12** (make `WikiManager` tenant-aware)

### Wave 3 — Telegram surface area
10. **Fix #4 + #5** together (gateway dispatch arms + runtime_client methods for the 4 command families)

### Wave 4 — Cleanup
11. **Fix #6** (StrategyLearner per-agent filtering — pick honor or remove)
12. **Fix #7** (scheduled distillation cron, independent of QualityMonitor)
13. **Fix #11** (Phoenix reachability probe at startup)
14. **Fix #13** (decide: implement or delete `/agents/upload`)
15. **Fix #15** (XGBoost gate — inject telemetry_provider or delete)
16. **Fix #17, #18** (delete or wire orphan wiki endpoints)

---

## Audit Statistics

- **Features audited:** 9 (StrategyLearner, QualityMonitor, Telemetry, RLM, Messaging, Wiki, Jobs, Sandbox, Tenant Extensibility) + mechanical scans
- **BROKEN:** 5 findings
- **DEGRADED:** 10 findings
- **ORPHANED:** 4 findings
- **OK (verified):** 7 features
- **Most common failure mode:** features built bottom-up but never wired into the startup lifespan or the dispatcher layer
- **Most surprising finding:** 5 of 6 main agents emit zero telemetry spans (#10) — fixing this is a one-line change in `AgentBase` that unblocks a huge fraction of the monitoring/quality stack
