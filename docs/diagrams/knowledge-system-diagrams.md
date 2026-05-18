# Knowledge System Diagrams
---

## Table of Contents
1. [Contradiction Reconciliation Agent Flow](#contradiction-reconciliation-agent-flow)
2. [9-Agent Knowledge Dispatch](#9-agent-knowledge-dispatch)
3. [DeepSynthesisWorkflow Loop](#deepsynthesisworkflow-loop)
4. [Sandbox Boot Policy Decision](#sandbox-boot-policy-decision)
5. [Optimizer Canary FSM](#optimizer-canary-fsm)
6. [Daily-Cleanup Workflow + Monthly-Reports Upload](#daily-cleanup-workflow--monthly-reports-upload)

---

## Contradiction Reconciliation Agent Flow

`ContradictionReconciliationAgent._process_impl`
(`libs/agents/cogniverse_agents/contradiction_reconciliation_agent.py`)
takes a `subject_key`'s `conflict_member_ids`, fetches each via
`memory_manager.memory.get(mid)`, then calls `reconcile(members, policy)`
from `libs/core/cogniverse_core/memory/contradiction.py`. The
`ContradictionDetector` upstream groups by `metadata.subject_key` and
emits one `ConflictSet` per subject with more than one distinct
`_content_signature`. The resolved view depends on the schema's
`contradiction_policy` (or a request-time `policy_override`).

```mermaid
sequenceDiagram
    participant Caller as Caller / Router
    participant Agent as ContradictionReconciliationAgent
    participant Memory as Mem0MemoryManager
    participant Detector as ContradictionDetector
    participant Reconcile as reconcile(members, policy)
    participant Registry as KnowledgeRegistry

    Caller->>Agent: ContradictionReconciliationInput(target_kind, conflict_member_ids, policy_override?)
    Agent->>Registry: get(target_kind) -> KnowledgeSchema
    Registry-->>Agent: schema (contradiction_policy)
    Agent->>Agent: policy = policy_override or schema.contradiction_policy

    Note over Detector: Upstream snapshot — detector previously emitted the ConflictSet<br/>by grouping candidates by metadata.subject_key + _content_signature
    Detector-->>Caller: ConflictSet(subject_key, conflicting_memory_ids, detected_at)

    loop For each mid in conflict_member_ids
        Agent->>Memory: memory.get(mid)
        Memory-->>Agent: member dict (or missing)
    end

    Agent->>Reconcile: reconcile(members, policy)
    Reconcile-->>Agent: resolved view (survivors, disputed flags)
    Agent-->>Caller: ContradictionReconciliationOutput(policy_used, resolved[], survivors[])
```

Within `reconcile`, the per-subject branch picks the survivor by policy:

```mermaid
flowchart TD
    Start["<span style='color:#000'>members for one subject_key<br/>(distinct content signatures)</span>"]
    Policy{"<span style='color:#000'>schema.contradiction_policy<br/>(or policy_override)</span>"}

    Latest["<span style='color:#000'>LATEST_WINS<br/>_pick_latest: max(members, key=created_at)</span>"]
    Trust["<span style='color:#000'>TRUST_RANKED<br/>_pick_trust_ranked:<br/>max(trust.score × provenance.confidence)</span>"]
    Both["<span style='color:#000'>PRESERVE_BOTH<br/>_mark_disputed: keep all,<br/>set metadata.disputed=True</span>"]

    WinnerOne["<span style='color:#000'>winner_memory_id = survivor.id<br/>disputed_members = []</span>"]
    WinnerTrust["<span style='color:#000'>winner_memory_id = survivor.id<br/>disputed_members = []</span>"]
    WinnerBoth["<span style='color:#000'>winner_memory_id = None<br/>disputed_members = all members</span>"]

    Out["<span style='color:#000'>ContradictionReconciliationOutput<br/>policy_used, survivors[], resolved[].disputed</span>"]

    Start --> Policy
    Policy -->|latest_wins| Latest --> WinnerOne --> Out
    Policy -->|trust_ranked| Trust --> WinnerTrust --> Out
    Policy -->|preserve_both| Both --> WinnerBoth --> Out

    style Start fill:#90caf9,stroke:#1565c0,color:#000
    style Policy fill:#ffcc80,stroke:#ef6c00,color:#000
    style Latest fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Trust fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Both fill:#ce93d8,stroke:#7b1fa2,color:#000
    style WinnerOne fill:#a5d6a7,stroke:#388e3c,color:#000
    style WinnerTrust fill:#a5d6a7,stroke:#388e3c,color:#000
    style WinnerBoth fill:#a5d6a7,stroke:#388e3c,color:#000
    style Out fill:#90caf9,stroke:#1565c0,color:#000
```

---

## 9-Agent Knowledge Dispatch

`libs/runtime/cogniverse_runtime/routers/knowledge.py` exposes nine
`/admin/tenants/{tenant_id}/knowledge/...` POST routes, one per
knowledge agent. Every route resolves the per-tenant Mem0 instance
through `_build_factory(tenant_id)` which constructs
`Mem0MemoryManager(tenant_id)` and calls
`lazy_init_memory(mm, tenant_id, _require_config_manager())` when
`mm.memory` is not yet wired. The same factory is either passed as
`memory_manager_factory=` or attached via `_inject_memory(...)` (which
sets `agent.memory_manager`, `_memory_initialized`,
`_memory_tenant_id`, `_memory_agent_name`).

```mermaid
flowchart LR
    Client["<span style='color:#000'>HTTP POST /admin/tenants/{t}/knowledge/&lt;route&gt;</span>"]
    Router["<span style='color:#000'>knowledge.py router</span>"]
    Factory["<span style='color:#000'>_build_factory(tenant_id)<br/>Mem0MemoryManager(t)<br/>lazy_init_memory(mm, t, cfg_mgr)</span>"]

    Audit["<span style='color:#000'>AuditExplanationAgent<br/>audit/explain</span>"]
    Citation["<span style='color:#000'>CitationTracingAgent<br/>citations/trace</span>"]
    Summarize["<span style='color:#000'>KnowledgeSummarizationAgent<br/>summarize</span>"]
    Reconcile["<span style='color:#000'>ContradictionReconciliationAgent<br/>contradictions/reconcile</span>"]
    MultiDoc["<span style='color:#000'>MultiDocumentSynthesisAgent<br/>synthesis/multi_doc</span>"]
    KG["<span style='color:#000'>KnowledgeGraphTraversalAgent<br/>kg/traverse</span>"]
    CrossTenant["<span style='color:#000'>CrossTenantComparisonAgent<br/>cross_tenant/compare</span>"]
    Federated["<span style='color:#000'>FederatedQueryAgent<br/>federated/query</span>"]
    Temporal["<span style='color:#000'>TemporalReasoningAgent<br/>temporal/reason</span>"]

    Client --> Router

    Router --> Audit
    Router --> Citation
    Router --> Summarize
    Router --> Reconcile
    Router --> MultiDoc
    Router --> KG
    Router --> CrossTenant
    Router --> Federated
    Router --> Temporal

    Audit --> Factory
    Citation --> Factory
    Summarize --> Factory
    Reconcile --> Factory
    MultiDoc --> Factory
    KG --> Factory
    CrossTenant --> Factory
    Federated --> Factory
    Temporal --> Factory

    style Client fill:#90caf9,stroke:#1565c0,color:#000
    style Router fill:#b0bec5,stroke:#546e7a,color:#000
    style Factory fill:#a5d6a7,stroke:#388e3c,color:#000
    style Audit fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Citation fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Summarize fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Reconcile fill:#ce93d8,stroke:#7b1fa2,color:#000
    style MultiDoc fill:#ce93d8,stroke:#7b1fa2,color:#000
    style KG fill:#ce93d8,stroke:#7b1fa2,color:#000
    style CrossTenant fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Federated fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Temporal fill:#ce93d8,stroke:#7b1fa2,color:#000
```

---

## DeepSynthesisWorkflow Loop

`DeepSynthesisWorkflow.run`
(`libs/agents/cogniverse_agents/deep_synthesis_workflow.py`) wraps the
orchestrator in an RLM trajectory bounded by three explicit caps: a
per-tenant sliding-window rate limit
(`DeepSynthesisRateLimiter.try_acquire`), a cumulative
`hard_call_cap` over `subagent_calls_made + llm_calls_used`, and a
`max_iterations` ceiling. Each RLM step either contains the `SUBMIT()`
token (answer ready) or emits `ASK(<subagent>: <subquery>)` markers
that get parsed and fanned out, capped per round by
`max_subagent_calls_per_round`.

```mermaid
flowchart TD
    Start["<span style='color:#000'>run(query, tenant_id, seed_subagents)</span>"]
    Rate{"<span style='color:#000'>DeepSynthesisRateLimiter.try_acquire(tenant_id)<br/>(rate_limit_per_hour, sliding window)</span>"}
    RateOut["<span style='color:#000'>return DeepSynthesisResult(<br/>was_rate_limited=True, answer='')</span>"]

    Seed["<span style='color:#000'>Iter 0 fan-out:<br/>seed_subagents[:max_subagent_calls_per_round]<br/>parallel _fan_out -> gathered[]</span>"]

    Loop["<span style='color:#000'>for it in 1..max_iterations</span>"]
    HardCap{"<span style='color:#000'>subagent_calls_made + llm_calls_used<br/>>= hard_call_cap?</span>"}
    Capped["<span style='color:#000'>return DeepSynthesisResult(<br/>was_capped=True, was_submitted=False)</span>"]

    RLMStep["<span style='color:#000'>rlm.process(query, context=gathered)<br/>llm_calls_used += 1</span>"]
    Submit{"<span style='color:#000'>SUBMIT() in iter_text?</span>"}
    Submitted["<span style='color:#000'>return DeepSynthesisResult(<br/>was_submitted=True, answer)</span>"]

    Asks{"<span style='color:#000'>_parse_asks(iter_text)<br/>yields ASK(name: subq) pairs?</span>"}
    Stalled["<span style='color:#000'>trajectory: stalled_no_asks<br/>return was_capped=True</span>"]

    FanOut["<span style='color:#000'>dispatch asks[:max_subagent_calls_per_round]<br/>(further trimmed by remaining cap budget)<br/>subagent_calls_made += len(results)</span>"]

    IterCap{"<span style='color:#000'>iterations exhausted<br/>without SUBMIT?</span>"}
    IterExhausted["<span style='color:#000'>trajectory: iteration_cap_exhausted<br/>return was_capped=True, was_submitted=False</span>"]

    Start --> Rate
    Rate -->|denied| RateOut
    Rate -->|admitted| Seed
    Seed --> Loop
    Loop --> HardCap
    HardCap -->|yes| Capped
    HardCap -->|no| RLMStep
    RLMStep --> Submit
    Submit -->|yes| Submitted
    Submit -->|no| Asks
    Asks -->|none| Stalled
    Asks -->|>=1| FanOut
    FanOut --> IterCap
    IterCap -->|yes| IterExhausted
    IterCap -->|no| Loop

    style Start fill:#90caf9,stroke:#1565c0,color:#000
    style Rate fill:#ffcc80,stroke:#ef6c00,color:#000
    style RateOut fill:#e57373,stroke:#c62828,color:#000
    style Seed fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Loop fill:#b0bec5,stroke:#546e7a,color:#000
    style HardCap fill:#ffcc80,stroke:#ef6c00,color:#000
    style Capped fill:#e57373,stroke:#c62828,color:#000
    style RLMStep fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Submit fill:#ffcc80,stroke:#ef6c00,color:#000
    style Submitted fill:#a5d6a7,stroke:#388e3c,color:#000
    style Asks fill:#ffcc80,stroke:#ef6c00,color:#000
    style Stalled fill:#e57373,stroke:#c62828,color:#000
    style FanOut fill:#ce93d8,stroke:#7b1fa2,color:#000
    style IterCap fill:#ffcc80,stroke:#ef6c00,color:#000
    style IterExhausted fill:#e57373,stroke:#c62828,color:#000
```

---

## Sandbox Boot Policy Decision

`SandboxManager.__init__`
(`libs/runtime/cogniverse_runtime/sandbox_manager.py`) resolves a
`SandboxPolicy` (with the deprecated `enabled` kwarg mapped to
`OPTIONAL`/`DISABLED`) and either short-circuits on `DISABLED` or
calls `_connect()`, which TCP-probes
`OPENSHELL_GATEWAY_ENDPOINT` via `_probe_gateway_endpoint`. When the
resolved policy is `REQUIRED` and `_available` is still False after
the probe, the constructor raises `SandboxGatewayUnavailableError` so
boot fails loud.

```mermaid
flowchart TD
    Init["<span style='color:#000'>SandboxManager(policy=?, enabled=?)</span>"]
    Resolve["<span style='color:#000'>_resolve_policy(policy, enabled)<br/>policy wins; else enabled True->OPTIONAL,<br/>False->DISABLED; default OPTIONAL</span>"]

    Decision{"<span style='color:#000'>SandboxPolicy</span>"}

    Disabled["<span style='color:#000'>DISABLED branch<br/>log 'disabled by configuration'<br/>_client=None, _available=False<br/>return (skip probe + policy load)</span>"]

    LoadPolicies["<span style='color:#000'>_load_policies()<br/>read configs/agent_policies/*.yaml</span>"]
    Connect["<span style='color:#000'>_connect()<br/>if OPENSHELL_GATEWAY_ENDPOINT:<br/>  _probe_gateway_endpoint(endpoint)<br/>  SandboxClient(endpoint=...)<br/>else:<br/>  SandboxClient.from_active_cluster()</span>"]
    Available{"<span style='color:#000'>connect succeeded?<br/>(probe + client construct)</span>"}

    Optional["<span style='color:#000'>OPTIONAL + available=False<br/>log warning 'will execute without sandbox'<br/>continue boot</span>"]
    OptionalOK["<span style='color:#000'>OPTIONAL + available=True<br/>continue boot</span>"]
    Required["<span style='color:#000'>REQUIRED + available=False<br/>raise SandboxGatewayUnavailableError<br/>(refuse to start)</span>"]
    RequiredOK["<span style='color:#000'>REQUIRED + available=True<br/>continue boot</span>"]

    Init --> Resolve --> Decision
    Decision -->|DISABLED| Disabled
    Decision -->|OPTIONAL| LoadPolicies
    Decision -->|REQUIRED| LoadPolicies
    LoadPolicies --> Connect --> Available
    Available -->|no, policy=OPTIONAL| Optional
    Available -->|yes, policy=OPTIONAL| OptionalOK
    Available -->|no, policy=REQUIRED| Required
    Available -->|yes, policy=REQUIRED| RequiredOK

    style Init fill:#90caf9,stroke:#1565c0,color:#000
    style Resolve fill:#b0bec5,stroke:#546e7a,color:#000
    style Decision fill:#ffcc80,stroke:#ef6c00,color:#000
    style Disabled fill:#b0bec5,stroke:#546e7a,color:#000
    style LoadPolicies fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Connect fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Available fill:#ffcc80,stroke:#ef6c00,color:#000
    style Optional fill:#ffcc80,stroke:#ef6c00,color:#000
    style OptionalOK fill:#a5d6a7,stroke:#388e3c,color:#000
    style Required fill:#e57373,stroke:#c62828,color:#000
    style RequiredOK fill:#a5d6a7,stroke:#388e3c,color:#000
```

---

## Optimizer Canary FSM

`ArtifactManager`
(`libs/agents/cogniverse_agents/optimizer/artifact_manager.py`)
stores a per-`(tenant, agent)` JSON state blob with `active`, `canary`,
and `retired[]` slots. `promote_to_canary`, `promote_canary_to_active`,
`retire_canary`, and `rollback_to_version` mutate the slots and snapshot
the relevant versions; `load_for_request` then routes each call to the
canary or active arm using a stable
`sha1(request_seed) % 100 < traffic_pct` decision.

```mermaid
stateDiagram-v2
    [*] --> active : promote (first version becomes active)

    state "active only" as active
    state "active + canary[traffic_pct]" as canary
    state "retired (audit list)" as retired

    active --> canary : promote_to_canary(version, traffic_pct)
    canary --> canary : promote_to_canary(new_version, pct)\nprev canary -> retired\n(reason=superseded_by_new_canary)

    canary --> active : promote_canary_to_active()\nprev active -> retired\n(reason=superseded_by_canary_promotion)\n_restore_active_from_version(canary.version)

    canary --> active : retire_canary(reason=metric_regression\n| manual_retire | ...)\ncanary -> retired (caller reason)

    active --> active : rollback_to_version(prompts_version, demos_version)\nsnapshot_active() -> backup_versions\n_restore_active_from_version(target)

    canary --> retired : (retired list grows)
    active --> retired : (when canary promoted)

    note right of canary
        load_for_request(request_seed, variant_id):
          bucket = int(sha1(request_seed)[:8], 16) % 100
          if canary and bucket < canary.traffic_pct:
              served_from = "canary" (version = canary.version)
          else:
              served_from = "active" (version = active.version)
        Same request_seed always lands in the same arm.
    end note
```

---

## Daily-Cleanup Workflow + Monthly-Reports Upload

`optimization_cli.run_cleanup`
(`libs/runtime/cogniverse_runtime/optimization_cli.py`) runs as a
single Argo CronWorkflow pod (`{fullname}-daily-cleanup`,
`charts/cogniverse/templates/optimization-workflows.yaml`) that
executes four sequential sections per tick: schema-driven per-tenant
Mem0 cleanup, log rotation, temp purge, and config-store version
vacuum. `run_monthly_reports` runs as a two-step Argo pipeline sharing
a `reports-stage` PVC: the runtime image generates the JSON, then a
`minio/mc:latest` pod uploads it to the `cogniverse-backups` bucket
using credentials from the `{fullname}-minio` Secret.

```mermaid
flowchart TD
    Cron["<span style='color:#000'>CronWorkflow {fullname}-daily-cleanup<br/>schedule (UTC), Forbid concurrency<br/>image=runtime, args=[--mode cleanup,<br/>--log-retention-days, --memory-retention-days]</span>"]

    Mem["<span style='color:#000'>1) Memory cleanup (per tenant)<br/>list_organizations_internal +<br/>list_tenants_for_org_internal<br/>for each tid:<br/>  Mem0MemoryManager(tid)<br/>  lazy_init_memory(mm, tid, cfg)<br/>  mm.cleanup_with_schema(build_default_registry())</span>"]

    Logs["<span style='color:#000'>2) Log rotation<br/>LOG_DIR (env, default /logs)<br/>_prune_aged_files(older_than_days=<br/>--log-retention-days)</span>"]

    Temp["<span style='color:#000'>3) Temp purge<br/>TEMP_DIR (env, default /tmp)<br/>TEMP_RETENTION_DAYS (env, default 1)<br/>_prune_aged_files(temp_dir, temp_age_days)</span>"]

    Vacuum["<span style='color:#000'>4) Config vacuum<br/>CONFIG_KEEP_VERSIONS (env, default 10)<br/>VespaConfigStore.prune_all_configs(<br/>keep=keep_versions)</span>"]

    Result["<span style='color:#000'>results dict:<br/>memory_cleanup, tenants_processed,<br/>log_cleanup, temp_cleanup, config_vacuum</span>"]

    Cron --> Mem --> Logs --> Temp --> Vacuum --> Result

    style Cron fill:#90caf9,stroke:#1565c0,color:#000
    style Mem fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Logs fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Temp fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Vacuum fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Result fill:#a5d6a7,stroke:#388e3c,color:#000
```

```mermaid
flowchart LR
    Schedule["<span style='color:#000'>CronWorkflow {fullname}-monthly-reports<br/>schedule (UTC), Forbid concurrency</span>"]

    subgraph Pipeline["<span style='color:#000'>reports-pipeline (sequential steps, shared PVC reports-stage)</span>"]
        Generate["<span style='color:#000'>Step 1: generate-reports pod<br/>image=runtime<br/>python -m cogniverse_runtime.optimization_cli<br/>--mode monthly-reports<br/>--reports-output-dir /reports<br/>--lookback-hours {value}<br/>writes usage-YYYYMM.json,<br/>performance-YYYYMM.json</span>"]
        Upload["<span style='color:#000'>Step 2: upload-reports pod<br/>image=minio/mc:latest<br/>env MINIO_ENDPOINT, MINIO_BUCKET,<br/>REPORTS_PREFIX,<br/>MINIO_ACCESS_KEY (secret rootUser),<br/>MINIO_SECRET_KEY (secret rootPassword)<br/>mc alias set dest $ENDPOINT $AK $SK<br/>mc cp /reports/*.json<br/>dest/$BUCKET/$REPORTS_PREFIX/</span>"]
    end

    PVC["<span style='color:#000'>PVC reports-stage<br/>ReadWriteOnce, 1Gi<br/>mounted at /reports in both pods</span>"]

    Secret["<span style='color:#000'>Secret {fullname}-minio<br/>keys: rootUser, rootPassword</span>"]

    MinIO["<span style='color:#000'>MinIO bucket cogniverse-backups<br/>(hostStorage.backup.bucket default)<br/>prefix REPORTS_PREFIX</span>"]

    Schedule --> Generate
    Generate -->|writes /reports/*.json| PVC
    PVC -->|reads /reports/*.json| Upload
    Secret --> Upload
    Upload --> MinIO

    style Schedule fill:#90caf9,stroke:#1565c0,color:#000
    style Generate fill:#ce93d8,stroke:#7b1fa2,color:#000
    style Upload fill:#ce93d8,stroke:#7b1fa2,color:#000
    style PVC fill:#b0bec5,stroke:#546e7a,color:#000
    style Secret fill:#ffcc80,stroke:#ef6c00,color:#000
    style MinIO fill:#a5d6a7,stroke:#388e3c,color:#000
```

---
