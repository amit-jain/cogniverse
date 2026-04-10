# Unified Optimization Pipeline — Design Spec

## Goal

Unify the optimization feedback loop so agents emit spans, QualityMonitor evaluates and triggers optimization, Argo workflows run batch jobs (production spans + approved synthetic data), artifacts are saved, and agents load them at startup. Replace the old scattered optimization paths with a single coherent pipeline.

## Architecture

```
AGENTS (runtime):
  Process queries → emit spans to Phoenix
  On startup: load latest artifacts from ArtifactManager (fallback to defaults)

SYNTHETIC DATA (separate Argo workflow):
  Generate synthetic data per optimizer type → save with approval status
  Human reviews in Dashboard (approval_queue_tab.py) → approves or rejects
  High-confidence items auto-approved via SyntheticDataConfidenceExtractor
  Approved datasets sit in Phoenix until consumed by optimization

EVALUATION (daily + reactive):
  QualityMonitor scores spans against golden dataset
  Scheduled distillation daily (strategies → Mem0)
  On degradation: trigger targeted optimization for affected agents

OPTIMIZATION (weekly parallel + daily gateway + reactive):
  Weekly CronWorkflow: all 6 batch jobs in parallel (then workflow sequential)
  Daily CronWorkflow: gateway-thresholds only (cheap, no LLM)
  Reactive: QualityMonitor triggers batch job for degraded agent only
  Each batch job: production spans + approved synthetic data → compile → save artifact
  After optimization: rolling restart of runtime pods to pick up new artifacts
```

## Tech Stack

- **Argo Workflows**: CronWorkflows for scheduled optimization, Workflows for reactive triggers
- **Phoenix**: Span storage, artifact storage (via ArtifactManager), synthetic dataset storage
- **DSPy**: BootstrapFewShot/GEPA for module compilation
- **SyntheticDataService**: Generates training data per optimizer type
- **Approval system**: `HumanApprovalAgent`, `ApprovalStatus`, `SyntheticDataConfidenceExtractor`
- **Dashboard**: `approval_queue_tab.py` for human review (already exists)

---

## 1. Batch Jobs — Agent-to-Job Mapping

| Job CLI Mode | Agent | Schedule | What it optimizes | Span source | Artifact key |
|---|---|---|---|---|---|
| `gateway-thresholds` | GatewayAgent | Daily | GLiNER confidence thresholds | `cogniverse.gateway` | `config/gateway_thresholds` |
| `entity-extraction` | EntityExtractionAgent | Weekly | DSPy fallback module + path thresholds | `cogniverse.entity_extraction` | `model/entity_extraction` |
| `simba` | QueryEnhancementAgent | Weekly | DSPy enhancement module | `cogniverse.query_enhancement` | `model/simba_query_enhancement` |
| `routing` | RoutingAgent | Weekly | DSPy routing decision module | `cogniverse.routing` | `model/routing_decision` |
| `profile` | ProfileSelectionAgent | Weekly | DSPy profile selection module | `cogniverse.profile_selection` | `model/profile_selection` |
| `workflow` | OrchestratorAgent | Weekly | Workflow templates + agent performance profiles | `cogniverse.orchestration` | `demos/workflow` + `config/workflow_templates` |

## 2. Each Batch Job — Execution Flow

Every batch job follows the same pattern:

```
1. Read production spans from Phoenix
   - Query cogniverse.{span_type} spans for tenant
   - Filter by lookback window (default 48h for weekly, 24h for daily)

2. Load approved synthetic datasets from Phoenix
   - Query datasets matching optimizer type + tenant
   - Filter by ApprovalStatus: APPROVED or AUTO_APPROVED only
   - Skip if none available (synthetic is optional, not blocking)

3. Merge production + synthetic into training set
   - Production spans converted to DSPy Examples
   - Synthetic data converted to DSPy Examples
   - Combined into single trainset

4. Compile
   - DSPy modules: BootstrapFewShot (< 50 examples) or GEPA (>= 50)
   - Gateway thresholds: statistical analysis of confidence scores
   - Workflow: template extraction + agent performance profiling

5. Save artifact via ArtifactManager
   - model blobs: ArtifactManager.save_blob(kind, key, serialized_module)
   - config blobs: ArtifactManager.save_blob("config", key, json_config)
   - demonstrations: ArtifactManager.save_demonstrations(agent_type, demos)
```

## 3. Agent Artifact Loading at Startup

Each agent's `__init__` loads the latest optimized artifact. If no artifact exists (first run, new tenant), the agent uses its default module.

**GatewayAgent** — loads threshold config:
```python
blob = await artifact_manager.load_blob("config", "gateway_thresholds")
if blob:
    config = json.loads(blob)
    self.deps.fast_path_confidence_threshold = config["fast_path_confidence_threshold"]
    if "gliner_threshold" in config:
        self.deps.gliner_threshold = config["gliner_threshold"]
```

**QueryEnhancementAgent, RoutingAgent, ProfileSelectionAgent, EntityExtractionAgent** — load DSPy module:
```python
blob = await artifact_manager.load_blob("model", artifact_key)
if blob:
    self.dspy_module.load_state(json.loads(blob))
    logger.info(f"Loaded optimized DSPy module from artifact: {artifact_key}")
```

**OrchestratorAgent** — loads workflow templates:
```python
demos = await artifact_manager.load_demonstrations("workflow")
if demos:
    self.workflow_intelligence.load_templates_from_demos(demos)
```

**Telemetry provider access**: Agents need a `TelemetryProvider` to create `ArtifactManager`. The dispatcher injects `TelemetryManager` into each agent; the agent gets a provider via `telemetry_manager.get_provider(tenant_id)`.

**Async in __init__**: `load_blob` is async. Agents can either:
- Use `asyncio.run()` in `__init__` (blocking, simple)
- Defer to first request (lazy load)
- Load in the FastAPI lifespan handler (for standalone deployment)

## 4. Synthetic Data Flow

```
Synthetic Data CronWorkflow (weekly, runs BEFORE optimization):
  For each optimizer type:
    1. Call SyntheticDataService.generate(optimizer=type, count=100, tenant_id=tenant)
    2. Run through SyntheticDataConfidenceExtractor
    3. High confidence → AUTO_APPROVED, low confidence → PENDING_REVIEW
    4. Save to Phoenix with approval status

Human Review (Dashboard — approval_queue_tab.py):
  - Shows PENDING_REVIEW synthetic items
  - Human reviews, approves or rejects each
  - Approved items available for next optimization run

Optimization Batch Job:
  - Queries approved synthetic datasets:
    ApprovalStatus.APPROVED or ApprovalStatus.AUTO_APPROVED
  - Merges with production spans
  - If no approved synthetic data exists, runs with production spans only
```

## 5. Argo Workflow Structure

### Weekly Optimization CronWorkflow

```yaml
schedule: "0 3 * * 0"  # Sunday 3 AM UTC
steps:
  # Parallel: all independent agent optimizations
  - - name: optimize-gateway
      template: run-optimizer
      arguments: {mode: gateway-thresholds}
    - name: optimize-entity-extraction
      template: run-optimizer
      arguments: {mode: entity-extraction}
    - name: optimize-simba
      template: run-optimizer
      arguments: {mode: simba}
    - name: optimize-routing
      template: run-optimizer
      arguments: {mode: routing}
    - name: optimize-profile
      template: run-optimizer
      arguments: {mode: profile}
  # Sequential: workflow may use other agents' artifacts
  - - name: optimize-workflow
      template: run-optimizer
      arguments: {mode: workflow}
  # Restart pods to pick up new artifacts
  - - name: restart-runtime
      template: restart-deployment
```

### Daily Gateway Thresholds CronWorkflow

```yaml
schedule: "0 4 * * *"  # Daily 4 AM UTC
steps:
  - - name: optimize-gateway
      template: run-optimizer
      arguments: {mode: gateway-thresholds}
  - - name: restart-runtime
      template: restart-deployment
```

### Synthetic Data Generation CronWorkflow

```yaml
schedule: "0 1 * * 6"  # Saturday 1 AM UTC (before Sunday optimization)
steps:
  - - name: generate-routing
      template: generate-synthetic
      arguments: {optimizer: routing, count: 100}
    - name: generate-enhancement
      template: generate-synthetic
      arguments: {optimizer: query_enhancement, count: 100}
    - name: generate-profile
      template: generate-synthetic
      arguments: {optimizer: profile_selection, count: 100}
    - name: generate-workflow
      template: generate-synthetic
      arguments: {optimizer: workflow, count: 100}
```

### Reactive Optimization (QualityMonitor triggered)

QualityMonitor detects degradation for specific agents → submits Argo Workflow:
```yaml
# Dynamic — only runs the degraded agent's batch job
steps:
  - - name: optimize-affected-agent
      template: run-optimizer
      arguments: {mode: <affected-agent-mode>}
  - - name: restart-runtime
      template: restart-deployment
```

## 6. Trigger Model

Two paths, both feeding the same batch jobs:

**Scheduled (CronWorkflows):**
- Daily: gateway-thresholds
- Weekly: all 6 batch jobs (parallel + workflow sequential)
- Weekly: synthetic data generation (Saturday, before Sunday optimization)
- Daily: strategy distillation via `quality_monitor_cli --once`

**Reactive (QualityMonitor):**
- QualityMonitor evaluates spans against golden dataset
- Uses XGBoost meta-models or heuristics to detect degradation
- On degradation: submits one-shot Argo Workflow for affected agent(s)
- Also runs StrategyLearner to distill strategies into Mem0

Both paths produce artifacts via ArtifactManager. Both trigger pod restarts.

## 7. What Gets Deleted

| File/Component | Reason |
|---|---|
| `scripts/run_module_optimization.py` | Replaced by CLI modes |
| `scripts/run_optimization.py` | Replaced by CLI modes |
| Old weekly Argo workflow steps (`optimize-modality`, `optimize-cross-modal`) | Replaced by new agent-specific jobs |
| Old `--mode dspy` in optimization_cli.py | Replaced by `--mode routing` (reads spans, not internal data) |
| Old `--mode once`/`--mode full` | Replaced by running individual CLI modes from Argo |
| `XGBoostMetaModels` references to `QueryModality` | Updated for new agent types |

## 8. What Gets Added

| Component | Description |
|---|---|
| `--mode entity-extraction` CLI mode | Compiles EntityExtractionAgent's DSPy module from spans |
| `--mode routing` CLI mode | Compiles RoutingAgent's DSPy module from spans (replaces `--mode dspy`) |
| Synthetic data merge in each batch job | Each job loads approved synthetic datasets + production spans |
| Artifact loading in 6 agent `__init__` methods | Load optimized module/config from ArtifactManager at startup |
| Updated Argo CronWorkflow manifests | New weekly parallel workflow, daily gateway, synthetic data generation |
| Pod restart step in Argo workflows | `kubectl rollout restart` after optimization |
| Dataset approval status filtering | Batch jobs query only APPROVED/AUTO_APPROVED synthetic datasets |

## 9. What Stays the Same

- QualityMonitor evaluation logic (span scoring, golden dataset comparison)
- StrategyLearner distillation (traces → strategies → Mem0)
- ArtifactManager storage/retrieval API
- Scheduled distillation via `quality_monitor_cli --once`
- Approval system (`HumanApprovalAgent`, `ApprovalStatus`, dashboard UI)
- `SyntheticDataService` and `SyntheticDataConfidenceExtractor`

## 10. Testing

- **E2E**: Fixture generates 100+ spans per agent → runs all 6 batch jobs → verifies artifacts have learned demos with correct content → verifies agents can load artifacts
- **Integration**: Agent with loaded artifact produces output using optimized module (not default)
- **Synthetic merge**: Batch job with both production spans and approved synthetic data produces artifact with demos from both sources
- **Approval filtering**: Batch job ignores PENDING_REVIEW and REJECTED datasets
- **Argo**: Helm template rendering validates workflow manifests
- **Artifact loading**: Agent without artifact uses default module; agent with artifact uses optimized module
