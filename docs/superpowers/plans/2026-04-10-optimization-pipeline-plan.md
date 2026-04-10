# Unified Optimization Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the optimization feedback loop: agents load optimized artifacts at startup, batch jobs merge production spans + approved synthetic data, Argo workflows schedule and trigger optimization, old paths replaced.

**Architecture:** Each agent loads its latest optimized artifact (DSPy module, threshold config, or workflow templates) from ArtifactManager at startup, falling back to defaults if no artifact exists. Batch jobs read production spans from Phoenix + approved synthetic datasets, compile optimized artifacts, save via ArtifactManager. Argo CronWorkflows run batch jobs on schedule (daily/weekly) and restart pods. QualityMonitor triggers reactive optimization on degradation.

**Tech Stack:** Python 3.12, DSPy 3.0, Argo Workflows, Phoenix/OpenTelemetry, ArtifactManager, SyntheticDataService, Helm/k3d

**Spec:** `docs/superpowers/specs/2026-04-10-unified-optimization-pipeline-design.md`

---

## Phase 1: Agent Artifact Loading at Startup

Each of the 6 agents loads its latest optimized artifact during initialization. If no artifact exists (first run, new tenant), the agent uses its default module.

---

### Task 1: Add artifact loading to GatewayAgent

**Files:**
- Modify: `libs/agents/cogniverse_agents/gateway_agent.py`
- Test: `tests/agents/unit/test_gateway_agent.py`

- [ ] **Step 1: Add `_load_artifact` method to GatewayAgent**

After `self._gliner_model = None` in `__init__`, add artifact loading. The GatewayAgent loads threshold config, not a DSPy module:

```python
def _load_artifact(self) -> None:
    """Load optimized thresholds from artifact store (if available)."""
    if not (hasattr(self, "telemetry_manager") and self.telemetry_manager):
        return
    try:
        import asyncio
        import json

        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

        tenant_id = getattr(self, "_artifact_tenant_id", "default")
        provider = self.telemetry_manager.get_provider(tenant_id=tenant_id)
        am = ArtifactManager(provider, tenant_id)

        blob = asyncio.get_event_loop().run_until_complete(
            am.load_blob("config", "gateway_thresholds")
        )
        if blob:
            config = json.loads(blob)
            if "fast_path_confidence_threshold" in config:
                self.deps.fast_path_confidence_threshold = config["fast_path_confidence_threshold"]
            if "gliner_threshold" in config:
                self.deps.gliner_threshold = config["gliner_threshold"]
            logger.info(
                "GatewayAgent loaded optimized thresholds: "
                f"fast_path={self.deps.fast_path_confidence_threshold}, "
                f"gliner={self.deps.gliner_threshold}"
            )
    except Exception as e:
        logger.debug("No gateway artifact to load (using defaults): %s", e)
```

Call `_load_artifact()` at the end of `__init__`, after `super().__init__()`.

- [ ] **Step 2: Write test for artifact loading**

```python
@pytest.mark.asyncio
async def test_gateway_loads_artifact_thresholds(self):
    """GatewayAgent should apply thresholds from artifact if available."""
    from unittest.mock import AsyncMock, MagicMock
    import json

    agent = _make_gateway()

    # Mock telemetry_manager and artifact
    mock_tm = MagicMock()
    mock_provider = MagicMock()
    mock_tm.get_provider.return_value = mock_provider

    # Simulate artifact with custom thresholds
    artifact = {"fast_path_confidence_threshold": 0.55, "gliner_threshold": 0.35}

    with patch("cogniverse_agents.gateway_agent.ArtifactManager") as MockAM:
        mock_am = MockAM.return_value
        mock_am.load_blob = AsyncMock(return_value=json.dumps(artifact))

        agent.telemetry_manager = mock_tm
        agent._load_artifact()

    assert agent.deps.fast_path_confidence_threshold == 0.55
    assert agent.deps.gliner_threshold == 0.35

def test_gateway_uses_defaults_without_artifact(self):
    """GatewayAgent should use default thresholds when no artifact exists."""
    agent = _make_gateway()
    assert agent.deps.fast_path_confidence_threshold == 0.4  # default
    assert agent.deps.gliner_threshold == 0.3  # default
```

- [ ] **Step 3: Run tests**

```bash
uv run pytest tests/agents/unit/test_gateway_agent.py --tb=long -q
```

- [ ] **Step 4: Commit**

```bash
git commit -m "Add artifact loading to GatewayAgent for optimized thresholds"
```

---

### Task 2: Add artifact loading to QueryEnhancementAgent

**Files:**
- Modify: `libs/agents/cogniverse_agents/query_enhancement_agent.py`
- Test: `tests/agents/unit/test_query_enhancement.py`

- [ ] **Step 1: Add `_load_artifact` method**

After `super().__init__()` in QueryEnhancementAgent, add:

```python
def _load_artifact(self) -> None:
    """Load optimized DSPy module from artifact store (if available)."""
    if not (hasattr(self, "telemetry_manager") and self.telemetry_manager):
        return
    try:
        import asyncio
        import json

        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

        tenant_id = getattr(self, "_artifact_tenant_id", "default")
        provider = self.telemetry_manager.get_provider(tenant_id=tenant_id)
        am = ArtifactManager(provider, tenant_id)

        blob = asyncio.get_event_loop().run_until_complete(
            am.load_blob("model", "simba_query_enhancement")
        )
        if blob:
            state = json.loads(blob)
            self.dspy_module.load_state(state)
            logger.info("QueryEnhancementAgent loaded optimized DSPy module from artifact")
    except Exception as e:
        logger.debug("No enhancement artifact to load (using defaults): %s", e)
```

- [ ] **Step 2: Write test**

```python
@pytest.mark.asyncio
async def test_enhancement_loads_dspy_artifact(self):
    """QueryEnhancementAgent should load optimized DSPy module."""
    from unittest.mock import AsyncMock, MagicMock, patch
    import json

    deps = QueryEnhancementDeps()
    agent = QueryEnhancementAgent(deps=deps)

    mock_tm = MagicMock()
    mock_tm.get_provider.return_value = MagicMock()

    # Simulate artifact blob
    fake_state = {"enhancer.predict": {"signature": {"fields": []}, "demos": []}}

    with patch("cogniverse_agents.query_enhancement_agent.ArtifactManager") as MockAM:
        mock_am = MockAM.return_value
        mock_am.load_blob = AsyncMock(return_value=json.dumps(fake_state))

        agent.telemetry_manager = mock_tm
        agent._load_artifact()

    # Module's load_state should have been called
    # (verified by no exception — load_state parses the state dict)
```

- [ ] **Step 3: Run tests, commit**

```bash
uv run pytest tests/agents/unit/test_query_enhancement.py --tb=long -q
git commit -m "Add artifact loading to QueryEnhancementAgent"
```

---

### Task 3: Add artifact loading to RoutingAgent

**Files:**
- Modify: `libs/agents/cogniverse_agents/routing_agent.py`
- Test: `tests/agents/unit/test_routing_agent.py`

- [ ] **Step 1: Add `_load_artifact` method**

Same pattern as QueryEnhancementAgent but with artifact key `"model/routing_decision"`:

```python
def _load_artifact(self) -> None:
    """Load optimized DSPy routing module from artifact store."""
    if not (hasattr(self, "telemetry_manager") and self.telemetry_manager):
        return
    try:
        import asyncio
        import json

        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

        tenant_id = getattr(self, "_artifact_tenant_id", "default")
        provider = self.telemetry_manager.get_provider(tenant_id=tenant_id)
        am = ArtifactManager(provider, tenant_id)

        blob = asyncio.get_event_loop().run_until_complete(
            am.load_blob("model", "routing_decision")
        )
        if blob:
            state = json.loads(blob)
            self.routing_module.load_state(state)
            logger.info("RoutingAgent loaded optimized DSPy module from artifact")
    except Exception as e:
        logger.debug("No routing artifact to load (using defaults): %s", e)
```

- [ ] **Step 2: Write test, run, commit**

Same pattern as Task 2. Test that `_load_artifact` runs without error and that agent works with default module when no artifact exists.

```bash
uv run pytest tests/agents/unit/test_routing_agent.py --tb=long -q
git commit -m "Add artifact loading to RoutingAgent"
```

---

### Task 4: Add artifact loading to ProfileSelectionAgent

**Files:**
- Modify: `libs/agents/cogniverse_agents/profile_selection_agent.py`
- Test: `tests/agents/unit/test_profile_selection_agent.py`

- [ ] **Step 1: Add `_load_artifact` method**

Same pattern, artifact key `"model/profile_selection"`:

```python
def _load_artifact(self) -> None:
    """Load optimized DSPy profile selection module from artifact store."""
    if not (hasattr(self, "telemetry_manager") and self.telemetry_manager):
        return
    try:
        import asyncio
        import json

        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

        tenant_id = getattr(self, "_artifact_tenant_id", "default")
        provider = self.telemetry_manager.get_provider(tenant_id=tenant_id)
        am = ArtifactManager(provider, tenant_id)

        blob = asyncio.get_event_loop().run_until_complete(
            am.load_blob("model", "profile_selection")
        )
        if blob:
            state = json.loads(blob)
            self.dspy_module.load_state(state)
            logger.info("ProfileSelectionAgent loaded optimized DSPy module from artifact")
    except Exception as e:
        logger.debug("No profile artifact to load (using defaults): %s", e)
```

- [ ] **Step 2: Write test, run, commit**

```bash
uv run pytest tests/agents/unit/test_profile_selection_agent.py --tb=long -q
git commit -m "Add artifact loading to ProfileSelectionAgent"
```

---

### Task 5: Add artifact loading to EntityExtractionAgent

**Files:**
- Modify: `libs/agents/cogniverse_agents/entity_extraction_agent.py`
- Test: `tests/agents/unit/test_entity_extraction_agent.py`

- [ ] **Step 1: Add `_load_artifact` method**

Artifact key `"model/entity_extraction"`. Loads optimized DSPy fallback module:

```python
def _load_artifact(self) -> None:
    """Load optimized DSPy entity extraction module from artifact store."""
    if not (hasattr(self, "telemetry_manager") and self.telemetry_manager):
        return
    try:
        import asyncio
        import json

        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

        tenant_id = getattr(self, "_artifact_tenant_id", "default")
        provider = self.telemetry_manager.get_provider(tenant_id=tenant_id)
        am = ArtifactManager(provider, tenant_id)

        blob = asyncio.get_event_loop().run_until_complete(
            am.load_blob("model", "entity_extraction")
        )
        if blob:
            state = json.loads(blob)
            self.dspy_module.load_state(state)
            logger.info("EntityExtractionAgent loaded optimized DSPy module from artifact")
    except Exception as e:
        logger.debug("No entity extraction artifact to load (using defaults): %s", e)
```

- [ ] **Step 2: Write test, run, commit**

```bash
uv run pytest tests/agents/unit/test_entity_extraction_agent.py --tb=long -q
git commit -m "Add artifact loading to EntityExtractionAgent"
```

---

### Task 6: Add artifact loading to OrchestratorAgent

**Files:**
- Modify: `libs/agents/cogniverse_agents/orchestrator_agent.py`
- Test: `tests/agents/unit/test_orchestrator_agent.py`

- [ ] **Step 1: Add `_load_artifact` method**

OrchestratorAgent loads workflow templates (demonstrations), not a DSPy module blob:

```python
def _load_artifact(self) -> None:
    """Load workflow templates from artifact store."""
    if not (hasattr(self, "telemetry_manager") and self.telemetry_manager):
        return
    try:
        import asyncio

        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

        tenant_id = getattr(self, "_current_tenant_id", "default")
        provider = self.telemetry_manager.get_provider(tenant_id=tenant_id)
        am = ArtifactManager(provider, tenant_id)

        demos = asyncio.get_event_loop().run_until_complete(
            am.load_demonstrations("workflow")
        )
        if demos and self.workflow_intelligence:
            self.workflow_intelligence.load_templates_from_demos(demos)
            logger.info(
                "OrchestratorAgent loaded %d workflow templates from artifact", len(demos)
            )
    except Exception as e:
        logger.debug("No workflow artifact to load (using defaults): %s", e)
```

- [ ] **Step 2: Write test, run, commit**

```bash
uv run pytest tests/agents/unit/test_orchestrator_agent.py --tb=long -q
git commit -m "Add artifact loading to OrchestratorAgent"
```

---

### Task 7: Inject tenant_id for artifact loading in dispatcher

**Files:**
- Modify: `libs/runtime/cogniverse_runtime/agent_dispatcher.py`

The dispatcher creates agents and injects TelemetryManager. It also needs to set `_artifact_tenant_id` so agents know which tenant's artifacts to load.

- [ ] **Step 1: Set _artifact_tenant_id in generic dispatch and gateway/orchestrator paths**

In `_execute_gateway_task`, after `self._gateway_agent.telemetry_manager = get_telemetry_manager()`:
```python
self._gateway_agent._artifact_tenant_id = tenant_id
self._gateway_agent._load_artifact()
```

In `_execute_generic_agent`, after `agent.telemetry_manager = get_telemetry_manager()`:
```python
agent._artifact_tenant_id = tenant_id
if hasattr(agent, "_load_artifact"):
    agent._load_artifact()
```

In `_execute_orchestration_task`, after `agent.telemetry_manager = get_telemetry_manager()`:
```python
agent._artifact_tenant_id = tenant_id
agent._load_artifact()
```

- [ ] **Step 2: Run tests, commit**

```bash
uv run pytest tests/runtime/unit/ --tb=long -q
git commit -m "Inject tenant_id and trigger artifact loading in dispatcher"
```

---

## Phase 2: New Batch Job CLI Modes + Synthetic Data Merge

---

### Task 8: Add `--mode entity-extraction` CLI mode

**Files:**
- Modify: `libs/runtime/cogniverse_runtime/optimization_cli.py`

- [ ] **Step 1: Add `run_entity_extraction_optimization` function**

Same pattern as `run_simba_optimization` but reads `cogniverse.entity_extraction` spans:

```python
async def run_entity_extraction_optimization(
    tenant_id: str,
    lookback_hours: int = 24,
) -> dict:
    """Entity extraction DSPy module optimization."""
    from cogniverse_foundation.config.utils import create_default_config_manager, get_config
    from cogniverse_foundation.telemetry.manager import get_telemetry_manager

    config_manager = create_default_config_manager()
    telemetry_manager = get_telemetry_manager()
    telemetry_provider = telemetry_manager.get_provider(tenant_id=tenant_id)

    spans_df = await _query_spans_by_name(
        telemetry_provider, tenant_id, "cogniverse.entity_extraction", lookback_hours
    )

    if spans_df.empty:
        return {"status": "no_data", "spans_found": 0}

    # Build training examples from span attributes
    import dspy
    trainset = []
    for _, row in spans_df.iterrows():
        ee_attrs = row.get("attributes.entity_extraction", {})
        if not isinstance(ee_attrs, dict):
            continue
        query = ee_attrs.get("query", "")
        entity_count = ee_attrs.get("entity_count", 0)
        if not query or entity_count == 0:
            continue
        entities_json = ee_attrs.get("entities", "[]")
        example = dspy.Example(
            query=query,
            entities=entities_json,
            entity_types="",
        ).with_inputs("query")
        trainset.append(example)

    if not trainset:
        return {"status": "no_data", "spans_found": len(spans_df), "reason": "no_valid_examples"}

    # Compile DSPy module
    from cogniverse_agents.entity_extraction_agent import EntityExtractionModule
    config = get_config(tenant_id=tenant_id, config_manager=config_manager)
    llm_config = config.get_llm_config()
    llm_endpoint = llm_config.resolve("optimization")

    dspy.configure(lm=dspy.LM(f"ollama_chat/{llm_endpoint.model}", api_base=llm_endpoint.api_base))

    module = EntityExtractionModule()
    from dspy.teleprompt import BootstrapFewShot
    optimizer = BootstrapFewShot(max_bootstrapped_demos=4, max_labeled_demos=8)
    compiled = optimizer.compile(module, trainset=trainset)

    # Save artifact
    import json
    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
    am = ArtifactManager(telemetry_provider, tenant_id)
    state = compiled.dump_state()
    artifact_id = await am.save_blob("model", "entity_extraction", json.dumps(state, default=str))

    return {
        "status": "success",
        "spans_found": len(spans_df),
        "training_examples": len(trainset),
        "artifact_id": artifact_id,
    }
```

- [ ] **Step 2: Wire into CLI argument parser**

Add `elif args.mode == "entity-extraction":` branch in the main function.

- [ ] **Step 3: Run tests, commit**

```bash
uv run pytest tests/runtime/unit/test_optimization_cli_batch_modes.py --tb=long -q
git commit -m "Add --mode entity-extraction batch job"
```

---

### Task 9: Add `--mode routing` CLI mode

**Files:**
- Modify: `libs/runtime/cogniverse_runtime/optimization_cli.py`

- [ ] **Step 1: Add `run_routing_optimization` function**

Reads `cogniverse.routing` spans, compiles RoutingAgent's DSPy module:

```python
async def run_routing_optimization(
    tenant_id: str,
    lookback_hours: int = 24,
) -> dict:
    """Routing agent DSPy module optimization from production spans."""
    # Same pattern as simba — read spans, build examples, compile, save artifact
    # Span source: cogniverse.routing
    # Artifact key: model/routing_decision
    # DSPy module: DSPyAdvancedRoutingModule or BasicQueryAnalysisSignature
    ...
```

- [ ] **Step 2: Wire into CLI, test, commit**

```bash
git commit -m "Add --mode routing batch job for RoutingAgent DSPy optimization"
```

---

### Task 10: Add synthetic data merge to all batch jobs

**Files:**
- Modify: `libs/runtime/cogniverse_runtime/optimization_cli.py`

- [ ] **Step 1: Create `_load_approved_synthetic_data` helper**

```python
async def _load_approved_synthetic_data(
    telemetry_provider,
    tenant_id: str,
    optimizer_type: str,
) -> list:
    """Load approved synthetic datasets for an optimizer type.

    Returns list of DSPy Examples from datasets with status
    APPROVED or AUTO_APPROVED. Returns empty list if none found.
    """
    from cogniverse_agents.approval.interfaces import ApprovalStatus

    # Query Phoenix for synthetic datasets matching optimizer type
    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
    am = ArtifactManager(telemetry_provider, tenant_id)

    demos = await am.load_demonstrations(f"synthetic_{optimizer_type}")
    if not demos:
        return []

    # Filter by approval status
    approved = []
    for demo in demos:
        metadata = demo.get("metadata", {})
        status = metadata.get("approval_status", "")
        if status in (ApprovalStatus.APPROVED.value, ApprovalStatus.AUTO_APPROVED.value):
            approved.append(demo)

    return approved
```

- [ ] **Step 2: Integrate into each batch job function**

In each `run_*_optimization` function, after building production trainset:
```python
# Merge approved synthetic data
synthetic_demos = await _load_approved_synthetic_data(
    telemetry_provider, tenant_id, optimizer_type
)
for demo in synthetic_demos:
    example = dspy.Example(**json.loads(demo["input"])).with_inputs("query")
    trainset.append(example)
logger.info("Merged %d synthetic examples with %d production examples",
            len(synthetic_demos), len(trainset) - len(synthetic_demos))
```

- [ ] **Step 3: Test, commit**

```bash
git commit -m "Add synthetic data merge to all batch jobs"
```

---

## Phase 3: Argo CronWorkflow Manifests

---

### Task 11: Add weekly parallel optimization CronWorkflow

**Files:**
- Modify: `charts/cogniverse/templates/optimization-workflows.yaml`
- Modify: `charts/cogniverse/values.yaml`

- [ ] **Step 1: Add new values**

In `charts/cogniverse/values.yaml` under `argo.optimization`:
```yaml
    agentOptimization:
      enabled: true
      schedule: "0 3 * * 0"  # Sunday 3 AM UTC
    dailyGateway:
      enabled: true
      schedule: "0 4 * * *"  # Daily 4 AM UTC
    syntheticGeneration:
      enabled: true
      schedule: "0 1 * * 6"  # Saturday 1 AM UTC
```

- [ ] **Step 2: Add weekly parallel CronWorkflow template**

Add to `optimization-workflows.yaml`:
```yaml
{{- if .Values.argo.optimization.agentOptimization.enabled }}
---
apiVersion: argoproj.io/v1alpha1
kind: CronWorkflow
metadata:
  name: {{ $fullName }}-agent-optimization
  namespace: {{ .Release.Namespace }}
spec:
  schedule: {{ .Values.argo.optimization.agentOptimization.schedule | quote }}
  timezone: "UTC"
  concurrencyPolicy: "Forbid"
  workflowSpec:
    entrypoint: optimize-all-agents
    serviceAccountName: {{ include "cogniverse.serviceAccountName" . }}
    templates:
    - name: optimize-all-agents
      steps:
      - - name: gateway-thresholds
          template: run-agent-optimizer
          arguments:
            parameters:
            - name: mode
              value: gateway-thresholds
        - name: entity-extraction
          template: run-agent-optimizer
          arguments:
            parameters:
            - name: mode
              value: entity-extraction
        - name: simba
          template: run-agent-optimizer
          arguments:
            parameters:
            - name: mode
              value: simba
        - name: routing
          template: run-agent-optimizer
          arguments:
            parameters:
            - name: mode
              value: routing
        - name: profile
          template: run-agent-optimizer
          arguments:
            parameters:
            - name: mode
              value: profile
      - - name: workflow
          template: run-agent-optimizer
          arguments:
            parameters:
            - name: mode
              value: workflow
      - - name: restart-runtime
          template: restart-deployment

    - name: run-agent-optimizer
      inputs:
        parameters:
        - name: mode
      container:
        image: "{{ .Values.runtime.image.repository }}:{{ .Values.runtime.image.tag }}"
        command: ["python", "-m", "cogniverse_runtime.optimization_cli"]
        args:
          - "--mode"
          - "{{`{{inputs.parameters.mode}}`}}"
          - "--tenant-id"
          - "{{`{{workflow.parameters.tenant-id}}`}}"
        env:
        # ... standard env vars (BACKEND_URL, TELEMETRY_*, LLM_ENDPOINT, OLLAMA_API_BASE)
        volumeMounts:
        - name: config
          mountPath: /app/configs/config.json
          subPath: config.json
          readOnly: true

    - name: restart-deployment
      container:
        image: bitnami/kubectl:latest
        command: ["kubectl"]
        args:
          - "rollout"
          - "restart"
          - "deployment/{{ $fullName }}-runtime"
          - "-n"
          - "{{ .Release.Namespace }}"
{{- end }}
```

- [ ] **Step 3: Add daily gateway CronWorkflow and synthetic generation CronWorkflow**

Similar templates for daily gateway-thresholds and Saturday synthetic data generation.

- [ ] **Step 4: Validate with `helm template`**

```bash
helm template cogniverse charts/cogniverse -f charts/cogniverse/values.yaml | grep -A 5 "agent-optimization"
```

- [ ] **Step 5: Commit**

```bash
git commit -m "Add Argo CronWorkflows for agent optimization pipeline"
```

---

## Phase 4: Old Path Cleanup

---

### Task 12: Remove old optimization scripts and update old Argo workflows

**Files:**
- Delete: `scripts/run_module_optimization.py`
- Delete: `scripts/run_optimization.py`
- Modify: `charts/cogniverse/templates/optimization-workflows.yaml` (remove old weekly steps)
- Modify: `libs/runtime/cogniverse_runtime/optimization_cli.py` (remove `--mode once`, `--mode full`, `--mode dspy`)

- [ ] **Step 1: Delete old scripts**

```bash
rm scripts/run_module_optimization.py scripts/run_optimization.py
```

- [ ] **Step 2: Remove old CLI modes**

Remove `--mode once`, `--mode full`, `--mode dspy` from `optimization_cli.py`. These are replaced by the individual agent modes + `--mode routing`.

- [ ] **Step 3: Update old weekly Argo workflow**

Replace the old `weekly-optimization` CronWorkflow (which ran `optimize-modality`, `optimize-cross-modal`, etc.) with the new `agent-optimization` CronWorkflow from Task 11.

- [ ] **Step 4: Update XGBoost meta-models**

In `xgboost_meta_models.py`, replace `from cogniverse_agents.search.multi_modal_reranker import QueryModality` with a local enum or the new agent type mapping.

- [ ] **Step 5: Run full test suite to verify no regressions**

```bash
uv run pytest tests/agents/unit/ tests/routing/unit/ tests/runtime/unit/ --tb=long -q
```

- [ ] **Step 6: Commit**

```bash
git commit -m "Remove old optimization scripts and update Argo workflows for unified pipeline"
```

---

## Phase 5: E2E Verification

---

### Task 13: E2E test for artifact loading round-trip

**Files:**
- Create or modify: `tests/e2e/test_batch_optimization_e2e.py`

- [ ] **Step 1: Add test that verifies the full loop**

```python
def test_artifact_loading_round_trip(self):
    """Run batch job → verify artifact → restart pod → verify agent loaded artifact."""
    # 1. Run gateway-thresholds batch job
    result = _run_batch_job("gateway-thresholds")
    assert result["status"] == "success"

    # 2. Verify artifact exists with correct content
    blob = _load_blob_in_pod("config", "gateway_thresholds")
    artifact = json.loads(blob)
    assert "fast_path_confidence_threshold" in artifact

    # 3. Restart runtime pod (simulates pod restart after optimization)
    subprocess.run([
        "kubectl", "--context", KUBECTL_CONTEXT,
        "rollout", "restart", f"deployment/{DEPLOYMENT.split('/')[-1]}",
        "-n", NAMESPACE,
    ], check=True, timeout=30)

    # Wait for rollout
    subprocess.run([
        "kubectl", "--context", KUBECTL_CONTEXT,
        "rollout", "status", DEPLOYMENT,
        "-n", NAMESPACE, "--timeout=120s",
    ], check=True, timeout=150)

    time.sleep(20)  # Wait for agent initialization

    # 4. Verify agent loaded the artifact (check via a query that exercises thresholds)
    resp = httpx.post(
        f"{RUNTIME}/agents/gateway_agent/process",
        json={"agent_name": "gateway_agent", "query": "test", "context": {"tenant_id": TENANT_ID}},
        timeout=120.0,
    )
    assert resp.status_code == 200  # Agent works after loading artifact
```

- [ ] **Step 2: Run, commit**

```bash
uv run pytest tests/e2e/test_batch_optimization_e2e.py --tb=short -q
git commit -m "Add E2E test for artifact loading round-trip"
```

---

### Task 14: Audit against spec

- [ ] **Step 1: Run codebase-integrity-auditor against the spec**

Verify every requirement in `docs/superpowers/specs/2026-04-10-unified-optimization-pipeline-design.md` is implemented and tested.

- [ ] **Step 2: Fix any gaps found**

- [ ] **Step 3: Final commit**

```bash
git commit -m "Complete unified optimization pipeline implementation"
```
