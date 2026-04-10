"""
E2E tests for Argo batch optimization jobs.

Tests the 4 optimization CLI modes (gateway-thresholds, workflow, simba, profile)
by running them inside the k3d pod via kubectl exec. Verifies the full loop:
spans exist in Phoenix -> batch job reads them -> produces artifact -> artifact
contains correct data -> agent can load the artifact.

Requires live k3d stack via `cogniverse up` with:
- Runtime at localhost:28000
- Phoenix at localhost:26006
- kubectl context: k3d-cogniverse
"""

import json
import subprocess

import pytest

from tests.e2e.conftest import (
    PHOENIX_URL,
    TENANT_ID,
    skip_if_no_runtime,
)

KUBECTL_CONTEXT = "k3d-cogniverse"
NAMESPACE = "cogniverse"
DEPLOYMENT = "deploy/cogniverse-runtime"
CONTAINER = "runtime"
LOOKBACK_HOURS = 48


def _kubectl_available() -> bool:
    """Check if kubectl can reach the k3d cluster."""
    try:
        result = subprocess.run(
            [
                "kubectl", "--context", KUBECTL_CONTEXT,
                "get", "ns", NAMESPACE,
            ],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


skip_if_no_kubectl = pytest.mark.skipif(
    not _kubectl_available(),
    reason=f"kubectl cannot reach {KUBECTL_CONTEXT} cluster",
)


def _run_batch_job(
    mode: str,
    tenant_id: str = TENANT_ID,
    lookback_hours: int = LOOKBACK_HOURS,
    timeout: int = 180,
) -> dict:
    """Run a batch optimization job inside the k3d pod and return parsed JSON."""
    result = subprocess.run(
        [
            "kubectl", "--context", KUBECTL_CONTEXT,
            "exec", "-n", NAMESPACE, DEPLOYMENT, "-c", CONTAINER,
            "--",
            "python3", "-m", "cogniverse_runtime.optimization_cli",
            "--mode", mode,
            "--tenant-id", tenant_id,
            "--lookback-hours", str(lookback_hours),
        ],
        capture_output=True, text=True, timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Batch job '{mode}' failed (rc={result.returncode}).\n"
            f"stderr: {result.stderr[-1000:]}\n"
            f"stdout: {result.stdout[-500:]}"
        )

    # The CLI prints JSON as the last output via json.dumps().
    # Log lines may precede it. Find the outermost JSON object.
    stdout = result.stdout.strip()

    # Try parsing from the last '{' that starts a top-level JSON object.
    # The CLI outputs a single json.dumps() call at the end.
    brace_depth = 0
    json_start = None
    for i in range(len(stdout) - 1, -1, -1):
        if stdout[i] == "}":
            if brace_depth == 0:
                json_end = i + 1
            brace_depth += 1
        elif stdout[i] == "{":
            brace_depth -= 1
            if brace_depth == 0:
                json_start = i
                break

    if json_start is not None:
        return json.loads(stdout[json_start:json_end])

    raise ValueError(
        f"No JSON found in batch job '{mode}' output.\n"
        f"stdout (last 500 chars): {stdout[-500:]}"
    )


def _load_blob_in_pod(kind: str, key: str, tenant_id: str = TENANT_ID) -> str:
    """Load an artifact blob from inside the k3d pod via ArtifactManager."""
    script = (
        "import asyncio, json; "
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
        "from cogniverse_agents.optimizer.artifact_manager import ArtifactManager; "
        f"tm = get_telemetry_manager(); "
        f"tp = tm.get_provider(tenant_id='{tenant_id}'); "
        f"am = ArtifactManager(tp, '{tenant_id}'); "
        f"blob = asyncio.get_event_loop().run_until_complete(am.load_blob('{kind}', '{key}')); "
        "print(blob if blob else '')"
    )
    result = subprocess.run(
        [
            "kubectl", "--context", KUBECTL_CONTEXT,
            "exec", "-n", NAMESPACE, DEPLOYMENT, "-c", CONTAINER,
            "--",
            "python3", "-c", script,
        ],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"load_blob({kind}, {key}) failed: {result.stderr[-500:]}"
        )
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# 1. Gateway threshold optimization
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
@skip_if_no_kubectl
class TestGatewayThresholds:
    """Verify gateway-thresholds batch job produces valid threshold artifact."""

    def test_gateway_thresholds_produces_artifact(self):
        """Run --mode gateway-thresholds, assert it produces a threshold config."""
        result = _run_batch_job("gateway-thresholds")

        assert result["status"] == "success", (
            f"Expected status='success', got '{result.get('status')}': {result}"
        )
        assert result["spans_found"] > 0, (
            f"Expected spans_found > 0, got {result.get('spans_found')}"
        )
        assert isinstance(result["artifact_id"], str) and result["artifact_id"], (
            f"Expected non-empty artifact_id, got {result.get('artifact_id')!r}"
        )

        # Threshold config is nested under 'thresholds'
        thresholds = result["thresholds"]
        analysis = thresholds["analysis"]

        assert analysis["total_spans"] > 0, (
            f"Expected total_spans > 0, got {analysis['total_spans']}"
        )
        assert analysis["mean_confidence"] > 0, (
            f"Expected mean_confidence > 0, got {analysis['mean_confidence']}"
        )

        fp_threshold = thresholds["fast_path_confidence_threshold"]
        assert isinstance(fp_threshold, float), (
            f"Expected float threshold, got {type(fp_threshold)}"
        )
        assert 0 < fp_threshold < 1, (
            f"Expected threshold in (0, 1), got {fp_threshold}"
        )

    def test_gateway_thresholds_artifact_loadable(self):
        """Load the gateway threshold artifact and verify its content."""
        # Run the job to ensure artifact exists
        job_result = _run_batch_job("gateway-thresholds")
        assert job_result["status"] == "success"

        # Load the artifact from inside the pod
        blob = _load_blob_in_pod("config", "gateway_thresholds")
        assert blob, "Loaded blob is empty"

        artifact = json.loads(blob)
        assert "fast_path_confidence_threshold" in artifact, (
            f"Missing fast_path_confidence_threshold, keys: {list(artifact.keys())}"
        )
        assert "analysis" in artifact, (
            f"Missing analysis key, keys: {list(artifact.keys())}"
        )

        # Verify analysis data matches what the job reported
        assert artifact["analysis"]["total_spans"] == job_result["thresholds"]["analysis"]["total_spans"], (
            f"Artifact total_spans ({artifact['analysis']['total_spans']}) "
            f"!= job total_spans ({job_result['thresholds']['analysis']['total_spans']})"
        )

        # Verify threshold is a valid float
        threshold = artifact["fast_path_confidence_threshold"]
        assert isinstance(threshold, (int, float)) and 0 < threshold < 1, (
            f"Invalid threshold in artifact: {threshold}"
        )

        # Analysis must have correct breakdown
        analysis = artifact["analysis"]
        assert analysis["simple_count"] + analysis["complex_count"] == analysis["total_spans"], (
            f"simple ({analysis['simple_count']}) + complex ({analysis['complex_count']}) "
            f"should equal total ({analysis['total_spans']})"
        )
        assert analysis["simple_count"] > 0, "Should have at least some simple spans"
        assert 0 <= analysis["simple_error_rate"] <= 1, (
            f"Error rate should be 0-1, got {analysis['simple_error_rate']}"
        )
        assert 0 <= analysis["complex_error_rate"] <= 1, (
            f"Error rate should be 0-1, got {analysis['complex_error_rate']}"
        )


# ---------------------------------------------------------------------------
# 2. Workflow optimization
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
@skip_if_no_kubectl
class TestWorkflowOptimization:
    """Verify workflow batch job extracts orchestration patterns."""

    def test_workflow_produces_demonstrations(self):
        """Run --mode workflow, assert demos contain real workflow data."""
        result = _run_batch_job("workflow")

        assert result["status"] == "success"
        assert result["spans_found"] > 0
        assert result["workflows_extracted"] >= 1
        assert result["execution_demos_saved"] >= 1

    def test_workflow_artifact_contains_real_data(self):
        """Workflow demos must contain agent_sequence, execution_time, success."""
        _run_batch_job("workflow")  # ensure artifact exists

        script = (
            "import asyncio, json; "
            "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
            "from cogniverse_agents.optimizer.artifact_manager import ArtifactManager; "
            f"tm = get_telemetry_manager(); "
            f"tp = tm.get_provider(tenant_id='{TENANT_ID}'); "
            f"am = ArtifactManager(tp, '{TENANT_ID}'); "
            "demos = asyncio.get_event_loop().run_until_complete("
            "  am.load_demonstrations('workflow')); "
            "print(json.dumps(demos) if demos else '[]')"
        )
        out = subprocess.run(
            ["kubectl", "--context", KUBECTL_CONTEXT,
             "exec", "-n", NAMESPACE, DEPLOYMENT, "-c", CONTAINER,
             "--", "python3", "-c", script],
            capture_output=True, text=True, timeout=60,
        )
        demos = json.loads(out.stdout.strip() or "[]")
        assert len(demos) >= 1, f"Expected workflow demos, got {len(demos)}"

        # Parse latest demo and verify it contains real workflow data
        first = json.loads(demos[-1]["input"])
        assert "workflow_id" in first, f"Demo missing workflow_id: {list(first.keys())}"
        assert "agent_sequence" in first, f"Demo missing agent_sequence: {list(first.keys())}"
        assert "execution_time" in first, f"Demo missing execution_time: {list(first.keys())}"
        assert "success" in first, f"Demo missing success flag: {list(first.keys())}"
        assert isinstance(first["agent_sequence"], list), (
            f"agent_sequence should be list, got {type(first['agent_sequence'])}"
        )
        assert len(first["agent_sequence"]) >= 1, (
            f"agent_sequence should have at least 1 agent, got {first['agent_sequence']}"
        )
        assert first["execution_time"] > 0, (
            f"execution_time should be positive, got {first['execution_time']}"
        )


# ---------------------------------------------------------------------------
# 3. SIMBA query enhancement optimization
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
@skip_if_no_kubectl
class TestSimbaOptimization:
    """Verify SIMBA batch job compiles the query enhancement module."""

    def test_simba_produces_model_artifact(self):
        """Run --mode simba, assert it produces a compiled DSPy model."""
        result = _run_batch_job("simba")

        assert result["status"] == "success"
        assert result["spans_found"] > 0
        assert result["training_examples"] >= 1
        assert isinstance(result["artifact_id"], str) and result["artifact_id"]

    def test_simba_artifact_contains_dspy_module(self):
        """SIMBA artifact must be a valid DSPy module with query enhancement signature."""
        _run_batch_job("simba")  # ensure artifact exists

        blob = _load_blob_in_pod("model", "simba_query_enhancement")
        assert blob, "SIMBA artifact blob is empty"

        artifact = json.loads(blob)

        # DSPy serialized module has a key like "enhancer.predict" or similar
        # with "signature" containing field definitions
        assert len(artifact) >= 1, f"Empty DSPy module artifact: {artifact}"

        # Find the signature in any module key
        found_signature = False
        for key, value in artifact.items():
            if isinstance(value, dict) and "signature" in value:
                sig = value["signature"]
                assert "fields" in sig, f"Signature missing 'fields': {list(sig.keys())}"

                field_names = [f.get("prefix", "").rstrip(":").strip().lower() for f in sig["fields"]]
                assert "query" in field_names, (
                    f"DSPy signature should have 'query' field, got: {field_names}"
                )
                found_signature = True
                break

        assert found_signature, (
            f"No DSPy signature found in artifact keys: {list(artifact.keys())}"
        )


# ---------------------------------------------------------------------------
# 4. Profile selection optimization
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
@skip_if_no_kubectl
class TestProfileOptimization:
    """Verify profile selection batch job compiles the profile module."""

    def test_profile_produces_model_artifact(self):
        """Run --mode profile, assert it produces a compiled DSPy model."""
        result = _run_batch_job("profile")

        assert result["status"] == "success"
        assert result["spans_found"] > 0
        assert result["training_examples"] >= 1
        assert isinstance(result["artifact_id"], str) and result["artifact_id"]

    def test_profile_artifact_contains_dspy_module(self):
        """Profile artifact must be a valid DSPy module with profile selection signature."""
        _run_batch_job("profile")  # ensure artifact exists

        blob = _load_blob_in_pod("model", "profile_selection")
        assert blob, "Profile artifact blob is empty"

        artifact = json.loads(blob)
        assert len(artifact) >= 1, f"Empty DSPy module artifact: {artifact}"

        # Find signature with profile selection fields
        found_signature = False
        for key, value in artifact.items():
            if isinstance(value, dict) and "signature" in value:
                sig = value["signature"]
                assert "fields" in sig, f"Signature missing 'fields': {list(sig.keys())}"

                field_names = [f.get("prefix", "").rstrip(":").strip().lower() for f in sig["fields"]]
                assert "query" in field_names, (
                    f"Profile DSPy signature should have 'query' field, got: {field_names}"
                )
                found_signature = True
                break

        assert found_signature, (
            f"No DSPy signature found in profile artifact keys: {list(artifact.keys())}"
        )


# ---------------------------------------------------------------------------
# 5. Span type verification
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestBatchJobsReadCorrectSpanTypes:
    """Verify the span types that each batch job reads exist in Phoenix."""

    @pytest.fixture(autouse=True)
    def _phoenix_client(self):
        """Create a Phoenix client for span queries."""
        import phoenix as px

        self.client = px.Client(endpoint=PHOENIX_URL)

    def _project_has_spans_named(self, span_name: str) -> bool:
        """Check if the tenant's Phoenix project has spans with the given name.

        Project naming follows TelemetryConfig.tenant_project_template:
        ``cogniverse-{tenant_id}`` (colon preserved, no service suffix).
        """
        project_name = f"cogniverse-{TENANT_ID}"
        try:
            df = self.client.get_spans_dataframe(project_name=project_name)
            if df is not None and not df.empty and "name" in df.columns:
                return span_name in df["name"].values
        except Exception:
            pass
        return False

    def test_gateway_spans_exist(self):
        """Phoenix has cogniverse.gateway spans for gateway-thresholds job."""
        assert self._project_has_spans_named("cogniverse.gateway"), (
            "No cogniverse.gateway spans found in Phoenix. "
            "Run some queries through the gateway first."
        )

    def test_query_enhancement_spans_exist(self):
        """Phoenix has cogniverse.query_enhancement spans for SIMBA job."""
        assert self._project_has_spans_named("cogniverse.query_enhancement"), (
            "No cogniverse.query_enhancement spans found in Phoenix. "
            "Run some complex queries that trigger enhancement first."
        )

    def test_orchestration_spans_exist(self):
        """Phoenix has cogniverse.orchestration spans for workflow job."""
        assert self._project_has_spans_named("cogniverse.orchestration"), (
            "No cogniverse.orchestration spans found in Phoenix. "
            "Run some complex queries that trigger orchestration first."
        )

    def test_profile_selection_spans_exist(self):
        """Phoenix has cogniverse.profile_selection spans for profile job."""
        assert self._project_has_spans_named("cogniverse.profile_selection"), (
            "No cogniverse.profile_selection spans found in Phoenix. "
            "Run some queries that trigger profile selection first."
        )
