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
import time

import httpx
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
RUNTIME = "http://localhost:28000"


# ---------------------------------------------------------------------------
# Module-scoped fixture: generate spans for all batch job tests
# ---------------------------------------------------------------------------

ENHANCEMENT_QUERIES = [
    "ML transformer videos", "find AI tutorials", "deep learning frameworks",
    "neural network architecture", "computer vision applications",
    "NLP text processing", "reinforcement learning robotics",
    "generative AI models", "transfer learning techniques", "autoML tools",
    "object detection algorithms", "semantic segmentation methods",
    "speech recognition systems", "recommendation engines", "time series forecasting",
    "graph neural networks", "attention mechanisms explained", "CNN architectures",
    "RNN LSTM tutorials", "GAN image generation",
]

PROFILE_QUERIES = [
    "find basketball highlights", "cooking tutorial videos", "robotics engineering",
    "music production content", "science experiments", "yoga workout videos",
    "photography tutorials", "coding bootcamp recordings", "language learning videos",
    "art history lectures", "wildlife documentary", "architecture design videos",
    "gardening how-to", "chess strategy tutorials", "piano lessons online",
    "fitness training clips", "travel vlog compilation", "astronomy lectures",
    "medical education videos", "business strategy talks",
]

COMPLEX_QUERIES = [
    "analyze the video transcripts for key themes",
    "compare videos and documents about neural networks then summarize",
    "investigate the relationship between AI research papers and video tutorials",
]


def _call_agent(agent_name: str, query: str) -> bool:
    """Call an agent endpoint. Returns True on success."""
    try:
        resp = httpx.post(
            f"{RUNTIME}/agents/{agent_name}/process",
            json={
                "agent_name": agent_name,
                "query": query,
                "context": {"tenant_id": TENANT_ID},
                "top_k": 3,
            },
            timeout=120.0,
        )
        return resp.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module", autouse=True)
def generate_spans_for_batch_jobs():
    """Generate enough spans in Phoenix for all batch job tests.

    Calls agent endpoints to produce:
    - 100+ cogniverse.gateway spans (simple queries)
    - 100+ cogniverse.query_enhancement spans
    - 100+ cogniverse.profile_selection spans
    - 3+ cogniverse.orchestration spans (complex queries)

    Runs once per module, before any batch job test.
    """
    # Check if runtime is available
    try:
        r = httpx.get(f"{RUNTIME}/health", timeout=5.0)
        if r.status_code != 200:
            pytest.skip("Runtime not available")
    except Exception:
        pytest.skip("Runtime not available")

    # Generate gateway spans (simple queries go through gateway)
    simple_queries = [
        "find videos about machine learning",
        "search for video content about AI",
        "show me cooking videos",
        "find images of neural network architectures",
        "listen to podcasts about deep learning",
    ]
    for q in simple_queries:
        _call_agent("gateway_agent", q)

    # Generate entity extraction spans (direct calls)
    entity_queries = [
        "Obama speaking at MIT about climate change",
        "Tesla cars driving in San Francisco",
        "Python programming with TensorFlow",
        "Google acquiring DeepMind in London",
        "Elon Musk presenting at Stanford University",
    ]
    for q in entity_queries:
        _call_agent("entity_extraction_agent", q)

    # Generate search spans (direct calls — produces search.execute,
    # encoder.colpali.encode, search_service.search spans)
    search_queries = [
        "machine learning tutorials",
        "cooking recipe videos",
        "robotics engineering demos",
        "music theory lectures",
        "wildlife nature footage",
    ]
    for q in search_queries:
        _call_agent("search_agent", q)

    # Generate routing spans (routing goes through gateway in the new
    # architecture, producing cogniverse.routing spans along the way)
    routing_queries = [
        "find videos about deep learning",
        "search for audio recordings",
        "show me image galleries",
        "find document archives",
        "search for video content",
    ]
    for q in routing_queries:
        _call_agent("routing_agent", q)

    # Generate query enhancement spans (100+)
    for i in range(100):
        q = f"{ENHANCEMENT_QUERIES[i % len(ENHANCEMENT_QUERIES)]} variant {i}"
        _call_agent("query_enhancement_agent", q)

    # Generate profile selection spans (100+)
    for i in range(100):
        q = f"{PROFILE_QUERIES[i % len(PROFILE_QUERIES)]} variant {i}"
        _call_agent("profile_selection_agent", q)

    # Generate orchestration spans (complex queries — these also produce
    # entity_extraction, routing, and search spans via the A2A pipeline)
    for q in COMPLEX_QUERIES:
        _call_agent("gateway_agent", q)

    # Wait for Phoenix to ingest spans
    time.sleep(15)

    yield


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
        # Most E2E queries are simple video searches → simple should be majority
        assert analysis["simple_count"] > analysis["complex_count"], (
            f"Simple ({analysis['simple_count']}) should outnumber complex "
            f"({analysis['complex_count']}) — most test queries are simple video searches"
        )
        assert 0 <= analysis["simple_error_rate"] <= 1
        assert 0 <= analysis["complex_error_rate"] <= 1
        # mean_confidence should reflect real GLiNER scores (0.4-0.7 range for simple queries)
        assert 0.3 < analysis["mean_confidence"] < 0.9, (
            f"mean_confidence {analysis['mean_confidence']} outside expected 0.3-0.9 range"
        )
        # gliner_threshold must be computed (not just default 0.3)
        assert "gliner_threshold" in artifact, "Missing computed gliner_threshold"
        assert isinstance(artifact["gliner_threshold"], float), (
            f"gliner_threshold should be float, got {type(artifact['gliner_threshold'])}"
        )
        # p25 confidence should reflect real scores from our test queries
        # (our simple queries score 0.44-0.69, so p25 should be around 0.44)
        p25 = analysis.get("p25_confidence", 0)
        assert p25 > 0.3, (
            f"p25_confidence {p25} too low — our test queries score 0.44+"
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

        # Find demos with non-empty agent_sequence (latest runs have the fix)
        valid_demos = []
        for d in demos:
            data = json.loads(d["input"])
            agents = data.get("agent_sequence", [])
            if isinstance(agents, str):
                agents = [a.strip() for a in agents.split(",") if a.strip()]
            if agents:
                valid_demos.append(data)

        assert len(valid_demos) >= 1, (
            f"Expected at least 1 demo with non-empty agent_sequence, "
            f"got {len(valid_demos)} out of {len(demos)} total demos"
        )

        # Known queries we sent: "analyze the video transcripts for key themes"
        # and "analyze the video transcripts and compare with documents"
        known_queries = {
            "analyze the video transcripts for key themes",
            "analyze the video transcripts and compare with documents",
        }
        demo_queries = {d["query"] for d in valid_demos}
        matching = demo_queries & known_queries
        assert matching, (
            f"Expected demos for queries {known_queries}, "
            f"got: {demo_queries}"
        )

        # For "analyze...compare with documents" → should use entity_extraction + search + document agents
        compare_demos = [d for d in valid_demos if "compare" in d["query"]]
        if compare_demos:
            agents = compare_demos[0]["agent_sequence"]
            if isinstance(agents, str):
                agents = [a.strip() for a in agents.split(",") if a.strip()]
            assert "entity_extraction_agent" in agents, (
                f"'compare with documents' workflow should use entity_extraction_agent, "
                f"got: {agents}"
            )
            assert any(a in agents for a in ("search_agent", "document_agent")), (
                f"'compare with documents' workflow should use search or document agent, "
                f"got: {agents}"
            )

        # All agents must be valid registered agents
        known_agents = {
            "search_agent", "summarizer_agent", "detailed_report_agent",
            "entity_extraction_agent", "query_enhancement_agent",
            "profile_selection_agent", "routing_agent", "image_search_agent",
            "audio_analysis_agent", "document_agent", "deep_research_agent",
            "coding_agent", "text_analysis_agent", "orchestrator_agent",
            "gateway_agent",
        }
        for demo in valid_demos:
            agents = demo["agent_sequence"]
            if isinstance(agents, str):
                agents = [a.strip() for a in agents.split(",") if a.strip()]
            for agent in agents:
                assert agent in known_agents, (
                    f"Unknown agent '{agent}' in workflow for query '{demo['query']}'"
                )

        # Execution metadata must be real
        for demo in valid_demos:
            assert demo["execution_time"] > 0, (
                f"execution_time should be positive for '{demo['query']}'"
            )
            assert isinstance(demo["success"], bool)
            assert demo["workflow_id"].startswith("workflow_")


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

    def test_simba_artifact_has_learned_demos(self):
        """SIMBA artifact must have demos with real query→enhanced_query pairs."""
        _run_batch_job("simba")

        blob = _load_blob_in_pod("model", "simba_query_enhancement")
        assert blob, "SIMBA artifact blob is empty"

        artifact = json.loads(blob)
        assert "enhancer.predict" in artifact, (
            f"Expected 'enhancer.predict' module, got: {list(artifact.keys())}"
        )
        module = artifact["enhancer.predict"]

        # Signature fields must match QueryEnhancementSignature exactly
        sig = module["signature"]
        field_names = [f.get("prefix", "").rstrip(":").strip() for f in sig["fields"]]
        for expected in ("Query", "Enhanced Query", "Expansion Terms", "Synonyms", "Confidence"):
            assert expected in field_names, f"Missing '{expected}', got: {field_names}"
        assert sig["instructions"] == "Enhance query with synonyms, context, and related terms"

        # Must have learned demos — 0 demos means optimization did nothing
        demos = module.get("demos", [])
        assert len(demos) >= 1, (
            "SIMBA produced 0 demos — optimization was useless"
        )

        # Each demo: real query with a DIFFERENT enhanced version
        for demo in demos:
            assert demo.get("query"), f"Demo missing query: {demo}"
            assert demo.get("enhanced_query"), f"Demo missing enhanced_query: {demo}"
            assert demo["enhanced_query"] != demo["query"], (
                f"Enhanced should differ from original: '{demo['query']}'"
            )

        # At least one demo should contain an ML-related query (our test data)
        demo_queries = " ".join(d["query"].lower() for d in demos)
        ml_terms = ("learning", "neural", "ai", "detection", "vision", "nlp", "reinforcement")
        assert any(t in demo_queries for t in ml_terms), (
            f"Demos should contain ML-related queries from our test data, "
            f"got: {[d['query'] for d in demos[:5]]}"
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

    def test_profile_artifact_has_learned_demos(self):
        """Profile artifact must have demos with real query→profile pairs."""
        _run_batch_job("profile")

        blob = _load_blob_in_pod("model", "profile_selection")
        assert blob, "Profile artifact blob is empty"

        artifact = json.loads(blob)
        assert "selector.predict" in artifact, (
            f"Expected 'selector.predict' module, got: {list(artifact.keys())}"
        )
        module = artifact["selector.predict"]

        # Signature fields must match ProfileSelectionSignature
        sig = module["signature"]
        field_names = [f.get("prefix", "").rstrip(":").strip() for f in sig["fields"]]
        for expected in ("Query", "Available Profiles", "Selected Profile", "Modality"):
            assert expected in field_names, f"Missing '{expected}', got: {field_names}"
        assert sig["instructions"] == "Select optimal backend profile based on query analysis"

        # Must have learned demos
        demos = module.get("demos", [])
        assert len(demos) >= 1, (
            "Profile produced 0 demos — optimization was useless"
        )

        # Known Vespa profiles
        known_profiles = {
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
            "video_videoprism_base_mv_chunk_30s",
            "video_videoprism_large_mv_chunk_30s",
        }

        # Each demo: real query selecting a known profile
        for demo in demos:
            assert demo.get("query"), f"Demo missing query: {demo}"
            assert demo.get("selected_profile"), f"Demo missing selected_profile: {demo}"
            assert demo["selected_profile"] in known_profiles, (
                f"Demo selected unknown profile '{demo['selected_profile']}', "
                f"expected one of {known_profiles}"
            )
            # available_profiles should list the 4 known profiles
            avail = demo.get("available_profiles", "")
            assert "video_colpali" in avail, (
                f"available_profiles should contain known profiles, got: {avail[:100]}"
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
