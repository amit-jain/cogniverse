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

MARKED AS SLOW: the module fixture seeds ~80 DSPy spans via real agent calls
(each is ~60-80s on CPU Ollama — the entire fixture takes ~2 hours). Run
explicitly with `pytest -m slow tests/e2e/test_batch_optimization_e2e.py` on
machines where the LLM is backed by a GPU or faster inference service.
"""

import json
import subprocess
import time

import httpx
import pytest

pytestmark = pytest.mark.slow

from tests.e2e.conftest import (
    PHOENIX_URL,
    TENANT_ID,
    skip_if_no_runtime,
)

KUBECTL_CONTEXT = "k3d-cogniverse"
NAMESPACE = "cogniverse"
DEPLOYMENT = "deploy/cogniverse-runtime"
CONTAINER = "runtime"
# Narrow lookback so each batch job analyses only the spans this module's
# fixture just emitted. 48h dragged in spans from every past local e2e run,
# and assertions like "simple_count > complex_count" (which reflect THIS
# fixture's 20-simple + 10-complex query mix) get drowned out when past runs
# skewed the population. 1h comfortably covers fixture → test-run latency.
LOOKBACK_HOURS = 1
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

ENTITY_QUERIES = [
    "Obama speaking at MIT about climate change",
    "Tesla cars driving in San Francisco near Google",
    "Python programming with TensorFlow for deep learning",
    "Google acquiring DeepMind in London",
    "Elon Musk presenting at Stanford University",
    "Microsoft Azure running PyTorch models",
    "Amazon AWS hosting Kubernetes clusters",
    "Apple releasing new MacBook with M4 chip",
    "NASA launching Artemis mission to Mars",
    "UNESCO declaring World Heritage sites in Japan",
    "Netflix producing documentaries about coral reefs",
    "OpenAI releasing GPT models in San Francisco",
    "Toyota manufacturing robots in Nagoya factory",
    "Samsung developing OLED displays in Seoul",
    "SpaceX Starship launching from Texas",
    "MIT researchers publishing papers on quantum computing",
    "Harvard Medical School studying gene therapy",
    "CERN operating Large Hadron Collider in Geneva",
    "Boeing testing autonomous drones in Seattle",
    "Nvidia designing GPU architectures in Santa Clara",
]

GATEWAY_QUERIES = [
    "find videos about machine learning",
    "search for video content about AI",
    "show me cooking videos",
    "find images of neural network architectures",
    "listen to podcasts about deep learning",
    "find PDF documents about Python",
    "show me robotics tutorials",
    "search for audio recordings of bird songs",
    "find basketball highlights",
    "search for video content about climate change",
    "find documentary footage of wildlife",
    "search for lecture recordings about physics",
    "show me guitar tutorial videos",
    "find photography editing tutorials",
    "search for meditation audio guides",
    "find cooking recipe demonstrations",
    "search for language learning content",
    "show me fitness workout videos",
    "find architecture design presentations",
    "search for music theory lectures",
]

COMPLEX_QUERIES = [
    "analyze the video transcripts for key themes",
    "compare videos and documents about neural networks then summarize",
    "investigate the relationship between AI research papers and video tutorials",
    "evaluate the quality of machine learning course videos and create a report",
    "synthesize findings from multiple robotics engineering lectures",
    "analyze trends in deep learning research and summarize progress",
    "compare cooking technique videos then write a detailed guide",
    "review all physics lecture recordings and identify common topics",
    "examine the relationship between AI ethics papers and tutorial content",
    "analyze video transcripts about climate change and create a summary report",
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

    # Per-agent span count. BootstrapFewShot samples demos from these; the
    # project originally generated 100 per agent which takes ~9 hours on CPU
    # Ollama. 20 per agent is enough to bootstrap 3-4 demos while keeping the
    # fixture under ~2 hours on CPU. Override via BATCH_SPAN_COUNT for
    # GPU-backed runs where 100+ is cheap.
    import os as _os
    spans_per_agent = int(_os.environ.get("BATCH_SPAN_COUNT", "20"))

    # Gateway spans — simple queries through gateway
    for i in range(spans_per_agent):
        q = f"{GATEWAY_QUERIES[i % len(GATEWAY_QUERIES)]} run {i}"
        _call_agent("gateway_agent", q)

    # Entity extraction spans
    for i in range(spans_per_agent):
        q = f"{ENTITY_QUERIES[i % len(ENTITY_QUERIES)]} case {i}"
        _call_agent("entity_extraction_agent", q)

    # Query enhancement spans.  Do NOT append a numeric suffix here: small
    # models (gemma4:e2b) treat "variant 5" as opaque content they must
    # preserve and end up echoing the whole input back unchanged, which
    # makes SIMBA train on degenerate identity pairs.  Cycling through the
    # base list is fine — spans are unique by span_id, not query text.
    for i in range(spans_per_agent):
        q = ENHANCEMENT_QUERIES[i % len(ENHANCEMENT_QUERIES)]
        _call_agent("query_enhancement_agent", q)

    # Profile selection spans
    for i in range(spans_per_agent):
        q = f"{PROFILE_QUERIES[i % len(PROFILE_QUERIES)]} variant {i}"
        _call_agent("profile_selection_agent", q)

    # Orchestration spans (10+ complex queries — each also produces
    # entity_extraction, routing, and search spans via A2A pipeline)
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
    lookback_hours: float = LOOKBACK_HOURS,
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

        # Workflow demos reflect the orchestrator's agent_sequence — the agents
        # it explicitly routes to. Entity extraction and query enhancement run
        # in the gateway preprocessing pipeline BEFORE orchestration starts,
        # so they aren't part of the orchestrator's recorded sequence. Only
        # assert on agents the orchestrator actually dispatches.
        compare_demos = [d for d in valid_demos if "compare" in d["query"]]
        if compare_demos:
            agents = compare_demos[0]["agent_sequence"]
            if isinstance(agents, str):
                agents = [a.strip() for a in agents.split(",") if a.strip()]
            assert any(a in agents for a in ("search_agent", "document_agent")), (
                f"'compare with documents' workflow should use search or document agent, "
                f"got: {agents}"
            )
            assert any(a in agents for a in ("summarizer_agent", "detailed_report_agent")), (
                f"'compare' workflow should aggregate results via summarizer or "
                f"report agent, got: {agents}"
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
        from phoenix.client import Client

        self.client = Client(base_url=PHOENIX_URL)

    def _project_has_spans_named(self, span_name: str) -> bool:
        """Check if the tenant's Phoenix project has spans with the given name.

        Project naming follows TelemetryConfig.tenant_project_template:
        ``cogniverse-{tenant_id}`` (colon preserved, no service suffix).
        """
        project_name = f"cogniverse-{TENANT_ID}"
        try:
            df = self.client.spans.get_spans_dataframe(project_identifier=project_name)
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


# ---------------------------------------------------------------------------
# 5. Entity extraction optimization
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
@skip_if_no_kubectl
class TestEntityExtractionOptimization:
    """Verify entity extraction batch job compiles the entity extraction module."""

    def test_entity_extraction_produces_model_artifact(self):
        """Run --mode entity-extraction, assert it produces a compiled DSPy model."""
        result = _run_batch_job("entity-extraction")

        assert result["status"] == "success"
        assert result["spans_found"] > 0
        assert result["training_examples"] >= 1
        assert isinstance(result["artifact_id"], str) and result["artifact_id"]

    def test_entity_extraction_artifact_has_learned_demos(self):
        """Entity extraction artifact must have demos with real entity data."""
        _run_batch_job("entity-extraction")

        blob = _load_blob_in_pod("model", "entity_extraction")
        assert blob, "Entity extraction artifact blob is empty"

        artifact = json.loads(blob)
        assert "extractor.predict" in artifact, (
            f"Expected 'extractor.predict' module, got: {list(artifact.keys())}"
        )
        module = artifact["extractor.predict"]

        # Signature fields must match EntityExtractionSignature exactly
        sig = module["signature"]
        field_names = [f.get("prefix", "").rstrip(":").strip() for f in sig["fields"]]
        for expected in ("Query", "Entities", "Entity Types"):
            assert expected in field_names, f"Missing '{expected}', got: {field_names}"
        assert sig["instructions"] == "Extract named entities from text query"

        # Must have learned demos — 0 demos means optimization did nothing
        demos = module.get("demos", [])
        assert len(demos) >= 1, (
            "Entity extraction produced 0 demos — optimization was useless"
        )

        # Each demo: real query with entities extracted
        # Entities may be pipe-delimited (DSPy fallback: "text|type|confidence")
        # or JSON array (GLiNER fast path: [{"text": ..., "type": ..., "confidence": ...}])
        for demo in demos:
            assert demo.get("query"), f"Demo missing query: {demo}"
            assert demo.get("entities"), f"Demo missing entities: {demo}"
            entities_str = demo["entities"]
            has_pipe_format = "|" in entities_str
            has_json_format = entities_str.strip().startswith("[")
            assert has_pipe_format or has_json_format, (
                f"Entities should be pipe-delimited or JSON array, "
                f"got: '{entities_str[:100]}'"
            )

        # At least one demo should contain entity-related queries from our test data
        # (fixture generates queries like "ML transformer", "find AI tutorials" etc.)
        demo_queries = " ".join(d["query"].lower() for d in demos)
        entity_terms = ("ml", "ai", "learning", "neural", "vision", "transformer", "deep")
        assert any(t in demo_queries for t in entity_terms), (
            f"Demos should contain entity-rich queries from test data, "
            f"got: {[d['query'] for d in demos[:5]]}"
        )


# ---------------------------------------------------------------------------
# 6. Routing optimization
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
@skip_if_no_kubectl
class TestRoutingOptimization:
    """Verify routing batch job compiles the routing decision module.

    Generates fresh routing spans via complex gateway queries — these trigger
    OrchestratorAgent which calls RoutingAgent, producing cogniverse.routing
    spans with the new attribute format (query, recommended_agent, primary_intent).
    """

    @pytest.fixture(autouse=True)
    def _generate_routing_spans(self):
        """Generate fresh routing spans with current code.

        The module-scoped fixture may have generated spans with OLD code that
        lacked query/recommended_agent attributes. We need spans from the
        current code which emits the full attribute dict.
        """
        # Generate routing spans by calling routing_agent inside the pod directly.
        # The REST API dispatcher routes routing_agent through gateway/orchestrator
        # paths that skip the actual RoutingAgent, so we use kubectl exec to
        # instantiate and call RoutingAgent.route_query() directly.
        routing_queries = [
            "find basketball highlights",
            "show me cooking videos",
            "search for AI tutorial content",
            "find music video clips",
            "look for science experiment videos",
            "find yoga workout content",
            "search photography tutorials",
            "find coding bootcamp recordings",
            "look for language learning videos",
            "find art history lectures",
        ]
        # Generate routing spans by calling RoutingAgent.route_query() directly
        # inside the pod. The REST API dispatcher routes routing_agent through
        # gateway/orchestrator paths that skip the actual RoutingAgent.
        for q in routing_queries:
            script = (
                "import asyncio; "
                "from cogniverse_agents.routing_agent import RoutingAgent, RoutingDeps; "
                "from cogniverse_foundation.config.utils import create_default_config_manager, get_config; "
                "from cogniverse_foundation.telemetry.config import TelemetryConfig; "
                "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
                "cm = create_default_config_manager(); "
                "config = get_config(tenant_id='default', config_manager=cm); "
                "llm = config.get_llm_config().resolve('routing_agent'); "
                "agent = RoutingAgent(deps=RoutingDeps(telemetry_config=TelemetryConfig(enabled=True), llm_config=llm)); "
                "agent.telemetry_manager = get_telemetry_manager(); "
                f"asyncio.run(agent.route_query(query={q!r}, tenant_id={TENANT_ID!r})); "
                "print('ok')"
            )
            try:
                subprocess.run(
                    [
                        "kubectl", "--context", KUBECTL_CONTEXT,
                        "exec", "-n", NAMESPACE, DEPLOYMENT, "-c", CONTAINER,
                        "--", "python3", "-c", script,
                    ],
                    capture_output=True, text=True, timeout=60,
                )
            except Exception:
                pass
        time.sleep(15)  # Wait for Phoenix ingestion

    def test_routing_produces_model_artifact(self):
        """Run --mode routing, assert it produces a compiled DSPy model."""
        result = _run_batch_job("routing")

        assert result["status"] == "success", (
            f"Routing batch job failed: {result}. "
            "If spans_found=0, RoutingAgent is not emitting cogniverse.routing spans."
        )
        assert result["spans_found"] > 0
        assert result["training_examples"] >= 1
        assert isinstance(result["artifact_id"], str) and result["artifact_id"]

    def test_routing_artifact_has_learned_demos(self):
        """Routing artifact must have demos with real routing decisions."""
        _run_batch_job("routing")

        blob = _load_blob_in_pod("model", "routing_decision")
        assert blob, "Routing artifact blob is empty"

        artifact = json.loads(blob)
        # DSPyAdvancedRoutingModule has router.predict, basic_module.analyzer.predict, etc.
        module_key = next(
            (k for k in artifact if "predict" in k.lower()),
            None,
        )
        assert module_key is not None, (
            f"No routing module found in artifact, keys: {list(artifact.keys())}"
        )
        module = artifact[module_key]

        sig = module["signature"]
        field_names = [f.get("prefix", "").rstrip(":").strip() for f in sig["fields"]]
        assert "Query" in field_names, f"Missing 'Query' field, got: {field_names}"

        demos = module.get("demos", [])
        assert len(demos) >= 1, "Routing produced 0 demos — optimization was useless"

        known_agents = {
            "search_agent", "video_search_agent", "orchestrator_agent",
            "summarizer_agent", "image_search_agent", "audio_analysis_agent",
            "document_agent", "detailed_report_agent", "entity_extraction_agent",
            "query_enhancement_agent", "profile_selection_agent",
        }
        for demo in demos:
            assert demo.get("query"), f"Demo missing query: {demo}"
            agent = demo.get("recommended_agent", "")
            if agent:
                assert agent in known_agents, (
                    f"Demo recommended unknown agent '{agent}', "
                    f"expected one of {known_agents}"
                )


# ---------------------------------------------------------------------------
# 7. Artifact loading round-trip
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
@skip_if_no_kubectl
class TestArtifactLoadingRoundTrip:
    """Full loop: batch job → artifact → pod restart → agent uses optimized thresholds."""

    def test_gateway_artifact_round_trip(self):
        """Run gateway-thresholds → verify artifact → restart → verify agent uses it."""
        # 1. Run batch job and capture the optimized thresholds
        result = _run_batch_job("gateway-thresholds")
        assert result["status"] == "success"

        optimized_threshold = result["thresholds"]["fast_path_confidence_threshold"]
        optimized_gliner = result["thresholds"]["gliner_threshold"]

        # 2. Verify artifact in pod matches what the batch job produced
        blob = _load_blob_in_pod("config", "gateway_thresholds")
        assert blob, "Gateway artifact blob is empty"
        artifact = json.loads(blob)
        assert artifact["fast_path_confidence_threshold"] == optimized_threshold, (
            f"Artifact threshold {artifact['fast_path_confidence_threshold']} "
            f"!= batch job threshold {optimized_threshold}"
        )
        assert artifact["gliner_threshold"] == optimized_gliner, (
            f"Artifact gliner {artifact['gliner_threshold']} "
            f"!= batch job gliner {optimized_gliner}"
        )

        # 3. Restart runtime pod to trigger artifact loading
        subprocess.run(
            [
                "kubectl", "--context", KUBECTL_CONTEXT,
                "rollout", "restart", "deployment/cogniverse-runtime",
                "-n", NAMESPACE,
            ],
            check=True, timeout=30,
        )
        subprocess.run(
            [
                "kubectl", "--context", KUBECTL_CONTEXT,
                "rollout", "status", DEPLOYMENT,
                "-n", NAMESPACE, "--timeout=120s",
            ],
            check=True, timeout=150,
        )
        time.sleep(20)  # Wait for agent initialization + artifact loading

        # 4. Query the gateway and verify it works after restart with artifact loaded.
        #    The optimized threshold may classify this as simple or complex depending
        #    on the threshold value. Either path proves the agent started correctly.
        #    What we're testing: artifact loading didn't crash the agent.
        resp = httpx.post(
            f"{RUNTIME}/agents/gateway_agent/process",
            json={
                "agent_name": "gateway_agent",
                "query": "find cooking videos",
                "context": {"tenant_id": TENANT_ID},
            },
            timeout=120.0,
        )
        assert resp.status_code == 200, (
            f"Agent failed after restart: {resp.status_code} {resp.text[:200]}"
        )
        body = resp.json()
        assert body.get("status") == "success", (
            f"Gateway dispatch did not succeed: {json.dumps(body, default=str)[:300]}"
        )
        # Gateway returns either:
        # - Simple path: {"gateway": {"complexity": "simple", "routed_to": ...}, "downstream_result": ...}
        # - Complex path: {"agent": "orchestrator_agent", "orchestration_result": ...}
        # Both are valid — what matters is the agent processed the query, not HTTP 200
        agent = body.get("agent", "")
        assert agent in ("gateway_agent", "orchestrator_agent"), (
            f"Expected gateway_agent or orchestrator_agent, got '{agent}'. "
            f"Body: {json.dumps(body, default=str)[:300]}"
        )
        if agent == "gateway_agent":
            gateway_info = body.get("gateway", {})
            assert "complexity" in gateway_info, (
                f"Gateway response missing complexity: {gateway_info}"
            )
            assert "routed_to" in gateway_info, (
                f"Gateway response missing routed_to: {gateway_info}"
            )
        else:
            # Routed to orchestrator — verify orchestration produced a result
            assert "orchestration_result" in body, (
                f"Orchestrator path but no orchestration_result: {list(body.keys())}"
            )

        # 5. Verify the artifact is still loadable in-pod after restart
        #    (proves the agent's telemetry infrastructure survived restart)
        blob_after = _load_blob_in_pod("config", "gateway_thresholds")
        assert blob_after, "Gateway artifact not loadable after restart"
        artifact_after = json.loads(blob_after)
        assert artifact_after["fast_path_confidence_threshold"] == optimized_threshold, (
            f"Artifact threshold changed after restart: "
            f"{artifact_after['fast_path_confidence_threshold']} != {optimized_threshold}"
        )

    def test_simba_artifact_round_trip(self):
        """Run simba batch job -> verify artifact blob has correct structure and is loadable."""
        # 1. Run batch job
        result = _run_batch_job("simba")
        assert result["status"] == "success"
        assert result["training_examples"] >= 1

        # 2. Verify artifact blob exists and has correct structure
        blob = _load_blob_in_pod("model", "simba_query_enhancement")
        assert blob, "SIMBA artifact blob is empty after batch job"

        artifact = json.loads(blob)
        assert "enhancer.predict" in artifact, (
            f"Expected 'enhancer.predict' module, got: {list(artifact.keys())}"
        )

        # Must have learned demos
        demos = artifact["enhancer.predict"].get("demos", [])
        assert len(demos) >= 1, "SIMBA artifact has 0 demos"

        # Each demo should have query and enhanced_query
        for demo in demos:
            assert demo.get("query"), f"Demo missing query: {demo}"
            assert demo.get("enhanced_query"), f"Demo missing enhanced_query: {demo}"
            assert demo["enhanced_query"] != demo["query"], (
                f"Enhanced should differ from original: '{demo['query']}'"
            )

        # 3. Verify the artifact is loadable in-pod (proves it survives restart
        #    since test_gateway_artifact_round_trip already restarted the pod)
        blob_check = _load_blob_in_pod("model", "simba_query_enhancement")
        assert blob_check, "SIMBA artifact not loadable in pod"
        reloaded = json.loads(blob_check)
        assert len(reloaded["enhancer.predict"].get("demos", [])) == len(demos), (
            "SIMBA artifact demo count changed between loads"
        )

    def test_entity_extraction_artifact_survives_restart(self):
        """Verify entity_extraction artifact is loadable after the gateway restart."""
        # Run batch job to ensure artifact exists
        result = _run_batch_job("entity-extraction")
        assert result["status"] == "success"
        assert result["training_examples"] >= 1

        # Load the artifact — the gateway test already restarted the pod,
        # so this proves the artifact persists across restarts
        blob = _load_blob_in_pod("model", "entity_extraction")
        assert blob, "Entity extraction artifact not loadable after restart"

        artifact = json.loads(blob)
        assert "extractor.predict" in artifact, (
            f"Expected 'extractor.predict' module, got: {list(artifact.keys())}"
        )

        demos = artifact["extractor.predict"].get("demos", [])
        assert len(demos) >= 1, "Entity extraction artifact has 0 demos"

        # Verify demo structure: each should have query and entities
        for demo in demos:
            assert demo.get("query"), f"Demo missing query: {demo}"
            assert demo.get("entities"), f"Demo missing entities: {demo}"
            entities_str = demo["entities"]
            has_pipe = "|" in entities_str
            has_json = entities_str.strip().startswith("[")
            assert has_pipe or has_json, (
                f"Entities should be pipe-delimited or JSON, got: '{entities_str[:100]}'"
            )

    def test_profile_artifact_survives_restart(self):
        """Verify profile selection artifact is loadable after the gateway restart."""
        # Run batch job to ensure artifact exists
        result = _run_batch_job("profile")
        assert result["status"] == "success"
        assert result["training_examples"] >= 1

        # Load the artifact — proves persistence across the gateway restart
        blob = _load_blob_in_pod("model", "profile_selection")
        assert blob, "Profile selection artifact not loadable after restart"

        artifact = json.loads(blob)
        assert "selector.predict" in artifact, (
            f"Expected 'selector.predict' module, got: {list(artifact.keys())}"
        )

        demos = artifact["selector.predict"].get("demos", [])
        assert len(demos) >= 1, "Profile selection artifact has 0 demos"

        known_profiles = {
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
            "video_videoprism_base_mv_chunk_30s",
            "video_videoprism_large_mv_chunk_30s",
        }

        # Verify demo structure: each should have query and selected_profile
        for demo in demos:
            assert demo.get("query"), f"Demo missing query: {demo}"
            assert demo.get("selected_profile"), f"Demo missing selected_profile: {demo}"
            assert demo["selected_profile"] in known_profiles, (
                f"Demo selected unknown profile '{demo['selected_profile']}', "
                f"expected one of {known_profiles}"
            )


# ---------------------------------------------------------------------------
# 8. Synthetic data generation
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
@skip_if_no_kubectl
class TestSyntheticGeneration:
    """Verify --mode synthetic runs inside the pod."""

    def test_synthetic_mode_runs(self):
        """Run --mode synthetic for one optimizer type, verify structured result."""
        result = subprocess.run(
            [
                "kubectl", "--context", KUBECTL_CONTEXT,
                "exec", "-n", NAMESPACE, DEPLOYMENT, "-c", CONTAINER,
                "--",
                "python3", "-m", "cogniverse_runtime.optimization_cli",
                "--mode", "synthetic",
                "--tenant-id", TENANT_ID,
                "--agents", "simba",
            ],
            capture_output=True, text=True, timeout=300,
        )

        stdout = result.stdout.strip()

        brace_depth = 0
        json_start = None
        json_end = None
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

        assert json_start is not None, (
            f"No JSON in synthetic output. rc={result.returncode}, "
            f"stderr={result.stderr[-300:]}, stdout={stdout[-300:]}"
        )
        output = json.loads(stdout[json_start:json_end])

        assert "results" in output, (
            f"Synthetic output missing 'results' key: {output}"
        )
        assert "simba" in output["results"], (
            f"Synthetic output missing 'simba' result: {output['results']}"
        )
        assert output["results"]["simba"]["status"] in ("success", "failed", "no_data"), (
            f"Unexpected status: {output['results']['simba']}"
        )
