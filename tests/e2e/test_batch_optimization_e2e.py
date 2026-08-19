"""
E2E tests for Argo batch optimization jobs.

Tests the 4 optimization CLI modes (gateway-thresholds, workflow, simba, profile)
by running them inside the k3d pod via kubectl exec. Verifies the full loop:
spans exist in Phoenix -> batch job reads them -> produces artifact -> artifact
contains correct data -> agent can load the artifact.

Requires live k3d stack via `cogniverse up` with:
- Runtime at localhost:33000
- Phoenix at localhost:33006
- kubectl context: k3d-cogniverse

MARKED AS SLOW: the module fixture seeds ~80 DSPy spans via real agent calls
(each is ~60-80s on a CPU-served LM — the entire fixture takes ~2 hours). Run
explicitly with `pytest -m slow tests/e2e/test_batch_optimization_e2e.py` on
machines where the LLM is backed by a GPU or faster inference service.
"""

import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import pytest

from cogniverse_agents.query_enhancement_agent import QueryEnhancementModule

pytestmark = pytest.mark.slow

from tests.e2e.conftest import (
    GATEWAY_VIDEO_QUERIES,
    IN_POD_TELEMETRY_PRELUDE,
    KUBECTL_CONTEXT,
    PHOENIX_URL,
    TENANT_ID,
    expected_gateway_calibration,
    expected_gateway_routing,
    register_tenant_and_wait,
)
from tests.e2e.test_api_e2e import _deploy_profile_for_tenant


def _enabled_agents_in_shipped_config() -> set[str]:
    """Agents the runtime routes to today: configs/config.json ``agents``
    entries not disabled — the optimizer's own stale-demo filter."""
    config = json.loads(
        (Path(__file__).resolve().parents[2] / "configs" / "config.json").read_text()
    )
    agents = config.get("agents", {})
    live = {
        name
        for name, body in agents.items()
        if isinstance(body, dict) and body.get("enabled", True)
    }
    assert live, "configs/config.json agents block is empty"
    return live


NAMESPACE = "cogniverse"
DEPLOYMENT = "deploy/cogniverse-runtime"
CONTAINER = "runtime"
# Each batch job analyses the spans this module's fixtures emitted: the
# lookback is measured from the moment span seeding started (plus a small
# margin), so it neither drags in earlier sessions' traffic nor expires the
# seeded spans when the module runs longer than a fixed window.
_SPAN_SEED_STARTED_AT: float | None = None
_LOOKBACK_MARGIN_HOURS = 0.25


def _module_lookback_hours() -> float:
    assert _SPAN_SEED_STARTED_AT is not None, (
        "batch job requested before this module's span seeding started"
    )
    return (time.time() - _SPAN_SEED_STARTED_AT) / 3600.0 + _LOOKBACK_MARGIN_HOURS


RUNTIME = "http://localhost:33000"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.json"


def _configured_profile_names(profile_type: str | None = None) -> tuple[str, ...]:
    config = json.loads(CONFIG_PATH.read_text())
    profiles = config.get("backend", {}).get("profiles", {})
    names = []
    for profile_name, profile_config in profiles.items():
        if not isinstance(profile_config, dict):
            continue
        if profile_type is None or profile_config.get("type") == profile_type:
            names.append(profile_name)
    return tuple(names)


# ---------------------------------------------------------------------------
# Module-scoped fixture: generate spans for all batch job tests
# ---------------------------------------------------------------------------

ENHANCEMENT_QUERIES = [
    "ML transformer videos",
    "find AI tutorials",
    "deep learning frameworks",
    "neural network architecture",
    "computer vision applications",
    "NLP text processing",
    "reinforcement learning robotics",
    "generative AI models",
    "transfer learning techniques",
    "autoML tools",
    "object detection algorithms",
    "semantic segmentation methods",
    "speech recognition systems",
    "recommendation engines",
    "time series forecasting",
    "graph neural networks",
    "attention mechanisms explained",
    "CNN architectures",
    "RNN LSTM tutorials",
    "GAN image generation",
]

PROFILE_QUERIES = [
    "find basketball highlights",
    "cooking tutorial videos",
    "robotics engineering",
    "music production content",
    "science experiments",
    "yoga workout videos",
    "photography tutorials",
    "coding bootcamp recordings",
    "language learning videos",
    "art history lectures",
    "wildlife documentary",
    "architecture design videos",
    "gardening how-to",
    "chess strategy tutorials",
    "piano lessons online",
    "fitness training clips",
    "travel vlog compilation",
    "astronomy lectures",
    "medical education videos",
    "business strategy talks",
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

# Live cue-less gateway/orchestrator calls measured 176s, 136s, and 229s here;
# use the shared 480s endpoint budget from ORCHESTRATOR_PROCESS_TIMEOUT_S.
GATEWAY_PROCESS_TIMEOUT_S = 480.0

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

GATEWAY_THRESHOLD_PROFILES = _configured_profile_names("video")


# Query-enhancement calls that carry upstream entities — the hardest served
# input: the enhancement must surface the entity names. Seeding them puts
# grounded records in the SIMBA training set and holdout.
GROUNDED_ENHANCEMENT_QUERIES = [
    (
        "find tutorials",
        [
            {"text": "TensorFlow", "type": "TECHNOLOGY", "confidence": 0.9},
            {"text": "neural networks", "type": "CONCEPT", "confidence": 0.85},
        ],
        [
            {
                "subject": "TensorFlow",
                "relation": "used_for",
                "object": "neural networks",
            }
        ],
    ),
    (
        "compare frameworks",
        [
            {"text": "PyTorch", "type": "TECHNOLOGY", "confidence": 0.9},
            {"text": "JAX", "type": "TECHNOLOGY", "confidence": 0.8},
        ],
        [],
    ),
    (
        "show lectures",
        [
            {"text": "linear algebra", "type": "CONCEPT", "confidence": 0.9},
            {"text": "eigenvalues", "type": "CONCEPT", "confidence": 0.8},
        ],
        [{"subject": "eigenvalues", "relation": "part_of", "object": "linear algebra"}],
    ),
    (
        "find demos",
        [{"text": "Kubernetes", "type": "TECHNOLOGY", "confidence": 0.9}],
        [],
    ),
    (
        "search recordings",
        [
            {"text": "jazz piano", "type": "GENRE", "confidence": 0.9},
            {"text": "Bill Evans", "type": "PERSON", "confidence": 0.9},
        ],
        [{"subject": "Bill Evans", "relation": "plays", "object": "jazz piano"}],
    ),
    (
        "get footage",
        [
            {"text": "volcano", "type": "LOCATION", "confidence": 0.9},
            {"text": "lava flow", "type": "EVENT", "confidence": 0.8},
        ],
        [],
    ),
]


def _call_agent(
    agent_name: str,
    query: str,
    tenant_id: str = TENANT_ID,
    context_extra: dict | None = None,
) -> None:
    resp = httpx.post(
        f"{RUNTIME}/agents/{agent_name}/process",
        json={
            "agent_name": agent_name,
            "query": query,
            "context": {"tenant_id": tenant_id, **(context_extra or {})},
            "top_k": 3,
        },
        timeout=GATEWAY_PROCESS_TIMEOUT_S,
    )
    assert resp.status_code == 200, (
        f"{agent_name} rejected span-seeding query {query!r}: "
        f"HTTP {resp.status_code} {resp.text[:500]}"
    )


@pytest.fixture(scope="module")
def _kubectl_cluster_ready() -> None:
    """Require kubectl access after the session E2E stack is initialized."""
    command = [
        "kubectl",
        "--context",
        KUBECTL_CONTEXT,
        "get",
        "namespace",
        NAMESPACE,
        "-o",
        "name",
    ]
    command_text = " ".join(command)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except FileNotFoundError as exc:
        pytest.fail(
            f"kubectl executable unavailable after E2E stack setup; "
            f"command={command_text!r}; context={KUBECTL_CONTEXT!r}; error={exc}",
            pytrace=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"kubectl cluster check timed out after E2E stack setup; "
            f"command={command_text!r}; context={KUBECTL_CONTEXT!r}; "
            f"timeout={exc.timeout}s; stdout={exc.stdout!r}; stderr={exc.stderr!r}",
            pytrace=False,
        )
    if result.returncode != 0:
        pytest.fail(
            f"kubectl cannot reach the E2E cluster after stack setup; "
            f"command={command_text!r}; context={KUBECTL_CONTEXT!r}; "
            f"returncode={result.returncode}; stdout={result.stdout!r}; "
            f"stderr={result.stderr!r}",
            pytrace=False,
        )


def _count_spans_by_name_in_pod(tenant_id: str, span_name_symbol: str) -> int:
    """Count spans of one training-span type for a tenant, via the runtime pod.

    ``span_name_symbol`` is a ``SPAN_NAME_*`` name in
    ``cogniverse_foundation.telemetry.config`` (e.g. ``SPAN_NAME_PROFILE_SELECTION``).
    Seeding emits best-effort onto the batch queue (~500ms), so callers poll
    this until the directly-seeded lower bound is present before optimizing.
    """
    script = IN_POD_TELEMETRY_PRELUDE + (
        "import asyncio; "
        f"from cogniverse_foundation.telemetry.config import {span_name_symbol}; "
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
        "from cogniverse_runtime.optimization_cli import _query_spans_by_name; "
        f"tm = get_telemetry_manager(); "
        f"tp = tm.get_provider(tenant_id={tenant_id!r}); "
        f"df = asyncio.run(_query_spans_by_name(tm, tp, {tenant_id!r}, {span_name_symbol}, 1)); "
        "print('__SPANS__' + str(len(df)))"
    )
    result = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "exec",
            "-n",
            NAMESPACE,
            DEPLOYMENT,
            "-c",
            CONTAINER,
            "--",
            "python3",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = result.stdout.strip().splitlines()[-1]
    assert line.startswith("__SPANS__"), result.stdout[-500:]
    return int(line[len("__SPANS__") :])


def _count_gateway_spans_in_pod(tenant_id: str) -> int:
    return _count_spans_by_name_in_pod(tenant_id, "SPAN_NAME_GATEWAY")


def _wait_for_seeded_span_lower_bound_in_pod(
    tenant_id: str, span_name_symbol: str, minimum: int, timeout_s: float = 240.0
) -> None:
    """Poll until at least ``minimum`` spans of this type are queryable.

    Emitters are async best-effort (batch export), so a directly-seeded span
    is eventually consistent. Waiting for the seeded lower bound makes the
    optimizer read deterministic without forcing synchronous export on the
    request path.
    """
    deadline = time.monotonic() + timeout_s
    seen = -1
    while time.monotonic() < deadline:
        seen = _count_spans_by_name_in_pod(tenant_id, span_name_symbol)
        if seen >= minimum:
            return
        time.sleep(5.0)
    raise AssertionError(
        f"Phoenix shows {seen} {span_name_symbol} spans for tenant {tenant_id!r}; "
        f"expected at least {minimum} within {timeout_s:.0f}s"
    )


def _wait_for_gateway_spans_in_pod(tenant_id: str, expected: int) -> None:
    deadline = time.monotonic() + 240.0
    seen = -1
    while time.monotonic() < deadline:
        seen = _count_gateway_spans_in_pod(tenant_id)
        if seen == expected:
            return
        time.sleep(5.0)
    raise AssertionError(
        f"Phoenix shows {seen} gateway spans for tenant {tenant_id!r}; "
        f"expected {expected} within 240s"
    )


class GatewayThresholdTenant:
    """A dedicated tenant plus the exact gateway decisions it recorded."""

    def __init__(self, tenant_id: str, decisions: list[tuple[str, float]]):
        self.tenant_id = tenant_id
        self.decisions = decisions

    @property
    def expected_thresholds(self) -> dict:
        return expected_gateway_calibration(self.decisions)


@pytest.fixture(scope="module")
def gateway_threshold_tenant(_kubectl_cluster_ready) -> GatewayThresholdTenant:
    """Create a dedicated tenant for gateway-threshold optimization runs and
    drive exactly BATCH_SPAN_COUNT (default 20) simple video decisions
    through its gateway, recording each one so the calibration is exact."""
    suffix = uuid.uuid4().hex[:8]
    org_id = f"opt_gw_{suffix}"
    tenant_id = f"{org_id}:t1"

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{RUNTIME}/admin/organizations",
            json={
                "org_id": org_id,
                "org_name": f"opt-gw-{suffix}",
                "created_by": "e2e",
            },
        )
        assert resp.status_code in (200, 201, 409), resp.text

    register_tenant_and_wait(tenant_id, created_by="e2e", timeout_s=600.0)

    with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
        for profile_name in GATEWAY_THRESHOLD_PROFILES:
            _deploy_profile_for_tenant(client, profile_name, tenant_id)

    span_count = int(os.environ.get("BATCH_SPAN_COUNT", "20"))
    assert span_count > 0, "BATCH_SPAN_COUNT must be a positive integer"
    decisions: list[tuple[str, float]] = []
    with httpx.Client(base_url=RUNTIME, timeout=GATEWAY_PROCESS_TIMEOUT_S) as client:
        for i in range(span_count):
            query = GATEWAY_VIDEO_QUERIES[i % len(GATEWAY_VIDEO_QUERIES)]
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
                    "query": query,
                    "context": {"tenant_id": tenant_id},
                    "top_k": 3,
                },
            )
            assert resp.status_code == 200, resp.text[:500]
            body = resp.json()
            gw = body["gateway"]
            assert (gw["complexity"], gw["modality"], gw["routed_to"]) == (
                "simple",
                "video",
                "search_agent",
            ), body
            assert gw["generation_type"] == "raw_results", body
            assert gw["confidence"] >= gw["fast_path_confidence_threshold"], body
            assert body["status"] == "success", body
            assert body["downstream_result"]["status"] == "success", body
            decisions.append((gw["complexity"], gw["confidence"]))
    _wait_for_gateway_spans_in_pod(tenant_id, span_count)
    try:
        yield GatewayThresholdTenant(tenant_id, decisions)
    finally:
        with httpx.Client(timeout=60.0) as client:
            try:
                client.delete(f"{RUNTIME}/admin/tenants/{tenant_id}")
            except httpx.HTTPError:
                pass
            try:
                client.delete(f"{RUNTIME}/admin/organizations/{org_id}")
            except httpx.HTTPError:
                pass


@pytest.fixture(scope="module", autouse=True)
def generate_spans_for_batch_jobs(_kubectl_cluster_ready):
    """Generate enough spans in Phoenix for all batch job tests.

    Calls agent endpoints to produce:
    - 100+ cogniverse.gateway spans (simple queries)
    - 100+ cogniverse.query_enhancement spans
    - 100+ cogniverse.profile_selection spans
    - 3+ cogniverse.orchestration spans (complex queries)

    Runs once per module, before any batch job test.
    """
    global _SPAN_SEED_STARTED_AT
    _SPAN_SEED_STARTED_AT = time.time()
    response = httpx.get(f"{RUNTIME}/health", timeout=5.0)
    assert response.status_code == 200, (
        f"runtime health returned HTTP {response.status_code}: {response.text[:500]}"
    )
    # The tenant's SIMBA artifact is this module's own state: seed the
    # query-enhancement spans from the base module, not from whatever an
    # earlier optimization run left persisted (and loaded into the pod).
    if _reset_query_enhancement_artifact_in_pod():
        _bounce_runtime_pod()

    # Per-agent span count. BootstrapFewShot samples demos from these; the
    # project originally generated 100 per agent which takes ~9 hours on CPU
    # the local LM. 20 per agent is enough to bootstrap 3-4 demos while keeping the
    # fixture under ~2 hours on CPU. Override via BATCH_SPAN_COUNT for
    # GPU-backed runs where 100+ is cheap.
    import os as _os

    spans_per_agent = int(_os.environ.get("BATCH_SPAN_COUNT", "20"))
    assert spans_per_agent > 0, "BATCH_SPAN_COUNT must be a positive integer"

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
    for q, entities, relationships in GROUNDED_ENHANCEMENT_QUERIES:
        _call_agent(
            "query_enhancement_agent",
            q,
            context_extra={"entities": entities, "relationships": relationships},
        )

    # Profile selection spans
    for i in range(spans_per_agent):
        q = f"{PROFILE_QUERIES[i % len(PROFILE_QUERIES)]} variant {i}"
        _call_agent("profile_selection_agent", q)

    # Orchestration spans (10+ complex queries — each also produces
    # entity_extraction, routing, and search spans via A2A pipeline)
    for q in COMPLEX_QUERIES:
        _call_agent("gateway_agent", q)

    # Wait for Phoenix to ingest EVERY seeded span type before any optimizer
    # reads them. Emitters are async best-effort (batch export), so seeded
    # spans are eventually consistent; polling here — not forcing synchronous
    # export on the request path — is what keeps the batch-job reads
    # deterministic. QE waits on the exact seeded query set; the others wait
    # on the directly-seeded lower bound (complex-query fan-out adds more).
    _wait_for_seeded_query_enhancement_queries_in_pod()
    _wait_for_seeded_span_lower_bound_in_pod(
        TENANT_ID, "SPAN_NAME_GATEWAY", spans_per_agent
    )
    _wait_for_seeded_span_lower_bound_in_pod(
        TENANT_ID, "SPAN_NAME_ENTITY_EXTRACTION", spans_per_agent
    )
    _wait_for_seeded_span_lower_bound_in_pod(
        TENANT_ID, "SPAN_NAME_PROFILE_SELECTION", spans_per_agent
    )
    _wait_for_seeded_span_lower_bound_in_pod(
        TENANT_ID, "SPAN_NAME_ORCHESTRATION", len(COMPLEX_QUERIES)
    )

    yield


def _run_batch_job(
    mode: str,
    tenant_id: str = TENANT_ID,
    lookback_hours: float | None = None,
    # A job is a Phoenix span scan (tens of seconds on a project holding a
    # day of traffic) plus a DSPy compile with real LM calls at ~12 tok/s —
    # ~2 min solo, more when the cluster is loaded.
    timeout: int = 600,
) -> dict:
    """Run a batch optimization job inside the k3d pod and return parsed JSON."""
    if lookback_hours is None:
        lookback_hours = _module_lookback_hours()
    result = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "exec",
            "-n",
            NAMESPACE,
            DEPLOYMENT,
            "-c",
            CONTAINER,
            "--",
            "python3",
            "-m",
            "cogniverse_runtime.optimization_cli",
            "--mode",
            mode,
            "--tenant-id",
            tenant_id,
            "--lookback-hours",
            str(lookback_hours),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
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
    script = IN_POD_TELEMETRY_PRELUDE + (
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
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "exec",
            "-n",
            NAMESPACE,
            DEPLOYMENT,
            "-c",
            CONTAINER,
            "--",
            "python3",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"load_blob({kind}, {key}) failed: {result.stderr[-500:]}")
    return result.stdout.strip()


def _blob_version_lineage_in_pod(
    kind: str, key: str, tenant_id: str = TENANT_ID
) -> list[dict]:
    """The versioned-artifact ledger for ``kind/key``, read from inside the pod."""
    script = IN_POD_TELEMETRY_PRELUDE + (
        "import asyncio, json; "
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
        "from cogniverse_agents.optimizer.artifact_manager import ArtifactManager; "
        f"tp = get_telemetry_manager().get_provider(tenant_id={tenant_id!r}); "
        f"am = ArtifactManager(tp, {tenant_id!r}); "
        "lineage = asyncio.get_event_loop().run_until_complete("
        f"am.get_version_lineage({kind!r}, {key!r})); "
        "print('__LINEAGE__' + json.dumps(lineage))"
    )
    result = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "exec",
            "-n",
            NAMESPACE,
            DEPLOYMENT,
            "-c",
            CONTAINER,
            "--",
            "python3",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"get_version_lineage({kind}, {key}) failed: {result.stderr[-500:]}"
        )
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("__LINEAGE__"))
    return json.loads(line[len("__LINEAGE__") :])


def _active_blob_version_in_pod(kind: str, key: str, tenant_id: str = TENANT_ID):
    """The activated version of ``kind/key``, or None, read from inside the pod."""
    script = IN_POD_TELEMETRY_PRELUDE + (
        "import asyncio, json; "
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
        "from cogniverse_agents.optimizer.artifact_manager import ArtifactManager; "
        f"tp = get_telemetry_manager().get_provider(tenant_id={tenant_id!r}); "
        f"am = ArtifactManager(tp, {tenant_id!r}); "
        "state = asyncio.get_event_loop().run_until_complete("
        f"am.get_blob_state({kind!r}, {key!r})); "
        "print('__ACTIVE__' + json.dumps(state['active']))"
    )
    result = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "exec",
            "-n",
            NAMESPACE,
            DEPLOYMENT,
            "-c",
            CONTAINER,
            "--",
            "python3",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"get_blob_state({kind}, {key}) failed: {result.stderr[-500:]}"
        )
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("__ACTIVE__"))
    active = json.loads(line[len("__ACTIVE__") :])
    return active["version"] if active else None


def _reset_query_enhancement_artifact_in_pod(tenant_id: str = TENANT_ID) -> bool:
    """Persist the base QueryEnhancementModule state as the tenant's SIMBA artifact.

    Returns True when the persisted artifact differed from the base state
    (so the running pod, which loaded it at start, must be bounced before
    it serves the seeding traffic).
    """
    script = IN_POD_TELEMETRY_PRELUDE + (
        "import asyncio, json; "
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
        "from cogniverse_agents.optimizer.artifact_manager import ArtifactManager; "
        "from cogniverse_agents.query_enhancement_agent import QueryEnhancementModule; "
        "from cogniverse_runtime.optimization_cli import SIMBA_ARTIFACT_KEY; "
        f"tp = get_telemetry_manager().get_provider(tenant_id={tenant_id!r}); "
        f"am = ArtifactManager(tp, {tenant_id!r}); "
        "base = json.dumps(QueryEnhancementModule().dump_state(), default=str); "
        "blob = asyncio.run(am.load_blob('model', SIMBA_ARTIFACT_KEY)); "
        "differs = (json.loads(blob) != json.loads(base)) if blob else False; "
        "asyncio.run(am.save_blob(kind='model', key=SIMBA_ARTIFACT_KEY, content=base)); "
        "print('__RESET__' + ('1' if differs else '0'))"
    )
    result = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "exec",
            "-n",
            NAMESPACE,
            DEPLOYMENT,
            "-c",
            CONTAINER,
            "--",
            "python3",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = result.stdout.strip().splitlines()[-1]
    assert line in ("__RESET__0", "__RESET__1"), result.stdout[-500:]
    return line == "__RESET__1"


def _bounce_runtime_pod(ready_timeout_s: int = 240) -> str:
    """Delete-pod 1:1 replacement of the runtime pod and wait for ready.

    Uses ``kubectl delete pod`` rather than ``kubectl rollout restart``
    because a rolling update tries to surge a second 8Gi pod alongside
    the current one, which never schedules on a memory-pinned k3d
    laptop. Returns the new pod name so callers can scrape its logs.
    """
    old_pod = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "get",
            "pods",
            "-n",
            NAMESPACE,
            "-l",
            "app.kubernetes.io/component=runtime",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        check=True,
        timeout=15,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "delete",
            "pod",
            old_pod,
            "-n",
            NAMESPACE,
            "--grace-period=10",
        ],
        check=True,
        timeout=30,
    )
    deadline = time.monotonic() + ready_timeout_s
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{RUNTIME}/health/live", timeout=10.0)
            if r.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(5)
    else:
        raise AssertionError(
            f"Runtime did not return /health/live=200 within {ready_timeout_s}s"
        )
    # Settle for agent registry, schema convergence, artifact loading.
    time.sleep(20)

    # Resolve the NEW pod name for log scraping. The deployment
    # controller schedules the replacement under a different name.
    new_pod = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "get",
            "pods",
            "-n",
            NAMESPACE,
            "-l",
            "app.kubernetes.io/component=runtime",
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ],
        check=True,
        timeout=15,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return new_pod


def _read_pod_logs(pod_name: str, since: str = "5m", tail_lines: int = 5000) -> str:
    """Return container logs for ``pod_name`` (runtime container)."""
    result = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "logs",
            pod_name,
            "-n",
            NAMESPACE,
            "-c",
            CONTAINER,
            f"--since={since}",
            f"--tail={tail_lines}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"kubectl logs {pod_name} failed: {result.stderr[-500:]}")
    return result.stdout


# ---------------------------------------------------------------------------
# 1. Gateway threshold optimization
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def seeded_gateway_traffic():
    """Route real queries through the gateway so its spans land inside the
    thresholds job's lookback window.

    The job reads ``cogniverse.gateway`` spans from the module's lookback
    window; without seeding, the test silently depends
    on some earlier suite (a2a_gateway) having run recently, and returns
    ``no_data`` when executed on its own.
    """
    queries = [
        "search for video content about AI",
        "find clips showing outdoor scenes",
        "show me videos about machine learning",
        "search for cooking demonstrations",
        "find footage of city traffic",
        "show videos with people talking",
    ]
    # The gateway span is emitted as soon as the gateway CLASSIFIES the
    # query — the downstream agent's answer is irrelevant here, so keep a
    # 480s per-query budget and tolerate individual slow dispatches; one
    # classified query is enough for the job's span analysis.
    seeded = 0
    with httpx.Client(base_url=RUNTIME, timeout=GATEWAY_PROCESS_TIMEOUT_S) as client:
        for query in queries:
            try:
                resp = client.post(
                    "/agents/gateway_agent/process",
                    json={
                        "agent_name": "gateway_agent",
                        "query": query,
                        "context": {"tenant_id": TENANT_ID},
                        "top_k": 3,
                    },
                )
                if resp.status_code == 200:
                    seeded += 1
            except httpx.HTTPError:
                continue
    assert seeded >= 1, (
        f"No gateway seeding query succeeded within {GATEWAY_PROCESS_TIMEOUT_S:.0f}s each"
    )
    # OTLP export is batched; give the exporter time to flush to Phoenix.
    time.sleep(15)


@pytest.mark.e2e
@pytest.mark.usefixtures("seeded_gateway_traffic")
class TestGatewayThresholds:
    """Verify gateway-thresholds batch job produces valid threshold artifact."""

    def test_gateway_thresholds_produces_artifact(self, gateway_threshold_tenant):
        """Run --mode gateway-thresholds: the job calibrates exactly from the
        tenant's recorded decisions and reports the persisted artifact."""
        result = _run_batch_job(
            "gateway-thresholds", tenant_id=gateway_threshold_tenant.tenant_id
        )

        assert result["status"] == "success", result
        assert result["spans_found"] == len(gateway_threshold_tenant.decisions), result
        assert isinstance(result["artifact_id"], str) and result["artifact_id"], result
        expected = gateway_threshold_tenant.expected_thresholds
        thresholds = result["thresholds"]
        assert (
            thresholds["fast_path_confidence_threshold"]
            == (expected["fast_path_confidence_threshold"])
        ), thresholds
        assert thresholds["gliner_threshold"] == expected["gliner_threshold"], (
            thresholds
        )
        assert thresholds["analysis"] == expected["analysis"], thresholds
        assert thresholds == expected

    def test_gateway_thresholds_artifact_loadable(self, gateway_threshold_tenant):
        """The persisted artifact is exactly what the job computed."""
        job_result = _run_batch_job(
            "gateway-thresholds", tenant_id=gateway_threshold_tenant.tenant_id
        )
        assert job_result["status"] == "success", job_result

        blob = _load_blob_in_pod(
            "config", "gateway_thresholds", tenant_id=gateway_threshold_tenant.tenant_id
        )
        artifact = json.loads(blob)
        assert set(artifact) == {
            "fast_path_confidence_threshold",
            "gliner_threshold",
            "analysis",
        }, artifact
        assert artifact["analysis"]["total_spans"] == len(
            gateway_threshold_tenant.decisions
        ), artifact
        assert artifact == gateway_threshold_tenant.expected_thresholds
        assert artifact == job_result["thresholds"]


# ---------------------------------------------------------------------------
# 2. Workflow optimization
# ---------------------------------------------------------------------------


@pytest.mark.e2e
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
        result = _run_batch_job("workflow")  # ensure artifact exists

        script = IN_POD_TELEMETRY_PRELUDE + (
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
            [
                "kubectl",
                "--context",
                KUBECTL_CONTEXT,
                "exec",
                "-n",
                NAMESPACE,
                DEPLOYMENT,
                "-c",
                CONTAINER,
                "--",
                "python3",
                "-c",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        demos = json.loads(out.stdout.strip() or "[]")
        assert demos != [], "Expected workflow demos, got 0"

        # Find demos with non-empty agent_sequence (latest runs have the fix)
        valid_demos = []
        for d in demos:
            data = json.loads(d["input"])
            agents = data.get("agent_sequence", [])
            if isinstance(agents, str):
                agents = [a.strip() for a in agents.split(",") if a.strip()]
            if agents:
                valid_demos.append(data)

        assert valid_demos != [], (
            f"Expected at least 1 demo with non-empty agent_sequence, "
            f"got 0 out of {len(demos)} total demos"
        )
        assert len(valid_demos) == result["execution_demos_saved"], result

        # Every orchestrated query the fixture seeded must appear as a demo.
        # Derived from the seeding constant so the expectation cannot name a
        # query the fixture never sent.
        seeded_queries = set(COMPLEX_QUERIES)
        assert seeded_queries != set()
        demo_queries = {d["query"] for d in valid_demos}
        assert seeded_queries <= demo_queries, (
            f"Expected a demo for every seeded orchestrated query; missing "
            f"{sorted(seeded_queries - demo_queries)}, got: {sorted(demo_queries)}"
        )

        # Workflow demos must show a real retrieval-heavy execution, not a
        # planner name guess.
        compare_demos = [d for d in valid_demos if "compare" in d["query"]]
        assert compare_demos != [], f"Expected compare demos, got: {demo_queries}"
        compare_demo = max(
            compare_demos,
            key=lambda d: (
                len((d.get("metadata") or {}).get("agent_observations") or []),
                d.get("task_count") or 0,
            ),
        )
        # The plan SHAPE (how many agents, which final synthesizer) is the
        # planner LM's free choice and legitimately varies run to run (a 3-agent
        # [query_enhancement, search, summarizer] plan and a 5-agent plan both
        # complete successfully). Assert the retrieval-heavy PROPERTY the query
        # demands, not an LM-chosen count. The cross-modal coverage question
        # (a "videos and documents" query must retrieve BOTH modalities, since
        # one search_agent call searches one modality) is a separate product
        # contract tracked in docs/plan/optimization-flow.md, not this test.
        metadata = compare_demo.get("metadata") or {}
        execution_order = metadata.get("execution_order") or []
        observations = metadata.get("agent_observations") or []
        executed = set(execution_order)
        seq = compare_demo["agent_sequence"]
        if isinstance(seq, str):
            seq = [a.strip() for a in seq.split(",") if a.strip()]
        assert compare_demo["success"] is True, compare_demo
        # It must actually retrieve (the "retrieval-heavy execution") ...
        assert {"search_agent"} <= executed, compare_demo
        # ... and synthesize (the query's explicit "then summarize / write a
        # guide" step ran a report/summary agent) ...
        assert executed & {"summarizer_agent", "detailed_report_agent"}, compare_demo
        # ... as the FINAL planned step (synthesis follows retrieval, not before).
        assert seq[-1] in {"summarizer_agent", "detailed_report_agent"}, compare_demo
        # Consistency invariants (exact, not LM-chosen magnitudes): one task per
        # planned agent, one observation per executed step.
        assert metadata.get("tasks_completed") == len(seq), compare_demo
        assert len(observations) == len(execution_order), compare_demo

        # Observed workflows may name any agent enabled in the shipped config —
        # the same set the optimizer's stale-demo filter keeps
        # (optimization_cli._agents_live); the synthetic generator's narrower
        # planning vocabulary does not apply to recorded orchestrations.
        live_agents = _enabled_agents_in_shipped_config()
        for demo in valid_demos:
            agents = demo["agent_sequence"]
            if isinstance(agents, str):
                agents = [a.strip() for a in agents.split(",") if a.strip()]
            assert agents, f"empty agent_sequence for query '{demo['query']}'"
            for agent in agents:
                assert agent in live_agents, (
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


def _approved_query_enhancement_examples_in_pod(
    tenant_id: str = TENANT_ID,
) -> list[dict]:
    """The tenant's approved synthetic query-enhancement examples, read in-pod."""
    script = IN_POD_TELEMETRY_PRELUDE + (
        "import asyncio, json; "
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
        "from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data; "
        f"tp = get_telemetry_manager().get_provider(tenant_id={tenant_id!r}); "
        f"rows = asyncio.run(_load_approved_synthetic_data(tp, {tenant_id!r}, 'query_enhancement')); "
        "print('__APPROVED__' + json.dumps(rows, default=str))"
    )
    result = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "exec",
            "-n",
            NAMESPACE,
            DEPLOYMENT,
            "-c",
            CONTAINER,
            "--",
            "python3",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = result.stdout.strip().splitlines()[-1]
    assert line.startswith("__APPROVED__"), result.stdout[-500:]
    return json.loads(line[len("__APPROVED__") :])


def _served_query_enhancement_queries_in_pod(
    tenant_id: str = TENANT_ID,
    lookback_hours: float | None = None,
) -> set[str]:
    """Every query the query-enhancement agent served in the module window.

    Read from the tenant's Phoenix spans in-pod — the population the SIMBA job
    builds its records from. It holds the fixture's seeded calls AND the
    sub-queries the orchestrator issued to the agent while the complex
    seeding queries ran; both are real served traffic.
    """
    if lookback_hours is None:
        lookback_hours = _module_lookback_hours()
    script = IN_POD_TELEMETRY_PRELUDE + (
        "import asyncio, json; "
        "from cogniverse_foundation.telemetry.config import SPAN_NAME_QUERY_ENHANCEMENT; "
        "from cogniverse_foundation.telemetry.manager import get_telemetry_manager; "
        "from cogniverse_runtime.optimization_cli import _query_enhancement_pairs, _query_spans_by_name; "
        f"tm = get_telemetry_manager(); "
        f"tp = tm.get_provider(tenant_id={tenant_id!r}); "
        f"df = asyncio.run(_query_spans_by_name(tm, tp, {tenant_id!r}, SPAN_NAME_QUERY_ENHANCEMENT, {lookback_hours!r})); "
        "print('__SERVED__' + json.dumps(sorted({r['query'] for r in _query_enhancement_pairs(df)})))"
    )
    result = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "exec",
            "-n",
            NAMESPACE,
            DEPLOYMENT,
            "-c",
            CONTAINER,
            "--",
            "python3",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = result.stdout.strip().splitlines()[-1]
    assert line.startswith("__SERVED__"), result.stdout[-500:]
    return set(json.loads(line[len("__SERVED__") :]))


def _wait_for_seeded_query_enhancement_queries_in_pod(
    tenant_id: str = TENANT_ID,
    lookback_hours: float | None = None,
    timeout_s: float = 240.0,
) -> set[str]:
    """Wait until this module's seeded query-enhancement queries are visible."""
    expected_queries = set(ENHANCEMENT_QUERIES) | {
        q for q, _, _ in GROUNDED_ENHANCEMENT_QUERIES
    }
    deadline = time.monotonic() + timeout_s
    served_queries: set[str] = set()
    while time.monotonic() < deadline:
        served_queries = _served_query_enhancement_queries_in_pod(
            tenant_id, lookback_hours
        )
        if served_queries and expected_queries <= served_queries:
            return served_queries
        time.sleep(5.0)
    missing = sorted(expected_queries - served_queries)
    raise AssertionError(
        f"Phoenix showed {len(served_queries & expected_queries)}/{len(expected_queries)} "
        f"seeded query-enhancement queries after {timeout_s:.0f}s; missing: {missing}"
    )


def _usable_profile_names_in_pod(tenant_id: str = TENANT_ID) -> list[str]:
    """The candidate pool the profile-selection agent shows the LM for this
    tenant, derived in-pod by the agent's own function (tenant profiles whose
    inference services are deployed, in the agent's type order). Visual-only
    profiles belong here — selection is not synthetic-generation groundability."""
    script = (
        "import json; "
        "from cogniverse_foundation.config.utils import create_default_config_manager; "
        "from cogniverse_agents.profile_selection_agent import tenant_usable_profile_names; "
        f"print('__USABLE__' + json.dumps(tenant_usable_profile_names(create_default_config_manager(), {tenant_id!r})))"
    )
    result = subprocess.run(
        [
            "kubectl",
            "--context",
            KUBECTL_CONTEXT,
            "exec",
            "-n",
            NAMESPACE,
            DEPLOYMENT,
            "-c",
            CONTAINER,
            "--",
            "python3",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    line = result.stdout.strip().splitlines()[-1]
    assert line.startswith("__USABLE__"), result.stdout[-500:]
    usable = json.loads(line[len("__USABLE__") :])
    assert usable, f"tenant {tenant_id!r} exposes no usable profiles"
    return usable


def _base_query_enhancement_state() -> dict:
    return json.loads(json.dumps(QueryEnhancementModule().dump_state(), default=str))


def _assert_simba_served_the_best_module(result: dict, blob_before: str) -> dict:
    """The contract of one ``--mode simba`` run against the seeded tenant.

    Every query-enhancement span in the module window (the fixture's seeded
    calls and the orchestrator's own sub-calls) plus every approved synthetic
    example is a record; the holdout is the deterministic quarter; the module
    persisted after the run is the one that scored best on that holdout and
    never scores below the base module. Returns the persisted state.
    """
    approved = _approved_query_enhancement_examples_in_pod()
    assert set(result) == {
        "status",
        "spans_found",
        "examples",
        "training_examples",
        "holdout_examples",
        "baseline_score",
        "current_score",
        "candidate_score",
        "decision",
        "version",
        "consumed_example_ids",
    }, result
    assert result["status"] == "success", result
    assert result["examples"] == result["spans_found"] + len(approved), result
    assert result["holdout_examples"] == max(1, result["examples"] // 4), result
    assert (
        result["training_examples"] <= result["examples"] - result["holdout_examples"]
    )
    served_score = {
        "promote": result["candidate_score"],
        "keep": result["current_score"],
        "rollback": result["baseline_score"],
    }[result["decision"]]
    assert served_score >= result["baseline_score"], result

    after = json.loads(_load_blob_in_pod("model", "simba_query_enhancement"))
    assert list(after) == ["enhancer.predict"], list(after)
    # The persisted signature is the served module's: ChainOfThought places
    # its Reasoning field ahead of the signature's own outputs, so the order
    # comes from the real predictor, not from the class body.
    served_signature = QueryEnhancementModule().enhancer.predict.signature
    sig = after["enhancer.predict"]["signature"]
    assert [f.get("prefix", "").rstrip(":").strip() for f in sig["fields"]] == [
        field.json_schema_extra["prefix"].rstrip(":").strip()
        for field in served_signature.fields.values()
    ], sig["fields"]
    assert sig["instructions"] == served_signature.instructions

    # The run persisted a version whose ledger names exactly what it consumed;
    # promote/rollback activate it, keep/reject leave the pointer.
    version = result["version"]
    assert isinstance(version, int) and version >= 1, result
    lineage = _blob_version_lineage_in_pod("model", "simba_query_enhancement")
    newest = lineage[-1]
    assert newest["version"] == version, lineage
    assert newest["decision"] == result["decision"], newest
    assert newest["consumed_example_ids"] == result["consumed_example_ids"], newest
    assert newest["consumed_example_ids"] != [], newest
    served_queries = _served_query_enhancement_queries_in_pod()
    seeded_queries = set(ENHANCEMENT_QUERIES) | {
        q for q, _, _ in GROUNDED_ENHANCEMENT_QUERIES
    }
    # Every consumed id is a served span or an approved example — never
    # a fabricated attribution.
    for example_id in result["consumed_example_ids"]:
        assert example_id.startswith(("span:", "approved:")), example_id

    demos = after["enhancer.predict"]["demos"]
    if result["decision"] == "promote":
        # The fixture's seeded calls all landed as served spans ...
        assert seeded_queries <= served_queries, sorted(seeded_queries - served_queries)
        # ... and every demo is a real served call or an approved example —
        # never a label synthesized from a query string.
        record_queries = served_queries | {row["query"] for row in approved}
        assert demos != [], result
        for demo in demos:
            assert demo["query"] in record_queries, demo
            assert (
                demo["enhanced_query"].strip().lower() != demo["query"].strip().lower()
            ), demo
        assert (
            _active_blob_version_in_pod("model", "simba_query_enhancement") == version
        ), lineage
    elif result["decision"] == "rollback":
        assert after == _base_query_enhancement_state()
        assert (
            _active_blob_version_in_pod("model", "simba_query_enhancement") == version
        ), lineage
    else:
        # keep / reject record the version but never activate it; the served
        # blob is whatever the run started from.
        assert after == json.loads(blob_before)
    return after


@pytest.mark.e2e
class TestSimbaOptimization:
    """Verify SIMBA batch job serves only a query-enhancement module that
    scores at least as well as the base module on held-out served calls."""

    def test_simba_serves_the_best_scoring_module(self):
        """Run --mode simba against the seeded spans and pin its contract."""
        blob_before = _load_blob_in_pod("model", "simba_query_enhancement")
        assert blob_before != "", "the module fixture persists the base artifact"

        result = _run_batch_job("simba", timeout=1200)

        _assert_simba_served_the_best_module(result, blob_before)

    def test_simba_second_run_is_consistent_with_the_first(self):
        """A rerun scores the artifact the first run persisted as ``current``
        and again serves the best module."""
        blob_before = _load_blob_in_pod("model", "simba_query_enhancement")
        first = json.loads(blob_before)

        result = _run_batch_job("simba", timeout=1200)

        after = _assert_simba_served_the_best_module(result, blob_before)
        assert isinstance(result["current_score"], float), result
        if result["decision"] == "keep":
            assert after == first


# ---------------------------------------------------------------------------
# 4. Profile selection optimization
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestProfileOptimization:
    """Verify profile selection batch job compiles the profile module."""

    def test_profile_produces_model_artifact(self):
        """Run --mode profile, assert it produces a compiled DSPy model."""
        result = _run_batch_job("profile")

        assert result["status"] == "success"
        assert result["spans_found"] > 0
        assert result["training_examples"] >= 1
        assert isinstance(result["version"], int) and result["version"] >= 1
        assert result["consumed_example_ids"] != []
        lineage = _blob_version_lineage_in_pod("model", "profile_selection")
        assert lineage[-1]["version"] == result["version"], lineage
        assert lineage[-1]["consumed_example_ids"] == result["consumed_example_ids"], (
            lineage
        )
        assert lineage[-1]["decision"] == "promote", lineage
        assert (
            _active_blob_version_in_pod("model", "profile_selection")
            == result["version"]
        ), lineage

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
        assert (
            sig["instructions"]
            == "Select optimal backend profile based on query analysis"
        )

        # Must have learned demos
        demos = module.get("demos", [])
        assert len(demos) >= 1, "Profile produced 0 demos — optimization was useless"

        usable_profiles = _usable_profile_names_in_pod(TENANT_ID)

        # Each demo: real query, and the pool it was labelled against is exactly
        # the pool the agent shows the LM (recorded on the span, read back by
        # the optimizer) — not a literal, not a subset, same order.
        for demo in demos:
            assert demo.get("query"), f"Demo missing query: {demo}"
            available = [
                profile.strip()
                for profile in demo.get("available_profiles", "").split(",")
                if profile.strip()
            ]
            assert available == usable_profiles, (
                f"demo available_profiles {available} != the agent's candidate "
                f"pool for the tenant {usable_profiles}"
            )
            assert demo["selected_profile"] in available, (
                f"Demo selected profile {demo['selected_profile']!r} is absent from "
                f"available_profiles {available}"
            )


@pytest.mark.e2e
class TestProfileSelectionArtifactReload:
    """Verify ProfileSelectionAgent's ``_load_artifact`` actually runs at
    startup and applies the optimized DSPy state to its in-memory
    module — not just that the artifact blob persists. Closes the
    verification gap between 'optimizer wrote a blob' and 'the live
    agent uses it on the next request'.

    The chart's ``agent-optimization`` CronWorkflow runs
    ``optimization_cli --mode profile`` weekly and then
    ``kubectl rollout restart deployment/runtime`` so agents pick up
    new artifacts. This test mirrors that exact sequence end-to-end.
    """

    def test_profile_agent_loads_optimized_module_after_restart(self):
        # 1. Run the profile optimizer to produce a fresh artifact.
        result = _run_batch_job("profile")
        assert result["status"] == "success"
        assert result["training_examples"] >= 1

        # 2. Verify the artifact has a non-trivial demo set before
        # restart, so we know there is something for _load_artifact to
        # actually load.
        blob_before = _load_blob_in_pod("model", "profile_selection")
        assert blob_before, "Profile artifact blob is empty before restart"
        artifact_before = json.loads(blob_before)
        demos_before = artifact_before.get("selector.predict", {}).get("demos", [])
        assert len(demos_before) >= 1, (
            f"Pre-restart artifact has 0 demos — nothing to load. "
            f"Keys present: {list(artifact_before.keys())}"
        )

        # 3. Bounce the runtime pod.
        new_pod = _bounce_runtime_pod()

        # 4. Issue a query first — the dispatcher constructs each agent
        # lazily on first dispatch, and ``_load_artifact`` only runs at
        # construction time. Without driving traffic, the agent is never
        # built and the load-success log line never fires.
        resp = httpx.post(
            f"{RUNTIME}/agents/profile_selection_agent/process",
            json={
                "agent_name": "profile_selection_agent",
                "query": "find a clip about machine learning",
                "context": {"tenant_id": TENANT_ID},
            },
            timeout=600.0,
        )
        assert resp.status_code == 200, (
            f"profile_selection_agent failed after restart: "
            f"{resp.status_code} {resp.text[:300]}"
        )
        body = resp.json()
        assert body.get("status") == "success", (
            f"Agent dispatch did not succeed: {json.dumps(body, default=str)[:300]}"
        )

        # 5. Now scrape logs. The success marker is emitted by
        # ``ProfileSelectionAgent._load_artifact`` (profile_selection_agent.py
        # line 254-255). Presence proves the artifact blob made it into
        # the live ``dspy_module.load_state`` call. Absence means the
        # agent silently fell back to the unoptimized module — the bug
        # this test was added to catch.
        logs = _read_pod_logs(new_pod, since="10m")
        assert (
            "ProfileSelectionAgent loaded optimized DSPy module from artifact" in logs
        ), (
            "Expected ProfileSelectionAgent load-success log line in new "
            f"pod {new_pod}; either _load_artifact didn't run or it "
            "swallowed an exception. Last 1500 chars of logs:\n"
            f"{logs[-1500:]}"
        )

        # 6. Re-read the artifact from inside the pod and assert demo
        # parity with the pre-restart state. This proves persistence
        # across the restart and that ProfileSelectionModule's load
        # didn't mutate the on-disk state.
        blob_after = _load_blob_in_pod("model", "profile_selection")
        assert blob_after, "Profile artifact missing after restart"
        artifact_after = json.loads(blob_after)
        demos_after = artifact_after.get("selector.predict", {}).get("demos", [])
        assert len(demos_after) == len(demos_before), (
            f"Demo count drifted across restart: "
            f"{len(demos_before)} -> {len(demos_after)}"
        )


# ---------------------------------------------------------------------------
# 5. Span type verification
# ---------------------------------------------------------------------------


@pytest.mark.e2e
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
        Scoped to the session window with a real timeout — an unscoped scan
        of the whole project blows the client's 5s method default once the
        span store holds a day of traffic, and the swallowed exception then
        reads as "no spans".
        """
        from datetime import datetime, timedelta, timezone

        project_name = f"cogniverse-{TENANT_ID}"
        window_start = datetime.now(timezone.utc) - timedelta(hours=3)
        last_error: Exception | None = None
        for _ in range(3):
            try:
                df = self.client.spans.get_spans_dataframe(
                    project_identifier=project_name,
                    start_time=window_start,
                    limit=2000,
                    timeout=90,
                )
                if df is not None and not df.empty and "name" in df.columns:
                    return span_name in df["name"].values
                return False
            except Exception as e:  # noqa: BLE001 — retried, then surfaced
                last_error = e
                time.sleep(3)
        raise AssertionError(
            f"Phoenix span query for {span_name!r} kept failing: {last_error!r}"
        )

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
class TestEntityExtractionOptimization:
    """Verify entity extraction batch job compiles the entity extraction module."""

    def test_entity_extraction_produces_model_artifact(self):
        """Run --mode entity-extraction, assert it produces a compiled DSPy model."""
        # ~2 min on an idle cluster (Phoenix span scan + DSPy compile with
        # real LM calls); leave headroom for a loaded Phoenix/LM.
        result = _run_batch_job("entity-extraction", timeout=600)

        assert result["status"] == "success"
        assert result["spans_found"] > 0
        assert result["training_examples"] >= 1
        assert isinstance(result["version"], int) and result["version"] >= 1
        assert result["consumed_example_ids"] != []
        lineage = _blob_version_lineage_in_pod("model", "entity_extraction")
        assert lineage[-1]["version"] == result["version"], lineage
        assert lineage[-1]["consumed_example_ids"] == result["consumed_example_ids"], (
            lineage
        )
        assert lineage[-1]["decision"] == "promote", lineage
        assert (
            _active_blob_version_in_pod("model", "entity_extraction")
            == result["version"]
        ), lineage

    def test_entity_extraction_artifact_has_learned_demos(self):
        """Entity extraction artifact must have demos with real entity data."""
        _run_batch_job("entity-extraction", timeout=600)

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
        entity_terms = (
            "ml",
            "ai",
            "learning",
            "neural",
            "vision",
            "transformer",
            "deep",
        )
        assert any(t in demo_queries for t in entity_terms), (
            f"Demos should contain entity-rich queries from test data, "
            f"got: {[d['query'] for d in demos[:5]]}"
        )


# ---------------------------------------------------------------------------
# 7. Artifact loading round-trip
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestArtifactLoadingRoundTrip:
    """Full loop: batch job → artifact → pod restart → agent uses optimized thresholds."""

    def test_gateway_artifact_round_trip(self, gateway_threshold_tenant):
        """Run gateway-thresholds → verify artifact → restart → verify agent uses it."""
        # 1. Run batch job and capture the optimized thresholds
        result = _run_batch_job(
            "gateway-thresholds", tenant_id=gateway_threshold_tenant.tenant_id
        )
        assert result["status"] == "success"

        optimized_threshold = result["thresholds"]["fast_path_confidence_threshold"]
        optimized_gliner = result["thresholds"]["gliner_threshold"]

        # 2. Verify artifact in pod matches what the batch job produced
        blob = _load_blob_in_pod(
            "config", "gateway_thresholds", tenant_id=gateway_threshold_tenant.tenant_id
        )
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

        # 3. Restart runtime pod to trigger artifact loading.
        #
        # Don't use ``kubectl rollout restart``: that stamps the
        # PodTemplate, and with the deployment's default RollingUpdate
        # strategy (maxSurge=25%, maxUnavailable=25%) it tries to bring
        # up a second 8Gi runtime pod alongside the current one before
        # killing the old. On a memory-pinned k3d laptop (node ~98% of
        # 48Gi allocated by colpali+llm+vespa+runtime) the surge pod
        # never schedules and the rollout times out.
        #
        # ``kubectl delete pod`` replaces 1:1 — the existing pod is
        # killed, the deployment controller spins up its replacement,
        # and the same memory slot is reused. No surge, no rollout
        # status to wait on.
        pod_name = subprocess.run(
            [
                "kubectl",
                "--context",
                KUBECTL_CONTEXT,
                "get",
                "pods",
                "-n",
                NAMESPACE,
                "-l",
                "app.kubernetes.io/component=runtime",
                "--field-selector=status.phase=Running",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ],
            check=True,
            timeout=15,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            [
                "kubectl",
                "--context",
                KUBECTL_CONTEXT,
                "delete",
                "pod",
                pod_name,
                "-n",
                NAMESPACE,
                "--grace-period=10",
            ],
            check=True,
            timeout=30,
        )
        # Wait for the replacement pod to be Ready. The deployment
        # controller schedules a new pod almost immediately after the
        # delete; the 60s readiness initialDelaySeconds + schema reload
        # + colpali probe means /health/live takes ~2 min to respond.
        # While the new pod is starting uvicorn can briefly accept a TCP
        # connection through the k3d nginx proxy and then close it before
        # any HTTP response — that surfaces as RemoteProtocolError, not
        # ConnectError. Catch the full HTTPError tree so the poll keeps
        # retrying through the startup window instead of crashing once.
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            try:
                r = httpx.get(f"{RUNTIME}/health/live", timeout=10.0)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(5)
        else:
            raise AssertionError(
                "Runtime did not return /health/live=200 within 240s of pod delete"
            )
        # One more pause so the agent registry, schema convergence and
        # artifact loading all settle before the gateway dispatch in
        # step 4.
        time.sleep(20)

        # 4. Query the gateway and verify it works after restart with artifact loaded.
        #    The response's gateway block reports the thresholds the restarted
        #    agent APPLIED, so the artifact having been loaded is asserted
        #    exactly, and the decision must obey the rule against them.
        # Cold-started runtime: first gateway call walks the full
        # GLiNER load + DSPy module compile + LM inference path,
        # 60-180s on CPU. 120s timeout was too tight after the pod
        # delete; 600s gives margin without masking real hangs.
        query = "find videos of dogs running on a beach"
        resp = httpx.post(
            f"{RUNTIME}/agents/gateway_agent/process",
            json={
                "agent_name": "gateway_agent",
                "query": query,
                "context": {"tenant_id": gateway_threshold_tenant.tenant_id},
            },
            timeout=600.0,
        )
        assert resp.status_code == 200, (
            f"Agent failed after restart: {resp.status_code} {resp.text[:200]}"
        )
        body = resp.json()
        assert body["status"] == "success", json.dumps(body, default=str)[:300]
        gw = body["gateway"]
        assert (gw["fast_path_confidence_threshold"], gw["gliner_threshold"]) == (
            optimized_threshold,
            optimized_gliner,
        ), gw
        assert (gw["complexity"], gw["routed_to"]) == expected_gateway_routing(
            query, gw
        )
        # GLiNER only ever tags this query video_content, so the modality and
        # generation type hold under any calibrated GLiNER threshold.
        assert gw["modality"] == "video", gw
        assert gw["generation_type"] == "raw_results", gw

        # 5. Verify the artifact is still loadable in-pod after restart
        #    (proves the agent's telemetry infrastructure survived restart)
        blob_after = _load_blob_in_pod(
            "config", "gateway_thresholds", tenant_id=gateway_threshold_tenant.tenant_id
        )
        assert blob_after, "Gateway artifact not loadable after restart"
        artifact_after = json.loads(blob_after)
        assert (
            artifact_after["fast_path_confidence_threshold"] == optimized_threshold
        ), (
            f"Artifact threshold changed after restart: "
            f"{artifact_after['fast_path_confidence_threshold']} != {optimized_threshold}"
        )

    def test_simba_artifact_round_trip(self):
        """Run simba after the pod bounce: the run honours its contract against
        the artifact the earlier runs persisted, and the persisted state reads
        back identically twice (it survived the restart)."""
        blob_before = _load_blob_in_pod("model", "simba_query_enhancement")
        assert blob_before != "", "earlier SIMBA runs persisted an artifact"

        result = _run_batch_job("simba", timeout=1200)

        after = _assert_simba_served_the_best_module(result, blob_before)
        assert json.loads(_load_blob_in_pod("model", "simba_query_enhancement")) == (
            after
        )

    def test_entity_extraction_artifact_survives_restart(self):
        """Verify entity_extraction artifact is loadable after the gateway restart.

        The prior test in this class bounces the runtime pod. The new pod
        re-subscribes to Phoenix spans but the catch-up takes 30-120s on
        first read. Running the batch job immediately can land in a
        window where Phoenix returns ``training_examples=0`` (status is
        still "success" — the optimizer just had nothing to learn from).
        Retry the batch job until training examples appear OR the
        wait-budget is exhausted; this distinguishes "Phoenix-catchup
        race" (transient) from "no spans exist at all" (real bug).
        """
        deadline = time.monotonic() + 180.0
        result = _run_batch_job("entity-extraction")
        while (
            result.get("status") == "success"
            and result.get("training_examples", 0) < 1
            and time.monotonic() < deadline
        ):
            time.sleep(15)
            result = _run_batch_job("entity-extraction")
        assert result["status"] == "success", (
            f"entity-extraction batch job failed: {result}"
        )
        assert result["training_examples"] >= 1, (
            f"Phoenix returned 0 training examples after 180s of post-bounce "
            f"catch-up — either spans were never indexed or the bounce dropped "
            f"persistent data. Last result: {result}"
        )

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

        usable_profiles = _usable_profile_names_in_pod(TENANT_ID)

        # Verify demo structure: each should have query and selected_profile
        for demo in demos:
            assert demo.get("query"), f"Demo missing query: {demo}"
            assert demo.get("selected_profile") in usable_profiles, (
                f"Demo selected unknown profile '{demo.get('selected_profile')}', "
                f"expected one of {usable_profiles}"
            )


# ---------------------------------------------------------------------------
# 8. Synthetic data generation
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestSyntheticGeneration:
    """``--mode synthetic`` accepts only optimizer types with an approved
    training-data consumer (query_enhancement, profile, routing,
    entity_extraction). The valid-type end-to-end run, through to the persisted
    pending-review batch, is ``test_optimizer_persistence_e2e``."""

    def test_synthetic_mode_rejects_an_optimizer_without_a_consumer(self):
        result = subprocess.run(
            [
                "kubectl",
                "--context",
                KUBECTL_CONTEXT,
                "exec",
                "-n",
                NAMESPACE,
                DEPLOYMENT,
                "-c",
                CONTAINER,
                "--",
                "python3",
                "-m",
                "cogniverse_runtime.optimization_cli",
                "--mode",
                "synthetic",
                "--tenant-id",
                TENANT_ID,
                "--agents",
                "simba",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 1, result
        assert result.stdout.strip() == "", result.stdout
        # kubectl exec appends its own exit line after the CLI's stderr.
        stderr_lines = result.stderr.rstrip().splitlines()
        assert stderr_lines[-1] == "command terminated with exit code 1", stderr_lines
        assert stderr_lines[-2] == (
            "Error: synthetic optimizer types have no approved training-data "
            "consumer: ['simba']"
        ), result.stderr[-1000:]
        # A configuration error is a one-line message, not a traceback.
        assert "Traceback" not in result.stderr, result.stderr[-2000:]
