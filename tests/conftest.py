"""Global pytest configuration for test isolation"""

pytest_plugins = [
    "tests.fixtures.inference",
    "tests.fixtures.llm",
    "tests.fixtures.sidecars",
]

import gc
import importlib.util
import json
import os
import shutil
import socket
import tempfile
import threading
import time
from pathlib import Path

import pytest
import requests

from tests.utils.async_polling import simulate_processing_delay


@pytest.fixture(scope="session")
def face_embed_container():
    """Self-provisioned face-embed sidecar container.

    Builds the image from deploy/face_embed/Dockerfile when absent, runs
    it with the shared HF/insightface cache volume, and yields the base
    URL — integration tests never depend on a pre-started service.
    """
    import subprocess
    import time as _time

    import requests as _requests

    repo = Path(__file__).resolve().parents[1]
    # Same dev-tag scheme as the chart's sidecar builds (<appVersion>-dev),
    # so the test image sits in the versioned family instead of an ad-hoc tag.
    image = "cogniverse/face-embed:0.1.0-dev"
    have = subprocess.run(["docker", "image", "inspect", image], capture_output=True)
    if have.returncode != 0:
        subprocess.run(
            [
                "docker",
                "build",
                "-f",
                str(repo / "deploy/face_embed/Dockerfile"),
                "-t",
                image,
                str(repo),
            ],
            check=True,
            timeout=1800,
        )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    name = f"face-embed-test-{port}"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(
        ["docker", "volume", "create", "face-embed-cache"], capture_output=True
    )
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "-p",
            f"{port}:8080",
            "-v",
            "face-embed-cache:/root/.insightface",
            "--oom-score-adj=500",
            image,
        ],
        check=True,
        timeout=120,
    )

    base_url = f"http://127.0.0.1:{port}"
    deadline = _time.time() + 120
    while _time.time() < deadline:
        try:
            if _requests.get(f"{base_url}/health", timeout=2).status_code == 200:
                break
        except Exception:
            pass
        _time.sleep(2)
    else:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        pytest.fail("face-embed sidecar container did not become healthy")

    try:
        yield base_url
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.fixture(scope="session")
def pylate_server():
    """LateOn served by the real PyLate sidecar container (deploy/pylate,
    the same engine the chart deploys) exposing the production ``/pooling``
    contract — session-scoped so LateOn loads once per run.

    The service owns PyLate's exact encode for both directions: query
    expansion over masked padding positions and the document punctuation
    skiplist. Generic vLLM ``/pooling`` cannot reproduce the query side
    because its request schema carries no attention mask. Integration tests
    provision their own inference; the cluster belongs to the e2e tier.
    """
    from cogniverse_cli.modal_inference_config import get_inference_service_spec

    from tests.fixtures.inference import LocalEndpointProvider

    provider = LocalEndpointProvider()
    try:
        endpoint = provider.resolve(get_inference_service_spec("colbert_pylate"))
        yield endpoint.base_url
    finally:
        provider.close()


@pytest.fixture(scope="session")
def shared_denseon(vllm_sidecar):
    """DenseOn served by a real vLLM container exposing the
    OpenAI-compatible ``/v1/embeddings`` contract Mem0's openai provider
    expects — session-scoped so the model loads once per test run.

    Mirrors the chart's ``vllm_embed`` engine: ``--runner pooling
    --convert embed`` pools to a single dense vector per input (no
    per-token reshape), matching DenseOn's dense-retrieval semantics.
    The chart pins float32 because DenseOn can emit NaNs for ordinary
    document-prefixed text under vLLM's lower-precision CPU default.
    """
    return vllm_sidecar.spawn(
        "lightonai/DenseOn",
        extra_args=[
            "--runner",
            "pooling",
            "--convert",
            "embed",
            "--dtype",
            "float32",
        ],
    )


# Configure torch and tokenizers to avoid threading issues in pytest
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Import torch and configure threading before any tests run
try:
    import torch

    torch.set_num_threads(1)
except ImportError:
    pass


def _whisper_local_installed() -> bool:
    """True when both packages from the
    ``cogniverse-runtime[whisper-local]`` extra are importable. The
    extra is opt-in so production runtime images don't ship them —
    tests that exercise the in-process Whisper loader (or patch
    ``whisper`` / ``faster_whisper`` import targets) must skip without
    it. The vLLM ASR sidecar handles transcription in production."""
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("whisper", "faster_whisper")
    )


_STALE_LM_SKIP_REASON = "Configured LM endpoint not reachable"
_BRIGHT_STACK_SKIP_PREFIXES = (
    "bright_video_probes.csv missing at ",
    "cannot import is_llm_available helper:",
    "OrchestratorAgent import failed:",
    "Live Vespa at ",
)
_UNPROVISIONED_TEACHER_MODEL = "openai/__teacher_role_not_provisioned__"


def _mark_stale_stack_for_runtime(item) -> list[tuple[object, object]]:
    """Mark stale import-time skips for replacement after all items are scanned."""
    iter_markers = getattr(item, "iter_markers_with_node", None)
    if iter_markers is None:
        return []
    matched = []
    for node, marker in list(iter_markers(name="skipif")):
        reason = marker.kwargs.get("reason", "")
        is_bright_stack_skip = str(getattr(item, "path", "")).endswith(
            "test_bright_video_probes.py"
        ) and reason.startswith(_BRIGHT_STACK_SKIP_PREFIXES)
        is_stale_stack_skip = reason == _STALE_LM_SKIP_REASON or is_bright_stack_skip
        if not is_stale_stack_skip or not marker.args or marker.args[0] is not True:
            continue
        item.add_marker(pytest.mark.requires_lm)
        matched.append((node, marker))
    return matched


def _requests_teacher_lm(item) -> bool:
    """Return whether the item's fixture closure reads a teacher LM.

    Only fixture code counts: a fixture is how a test consumes a provisioned
    endpoint. A test body that merely reads a ``teacher_lm`` attribute on an
    object it builds itself (wiring assertions against fake endpoints) needs
    no live teacher; ``requires_teacher_model`` is the explicit opt-in for
    body-level consumers, and an unprovisioned teacher fails loudly via the
    ``_UNPROVISIONED_TEACHER_MODEL`` sentinel.
    """
    fixture_defs = getattr(
        getattr(item, "_fixtureinfo", None),
        "name2fixturedefs",
        {},
    )
    callables = [
        getattr(fixture_def, "func", None)
        for definitions in fixture_defs.values()
        for fixture_def in (definitions or ())
    ]
    return "teacher_lm" in item.fixturenames or any(
        "teacher_lm" in getattr(getattr(callable_obj, "__code__", None), "co_names", ())
        for callable_obj in callables
        if callable_obj is not None
    )


def pytest_collection_modifyitems(items):
    """Location-derived markers, selective LM setup, and whisper auto-skip.

    Tests under a ``unit/`` (``integration/``) directory get the ``unit``
    (``integration``) marker from their location — the directory IS the
    taxonomy, so a file or test that forgets the marker cannot silently fall
    out of its directory's CI ``-m`` selection. ``local_only`` opts out: it
    declares a deliberate exclusion from CI selections, so no location marker
    is added (tests/runtime/unit/test_marker_coverage.py mirrors this rule).
    """
    from tests.fixtures.markers import apply_location_markers

    apply_location_markers(items)

    stale_stack_markers = []
    for item in items:
        stale_stack_markers.extend(_mark_stale_stack_for_runtime(item))

    for item in items:
        roles: set[str] = set()
        fixture_requested = "ensure_host_ollama" in item.fixturenames
        directly_requested = "ensure_host_ollama" in item._fixtureinfo.initialnames
        if fixture_requested:
            roles.add("primary")
        if item.get_closest_marker("requires_lm") is not None:
            roles.add("primary")
        if item.get_closest_marker("requires_teacher_model") is not None:
            roles.update(("primary", "teacher"))
        if _requests_teacher_lm(item):
            roles.update(("primary", "teacher"))
        if directly_requested:
            # A direct request only pins the primary role; teacher-consuming
            # tests declare that via requires_teacher_model or a teacher_lm
            # fixture, and primary-only sessions park the teacher on a dead
            # port (see teacher_role_on_test_lm).
            roles.add("primary")
        if not roles:
            continue
        item._cogniverse_lm_roles = frozenset(roles)
        if not fixture_requested:
            item.fixturenames.insert(0, "ensure_host_ollama")

    for node, marker in stale_stack_markers:
        if marker in node.own_markers:
            node.own_markers.remove(marker)

    # Auto-skip ``requires_whisper`` tests when the whisper-local extra isn't
    # installed, mirroring the runtime image's opt-in boundary.
    if _whisper_local_installed():
        return
    skip = pytest.mark.skip(
        reason=(
            "whisper-local extra not installed; install with "
            "`uv sync --package cogniverse-runtime --extra whisper-local` "
            "or run the e2e ASR tests against the cluster sidecar"
        )
    )
    for item in items:
        if "requires_whisper" in item.keywords:
            item.add_marker(skip)


@pytest.hookimpl(trylast=True)
def pytest_runtest_setup(item):
    """Runtime LM gate for ``requires_lm``-marked tests.

    Collection injects ``ensure_host_ollama`` only for LM-marked tests.
    ``trylast=True`` lets pytest provision the selected exact endpoints before
    this hook verifies that the primary endpoint remains reachable.
    """
    from tests.fixtures.markers import enforce_lm_gate

    enforce_lm_gate(item)


def cleanup_background_threads():
    """
    Clean up background threads from tqdm (transformers) and posthog (mem0ai).

    These libraries create daemon threads that can cause segfaults during pytest
    cleanup in async tests. We need to give them time to finish and exit cleanly.
    """
    max_wait = 2.0  # seconds
    start_time = time.time()

    while time.time() - start_time < max_wait:
        background_threads = [
            t
            for t in threading.enumerate()
            if t != threading.current_thread()
            and t.daemon
            and any(name in t.name.lower() for name in ["tqdm", "posthog", "monitor"])
        ]

        if not background_threads:
            break

        # Give threads time to finish their work
        simulate_processing_delay(delay=0.1, description="test processing")

    # Force garbage collection to clean up any remaining references
    gc.collect()


@pytest.fixture(autouse=True, scope="function")
def _reset_circuit_breakers():
    """Circuit breakers are process-wide singletons keyed by dependency name;
    reset them between tests so an opened breaker in one test can't reject
    calls in the next."""
    try:
        from cogniverse_core.common.utils.circuit_breaker import CircuitBreaker

        CircuitBreaker.reset_registry()
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True, scope="function")
def cleanup_dspy_state():
    """Clean up DSPy state between tests to prevent isolation issues"""
    yield

    # Clean up any DSPy state after each test
    try:
        import dspy

        # Reset ALL DSPy settings attributes to prevent any state pollution
        if hasattr(dspy, "settings"):
            # Clear the LM
            if hasattr(dspy.settings, "lm"):
                dspy.settings.lm = None

            # Clear adapters if they exist
            if hasattr(dspy.settings, "adapter"):
                dspy.settings.adapter = None

            # Clear any other cached settings
            if hasattr(dspy.settings, "rm"):
                dspy.settings.rm = None

            # Clear experimental settings
            if hasattr(dspy.settings, "experimental"):
                dspy.settings.experimental = False

        # Clear any context stack from async tests
        if hasattr(dspy, "_context_stack"):
            if hasattr(dspy._context_stack, "clear"):
                dspy._context_stack.clear()
            elif isinstance(dspy._context_stack, list):
                dspy._context_stack.clear()

    except (ImportError, AttributeError, RuntimeError):
        # RuntimeError can occur if called from different async context
        pass

    # Clean up background threads from tqdm and posthog
    cleanup_background_threads()


@pytest.fixture(autouse=True, scope="function")
def cleanup_vlm_state():
    """Clean up VLM interface state between tests"""
    yield
    # Clean up any cached VLM instances
    try:
        from cogniverse_core.common.vlm_interface import VLMInterface

        # Clear any class-level state if it exists
        if hasattr(VLMInterface, "_instance"):
            VLMInterface._instance = None
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True, scope="session")
def test_output_dir():
    """
    Configure output directory for test artifacts (logs, databases, etc.).

    Overrides config's output_base_dir to use temporary directory.
    Automatically cleans up after test session completes.
    """
    # Create temp directory for all test artifacts
    temp_dir = tempfile.mkdtemp(prefix="cogniverse_test_")
    artifacts_dir = Path(temp_dir)

    # Override config's output_base_dir for tests
    # OutputManager reads from config.get("output_base_dir", "outputs")
    os.environ["TEST_OUTPUT_BASE_DIR"] = str(artifacts_dir)

    print(f"\n🗂️  Test output directory: {artifacts_dir}")

    yield artifacts_dir

    # Cleanup: Remove entire test output directory
    try:
        shutil.rmtree(artifacts_dir, ignore_errors=True)
        print(f"\n🧹 Cleaned up test artifacts: {artifacts_dir}")
    except Exception as e:
        print(f"\n⚠️  Failed to cleanup {artifacts_dir}: {e}")
    finally:
        os.environ.pop("TEST_OUTPUT_BASE_DIR", None)


@pytest.fixture(autouse=True, scope="function")
def cleanup_environment():
    """Clean up environment variables that might pollute tests"""
    # Save current environment
    saved_env = {}
    env_vars_to_track = [
        "VESPA_SCHEMA",
        "MLFLOW_TRACKING_URI",
        "TELEMETRY_OTLP_ENDPOINT",
    ]
    for var in env_vars_to_track:
        if var in os.environ:
            saved_env[var] = os.environ[var]

    yield

    # Restore saved environment variables
    for var in env_vars_to_track:
        if var in saved_env:
            os.environ[var] = saved_env[var]
        elif var in os.environ:
            del os.environ[var]


@pytest.fixture
def telemetry_manager_without_phoenix():
    """
    Standard telemetry manager fixture for tests that don't need real Phoenix.

    Sets up telemetry with mock endpoints - tests can use real telemetry components
    without connecting to Phoenix. Use this for unit and integration tests that
    just need telemetry configured but don't export/query real spans.
    """
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager
    from cogniverse_foundation.telemetry.registry import get_telemetry_registry

    # Reset TelemetryManager singleton AND clear provider cache
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()

    # Create config with mock endpoints (tests don't actually connect)
    config = TelemetryConfig(
        otlp_endpoint="http://localhost:24317",  # gRPC endpoint for span export
        provider_config={
            "http_endpoint": "http://localhost:26006",  # HTTP endpoint for queries
            "grpc_endpoint": "http://localhost:24317",  # gRPC endpoint (same as OTLP)
        },
        batch_config=BatchExportConfig(
            use_sync_export=True
        ),  # Synchronous export for tests
    )

    # Set as the global singleton
    manager = TelemetryManager(config=config)
    telemetry_manager_module._telemetry_manager = manager

    yield manager

    # Cleanup
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()


@pytest.fixture(scope="module")
def phoenix_container():
    """
    Start Phoenix Docker container with gRPC support for integration tests.

    Allocates per-process unique ports so concurrent pytest sweeps don't
    fight over the same Docker port bindings or rm -f each other's
    containers. ``port_offset = (os.getpid() % 1000) * 10`` gives a
    1000-process range with 10-port spacing (room for HTTP + gRPC + future).

    - HTTP port: 16006 + port_offset (range ~16006-25996)
    - gRPC port: 14317 + port_offset (range ~14317-24307)

    Yields a dict with both the resolved endpoints and the container name so
    downstream fixtures and tests can wire themselves to the actual ports
    without hardcoding 16006/14317.

    Sets TELEMETRY_OTLP_ENDPOINT/TELEMETRY_SYNC_EXPORT env vars for tests and
    resets TelemetryManager. Cleans up only this process's container on
    teardown (never rm -f's other processes' containers).
    """
    import subprocess

    import requests

    from cogniverse_foundation.telemetry.manager import TelemetryManager

    original_endpoint = os.environ.get("TELEMETRY_OTLP_ENDPOINT")
    original_sync_export = os.environ.get("TELEMETRY_SYNC_EXPORT")

    # Per-process port allocation: keeps parallel pytest sweeps from colliding.
    port_offset = (os.getpid() % 1000) * 10
    http_port = 16006 + port_offset
    grpc_port = 14317 + port_offset
    http_endpoint = f"http://localhost:{http_port}"
    grpc_endpoint = f"http://localhost:{grpc_port}"

    # Tag containers with the owning pid so we only ever clean up our own
    # leftovers — never another concurrent pytest process's container.
    container_name = f"phoenix_test_pid{os.getpid()}_{int(time.time() * 1000)}"

    # Kill leftover phoenix_test_pid<our-pid>_* containers from PRIOR runs of
    # this same pid (rare but possible if a previous run crashed). Scoping to
    # our own pid prevents stomping on parallel sweeps.
    leftover = subprocess.run(
        ["docker", "ps", "-q", "--filter", f"name=phoenix_test_pid{os.getpid()}_"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    for cid in leftover.stdout.strip().splitlines():
        subprocess.run(
            ["docker", "rm", "-f", cid],
            check=False,
            capture_output=True,
            timeout=10,
        )

    # Set environment for tests
    os.environ["TELEMETRY_OTLP_ENDPOINT"] = grpc_endpoint
    os.environ["TELEMETRY_SYNC_EXPORT"] = "true"

    # Reset TelemetryManager to pick up new env vars
    TelemetryManager.reset()

    try:
        # Start Phoenix container
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "--label",
                f"cogniverse-test-owner-pid={os.getpid()}",
                "-p",
                f"{http_port}:6006",  # HTTP port
                "-p",
                f"{grpc_port}:4317",  # gRPC port
                "-e",
                "PHOENIX_WORKING_DIR=/phoenix",
                "arizephoenix/phoenix:14.2.1",
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )

        # Wait for Phoenix to be ready
        max_wait_time = 60
        poll_interval = 2
        start_time = time.time()
        phoenix_ready = False

        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(http_endpoint, timeout=2)
                if response.status_code == 200:
                    phoenix_ready = True
                    break
            except Exception:
                pass
            time.sleep(poll_interval)

        if not phoenix_ready:
            logs_result = subprocess.run(
                ["docker", "logs", container_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            raise RuntimeError(
                f"Phoenix failed to start after {max_wait_time} seconds. Logs:\n{logs_result.stdout}\n{logs_result.stderr}"
            )

        yield {
            "container_name": container_name,
            "http_endpoint": http_endpoint,
            "grpc_endpoint": grpc_endpoint,
            # Bare host:port form (no scheme) for OTLP gRPC exporter consumers
            # like ConnectionConfig.otlp_endpoint, which expects "host:port".
            "otlp_endpoint": f"localhost:{grpc_port}",
            "http_port": http_port,
            "grpc_port": grpc_port,
        }

    finally:
        # Cleanup
        try:
            subprocess.run(
                ["docker", "stop", container_name],
                check=False,
                capture_output=True,
                timeout=30,
            )
            subprocess.run(
                ["docker", "rm", container_name],
                check=False,
                capture_output=True,
                timeout=10,
            )
        except Exception:
            try:
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    check=False,
                    capture_output=True,
                    timeout=10,
                )
            except Exception:
                pass

        # Restore environment
        if original_endpoint:
            os.environ["TELEMETRY_OTLP_ENDPOINT"] = original_endpoint
        else:
            os.environ.pop("TELEMETRY_OTLP_ENDPOINT", None)

        if original_sync_export:
            os.environ["TELEMETRY_SYNC_EXPORT"] = original_sync_export
        else:
            os.environ.pop("TELEMETRY_SYNC_EXPORT", None)


@pytest.fixture
def phoenix_client(phoenix_container):
    """Phoenix client for querying spans from Docker container"""
    from phoenix.client import Client

    return Client(base_url=phoenix_container["http_endpoint"])


@pytest.fixture
def telemetry_config_with_phoenix(phoenix_container):
    """
    Telemetry config for tests using real Phoenix Docker container.

    Depends on phoenix_container to ensure env vars are set.
    """
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )

    otlp_endpoint = os.getenv(
        "TELEMETRY_OTLP_ENDPOINT", phoenix_container["grpc_endpoint"]
    )
    config = TelemetryConfig(
        otlp_endpoint=otlp_endpoint,
        provider_config={
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["grpc_endpoint"],
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    return config


@pytest.fixture
def telemetry_manager_with_phoenix(telemetry_config_with_phoenix):
    """
    Telemetry manager for tests using real Phoenix Docker container.

    Sets up telemetry manager as global singleton for the test.
    """
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.manager import TelemetryManager
    from cogniverse_foundation.telemetry.registry import get_telemetry_registry

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()

    # Also clear evaluation registry cache to ensure evaluation providers
    # pick up the test's endpoint configuration
    from cogniverse_evaluation.providers.registry import get_evaluation_registry

    get_evaluation_registry().clear_cache()

    manager = TelemetryManager(config=telemetry_config_with_phoenix)
    telemetry_manager_module._telemetry_manager = manager

    yield manager

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()
    get_evaluation_registry().clear_cache()


# ==================== Backend Configuration Fixtures ====================


@pytest.fixture(scope="session", autouse=True)
def backend_config_env():
    """
    Set environment variables for backend configuration.

    Sets BACKEND_URL and BACKEND_PORT environment variables
    required by create_default_config_manager().

    Uses TEST_BACKEND_URL and TEST_BACKEND_PORT if available, otherwise
    defaults BACKEND_PORT to a deliberate dead sentinel (see below).

    This fixture is autouse=True so it applies to all tests automatically.
    """
    original_url = os.environ.get("BACKEND_URL")
    original_port = os.environ.get("BACKEND_PORT")

    # Dead sentinel: nothing listens here, and it is below the 40000-54544
    # test-Vespa allocation range so no test container ever binds it. A test
    # that resolves config without binding ``shared_vespa`` fails loudly here
    # — identically local and CI — instead of silently masking against an
    # ambient Vespa (a developer's k3d on :8080). Tests that need the real
    # store depend on ``shared_vespa``, which overrides this fixture (see
    # tests/backends/conftest.py).
    os.environ["BACKEND_URL"] = os.environ.get("TEST_BACKEND_URL", "http://localhost")
    os.environ["BACKEND_PORT"] = os.environ.get("TEST_BACKEND_PORT", "29071")

    yield

    # Restore original values
    if original_url is not None:
        os.environ["BACKEND_URL"] = original_url
    elif "BACKEND_URL" in os.environ:
        del os.environ["BACKEND_URL"]

    if original_port is not None:
        os.environ["BACKEND_PORT"] = original_port
    elif "BACKEND_PORT" in os.environ:
        del os.environ["BACKEND_PORT"]


@pytest.fixture(scope="session", autouse=True)
def cogniverse_test_config(backend_config_env, tmp_path_factory):
    """Point ``COGNIVERSE_CONFIG`` at an isolated production-config clone.

    The exact-model LM fixture validates the distinct configured Gemma
    endpoints and uses identical local sidecars only when necessary.
    """
    if os.environ.get("COGNIVERSE_CONFIG"):
        yield None
        return

    src_path = Path(__file__).resolve().parent.parent / "configs" / "config.json"
    if not src_path.exists():
        yield None
        return

    blob = json.loads(src_path.read_text())

    test_dir = tmp_path_factory.mktemp("cogniverse_test_config")
    schemas_link = test_dir / "schemas"
    schemas_src = (src_path.parent / "schemas").resolve()
    if schemas_src.exists() and not schemas_link.exists():
        schemas_link.symlink_to(schemas_src, target_is_directory=True)
    test_path = test_dir / "config.json"
    test_path.write_text(json.dumps(blob))

    original = os.environ.get("COGNIVERSE_CONFIG")
    os.environ["COGNIVERSE_CONFIG"] = str(test_path)

    yield str(test_path)

    if original is not None:
        os.environ["COGNIVERSE_CONFIG"] = original
    elif "COGNIVERSE_CONFIG" in os.environ:
        del os.environ["COGNIVERSE_CONFIG"]


_OLLAMA_RELEASE_BASE = "https://github.com/ollama/ollama/releases/latest/download"


def _resolve_ollama_artefact() -> str:
    import platform as _pl

    system = _pl.system()
    machine = _pl.machine().lower()
    if system == "Linux" and machine in ("x86_64", "amd64"):
        return "ollama-linux-amd64.tar.zst"
    if system == "Linux" and machine in ("aarch64", "arm64"):
        return "ollama-linux-arm64.tar.zst"
    raise RuntimeError(f"Unsupported platform for Ollama install: {system}/{machine}")


def _install_ollama_to_home() -> Path:
    """Download the Ollama binary archive into ``~/.ollama/bin/ollama``.

    No sudo required — the canary overlay test consumes this explicit
    installer without coupling the session-wide LM fixture to Ollama.
    """
    import shutil as _sh
    import subprocess as _sp
    import tempfile as _tmp
    import urllib.request as _ur

    home_root = Path.home() / ".ollama"
    home_bin = home_root / "bin"
    home_bin.mkdir(parents=True, exist_ok=True)
    bin_path = home_bin / "ollama"
    if bin_path.exists():
        return bin_path

    artefact = _resolve_ollama_artefact()
    url = f"{_OLLAMA_RELEASE_BASE}/{artefact}"
    with _tmp.TemporaryDirectory() as td:
        archive_path = Path(td) / artefact
        with _ur.urlopen(url, timeout=600) as resp, open(archive_path, "wb") as f:
            _sh.copyfileobj(resp, f)
        # Ollama ships .tar.zst; needs --zstd (tar 1.31+) or zstd | tar.
        extract_dir = Path(td) / "extracted"
        extract_dir.mkdir()
        _sp.run(
            ["tar", "--zstd", "-xf", str(archive_path), "-C", str(extract_dir)],
            check=True,
            capture_output=True,
        )
        src_bin = extract_dir / "bin" / "ollama"
        if not src_bin.exists():
            raise RuntimeError(
                f"ollama archive extracted but bin/ollama missing under "
                f"{extract_dir}; archive layout may have changed"
            )
        _sh.copy2(src_bin, bin_path)
        # Copy bundled libs (CUDA shims, llama.cpp shared libs) alongside the binary.
        src_lib = extract_dir / "lib"
        if src_lib.exists():
            dst_lib = home_root / "lib"
            if dst_lib.exists():
                _sh.rmtree(dst_lib)
            _sh.copytree(src_lib, dst_lib)

    bin_path.chmod(bin_path.stat().st_mode | 0o755)
    return bin_path


def _complete_primary_only_lm_config(config_path: Path) -> None:
    """Keep the required LMConfig shape while provisioning only the primary."""
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError) as exc:
        pytest.fail(
            f"primary-only LM activation produced unreadable config "
            f"{config_path}: {exc}",
            pytrace=False,
        )
    llm_config = config.get("llm_config")
    primary = llm_config.get("primary") if isinstance(llm_config, dict) else None
    if not isinstance(primary, dict) or not primary:
        pytest.fail(
            "primary-only LM activation produced no llm_config.primary",
            pytrace=False,
        )
    teacher = dict(primary)
    teacher["model"] = _UNPROVISIONED_TEACHER_MODEL
    llm_config["teacher"] = teacher
    pending = config_path.with_name(
        f".{config_path.name}.{threading.get_ident()}.complete.tmp"
    )
    try:
        pending.write_text(json.dumps(config, indent=2))
        os.replace(pending, config_path)
    finally:
        pending.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def ensure_host_ollama(request, cogniverse_test_config):
    """Guarantee only the exact production LM roles selected by tests."""
    from tests.utils.hermetic_llm import (
        MODEL,
        TEACHER_MODEL,
        activate_llms,
        ensure_llm,
    )

    original_config = os.environ.get("COGNIVERSE_CONFIG")
    original_api_base = os.environ.get("TEST_LLM_API_BASE")
    original_model = os.environ.get("TEST_LLM_MODEL")
    original_openai_key = os.environ.get("OPENAI_API_KEY")
    source_config = Path(
        cogniverse_test_config
        or original_config
        or Path(__file__).resolve().parent.parent / "configs" / "config.json"
    )
    required_roles = {
        role
        for item in request.session.items
        for role in getattr(item, "_cogniverse_lm_roles", ())
    }
    if not required_roles:
        required_roles = {"primary", "teacher"}
    primary_api_base = ensure_llm(model=MODEL)
    teacher_api_base = (
        ensure_llm(model=TEACHER_MODEL) if "teacher" in required_roles else None
    )
    session_config = activate_llms(
        primary_api_base,
        teacher_api_base,
        source_config=source_config,
    )
    try:
        if "teacher" not in required_roles:
            _complete_primary_only_lm_config(session_config)
        yield
    finally:
        if original_config is None:
            os.environ.pop("COGNIVERSE_CONFIG", None)
        else:
            os.environ["COGNIVERSE_CONFIG"] = original_config
        if original_api_base is None:
            os.environ.pop("TEST_LLM_API_BASE", None)
        else:
            os.environ["TEST_LLM_API_BASE"] = original_api_base
        if original_model is None:
            os.environ.pop("TEST_LLM_MODEL", None)
        else:
            os.environ["TEST_LLM_MODEL"] = original_model
        if original_openai_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = original_openai_key
        session_config.unlink(missing_ok=True)


@pytest.fixture
def config_manager(backend_config_env):
    """
    Create ConfigManager with backend store for testing.

    Requires backend_config_env fixture to set environment variables.
    """
    from cogniverse_foundation.config.utils import create_default_config_manager

    return create_default_config_manager()


@pytest.fixture
def config_manager_memory():
    """
    Create ConfigManager with in-memory store for unit testing.

    Does not require any backend infrastructure (Vespa, etc.).
    Use this for unit tests that test business logic without
    needing real backend connectivity.
    """
    from cogniverse_foundation.config.manager import ConfigManager
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    store.initialize()
    return ConfigManager(store=store)


@pytest.fixture
def workflow_store(telemetry_manager_with_phoenix):
    """Resolve a workflow store via the registry — same path production uses.

    Going through ``WorkflowStoreRegistry`` (entry-point discovery + cache)
    rather than constructing ``TelemetryWorkflowStore(...)`` directly means a
    new backend lights up here automatically once it registers against the
    ``cogniverse.workflow.stores`` entry-point group; the fixture is unchanged.

    Backed by a real Phoenix provider so the store exercises the true
    ``ArtifactManager`` → Phoenix round-trip rather than an in-memory double.
    """
    from cogniverse_core.registries import WorkflowStoreRegistry

    provider = telemetry_manager_with_phoenix.get_provider(
        tenant_id="workflow-store-test"
    )
    # Telemetry store is process-wide (multi-tenant internally); evict any
    # instance cached under a stale provider so each test gets a clean resolve.
    WorkflowStoreRegistry.clear_cache()
    store = WorkflowStoreRegistry.get(
        name="telemetry",
        config={"telemetry_provider": provider},
    )
    store.initialize()
    return store


# shared_vespa — the single canonical Vespa container for the whole sweep.
#
# Every per-package conftest that used to spawn its own Vespa Docker container
# (vespa_instance, ingestion_vespa_backend, shared_system_vespa, vespa_with_schema,
# eval_vespa_instance, shared_memory_vespa) now re-exports this fixture and uses
# tenant-scoped schemas for isolation. Vespa is multi-tenant by design — a
# unique tenant_id per test gives the same isolation as a fresh container, at a
# fraction of the RAM cost. The kernel OOM-killer was picking individual Vespa
# containers under host RAM pressure, breaking unrelated tests; one container
# pinned with --oom-score-adj=-1000 ends that class of cascade.


def _vespa_wait_for_config_ready(config_port: int, timeout: int = 120) -> bool:
    """Poll the Vespa config server until it serves /state/v1/health."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            resp = requests.get(
                f"http://localhost:{config_port}/state/v1/health", timeout=2
            )
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def _vespa_wait_for_data_port_ready(data_port: int, timeout: int = 120) -> bool:
    """Poll the Vespa data port until it serves /state/v1/health."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            resp = requests.get(
                f"http://localhost:{data_port}/state/v1/health", timeout=2
            )
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def _vespa_wait_for_query_ready(data_port: int, timeout: int = 120) -> bool:
    """Poll until the content cluster can actually serve a query.

    ``/state/v1/health`` flips to 200 as soon as the container process is up,
    but the content node can still report zero ready nodes — a query then comes
    back 503 with "Connection failure on nodes with distribution-keys". Config
    seeding queries the store immediately, so gate the yield on a real query
    returning 200 with no ``root.errors`` to keep it from racing convergence.
    """
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            resp = requests.get(
                f"http://localhost:{data_port}/search/",
                params={
                    "yql": "select * from sources * where true limit 0",
                    "timeout": "1s",
                },
                timeout=5,
            )
            if resp.status_code == 200:
                errors = resp.json().get("root", {}).get("errors", [])
                if not errors:
                    return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def _shared_vespa_application_package(metadata_schemas):
    """Build the shared test package with room above the host disk watermark."""
    from vespa.configuration.services import (
        container,
        content,
        disk,
        document,
        document_api,
        document_processing,
        documents,
        node,
        nodes,
        redundancy,
        resource_limits,
        search,
        services,
        tuning,
    )
    from vespa.package import ApplicationPackage, ServicesConfiguration

    services_vt = services(
        container(id="cogniverse_container", version="1.0")(
            search(),
            document_api(),
            document_processing(),
        ),
        content(id="cogniverse_content", version="1.0")(
            redundancy("1"),
            documents(
                *[
                    document(type=schema.name, mode=schema.mode)
                    for schema in metadata_schemas
                ]
            ),
            nodes(node(distribution_key="0", hostalias="node1")),
            tuning(resource_limits(disk("0.90"))),
        ),
        version="1.0",
    )
    return ApplicationPackage(
        name="cogniverse",
        schema=metadata_schemas,
        services_config=ServicesConfiguration(
            application_name="cogniverse",
            services_config=services_vt,
        ),
    )


def _shared_vespa_run_args(*, owner_pid: int, docker_platform: str) -> list[str]:
    """Return the isolated runtime contract for the shared Vespa container."""
    return [
        # The owner label lets the next session reap this container
        # when SIGKILL prevents the fixture's finally block.
        "--label",
        f"cogniverse-test-owner-pid={owner_pid}",
        "--platform",
        docker_platform,
        # Losing the shared Vespa mid-session breaks every downstream
        # test; transient inference sidecars are cheaper to restart.
        "--oom-score-adj=-1000",
        "--tmpfs",
        "/opt/vespa/var/db/vespa/search:rw,size=8g,uid=1000,gid=1000,mode=0755",
    ]


@pytest.fixture(scope="session")
def shared_vespa():
    """One Vespa container per pytest session, pinned against OOM-kill.

    Deploys ONLY the four metadata schemas (organization, tenant, config,
    adapter_registry) at startup. Per-test data schemas (agent_memories,
    wiki_pages, provenance, video_*, code_*, etc.) are deployed at test
    time via SchemaRegistry.deploy_schema(tenant_id, base_schema_name)
    using a unique tenant_id derived from the test's module + function
    name. Per-test teardown wipes only that tenant's schemas, leaving the
    shared Vespa otherwise untouched.

    Yields a dict::

        {
            "http_port": <int>,         # Vespa data port
            "config_port": <int>,       # Vespa config-server port
            "container_name": <str>,
            "base_url": "http://localhost:<http_port>",
        }

    Per-package conftests should re-export this fixture via::

        from tests.conftest import shared_vespa  # noqa: F401

    Tests acquire their own tenant via the per-package ``vespa_tenant``
    fixture (see ``tests/utils/vespa_test_helpers.py``).
    """
    import platform
    import subprocess

    from tests.utils.docker_utils import start_docker_container_with_port_retry

    # Reap labelled containers whose owning pytest died without teardown
    # (SIGKILL skips the finally) — a dead session's Vespa JVM holds GBs.
    # Exact-model LLM sidecars are reused across sessions and carry no owner
    # pid, so they are reclaimed by age instead: one left running for days
    # holds its weights in host RAM and starves Vespa's memory pre-flight.
    from tests.utils.vllm_sidecar import (
        reap_dead_owner_containers,
        reclaim_stale_exact_model_containers,
    )

    reap_dead_owner_containers()
    reclaim_stale_exact_model_containers()

    machine = platform.machine().lower()
    docker_platform = (
        "linux/arm64" if machine in ("arm64", "aarch64") else "linux/amd64"
    )

    container_name, http_port, config_port = start_docker_container_with_port_retry(
        "tests.conftest",
        name_prefix="backend-tests",
        image="vespaengine/vespa:8.668.5",
        container_ports=(8080, 19071),
        extra_run_args=_shared_vespa_run_args(
            owner_pid=os.getpid(), docker_platform=docker_platform
        ),
        max_attempts=5,
    )

    try:
        if not _vespa_wait_for_config_ready(config_port, timeout=120):
            pytest.fail(
                f"shared_vespa config-server (port {config_port}) not ready in 120s"
            )

        # Config-server ready != data port ready. Vespa's internal services
        # need a few seconds to wire up after the container reports config
        # readiness; without this the first deploy can race them.
        time.sleep(10)

        # Clear singleton state in case a prior session left stale references.
        from cogniverse_core.memory.manager import Mem0MemoryManager
        from cogniverse_core.registries.backend_registry import BackendRegistry

        Mem0MemoryManager._instances.clear()
        BackendRegistry._backend_instances.clear()
        BackendRegistry._shared_schema_registry = None

        # Deploy ONLY the four metadata schemas. Data schemas are per-test.
        from cogniverse_vespa.metadata_schemas import (
            create_adapter_registry_schema,
            create_config_metadata_schema,
            create_organization_metadata_schema,
            create_tenant_metadata_schema,
        )
        from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

        metadata_schemas = [
            create_organization_metadata_schema(),
            create_tenant_metadata_schema(),
            create_config_metadata_schema(),
            create_adapter_registry_schema(),
        ]
        app_package = _shared_vespa_application_package(metadata_schemas)
        schema_mgr = VespaSchemaManager(
            backend_endpoint="http://localhost",
            backend_port=config_port,
        )
        schema_mgr._deploy_package(app_package)

        if not _vespa_wait_for_data_port_ready(http_port, timeout=120):
            pytest.fail(
                f"shared_vespa data port {http_port} not ready 120s after metadata deploy"
            )

        # Health-200 != query-ready: the content node may still have zero ready
        # nodes. Gate on a real query so config seeding never races convergence.
        if not _vespa_wait_for_query_ready(http_port, timeout=120):
            pytest.fail(
                f"shared_vespa content cluster (port {http_port}) not query-ready "
                f"120s after metadata deploy"
            )

        yield {
            "http_port": http_port,
            "config_port": config_port,
            "container_name": container_name,
            "base_url": f"http://localhost:{http_port}",
        }

    finally:
        try:
            from cogniverse_core.memory.manager import Mem0MemoryManager
            from cogniverse_core.registries.backend_registry import BackendRegistry

            Mem0MemoryManager._instances.clear()
            BackendRegistry.clear_instances()
        finally:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)


@pytest.fixture(scope="session")
def seeded_config_vespa(shared_vespa):
    """``shared_vespa`` with baseline system + telemetry config seeded, and
    ``BACKEND_URL``/``BACKEND_PORT`` pointed at it.

    Tests that read config depend on this so they read real, *present* config
    from the store — never a phantom default (``get_system_config`` /
    ``get_telemetry_config`` fall back to defaults on an absent key, which
    silently hides a test that never provisioned its config) and never an
    ambient Vespa. Pure-unit tests that don't read config skip it.
    """
    from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import SystemConfig
    from cogniverse_foundation.telemetry.config import TelemetryConfig
    from cogniverse_vespa.config.config_store import VespaConfigStore

    port = shared_vespa["http_port"]
    cm = ConfigManager(
        store=VespaConfigStore(backend_url="http://localhost", backend_port=port)
    )
    cm.set_system_config(
        SystemConfig(backend_url="http://localhost", backend_port=port)
    )
    cm.set_telemetry_config(TelemetryConfig(), tenant_id=SYSTEM_TENANT_ID)

    prev = (os.environ.get("BACKEND_URL"), os.environ.get("BACKEND_PORT"))
    os.environ["BACKEND_URL"] = "http://localhost"
    os.environ["BACKEND_PORT"] = str(port)
    yield shared_vespa
    for key, value in zip(("BACKEND_URL", "BACKEND_PORT"), prev):
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)


@pytest.fixture(scope="session")
def ephemeral_k8s_cluster(tmp_path_factory):
    """Test-owned Kubernetes API server (agentless k3s in Docker).

    Yields ``{"container", "kubeconfig", "port"}`` with the ``cogniverse``
    namespace and the minimal Argo CronWorkflow CRD already installed.
    Consuming modules point ``kubectl`` at it by monkeypatching the
    ``KUBECONFIG`` env var per test, so production code that shells out to
    kubectl hits a cluster this session owns — never a developer's k3d
    cluster or its live Secrets/CronWorkflows.
    """
    from tests.utils.k8s_api_server import start_k8s_api_server, stop_k8s_api_server

    info = start_k8s_api_server(tmp_path_factory.mktemp("k8s-api-server"))
    yield info
    stop_k8s_api_server(info["container"])


@pytest.fixture(autouse=True)
def _reset_request_contextvars():
    """Reset MemoryAwareMixin's per-request ContextVars around every test.

    The artefact overlay, session id, and memory tenant id live in
    module-level ContextVars (so a dispatcher-shared agent doesn't bleed
    state across concurrent requests). In sync tests that share the
    main-thread context, one test's set() would otherwise leak into the
    next — a leaked tenant id makes every ``_current_memory_tenant_id``
    read resolve a foreign tenant instead of the instance attribute; this
    guarantees a clean baseline per test.
    """
    from cogniverse_agents import memory_aware_mixin as _m

    _m._DISPATCHED_ARTEFACT.set(None)
    _m._MEMORY_SESSION_ID.set(None)
    _m._MEMORY_TENANT_ID.set(None)
    yield
    _m._DISPATCHED_ARTEFACT.set(None)
    _m._MEMORY_SESSION_ID.set(None)
    _m._MEMORY_TENANT_ID.set(None)


@pytest.fixture(autouse=True)
def _reset_dataset_frame_cache():
    """Give every test its own ``evaluation_task`` dataset-frame memo.

    ``evaluation_task`` memoises the fetched Phoenix DataFrame in a
    module-level dict keyed by ``(provider.http_endpoint, dataset_name)`` so
    one experiment sweep fetches a dataset once across its profile x strategy
    tasks. The test Phoenix endpoint is derived from the pid, so it is
    byte-identical for the whole session while ``phoenix_container`` is
    module-scoped — a later module's task for the same dataset name hits the
    memo and returns the previous module's rows without ever contacting the
    live container. Scoping the memo to a test keeps the within-sweep reuse
    and drops the cross-test reuse.

    Goes through ``sys.modules`` rather than importing: a memo can only exist
    once something loaded the module, and importing it eagerly would put
    inspect_ai's ~0.7s import on the front of every session, including ones
    that never evaluate anything.
    """
    import sys

    def _clear():
        task_mod = sys.modules.get("cogniverse_evaluation.core.task")
        if task_mod is not None:
            task_mod._DATASET_FRAMES.clear()

    _clear()
    yield
    _clear()


@pytest.fixture(autouse=True)
def _release_dspy_config_ownership():
    """Give every test a free dspy ambient-config ownership slot.

    ``dspy.configure`` records the thread and async task that first called
    it in module globals, and rejects later calls from anywhere else. Those
    globals are never cleared when the owning task finishes, so one async
    test that configures dspy makes every subsequent ``dspy.configure`` in
    the session raise — for code paths that legitimately configure once per
    process (the runtime lifespan, the ingestion worker) that turns into a
    failure with no relation to the test that caused it. Releasing the slot
    per test keeps the ownership rule meaningful within a test while
    preventing it from leaking across the session. The bound LM itself is
    left alone; only the ownership markers are cleared.
    """
    import importlib

    # Import the MODULE: ``from dspy.dsp.utils import settings`` binds the
    # Settings singleton, and assigning the ownership names onto it would
    # silently create dead instance attributes instead of clearing the
    # module globals dspy.configure actually reads.
    dspy_settings = importlib.import_module("dspy.dsp.utils.settings")

    dspy_settings.config_owner_thread_id = None
    dspy_settings.config_owner_async_task = None
    yield
    dspy_settings.config_owner_thread_id = None
    dspy_settings.config_owner_async_task = None
