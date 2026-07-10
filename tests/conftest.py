"""Global pytest configuration for test isolation"""

pytest_plugins = ["tests.fixtures.llm", "tests.fixtures.sidecars"]

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
    image = "cogniverse-face-embed:local"
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
def pylate_server(vllm_sidecar):
    """LateOn served by a real vLLM container exposing the production
    ``/pooling`` per-token contract — session-scoped so LateOn loads once
    per run.

    Mirrors the chart's ``vllm_token_embed`` engine: ``--runner pooling
    --convert embed`` selects vLLM's pooling runner and the
    ``ColBERTModernBertModel`` hf-override forces the multi-vector
    architecture (without it vLLM serves a plain dense ModernBert and the
    per-token outputs LateOn retrieval needs vanish). Integration tests
    provision their own inference; the cluster belongs to the e2e tier.
    """
    return vllm_sidecar.spawn(
        "lightonai/LateOn",
        extra_args=[
            "--runner",
            "pooling",
            "--convert",
            "embed",
            "--max-model-len",
            "8192",
            "--hf-overrides",
            '{"architectures": ["ColBERTModernBertModel"]}',
        ],
        # LateOn's sentence_bert_config.json declares max_seq_length=299 (an
        # ST truncation default); vLLM's CPU build clamps derived
        # max_model_len to it and rejects 8192 even though the model's
        # position table is 8192. The env var is vLLM's documented override.
        env={"VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"},
    )


@pytest.fixture(scope="session")
def shared_denseon(vllm_sidecar):
    """DenseOn served by a real vLLM container exposing the
    OpenAI-compatible ``/v1/embeddings`` contract Mem0's openai provider
    expects — session-scoped so the model loads once per test run.

    Mirrors the chart's ``vllm_embed`` engine: ``--runner pooling
    --convert embed`` pools to a single dense vector per input (no
    per-token reshape), matching DenseOn's dense-retrieval semantics.
    """
    return vllm_sidecar.spawn(
        "lightonai/DenseOn",
        extra_args=["--runner", "pooling", "--convert", "embed"],
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


def pytest_collection_modifyitems(items):
    """Auto-skip ``requires_whisper`` tests when the whisper-local
    extra isn't installed, mirroring the runtime image's opt-in
    boundary."""
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
    """Point ``COGNIVERSE_CONFIG`` at a tmp clone of ``configs/config.json``
    with ``llm_config.primary`` / ``.teacher`` rewritten to the local Ollama
    endpoint integration tests use by default.

    Production ``configs/config.json`` carries vLLM-served LLM endpoints
    (``openai/google/gemma-4-e4b-it`` at ``http://localhost:8101/v1``) that
    match the chart's vllm_llm_student/teacher pods. Local test runs hit a
    host Ollama instead; the ``ensure_host_ollama`` fixture below
    auto-installs / auto-starts Ollama and pulls the test model so
    ``http://localhost:11434`` actually answers.

    Defaults are overridable via env vars ``TEST_LLM_MODEL`` /
    ``TEST_LLM_API_BASE`` for operators who want a different LM target.

    Skipped when ``COGNIVERSE_CONFIG`` is already set externally — the
    operator wants their own config (e.g. CI matrix runs).
    """
    if os.environ.get("COGNIVERSE_CONFIG"):
        yield None
        return

    src_path = Path(__file__).resolve().parent.parent / "configs" / "config.json"
    if not src_path.exists():
        yield None
        return

    blob = json.loads(src_path.read_text())

    # When no env override is set AND the production-config vLLM
    # endpoint (``configs/config.json`` → ``llm_config.primary``) is
    # live, leave the config alone — integration tests then run against
    # the same LM as the deployed app. The Ollama install/start path
    # only fires when the production endpoint is unreachable.
    env_model = os.environ.get("TEST_LLM_MODEL")
    env_api_base = os.environ.get("TEST_LLM_API_BASE")
    cfg_primary = blob.get("llm_config", {}).get("primary", {})
    cfg_api_base = cfg_primary.get("api_base")
    cfg_model = cfg_primary.get("model")
    if (
        env_model is None
        and env_api_base is None
        and cfg_api_base
        and cfg_model
        and _probe_openai_compat(cfg_api_base)
        and _openai_compat_has_model(cfg_api_base, cfg_model)
    ):
        # Keep the production config verbatim — the OAI-compat path in
        # ``ensure_host_ollama`` will export TEST_LLM_API_BASE / _MODEL
        # to match so ``tests/fixtures/llm.py`` resolves the same target.
        test_model = (
            cfg_model[len("openai/") :]
            if cfg_model.startswith("openai/")
            else cfg_model
        )
        test_api_base = cfg_api_base
    else:
        test_model = env_model or "qwen2.5:7b"
        test_api_base = env_api_base or "http://localhost:11434"

        prefixed = f"openai/{test_model}"

        # litellm with the ``openai/`` provider prefix sends requests to
        # ``{api_base}/chat/completions``. Ollama's OAI-compat surface
        # lives at ``/v1/chat/completions``, so the api_base MUST carry
        # ``/v1`` — without it litellm hits
        # ``localhost:11434/chat/completions`` and Ollama returns a
        # literal "404 page not found". Append it here so the rest of
        # the test suite doesn't have to remember.
        if not test_api_base.rstrip("/").endswith("/v1"):
            test_api_base = test_api_base.rstrip("/") + "/v1"

        llm_cfg = blob.setdefault("llm_config", {})
        primary = llm_cfg.setdefault("primary", {})
        primary["model"] = prefixed
        primary["api_base"] = test_api_base
        teacher = llm_cfg.setdefault("teacher", {})
        teacher["model"] = prefixed
        teacher["api_base"] = test_api_base

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

    No sudo required — drops the binary into the user's home so the
    ``ollama serve`` and ``ollama pull`` calls below can find it via
    ``shutil.which`` after we prepend ``~/.ollama/bin`` to ``PATH``.
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


def _strip_v1(base_url: str) -> str:
    """Drop a trailing ``/v1`` so Ollama-native ``/api/...`` paths resolve.

    The cogniverse_test_config fixture pins api_base to end in ``/v1`` so
    litellm's openai-prefixed model strings route correctly. Ollama's
    native tag/pull endpoints live OUTSIDE ``/v1``, so probes/installs
    need the bare host:port.
    """
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return base


def _probe_ollama(base_url: str, timeout: float = 2.0) -> bool:
    import httpx

    try:
        return (
            httpx.get(f"{_strip_v1(base_url)}/api/tags", timeout=timeout).status_code
            == 200
        )
    except (httpx.HTTPError, OSError):
        return False


def _probe_openai_compat(base_url: str, timeout: float = 2.0) -> bool:
    """Return True iff the endpoint answers ``GET /v1/models`` with 200.

    vLLM and other pure OAI-compat servers (the production
    ``llm_config.primary.api_base`` at ``http://localhost:8101/v1``)
    expose ``/v1/models`` but NOT Ollama's native ``/api/tags``. When
    the operator points the test config at such a server, we must NOT
    install/start Ollama — the configured server is the LM and the
    rest of the fixture machinery (env exports) still has to run so
    downstream tests resolve the correct base.
    """
    import httpx

    try:
        return (
            httpx.get(f"{_strip_v1(base_url)}/v1/models", timeout=timeout).status_code
            == 200
        )
    except (httpx.HTTPError, OSError):
        return False


def _openai_compat_has_model(base_url: str, model: str) -> bool:
    """Return True iff ``GET /v1/models`` lists ``model`` (or its prefix).

    The configured model may carry a litellm provider prefix
    (``openai/google/gemma-4-e4b-it``); vLLM lists the bare HF id
    (``google/gemma-4-e4b-it``). Strip a single leading ``openai/`` (the
    only litellm provider whose prefix vLLM-served models use) before
    comparing.
    """
    import httpx

    bare = model[len("openai/") :] if model.startswith("openai/") else model
    try:
        resp = httpx.get(f"{_strip_v1(base_url)}/v1/models", timeout=5.0)
        if resp.status_code != 200:
            return False
        ids = {row.get("id", "") for row in (resp.json().get("data") or [])}
        return bare in ids
    except (httpx.HTTPError, OSError, ValueError):
        return False


def _ollama_has_model(base_url: str, model: str) -> bool:
    """Return True iff Ollama has *exactly* the requested model+tag pulled.

    Ollama models are tag-versioned (``qwen2.5:1.5b`` vs ``qwen2.5:7b``);
    if the requested tag is missing, ``litellm`` requests the full
    ``model:tag`` string and Ollama replies 404 even though the base
    model name matches a different-tagged pull. Match on the full
    ``name:tag`` (Ollama's ``/api/tags`` returns names with the tag
    suffix already, e.g. ``qwen2.5:7b``); when the caller asks for
    ``qwen2.5`` without a tag, default to Ollama's ``:latest`` convention.
    """
    import httpx

    wanted = model if ":" in model else f"{model}:latest"
    try:
        resp = httpx.get(f"{_strip_v1(base_url)}/api/tags", timeout=5.0)
        if resp.status_code != 200:
            return False
        names = {m.get("name", "") for m in resp.json().get("models") or []}
        return wanted in names
    except (httpx.HTTPError, OSError, ValueError):
        return False


def _claim_free_port() -> int:
    """Bind a socket to port 0, read back the OS-assigned port, then close."""
    import socket as _sock

    with _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session", autouse=True)
def ensure_host_ollama(cogniverse_test_config):
    """Guarantee an Ollama LM endpoint is reachable and has the test model.

    Integration tests default to ``http://localhost:11434`` (Ollama).
    Local dev environments often have other listeners on 11434 (e.g.
    a k3d cluster's serverlb forwarding into the cluster), so this
    fixture:
      1. If ``cogniverse_test_config`` is rewritable and 11434 is
         already taken by something that ISN'T Ollama, claims a free
         port and rewrites ``llm_config.primary``/``.teacher`` in the
         tmp-config to point at it.
      2. Installs the Ollama binary into ``~/.ollama/bin`` if missing
         (no sudo — single-file binary download).
      3. Starts ``ollama serve`` on the chosen port (via ``OLLAMA_HOST``)
         and waits for ``/api/tags`` to answer.
      4. Pulls the test model (``TEST_LLM_MODEL`` env, default
         ``qwen2.5:7b``) if missing.

    Skipped when ``cogniverse_test_config`` is None (external
    ``COGNIVERSE_CONFIG``) — that operator owns their own LM provisioning.
    """
    import shutil
    import subprocess as _sp
    import time as _t

    if cogniverse_test_config is None:
        yield
        return

    cfg_path = Path(cogniverse_test_config)
    cfg = json.loads(cfg_path.read_text())
    primary = cfg.get("llm_config", {}).get("primary", {})
    configured_base = primary.get("api_base") or "http://localhost:11434"
    # Prefer an explicit ``TEST_LLM_MODEL`` env override; otherwise fall
    # back to the model pinned in the rewritten test config (the
    # ``cogniverse_test_config`` fixture sets ``primary.model`` after
    # detecting the live OAI-compat server). Only when neither is
    # available do we drop to the historical Ollama default — that path
    # is reserved for sessions where the production endpoint is
    # unreachable. Without this preference order, a session that unsets
    # ``TEST_LLM_MODEL`` (as the BRIGHT probe test invocation does)
    # silently routed every request to a freshly-spawned Ollama instance
    # serving the wrong model and got back HTTP 404 "model not found".
    cfg_model_raw = primary.get("model")
    if os.environ.get("TEST_LLM_MODEL"):
        test_model = os.environ["TEST_LLM_MODEL"]
    elif cfg_model_raw:
        test_model = cfg_model_raw
    else:
        test_model = "qwen2.5:7b"

    # OAI-compat server (vLLM, sglang, ...) takes precedence — when the
    # configured base answers ``/v1/models`` and lists the requested
    # model, this is the production path and Ollama must not run.
    if _probe_openai_compat(configured_base) and _openai_compat_has_model(
        configured_base, test_model
    ):
        original_api_base = os.environ.get("TEST_LLM_API_BASE")
        original_model = os.environ.get("TEST_LLM_MODEL")
        original_openai_key = os.environ.get("OPENAI_API_KEY")
        os.environ["TEST_LLM_API_BASE"] = configured_base
        # Strip a leading ``openai/`` so the bare model id lands in the
        # env (vLLM lists ``google/gemma-4-e4b-it`` under ``/v1/models``;
        # ``tests/fixtures/llm.py`` re-prefixes it with the resolved
        # provider before constructing the dspy.LM).
        bare_model = (
            test_model[len("openai/") :]
            if test_model.startswith("openai/")
            else test_model
        )
        os.environ["TEST_LLM_MODEL"] = bare_model
        os.environ.setdefault("OPENAI_API_KEY", "not-required")
        try:
            yield
        finally:
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
        return

    if _probe_ollama(configured_base) and _ollama_has_model(
        configured_base, test_model
    ):
        # Configured Ollama is up and has the model. Still export the
        # env vars so ``tests/fixtures/llm.py`` helpers resolve to the
        # same endpoint instead of their hardcoded 11434 default. Pass
        # the URL WITH any trailing /v1 — make_dspy_lm sends to
        # ``{api_base}/chat/completions`` (Ollama only serves that under
        # /v1); is_test_lm_available strips /v1 internally before
        # probing /api/tags + /v1/models, so passing the full URL works
        # for both consumers.
        # OPENAI_API_KEY is also exported because litellm's openai
        # provider refuses to issue requests without one set in the
        # environment, even when api_key is passed via kwargs — Ollama
        # ignores the value but litellm requires it to be non-empty.
        original_api_base = os.environ.get("TEST_LLM_API_BASE")
        original_model = os.environ.get("TEST_LLM_MODEL")
        original_openai_key = os.environ.get("OPENAI_API_KEY")
        os.environ["TEST_LLM_API_BASE"] = configured_base
        os.environ["TEST_LLM_MODEL"] = test_model
        os.environ.setdefault("OPENAI_API_KEY", "not-required")
        try:
            yield
        finally:
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
        return

    # Soft-fail when Ollama can't be installed or started — CI runners
    # for ``unit and ci_fast`` tests don't need an LM and shouldn't pay
    # for one. Tests that DO need an LM gate themselves via
    # ``tests/fixtures/llm.is_test_lm_available`` (or pytest skipif),
    # so degrading silently here lets unit collection succeed while
    # LM-dependent tests skip with a clear reason.
    import logging as _logging

    _log = _logging.getLogger(__name__)
    if shutil.which("ollama") is None:
        home_bin = Path.home() / ".ollama" / "bin"
        if not (home_bin / "ollama").exists():
            try:
                _install_ollama_to_home()
            except Exception as exc:  # noqa: BLE001 — log + degrade
                _log.warning(
                    "ensure_host_ollama: Ollama install failed (%s); "
                    "yielding without an LM. Tests that need an LM will "
                    "skip via is_test_lm_available().",
                    exc,
                )
                yield
                return
        os.environ["PATH"] = f"{home_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        if shutil.which("ollama") is None:
            _log.warning(
                "ensure_host_ollama: Ollama binary still not on PATH after "
                "install — yielding without an LM."
            )
            yield
            return

    # Strip any ``openai/`` provider prefix before passing to
    # ``ollama pull``. ``cogniverse_test_config`` writes
    # ``model=openai/qwen2.5:7b`` into the test config so the rest of
    # the suite resolves the LM through litellm's openai provider, but
    # the Ollama CLI only accepts bare tags (``qwen2.5:7b``). The
    # OAI-compat branch above already does the same strip; this keeps
    # the two paths consistent.
    test_model_for_pull = (
        test_model[len("openai/") :] if test_model.startswith("openai/") else test_model
    )

    # Pick a port we can actually bind to. The configured 11434 is often held
    # by the k3d serverlb on dev machines.
    from urllib.parse import urlparse

    configured_port = urlparse(configured_base).port or 11434
    if _port_bindable(configured_port):
        chosen_port = configured_port
    else:
        chosen_port = _claim_free_port()
        # Preserve the /v1 suffix the cogniverse_test_config fixture pinned —
        # litellm's openai prefix sends to ``{api_base}/chat/completions``
        # and Ollama only serves that under /v1.
        new_base = f"http://localhost:{chosen_port}/v1"
        primary["api_base"] = new_base
        teacher = cfg.setdefault("llm_config", {}).setdefault("teacher", {})
        teacher["api_base"] = new_base
        cfg_path.write_text(json.dumps(cfg))

    chosen_base = f"http://localhost:{chosen_port}"
    serve_env = {
        **os.environ,
        "OLLAMA_HOST": f"127.0.0.1:{chosen_port}",
        # Unload the model from RAM as soon as a request finishes. The
        # 7b model is ~5GB resident; leaving it warm starves the vllm
        # CPU sidecars that other tests in the same sweep need (their
        # health-check fails with "Available memory ... less than
        # desired CPU memory utilization"). Reload latency on the next
        # request is a few seconds — acceptable for tests.
        "OLLAMA_KEEP_ALIVE": "0s",
    }
    serve_proc = _sp.Popen(
        ["ollama", "serve"],
        env=serve_env,
        stdout=_sp.DEVNULL,
        stderr=_sp.DEVNULL,
    )
    try:
        ready = False
        for _ in range(30):
            if _probe_ollama(chosen_base, timeout=1.0):
                ready = True
                break
            _t.sleep(1)
        if not ready:
            _log.warning(
                "ensure_host_ollama: ollama serve did not answer at %s "
                "within 30s — yielding without an LM.",
                chosen_base,
            )
            yield
            return

        if not _ollama_has_model(chosen_base, test_model_for_pull):
            pull = _sp.run(
                ["ollama", "pull", test_model_for_pull],
                env=serve_env,
                capture_output=True,
                timeout=900,
            )
            if pull.returncode != 0:
                _log.warning(
                    "ensure_host_ollama: `ollama pull %s` failed (%s) — "
                    "yielding without an LM. Tests that need an LM will "
                    "skip via is_test_lm_available().",
                    test_model_for_pull,
                    pull.stderr.decode(errors="replace")[:200],
                )
                yield
                return

        # Also export TEST_LLM_API_BASE / TEST_LLM_MODEL so the
        # ``tests/fixtures/llm.py`` helpers (resolve_base_url,
        # is_test_lm_available) used by tests that don't read the JSON
        # config resolve to the same dynamic Ollama port. Without this,
        # those tests fall back to the hardcoded ``http://localhost:11434``
        # default and fail when k3d's serverlb owns 11434. Include the
        # ``/v1`` suffix because litellm's openai prefix sends requests
        # to ``{api_base}/chat/completions`` and Ollama only serves that
        # path under /v1; is_test_lm_available strips /v1 internally
        # before probing /api/tags + /v1/models, so it works for both.
        # OPENAI_API_KEY is also exported because litellm refuses to
        # issue without one in the env even when api_key is passed.
        original_api_base = os.environ.get("TEST_LLM_API_BASE")
        original_model = os.environ.get("TEST_LLM_MODEL")
        original_openai_key = os.environ.get("OPENAI_API_KEY")
        os.environ["TEST_LLM_API_BASE"] = f"{chosen_base}/v1"
        # Export the bare model tag (no ``openai/`` prefix) — tests/
        # fixtures/llm.py re-prefixes it with the resolved provider
        # before constructing the dspy.LM, mirroring the OAI-compat
        # branch above.
        os.environ["TEST_LLM_MODEL"] = test_model_for_pull
        os.environ.setdefault("OPENAI_API_KEY", "not-required")

        try:
            yield
        finally:
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

    finally:
        serve_proc.terminate()
        try:
            serve_proc.wait(timeout=5)
        except _sp.TimeoutExpired:
            serve_proc.kill()


def _port_bindable(port: int) -> bool:
    import socket as _sock

    try:
        with _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM) as s:
            s.setsockopt(_sock.SOL_SOCKET, _sock.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))
            return True
    except OSError:
        return False


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


def _vespa_cleanup_my_container(container_name: str) -> None:
    """Remove only OUR container by exact name, not any container that
    happens to share the prefix.

    Earlier this helper used the prefix filter ``name=^backend-tests-``
    and killed every matching container. That blew away in-use containers
    when pytest re-evaluated the fixture mid-session (e.g. after a
    transient setup failure), turning one bad fixture call into a
    cascade of failed tests downstream. Exact name keeps the blast radius
    to what we own."""
    import subprocess

    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        timeout=30,
    )


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

    from tests.utils.docker_utils import generate_unique_ports

    http_port, config_port = generate_unique_ports("tests.conftest")
    container_name = f"backend-tests-{http_port}"

    # Only remove OUR exact container if a prior crashed pytest left it
    # behind. Don't touch other backend-tests-* containers — they belong
    # to concurrent sessions or other users.
    _vespa_cleanup_my_container(container_name)

    # Reap labelled containers whose owning pytest died without teardown
    # (SIGKILL skips the finally) — a dead session's Vespa JVM holds GBs.
    from tests.utils.vllm_sidecar import reap_dead_owner_containers

    reap_dead_owner_containers()

    machine = platform.machine().lower()
    docker_platform = (
        "linux/arm64" if machine in ("arm64", "aarch64") else "linux/amd64"
    )

    # --oom-score-adj=-1000 makes the kernel pick literally anything else
    # before this container under memory pressure. Losing the shared Vespa
    # mid-session breaks every downstream test; the per-test sidecars
    # (vllm, pylate) are cheaper to lose and restart.
    result = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "-p",
            f"{http_port}:8080",
            "-p",
            f"{config_port}:19071",
            "--platform",
            docker_platform,
            "--oom-score-adj=-1000",
            "vespaengine/vespa:8.668.5",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to start shared_vespa container: {result.stderr}")

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
        from vespa.package import ApplicationPackage

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
        app_package = ApplicationPackage(name="cogniverse", schema=metadata_schemas)
        schema_mgr = VespaSchemaManager(
            backend_endpoint="http://localhost",
            backend_port=config_port,
        )
        schema_mgr._deploy_package(app_package)

        if not _vespa_wait_for_data_port_ready(http_port, timeout=120):
            pytest.fail(
                f"shared_vespa data port {http_port} not ready 120s after metadata deploy"
            )

        yield {
            "http_port": http_port,
            "config_port": config_port,
            "container_name": container_name,
            "base_url": f"http://localhost:{http_port}",
        }

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


@pytest.fixture(autouse=True)
def _reset_request_contextvars():
    """Reset MemoryAwareMixin's per-request ContextVars around every test.

    The artefact overlay + session id moved to module-level ContextVars (so a
    dispatcher-shared agent doesn't bleed state across concurrent requests). In
    sync tests that share the main-thread context, one test's set() would
    otherwise leak into the next; this guarantees a clean baseline per test.
    """
    from cogniverse_agents import memory_aware_mixin as _m

    _m._DISPATCHED_ARTEFACT.set(None)
    _m._MEMORY_SESSION_ID.set(None)
    yield
    _m._DISPATCHED_ARTEFACT.set(None)
    _m._MEMORY_SESSION_ID.set(None)
