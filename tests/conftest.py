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


def _shared_denseon_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def shared_denseon():
    """Run the real deploy/pylate/server.py in mode=dense for any tests
    that need a DenseOn embedding endpoint (Mem0, semantic_embedder, etc).

    Loads ``lightonai/DenseOn`` via SentenceTransformer (~150MB
    ModernBERT-base download on first run, cached thereafter) and
    serves the OpenAI-compatible ``/v1/embeddings`` contract Mem0's
    openai provider expects. Session-scoped so the model loads once
    per test run regardless of how many test files request it.
    """
    import uvicorn

    spec = importlib.util.spec_from_file_location(
        "denseon_server_under_test", "deploy/pylate/server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    app = mod.build_app(model_name="lightonai/DenseOn", device="cpu", mode="dense")
    port = _shared_denseon_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 240
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        pytest.fail("denseon /health did not come up within 240s")

    try:
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=5)


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

    Uses non-default ports to avoid conflicts:
    - HTTP: 16006 (instead of 6006)
    - gRPC: 14317 (instead of 4317)

    Sets OTLP_ENDPOINT env var for tests and resets TelemetryManager.
    Kills any leftover containers on those ports before starting.
    """
    import subprocess

    import requests

    from cogniverse_foundation.telemetry.manager import TelemetryManager

    original_endpoint = os.environ.get("TELEMETRY_OTLP_ENDPOINT")
    original_sync_export = os.environ.get("TELEMETRY_SYNC_EXPORT")

    # Kill any leftover phoenix_test_* containers from previous runs that
    # are holding ports 16006/14317 and would cause "port already in use" (exit 125).
    leftover = subprocess.run(
        ["docker", "ps", "-q", "--filter", "name=phoenix_test_"],
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
    os.environ["TELEMETRY_OTLP_ENDPOINT"] = "http://localhost:14317"
    os.environ["TELEMETRY_SYNC_EXPORT"] = "true"

    # Reset TelemetryManager to pick up new env vars
    TelemetryManager.reset()

    container_name = f"phoenix_test_{int(time.time() * 1000)}"

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
                "16006:6006",  # HTTP port
                "-p",
                "14317:4317",  # gRPC port
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
                response = requests.get("http://localhost:16006", timeout=2)
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

        yield container_name

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

    return Client(base_url="http://localhost:16006")


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

    otlp_endpoint = os.getenv("TELEMETRY_OTLP_ENDPOINT", "localhost:4317")
    config = TelemetryConfig(
        otlp_endpoint=otlp_endpoint,
        provider_config={
            "http_endpoint": "http://localhost:16006",
            "grpc_endpoint": "http://localhost:14317",
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

    Uses TEST_BACKEND_URL and TEST_BACKEND_PORT if available,
    otherwise defaults to localhost:8080.

    This fixture is autouse=True so it applies to all tests automatically.
    """
    original_url = os.environ.get("BACKEND_URL")
    original_port = os.environ.get("BACKEND_PORT")

    # Set test values
    os.environ["BACKEND_URL"] = os.environ.get("TEST_BACKEND_URL", "http://localhost")
    os.environ["BACKEND_PORT"] = os.environ.get("TEST_BACKEND_PORT", "8080")

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

    test_model = os.environ.get("TEST_LLM_MODEL", "qwen2.5:1.5b")
    test_api_base = os.environ.get("TEST_LLM_API_BASE", "http://localhost:11434")

    prefixed = f"openai/{test_model}"

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


def _probe_ollama(base_url: str, timeout: float = 2.0) -> bool:
    import httpx

    try:
        return (
            httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=timeout).status_code
            == 200
        )
    except (httpx.HTTPError, OSError):
        return False


def _ollama_has_model(base_url: str, model: str) -> bool:
    import httpx

    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=5.0)
        if resp.status_code != 200:
            return False
        names = {
            m.get("name", "").split(":")[0] for m in resp.json().get("models") or []
        }
        return model.split(":")[0] in names
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
         ``qwen2.5:1.5b``) if missing.

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
    test_model = os.environ.get("TEST_LLM_MODEL", "qwen2.5:1.5b")

    if _probe_ollama(configured_base) and _ollama_has_model(
        configured_base, test_model
    ):
        yield
        return

    if shutil.which("ollama") is None:
        home_bin = Path.home() / ".ollama" / "bin"
        if not (home_bin / "ollama").exists():
            _install_ollama_to_home()
        os.environ["PATH"] = f"{home_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        if shutil.which("ollama") is None:
            raise RuntimeError(
                f"installed Ollama at {home_bin}/ollama but PATH update did not take effect"
            )

    # Pick a port we can actually bind to. The configured 11434 is often held
    # by the k3d serverlb on dev machines.
    from urllib.parse import urlparse

    configured_port = urlparse(configured_base).port or 11434
    if _port_bindable(configured_port):
        chosen_port = configured_port
    else:
        chosen_port = _claim_free_port()
        new_base = f"http://localhost:{chosen_port}"
        primary["api_base"] = new_base
        teacher = cfg.setdefault("llm_config", {}).setdefault("teacher", {})
        teacher["api_base"] = new_base
        cfg_path.write_text(json.dumps(cfg))

    chosen_base = f"http://localhost:{chosen_port}"
    serve_env = {**os.environ, "OLLAMA_HOST": f"127.0.0.1:{chosen_port}"}
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
            raise RuntimeError(
                f"`ollama serve` did not start answering at {chosen_base} within 30s"
            )

        if not _ollama_has_model(chosen_base, test_model):
            pull = _sp.run(
                ["ollama", "pull", test_model],
                env=serve_env,
                capture_output=True,
                timeout=900,
            )
            if pull.returncode != 0:
                raise RuntimeError(
                    f"`ollama pull {test_model}` failed: "
                    f"{pull.stderr.decode(errors='replace')[:500]}"
                )

        yield

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
def workflow_store(backend_config_env):
    """
    Create VespaWorkflowStore for testing.

    Requires backend_config_env fixture to set environment variables.
    """
    from cogniverse_vespa.workflow.workflow_store import VespaWorkflowStore

    store = VespaWorkflowStore(
        backend_url=os.environ.get("BACKEND_URL", "http://localhost"),
        backend_port=int(os.environ.get("BACKEND_PORT", "8080")),
    )
    store.initialize()
    return store
