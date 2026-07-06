"""
Shared fixtures and utilities for agent integration tests.

Re-exports ``shared_memory_vespa`` from tests/memory/conftest.py so agent
integration tests that need a real Mem0+Vespa backend can request the
same module-scoped Vespa instance the memory tests use, without spinning
up a duplicate.
"""

import logging
import os
import platform
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import dspy
import pytest

from cogniverse_agents.inference.deno_check import is_deno_available
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.config.utils import create_default_config_manager, get_config

# Re-export the canonical shared_memory_vespa fixture so it's discoverable
# by tests under tests/agents/integration/ (pytest only walks UP from a
# test file's directory, not laterally into siblings).
from tests.memory.conftest import shared_memory_vespa  # noqa: F401

logger = logging.getLogger(__name__)


def is_llm_available() -> bool:
    """Cheap reachability probe for the test LM.

    The LM is provisioned by the session-scoped ``ensure_host_ollama``
    fixture (tests/conftest.py); this only probes whether it answers. It
    MUST NOT spawn — module-level ``skipif`` markers call it at collection
    time, so a model-loading call here would block the whole collection.
    """
    from tests.fixtures.llm import is_test_lm_available

    return is_test_lm_available()


def is_teacher_api_available() -> bool:
    """Check if router optimizer teacher API key is available."""
    import os

    return bool(os.getenv("ROUTER_OPTIMIZER_TEACHER_KEY"))


skip_if_no_lm = pytest.mark.skipif(
    not is_llm_available(),
    reason="Configured LLM endpoint not reachable",
)

skip_if_no_teacher_api = pytest.mark.skipif(
    not is_teacher_api_available(),
    reason="ROUTER_OPTIMIZER_TEACHER_KEY environment variable not set",
)


_DENO_RELEASE_BASE = "https://github.com/denoland/deno/releases/latest/download"


def _resolve_deno_artefact() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Linux" and machine in ("x86_64", "amd64"):
        return "deno-x86_64-unknown-linux-gnu.zip"
    if system == "Linux" and machine in ("aarch64", "arm64"):
        return "deno-aarch64-unknown-linux-gnu.zip"
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return "deno-aarch64-apple-darwin.zip"
    if system == "Darwin" and machine in ("x86_64", "amd64"):
        return "deno-x86_64-apple-darwin.zip"
    raise RuntimeError(f"Unsupported platform for Deno install: {system}/{machine}")


def _install_deno_to_home() -> Path:
    """Download the latest Deno release zip into ~/.deno/bin/ and chmod +x.

    The cogniverse Deno probe (``is_deno_available``) already checks
    ``~/.deno/bin/deno`` and amends ``PATH`` when found, so this matches the
    install location the production code expects.
    """
    home_bin = Path.home() / ".deno" / "bin"
    home_bin.mkdir(parents=True, exist_ok=True)
    deno_path = home_bin / "deno"
    if deno_path.exists():
        return deno_path

    artefact = _resolve_deno_artefact()
    url = f"{_DENO_RELEASE_BASE}/{artefact}"
    logger.info("Downloading Deno from %s", url)
    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td) / artefact
        with (
            urllib.request.urlopen(url, timeout=120) as resp,
            open(zip_path, "wb") as f,
        ):
            shutil.copyfileobj(resp, f)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(home_bin)

    if not deno_path.exists():
        raise RuntimeError(
            f"Deno install completed but binary missing at {deno_path}; "
            f"zip layout may have changed"
        )
    deno_path.chmod(
        deno_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
    )
    return deno_path


@pytest.fixture(scope="session")
def ensure_deno() -> Path:
    """Session-scoped fixture: guarantees Deno is reachable for RLM REPL tests.

    Installs the latest Deno release into ``~/.deno/bin/`` if missing
    (single binary, no package manager). Tests use this instead of
    skipping when Deno is absent — infrastructure managed inside the
    test suite, per project policy that infra-skips count as bugs.
    """
    if is_deno_available():
        return Path(shutil.which("deno") or Path.home() / ".deno" / "bin" / "deno")
    logger.info("Deno not on PATH or in ~/.deno/bin — installing for test session")
    deno_path = _install_deno_to_home()
    if not is_deno_available():
        raise RuntimeError(
            f"Installed Deno at {deno_path} but is_deno_available() still False"
        )
    return deno_path


@pytest.fixture(scope="module")
def _dspy_lm_instance():
    """Module-scoped: create the LM once per module (expensive)."""
    cm = create_default_config_manager()
    config = get_config(tenant_id="test:unit", config_manager=cm)
    llm_cfg = config.get("llm_config", {}).get("primary", {})

    # Disable qwen3 thinking mode — it puts output in a 'thinking' field
    # that DSPy can't read, leaving content empty.
    extra_body = None
    model = llm_cfg["model"]
    if "qwen3" in model or "qwen-3" in model:
        extra_body = {"think": False}

    endpoint = LLMEndpointConfig(
        model=model,
        api_base=llm_cfg.get("api_base"),
        # Local OAI-compat LM servers accept any bearer token but litellm
        # refuses to construct an OpenAI client without an api_key set, so
        # it falls back to OPENAI_API_KEY and raises AuthenticationError
        # when neither is present. Test endpoints don't validate the key —
        # pass the cogniverse convention sentinel.
        api_key=llm_cfg.get("api_key") or "not-required",
        temperature=0.1,
        # 200 tokens was too small: synthesis / summarisation tests
        # truncate mid-sentence, masking real failures behind a
        # plausible-looking partial answer (the 3-doc synthesis test
        # would only ever surface 1 of 3 distinguishing facts before
        # hitting the cap). 800 is enough headroom for the largest
        # synthesis prompts in the test suite without slowing simple
        # Q&A tests appreciably (local LMs stop early on short answers).
        max_tokens=800,
        extra_body=extra_body,
    )
    return create_dspy_lm(endpoint)


@pytest.fixture
def dspy_lm(_dspy_lm_instance):
    """Function-scoped: re-apply dspy.configure before each test.

    The root conftest cleanup_dspy_state clears dspy.settings.lm after
    each test, so we must re-configure before every test that needs an LLM.
    """
    dspy.configure(lm=_dspy_lm_instance)
    return _dspy_lm_instance


@pytest.fixture(autouse=True)
def clear_singleton_state_between_tests():
    """
    Function-scoped autouse fixture to clear singleton state between each test.

    This prevents test isolation issues when using module-scoped fixtures like vespa_with_schema.
    Runs automatically before each test to ensure clean state.
    """
    # Clear before each test
    from cogniverse_core.registries.backend_registry import (
        BackendRegistry,
        get_backend_registry,
    )
    from cogniverse_foundation.config.manager import ConfigManager

    registry = get_backend_registry()
    if hasattr(registry, "_backend_instances"):
        initial_count = len(registry._backend_instances)
        registry._backend_instances.clear()
        if initial_count > 0:
            logger.debug(
                f"🧹 Cleared {initial_count} cached backend instances before test"
            )

    # Drop the shared SchemaRegistry singleton too. Without this, a
    # SchemaRegistry created by an earlier test against one Vespa instance
    # survives, and its captured ``_backend`` reference still points at
    # that earlier backend. The next test's new backend then inherits
    # the stale registry; any schema deploy through it hits the OLD
    # vespa endpoint (e.g. ``cogniverse-vespa`` from a k3d cluster
    # context) instead of the test's localhost vespa.
    BackendRegistry._shared_schema_registry = None

    if hasattr(ConfigManager, "_instance"):
        if ConfigManager._instance is not None:
            logger.debug("🧹 Cleared ConfigManager singleton before test")
        ConfigManager._instance = None

    import cogniverse_vespa.search_backend as _sb

    with _sb._CACHE_LOCK:
        _sb._RANKING_STRATEGIES_CACHE = None

    yield

    # Clear after each test as well
    registry = get_backend_registry()
    if hasattr(registry, "_backend_instances"):
        registry._backend_instances.clear()
    BackendRegistry._shared_schema_registry = None
    if hasattr(ConfigManager, "_instance"):
        ConfigManager._instance = None

    with _sb._CACHE_LOCK:
        _sb._RANKING_STRATEGIES_CACHE = None


@pytest.fixture(autouse=True, scope="module")
def _set_test_backend_env(shared_memory_vespa):  # noqa: F811
    """Point ``BACKEND_URL``/``BACKEND_PORT`` at the test-owned Vespa so
    ``create_default_config_manager()`` resolves to it, never the running
    cluster's persisted config store.

    Without this a developer host resolves ``BACKEND_URL`` to the live cluster,
    whose deployed config enables the semantic router at an in-cluster envoy
    the host cannot reach — turning every agent LM call into a connection
    error. Resets the config-manager singleton so the new env is picked up per
    module. Mirrors ``tests/runtime/integration/conftest.py``.
    """
    import os

    from cogniverse_foundation.config import utils as config_utils

    original_url = os.environ.get("BACKEND_URL")
    original_port = os.environ.get("BACKEND_PORT")

    os.environ["BACKEND_URL"] = "http://localhost"
    os.environ["BACKEND_PORT"] = str(shared_memory_vespa["http_port"])
    config_utils._config_manager_singleton = None

    yield

    config_utils._config_manager_singleton = None
    if original_url is not None:
        os.environ["BACKEND_URL"] = original_url
    else:
        os.environ.pop("BACKEND_URL", None)
    if original_port is not None:
        os.environ["BACKEND_PORT"] = original_port
    else:
        os.environ.pop("BACKEND_PORT", None)


class _SharedVespaManagerAdapter:
    """Drop-in replacement for VespaTestManager when consumers only need
    ``config_manager`` + ``get_backend_via_registry`` against a Vespa they
    don't own.

    The ~4 tests that consume ``vespa_with_schema["manager"]`` either read
    ``manager.config_manager`` (a ConfigManager bound to the right ports)
    or call ``manager.get_backend_via_registry(...)`` (a thin wrapper
    around BackendRegistry that injects the right port info). Both work
    against any Vespa endpoint, so we just point them at ``shared_vespa``.
    """

    def __init__(self, shared_vespa: dict):
        self._http_port = shared_vespa["http_port"]
        self._config_port = shared_vespa["config_port"]

        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_foundation.config.unified_config import SystemConfig
        from cogniverse_vespa.config.config_store import VespaConfigStore

        store = VespaConfigStore(
            backend_url="http://localhost", backend_port=self._http_port
        )
        cm = ConfigManager(store=store)
        cm.set_system_config(
            SystemConfig(backend_url="http://localhost", backend_port=self._http_port)
        )
        self.config_manager = cm
        self._schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))

    def get_backend_via_registry(
        self,
        tenant_id: str,
        config_manager,
        schema_loader=None,
        backend_type: str = "ingestion",
    ):
        """Mirror VespaTestManager.get_backend_via_registry behavior."""
        from cogniverse_core.registries.backend_registry import BackendRegistry

        backend_config = {
            "backend": {
                "url": "http://localhost",
                "port": self._http_port,
                "config_port": self._config_port,
            }
        }
        registry = BackendRegistry.get_instance()
        if backend_type == "ingestion":
            return registry.get_ingestion_backend(
                name="vespa",
                tenant_id=tenant_id,
                config=backend_config,
                config_manager=config_manager,
                schema_loader=schema_loader or self._schema_loader,
            )
        return registry.get_search_backend(
            name="vespa",
            config=backend_config,
            config_manager=config_manager,
            schema_loader=schema_loader or self._schema_loader,
        )


@pytest.fixture(scope="module")
def vespa_with_schema(shared_memory_vespa):  # noqa: F811
    """Compatibility shim: yields the dict shape the 4 consumer tests
    expect, but backed by the project-wide ``shared_vespa`` container
    (re-exported through ``shared_memory_vespa``).

    Deploys ``video_colpali_smol500_mv_frame`` for tenant ``test_tenant``
    once per module via SchemaRegistry. Consumers read ``default_schema``
    as the BASE name (``"video_colpali_smol500_mv_frame"``) and append
    ``_test_tenant`` themselves — preserved to avoid touching consumer
    code in this phase.

    The previous implementation spawned its own Vespa container with
    VespaTestManager + full_setup() (which also ingested test video
    data). Tests don't strictly need that data — the surviving assertion
    in test_orchestrator_with_search.py:388 is ``total_results >= 0``
    which holds for an empty Vespa. If a future test does need
    pre-ingested data, add it as a separate module-scoped fixture
    rather than reviving the one-container-per-module model.
    """
    import cogniverse_vespa.search_backend as _sb
    from cogniverse_core.registries.backend_registry import (
        BackendRegistry,
        get_backend_registry,
    )
    from cogniverse_foundation.config.manager import ConfigManager

    # Clear singletons (mirror the prior fixture's setup) — agents/integration
    # tests assume fresh registry state per module.
    registry = get_backend_registry()
    if hasattr(registry, "_backend_instances"):
        registry._backend_instances.clear()
    BackendRegistry._shared_schema_registry = None
    if hasattr(ConfigManager, "_instance"):
        ConfigManager._instance = None
    with _sb._CACHE_LOCK:
        _sb._RANKING_STRATEGIES_CACHE = None

    # Deploy the video schema for tenant_id="test_tenant" via the
    # canonical SchemaRegistry pathway (handles merge-with-existing
    # schemas, tenant-name normalization, ConfigStore registration).
    from tests.utils.vespa_test_helpers import deploy_tenant_schema

    deploy_tenant_schema(
        shared_memory_vespa,
        tenant_id="test_tenant",
        base_schema_name="video_colpali_smol500_mv_frame",
        config_manager=shared_memory_vespa["config_manager"],
    )

    # Reset singletons again so consumer tests don't inherit stale
    # state from the deploy above (the deploy populates registries that
    # may collide with what the test expects to construct fresh).
    if hasattr(registry, "_backend_instances"):
        registry._backend_instances.clear()

    yield {
        "http_port": shared_memory_vespa["http_port"],
        "config_port": shared_memory_vespa["config_port"],
        "base_url": shared_memory_vespa["base_url"],
        "manager": _SharedVespaManagerAdapter(shared_memory_vespa),
        # Base name (NOT tenant-scoped) — consumer tests append "_test_tenant"
        # themselves, and config.json profile lookups key on base names.
        "default_schema": "video_colpali_smol500_mv_frame",
    }
    # No teardown — shared_vespa owns the container; the deployed schema
    # stays in Vespa until session end. Per-module re-deploy is idempotent
    # at the SchemaRegistry layer (tenant-scoped registry entry already exists).


TOMORO_MODEL = "TomoroAI/tomoro-colqwen3-embed-4b"


@pytest.fixture(scope="session")
def tomoro_inference_url(vllm_sidecar):
    """Session-scoped Tomoro ColQwen3 vLLM sidecar URL.

    Tomoro (qwen3_vl) is remote-only — any SearchAgent / encoder built from
    a profile whose ``embedding_model`` is Tomoro must route its query
    encoding through this sidecar or the production factory falls back to a
    local load and hits the remote-only guard. Same ``--runner pooling
    --convert embed`` serving config the runtime / ingestion conftests use;
    cached across the session by the vllm_sidecar factory.
    """
    return vllm_sidecar.spawn(
        model=TOMORO_MODEL,
        extra_args=[
            "--runner",
            "pooling",
            "--convert",
            "embed",
            "--max-model-len",
            "4096",
        ],
    )


def inject_tomoro_url(config_manager, url: str) -> None:
    """Point ``SystemConfig.inference_service_urls['vllm_colpali']`` at ``url``.

    ``vllm_colpali`` is the service name the production config.json visual
    profiles (video_colpali / video_colqwen) reference under
    ``inference_services.embedding``; the QueryEncoderFactory resolves that
    name against this map to route Tomoro encoding remotely. Drops any
    encoder cached before the URL existed (it would be a local encoder).
    """
    from cogniverse_core.query.encoders import QueryEncoderFactory

    sys_cfg = config_manager.get_system_config()
    sys_cfg.inference_service_urls = dict(sys_cfg.inference_service_urls)
    sys_cfg.inference_service_urls["vllm_colpali"] = url
    config_manager.set_system_config(sys_cfg)
    QueryEncoderFactory._encoder_cache.clear()


@pytest.fixture(scope="module")
def real_telemetry(phoenix_container):
    """Module-scoped real TelemetryManager backed by Phoenix Docker.

    Depends on the root-conftest phoenix_container fixture which allocates
    per-pid HTTP and gRPC ports. Exposes a live TelemetryManager so agent
    telemetry span tests can emit and query real spans.
    """
    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager
    from cogniverse_foundation.telemetry.registry import get_telemetry_registry

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()

    config = TelemetryConfig(
        otlp_endpoint=os.getenv(
            "TELEMETRY_OTLP_ENDPOINT", phoenix_container["otlp_endpoint"]
        ),
        provider_config={
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["grpc_endpoint"],
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    manager = TelemetryManager(config=config)
    telemetry_manager_module._telemetry_manager = manager

    yield manager

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()
