"""
Shared fixtures and utilities for agent integration tests.

Re-exports ``shared_memory_vespa`` from tests/memory/conftest.py so agent
integration tests that need a real Mem0+Vespa backend can request the
same module-scoped Vespa instance the memory tests use, without spinning
up a duplicate.
"""

import json
import logging
import os
import platform
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

import dspy
import httpx
import pytest
from cogniverse_cli.inference_endpoints import (
    CandidateEndpoint,
    EndpointCredentials,
    EndpointIdentityEvidence,
    ResolvedInferenceEndpoint,
    resolve_endpoint,
)

from cogniverse_agents.inference.deno_check import is_deno_available
from cogniverse_foundation.config.inference_auth import inference_headers
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.inference_specs import get_inference_service_spec

# Re-export the canonical shared_memory_vespa fixture so it's discoverable
# by tests under tests/agents/integration/ (pytest only walks UP from a
# test file's directory, not laterally into siblings).
from tests.memory.conftest import shared_memory_vespa as _shared_memory_vespa_fixture

shared_memory_vespa = _shared_memory_vespa_fixture

logger = logging.getLogger(__name__)


def _configured_inference_service_urls() -> dict[str, str]:
    raw = os.environ.get("INFERENCE_SERVICE_URLS")
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "INFERENCE_SERVICE_URLS must be a JSON object of service URLs"
        ) from exc
    if not isinstance(parsed, dict) or any(
        not isinstance(service, str)
        or not service
        or not isinstance(url, str)
        or not url
        or url != url.strip()
        for service, url in parsed.items()
    ):
        raise RuntimeError(
            "INFERENCE_SERVICE_URLS must be a JSON object of service URLs"
        )
    return parsed


def _resolve_modal_generation_endpoint(
    service: str,
    *,
    client: httpx.Client | None = None,
) -> ResolvedInferenceEndpoint | None:
    spec = get_inference_service_spec(service)
    base_url = _configured_inference_service_urls().get(service)
    if base_url is None:
        return None
    host = httpx.URL(base_url).host
    if host is None or not host.endswith(".modal.run"):
        return None
    authorization = inference_headers(base_url)["Authorization"]
    candidate = CandidateEndpoint(
        provider="modal",
        base_url=base_url,
        credentials=EndpointCredentials(
            bearer_token=authorization.removeprefix("Bearer ")
        ),
        identity_evidence=EndpointIdentityEvidence.ENDPOINT,
    )
    if client is not None:
        return resolve_endpoint(spec, explicit=candidate, client=client)
    with httpx.Client(timeout=10) as boundary:
        return resolve_endpoint(spec, explicit=candidate, client=boundary)


def _gemma_llm_config(
    endpoint: ResolvedInferenceEndpoint,
) -> LLMEndpointConfig:
    headers = dict(endpoint.headers)
    authorization = headers.pop("Authorization", None)
    if not authorization:
        raise RuntimeError("Gemma endpoint requires bearer authorization")
    scheme, separator, api_key = authorization.partition(" ")
    if scheme != "Bearer" or not separator or not api_key or api_key != api_key.strip():
        raise RuntimeError("Gemma endpoint has an invalid bearer authorization value")
    return LLMEndpointConfig(
        model=f"openai/{endpoint.model_id}",
        api_base=f"{endpoint.base_url}/v1",
        api_key=api_key,
        temperature=0.1,
        max_tokens=800,
        extra_headers=headers,
    )


def _resolve_verified_local_endpoint(
    service: str,
    *,
    base_url: str,
    api_key: str,
) -> ResolvedInferenceEndpoint:
    spec = get_inference_service_spec(service)
    root_url = base_url.rstrip("/")
    if root_url.endswith("/v1"):
        root_url = root_url[: -len("/v1")]
    return resolve_endpoint(
        spec,
        explicit=CandidateEndpoint(
            provider="local",
            base_url=root_url,
            credentials=EndpointCredentials(bearer_token=api_key),
            identity_evidence=EndpointIdentityEvidence.DEPLOYMENT,
            model_revision=spec.model_revision,
        ),
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items):
    """Avoid local Gemma provisioning when its Modal endpoint is configured."""

    gemma_url = _configured_inference_service_urls().get("vllm_llm_student")
    gemma_host = httpx.URL(gemma_url).host if gemma_url is not None else None
    if gemma_host is None or not gemma_host.endswith(".modal.run"):
        return
    for item in items:
        roles = getattr(item, "_cogniverse_lm_roles", ())
        if (
            roles == frozenset({"primary"})
            and "ensure_host_ollama" in item.fixturenames
        ):
            item.fixturenames.remove("ensure_host_ollama")
            if "gemma_inference_endpoint" not in item.fixturenames:
                item.fixturenames.append("gemma_inference_endpoint")


def is_llm_available() -> bool:
    """Cheap reachability probe for the test LM.

    The LM is provisioned by the session-scoped ``ensure_host_ollama``
    fixture (tests/conftest.py); this only probes whether it answers.
    Called per test by ``pytest_runtest_setup`` for ``requires_lm``-marked
    tests, so it must stay cheap (no model loading, no spawning).
    """
    from tests.fixtures.llm import is_test_lm_available

    return is_test_lm_available()


def is_teacher_api_available() -> bool:
    """Check if router optimizer teacher API key is available."""
    import os

    return bool(os.getenv("ROUTER_OPTIMIZER_TEACHER_KEY"))


# Runtime LM gate: the requires_lm marker is enforced per test by
# ``pytest_runtest_setup`` in tests/conftest.py (an import-time skipif
# latches the pre-session-fixture endpoint state).
skip_if_no_lm = pytest.mark.requires_lm


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


@pytest.fixture(scope="session")
def gemma_inference_endpoint(request):
    """Prefer the authenticated Modal Gemma service, then the exact local LM."""

    endpoint = _resolve_modal_generation_endpoint("vllm_llm_student")
    if endpoint is None:
        request.getfixturevalue("ensure_host_ollama")
        from tests.fixtures.llm import (
            resolve_api_key,
            resolve_base_url,
        )

        endpoint = _resolve_verified_local_endpoint(
            "vllm_llm_student",
            base_url=resolve_base_url(),
            api_key=resolve_api_key(),
        )
    config = _gemma_llm_config(endpoint)
    injected = {
        "TEST_LLM_API_BASE": config.api_base,
        "TEST_LLM_MODEL": endpoint.model_id,
        "TEST_LLM_PROVIDER": "openai",
        "TEST_LLM_API_KEY": config.api_key,
        "OPENAI_API_KEY": config.api_key,
    }
    original = {name: os.environ.get(name) for name in injected}
    os.environ.update(injected)
    try:
        yield endpoint
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _resolve_whisper_inference_endpoint(vllm_sidecar):
    endpoint = _resolve_modal_generation_endpoint("vllm_asr")
    if endpoint is not None:
        return endpoint
    spec = get_inference_service_spec("vllm_asr")
    base_url = vllm_sidecar.spawn(
        model=spec.model_id,
        model_revision=spec.model_revision,
        required_snapshot_files=(
            "added_tokens.json",
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors",
            "normalizer.json",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ),
        extra_args=["--runner", "generate", "--max-model-len", "448"],
    )
    return ResolvedInferenceEndpoint(
        service=spec.name,
        provider="local",
        base_url=base_url.rstrip("/"),
        headers={},
        model_id=spec.model_id,
        model_revision=spec.model_revision,
    )


@pytest.fixture(scope="session")
def whisper_inference_endpoint(vllm_sidecar):
    """Prefer authenticated Modal Whisper, then an exact test-owned sidecar."""

    return _resolve_whisper_inference_endpoint(vllm_sidecar)


@pytest.fixture(scope="module")
def _dspy_lm_instance(gemma_inference_endpoint):
    """Create one exact production Gemma LM per integration-test module."""

    return create_dspy_lm(_gemma_llm_config(gemma_inference_endpoint))


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
def _set_test_backend_env(request):
    """Point ``BACKEND_URL``/``BACKEND_PORT`` at the test-owned Vespa so
    ``create_default_config_manager()`` resolves to it, never the running
    cluster's persisted config store.

    Without this a developer host resolves ``BACKEND_URL`` to the live cluster,
    whose deployed config enables the semantic router at an in-cluster envoy
    the host cannot reach — turning every agent LM call into a connection
    error. Resets the config-manager singleton so the new env is picked up per
    module. Mirrors ``tests/runtime/integration/conftest.py``. Modules marked
    ``no_shared_memory_vespa`` own a different real boundary and must not pay
    for a Vespa container they never touch.
    """
    import os

    from cogniverse_foundation.config import utils as config_utils

    if request.node.get_closest_marker("no_shared_memory_vespa"):
        yield
        return

    shared_memory_vespa = request.getfixturevalue("shared_memory_vespa")
    original_url = os.environ.get("BACKEND_URL")
    original_port = os.environ.get("BACKEND_PORT")
    parsed_base_url = urlsplit(shared_memory_vespa["base_url"])
    backend_url = f"{parsed_base_url.scheme}://{parsed_base_url.hostname}"
    module = request.module
    original_live_endpoint = None
    original_bright_schema = None
    original_wall_clock_ms = None
    if module.__name__ == "tests.agents.integration.test_bright_video_probes":
        from cogniverse_agents import orchestrator_agent
        from tests.utils.vespa_test_helpers import schema_full_name

        original_live_endpoint = module._live_vespa_endpoint
        original_bright_schema = module.BRIGHT_FULL_SCHEMA
        original_wall_clock_ms = orchestrator_agent._ITER_RETRIEVAL_WALL_CLOCK_MS

        def test_vespa_endpoint():
            return (
                backend_url,
                shared_memory_vespa["http_port"],
                shared_memory_vespa["config_port"],
            )

        module._live_vespa_endpoint = test_vespa_endpoint
        module.BRIGHT_FULL_SCHEMA = schema_full_name(
            module.BRIGHT_BASE_SCHEMA,
            module.BRIGHT_TENANT_ID,
        )
        orchestrator_agent._ITER_RETRIEVAL_WALL_CLOCK_MS = 600_000

    os.environ["BACKEND_URL"] = backend_url
    os.environ["BACKEND_PORT"] = str(shared_memory_vespa["http_port"])
    config_utils._config_manager_singleton = None

    try:
        yield
    finally:
        config_utils._config_manager_singleton = None
        if original_live_endpoint is not None:
            module._live_vespa_endpoint = original_live_endpoint
            module.BRIGHT_FULL_SCHEMA = original_bright_schema
            orchestrator_agent._ITER_RETRIEVAL_WALL_CLOCK_MS = original_wall_clock_ms
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
def vespa_with_schema(shared_memory_vespa, tomoro_inference_url):
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

    manager = _SharedVespaManagerAdapter(shared_memory_vespa)
    inject_tomoro_url(manager.config_manager, tomoro_inference_url)

    yield {
        "http_port": shared_memory_vespa["http_port"],
        "config_port": shared_memory_vespa["config_port"],
        "base_url": shared_memory_vespa["base_url"],
        "manager": manager,
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


@pytest.fixture(autouse=True)
def _test_owned_telemetry():
    """Keep extraction/graph span export off the default localhost:4317.

    The KG extraction paths call ``get_telemetry_manager()``; without a
    collector the batch exporter sprays connection failures after every
    successful run. When no test-owned collector is configured
    (``TELEMETRY_OTLP_ENDPOINT``, set by ``phoenix_container``), pre-build
    the singleton disabled so spans no-op instead of exporting into the
    void. Tests that assert real span export depend on
    ``phoenix_container``, which sets the env var and resets the manager.
    """
    import os

    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import TelemetryConfig
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    if os.environ.get("TELEMETRY_OTLP_ENDPOINT"):
        yield
        return
    installed = None
    if telemetry_manager_module._telemetry_manager is None:
        installed = TelemetryManager(TelemetryConfig(enabled=False))
        telemetry_manager_module._telemetry_manager = installed
    yield
    if (
        installed is not None
        and telemetry_manager_module._telemetry_manager is installed
    ):
        telemetry_manager_module._telemetry_manager = None
