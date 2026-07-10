"""
Integration test configuration for runtime integration tests.

Provides shared Vespa Docker instance with metadata schemas deployed,
plus ConfigManager, SchemaLoader, and FastAPI TestClient fixtures
wired with real dependencies including real ColPali query encoder.
"""

import json
import logging
from pathlib import Path

import dspy
import pytest
from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.testclient import TestClient as StarletteTestClient

# Import vespa backend to trigger self-registration
import cogniverse_vespa  # noqa: F401
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.query.encoders import QueryEncoderFactory
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.llm_factory import create_dspy_lm
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
    BackendProfileConfig,
    LLMEndpointConfig,
    SystemConfig,
)
from cogniverse_foundation.config.utils import get_config
from cogniverse_runtime.a2a_executor import CogniverseAgentExecutor
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.routers import health, search
from cogniverse_vespa.config.config_store import VespaConfigStore

# Re-export the canonical session-scoped Vespa from the project root.
from tests.conftest import shared_vespa  # noqa: F401, E402
from tests.utils.llm_config import get_llm_base_url, get_llm_model

logger = logging.getLogger(__name__)

SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "configs" / "schemas"


@pytest.fixture(scope="module")
def vespa_instance(shared_vespa):  # noqa: F811
    """Compatibility shim: yields the dict shape runtime/integration tests
    expect (``http_port``, ``config_port``, ``base_url``, ``container_name``)
    backed by the project-wide ``shared_vespa``.

    Deploys two baseline schemas under tenant ``test:unit`` (which
    canonicalizes to itself, sanitising to the ``test_unit`` schema
    suffix runtime tests query): ``video_colpali_smol500_mv_frame_test_unit``
    and ``agent_memories_test_unit``. Goes through SchemaRegistry so the
    deploy merges with any other tenants' schemas already on
    shared_vespa instead of full-replacing them.

    Singleton state is cleared at fixture entry so tests don't inherit
    a backend pointing at a torn-down container from a prior module.
    """
    # Clear stale singleton state inherited from other modules.
    BackendRegistry._instance = None
    BackendRegistry._backend_instances.clear()
    BackendRegistry._shared_schema_registry = None

    # Pre-deploy the two baseline schemas via SchemaRegistry (merge-safe).
    from tests.utils.vespa_test_helpers import deploy_tenant_schema

    deploy_tenant_schema(
        shared_vespa,
        tenant_id="test:unit",
        base_schema_name="video_colpali_smol500_mv_frame",
    )
    deploy_tenant_schema(
        shared_vespa,
        tenant_id="test:unit",
        base_schema_name="agent_memories",
    )

    # Reset singletons so tests start with a clean slate (the deploy
    # above populated registry caches that may collide with what tests
    # construct themselves).
    BackendRegistry._backend_instances.clear()

    yield {
        "http_port": shared_vespa["http_port"],
        "config_port": shared_vespa["config_port"],
        "base_url": shared_vespa["base_url"],
        "container_name": shared_vespa["container_name"],
    }
    # No teardown — shared_vespa owns the container. Clear singletons
    # so the next module gets a fresh registry.
    try:
        BackendRegistry._instance = None
        BackendRegistry._backend_instances.clear()
        BackendRegistry._shared_schema_registry = None
    except Exception as cleanup_err:
        logger.warning(f"BackendRegistry cleanup failed: {cleanup_err}")


@pytest.fixture(scope="module")
def tomoro_search_url(config_manager, vllm_sidecar):
    """Spawn the ColQwen3/Tomoro vLLM sidecar and register its URL under the
    embedding service names the ColPali profiles reference. The SearchAgent
    resolves its encoder from the SYSTEM_TENANT_ID config, whose
    ``video_colpali_smol500_mv_frame`` profile (from config.json) names
    ``vllm_colpali``; the ``test_colpali`` profile names ``tomoro_embedding``.
    Register both to the one sidecar. ColQwen3 is remote-only, so the
    SearchAgent's eager encoder build needs this URL or it falls back to an
    unsupported local load and raises. Tests that dispatch the search agent
    depend on this."""
    url = vllm_sidecar.spawn(
        model="TomoroAI/tomoro-colqwen3-embed-4b",
        extra_args=[
            "--runner",
            "pooling",
            "--convert",
            "embed",
            "--max-model-len",
            "4096",
        ],
    )
    sys_cfg = config_manager.get_system_config()
    sys_cfg.inference_service_urls = dict(sys_cfg.inference_service_urls)
    sys_cfg.inference_service_urls["vllm_colpali"] = url
    sys_cfg.inference_service_urls["tomoro_embedding"] = url
    config_manager.set_system_config(sys_cfg)
    QueryEncoderFactory._encoder_cache.clear()
    yield url
    QueryEncoderFactory._encoder_cache.clear()


@pytest.fixture(scope="module")
def config_manager(vespa_instance):
    """
    ConfigManager backed by real VespaConfigStore.

    Seeds SystemConfig with backend_url/port pointing at the test Vespa container,
    and adds two test profiles for the default tenant.
    """
    store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=vespa_instance["http_port"],
    )

    cm = ConfigManager(store=store)

    system_config = SystemConfig(
        backend_url="http://localhost",
        backend_port=vespa_instance["http_port"],
    )
    cm.set_system_config(system_config)

    cm.add_backend_profile(
        BackendProfileConfig(
            profile_name="test_colpali",
            type="video",
            schema_name="video_colpali_smol500_mv_frame",
            embedding_model="TomoroAI/tomoro-colqwen3-embed-4b",
            model_loader="colpali",
            # Tomoro (qwen3_vl) is remote-only — route the query encoder
            # through the vLLM sidecar. The URL for this service name is
            # injected into SystemConfig.inference_service_urls by the
            # search test once the sidecar is spawned (tomoro_search_url).
            extra_config={"inference_services": {"embedding": "tomoro_embedding"}},
        ),
        tenant_id="test:unit",
    )
    cm.add_backend_profile(
        BackendProfileConfig(
            profile_name="test_videoprism",
            type="video",
            schema_name="video_videoprism_base_mv_chunk_30s",
            embedding_model="google/videoprism-base",
        ),
        tenant_id="test:unit",
    )

    # Tests under this module use per-test tenant_ids to isolate Mem0 /
    # Phoenix / config state between cases. Each needs the production default
    # "video_colpali_smol500_mv_frame" profile registered under its own
    # tenant so the search router can resolve the profile. The list is
    # test-ergonomics only — it has nothing to do with schema deployment,
    # which VespaBackend.deploy_schemas now discovers from Vespa directly.
    _per_test_tenants = [
        "test:unit",
        "qm_real_test",
        "force_cycle_test",
        "force_cycle_live_test",
        "xgboost_gate_test",
        "xgboost_skip_test",
    ]
    for _tenant in _per_test_tenants:
        cm.add_backend_profile(
            BackendProfileConfig(
                profile_name="video_colpali_smol500_mv_frame",
                type="video",
                schema_name="video_colpali_smol500_mv_frame",
                embedding_model="TomoroAI/tomoro-colqwen3-embed-4b",
            ),
            tenant_id=_tenant,
        )

    cm.add_backend_profile(
        BackendProfileConfig(
            profile_name="tenant_b_profile",
            type="video",
            schema_name="video_colpali_smol500_mv_frame",
            embedding_model="TomoroAI/tomoro-colqwen3-embed-4b",
        ),
        tenant_id="tenant_b",
    )

    # Seed SchemaRegistry with the baseline schemas the vespa_instance
    # fixture pre-deployed directly to Vespa. Without these entries, the
    # SchemaRegistry cache has no record of "agent_memories_test_unit" /
    # "video_colpali_smol500_mv_frame_test_unit", so when a test-time
    # deploy_schemas() call discovers them in the live cluster, the backend
    # has no schema definition to reconstruct and (correctly) refuses to
    # silently drop them. We register each (tenant_id, base_schema) pair
    # whose physical name was materialised in Vespa.
    from datetime import datetime, timezone

    from cogniverse_core.common.tenant_utils import canonical_tenant_id
    from cogniverse_sdk.interfaces.config_store import ConfigScope

    # Register under the canonical tenant id so the ConfigStore key matches
    # what deploy_schema/register_schema write (both canonicalize their
    # tenant_id). The physical Vespa schema suffix is the sanitised canonical
    # form ("test:unit" -> "test_unit").
    seed_tenant_id = "test:unit"
    seed_suffix = canonical_tenant_id(seed_tenant_id).replace(":", "_")
    baseline_schemas = [
        (
            "video_colpali_smol500_mv_frame",
            "video_colpali_smol500_mv_frame_schema.json",
        ),
        ("agent_memories", "agent_memories_schema.json"),
    ]
    for base_name, schema_filename in baseline_schemas:
        schema_path = SCHEMAS_DIR / schema_filename
        with open(schema_path) as f:
            reg_schema_json = json.load(f)
        full_name = f"{base_name}_{seed_suffix}"
        reg_schema_json["name"] = full_name
        reg_schema_json["document"]["name"] = full_name

        cm.store.set_config(
            tenant_id=seed_tenant_id,
            scope=ConfigScope.SCHEMA,
            service="schema_registry",
            config_key=f"schema_{base_name}",
            config_value={
                "tenant_id": seed_tenant_id,
                "base_schema_name": base_name,
                "full_schema_name": full_name,
                "schema_definition": json.dumps(reg_schema_json),
                "config": {},
                "deployment_time": datetime.now(timezone.utc).isoformat(),
            },
        )

    return cm


@pytest.fixture(scope="module")
def schema_loader():
    """FilesystemSchemaLoader from configs/schemas/."""
    return FilesystemSchemaLoader(SCHEMAS_DIR)


@pytest.fixture(autouse=True, scope="module")
def _set_test_backend_env(vespa_instance):
    """Set BACKEND_URL/BACKEND_PORT env vars so create_default_config_manager()
    naturally resolves to the test Vespa container. No mocks needed.

    Also resets config singletons so they pick up the new env vars
    when a new module starts with a different Vespa container.
    """
    import os

    from cogniverse_foundation.config import utils as config_utils

    original_url = os.environ.get("BACKEND_URL")
    original_port = os.environ.get("BACKEND_PORT")

    os.environ["BACKEND_URL"] = "http://localhost"
    os.environ["BACKEND_PORT"] = str(vespa_instance["http_port"])

    # Reset config singleton so it re-creates with the new env vars
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


@pytest.fixture(scope="module")
def memory_manager(vespa_instance, config_manager, schema_loader, shared_denseon):
    """Real Mem0MemoryManager backed by test Vespa Docker + denseon.

    Follows the same pattern as tests/memory/conftest.py shared_memory_vespa.
    Uses the same Vespa instance as search tests. Requires the configured LM endpoint
    for Mem0's LLM-based memory extraction; embeddings go through the
    denseon sidecar (DenseOn / 768-dim) at shared_denseon.
    """
    from cogniverse_core.memory.manager import Mem0MemoryManager

    Mem0MemoryManager._instances.clear()
    BackendRegistry._backend_instances.clear()

    # Refresh SystemConfig with the denseon URL so AgentDispatcher's
    # memory auto-init (which reads ``system_config.inference_service_urls
    # ["denseon"]``) finds the embedder. The base ``config_manager``
    # fixture seeds SystemConfig before shared_denseon is up, so the URL
    # would otherwise be missing and ``_init_agent_memory`` silently
    # skips with a warning.
    sys_cfg = config_manager.get_system_config()
    sys_cfg.inference_service_urls = dict(sys_cfg.inference_service_urls)
    sys_cfg.inference_service_urls["denseon"] = shared_denseon
    config_manager.set_system_config(sys_cfg)

    mm = Mem0MemoryManager(tenant_id="test:unit")
    mm.initialize(
        backend_host="http://localhost",
        backend_port=vespa_instance["http_port"],
        backend_config_port=vespa_instance["config_port"],
        llm_model=get_llm_model(),
        embedding_model="lightonai/DenseOn",
        llm_base_url=get_llm_base_url(),
        embedder_base_url=shared_denseon,
        config_manager=config_manager,
        schema_loader=schema_loader,
        auto_create_schema=True,
    )

    # Vespa's prepareandactivate returns before content nodes finish
    # activating new schemas. The deploy_schema calls above return True
    # but feed_data_point against the just-deployed agent_memories_test_unit
    # / provenance_test_unit schemas can fail with "Document type ... does
    # not exist" for several seconds afterwards. Probe both schemas with
    # a Document v1 GET (404 means schema-known-but-doc-absent → ready;
    # connection-error or 400 means schema not yet activated).
    import time as _t

    import requests as _req

    http_port = vespa_instance["http_port"]
    for schema in ("agent_memories_test_unit", "provenance_test_unit"):
        deadline = _t.monotonic() + 60
        while _t.monotonic() < deadline:
            try:
                resp = _req.get(
                    f"http://localhost:{http_port}/document/v1/{schema}/{schema}/docid/__readiness__",
                    timeout=5,
                )
                if resp.status_code in (200, 404):
                    break
            except _req.RequestException:
                pass
            _t.sleep(1)
        else:
            raise RuntimeError(
                f"schema {schema!r} did not become feedable on Vespa within 60s "
                f"of memory_manager.initialize() — every test that writes to "
                f"this schema will race the activation"
            )

    yield mm

    Mem0MemoryManager._instances.clear()
    BackendRegistry._backend_instances.clear()


@pytest.fixture(scope="module")
def real_telemetry(phoenix_container):
    """Module-scoped real TelemetryManager backed by Phoenix Docker.

    Sets up the global TelemetryManager singleton so that
    get_telemetry_manager() returns a real manager throughout search tests.
    """
    import os

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


@pytest.fixture(scope="module")
def search_client(vespa_instance, config_manager, schema_loader, real_telemetry):
    """
    FastAPI TestClient with search router wired to real ConfigManager and SchemaLoader.

    Only the search router is mounted (no lifespan needed).
    Dependency overrides point at the real Vespa-backed ConfigManager.
    Real TelemetryManager singleton is set up via real_telemetry fixture.
    assert_tenant_exists is patched to a no-op: these tests exercise search
    plumbing, not tenant lifecycle, and the local Vespa instance isn't
    seeded with tenant_metadata entries.
    """
    from unittest.mock import patch as _patch

    test_app = FastAPI()
    test_app.include_router(search.router, prefix="/search")

    test_app.dependency_overrides[search.get_config_manager_dependency] = lambda: (
        config_manager
    )
    test_app.dependency_overrides[search.get_schema_loader_dependency] = lambda: (
        schema_loader
    )

    async def _noop_tenant_check(tenant_id: str) -> None:
        return None

    with _patch(
        "cogniverse_runtime.routers.search.assert_tenant_exists",
        new=_noop_tenant_check,
    ):
        with TestClient(test_app) as client:
            yield client


@pytest.fixture(scope="module")
def health_client(vespa_instance, config_manager):
    """
    FastAPI TestClient with health router.

    Mounts the health router for integration testing against real registries.
    """
    test_app = FastAPI()
    test_app.include_router(health.router)

    with TestClient(test_app) as client:
        yield client


def _is_llm_available() -> bool:
    """Cheap reachability probe for the test LM provisioned by the
    session-scoped ``ensure_host_ollama`` fixture (tests/conftest.py).

    MUST NOT spawn: the ``skip_if_no_lm`` marker below calls it at module
    import (collection) time, so a model-loading call here would block
    collection of the whole suite."""
    from tests.fixtures.llm import is_test_lm_available

    return is_test_lm_available()


skip_if_no_lm = pytest.mark.skipif(
    not _is_llm_available(),
    reason="Configured LLM endpoint not reachable",
)


def _build_dspy_lm(max_tokens: int):
    """Build a DSPy LM from configs/config.json's primary endpoint.

    ``max_tokens`` is per-test: the small default (200) keeps the
    single-output agent tests fast, while planning agents (orchestrator
    ChainOfThought) need a larger budget or the structured output is
    truncated and fails to parse.
    """
    from cogniverse_foundation.config.utils import (
        create_default_config_manager as _cdcm,
    )

    cm = _cdcm()
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
        temperature=0.1,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    return create_dspy_lm(endpoint)


@pytest.fixture(scope="module")
def _dspy_lm_instance():
    """Module-scoped: create the LM once per module (expensive)."""
    return _build_dspy_lm(max_tokens=200)


@pytest.fixture(scope="module")
def _dspy_lm_planning_instance():
    """Module-scoped LM with a planning-sized token budget.

    The orchestrator's ChainOfThought planner emits reasoning plus the
    agent_sequence / parallel_steps structured fields; 200 tokens truncates
    that mid-output, so streamify can't parse it and the workflow aborts
    before the execution phase. 1024 tokens fits a real plan.
    """
    return _build_dspy_lm(max_tokens=1024)


@pytest.fixture
def dspy_lm(_dspy_lm_instance):
    """Function-scoped: re-apply dspy.configure before each test.

    The root conftest cleanup_dspy_state clears dspy.settings.lm after
    each test, so we must re-configure before every test that needs an LLM.
    """
    dspy.configure(lm=_dspy_lm_instance)
    return _dspy_lm_instance


@pytest.fixture
def dspy_lm_planning(_dspy_lm_planning_instance):
    """Function-scoped planning LM (1024 tokens) for orchestrator planning.

    Same re-configure-per-test contract as ``dspy_lm``.
    """
    dspy.configure(lm=_dspy_lm_planning_instance)
    return _dspy_lm_planning_instance


@pytest.fixture(scope="module")
def agent_registry(config_manager):
    """AgentRegistry with a search_agent registered."""
    registry = AgentRegistry(tenant_id="test:unit", config_manager=config_manager)
    registry.register_agent(
        AgentEndpoint(
            name="search_agent",
            url="http://localhost:8000",
            capabilities=["search", "video_search"],
        )
    )
    return registry


@pytest.fixture(scope="module")
def dispatcher(agent_registry, config_manager, schema_loader):
    """Real AgentDispatcher wired to real registry, config, and schema loader."""
    return AgentDispatcher(
        agent_registry=agent_registry,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )


@pytest.fixture(scope="module")
def a2a_client(dispatcher):
    """Starlette TestClient wrapping a real A2A server with InMemoryTaskStore.

    This is the production A2A stack: real executor, real task store,
    real request handler. Only the transport is in-process (TestClient).
    """
    executor = CogniverseAgentExecutor(dispatcher=dispatcher)

    card = AgentCard(
        name="Test Cogniverse",
        description="Integration test agent",
        url="http://localhost:9999/a2a",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id="search_agent",
                name="search_agent",
                description="Search for videos",
                tags=["search", "video_search"],
            ),
        ],
    )

    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=card,
        http_handler=handler,
    )

    with StarletteTestClient(server.build()) as client:
        yield client
