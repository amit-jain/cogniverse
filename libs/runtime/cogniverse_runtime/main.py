"""Unified FastAPI Runtime - Single entry point for all Cogniverse services.

This replaces 10+ scattered FastAPI apps with a single, unified runtime that:
- Dynamically loads backends/agents from configs/config.json
- Consolidates all endpoints under one service
- Enables clean deployment and scaling
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.config.utils import get_config

# Import routers
from cogniverse_runtime.admin import tenant_manager
from cogniverse_runtime.config_loader import get_config_loader
from cogniverse_runtime.routers import (
    admin,
    agents,
    debug,
    events,
    graph,
    health,
    ingestion,
    search,
    tenant,
    wiki,
)
from cogniverse_synthetic.api import router as synthetic_router

logger = logging.getLogger(__name__)


def _probe_phoenix_reachability() -> None:
    """Verify the TelemetryManager can actually emit a span at startup.

    TelemetryManager falls back to ``NoOpSpan`` when Phoenix is unreachable,
    which would leave observability dashboards empty with no signal in the
    runtime logs. This probe emits a real span via the global manager once
    at startup and surfaces the result.

    Behaviour:
    - If the probe succeeds: log INFO with the configured endpoint.
    - If it fails AND ``TELEMETRY_REQUIRED=true`` is set: raise
      ``RuntimeError`` to fail-fast at startup. Operators set this in
      production deployments where missing telemetry is a deploy-blocker.
    - If it fails and the env var is unset: log WARNING with the error
      and continue. Local development should not require Phoenix.

    Extracted into a helper so it can be unit-tested without spinning up
    the full FastAPI lifespan.
    """
    required = os.environ.get("TELEMETRY_REQUIRED", "").lower() in (
        "true",
        "1",
        "yes",
    )
    try:
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        tm = get_telemetry_manager()
        if not tm.config.enabled:
            logger.info(
                "Telemetry disabled in config — skipping Phoenix reachability probe"
            )
            return

        with tm.span("startup.probe", tenant_id=SYSTEM_TENANT_ID) as span:
            # NoOpSpan has no record_exception/set_attribute side effects;
            # if we got a real span we set an attribute to force any error
            # in the export pipeline to surface here rather than later.
            if hasattr(span, "set_attribute"):
                span.set_attribute("startup.probe", True)

        logger.info(
            f"Phoenix reachability probe OK (otlp={tm.config.otlp_endpoint})"
        )
    except Exception as exc:
        msg = (
            f"Phoenix reachability probe FAILED: {exc}. "
            "Telemetry spans will fall back to NoOpSpan and dashboards "
            "will be empty until this is fixed."
        )
        if required:
            raise RuntimeError(
                f"{msg} (TELEMETRY_REQUIRED=true is set, refusing to start)"
            )
        logger.warning(msg)


def _wire_argo_from_environment() -> None:
    """Wire the tenant router's Argo config from environment variables.

    Populates ``tenant._argo_api_url`` so ``POST /{tenant}/jobs`` submits
    the CronWorkflow step. Extracted into a helper so it can be unit-tested
    without spinning up the whole FastAPI lifespan.
    """
    argo_api_url = os.environ.get("ARGO_API_URL") or None
    argo_namespace = os.environ.get("ARGO_NAMESPACE", "cogniverse")
    tenant.set_argo_config(api_url=argo_api_url, namespace=argo_namespace)
    if argo_api_url:
        logger.info(
            f"Argo CronWorkflow submission enabled (url={argo_api_url}, "
            f"namespace={argo_namespace})"
        )
    else:
        logger.warning(
            "ARGO_API_URL env var not set — scheduled jobs will be persisted "
            "but never trigger. Set ARGO_API_URL in deployment to enable."
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifecycle manager for FastAPI app - handles startup and shutdown."""

    # Startup
    import time as _time

    # Bound the default asyncio executor — every ``asyncio.to_thread`` /
    # ``run_in_executor(None, ...)`` in the codebase shares this pool.
    # Without a cap, bursts of sync work (agent instantiation, Mem0 HTTP,
    # wiki auto-file, GLiNER predict) spawn up to 32 workers and
    # third-party libraries pile on more, landing the runtime north of
    # 200 Python threads under e2e load. GIL contention then starves the
    # event loop past the readiness probe timeout.
    from concurrent.futures import ThreadPoolExecutor

    _event_loop = asyncio.get_running_loop()
    _event_loop.set_default_executor(
        ThreadPoolExecutor(max_workers=16, thread_name_prefix="cv-worker")
    )
    logger.info("Default asyncio executor capped at 16 workers")

    logger.info("Starting Cogniverse Runtime...")

    # 1. Wait for backend to be reachable before loading config.
    # In k8s, the runtime pod may start before Vespa is ready.
    from cogniverse_foundation.config.bootstrap import BootstrapConfig

    bootstrap = BootstrapConfig.from_environment()
    import httpx

    vespa_base = f"{bootstrap.backend_url}:{bootstrap.backend_port}"
    # Vespa two-port architecture: container node (GET) converges before
    # content/distributor nodes (PUT/feed). We need feed readiness, so
    # probe with a document GET that exercises the content node path.
    vespa_feed_probe = f"{vespa_base}/document/v1/config_metadata/config_metadata/docid/probe"
    logger.info(f"Waiting for backend feed readiness at {vespa_base}...")

    for attempt in range(60):
        try:
            # First check container node is up
            resp = httpx.get(f"{vespa_base}/ApplicationStatus", timeout=5)
            if resp.status_code != 200:
                raise ConnectionError("Container node not ready")
            # Then check feed path is ready (404 = schema exists, feed works;
            # 200 = doc exists; both mean feed is ready)
            resp = httpx.get(vespa_feed_probe, timeout=5)
            if resp.status_code in (200, 404):
                logger.info("Backend feed endpoint is ready")
                break
        except (httpx.HTTPError, OSError, ConnectionError):
            pass
        logger.info(f"Backend not ready, retrying ({attempt + 1}/60)...")
        _time.sleep(5)
    else:
        logger.warning("Backend not ready after 5 minutes, proceeding anyway")

    # 2. Load configuration
    from pathlib import Path

    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.utils import create_default_config_manager

    config_manager = create_default_config_manager()
    # Wire profile-change propagation: when /admin/profiles adds or removes
    # a backend profile, push the update into live search-backend instances
    # via BackendRegistry so the change is queryable without a pod restart.
    def _profile_change_listener(
        event: str, profile_name: str, profile_config
    ) -> None:
        if event == "added" and profile_config is not None:
            BackendRegistry.add_profile_to_backends(profile_name, profile_config)
        elif event == "removed":
            BackendRegistry.remove_profile_from_backends(profile_name)

    config_manager.set_profile_change_listener(_profile_change_listener)
    # SystemConfig is cluster-wide, not user-tenant-specific; scope it under
    # the reserved SYSTEM_TENANT_ID so it can't collide with a user tenant.
    config = get_config(tenant_id=SYSTEM_TENANT_ID, config_manager=config_manager)
    logger.info(f"Loaded configuration for tenant: {config.tenant_id}")

    # 3. Initialize SchemaLoader
    schema_loader = FilesystemSchemaLoader(Path("configs/schemas"))
    logger.info("SchemaLoader initialized")

    # 3. Set dependencies on routers
    admin.set_config_manager(config_manager)
    admin.set_schema_loader(schema_loader)
    tenant.set_config_manager(config_manager)
    _wire_argo_from_environment()

    # Wire ingestion and search routers via FastAPI dependency overrides
    app.dependency_overrides[ingestion.get_config_manager_dependency] = (
        lambda: config_manager
    )
    app.dependency_overrides[ingestion.get_schema_loader_dependency] = (
        lambda: schema_loader
    )
    app.dependency_overrides[search.get_config_manager_dependency] = (
        lambda: config_manager
    )
    app.dependency_overrides[search.get_schema_loader_dependency] = (
        lambda: schema_loader
    )
    logger.info("Router dependencies configured")

    # 4. Initialize registries
    backend_registry = BackendRegistry.get_instance()
    # Startup registry lookup is cluster-scope (no request tenant yet);
    # per-request code creates tenant-scoped registries via the
    # dispatcher.
    agent_registry = AgentRegistry(
        tenant_id=SYSTEM_TENANT_ID, config_manager=config_manager
    )
    logger.info("Registries initialized")

    # 5. Initialize SandboxManager (optional — gracefully degrades)
    from cogniverse_runtime.sandbox_manager import SandboxManager

    sandbox_enabled = (
        config.get("sandbox", {}).get("enabled", False)
        or os.environ.get("COGNIVERSE_SANDBOX_ENABLED", "").lower() in ("true", "1", "yes")
        or bool(os.environ.get("OPENSHELL_GATEWAY_ENDPOINT"))
    )
    sandbox_manager = SandboxManager(enabled=sandbox_enabled)

    # 5a. Wire agent registry and dependencies to agents router + A2A
    agents.set_agent_registry(agent_registry)
    agents.set_agent_dependencies(config_manager, schema_loader)
    agents.set_sandbox_manager(sandbox_manager)
    logger.info("AgentRegistry and dependencies wired to agents router")

    # 5b. Mount A2A protocol server (JSON-RPC 2.0)
    from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.tasks import InMemoryTaskStore
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill

    from cogniverse_runtime.a2a_executor import CogniverseAgentExecutor

    dispatcher = agents.get_dispatcher()
    executor = CogniverseAgentExecutor(dispatcher=dispatcher)

    skills = [
        AgentSkill(
            id=name,
            name=name,
            description=f"Agent: {name} ({', '.join(agent_registry.get_agent(name).capabilities)})",
            tags=list(agent_registry.get_agent(name).capabilities),
        )
        for name in agent_registry.list_agents()
        if agent_registry.get_agent(name) is not None
    ]

    agent_card = AgentCard(
        name="Cogniverse Runtime",
        description="Multi-agent AI platform for content intelligence",
        url="http://localhost:8000/a2a",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=skills
        or [
            AgentSkill(
                id="default",
                name="default",
                description="Default agent skill",
                tags=["general"],
            )
        ],
    )

    a2a_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    a2a_server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=a2a_handler,
    )
    app.mount("/a2a", a2a_server.build())
    logger.info(f"A2A server mounted at /a2a with {len(skills)} skills")

    # 6. Use config loader to dynamically load backends and agents
    config_loader = get_config_loader()
    config_loader.load_backends()
    config_loader.load_agents(agent_registry=agent_registry)

    logger.info(
        f"Loaded {len(backend_registry.list_backends())} backends, "
        f"{len(agent_registry.list_agents())} agents"
    )

    # 7. Create system backend and deploy metadata schemas
    from cogniverse_foundation.config.bootstrap import BootstrapConfig

    bootstrap = BootstrapConfig.from_environment()
    system_backend = BackendRegistry.get_instance().get_ingestion_backend(
        name=bootstrap.backend_type,
        tenant_id="system",
        config={
            "backend": {"url": bootstrap.backend_url, "port": bootstrap.backend_port}
        },
        config_manager=config_manager,
        schema_loader=schema_loader,
    )

    # Deploy metadata schemas once at startup (not in every VespaBackend.__init__)
    system_config = config_manager.get_system_config()
    system_backend.schema_manager.upload_metadata_schemas(
        app_name=system_config.application_name
    )
    logger.info("Metadata schemas deployed via system backend")

    # Store SystemConfig with env var overrides so all components
    # (search backend, agents, dashboard) read the correct service URLs.
    # Env vars are set by the deployment layer (Helm template).
    updated = False
    if os.environ.get("BACKEND_URL"):
        system_config.backend_url = os.environ["BACKEND_URL"]
        updated = True
    if os.environ.get("BACKEND_PORT"):
        system_config.backend_port = int(os.environ["BACKEND_PORT"])
        updated = True
    if os.environ.get("LLM_ENDPOINT"):
        system_config.base_url = os.environ["LLM_ENDPOINT"]
        updated = True
    if os.environ.get("TELEMETRY_HTTP_ENDPOINT"):
        system_config.telemetry_url = os.environ["TELEMETRY_HTTP_ENDPOINT"]
        updated = True
    if os.environ.get("TELEMETRY_OTLP_ENDPOINT"):
        system_config.telemetry_collector_endpoint = os.environ["TELEMETRY_OTLP_ENDPOINT"]
        updated = True
    if os.environ.get("RUNTIME_URL"):
        system_config.agent_registry_url = os.environ["RUNTIME_URL"]
        updated = True
    # Inference URLs: set from env if present, clear if absent (prevents
    # stale URLs from previous deployments persisting in Vespa config).
    # COLPALI keeps its dedicated var; ColBERT-family services are carried
    # in the INFERENCE_SERVICE_URLS JSON dict keyed by service name.
    new_colpali = os.environ.get("COLPALI_INFERENCE_URL", "")
    new_service_urls_json = os.environ.get("INFERENCE_SERVICE_URLS", "")
    if new_service_urls_json:
        try:
            new_service_urls = json.loads(new_service_urls_json)
            if not isinstance(new_service_urls, dict):
                raise ValueError("INFERENCE_SERVICE_URLS must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.error(
                f"Invalid INFERENCE_SERVICE_URLS env var: {exc}. "
                f"Expected JSON object like {{'general': 'http://...'}}"
            )
            raise
    else:
        new_service_urls = {}
    if new_colpali != system_config.colpali_inference_url:
        system_config.colpali_inference_url = new_colpali
        updated = True
    if new_service_urls != system_config.inference_service_urls:
        system_config.inference_service_urls = new_service_urls
        updated = True
    if updated:
        config_manager.set_system_config(system_config)
        BackendRegistry.get_instance()._backend_instances.clear()
        logger.info("SystemConfig stored with deployment env var overrides")

    # 7b. Propagate telemetry OTLP endpoint to the TelemetryManager singleton
    # (created during load_backends/load_agents above, before env overrides)
    if os.environ.get("TELEMETRY_OTLP_ENDPOINT"):
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        tm = get_telemetry_manager()
        tm.config.otlp_endpoint = system_config.telemetry_collector_endpoint
        tm._tenant_providers.clear()
        logger.info(
            f"TelemetryManager otlp_endpoint set to {system_config.telemetry_collector_endpoint}"
        )

    # 7c. Probe Phoenix reachability so a silent NoOpSpan fallback surfaces
    # at startup. If TELEMETRY_REQUIRED is set, missing telemetry fails
    # startup; otherwise it logs a warning so operators can decide.
    _probe_phoenix_reachability()

    # 7d. Validate each inference service actually serves the model the
    # profiles expect. Closes the silent-wrong-embedding failure mode.
    # Disabled with SKIP_INFERENCE_VALIDATION=1 (e.g., when running the
    # runtime without any inference pods deployed).
    if os.environ.get("SKIP_INFERENCE_VALIDATION") != "1":
        from cogniverse_foundation.config.utils import ConfigUtils
        from cogniverse_runtime.inference_health_check import (
            collect_profile_bindings,
            validate_inference_services,
        )

        config_path = ConfigUtils._discover_config_file()
        if config_path and config_path.exists():
            with open(config_path) as f:
                raw_config = json.load(f)
            profiles = raw_config.get("backend", {}).get("profiles", {})
            bindings = collect_profile_bindings(profiles)
            validate_inference_services(bindings, system_config.inference_service_urls)
        else:
            logger.warning(
                "Skipping inference-service validation: no config.json found"
            )

    # 8. Wire tenant manager dependencies
    tenant_manager.set_config_manager(config_manager)
    tenant_manager.set_schema_loader(schema_loader)
    tenant_manager.backend = system_backend
    logger.info("Tenant manager wired to Runtime")

    # 8b. Install per-tenant WikiManager factory.
    # Wiki pages are genuinely per-tenant — each tenant gets its own
    # wiki_pages_<tenant> Vespa schema for hard isolation. The factory
    # below deploys that schema lazily on first access per tenant; no
    # startup pre-deploy is needed.
    try:
        from cogniverse_agents.wiki.wiki_manager import WikiManager
        from cogniverse_runtime.routers import wiki as wiki_router

        # The backend handle itself is cluster-wide: one Vespa client used
        # by every tenant's WikiManager. Scope it under SYSTEM_TENANT_ID so
        # the registry key is semantically correct.
        wiki_backend = BackendRegistry.get_instance().get_ingestion_backend(
            name=bootstrap.backend_type,
            tenant_id=SYSTEM_TENANT_ID,
            config={
                "backend": {
                    "url": bootstrap.backend_url,
                    "port": bootstrap.backend_port,
                }
            },
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        # Register the "wiki" backend profile (type="wiki") so
        # WikiManager.search can resolve via the shared profile registry.
        # Schema deploy and profile registration are separate concerns in
        # VespaSearchBackend — Mem0 does the same thing in
        # memory/manager.py for "agent_memories". The profile_change_listener
        # wired above fans this into every cached search backend. The
        # registration itself is cluster-wide (all tenants see the same
        # profile shape), so it lives under SYSTEM_TENANT_ID.
        try:
            from cogniverse_foundation.config.unified_config import (
                BackendProfileConfig,
            )

            wiki_profile = {
                "type": "wiki",
                "model": "google/embeddinggemma-300m",
                "embedding_model": "google/embeddinggemma-300m",
                "embedding_dims": 768,
                "strategy": "semantic_search",
                "schema_name": "wiki_pages",
                "embedding_type": "dense",
                "schema_config": {"embedding_dims": 768},
            }
            config_manager.add_backend_profile(
                BackendProfileConfig.from_dict("wiki_pages", wiki_profile),
                tenant_id=SYSTEM_TENANT_ID,
                service="backend",
            )
            logger.info("Wiki backend profile registered")
        except Exception as exc:
            logger.debug("Wiki profile register skipped: %s", exc)

        _wiki_managers: dict = {}

        def _wiki_manager_factory(tenant_id: str) -> WikiManager:
            """Return a WikiManager for the given tenant, building it on demand.

            Each tenant gets a dedicated wiki_pages_<tenant> schema. The first
            access for a new tenant deploys the schema; subsequent accesses
            reuse the cached manager. Errors during schema deploy are non-fatal
            — the manager still constructs and writes will surface the error
            naturally on first feed attempt.
            """
            if tenant_id in _wiki_managers:
                return _wiki_managers[tenant_id]

            try:
                wiki_backend.schema_registry.deploy_schema(
                    tenant_id=tenant_id, base_schema_name="wiki_pages"
                )
            except Exception as schema_err:
                logger.warning(
                    f"Wiki schema deploy for tenant {tenant_id} skipped: {schema_err}"
                )

            # Colons are reserved in Vespa's /document/v1 URL segments
            # (id:namespace:doctype::docid format). A tenant_id like
            # "flywheel_org:production" produced schema_name
            # "wiki_pages_flywheel_org:production" and every feed call
            # returned 400 "Illegal key-value pair 'production'".
            # schema_registry.deploy_schema sanitizes internally the same
            # way, so both sides line up on the underscore form.
            mgr = WikiManager(
                backend=wiki_backend,
                tenant_id=tenant_id,
                schema_name=wiki_backend.get_tenant_schema_name(
                    tenant_id, "wiki_pages"
                ),
            )
            _wiki_managers[tenant_id] = mgr
            return mgr

        wiki_router.set_wiki_manager_factory(_wiki_manager_factory)
        logger.info("WikiManager factory initialized (per-tenant)")
    except Exception as e:
        logger.warning(f"WikiManager init failed (non-fatal): {e}")

    # 8c. Install per-tenant GraphManager factory.
    # Knowledge-graph is now per-tenant: each tenant gets its own
    # knowledge_graph_<tenant> Vespa schema, mirroring the wiki pattern.
    # Hard Vespa-schema isolation is the right level — a field-only
    # filter inside a shared schema is fragile (any new query path has
    # to remember the tenant_id filter, admin tooling sees every
    # tenant's data, and a noisy tenant can dominate shared storage).
    # The factory below deploys the per-tenant schema lazily on first
    # access; no startup pre-deploy.
    try:
        from cogniverse_agents.graph.graph_manager import GraphManager
        from cogniverse_runtime.routers import graph as graph_router

        # Cluster-wide backend handle (one Vespa client shared by every
        # tenant's GraphManager). Registry key lives under SYSTEM_TENANT_ID.
        graph_backend = BackendRegistry.get_instance().get_ingestion_backend(
            name=bootstrap.backend_type,
            tenant_id=SYSTEM_TENANT_ID,
            config={
                "backend": {
                    "url": bootstrap.backend_url,
                    "port": bootstrap.backend_port,
                }
            },
            config_manager=config_manager,
            schema_loader=schema_loader,
        )

        _graph_managers: dict = {}

        def _graph_manager_factory(tenant_id: str) -> GraphManager:
            """Return a GraphManager for the given tenant, building on demand.

            Each tenant gets a dedicated knowledge_graph_<tenant> schema.
            The first access for a new tenant deploys the schema; subsequent
            accesses reuse the cached manager. Errors during schema deploy
            are non-fatal — the manager still constructs and the first
            feed/query attempt surfaces the real error.
            """
            if tenant_id in _graph_managers:
                return _graph_managers[tenant_id]

            try:
                graph_backend.schema_registry.deploy_schema(
                    tenant_id=tenant_id, base_schema_name="knowledge_graph"
                )
            except Exception as schema_err:
                logger.warning(
                    f"Knowledge graph schema deploy for tenant {tenant_id} "
                    f"skipped: {schema_err}"
                )

            mgr = GraphManager(
                backend=graph_backend,
                tenant_id=tenant_id,
                schema_name=graph_backend.get_tenant_schema_name(
                    tenant_id, "knowledge_graph"
                ),
            )
            _graph_managers[tenant_id] = mgr
            return mgr

        graph_router.set_graph_manager_factory(_graph_manager_factory)
        logger.info("GraphManager factory initialized (per-tenant)")
    except Exception as e:
        logger.warning(f"GraphManager init failed (non-fatal): {e}")

    # 9. Configure DSPy LM and synthetic data service
    import dspy

    from cogniverse_foundation.config.llm_factory import create_dspy_lm
    from cogniverse_foundation.config.unified_config import (
        AgentMappingRule,
        DSPyModuleConfig,
        OptimizerGenerationConfig,
        SyntheticGeneratorConfig,
    )
    from cogniverse_synthetic.api import configure_service as configure_synthetic

    llm_config = config.get_llm_config()
    primary_lm = create_dspy_lm(llm_config.primary)
    # LenientJSONAdapter normalizes LM field-name variants (e.g. gemma4
    # emits `reason` instead of `reasoning`) before DSPy's strict output
    # validation. Without this, ChainOfThought calls fail with
    # AdapterParseError on small local models.
    from cogniverse_foundation.dspy import LenientJSONAdapter

    dspy.configure(lm=primary_lm, adapter=LenientJSONAdapter())
    logger.info(f"DSPy configured with LM: {llm_config.primary.model}")

    modality_config = OptimizerGenerationConfig(
        optimizer_type="modality",
        dspy_modules={
            "query_generator": DSPyModuleConfig(
                signature_class="cogniverse_synthetic.dspy_signatures.GenerateModalityQuery",
                module_type="ChainOfThought",
            ),
        },
        agent_mappings=[
            AgentMappingRule(modality="VIDEO", agent_name="video_search_agent"),
            AgentMappingRule(modality="DOCUMENT", agent_name="document_agent"),
            AgentMappingRule(modality="IMAGE", agent_name="image_search_agent"),
            AgentMappingRule(modality="AUDIO", agent_name="audio_analysis_agent"),
        ],
    )
    routing_config = OptimizerGenerationConfig(
        optimizer_type="routing",
        dspy_modules={
            "query_generator": DSPyModuleConfig(
                signature_class="cogniverse_synthetic.dspy_signatures.GenerateEntityQuery",
                module_type="ChainOfThought",
            ),
        },
    )
    synthetic_gen_config = SyntheticGeneratorConfig(
        tenant_id=SYSTEM_TENANT_ID,
        optimizer_configs={
            "modality": modality_config,
            "routing": routing_config,
        },
    )
    configure_synthetic(generator_config=synthetic_gen_config)
    logger.info("Synthetic data service configured")

    # 10. Optimization runs via Argo CronWorkflows (not as background task).
    # See: charts/cogniverse/templates/optimization-workflows.yaml
    # CLI: python -m cogniverse_runtime.optimization_cli --mode once

    # 11. Start the InMemoryQueueManager cleanup loop. Every search / ingestion /
    # mem0 operation creates a task queue holding up to max_buffer_size events
    # (~1 KB each). Without this loop, queues live forever — the suite creates
    # thousands over a run and the runtime OOMs on the accumulated buffers.
    from cogniverse_core.events import get_queue_manager

    queue_manager = get_queue_manager()
    await queue_manager.start_cleanup_loop(interval_seconds=60)
    logger.info("Event queue cleanup loop started")

    logger.info("Cogniverse Runtime started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Cogniverse Runtime...")
    await queue_manager.stop_cleanup_loop()
    logger.info("Cogniverse Runtime shut down successfully")


# Create FastAPI app
app = FastAPI(
    title="Cogniverse Runtime",
    description="Multi-agent AI platform for content intelligence",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(agents.router, prefix="/agents", tags=["agents"])
app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(ingestion.router, prefix="/ingestion", tags=["ingestion"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(tenant_manager.router, prefix="/admin", tags=["tenant-management"])
app.include_router(events.router, prefix="/events", tags=["events"])
app.include_router(synthetic_router, tags=["synthetic-data"])
app.include_router(wiki.router, prefix="/wiki", tags=["wiki"])
app.include_router(graph.router, prefix="/graph", tags=["graph"])
app.include_router(tenant.router, prefix="/admin/tenant", tags=["tenant-extensibility"])
app.include_router(debug.router, prefix="/admin/debug", tags=["debug"])


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Cogniverse Runtime",
        "version": "1.0.0",
        "description": "Multi-agent AI platform for content intelligence",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    from cogniverse_foundation.config.utils import create_default_config_manager

    # Load config to get port. SystemConfig is cluster-wide (port, host,
    # runtime-level settings), so read it under SYSTEM_TENANT_ID.
    config_manager = create_default_config_manager()
    config = get_config(tenant_id=SYSTEM_TENANT_ID, config_manager=config_manager)

    port = config.get("runtime", {}).get("port", 8000)
    host = config.get("runtime", {}).get("host", "0.0.0.0")

    uvicorn.run(
        "cogniverse_runtime.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
