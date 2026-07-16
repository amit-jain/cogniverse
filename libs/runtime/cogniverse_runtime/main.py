"""Unified FastAPI Runtime - Single entry point for all Cogniverse services.

This replaces 10+ scattered FastAPI apps with a single, unified runtime that:
- Dynamically loads backends/agents from configs/config.json
- Consolidates all endpoints under one service
- Enables clean deployment and scaling
"""

import os as _bootstrap_os

# OpenInference DSPy instrumentation must run BEFORE any module
# imports dspy.Predict / dspy.ChainOfThought etc. — those classes get
# bound to unwrapped references on import and a later instrument()
# call can't patch already-bound names. So we run instrumentation
# at the very top of main.py, gated on OPENINFERENCE_DSPY=1, before
# any other imports.
if _bootstrap_os.environ.get("OPENINFERENCE_DSPY") == "1":
    try:
        from openinference.instrumentation.dspy import (
            DSPyInstrumentor as _DSPyInstrumentor,
        )

        _DSPyInstrumentor().instrument()
        print("OpenInference DSPy instrumentation enabled at bootstrap")
    except ImportError as _exc:
        print(f"OpenInference DSPy not installed; skipping: {_exc}")
    except Exception as _exc:  # noqa: BLE001
        print(f"OpenInference DSPy instrument failed: {_exc}")

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
    knowledge,
    search,
    tenant,
    wiki,
)
from cogniverse_synthetic.api import router as synthetic_router

logger = logging.getLogger(__name__)


def _semantic_router_config_from_env():
    """Build a ``SemanticRouterConfig`` from deployment env vars, or ``None``.

    The chart sets ``SEMANTIC_ROUTER_ENABLED`` + ``SEMANTIC_ROUTER_URL`` (and
    an optional ``SEMANTIC_ROUTER_TENANT_TIERS`` JSON-object map) so a deployed
    runtime boots routing every agent's LLM call through the in-cluster
    semantic router. Returns ``None`` when routing is off (flag unset/false) or
    no URL is set — leaving the direct-to-backend path untouched. Extracted so
    it can be unit-tested without the FastAPI lifespan.

    A malformed ``SEMANTIC_ROUTER_TENANT_TIERS`` raises: a misconfigured
    deployment must fail at boot, not silently route every tenant to the
    default tier.
    """
    enabled = os.environ.get("SEMANTIC_ROUTER_ENABLED", "").lower() in (
        "1",
        "true",
        "yes",
    )
    url = os.environ.get("SEMANTIC_ROUTER_URL", "").strip()
    if not (enabled and url):
        return None

    from cogniverse_foundation.config.unified_config import SemanticRouterConfig

    raw_tiers = os.environ.get("SEMANTIC_ROUTER_TENANT_TIERS", "").strip()
    if raw_tiers:
        tenant_tiers = json.loads(raw_tiers)
        if not isinstance(tenant_tiers, dict):
            raise ValueError(
                "SEMANTIC_ROUTER_TENANT_TIERS must be a JSON object mapping "
                f"tenant_id -> tier; got {type(tenant_tiers).__name__}"
            )
    else:
        tenant_tiers = {}

    return SemanticRouterConfig(
        enabled=True,
        semantic_router_url=url,
        tenant_tiers=tenant_tiers,
    )


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

        with tm.span(
            "startup.probe",
            tenant_id=SYSTEM_TENANT_ID,
            component="search_service",
        ) as span:
            # NoOpSpan has no record_exception/set_attribute side effects;
            # if we got a real span we set an attribute to force any error
            # in the export pipeline to surface here rather than later.
            if hasattr(span, "set_attribute"):
                span.set_attribute("startup.probe", True)

        logger.info(f"Phoenix reachability probe OK (otlp={tm.config.otlp_endpoint})")
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


def reaffirm_wiki_profile(config_manager, config: dict) -> None:
    """Re-affirm config.json's ``wiki_semantic`` profile into cached backends.

    The add fans through the profile-change listener into every cached search
    backend. The profile is READ from the loaded config dict (the same source
    the search backend resolves profiles from) — a hardcoded copy here
    drifted from config.json silently. Raises when the profile is missing:
    wiki search cannot resolve without it.
    """
    from cogniverse_foundation.config.unified_config import BackendProfileConfig

    profiles = (config.get("backend") or {}).get("profiles") or {}
    raw = profiles.get("wiki_semantic")
    if raw is None:
        raise RuntimeError(
            "wiki_semantic profile missing from config — wiki search "
            "cannot resolve a profile of type 'wiki'"
        )
    config_manager.add_backend_profile(
        BackendProfileConfig.from_dict("wiki_semantic", raw),
        tenant_id=SYSTEM_TENANT_ID,
        service="backend",
    )


def _log_workflow_submission_status() -> None:
    """Log whether workflow-engine submission is enabled, at startup."""
    from cogniverse_runtime.config_loader import get_workflow_settings

    settings = get_workflow_settings()
    if settings.api_url:
        logger.info(
            "Workflow submission enabled (url=%s, namespace=%s)",
            settings.api_url,
            settings.namespace,
        )
    else:
        logger.warning(
            "WORKFLOW_API_URL not set — scheduled jobs will be persisted but "
            "never trigger. Set WORKFLOW_API_URL in deployment to enable."
        )


async def _wait_for_backend_ready(
    vespa_base: str,
    *,
    max_attempts: int = 60,
    retry_interval: float = 5.0,
    timeout: float = 5.0,
) -> bool:
    """Poll Vespa container + feed readiness without blocking the event loop.

    Vespa's two-port architecture means the container node (GET
    ``/ApplicationStatus``) converges before the content/distributor nodes
    that serve PUT/feed, so we additionally probe a document GET (404 =
    schema exists and feed works; 200 = doc exists). Returns True once feed
    is ready, False after ``max_attempts`` retries.
    """
    import httpx

    vespa_feed_probe = (
        f"{vespa_base}/document/v1/config_metadata/config_metadata/docid/probe"
    )
    async with httpx.AsyncClient() as client:
        for attempt in range(max_attempts):
            try:
                resp = await client.get(
                    f"{vespa_base}/ApplicationStatus", timeout=timeout
                )
                if resp.status_code != 200:
                    raise ConnectionError("Container node not ready")
                resp = await client.get(vespa_feed_probe, timeout=timeout)
                if resp.status_code in (200, 404):
                    return True
            except (httpx.HTTPError, OSError, ConnectionError):
                pass
            logger.info(
                f"Backend not ready, retrying ({attempt + 1}/{max_attempts})..."
            )
            await asyncio.sleep(retry_interval)
    return False


def _mirror_minio_credentials_to_aws() -> None:
    """Mirror the ``MINIO_*`` secret onto the ``AWS_*`` names fsspec's s3 client
    reads, so answer-time keyframe resolution (agents localize ``s3://``
    keyframes via ``MediaLocator``) authenticates against MinIO. Done once at the
    entrypoint so the agents stay env-agnostic and build their s3 MediaConfig
    from ``SystemConfig.minio_endpoint``. ``setdefault`` never overwrites an
    explicit ``AWS_*`` value already in the environment."""
    access = os.environ.get("MINIO_ACCESS_KEY")
    if access:
        os.environ.setdefault("AWS_ACCESS_KEY_ID", access)
    secret = os.environ.get("MINIO_SECRET_KEY")
    if secret:
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", secret)


def build_wiki_manager_factory(wiki_backend, config, config_manager):
    """Build the per-tenant ``WikiManager`` factory the runtime installs.

    Each tenant gets a dedicated ``wiki_pages_<tenant>`` schema. The first
    access for a new tenant deploys the schema (non-fatal on error — the
    first feed then surfaces the real error); subsequent accesses reuse the
    cached manager.

    The factory canonicalizes ``tenant_id`` so the schema name matches what
    ``POST /admin/tenants`` stored it under. Without this, a simple-form
    tenant_id ("acme") deploys ``wiki_pages_acme`` while the rest of the
    stack expects ``wiki_pages_acme_acme`` — writes and reads split across
    two schemas and the simple-form one becomes an orphan the canonical-form
    DELETE cannot reap. Extracted to module scope so this behaviour is
    unit-testable against a fake backend without booting the app (and so the
    test exercises the real factory rather than a drifting copy).
    """
    from cogniverse_agents.wiki.wiki_manager import WikiManager
    from cogniverse_core.common.tenant_utils import canonical_tenant_id

    managers: dict = {}

    def _wiki_manager_factory(tenant_id: str) -> "WikiManager":
        tenant_id = canonical_tenant_id(tenant_id)
        if tenant_id in managers:
            return managers[tenant_id]

        try:
            wiki_backend.schema_registry.deploy_schema(
                tenant_id=tenant_id, base_schema_name="wiki_pages"
            )
        except Exception as schema_err:
            logger.warning(
                f"Wiki schema deploy for tenant {tenant_id} skipped: {schema_err}"
            )

        mgr = WikiManager(
            backend=wiki_backend,
            tenant_id=tenant_id,
            schema_name=wiki_backend.get_tenant_schema_name(tenant_id, "wiki_pages"),
            llm_endpoint_config=config.get_llm_config().primary,
            config_manager=config_manager,
        )
        managers[tenant_id] = mgr
        return mgr

    return _wiki_manager_factory


def build_pin_lookup(knowledge_registry):
    """Pin-lookup callable for the lifecycle scheduler.

    Returns the pinned-id set for one tenant's Mem0 manager. A lookup failure
    (e.g. a backend outage) RAISES instead of returning an empty set — an
    empty set would read as "no pins" and let the scheduler prune genuinely
    pinned memories; the raise makes tick_once skip that tenant's cleanup for
    the tick (fail-safe: never prune when pins can't be confirmed).
    """

    def _pin_lookup(mm: object) -> set:
        tenant_id = getattr(mm, "tenant_id", None)
        if not tenant_id:
            return set()
        try:
            # Honor admin PinService quota overrides set via
            # PUT /admin/tenants/{t}/pin_quotas. PinQuotas.for_tenant
            # checks the admin runtime dict first, then TenantConfig
            # metadata, then defaults.
            from cogniverse_core.memory.pinning import PinQuotas, PinService

            pin_svc = PinService(
                mm,
                knowledge_registry,
                quotas=PinQuotas.for_tenant(tenant_id),
            )
            return {rec.target_memory_id for rec in pin_svc.list_pins(tenant_id)}
        except Exception as exc:
            logger.warning(
                "Pin lookup failed for tenant %s during lifecycle tick; "
                "skipping cleanup this tick: %r",
                tenant_id,
                exc,
            )
            raise

    return _pin_lookup


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifecycle manager for FastAPI app - handles startup and shutdown."""

    # Startup
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

    vespa_base = f"{bootstrap.backend_url}:{bootstrap.backend_port}"
    # Expose the resolved backend base so the health/readiness probes can ping
    # it for real connectivity instead of only checking class registration.
    app.state.backend_base_url = vespa_base
    logger.info(f"Waiting for backend feed readiness at {vespa_base}...")

    if await _wait_for_backend_ready(vespa_base):
        logger.info("Backend feed endpoint is ready")
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
    def _profile_change_listener(event: str, profile_name: str, profile_config) -> None:
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
    _log_workflow_submission_status()

    # Wire ingestion and search routers via FastAPI dependency overrides
    app.dependency_overrides[ingestion.get_config_manager_dependency] = lambda: (
        config_manager
    )
    app.dependency_overrides[ingestion.get_schema_loader_dependency] = lambda: (
        schema_loader
    )
    app.dependency_overrides[search.get_config_manager_dependency] = lambda: (
        config_manager
    )
    app.dependency_overrides[search.get_schema_loader_dependency] = lambda: (
        schema_loader
    )
    app.dependency_overrides[knowledge._get_config_manager] = lambda: config_manager
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

    # 5. Initialize SandboxManager.
    # Policy resolution order (first non-empty wins):
    #   1. COGNIVERSE_SANDBOX_POLICY env var (required|optional|disabled)
    #   2. config["sandbox"]["policy"]
    #   3. COGNIVERSE_SANDBOX_ENABLED + OPENSHELL_GATEWAY_ENDPOINT
    #      → maps to optional / disabled.
    # Default: optional (degrade with warning if gateway is missing).
    from cogniverse_runtime.sandbox_manager import SandboxManager, SandboxPolicy

    sandbox_policy: SandboxPolicy
    env_policy = os.environ.get("COGNIVERSE_SANDBOX_POLICY", "").lower().strip()
    cfg_policy = config.get("sandbox", {}).get("policy")
    if env_policy:
        sandbox_policy = SandboxPolicy(env_policy)
    elif cfg_policy:
        sandbox_policy = SandboxPolicy(str(cfg_policy).lower())
    else:
        legacy_enabled = (
            config.get("sandbox", {}).get("enabled", False)
            or os.environ.get("COGNIVERSE_SANDBOX_ENABLED", "").lower()
            in ("true", "1", "yes")
            or bool(os.environ.get("OPENSHELL_GATEWAY_ENDPOINT"))
        )
        sandbox_policy = (
            SandboxPolicy.OPTIONAL if legacy_enabled else SandboxPolicy.DISABLED
        )
    logger.info("SandboxManager booting with policy=%s", sandbox_policy.value)
    sandbox_manager = SandboxManager(policy=sandbox_policy)

    # 5a. Wire agent registry and dependencies to agents router + A2A
    agents.set_agent_registry(agent_registry)
    agents.set_agent_dependencies(config_manager, schema_loader)
    agents.set_sandbox_manager(sandbox_manager)
    logger.info("AgentRegistry and dependencies wired to agents router")

    # 6. Use config loader to dynamically load backends and agents
    config_loader = get_config_loader()
    config_loader.load_backends()
    config_loader.load_agents(agent_registry=agent_registry)

    logger.info(
        f"Loaded {len(backend_registry.list_backends())} backends, "
        f"{len(agent_registry.list_agents())} agents"
    )

    # 5b. Mount A2A protocol server (JSON-RPC 2.0). Built AFTER load_agents so
    # the card advertises the real agents (search_agent, ...) instead of the
    # 'default' fallback the empty registry produced when this ran first.
    from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill

    from cogniverse_runtime.a2a_executor import (
        BoundedInMemoryTaskStore,
        CogniverseAgentExecutor,
    )

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
        task_store=BoundedInMemoryTaskStore(
            max_tasks=int(os.environ.get("A2A_MAX_TASKS", "10000"))
        ),
    )
    a2a_server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=a2a_handler,
    )
    app.mount("/a2a", a2a_server.build())
    logger.info(f"A2A server mounted at /a2a with {len(skills)} skills")

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
    # LLM_ENGINE / LLM_MODEL come from the chart's llm.engine / llm.model
    # values (the chart passes the BARE model id). Agents that build a
    # per-tenant DSPy LM off system_config.llm_model (e.g. TextAnalysisAgent
    # via DynamicDSPyMixin) hand it straight to litellm, which rejects a bare
    # id with "LLM Provider NOT provided" — so attach the provider prefix
    # here, matching the worker and the chart's config.json model helper.
    if os.environ.get("LLM_ENGINE"):
        system_config.llm_engine = os.environ["LLM_ENGINE"]
        updated = True
    if os.environ.get("LLM_MODEL"):
        from cogniverse_foundation.dspy.model_format import ensure_provider_prefix

        system_config.llm_model = ensure_provider_prefix(os.environ["LLM_MODEL"])
        updated = True
    if os.environ.get("TELEMETRY_HTTP_ENDPOINT"):
        system_config.telemetry_url = os.environ["TELEMETRY_HTTP_ENDPOINT"]
        updated = True
    if os.environ.get("TELEMETRY_OTLP_ENDPOINT"):
        system_config.telemetry_collector_endpoint = os.environ[
            "TELEMETRY_OTLP_ENDPOINT"
        ]
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
    # Orchestrator iterative-loop knobs. Env reads belong here at the
    # startup boundary — the orchestrator itself reads them from
    # SystemConfig (no env access).
    if os.environ.get("ITER_RETRIEVAL_MAX_ITER"):
        system_config.iter_retrieval_max_iter = int(
            os.environ["ITER_RETRIEVAL_MAX_ITER"]
        )
        updated = True
    if os.environ.get("ITER_RETRIEVAL_TOKEN_BUDGET"):
        system_config.iter_retrieval_token_budget = int(
            os.environ["ITER_RETRIEVAL_TOKEN_BUDGET"]
        )
        updated = True
    if os.environ.get("ITER_RETRIEVAL_WALL_CLOCK_MS"):
        system_config.iter_retrieval_wall_clock_ms = int(
            os.environ["ITER_RETRIEVAL_WALL_CLOCK_MS"]
        )
        updated = True
    # REDIS_URL: env override for cross-pod inbound messaging. The
    # orchestrator reads this from SystemConfig (no env access in
    # the agent module).
    if os.environ.get("REDIS_URL"):
        system_config.redis_url = os.environ["REDIS_URL"]
        updated = True
    # COGNIVERSE_ADAPTER_CACHE: local download dir for the finetuning
    # adapter resolver. resolve_adapter_path() now REQUIRES a
    # non-empty cache_dir from the caller (no hardcoded fallback),
    # so production must set this if any agent loads adapters.
    if os.environ.get("COGNIVERSE_ADAPTER_CACHE"):
        system_config.adapter_cache_dir = os.environ["COGNIVERSE_ADAPTER_CACHE"]
        updated = True
    # MINIO_ENDPOINT: object-store target for the ingestion-upload
    # route. The route reads from SystemConfig (no env access).
    if os.environ.get("MINIO_ENDPOINT"):
        system_config.minio_endpoint = os.environ["MINIO_ENDPOINT"]
        updated = True
    _mirror_minio_credentials_to_aws()
    # Semantic router: the chart turns routing on by default and points the
    # runtime at the in-cluster router Service. Absent env (local/dev), routing
    # stays disabled and agents call the backend directly.
    sr_config = _semantic_router_config_from_env()
    if sr_config is not None and sr_config != system_config.semantic_router:
        system_config.semantic_router = sr_config
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
    # Disabled with SKIP_INFERENCE_VALIDATION=1; deadline overridable via
    # INFERENCE_HEALTH_BOOT_DEADLINE_SECONDS for slow-loading vLLM models.
    if os.environ.get("SKIP_INFERENCE_VALIDATION") != "1":
        from cogniverse_foundation.config.utils import ConfigUtils
        from cogniverse_runtime.inference_health_check import (
            DEFAULT_BOOT_DEADLINE_SECONDS,
            collect_profile_bindings,
            validate_inference_services,
        )

        config_path = ConfigUtils._discover_config_file()
        if config_path and config_path.exists():
            with open(config_path) as f:
                raw_config = json.load(f)
            profiles = raw_config.get("backend", {}).get("profiles", {})
            bindings = collect_profile_bindings(profiles)
            boot_deadline = float(
                os.environ.get(
                    "INFERENCE_HEALTH_BOOT_DEADLINE_SECONDS",
                    DEFAULT_BOOT_DEADLINE_SECONDS,
                )
            )
            validate_inference_services(
                bindings,
                system_config.inference_service_urls,
                boot_deadline_seconds=boot_deadline,
            )
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
            reaffirm_wiki_profile(config_manager, config)
            logger.info("Wiki backend profile registered")
        except Exception as exc:
            logger.warning("Wiki profile register failed: %s", exc)

        wiki_router.set_wiki_manager_factory(
            build_wiki_manager_factory(wiki_backend, config, config_manager)
        )
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

        def _graph_manager_factory(tenant_id: str, deploy: bool = True) -> GraphManager:
            """Return a GraphManager for the given tenant, building on demand.

            Each tenant gets a dedicated knowledge_graph_<tenant> schema.
            The first access for a new tenant deploys the schema; subsequent
            accesses reuse the cached manager. Errors during schema deploy
            are non-fatal — the manager still constructs and the first
            feed/query attempt surfaces the real error.

            ``deploy`` MUST be False on read-only paths. deploy_schema
            triggers a Vespa global app-redeploy that reconfigures the
            content cluster and can drop rows another process just fed but
            Vespa hasn't flushed — a read then loses the documents it was
            meant to return. Read-built managers are not cached so the
            first writer still deploys.

            Canonicalizes the tenant_id so the schema name matches what
            POST /admin/tenants stored it under. Without this, /graph/upsert
            with a simple-form tenant_id ("acme") deploys
            ``knowledge_graph_acme`` while the rest of the stack expects
            ``knowledge_graph_acme_acme`` — and the simple-form schema
            becomes an orphan the canonical-form DELETE cannot reap.
            """
            from cogniverse_core.common.tenant_utils import canonical_tenant_id

            tenant_id = canonical_tenant_id(tenant_id)
            if tenant_id in _graph_managers:
                return _graph_managers[tenant_id]

            if deploy:
                try:
                    graph_backend.schema_registry.deploy_schema(
                        tenant_id=tenant_id, base_schema_name="knowledge_graph"
                    )
                except Exception as schema_err:
                    logger.warning(
                        f"Knowledge graph schema deploy for tenant {tenant_id} "
                        f"skipped: {schema_err}"
                    )

            sys_cfg = config_manager.get_system_config()
            colbert_url = sys_cfg.inference_service_urls.get("colbert_pylate")
            if not colbert_url:
                raise RuntimeError(
                    "knowledge_graph requires the colbert_pylate inference "
                    "service to be deployed and present in "
                    "INFERENCE_SERVICE_URLS. Available services: "
                    f"{sorted(sys_cfg.inference_service_urls)}"
                )
            mgr = GraphManager(
                backend=graph_backend,
                tenant_id=tenant_id,
                schema_name=graph_backend.get_tenant_schema_name(
                    tenant_id, "knowledge_graph"
                ),
                colbert_endpoint_url=colbert_url,
            )
            if deploy:
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
    # NOTE: OpenInference DSPy instrumentation runs at module-top
    # bootstrap (see the top of this file) so DSPy classes are
    # wrapped BEFORE any agent imports bind references to the
    # unwrapped originals. dspy.configure here uses the already-
    # wrapped classes.

    # Re-instrument DSPy NOW that Phoenix's tracer is up so DSPy
    # LM spans actually flow to Phoenix. The bootstrap
    # instrumentation at module-top wrapped dspy.LM.__call__ with
    # the default ProxyTracerProvider (no-op). Here we create a
    # Phoenix tracer for the orchestration project and re-bind
    # the wrappers. All DSPy LM spans across tenants land in
    # this single project — acceptable for the test cluster +
    # surfaces LM input/output for byte-equal assertions.
    if os.environ.get("OPENINFERENCE_DSPY") == "1":
        try:
            from openinference.instrumentation.dspy import DSPyInstrumentor
            from phoenix.otel import register as _px_register

            otlp_endpoint = os.environ.get(
                "TELEMETRY_OTLP_ENDPOINT", "cogniverse-phoenix:4317"
            )
            if "://" not in otlp_endpoint:
                otlp_endpoint = f"http://{otlp_endpoint}"
            # Create a dedicated tracer provider for DSPy LM spans
            # and use it explicitly with the instrumentor (NOT set
            # as global so the existing telemetry-phoenix per-tenant
            # tracers remain authoritative for their domains).
            dspy_tp = _px_register(
                endpoint=otlp_endpoint,
                project_name="cogniverse-dspy-instrumentation",
                batch=True,
                protocol="grpc",
                auto_instrument=False,
                set_global_tracer_provider=False,
            )
            DSPyInstrumentor().uninstrument()
            DSPyInstrumentor().instrument(tracer_provider=dspy_tp)
            logger.info(
                "DSPy re-instrumented with Phoenix tracer "
                "(project: cogniverse-dspy-instrumentation)"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to rebind DSPy instrumentation to Phoenix: %s", exc)

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
    # Wire the real search backend so /synthetic/generate samples the tenant's
    # actual corpus. Without a backend the service falls back to a hardcoded
    # mock profile/topic list and every tenant gets fabricated training data.
    # Mirrors the optimization CLI's wiring (optimization_cli.py).
    from cogniverse_foundation.config.unified_config import BackendConfig

    synthetic_backend_section = config.get("backend", {})
    if isinstance(synthetic_backend_section, dict):
        synthetic_backend_section = {**synthetic_backend_section}
        synthetic_backend_section.setdefault("tenant_id", SYSTEM_TENANT_ID)
        synthetic_backend_config = BackendConfig.from_dict(synthetic_backend_section)
    else:
        synthetic_backend_config = synthetic_backend_section
    synthetic_backend = BackendRegistry.get_instance().get_search_backend(
        name=synthetic_backend_config.backend_type,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )
    configure_synthetic(
        backend=synthetic_backend,
        backend_config=synthetic_backend_config,
        generator_config=synthetic_gen_config,
    )
    logger.info(
        "Synthetic data service configured with %s backend",
        synthetic_backend_config.backend_type,
    )

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

    # 12. Start the OpenShell gateway health probe (only when sandbox is not
    # disabled). Each probe records availability + latency as a Phoenix span
    # (openshell.gateway_health) so the dashboard can surface gateway state.
    gateway_probe = None
    if sandbox_policy is not SandboxPolicy.DISABLED:
        from cogniverse_runtime.openshell_health import GatewayHealthProbe

        probe_interval = float(
            os.environ.get("COGNIVERSE_SANDBOX_PROBE_INTERVAL", "30")
        )
        gateway_probe = GatewayHealthProbe(
            sandbox_manager=sandbox_manager,
            interval_seconds=probe_interval,
        )
        gateway_probe.start()
        app.state.gateway_probe = gateway_probe

    # 12b. Start the OpenShell mTLS cert rotator. Watches the gateway
    # cert directory; on detected change, calls SandboxManager.reconnect()
    # so the next exec uses the rotated client. Disabled when the sandbox
    # itself is disabled, or when COGNIVERSE_SANDBOX_CERT_ROTATION_DISABLED
    # is set (operators who use a different cert-management story).
    cert_rotator = None
    if sandbox_policy is not SandboxPolicy.DISABLED and os.environ.get(
        "COGNIVERSE_SANDBOX_CERT_ROTATION_DISABLED", ""
    ).lower() not in ("1", "true", "yes"):
        from cogniverse_runtime.openshell_cert_rotator import CertRotator

        cert_rotation_interval = float(
            os.environ.get("COGNIVERSE_SANDBOX_CERT_ROTATION_INTERVAL", "300")
        )
        cert_rotator = CertRotator(
            sandbox_manager=sandbox_manager,
            interval_seconds=cert_rotation_interval,
        )
        sandbox_manager.attach_cert_rotator(cert_rotator)
        cert_rotator.start()
        app.state.cert_rotator = cert_rotator
        logger.info(
            "OpenShell cert rotator started (interval=%.0fs)",
            cert_rotation_interval,
        )

    # 13. Start the memory lifecycle scheduler. Periodically iterates
    # warm-tenant Mem0 managers and runs schema-driven cleanup on each.
    # Disabled with COGNIVERSE_MEMORY_LIFECYCLE_DISABLED=1 for tests that
    # don't want the loop running concurrently.
    lifecycle_scheduler = None
    if os.environ.get("COGNIVERSE_MEMORY_LIFECYCLE_DISABLED", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        from cogniverse_core.memory.lifecycle_scheduler import LifecycleScheduler
        from cogniverse_core.memory.manager import Mem0MemoryManager
        from cogniverse_core.memory.schema import build_default_registry

        lifecycle_interval = float(
            os.environ.get("COGNIVERSE_MEMORY_LIFECYCLE_INTERVAL", "3600")
        )
        knowledge_registry = build_default_registry()

        _pin_lookup = build_pin_lookup(knowledge_registry)

        lifecycle_scheduler = LifecycleScheduler(
            get_warm_managers=Mem0MemoryManager._instances.values,
            registry=knowledge_registry,
            interval_seconds=lifecycle_interval,
            pin_lookup=_pin_lookup,
        )
        lifecycle_scheduler.start()
        app.state.lifecycle_scheduler = lifecycle_scheduler
        logger.info(
            "Lifecycle scheduler started (interval=%.0fs)",
            lifecycle_interval,
        )

    # SIGUSR1 hot-reload handler. Operators send `kill -USR1 <pid>`
    # to trigger a non-disruptive config + sandbox-policy reload (loaded
    # backends/agents are re-read from configs/config.json; OpenShell
    # policies are re-read from configs/agent_policies/). The handler is
    # registered on the running event loop so the reload runs
    # cooperatively without blocking ongoing requests.
    import signal as _signal

    _reload_count = {"n": 0}
    _reload_tasks: set = set()

    def _do_hot_reload():
        try:
            config_loader.reload_config()
        except Exception as exc:
            logger.warning("Config hot-reload failed: %s", exc)
        if sandbox_manager is not None:
            try:
                sandbox_manager.reload_policies()
            except Exception as exc:
                logger.warning("Sandbox policy hot-reload failed: %s", exc)
        logger.info("Hot-reload complete")

    def _on_sigusr1():
        # The signal callback runs ON the loop and must return immediately —
        # the reload does blocking Vespa round-trips, so run it in a worker
        # thread. The task ref is held until done so it can't be GC'd.
        _reload_count["n"] += 1
        logger.info(
            "SIGUSR1 received — hot-reloading configuration (count=%d)",
            _reload_count["n"],
        )
        task = asyncio.get_running_loop().create_task(asyncio.to_thread(_do_hot_reload))
        _reload_tasks.add(task)
        task.add_done_callback(_reload_tasks.discard)

    try:
        asyncio.get_running_loop().add_signal_handler(_signal.SIGUSR1, _on_sigusr1)
        app.state.sigusr1_registered = True
        app.state.hot_reload_count = _reload_count
        logger.info(
            "SIGUSR1 hot-reload handler registered "
            "(send `kill -USR1 <pid>` to reload config + sandbox policies)"
        )
    except (NotImplementedError, ValueError) as exc:
        # add_signal_handler is unavailable on Windows and inside some
        # nested event-loop contexts (test runners). Fall back gracefully.
        logger.info("SIGUSR1 hot-reload not available in this loop: %s", exc)
        app.state.sigusr1_registered = False

    logger.info("Cogniverse Runtime started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Cogniverse Runtime...")
    try:
        asyncio.get_running_loop().remove_signal_handler(_signal.SIGUSR1)
    except (NotImplementedError, ValueError, RuntimeError):
        pass
    if gateway_probe is not None:
        await gateway_probe.stop()
    if cert_rotator is not None:
        await cert_rotator.stop()
    if lifecycle_scheduler is not None:
        await lifecycle_scheduler.stop()
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


def register_degraded_search_handler(app: FastAPI) -> None:
    """Map VespaSearchDegraded from any route to a 503 with the error detail.

    Search consumers (media agents, graph manager) raise it on a Vespa
    soft-timeout; without this handler those raises surface as opaque 500s.
    """
    from cogniverse_agents.search.vespa_query import VespaSearchDegraded

    @app.exception_handler(VespaSearchDegraded)
    async def _degraded_search_to_503(request, exc: VespaSearchDegraded):
        return JSONResponse(status_code=503, content={"detail": str(exc)})


register_degraded_search_handler(app)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(agents.router, prefix="/agents", tags=["agents"])
app.include_router(search.router, prefix="/search", tags=["search"])
app.include_router(ingestion.router, prefix="/ingestion", tags=["ingestion"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])
app.include_router(knowledge.router, prefix="/admin", tags=["knowledge-agents"])
app.include_router(tenant_manager.router, prefix="/admin", tags=["tenant-management"])
app.include_router(events.router, prefix="/events", tags=["events"])
app.include_router(synthetic_router, tags=["synthetic-data"])
app.include_router(wiki.router, prefix="/wiki", tags=["wiki"])
app.include_router(graph.router, prefix="/graph", tags=["graph"])
app.include_router(tenant.router, prefix="/admin/tenant", tags=["tenant-extensibility"])
app.include_router(debug.router, prefix="/admin/debug", tags=["debug"])

# Queue-driven ingestion (Step 5). When REDIS_URL is set, the existing
# /ingestion/upload endpoint streams uploaded bytes to MinIO and submits
# to the redis queue (instead of running the pipeline in-process). The
# new /ingestion/{id}/events SSE + /ingestion/{id}/status snapshot are
# mounted under the same prefix. Without REDIS_URL, /ingestion/upload
# falls back to the in-process pipeline.
if os.environ.get("REDIS_URL"):
    from cogniverse_runtime.ingestion_worker import status_api as ingest_v2_status

    app.include_router(ingest_v2_status.router, prefix="/ingestion", tags=["ingestion"])


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
