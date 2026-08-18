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
from enum import StrEnum
from typing import AsyncIterator, Callable, Mapping

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from cogniverse_core.common.cache.backends.s3 import configure_s3_backend_defaults
from cogniverse_core.common.models.semantic_embedder import (
    configure_semantic_embedder_defaults,
)
from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_foundation.config.utils import get_config
from cogniverse_foundation.telemetry.manager import get_telemetry_manager

# Import routers
from cogniverse_runtime.admin import tenant_manager
from cogniverse_runtime.config_loader import get_config_loader
from cogniverse_runtime.entrypoint_env import (
    resolve_library_env_defaults as _resolve_library_env_defaults_from_entrypoint,
)
from cogniverse_runtime.inference_services import parse_inference_service_urls
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
from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config
from cogniverse_synthetic.api import router as synthetic_router

logger = logging.getLogger(__name__)

_RUNTIME_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def _configure_runtime_logging() -> None:
    """Attach a stream handler to the root logger under ``LOG_LEVEL``.

    Uvicorn configures only its own loggers; a pre-existing root handler
    (pytest, an embedding host) is left in charge.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = logging.getLevelNamesMapping().get(level_name)
    if level is None:
        raise ValueError(f"LOG_LEVEL={level_name!r} is not a logging level name")
    logging.basicConfig(level=level, format=_RUNTIME_LOG_FORMAT)


_configure_runtime_logging()

# Bound on cached per-tenant GraphManagers. Least-recently-used tenants
# rebuild on their next graph access; tenant delete evicts eagerly via the
# registered-cache hook.
GRAPH_MANAGER_CACHE_CAPACITY = 64


def _build_graph_manager_factory(graph_backend, config_manager):
    """Build the per-tenant GraphManager factory used by the graph router.

    Extracted from the lifespan so the caching/eviction behavior is
    unit-testable without booting the runtime.
    """
    from cogniverse_agents.graph.graph_manager import GraphManager
    from cogniverse_foundation.caching import TenantLRUCache, register_tenant_cache

    _graph_managers: TenantLRUCache[GraphManager] = register_tenant_cache(
        TenantLRUCache(capacity=GRAPH_MANAGER_CACHE_CAPACITY)
    )

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

        def _build() -> GraphManager:
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
            gliner_url = sys_cfg.inference_service_urls.get("gliner")
            if not colbert_url:
                raise RuntimeError(
                    "knowledge_graph requires the colbert_pylate inference "
                    "service to be deployed and present in "
                    "INFERENCE_SERVICE_URLS. Available services: "
                    f"{sorted(sys_cfg.inference_service_urls)}"
                )
            if not gliner_url:
                raise RuntimeError(
                    "knowledge_graph requires gliner in INFERENCE_SERVICE_URLS. "
                    f"Available: {sorted(sys_cfg.inference_service_urls)}"
                )
            return GraphManager(
                backend=graph_backend,
                tenant_id=tenant_id,
                schema_name=graph_backend.get_tenant_schema_name(
                    tenant_id, "knowledge_graph"
                ),
                colbert_endpoint_url=colbert_url,
                gliner_inference_url=gliner_url,
            )

        if deploy:
            # get_or_set holds the cache lock across the build, so N
            # concurrent first-touches for a fresh tenant run deploy_schema
            # + the manager construction exactly once.
            return _graph_managers.get_or_set(tenant_id, _build)
        cached = _graph_managers.get(tenant_id)
        if cached is not None:
            return cached
        return _build()

    return _graph_manager_factory


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
        import socket
        from urllib.parse import urlparse

        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        tm = get_telemetry_manager()
        if not tm.config.enabled:
            logger.info(
                "Telemetry disabled in config — skipping Phoenix reachability probe"
            )
            return

        # Emitting a span proves nothing: TelemetryManager.span() swallows
        # tracer/export errors and yields a NoOpSpan, so the block never raises
        # even with Phoenix down — the probe always logged "OK". Actually check
        # the OTLP collector is listening with a TCP connect.
        endpoint = tm.config.otlp_endpoint
        parsed = urlparse(endpoint if "://" in endpoint else f"//{endpoint}")
        host = parsed.hostname or "localhost"
        port = parsed.port or 4317
        with socket.create_connection((host, port), timeout=5.0):
            pass

        logger.info(f"Phoenix reachability probe OK (otlp={endpoint})")
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


class BackendStartupState(StrEnum):
    FEED_READY = "feed_ready"
    FRESH_INSTALL = "fresh_install"
    UNAVAILABLE = "unavailable"


async def _wait_for_backend_startup(
    vespa_base: str,
    config_server_base: str,
    *,
    max_attempts: int = 60,
    retry_interval: float = 5.0,
    timeout: float = 5.0,
) -> BackendStartupState:
    """Distinguish a ready data plane from a fresh Vespa installation.

    A fresh config server cannot expose ``/ApplicationStatus`` or document
    endpoints until its first application package is deployed. Polling only
    those endpoints therefore creates a startup cycle. The config-server
    application resource returns 404 only for that fresh state; 200 means an
    application exists and its data plane still needs to converge.
    """
    import httpx

    vespa_feed_probe = (
        f"{vespa_base}/document/v1/config_metadata/config_metadata/docid/probe"
    )
    application_resource = (
        f"{config_server_base}/application/v2/tenant/default/application/default"
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
                    return BackendStartupState.FEED_READY
            except (httpx.HTTPError, OSError, ConnectionError):
                pass
            try:
                resp = await client.get(application_resource, timeout=timeout)
                if resp.status_code == 404:
                    return BackendStartupState.FRESH_INSTALL
            except (httpx.HTTPError, OSError):
                pass
            logger.info(
                f"Backend not ready, retrying ({attempt + 1}/{max_attempts})..."
            )
            if attempt < max_attempts - 1:
                await asyncio.sleep(retry_interval)
    return BackendStartupState.UNAVAILABLE


def _wait_for_config_server(
    host: str, port: int, *, max_attempts: int = 60, interval: float = 5.0
) -> bool:
    """Poll until the backend's config/deploy server accepts TCP connections.

    A cold Vespa opens its query port (8080) before its config/deploy server
    (19071), so a metadata deploy fired the instant the query port answers
    hits ``Connection refused`` on 19071. Waiting here keeps the retry
    IN-PROCESS — otherwise the deploy raises, the whole app startup exits,
    and the only thing retrying is the kubelet restarting the crashed pod
    (5+ crash-loops with full tracebacks before the config server is up).
    """
    import socket
    import time

    for _ in range(max_attempts):
        try:
            with socket.create_connection((host, port), timeout=3):
                return True
        except OSError:
            time.sleep(interval)
    return False


def _application_exists(
    host: str, port: int, *, max_attempts: int = 6, interval: float = 5.0
) -> bool:
    """Ask the config server whether an application package is deployed.

    Discriminates a genuinely FRESH backend (404 → safe to bootstrap) from a
    populated one whose config read merely failed (200 → a registry-less
    metadata-only deploy would drop every tenant schema and lose their
    documents). An answer that is neither leaves fresh-vs-populated unknown —
    raise rather than deploy blind.
    """
    import time

    import httpx

    url = f"http://{host}:{port}/application/v2/tenant/default/application/default"
    last: object = None
    for attempt in range(max_attempts):
        try:
            resp = httpx.get(url, timeout=10)
        except httpx.HTTPError as exc:
            last = repr(exc)
        else:
            if resp.status_code == 200:
                return True
            if resp.status_code == 404:
                return False
            last = f"HTTP {resp.status_code}"
        if attempt < max_attempts - 1:
            time.sleep(interval)
    raise RuntimeError(
        f"Cannot determine whether {host}:{port} has an application deployed "
        f"(last answer: {last}) — refusing to bootstrap metadata schemas blind"
    )


def _bootstrap_metadata_schemas(bootstrap, application_name: str) -> None:
    """Deploy the metadata schemas to a backend with no application package.

    A fresh backend serves nothing on the query chain until the first
    application deploys, so every config read fails — the config_metadata
    schema is itself part of the metadata application. Runs BEFORE the
    first config read on first install only. Waits for the config/deploy
    server to accept connections first (a cold backend brings it up after
    the query port), then raises if the deploy itself fails (genuine
    outage / misconfig → fail fast).

    A failed config read is NOT proof of a fresh backend — a populated
    cluster mid cold-start or answering degraded fails the same way, and
    deploying the registry-less metadata-only package over it would remove
    every tenant content schema. Two guards make that impossible: the
    config server must report NO deployed application before anything is
    deployed, and the deploy runs with schema removal disabled so Vespa
    itself refuses a package that would drop schemas.

    Constructs the schema manager DIRECTLY: every registry/backend path
    reads the config store internally, which is exactly what cannot work
    yet on a fresh backend.
    """
    from urllib.parse import urlparse

    from cogniverse_vespa.config_utils import calculate_config_port
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

    config_port = calculate_config_port(bootstrap.backend_port)
    host = urlparse(bootstrap.backend_url).hostname or bootstrap.backend_url
    if not _wait_for_config_server(host, config_port):
        raise RuntimeError(
            f"Backend config server {host}:{config_port} never accepted "
            "connections — cannot bootstrap metadata schemas"
        )

    if _application_exists(host, config_port):
        raise RuntimeError(
            f"Backend {host}:{config_port} already has an application deployed "
            "— the failed config read is a real outage, not a fresh install; "
            "a metadata-only deploy here would drop the existing tenant schemas"
        )

    manager = VespaSchemaManager(
        backend_endpoint=bootstrap.backend_url,
        backend_port=config_port,
        schema_registry=None,
    )
    manager.upload_metadata_schemas(
        app_name=application_name, allow_schema_removal=False
    )
    logger.info("Metadata schemas bootstrapped for fresh backend")


def _resolve_library_env_defaults() -> dict[str, str | int | None]:
    """Read the library-module defaults from the shared runtime resolver."""
    return _resolve_library_env_defaults_from_entrypoint()


def _mirror_minio_credentials_to_aws(
    access_key: str | None, secret_key: str | None
) -> None:
    """Mirror the MinIO secret onto the AWS names fsspec's S3 client reads."""
    if access_key:
        os.environ.setdefault("AWS_ACCESS_KEY_ID", access_key)
    if secret_key:
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", secret_key)


def _configure_library_module_defaults(
    config_manager,
    *,
    minio_endpoint: str | None,
    minio_access_key: str | None,
    minio_secret_key: str | None,
    telemetry_otlp_endpoint: str | None,
    telemetry_http_endpoint: str | None = None,
    semantic_embed_url: str | None,
    semantic_embed_model: str | None,
    tenant_cache_capacity: int,
) -> None:
    """Inject the runtime defaults into the library modules that use them."""
    from cogniverse_agents.text_analysis_agent import (
        configure_tenant_cache_capacity as configure_text_analysis_agent_tenant_cache_capacity,
    )
    from cogniverse_core.memory.manager import (
        configure_tenant_cache_capacity as configure_memory_manager_tenant_cache_capacity,
    )
    from cogniverse_core.registries.backend_registry import (
        configure_tenant_cache_capacity as configure_backend_registry_tenant_cache_capacity,
    )
    from cogniverse_foundation.registry.entry_point_registry import (
        configure_tenant_cache_capacity as configure_entry_point_registry_tenant_cache_capacity,
    )

    _mirror_minio_credentials_to_aws(minio_access_key, minio_secret_key)
    configure_s3_backend_defaults(
        endpoint=minio_endpoint,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
    )
    configure_semantic_embedder_defaults(
        remote_url=semantic_embed_url,
        model_name=semantic_embed_model,
    )
    configure_text_analysis_agent_tenant_cache_capacity(tenant_cache_capacity)
    configure_memory_manager_tenant_cache_capacity(tenant_cache_capacity)
    configure_backend_registry_tenant_cache_capacity(tenant_cache_capacity)
    configure_entry_point_registry_tenant_cache_capacity(tenant_cache_capacity)
    get_telemetry_manager(config_manager, otlp_endpoint=telemetry_otlp_endpoint)


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


_PIN_QUOTA_LOAD_TIMEOUT_S = 10.0


def build_pin_lookup(
    knowledge_registry,
    quota_loader: Callable[[str], Mapping[str, int]],
):
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
            from cogniverse_core.memory.pinning import PinQuotas, PinService

            admin_overrides = quota_loader(tenant_id)
            pin_svc = PinService(
                mm,
                knowledge_registry,
                quotas=PinQuotas.for_tenant(tenant_id, admin_overrides=admin_overrides),
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


def _dispatcher_entity_extractor(dispatcher):
    """Label source text through the runtime's production agent dispatcher."""

    async def extract_entities(source_text: str, tenant_id: str):
        try:
            return await dispatcher.dispatch(
                agent_name="entity_extraction_agent",
                query=source_text,
                context={"tenant_id": tenant_id},
            )
        except Exception as exc:
            raise RuntimeError(
                "Entity extraction dispatch failed for "
                f"tenant={tenant_id!r} source_text={source_text!r}: {exc}"
            ) from exc

    return extract_entities


def _dispatcher_routing_decider(dispatcher):
    """Label generated queries through the runtime's production gateway."""

    async def route_query(query: str, tenant_id: str):
        from cogniverse_agents.gateway_agent import GatewayInput

        try:
            gateway_agent = await dispatcher._get_or_build_gateway_agent(tenant_id)
            return await gateway_agent.process(
                GatewayInput(query=query, tenant_id=tenant_id)
            )
        except Exception as exc:
            raise RuntimeError(
                "Gateway routing decision failed for "
                f"tenant={tenant_id!r} query={query!r}: {exc}"
            ) from exc

    return route_query


def _dispatcher_query_enhancer(dispatcher):
    """Label generated queries through the production enhancement agent."""

    async def enhance_query(query: str, tenant_id: str, source_text: str):
        try:
            return await dispatcher.dispatch(
                agent_name="query_enhancement_agent",
                query=query,
                context={"tenant_id": tenant_id, "source_text": source_text},
            )
        except Exception as exc:
            raise RuntimeError(
                "Query enhancement dispatch failed for "
                f"tenant={tenant_id!r} query={query!r}: {exc}"
            ) from exc

    return enhance_query


def _dispatcher_profile_labeler(dispatcher):
    """Label source-grounded queries through the production profile selector."""

    async def label_profile(query: str, available_profiles: list[str], tenant_id: str):
        try:
            return await dispatcher.dispatch(
                agent_name="profile_selection_agent",
                query=query,
                context={
                    "tenant_id": tenant_id,
                    "profiles": available_profiles,
                },
            )
        except Exception as exc:
            raise RuntimeError(
                "Profile selection dispatch failed for "
                f"tenant={tenant_id!r} query={query!r}: {exc}"
            ) from exc

    return label_profile


# dspy.configure grants ambient-binding ownership to the first async task
# that calls it; the ambient LM is process-wide, so it is bound exactly once
# per process (tests boot several lifespans in one process). Per-tenant and
# per-request paths override via dspy.context(lm=...).
_DSPY_AMBIENT_CONFIGURED = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifecycle manager for FastAPI app - handles startup and shutdown."""

    inference_service_urls = parse_inference_service_urls(
        os.environ.get("INFERENCE_SERVICE_URLS")
    )

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

    # 1. Resolve fresh-install versus existing data-plane readiness before
    # loading config. In k8s, the runtime pod may start before Vespa is ready.
    from urllib.parse import urlparse

    from cogniverse_foundation.config.bootstrap import BootstrapConfig
    from cogniverse_vespa.config_utils import calculate_config_port

    bootstrap = BootstrapConfig.from_environment()

    vespa_base = f"{bootstrap.backend_url}:{bootstrap.backend_port}"
    parsed_backend_url = urlparse(bootstrap.backend_url)
    config_server_host = parsed_backend_url.hostname or bootstrap.backend_url
    config_server_base = (
        f"http://{config_server_host}:{calculate_config_port(bootstrap.backend_port)}"
    )
    # Expose the resolved backend base so the health/readiness probes can ping
    # it for real connectivity instead of only checking class registration.
    app.state.backend_base_url = vespa_base
    logger.info(f"Waiting for backend startup readiness at {vespa_base}...")

    backend_state = await _wait_for_backend_startup(vespa_base, config_server_base)
    if backend_state is BackendStartupState.FRESH_INSTALL:
        from cogniverse_foundation.config.unified_config import SystemConfig

        logger.info("Fresh backend detected; deploying metadata schemas")
        await asyncio.to_thread(
            _bootstrap_metadata_schemas,
            bootstrap,
            SystemConfig().application_name,
        )
        backend_state = await _wait_for_backend_startup(vespa_base, config_server_base)

    if backend_state is not BackendStartupState.FEED_READY:
        raise RuntimeError(
            f"Backend data and config planes did not become ready at {vespa_base}"
        )
    logger.info("Backend feed endpoint is ready")

    # 2. Load configuration
    from pathlib import Path

    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.utils import create_default_config_manager

    config_manager = create_default_config_manager()

    # 2a. Re-probe the store after feed readiness. A read failure here is not
    # proof of a fresh install: the startup state already handled that case.
    # The guarded bootstrap refuses to touch a deployed application, while a
    # rare fresh-install race may still safely complete metadata deployment.
    try:
        await asyncio.to_thread(config_manager.get_system_config)
    except Exception as probe_exc:
        logger.warning(
            "Config store not queryable (%s) — deploying metadata schemas "
            "in case this is a fresh backend with no application package",
            probe_exc,
        )
        from cogniverse_foundation.config.unified_config import SystemConfig

        await asyncio.to_thread(
            _bootstrap_metadata_schemas,
            bootstrap,
            SystemConfig().application_name,
        )
        for attempt in range(12):
            try:
                await asyncio.to_thread(config_manager.get_system_config)
                logger.info("Config store queryable after metadata bootstrap")
                break
            except Exception:
                if attempt == 11:
                    raise
                await asyncio.sleep(10)

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
    library_env = _resolve_library_env_defaults()
    _configure_library_module_defaults(config_manager, **library_env)

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

    # Deploy metadata schemas once at startup (not in every VespaBackend.__init__).
    # allow_schema_removal=True is safe ONLY here: this manager is
    # registry-aware and schema enumeration raises on any schema it cannot
    # include, so the deploy can garbage-collect deleted-tenant schemas
    # without ever silently dropping a live one.
    system_config = config_manager.get_system_config()
    system_backend.schema_manager.upload_metadata_schemas(
        app_name=system_config.application_name, allow_schema_removal=True
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
    if library_env["telemetry_otlp_endpoint"]:
        system_config.telemetry_collector_endpoint = library_env[
            "telemetry_otlp_endpoint"
        ]
        updated = True
    if os.environ.get("RUNTIME_URL"):
        system_config.agent_registry_url = os.environ["RUNTIME_URL"]
        updated = True
    # Inference URLs explicitly supplied by the deployment replace persisted
    # service discovery. An absent variable leaves persisted discovery intact.
    # COLPALI keeps its dedicated variable.
    new_colpali = os.environ.get("COLPALI_INFERENCE_URL", "")
    if new_colpali != system_config.colpali_inference_url:
        system_config.colpali_inference_url = new_colpali
        updated = True
    if (
        inference_service_urls is not None
        and inference_service_urls != system_config.inference_service_urls
    ):
        system_config.inference_service_urls = inference_service_urls
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
    # MINIO_ENDPOINT: object-store target for the ingestion-upload
    # route. The route reads from SystemConfig (no env access).
    if library_env["minio_endpoint"]:
        system_config.minio_endpoint = library_env["minio_endpoint"]
        updated = True
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

    # Wire Phoenix endpoints after config resolution so the admin router
    # follows the same single source of truth as the rest of telemetry config.
    admin.set_phoenix_endpoints(
        system_config.telemetry_url,
        system_config.telemetry_collector_endpoint,
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

        graph_router.set_graph_manager_factory(
            _build_graph_manager_factory(graph_backend, config_manager)
        )
        logger.info("GraphManager factory initialized (per-tenant)")
    except Exception as e:
        logger.warning(f"GraphManager init failed (non-fatal): {e}")

    # 9. Configure DSPy LM and synthetic data service
    import dspy

    from cogniverse_foundation.config.llm_factory import create_dspy_lm
    from cogniverse_synthetic.api import configure_service as configure_synthetic

    llm_config = config.get_llm_config()
    primary_lm = create_dspy_lm(llm_config.primary)
    # LenientJSONAdapter normalizes LM field-name variants (e.g. gemma4
    # emits `reason` instead of `reasoning`) before DSPy's strict output
    # validation. Without this, ChainOfThought calls fail with
    # AdapterParseError on small local models.
    from cogniverse_foundation.dspy import LenientJSONAdapter

    global _DSPY_AMBIENT_CONFIGURED
    if not _DSPY_AMBIENT_CONFIGURED:
        try:
            dspy.configure(lm=primary_lm, adapter=LenientJSONAdapter())
            _DSPY_AMBIENT_CONFIGURED = True
            logger.info(f"DSPy configured with LM: {llm_config.primary.model}")
        except RuntimeError as exc:
            # Another component in this process (a worker job, an earlier
            # test) already owns the ambient binding's async-task slot.
            # First writer wins by design; boot continues on dspy.context
            # overrides.
            logger.warning(f"DSPy ambient configure skipped: {exc}")
    else:
        logger.info("DSPy ambient LM already configured for this process")
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

            otlp_endpoint = library_env["telemetry_otlp_endpoint"] or (
                "cogniverse-phoenix:4317"
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

    synthetic_runtime_config = parse_synthetic_runtime_config(
        config,
        tenant_id=SYSTEM_TENANT_ID,
        loaded_agent_names=set(agent_registry.list_agents()),
    )
    # Wire the search backend and complete configured profile map required by
    # SyntheticDataService so /synthetic/generate samples tenant schemas.
    # Mirrors the optimization CLI's wiring (optimization_cli.py).
    try:
        synthetic_backend = BackendRegistry.get_instance().get_search_backend(
            name=synthetic_runtime_config.backend_config.backend_type,
            config_manager=config_manager,
            schema_loader=schema_loader,
        )
    except Exception as exc:
        raise RuntimeError(
            "Synthetic backend access failed for "
            f"tenant={SYSTEM_TENANT_ID!r} "
            f"backend={synthetic_runtime_config.backend_config.backend_type!r}: {exc}"
        ) from exc
    configure_synthetic(
        backend=synthetic_backend,
        backend_config=synthetic_runtime_config.backend_config,
        generator_config=synthetic_runtime_config.generator_config,
        agents_config=synthetic_runtime_config.agents_config,
        entity_extractor=_dispatcher_entity_extractor(dispatcher),
        routing_decider=_dispatcher_routing_decider(dispatcher),
        query_enhancer=_dispatcher_query_enhancer(dispatcher),
        profile_labeler=_dispatcher_profile_labeler(dispatcher),
    )
    logger.info(
        "Synthetic data service configured with %s backend",
        synthetic_runtime_config.backend_config.backend_type,
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
        from cogniverse_runtime.routers.admin import _load_pin_quotas

        lifecycle_interval = float(
            os.environ.get("COGNIVERSE_MEMORY_LIFECYCLE_INTERVAL", "3600")
        )
        knowledge_registry = build_default_registry()

        loop = asyncio.get_running_loop()

        def _quota_loader(tenant_id: str) -> dict[str, int]:
            return asyncio.run_coroutine_threadsafe(
                _load_pin_quotas(tenant_id), loop
            ).result(timeout=_PIN_QUOTA_LOAD_TIMEOUT_S)

        _pin_lookup = build_pin_lookup(knowledge_registry, _quota_loader)

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
    if _reload_tasks:
        # A SIGUSR1 reload still running would otherwise be torn down at
        # loop close mid-re-read; it is quick and idempotent, so drain it.
        await asyncio.gather(*_reload_tasks, return_exceptions=True)
    if gateway_probe is not None:
        await gateway_probe.stop()
    if cert_rotator is not None:
        await cert_rotator.stop()
    if lifecycle_scheduler is not None:
        await lifecycle_scheduler.stop()
    await queue_manager.stop_cleanup_loop()
    # Tear down pooled OpenShell sessions so a restart doesn't orphan one live
    # gateway container per agent_type. close() does gateway RPCs — off the loop.
    try:
        await asyncio.to_thread(sandbox_manager.close)
    except Exception as exc:
        logger.warning("SandboxManager close failed during shutdown: %s", exc)
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

# Queue-driven ingestion. When REDIS_URL is set, /ingestion/upload
# streams uploaded bytes to MinIO and submits to the redis queue, and
# the /ingestion/{id}/events SSE + /ingestion/{id}/status snapshot
# routes are mounted under the same prefix. Without REDIS_URL (and a
# MinIO endpoint) /ingestion/upload responds 503 — it has no
# in-process fallback — and the status routes are simply absent.
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
