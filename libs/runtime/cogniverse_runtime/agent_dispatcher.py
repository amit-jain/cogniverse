"""Agent dispatch logic — routes tasks to in-process agent implementations.

Extracted from routers/agents.py so both the REST endpoint and the A2A
executor can share the same dispatch logic without duplication.

Multi-turn support: The context dict may carry 'context_id' and
'conversation_history' (list of {role, content} dicts) which are
forwarded to routing/orchestration and search paths.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from cogniverse_core.common.tenant_utils import (
    canonical_tenant_id,
    require_tenant_id,
)
from cogniverse_core.registries.agent_registry import AgentRegistry

if TYPE_CHECKING:
    from cogniverse_runtime.sandbox_manager import SandboxManager
from cogniverse_foundation.caching import TenantLRUCache, register_tenant_cache
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader

if TYPE_CHECKING:
    from collections.abc import Callable

    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

logger = logging.getLogger(__name__)

# How long a cached GatewayAgent serves its loaded thresholds before a cache
# hit re-reads the artifact. Bounds post-recalibration staleness on a warm pod
# (the optimization crons run on 15-minute cadences) without putting the
# artifact read on the per-request path.
GATEWAY_ARTIFACT_TTL_S = 300.0

# Bound on cached per-tenant GatewayAgents. Least-recently-dispatched tenants
# rebuild on their next request; tenant delete evicts eagerly via the
# registered-cache hook.
GATEWAY_AGENT_CACHE_CAPACITY = 64

# How long a cached generic A2A agent (entity_extraction, query_enhancement,
# profile_selection, …) serves its loaded artifact before a cache hit re-reads
# it. Same rationale + cadence as the gateway: keep the artifact read off the
# per-request path while bounding post-recalibration staleness on a warm pod.
GENERIC_AGENT_TTL_S = 300.0

# Bound on cached tenants for the generic-agent cache. The per-tenant value is a
# dict of {agent_name -> _GenericAgentEntry}, so this caps distinct tenants, not
# distinct agents; tenant delete evicts a tenant's whole dict via the
# registered-cache hook.
GENERIC_AGENT_CACHE_CAPACITY = 64

# Orchestrator caching mirrors the gateway: build the per-tenant
# OrchestratorAgent (+ its WorkflowIntelligence corpus and policy http client)
# once and TTL-reload its artifact, instead of rebuilding the agent and
# re-reading the workflow corpus (4+ Phoenix reads) on every complex query.
ORCHESTRATOR_ARTIFACT_TTL_S = 300.0
ORCHESTRATOR_AGENT_CACHE_CAPACITY = 64

# When a TTL reload of a cached agent's artifact fails (transient artifact-store
# blip on exactly the request that crosses the TTL boundary), keep serving the
# still-valid cached agent and reschedule the reload this many seconds out —
# short enough to recover quickly, long enough not to hammer a down store on
# every subsequent request.
RELOAD_RETRY_COOLDOWN_S = 10.0

# Resolved (agent, deps, input) classes for generic A2A dispatch, keyed by the
# ``module:Class`` path. Bounded by the fixed AGENT_CLASSES set — each dispatch
# would otherwise re-scan dir(module) twice for the convention-named Deps/Input
# classes. Keyed on the class path (not the bare agent_name) so a re-registered
# mapping resolves to the new module instead of a stale entry.
_GENERIC_AGENT_CLASSES: Dict[str, tuple] = {}


def _scan_module_for_generic_classes(module: Any, class_name: str) -> tuple:
    """Resolve the (agent, deps, input) classes from ``module`` by convention:
    the agent class is ``class_name``; the Deps/Input classes are the module
    attributes whose names end with ``Deps``/``Input`` (excluding the shared
    ``AgentDeps``/``AgentInput`` bases). Either may be None if absent."""
    agent_cls = getattr(module, class_name)

    deps_cls = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and attr_name.endswith("Deps")
            and attr_name != "AgentDeps"
        ):
            deps_cls = attr
            break

    input_cls = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and attr_name.endswith("Input")
            and attr_name != "AgentInput"
        ):
            input_cls = attr
            break

    return agent_cls, deps_cls, input_cls


def _resolve_generic_agent_classes(
    class_path: str, module: Any, class_name: str
) -> tuple:
    """Return the memoized (agent, deps, input) classes for ``class_path``,
    scanning the module once on the first request for that path."""
    cached = _GENERIC_AGENT_CLASSES.get(class_path)
    if cached is not None:
        return cached
    resolved = _scan_module_for_generic_classes(module, class_name)
    _GENERIC_AGENT_CLASSES[class_path] = resolved
    return resolved


@dataclasses.dataclass
class _GatewayAgentEntry:
    """A cached per-tenant GatewayAgent with its artifact-load stamp.

    One cache slot keeps the agent and its reload stamp moving together —
    an agent can never outlive or lose its stamp.
    """

    agent: Any
    loaded_at: float


def _new_gateway_agent_cache() -> TenantLRUCache["_GatewayAgentEntry"]:
    return register_tenant_cache(TenantLRUCache(capacity=GATEWAY_AGENT_CACHE_CAPACITY))


@dataclasses.dataclass
class _GenericAgentEntry:
    """A cached generic A2A agent with its artifact-load stamp.

    ``loaded_at`` is mutated in place on a TTL reload (stamp-before-await), so
    the entry must stay mutable — the same agent instance keeps serving while
    only its artifact is refreshed.
    """

    agent: Any
    loaded_at: float


def _new_generic_agent_cache() -> "TenantLRUCache[Dict[str, _GenericAgentEntry]]":
    return register_tenant_cache(TenantLRUCache(capacity=GENERIC_AGENT_CACHE_CAPACITY))


@dataclasses.dataclass
class _OrchestratorAgentEntry:
    """A cached per-tenant OrchestratorAgent with its artifact-load stamp."""

    agent: Any
    loaded_at: float


def _new_orchestrator_agent_cache(
    on_evict: "Callable[[str, _OrchestratorAgentEntry], None] | None" = None,
) -> "TenantLRUCache[_OrchestratorAgentEntry]":
    return register_tenant_cache(
        TenantLRUCache(capacity=ORCHESTRATOR_AGENT_CACHE_CAPACITY, on_evict=on_evict)
    )


def _retrieve_future_exc(fut: "asyncio.Future[Any]") -> None:
    """Consume a resolved build future's exception so a failed cold build with
    no concurrent waiter doesn't log ``Future exception was never retrieved``.
    Real waiters still receive it via ``await``."""
    if not fut.cancelled():
        fut.exception()


def _flatten_search_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    """Lift a ``SearchResult``/gateway-shaped hit's ``metadata`` to the top level
    so the answer agents' text helpers see the retrieved content.

    ``_format_public_result`` nests ``video_id``, ``video_title``,
    ``audio_transcript`` etc. under ``metadata``, but the report/summary content
    builders read ``title`` / ``description`` / ``video_id`` at the top level and
    would otherwise render every source as "Unknown" with no transcript. Keeps
    the nested ``metadata`` untouched (keyframe resolution reads it), aliases the
    search field names to what the agents read, and never lets a top-level
    identity/score field be shadowed by metadata.
    """
    metadata = hit.get("metadata")
    if not isinstance(metadata, dict):
        return hit
    flat: Dict[str, Any] = {**metadata, **hit}
    if not flat.get("title"):
        # Each content type names its title differently; document/image/audio
        # hits carried document_title/full_text etc. that were never lifted, so
        # document-profile summaries rendered every source as "Unknown".
        for key in ("video_title", "document_title", "audio_title", "image_title"):
            if metadata.get(key):
                flat["title"] = metadata[key]
                break
    text = (
        metadata.get("segment_description")
        or metadata.get("audio_transcript")
        or metadata.get("full_text")
    )
    if text:
        flat.setdefault("description", text)
        flat.setdefault("text_content", text)
    return flat


class AgentDispatcher:
    """Routes agent tasks to the correct in-process agent implementation.

    Each dispatch call instantiates a fresh agent (stateless, per-request)
    and executes the task against it.
    """

    def __init__(
        self,
        agent_registry: AgentRegistry,
        config_manager: ConfigManager,
        schema_loader: SchemaLoader,
        sandbox_manager: "SandboxManager | None" = None,
        artifact_manager_factory: Optional["Callable[[str], ArtifactManager]"] = None,
    ) -> None:
        self._registry = agent_registry
        self._config_manager = config_manager
        self._schema_loader = schema_loader
        self._sandbox_manager = sandbox_manager
        # When None, no canary routing — every request gets active artefacts.
        self._artifact_manager_factory = artifact_manager_factory
        self._query_rewriter = None
        self._gateway_agents: TenantLRUCache[_GatewayAgentEntry] = (
            _new_gateway_agent_cache()
        )
        self._gateway_artifact_ttl_s: float = GATEWAY_ARTIFACT_TTL_S
        # Generic A2A agents cached per (tenant, agent_name): the per-tenant
        # value is a dict keyed by agent_name, since a tenant runs several
        # generic agents (entity_extraction, query_enhancement, …).
        self._generic_agents: TenantLRUCache[Dict[str, _GenericAgentEntry]] = (
            _new_generic_agent_cache()
        )
        self._generic_agent_ttl_s: float = GENERIC_AGENT_TTL_S
        # In-flight cold-build guards: a cache miss serializes concurrent
        # first-touches for the same key through one build so N first requests
        # don't each run a full build (schema deploy, mem0 init) and discard
        # N-1. Each maps key -> the Future of the build currently running;
        # removed in the build's finally, so the dict only holds active builds.
        self._gateway_build_inflight: Dict[str, "asyncio.Future[Any]"] = {}
        self._generic_build_inflight: Dict[Tuple[str, str], "asyncio.Future[Any]"] = {}
        # _get_search_agent and _get_rail_chains run inside asyncio.to_thread OS
        # threads (not on the loop), so their per-key cold-builds — each a
        # heavyweight SearchAgent init or a Vespa get_config read — are guarded
        # by threading.Locks. An asyncio Future guard (as the agent caches use)
        # would not serialize concurrent thread first-touches.
        self._search_agent_cache: Dict[str, Any] = {}
        self._search_agent_cache_lock = threading.Lock()
        self._rail_chains_cache: Dict[str, Optional[tuple]] = {}
        self._rail_chains_cache_lock = threading.Lock()
        # Per-tenant OrchestratorAgent cache — the orchestration path (dispatch
        # and streaming) resolves through it so the workflow corpus is read once
        # per TTL, not per complex query.
        self._orchestrator_agents: TenantLRUCache[_OrchestratorAgentEntry] = (
            _new_orchestrator_agent_cache(
                on_evict=self._close_evicted_orchestrator_client
            )
        )
        self._orchestrator_artifact_ttl_s: float = ORCHESTRATOR_ARTIFACT_TTL_S
        self._orchestrator_build_inflight: Dict[str, "asyncio.Future[Any]"] = {}
        # Strong references to fire-and-forget tasks so CPython does not GC
        # the coroutine before it runs. asyncio.create_task() with the result
        # discarded is documented to allow that — keep the handles, discard
        # them on completion via add_done_callback.
        self._background_tasks: set[asyncio.Task[Any]] = set()

    def _resolve_gliner_url(self) -> Optional[str]:
        """Look up the deployed GLiNER sidecar URL from system config.

        Returns the URL when ``inference.gliner`` is enabled in the
        chart, otherwise None — in which case GatewayAgent falls back
        to in-process loading (which only works when the runtime image
        has the ``torch-local`` extras installed).
        """
        try:
            sys_cfg = self._config_manager.get_system_config()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("get_system_config failed for gliner lookup: %s", exc)
            return None
        return (sys_cfg.inference_service_urls or {}).get("gliner")

    async def _reload_entry_artifact(self, entry: Any, ttl: float) -> None:
        """Re-read a cached agent's artifact once its reload interval elapsed.

        Stamps ``loaded_at`` BEFORE the await so concurrent dispatches serve the
        current instance instead of stampeding duplicate reloads. On a transient
        artifact-store failure, keep serving the still-valid cached agent and
        reschedule the reload ``RELOAD_RETRY_COOLDOWN_S`` out (not a full TTL, not
        every request) — a blip on the boundary-crossing request must not 500 it.
        """
        if not hasattr(entry.agent, "_load_artifact"):
            return
        entry.loaded_at = time.monotonic()
        try:
            await asyncio.to_thread(entry.agent._load_artifact)
        except Exception as exc:  # noqa: BLE001 — serve cached, retry soon
            entry.loaded_at = time.monotonic() - max(0.0, ttl - RELOAD_RETRY_COOLDOWN_S)
            logger.warning(
                "Artifact reload failed; serving cached agent, retry in ~%.0fs: %s",
                RELOAD_RETRY_COOLDOWN_S,
                exc,
            )

    async def _build_gateway_agent(self, tenant_id: str):
        """Construct a GatewayAgent for ``tenant_id`` with its per-tenant wiring:
        routing-config GLiNER seed, telemetry, and the first artifact read. Cached
        once per tenant by :meth:`_get_or_build_gateway_agent`."""
        from cogniverse_agents.gateway_agent import GatewayAgent, GatewayDeps
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        deps = GatewayDeps(gliner_inference_url=self._resolve_gliner_url())
        # Seed GLiNER config from the tenant's routing config (dashboard-editable
        # base values); the optimization artifact loaded below overrides the
        # fields it tunes, so tuned tenants keep their compiled thresholds.
        routing_cfg = self._config_manager.get_routing_config(tenant_id)
        deps.gliner_model_name = routing_cfg.gliner_model
        deps.gliner_threshold = routing_cfg.gliner_threshold
        deps.gliner_device = routing_cfg.gliner_device
        deps.fast_path_confidence_threshold = routing_cfg.fast_path_confidence_threshold
        deps.enable_fast_path = routing_cfg.enable_fast_path
        agent = GatewayAgent(deps=deps)
        agent.telemetry_manager = get_telemetry_manager()
        agent._artifact_tenant_id = tenant_id
        await asyncio.to_thread(agent._load_artifact)
        return agent

    async def _get_or_build_gateway_agent(self, tenant_id: str):
        """Return the GatewayAgent for ``tenant_id``, building it once per tenant.

        The gateway's routing thresholds (fast_path_confidence_threshold,
        gliner_threshold) are loaded per tenant from the artifact store into
        ``deps``, so the instance is tenant-specific. A single shared instance
        would bake in whichever tenant constructed it first and serve every
        other tenant that tenant's thresholds. Both the dispatch and the
        streaming path resolve their gateway through here so streaming also
        loads the tenant artifact (and telemetry) instead of running on
        defaults.

        Cache hits re-run ``_load_artifact`` once the reload interval has
        elapsed, so a warm pod starts serving recalibrated thresholds without
        a restart while keeping the artifact read off the per-request path.
        Concurrent first-touches for a cold tenant funnel through a single
        in-flight build so they don't run duplicate GLiNER seeds + reads.
        """
        cache = getattr(self, "_gateway_agents", None)
        if cache is None:
            cache = _new_gateway_agent_cache()
            self._gateway_agents = cache
        entry = cache.get(tenant_id)
        if entry is not None:
            ttl = getattr(self, "_gateway_artifact_ttl_s", GATEWAY_ARTIFACT_TTL_S)
            if time.monotonic() - entry.loaded_at >= ttl:
                await self._reload_entry_artifact(entry, ttl)
            return entry.agent

        inflight = getattr(self, "_gateway_build_inflight", None)
        if inflight is None:
            inflight = {}
            self._gateway_build_inflight = inflight
        pending = inflight.get(tenant_id)
        if pending is not None:
            return await pending
        fut: "asyncio.Future[Any]" = asyncio.get_event_loop().create_future()
        fut.add_done_callback(_retrieve_future_exc)
        inflight[tenant_id] = fut
        try:
            agent = await self._build_gateway_agent(tenant_id)
            cache.set(
                tenant_id, _GatewayAgentEntry(agent=agent, loaded_at=time.monotonic())
            )
            fut.set_result(agent)
            return agent
        except Exception as exc:
            fut.set_exception(exc)
            raise
        finally:
            inflight.pop(tenant_id, None)

    def _close_evicted_orchestrator_client(
        self, tenant_id: str, entry: "_OrchestratorAgentEntry"
    ) -> None:
        """Close the evicted orchestrator's pooled http client.

        Wired as the orchestrator cache's ``on_evict``. The cached
        OrchestratorAgent owns a per-tenant policy-enforcing
        ``httpx.AsyncClient``; an LRU capacity evict or a same-key
        set-displace drops the entry, and without closing the client its
        connection pool and file descriptors leak for the pod's lifetime.
        ``on_evict`` runs synchronously while ``aclose()`` is async, so
        schedule the close on the running loop and keep a strong reference to
        the task so CPython does not GC the coroutine before it runs; fall
        back to ``asyncio.run`` when no loop is running.
        """
        agent = getattr(entry, "agent", None)
        client = getattr(agent, "_http_client_override", None)
        aclose = getattr(client, "aclose", None)
        if not callable(aclose):
            return
        coro = aclose()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            asyncio.run(coro)
            return
        task = loop.create_task(coro)
        tasks = getattr(self, "_background_tasks", None)
        if tasks is not None:
            tasks.add(task)
            task.add_done_callback(tasks.discard)

    async def _build_orchestrator_agent(self, tenant_id: str):
        """Construct the per-tenant OrchestratorAgent with its WorkflowIntelligence
        corpus, policy-enforcing client, memory wiring, and first artifact read.
        Cached once per tenant by :meth:`_get_or_build_orchestrator`; the workflow
        corpus and templates then refresh on the TTL reload, not per request."""
        from cogniverse_agents.orchestrator_agent import (
            OrchestratorAgent,
            OrchestratorDeps,
        )
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        tm = get_telemetry_manager()
        # WorkflowIntelligence lets the OrchestratorAgent load workflow templates
        # from the artifact store; built per tenant and refreshed on TTL reload.
        workflow_intelligence = None
        if tm is not None:
            try:
                from cogniverse_agents.workflow.intelligence import WorkflowIntelligence

                workflow_intelligence = WorkflowIntelligence(
                    tm.get_provider(tenant_id=tenant_id), tenant_id
                )
            except Exception as exc:
                logger.debug("WorkflowIntelligence init failed (non-fatal): %s", exc)

        # Policy-enforcing client restricts outbound A2A calls to the endpoints in
        # configs/agent_policies/orchestrator_agent.yaml. It lives with the cached
        # agent (one pooled client per tenant) rather than being rebuilt and
        # closed per request.
        orch_http_client = None
        if self._sandbox_manager is not None:
            try:
                orch_http_client = self._sandbox_manager.make_http_client(
                    "orchestrator_agent"
                )
            except Exception as exc:
                logger.debug(
                    "Could not build policy-enforcing client for orchestrator: %s",
                    exc,
                )

        agent = OrchestratorAgent(
            deps=OrchestratorDeps(),
            registry=self._registry,
            config_manager=self._config_manager,
            workflow_intelligence=workflow_intelligence,
            http_client=orch_http_client,
        )
        await asyncio.to_thread(
            self._init_agent_memory, agent, "orchestrator_agent", tenant_id
        )
        agent.telemetry_manager = tm
        agent._artifact_tenant_id = tenant_id
        await asyncio.to_thread(agent._load_artifact)
        return agent

    async def _get_or_build_orchestrator(self, tenant_id: str):
        """Return the OrchestratorAgent for ``tenant_id``, building it once per
        tenant. Both the dispatch and the streaming path resolve through here, so
        streaming also loads the workflow templates (it previously planned with
        none) and a warm pod amortizes the workflow-corpus reads across the TTL
        instead of paying 4+ Phoenix reads per complex query. Cache hits re-run
        ``_load_artifact`` past the reload interval; concurrent first-touches for
        a cold tenant funnel through a single in-flight build.
        """
        cache = getattr(self, "_orchestrator_agents", None)
        if cache is None:
            cache = _new_orchestrator_agent_cache(
                on_evict=self._close_evicted_orchestrator_client
            )
            self._orchestrator_agents = cache
        entry = cache.get(tenant_id)
        if entry is not None:
            ttl = getattr(
                self, "_orchestrator_artifact_ttl_s", ORCHESTRATOR_ARTIFACT_TTL_S
            )
            if time.monotonic() - entry.loaded_at >= ttl:
                await self._reload_entry_artifact(entry, ttl)
            return entry.agent

        inflight = getattr(self, "_orchestrator_build_inflight", None)
        if inflight is None:
            inflight = {}
            self._orchestrator_build_inflight = inflight
        pending = inflight.get(tenant_id)
        if pending is not None:
            return await pending
        fut: "asyncio.Future[Any]" = asyncio.get_event_loop().create_future()
        fut.add_done_callback(_retrieve_future_exc)
        inflight[tenant_id] = fut
        try:
            agent = await self._build_orchestrator_agent(tenant_id)
            cache.set(
                tenant_id,
                _OrchestratorAgentEntry(agent=agent, loaded_at=time.monotonic()),
            )
            fut.set_result(agent)
            return agent
        except Exception as exc:
            fut.set_exception(exc)
            raise
        finally:
            inflight.pop(tenant_id, None)

    def _bind_graph_manager(self, agent: Any, tenant_id: str) -> None:
        """Bind the tenant's Vespa knowledge-graph manager to a graph-aware
        agent (one exposing ``set_graph_manager``) so its graph methods can
        enrich the mem0 answer with the shared, provenance-rich KG. No-op
        (logged) when the agent isn't graph-bindable or the graph backend isn't
        configured — agents fall back to the mem0-only path.
        """
        setter = getattr(agent, "set_graph_manager", None)
        if not callable(setter):
            return
        try:
            from cogniverse_runtime.routers.graph import get_graph_manager

            # Read-side enrichment bind: deploy=False so consulting the graph
            # never triggers a schema redeploy that could drop just-fed rows.
            setter(get_graph_manager(tenant_id, deploy=False))
        except Exception as exc:  # graph backend not configured / unavailable
            logger.warning(
                "Graph manager bind skipped for tenant %s: %s", tenant_id, exc
            )

    def _init_agent_memory(self, agent: Any, agent_name: str, tenant_id: str) -> None:
        """Auto-initialize MemoryAwareMixin for any agent that supports it.

        Checks at runtime whether the constructed agent inherits the mixin
        and, if so, runs ``initialize_memory`` with the dispatcher's own
        config_manager and schema_loader. No-ops for agents that don't
        inherit the mixin (e.g. ImageSearchAgent). Errors during init are
        logged but never raised — memory is best-effort enrichment, not a
        hard dependency.
        """
        from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin

        if not isinstance(agent, MemoryAwareMixin):
            return

        # Always set tenant for instruction lookup, even if full memory init fails.
        try:
            agent.set_tenant_for_context(tenant_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("set_tenant_for_context failed for %s: %s", agent_name, exc)

        sys_cfg = self._config_manager.get_system_config()
        # Prefer llm_config.primary.model over SystemConfig.llm_model — the
        # latter is a dataclass default that drifts from the chart.
        # Strip provider prefix: Mem0's openai provider wants just the model
        # name; the OAI-compat /v1 endpoint adds its own routing.
        from cogniverse_foundation.config.utils import get_config

        config = get_config(tenant_id=tenant_id, config_manager=self._config_manager)
        llm_primary = config.get_llm_config().primary
        model_name = llm_primary.model
        if "/" in model_name:
            model_name = model_name.split("/", 1)[1]

        denseon_url = sys_cfg.inference_service_urls.get("denseon")
        if not denseon_url:
            logger.warning(
                "Memory init for %s skipped: denseon inference service not "
                "deployed (available: %s)",
                agent_name,
                sorted(sys_cfg.inference_service_urls),
            )
            return

        try:
            agent.initialize_memory(
                agent_name=agent_name,
                tenant_id=tenant_id,
                backend_host=sys_cfg.backend_url,
                backend_port=sys_cfg.backend_port,
                llm_model=model_name,
                embedding_model="lightonai/DenseOn",
                llm_base_url=llm_primary.api_base
                or getattr(sys_cfg, "base_url", "http://localhost:11434"),
                embedder_base_url=denseon_url,
                config_manager=self._config_manager,
                schema_loader=self._schema_loader,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize memory for %s (tenant=%s): %s",
                agent_name,
                tenant_id,
                exc,
            )

    def consult_egress_policy(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Look up the OpenShell egress policy for an agent (P3 wire).

        Returns the parsed policy dict (the YAML in
        ``configs/agent_policies/<agent>.yaml``) or ``None`` when no sandbox
        manager is wired or no policy is registered.

        Two-layer enforcement story for the agents that don't own their
        own httpx client (search / summarizer / routing — they go through
        DSPy / pyvespa):

          * **L4 / kernel**: the unified-runtime NetworkPolicy generated
            by the egress-netpol CLI covers the union of every
            agent's egress allowlist at the cluster boundary. A call to
            an off-allowlist destination is denied by the CNI before it
            leaves the pod.
          * **Dispatch-time validation** (this method, when paired with
            ``validate_dispatch_endpoints``): for the destinations the
            dispatcher KNOWS the agent will use (Vespa, LLM endpoint,
            etc.), confirm they're in the policy's egress allow-list
            before dispatching. Catches misconfiguration at boot/test
            time, before a runtime call hits L4 enforcement.
        """
        if self._sandbox_manager is None:
            return None
        try:
            policy = self._sandbox_manager.get_policy(agent_name)
        except Exception as exc:
            logger.debug("egress policy lookup failed for %s: %s", agent_name, exc)
            return None
        if policy is not None:
            logger.info(
                "Dispatch: egress policy active for %s (rules=%d)",
                agent_name,
                len(policy.get("egress", {}).get("allow", []))
                if isinstance(policy.get("egress"), dict)
                else 0,
            )
        return policy

    def _system_endpoints(self, tenant_id: str) -> Dict[str, Dict[str, Any]]:
        """Return host/port for the runtime's standard outbound services.

        These are the destinations the dispatcher already knows the
        agents will reach (Vespa for retrieval, the LLM endpoint for
        DSPy calls, denseon for embeddings). Used by the per-agent
        ``_verify_*_egress`` helpers below to drift-check policies.
        Best-effort: missing config keys return an empty dict and the
        verification step is skipped silently.
        """
        from urllib.parse import urlparse

        try:
            sys_cfg = self._config_manager.get_system_config()
        except Exception:
            return {}

        out: Dict[str, Dict[str, Any]] = {}

        def _add(name: str, url: Optional[str], default_port: int) -> None:
            if not url:
                return
            parsed = urlparse(url if "://" in url else f"http://{url}")
            host = parsed.hostname or "localhost"
            port = parsed.port or default_port
            out[name] = {"host": host, "port": int(port), "protocol": "tcp"}

        _add(
            "vespa",
            f"{sys_cfg.backend_url}:{sys_cfg.backend_port}",
            sys_cfg.backend_port or 8080,
        )
        try:
            from cogniverse_foundation.config.utils import get_config

            cfg = get_config(tenant_id=tenant_id, config_manager=self._config_manager)
            llm_primary = cfg.get_llm_config().primary
            if llm_primary.api_base:
                _add("llm", llm_primary.api_base, 11434)
        except Exception as exc:  # noqa: BLE001 — log + degrade
            # Failure here silently drops the LLM endpoint from the
            # egress allow-list the surrounding code uses to authorize
            # outbound calls. Log so the policy hole is visible.
            logger.warning(
                "LLM endpoint resolution failed for tenant %s (egress "
                "allow-list will not include the LLM endpoint): %s",
                tenant_id,
                exc,
            )
        denseon_url = (sys_cfg.inference_service_urls or {}).get("denseon")
        if denseon_url:
            _add("denseon", denseon_url, 8000)
        return out

    def _verify_search_egress(self, tenant_id: str) -> None:
        sysep = self._system_endpoints(tenant_id)
        # SearchAgent reaches Vespa for retrieval and (via DSPy) the LLM
        # for query rewriting. Denseon embeddings are pulled by Mem0 in
        # this runtime, not the agent itself, so they're not flagged here.
        endpoints = [v for k, v in sysep.items() if k in {"vespa", "llm"}]
        if endpoints:
            self.validate_dispatch_endpoints("search_agent", endpoints)

    def _verify_summarizer_egress(self, tenant_id: str) -> None:
        sysep = self._system_endpoints(tenant_id)
        # SummarizerAgent reaches the LLM endpoint via DSPy.
        endpoints = [v for k, v in sysep.items() if k == "llm"]
        if endpoints:
            self.validate_dispatch_endpoints("summarizer_agent", endpoints)

    def _verify_routing_egress(self, tenant_id: str) -> None:
        sysep = self._system_endpoints(tenant_id)
        # RoutingAgent (gateway) classifies via the LLM and may dispatch
        # to in-cluster agent peers; the in-cluster A2A endpoints aren't
        # listed in agent policies (they're loopback within the unified
        # runtime pod), so only the LLM hop is verified.
        endpoints = [v for k, v in sysep.items() if k == "llm"]
        if endpoints:
            self.validate_dispatch_endpoints("routing_agent", endpoints)

    def validate_dispatch_endpoints(
        self,
        agent_name: str,
        endpoints: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Verify each (host, port) endpoint matches the agent's egress policy.

        Returns a list of violations: each entry is the original endpoint
        dict plus a ``reason`` field. Empty list means every endpoint is
        on the allowlist (or no policy is registered for the agent).

        Application-layer half of egress enforcement for shared-runtime
        agents. The CNI-level NetworkPolicy is the actual deny mechanism
        in production; this dispatch-time check catches the
        common case where an operator updated a backend port without
        updating the policy YAML, surfacing the drift as a logged
        warning before a single request runs.

        Each ``endpoint`` dict must carry ``host`` and ``port`` keys; an
        optional ``protocol`` (default ``tcp``) is matched case-insensitively.
        """
        if self._sandbox_manager is None:
            return []
        policy = self._sandbox_manager.get_policy(agent_name)
        if not policy:
            return []
        net = policy.get("network_policies") or policy.get("egress") or {}
        if isinstance(net, dict):
            allowed = net.get("egress") or net.get("allow") or []
        else:
            allowed = net or []
        # Normalise the allowlist to a set of (host, port, protocol).
        allow_set: set = set()
        for rule in allowed:
            if not isinstance(rule, dict):
                continue
            allow_set.add(
                (
                    str(rule.get("host", "")).lower(),
                    int(rule.get("port", 0) or 0),
                    str(rule.get("protocol", "tcp")).lower(),
                )
            )

        violations: List[Dict[str, Any]] = []
        for ep in endpoints:
            key = (
                str(ep.get("host", "")).lower(),
                int(ep.get("port", 0) or 0),
                str(ep.get("protocol", "tcp")).lower(),
            )
            if key not in allow_set:
                violations.append(
                    {
                        **ep,
                        "reason": (
                            f"host={key[0]} port={key[1]} protocol={key[2]} "
                            f"not in egress allowlist for agent={agent_name}"
                        ),
                    }
                )
        if violations:
            logger.warning(
                "egress policy DRIFT for %s: %d endpoint(s) outside allowlist; "
                "first violation: %s",
                agent_name,
                len(violations),
                violations[0]["reason"],
            )
        return violations

    async def resolve_artefact_for_request(
        self,
        agent_name: str,
        tenant_id: str,
        request_seed: str,
    ) -> Optional[Dict[str, Any]]:
        """Per-request canary-aware artefact resolution.

        Returns ``{"served_from": "active|canary|default", "version": int|None,
        "prompts": {...}|None}`` when an ``artifact_manager_factory`` was
        provided to the dispatcher; otherwise returns ``None``. The caller
        passes the result down to the agent constructor (or stashes it in
        the dispatch context) so the agent uses the chosen variant of the
        compiled prompts.

        Without this method, every request hit the active artefacts and the
        canary state machine in :class:`ArtifactManager` was unreachable
        from the live dispatch path — a P1.4 wiring gap.
        """
        if self._artifact_manager_factory is None:
            return None
        try:
            am = self._artifact_manager_factory(tenant_id)
        except Exception as exc:
            # A telemetry/Phoenix outage here silently reverts every request to
            # the un-optimized DEFAULT prompts. That's an acceptable degrade, but
            # it must be VISIBLE (warning, not debug) so an operator sees that
            # optimized prompts stopped being served.
            logger.warning(
                "Artefact factory failed for tenant=%s — serving default prompts: %s",
                tenant_id,
                exc,
            )
            return None

        # Consumer for PUT /admin/tenants/{t}/signature_variants/{agent};
        # falls back to default when no admin selection exists. Warm the
        # blob-backed cache first (TTL-bounded) so the selection an admin
        # persisted reaches this replica, not just the one that served the PUT.
        try:
            from cogniverse_runtime.routers.admin import load_signature_variants

            await load_signature_variants(tenant_id)
        except Exception as exc:
            logger.debug("signature-variant cache warm skipped: %s", exc)
        variant_id = self._resolve_signature_variant(tenant_id, agent_name)

        try:
            return await am.load_for_request(
                agent_name,
                request_seed=request_seed,
                variant_id=variant_id,
            )
        except Exception as exc:
            # Same degraded-mode reversion as above — warn so the fallback to
            # default prompts is not invisible.
            logger.warning(
                "load_for_request(%s, seed=%s, variant=%s) failed — serving "
                "default prompts: %s",
                agent_name,
                request_seed,
                variant_id,
                exc,
            )
            return None

    @staticmethod
    def _apply_artefact_overlay(agent: Any, context: Optional[Dict[str, Any]]) -> None:
        """Inject the dispatcher's per-request artefact overlay onto an agent.

        The dispatcher already produces the canary/variant decision via
        :meth:`resolve_artefact_for_request` and stashes it in
        ``context["_artefact_overlay"]``. This helper hands it to the
        agent via the ``MemoryAwareMixin.set_dispatched_artefact`` hook.
        Agents that don't inherit the mixin silently no-op (no regression).

        Without this call, the overlay is unreachable from agent code and
        canary traffic-split / admin variant selection have no production
        effect.
        """
        if context is None:
            return
        overlay = context.get("_artefact_overlay")
        if overlay is None:
            return
        setter = getattr(agent, "set_dispatched_artefact", None)
        if setter is None:
            return
        try:
            setter(overlay)
        except Exception as exc:
            logger.debug("set_dispatched_artefact failed (non-fatal): %s", exc)

    @staticmethod
    @contextmanager
    def _scoped_session(agent: Any, session_id: Optional[str]):
        """Stamp ``session_id`` on a memory-aware agent for one request.

        The mixin's ``set_session_id`` hook auto-applies the id to every
        ``add_memory`` write so EPHEMERAL_SESSION-kind writes pass schema
        validation without each agent threading session_id through its
        own metadata. Cleared on exit so a long-lived agent instance
        doesn't bleed one request's session into the next.

        Agents that don't inherit MemoryAwareMixin silently no-op.
        """
        setter = getattr(agent, "set_session_id", None)
        if setter is None or not session_id:
            yield
            return
        try:
            setter(session_id)
        except Exception as exc:
            logger.debug("set_session_id failed (non-fatal): %s", exc)
            yield
            return
        try:
            yield
        finally:
            try:
                setter(None)
            except Exception as exc:  # noqa: BLE001 — log + degrade
                # Session-clear is best-effort cleanup; failures here
                # leak session-id into the next request handled by this
                # event-loop slot. Log so the leak is investigable.
                logger.warning(
                    "Session-id clear failed (next request may inherit "
                    "stale session_id): %s",
                    exc,
                )

    @staticmethod
    def _resolve_signature_variant(tenant_id: str, agent_name: str) -> str:
        """Read the tenant's selected variant for an agent from admin overrides.

        Falls back to ``DEFAULT_VARIANT_ID`` when no admin PUT has set one.
        Lazy import — admin lives in the runtime layer, this dispatcher
        also lives there but we keep the import local so test setups
        that don't load admin still work.
        """
        from cogniverse_agents.optimizer.signature_variants import (
            DEFAULT_VARIANT_ID,
        )

        try:
            from cogniverse_runtime.routers.admin import (
                _signature_variant_overrides as _admin_overrides,
            )
        except Exception:
            return DEFAULT_VARIANT_ID

        # Canonicalize the key so a variant set via admin PUT resolves here
        # regardless of whether the tenant arrived as simple ('acme') or
        # colon ('acme:acme') form — the admin endpoints (which canonicalize
        # their sibling routes) store under the canonical key too.
        per_tenant = _admin_overrides.get(canonical_tenant_id(tenant_id)) or {}
        return per_tenant.get(agent_name) or DEFAULT_VARIANT_ID

    async def dispatch(
        self,
        agent_name: str,
        query: str,
        context: Dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Dispatch a task to the named agent and return the result.

        Raises:
            ValueError: If agent is not found or has no supported execution path.
        """
        if context is None:
            context = {}

        agent = self._registry.get_agent(agent_name)
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found in registry")

        tenant_id = require_tenant_id(
            context.get("tenant_id"), source="AgentTask.context"
        )

        request_seed = str(
            context.get("request_id") or context.get("request_seed") or ""
        )
        if request_seed and self._artifact_manager_factory is not None:
            artefact = await self.resolve_artefact_for_request(
                agent_name, tenant_id, request_seed
            )
            if artefact is not None:
                context["_artefact_overlay"] = artefact
        capabilities = set(agent.capabilities)

        conversation_history = context.get("conversation_history", [])

        enrichment = {
            k: context[k]
            for k in (
                "enhanced_query",
                "entities",
                "relationships",
                "query_variants",
                "profiles",
            )
            if k in context
        }

        if capabilities & {"gateway", "routing", "intelligent_routing"}:
            result = await self._execute_gateway_task(query, context, tenant_id)
        elif "orchestration" in capabilities:
            result = await self._execute_orchestration_task(query, context, tenant_id)
        elif capabilities & {"search", "video_search", "retrieval"}:
            result = await self._execute_search_task(
                query,
                tenant_id,
                top_k,
                conversation_history=conversation_history,
                enrichment=enrichment or None,
                context=context,
            )
        elif capabilities & {"image_search", "visual_analysis"}:
            result = await self._execute_image_search_task(query, tenant_id, top_k)
        elif capabilities & {"audio_analysis", "transcription"}:
            result = await self._execute_audio_search_task(query, tenant_id, top_k)
        elif capabilities & {"document_analysis", "pdf_processing"}:
            result = await self._execute_document_search_task(query, tenant_id, top_k)
        elif capabilities & {"detailed_report"}:
            result = await self._execute_detailed_report_task(
                query, tenant_id, context=context
            )
        elif capabilities & {"summarization", "text_generation"}:
            result = await self._execute_summarization_task(
                query, tenant_id, context=context
            )
        elif capabilities & {"text_analysis", "sentiment", "classification"}:
            result = await self._execute_text_analysis_task(query, context, tenant_id)
        elif "deep_research" in capabilities:
            result = await self._execute_deep_research_task(query, tenant_id)
        elif "coding" in capabilities:
            result = await self._execute_coding_task(query, tenant_id, context)
        else:
            result = await self._execute_generic_agent(
                agent_name, query, context, tenant_id
            )

        entities = result.get("entities", [])
        turn_count = len(conversation_history or []) // 2 + 1
        self._spawn_background(
            self._maybe_auto_file_wiki(
                query, result, entities, agent_name, tenant_id, turn_count
            )
        )
        return result

    def _spawn_background(self, coro) -> asyncio.Task:
        """Schedule a fire-and-forget coroutine while keeping a strong
        reference, so CPython does not GC the task before it runs."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _maybe_auto_file_wiki(
        self,
        query: str,
        response: Dict[str, Any],
        entities: List[str],
        agent_name: str,
        tenant_id: str,
        turn_count: int,
    ) -> None:
        """Fire-and-forget wiki auto-filing. Non-fatal: logs and returns on any error.

        Resolves the per-tenant WikiManager via the router factory so each
        tenant's auto-filed pages land in their own wiki.
        """
        try:
            from cogniverse_runtime.routers import wiki as wiki_router

            if wiki_router._wiki_manager_factory is None:
                return

            wm = wiki_router._wiki_manager_factory(tenant_id)
            if wm is None:
                return

            if not wm._should_auto_file(entities, agent_name, turn_count):
                return

            response_text = str(response.get("answer", response))
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: wm.save_session(
                    query=query,
                    response=response_text,
                    entities=entities,
                    agent_name=agent_name,
                ),
            )
        except Exception:
            logger.warning("Wiki auto-filing failed (non-fatal)", exc_info=True)

    async def _build_generic_agent(
        self,
        agent_name: str,
        tenant_id: str,
        agent_cls: Any,
        deps_cls: Any,
    ) -> Any:
        """Construct a generic A2A agent with its per-tenant wiring.

        Runs the expensive one-time setup: default deps (with the GLiNER
        sidecar URL when the schema accepts it), ``_init_agent_memory`` and
        ``_bind_graph_manager`` off the event loop, telemetry + config-manager
        injection, and the first artifact read. The result is cached per
        (tenant, agent_name) by :meth:`_get_or_build_generic_agent`.
        """
        # If the deps schema accepts a gliner_inference_url field
        # (EntityExtractionAgent + any future gliner-using agent), inject the
        # deployed sidecar URL so the agent doesn't fall back to in-process
        # loading.
        deps_kwargs: Dict[str, Any] = {}
        gliner_url = self._resolve_gliner_url()
        if gliner_url and "gliner_inference_url" in deps_cls.model_fields:
            deps_kwargs["gliner_inference_url"] = gliner_url
        deps = deps_cls(**deps_kwargs)
        agent = agent_cls(deps=deps)
        await asyncio.to_thread(self._init_agent_memory, agent, agent_name, tenant_id)
        # Bind the per-tenant Vespa knowledge-graph manager so KG-aware agents
        # can consult the shared graph (with cross-document Mention provenance)
        # complementary to their own mem0 memory. Fail-safe: agents fall back to
        # the mem0-only path when the graph backend isn't configured.
        await asyncio.to_thread(self._bind_graph_manager, agent, tenant_id)

        # Inject global TelemetryManager for span emission
        from cogniverse_foundation.telemetry.manager import get_telemetry_manager

        agent.telemetry_manager = get_telemetry_manager()
        agent._artifact_tenant_id = tenant_id
        # Config access for per-tenant adapter/LM resolution (e.g. the
        # profile_selection adapter context) — same injection pattern as
        # telemetry_manager. Never clobber an agent-owned manager.
        if getattr(agent, "_config_manager", None) is None:
            agent._config_manager = self._config_manager
        if hasattr(agent, "_load_artifact"):
            await asyncio.to_thread(agent._load_artifact)
        return agent

    async def _get_or_build_generic_agent(
        self,
        agent_name: str,
        tenant_id: str,
        agent_cls: Any,
        deps_cls: Any,
    ) -> Any:
        """Return the cached generic agent for (tenant, agent_name), building
        it once per pair — mirroring :meth:`_get_or_build_gateway_agent`.

        The build (class wiring, memory init, graph bind, telemetry inject,
        first artifact read) ran on every dispatch before; it now runs once and
        a cache hit past the reload interval re-reads only the artifact. The
        reload stamps ``loaded_at`` before the ``to_thread`` await, so two
        concurrent dispatches never stampede duplicate reloads. Per-request
        state (artefact overlay, session id) is applied by the caller on every
        dispatch — it lives in Task-isolated ContextVars, so a shared cached
        instance is safe.
        """
        cache = getattr(self, "_generic_agents", None)
        if cache is None:
            cache = _new_generic_agent_cache()
            self._generic_agents = cache
        per_tenant = cache.get_or_set(tenant_id, dict)
        entry = per_tenant.get(agent_name)
        if entry is not None:
            ttl = getattr(self, "_generic_agent_ttl_s", GENERIC_AGENT_TTL_S)
            if time.monotonic() - entry.loaded_at >= ttl:
                await self._reload_entry_artifact(entry, ttl)
            return entry.agent

        # Cache miss: funnel concurrent first-touches for this (tenant,
        # agent_name) through a single in-flight build so they don't each run a
        # full build (class wiring, memory init, graph bind, schema-deploying
        # artifact read) and discard all but one.
        key = (tenant_id, agent_name)
        inflight = getattr(self, "_generic_build_inflight", None)
        if inflight is None:
            inflight = {}
            self._generic_build_inflight = inflight
        pending = inflight.get(key)
        if pending is not None:
            return await pending
        fut: "asyncio.Future[Any]" = asyncio.get_event_loop().create_future()
        fut.add_done_callback(_retrieve_future_exc)
        inflight[key] = fut
        try:
            agent = await self._build_generic_agent(
                agent_name, tenant_id, agent_cls, deps_cls
            )
            # Re-fetch the per-tenant dict in case the tenant was LRU-evicted
            # during the build's awaits.
            cache.get_or_set(tenant_id, dict)[agent_name] = _GenericAgentEntry(
                agent=agent, loaded_at=time.monotonic()
            )
            fut.set_result(agent)
            return agent
        except Exception as exc:
            fut.set_exception(exc)
            raise
        finally:
            inflight.pop(key, None)

    async def _execute_generic_agent(
        self,
        agent_name: str,
        query: str,
        context: Dict[str, Any],
        tenant_id: str,
    ) -> Dict[str, Any]:
        """Generic A2A agent execution via dynamic class import.

        Handles any registered agent by importing its class from
        ConfigLoader.AGENT_CLASSES, instantiating with default deps,
        and calling process(). This is the path taken by preprocessing
        agents (entity_extraction, query_enhancement, profile_selection)
        when the OrchestratorAgent calls them via A2A HTTP.
        """
        import importlib

        from cogniverse_runtime.config_loader import ConfigLoader

        # AGENT_CLASSES is keyed on the fully-qualified type name (e.g.
        # "query_enhancement_agent"), but the orchestrator plans steps
        # with the bare capability name ("query_enhancement"). Accept
        # both: try the given name first, then the "_agent" suffixed
        # form.
        class_path = ConfigLoader.AGENT_CLASSES.get(agent_name) or (
            ConfigLoader.AGENT_CLASSES.get(f"{agent_name}_agent")
        )
        if not class_path:
            raise ValueError(
                f"Agent '{agent_name}' has no supported execution path "
                f"(not in AGENT_CLASSES)"
            )

        module_path, class_name = class_path.split(":")
        module = importlib.import_module(module_path)

        # Convention lookup of the (agent, Deps, Input) classes is memoized per
        # class path — two dir(module) scans per dispatch otherwise.
        agent_cls, deps_cls, input_cls = _resolve_generic_agent_classes(
            class_path, module, class_name
        )

        if deps_cls is None:
            raise ValueError(
                f"Agent '{agent_name}' has no supported execution path "
                f"(no Deps class in {module_path})"
            )
        if input_cls is None:
            raise ValueError(
                f"Agent '{agent_name}' has no supported execution path "
                f"(no Input class in {module_path})"
            )

        agent = await self._get_or_build_generic_agent(
            agent_name, tenant_id, agent_cls, deps_cls
        )
        # ``set_dispatched_artefact`` lives on MemoryAwareMixin; non-mixin
        # agents silently no-op (overlay dropped, falling back to defaults).
        # Applied on EVERY dispatch, cache hit or miss: it writes a Task-isolated
        # ContextVar, never the shared cached instance, so a warm agent never
        # bleeds one request's canary/variant overlay into another.
        self._apply_artefact_overlay(agent, context)

        # Build input — pass query + tenant_id + any extra context fields
        input_kwargs = {"query": query}
        if "tenant_id" in input_cls.model_fields:
            input_kwargs["tenant_id"] = tenant_id

        # Forward upstream agent results from context (e.g., entities from entity extraction)
        for key in ("entities", "relationships", "enhanced_query"):
            if key in context and key in input_cls.model_fields:
                input_kwargs[key] = context[key]

        typed_input = input_cls(**input_kwargs)
        # EPHEMERAL_SESSION writes need metadata.session_id to pass schema
        # validation; mixin auto-stamps it from this field. Cleared on exit
        # so the next request on the same agent instance doesn't inherit it.
        with self._scoped_session(agent, context.get("session_id")):
            result = await agent.process(typed_input)

        # Convert pydantic model to dict
        if hasattr(result, "model_dump"):
            return {"status": "success", "agent": agent_name, **result.model_dump()}
        return {"status": "success", "agent": agent_name, "result": str(result)}

    async def create_streaming_agent(
        self,
        agent_name: str,
        query: str,
        tenant_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Create a streaming-capable agent and its typed input.

        Returns (agent, typed_input) for use with agent.process(typed_input, stream=True).
        All agents support streaming via emit_progress() and call_dspy().

        For the answer agents (summary / detailed report) the typed input is
        grounded in real search results — threaded through ``context`` by the
        caller, else a fresh search — so streaming answers see the same hits
        (and answer-time keyframes) as the non-streaming dispatch path.
        """
        agent_entry = self._registry.get_agent(agent_name)
        if not agent_entry:
            raise ValueError(f"Agent '{agent_name}' not found in registry")

        capabilities = set(agent_entry.capabilities)

        if capabilities & {"gateway"}:
            from cogniverse_agents.gateway_agent import GatewayInput

            gateway_agent = await self._get_or_build_gateway_agent(tenant_id)
            typed_input = GatewayInput(query=query, tenant_id=tenant_id)
            return gateway_agent, typed_input

        # detailed_report is checked before summarization/text_generation:
        # config.json registers detailed_report_agent with a "text_generation"
        # capability, so the reverse order (matched only by summarization) would
        # serve a streamed detailed report with the SummarizerAgent. Mirrors the
        # non-streaming dispatch() order.
        if capabilities & {"detailed_report"}:
            from cogniverse_agents.detailed_report_agent import (
                DetailedReportAgent,
                DetailedReportDeps,
                DetailedReportInput,
            )

            deps = DetailedReportDeps(
                tenant_id=tenant_id,
                **self._agent_behavior_kwargs(tenant_id, "detailed_report_agent"),
            )
            agent = DetailedReportAgent(deps=deps, config_manager=self._config_manager)
            typed_input = DetailedReportInput(
                query=query,
                search_results=await self._resolve_answer_search_results(
                    query, tenant_id, context, top_k=20
                ),
            )
            return agent, typed_input

        if capabilities & {"summarization", "text_generation"}:
            from cogniverse_agents.summarizer_agent import (
                SummarizerAgent,
                SummarizerDeps,
                SummarizerInput,
            )

            deps = SummarizerDeps(
                tenant_id=tenant_id,
                **self._agent_behavior_kwargs(tenant_id, "summarizer_agent"),
            )
            agent = SummarizerAgent(deps=deps, config_manager=self._config_manager)
            typed_input = SummarizerInput(
                query=query,
                search_results=await self._resolve_answer_search_results(
                    query, tenant_id, context, top_k=10
                ),
                summary_type="general",
            )
            return agent, typed_input

        if capabilities & {"search", "video_search", "retrieval"}:
            from cogniverse_agents.search_agent import (
                SearchAgent,
                SearchAgentDeps,
                SearchInput,
            )

            system_config = self._config_manager.get_system_config()
            deps = SearchAgentDeps(
                tenant_id=tenant_id,
                backend_url=system_config.backend_url,
                backend_port=system_config.backend_port,
            )
            agent = SearchAgent(
                deps=deps,
                schema_loader=self._schema_loader,
                config_manager=self._config_manager,
            )
            typed_input = SearchInput(query=query, tenant_id=tenant_id)
            return agent, typed_input

        if capabilities & {"coding"}:
            from cogniverse_agents.coding_agent import (
                CodingAgent,
                CodingDeps,
                CodingInput,
            )
            from cogniverse_foundation.config.semantic_router import (
                create_routed_lm,
                resolve_semantic_router_config,
            )
            from cogniverse_foundation.config.utils import get_config

            # Offload the config ensure-chain (cold-config Vespa read) so the
            # streaming setup doesn't block the loop.
            config = await asyncio.to_thread(
                get_config, tenant_id=tenant_id, config_manager=self._config_manager
            )
            coding_lm = create_routed_lm(
                config.get_llm_config().resolve("coding_agent"),
                resolve_semantic_router_config(config),
                tenant_id,
            )

            deps = CodingDeps(
                tenant_id=tenant_id,
                sandbox_manager=self._sandbox_manager,
            )

            agent = CodingAgent(
                deps=deps,
                search_fn=self._code_search,
                sandbox_manager=self._sandbox_manager,
                config_manager=self._config_manager,
            )
            agent._dspy_lm = coding_lm  # type: ignore[attr-defined]
            typed_input = CodingInput(task=query, tenant_id=tenant_id)
            return agent, typed_input

        if capabilities & {"orchestration", "planning"}:
            from cogniverse_agents.orchestrator_agent import OrchestratorInput

            # Resolve through the per-tenant cache so streaming also loads the
            # workflow templates (it previously planned with none) and shares the
            # dispatch path's cached instance instead of building a bare one.
            agent = await self._get_or_build_orchestrator(tenant_id)
            return agent, OrchestratorInput(query=query, tenant_id=tenant_id)

        # Generic fallback: any registered agent that follows the
        # Agent/Deps/Input convention (image/audio/document search, entity
        # extraction, query enhancement, profile selection, …) streams via the
        # base framework. Construct it exactly as the non-streaming generic
        # path does so streaming and non-streaming stay in lockstep.
        agent_input = self._build_generic_streaming_agent(agent_name, query, tenant_id)
        if agent_input is not None:
            return agent_input

        raise ValueError(
            f"Agent '{agent_name}' streaming not configured. "
            f"Capabilities: {agent_entry.capabilities}"
        )

    def _build_generic_streaming_agent(
        self, agent_name: str, query: str, tenant_id: str
    ) -> Optional[tuple]:
        """Build (agent, typed_input) for a registered agent via the
        Agent/Deps/Input naming convention. Returns None when the agent isn't
        generically constructible (no AGENT_CLASSES entry / no Deps or Input)."""
        import importlib

        from cogniverse_runtime.config_loader import ConfigLoader

        class_path = ConfigLoader.AGENT_CLASSES.get(agent_name) or (
            ConfigLoader.AGENT_CLASSES.get(f"{agent_name}_agent")
        )
        if not class_path:
            return None

        module_path, class_name = class_path.split(":")
        module = importlib.import_module(module_path)
        agent_cls = getattr(module, class_name)

        def _by_suffix(suffix: str, exclude: str):
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and attr_name.endswith(suffix)
                    and attr_name != exclude
                ):
                    return attr
            return None

        deps_cls = _by_suffix("Deps", "AgentDeps")
        input_cls = _by_suffix("Input", "AgentInput")
        if deps_cls is None or input_cls is None:
            return None

        deps_kwargs: Dict[str, Any] = {}
        gliner_url = self._resolve_gliner_url()
        if gliner_url and "gliner_inference_url" in deps_cls.model_fields:
            deps_kwargs["gliner_inference_url"] = gliner_url
        # Search-backed agents (image/audio/document) read deps.vespa_endpoint
        # and deps.tenant_id — the latter is carried as an extra field on Deps
        # configured with ``extra="allow"``, so pass it whenever the Deps either
        # declares it or accepts extras.
        if "vespa_endpoint" in deps_cls.model_fields:
            deps_kwargs["vespa_endpoint"] = self._get_vespa_endpoint(tenant_id)
        if (
            "tenant_id" in deps_cls.model_fields
            or deps_cls.model_config.get("extra") == "allow"
        ):
            deps_kwargs["tenant_id"] = tenant_id
        agent = agent_cls(deps=deps_cls(**deps_kwargs))

        input_kwargs: Dict[str, Any] = {"query": query}
        if "tenant_id" in input_cls.model_fields:
            input_kwargs["tenant_id"] = tenant_id
        return agent, input_cls(**input_kwargs)

    async def _execute_search_task(
        self,
        query: str,
        tenant_id: str,
        top_k: int,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        enrichment: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Drift surfaces as a logged warning here; CNI is the kernel deny.
        self.consult_egress_policy("search_agent")
        self._verify_search_egress(tenant_id)

        from cogniverse_agents.search_agent import (
            SearchInput,
        )
        from cogniverse_foundation.config.utils import get_config

        # Rewrite query using conversation history to resolve anaphoric references.
        # If the DSPy rewriter fails (e.g., no LM configured), propagate with
        # a clear error rather than silently searching with the unresolved query.
        resolved_query = query
        if conversation_history:
            resolved_query = await self._rewrite_query_with_history(
                query, conversation_history
            )
            if resolved_query != query:
                logger.info(f"Query rewritten: '{query}' -> '{resolved_query}'")

        # get_config runs the ConfigUtils ensure-chain (a Vespa read on a cold
        # or TTL-expired config) — offload it so it never stalls the API loop
        # (mirrors _build_encoder_config).
        config = await asyncio.to_thread(
            get_config, tenant_id=tenant_id, config_manager=self._config_manager
        )
        # ``active_video_profile`` is the tenant's configured default video
        # profile (config.json). A per-request ``profiles`` override still wins
        # inside the SearchAgent via SearchInput.profiles.
        profile = config.get("active_video_profile", "video_colpali_smol500_mv_frame")

        # _get_search_agent builds SearchAgent on a cache miss — a synchronous
        # get_system_config Vespa read + query-encoder init in __init__; offload
        # the whole call so the first search per profile doesn't stall the loop.
        search_agent = await asyncio.to_thread(self._get_search_agent, profile)
        # Apply the dispatcher's per-request artefact overlay so the
        # canary/variant prompts shape the SearchAgent's DSPy call. Without
        # this, the overlay sits in context unread.
        self._apply_artefact_overlay(search_agent, context)

        enrichment = enrichment or {}
        input_data = SearchInput(
            query=resolved_query,
            tenant_id=tenant_id,
            top_k=top_k,
            enhanced_query=enrichment.get("enhanced_query"),
            entities=enrichment.get("entities") or [],
            relationships=enrichment.get("relationships") or [],
            query_variants=enrichment.get("query_variants") or [],
            profiles=enrichment.get("profiles"),
        )

        output = await search_agent._process_impl(input_data)

        result_list = output.results
        result_count = len(result_list)
        effective_query = output.enhanced_query or resolved_query

        if result_count > 0:
            message = f"Found {result_count} results for '{effective_query}'"
        else:
            message = f"No results found for '{effective_query}'"

        response: Dict[str, Any] = {
            "status": "success",
            "agent": "search_agent",
            "message": message,
            "results_count": result_count,
            "results": result_list,
            "profile": output.profile or profile,
            "search_mode": output.search_mode,
        }

        # Multi-turn contract: whenever conversation_history was supplied we
        # ran the rewriter, so expose both the original and the resolved query
        # in the response — even when they match. Callers can diff the two to
        # see whether the rewriter actually changed anything.
        if conversation_history:
            response["original_query"] = query
            response["rewritten_query"] = resolved_query

        return response

    def _agent_behavior_kwargs(self, tenant_id: str, agent_name: str) -> Dict[str, Any]:
        """Per-tenant thinking/visual toggles for an answer agent's Deps.

        Reads the tenant's persisted ``AgentConfig`` (set via
        ``ConfigManager.set_agent_config``) and threads ``thinking_enabled`` /
        ``visual_analysis_enabled`` into the agent's Deps. Returns an empty dict
        — so the Deps field defaults apply — when no per-tenant config is set,
        or when the config read fails (a behavior toggle must not break
        dispatch).
        """
        from cogniverse_foundation.config.agent_config import AgentConfig

        try:
            cfg = self._config_manager.get_agent_config(tenant_id, agent_name)
        except Exception as exc:
            logger.warning(
                "agent config read failed for %s/%s: %r — using Deps defaults",
                tenant_id,
                agent_name,
                exc,
            )
            return {}
        # get_agent_config is typed Optional[AgentConfig]; anything else (None,
        # or a test double that doesn't honor the contract) means no per-tenant
        # override, so the Deps field defaults apply.
        if not isinstance(cfg, AgentConfig):
            return {}
        return {
            "thinking_enabled": cfg.thinking_enabled,
            "visual_analysis_enabled": cfg.visual_analysis_enabled,
        }

    async def _resolve_answer_search_results(
        self,
        query: str,
        tenant_id: str,
        context: Optional[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Search results to ground an answer agent (detailed report / summary).

        Grounding these agents in real hits is what makes the report reflect the
        corpus (and lets keyframes reach the answer LLM). Uses results the caller
        already threaded through ``context["search_results"]`` (the orchestrator
        hands a completed search step's hits to a dependent report/summary step)
        when present; otherwise runs the tenant's default video search.

        This fallback search is best-effort: a report/summary request is not
        inherently a video-search request (a directly-dispatched summary may be
        a plain conversational ask), so an unreachable search backend degrades to
        an ungrounded answer over ``[]`` — logged, not raised — rather than
        hard-failing the whole request on unrelated video-search infra. A genuine
        zero-match likewise returns ``[]`` (the agent reports "no results").
        """
        if context:
            threaded = context.get("search_results")
            if isinstance(threaded, list):
                hits = [_flatten_search_hit(h) for h in threaded if isinstance(h, dict)]
                if hits:
                    return hits
        enrichment = None
        if context:
            enrichment = {
                k: context[k]
                for k in (
                    "enhanced_query",
                    "entities",
                    "relationships",
                    "query_variants",
                    "profiles",
                )
                if context.get(k)
            }
        try:
            search = await self._execute_search_task(
                query,
                tenant_id,
                top_k=top_k,
                enrichment=enrichment or None,
                context=context,
            )
        except Exception as exc:
            logger.warning(
                "Answer-agent grounding search failed for tenant %s; proceeding "
                "with an ungrounded answer: %r",
                tenant_id,
                exc,
            )
            return []
        return [
            _flatten_search_hit(h)
            for h in search.get("results", [])
            if isinstance(h, dict)
        ]

    def _get_search_agent(self, profile: str):
        """Return a per-profile cached SearchAgent instance. SearchAgent is
        heavyweight (query encoder, schema loader, backend) so we avoid
        re-instantiating on every dispatch."""
        from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps

        agent = self._search_agent_cache.get(profile)
        if agent is not None:
            return agent
        with self._search_agent_cache_lock:
            # Double-check: a concurrent thread may have built it while we
            # waited on the lock — otherwise N first-touches each run the full
            # SearchAgent build (encoder init + Vespa get_system_config read).
            agent = self._search_agent_cache.get(profile)
            if agent is None:
                # SearchAgentDeps.backend_url defaults to "http://localhost" if
                # not set — that's the pod itself, never Vespa. Read the real
                # backend URL from system config, same as the capability-based
                # dispatch path above.
                system_config = self._config_manager.get_system_config()
                agent = SearchAgent(
                    deps=SearchAgentDeps(
                        profile=profile,
                        backend_url=system_config.backend_url,
                        backend_port=system_config.backend_port,
                    ),
                    schema_loader=self._schema_loader,
                    config_manager=self._config_manager,
                )
                self._search_agent_cache[profile] = agent
            return agent

    async def _code_search(self, query: str, tenant_id: str) -> List[Dict[str, Any]]:
        """Code-context search for the coding agent via ``code_lateon_mv``.

        Returns ``[]`` when code search is unavailable (encoder not registered,
        Vespa down) so coding tasks proceed without indexed context. The
        SearchService build + search (encoder inference + Vespa HTTP) run off
        the event loop.
        """
        from cogniverse_agents.search.service import SearchService
        from cogniverse_foundation.config.utils import get_config

        def _run() -> List[Dict[str, Any]]:
            cfg = get_config(tenant_id=tenant_id, config_manager=self._config_manager)
            svc = SearchService(
                config=cfg,
                config_manager=self._config_manager,
                schema_loader=self._schema_loader,
            )
            return [
                r.to_dict()
                for r in svc.search(
                    query=query,
                    profile="code_lateon_mv",
                    tenant_id=tenant_id,
                    top_k=10,
                )
            ]

        try:
            return await asyncio.to_thread(_run)
        except Exception as exc:
            logger.info("Code search unavailable, proceeding without context: %s", exc)
            return []

    async def _rewrite_query_with_history(
        self, query: str, conversation_history: List[Dict[str, str]]
    ) -> str:
        """Rewrite a query that may contain anaphoric references using conversation history.

        Uses a DSPy ChainOfThought module to resolve references like 'that',
        'those', 'more' etc. Raises on failure — callers must handle explicitly.
        """
        from cogniverse_agents.search_agent import (
            ConversationalQueryRewriteModule,
        )

        if self._query_rewriter is None:
            self._query_rewriter = ConversationalQueryRewriteModule()

        history_lines = []
        for turn in conversation_history[-5:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")[:200]
            history_lines.append(f"{role}: {content}")
        history_text = "\n".join(history_lines)

        result = await self._query_rewriter.acall(
            query=query, conversation_history=history_text
        )
        # DSPy may return None for the output field when the LM response fails
        # to parse — fall back to the original query in that case.
        rewritten = (result.rewritten_query or "").strip()
        return rewritten if rewritten else query

    def _get_rail_chains(self, tenant_id: str):
        """Build and cache the (input, output) content-rail chains per tenant.

        Loads the ``rails`` block from config (config.json / ConfigStore) and
        compiles it into RailChains once per tenant. Returns ``None`` when
        rails are absent or disabled. Rails are enforced at the gateway —
        the request front door — so internal agent-to-agent calls aren't
        gated.
        """
        if tenant_id in self._rail_chains_cache:
            return self._rail_chains_cache[tenant_id]
        with self._rail_chains_cache_lock:
            # Double-check under the lock — a concurrent to_thread first-touch
            # for the same tenant otherwise re-runs the get_config Vespa read
            # and chain compile.
            if tenant_id in self._rail_chains_cache:
                return self._rail_chains_cache[tenant_id]
            chains: Optional[tuple] = None
            try:
                from cogniverse_core.agents.rails import RailsConfig
                from cogniverse_foundation.config.utils import get_config

                rails_cfg = get_config(tenant_id, self._config_manager).get("rails", {})
                if rails_cfg:
                    rc = RailsConfig(**rails_cfg)
                    if rc.enabled:
                        chains = (rc.build_input_chain(), rc.build_output_chain())
            except Exception as exc:  # noqa: BLE001 — rails must never break dispatch
                logger.warning(
                    "Failed to build content rails for tenant %s: %s", tenant_id, exc
                )
                chains = None
            self._rail_chains_cache[tenant_id] = chains
            return self._rail_chains_cache[tenant_id]

    async def _execute_gateway_task(
        self,
        query: str,
        context: Dict[str, Any],
        tenant_id: str,
    ) -> Dict[str, Any]:
        """Route query through GatewayAgent for triage.

        Simple queries are dispatched directly to the target execution agent.
        Complex queries are forwarded to OrchestratorAgent for multi-agent
        coordination. Content rails run here (the front door): input rails on
        the incoming query, output rails on the final response.
        """
        # Consult + verify routing_agent egress policy at dispatch.
        self.consult_egress_policy("routing_agent")
        self._verify_routing_egress(tenant_id)

        from cogniverse_core.agents.rails import RailBlockedError

        # _get_rail_chains does a get_config Vespa read on its first call per
        # tenant (then caches) — offload so the gateway path doesn't stall.
        rail_chains = await asyncio.to_thread(self._get_rail_chains, tenant_id)
        if rail_chains is not None:
            try:
                rail_chains[0].check({"query": query})
            except RailBlockedError as exc:
                logger.info(
                    "Request blocked by input rail '%s': %s", exc.rail_name, exc.reason
                )
                return {
                    "status": "blocked",
                    "agent": "gateway_agent",
                    "rail": exc.rail_name,
                    "reason": exc.reason,
                    "message": f"Request blocked by {exc.rail_name}",
                }

        from cogniverse_agents.gateway_agent import GatewayInput

        gateway_agent = await self._get_or_build_gateway_agent(tenant_id)

        input_data = GatewayInput(query=query, tenant_id=tenant_id)
        result = await gateway_agent._process_impl(input_data)

        if result.complexity == "complex":
            final = await self._execute_orchestration_task(
                query,
                context,
                tenant_id,
                gateway_context={
                    "modality": result.modality,
                    "generation_type": result.generation_type,
                    "confidence": result.confidence,
                },
            )
        else:
            # Simple: route directly to the execution agent
            conversation_history = context.get("conversation_history", [])
            downstream = await self._execute_downstream_agent(
                agent_name=result.routed_to,
                query=query,
                tenant_id=tenant_id,
                top_k=context.get("top_k", 10),
                conversation_history=conversation_history,
            )
            final = {
                "status": "success",
                "agent": "gateway_agent",
                "message": f"Routed '{query[:50]}' to {result.routed_to} (simple)",
                "gateway": {
                    "complexity": result.complexity,
                    "modality": result.modality,
                    "generation_type": result.generation_type,
                    "routed_to": result.routed_to,
                    "confidence": result.confidence,
                },
                "downstream_result": downstream,
            }

        # Output rails on the final user-facing response (front-door exit).
        if rail_chains is not None and isinstance(final, dict):
            try:
                rail_chains[1].check(final)
            except RailBlockedError as exc:
                logger.info(
                    "Response blocked by output rail '%s': %s",
                    exc.rail_name,
                    exc.reason,
                )
                return {
                    "status": "blocked",
                    "agent": "gateway_agent",
                    "rail": exc.rail_name,
                    "reason": exc.reason,
                    "message": f"Response blocked by {exc.rail_name}",
                }
        return final

    async def _execute_orchestration_task(
        self,
        query: str,
        context: Dict[str, Any],
        tenant_id: str,
        gateway_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute full orchestration pipeline via OrchestratorAgent."""
        from cogniverse_agents.orchestrator_agent import OrchestratorInput

        # Cached per tenant: the agent, its WorkflowIntelligence corpus, and its
        # policy http client are built once and TTL-reloaded, not rebuilt per
        # complex query.
        agent = await self._get_or_build_orchestrator(tenant_id)
        # Apply per-request artefact overlay so OrchestratorAgent's planner DSPy
        # module honors the canary/variant decision. Task-isolated ContextVar,
        # never the shared cached instance, so concurrent requests don't bleed.
        self._apply_artefact_overlay(agent, context)

        gateway_ctx = gateway_context or {}
        # Propagate the synthesis_depth opt-in from the caller's context.
        # Three precedence levels, gateway-trust > admin override > none:
        #   1. function-arg gateway_context (set by the dispatcher's own
        #      gateway → orchestration handoff)
        #   2. context["gateway_context"]["synthesis_depth"] (HTTP callers
        #      that want to mimic the gateway-classified shape)
        #   3. context["synthesis_depth"] (plain admin / direct callers)
        nested_gateway = context.get("gateway_context") or {}
        synthesis_depth = (
            gateway_ctx.get("synthesis_depth")
            or (
                nested_gateway.get("synthesis_depth")
                if isinstance(nested_gateway, dict)
                else None
            )
            or context.get("synthesis_depth")
        )
        input_data = OrchestratorInput(
            query=query,
            tenant_id=tenant_id,
            session_id=context.get("session_id"),
            conversation_history=context.get("conversation_history"),
            modality=gateway_ctx.get("modality"),
            generation_type=gateway_ctx.get("generation_type"),
            synthesis_depth=synthesis_depth,
        )

        with self._scoped_session(agent, context.get("session_id")):
            result = await agent._process_impl(input_data)

        return {
            "status": "success",
            "agent": "orchestrator_agent",
            "message": f"Orchestrated '{query[:50]}' via A2A pipeline",
            "orchestration_result": (
                result.model_dump() if hasattr(result, "model_dump") else vars(result)
            ),
            "gateway_context": gateway_context,
        }

    async def _execute_downstream_agent(
        self,
        agent_name: str,
        query: str,
        tenant_id: str,
        top_k: int = 10,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Execute the downstream agent that the router recommended.

        Re-uses the existing _execute_*_task methods based on the agent's
        capabilities, passing conversation_history through for query rewrite.
        """
        agent = self._registry.get_agent(agent_name)
        if not agent:
            raise ValueError(
                f"Routing recommended '{agent_name}' but it is not in the registry. "
                "Register the agent under its exact name before routing to it."
            )

        capabilities = set(agent.capabilities)

        if capabilities & {"search", "video_search", "retrieval"}:
            return await self._execute_search_task(
                query,
                tenant_id,
                top_k,
                conversation_history=conversation_history,
                enrichment=None,
            )
        elif capabilities & {"image_search", "visual_analysis"}:
            return await self._execute_image_search_task(query, tenant_id, top_k)
        elif capabilities & {"audio_analysis", "transcription"}:
            return await self._execute_audio_search_task(query, tenant_id, top_k)
        elif capabilities & {"document_analysis", "pdf_processing"}:
            return await self._execute_document_search_task(query, tenant_id, top_k)
        elif capabilities & {"detailed_report"}:
            return await self._execute_detailed_report_task(query, tenant_id)
        elif capabilities & {"summarization", "text_generation"}:
            return await self._execute_summarization_task(query, tenant_id)
        elif capabilities & {"text_analysis", "sentiment", "classification"}:
            return await self._execute_text_analysis_task(
                query, {"tenant_id": tenant_id}, tenant_id
            )
        elif "coding" in capabilities:
            return await self._execute_coding_task(
                query, tenant_id, {"tenant_id": tenant_id}
            )
        else:
            raise ValueError(
                f"Routed agent '{agent_name}' has no supported execution path. "
                f"Capabilities: {agent.capabilities}"
            )

    def _get_vespa_endpoint(self, tenant_id: str) -> str:
        system_config = self._config_manager.get_system_config()
        return f"{system_config.backend_url}:{system_config.backend_port}"

    async def _execute_summarization_task(
        self,
        query: str,
        tenant_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Consult + verifysummarizer_agent egress policy at dispatch.
        self.consult_egress_policy("summarizer_agent")
        self._verify_summarizer_egress(tenant_id)

        from cogniverse_agents.summarizer_agent import (
            SummarizerAgent,
            SummarizerDeps,
            SummaryRequest,
        )

        deps = SummarizerDeps(
            tenant_id=tenant_id,
            **self._agent_behavior_kwargs(tenant_id, "summarizer_agent"),
        )
        agent = SummarizerAgent(deps=deps, config_manager=self._config_manager)
        await asyncio.to_thread(
            self._init_agent_memory, agent, "summarizer_agent", tenant_id
        )
        # Apply per-request artefact overlay to SummarizerAgent's
        # DSPy module(s) for canary/variant prompts.
        self._apply_artefact_overlay(agent, context)

        request = SummaryRequest(
            query=query,
            search_results=await self._resolve_answer_search_results(
                query, tenant_id, context, top_k=10
            ),
            summary_type="general",
        )
        result = await agent.summarize(request)

        return {
            "status": "success",
            "agent": "summarizer_agent",
            "message": f"Generated summary for '{query}'",
            "result": dataclasses.asdict(result),
        }

    async def _execute_text_analysis_task(
        self, query: str, context: Dict[str, Any], tenant_id: str
    ) -> Dict[str, Any]:
        from cogniverse_agents.text_analysis_agent import TextAnalysisAgent

        agent = TextAnalysisAgent(
            tenant_id=tenant_id, config_manager=self._config_manager
        )

        analysis_type = context.get("analysis_type", "summary")
        # analyze_text() invokes a DSPy module (blocking LLM call); run it off
        # the event loop so concurrent A2A requests aren't stalled.
        result = await asyncio.to_thread(
            agent.analyze_text, text=query, analysis_type=analysis_type
        )

        return {
            "status": "success",
            "agent": "text_analysis_agent",
            "message": f"Completed {analysis_type} analysis for '{query}'",
            "result": result,
        }

    async def _execute_detailed_report_task(
        self, query: str, tenant_id: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        from cogniverse_agents.detailed_report_agent import (
            DetailedReportAgent,
            DetailedReportDeps,
            ReportRequest,
        )

        deps = DetailedReportDeps(
            tenant_id=tenant_id,
            **self._agent_behavior_kwargs(tenant_id, "detailed_report_agent"),
        )
        agent = DetailedReportAgent(deps=deps, config_manager=self._config_manager)
        await asyncio.to_thread(
            self._init_agent_memory, agent, "detailed_report_agent", tenant_id
        )

        request = ReportRequest(
            query=query,
            search_results=await self._resolve_answer_search_results(
                query, tenant_id, context, top_k=20
            ),
            report_type="comprehensive",
        )
        result = await agent.generate_report(request)

        return {
            "status": "success",
            "agent": "detailed_report_agent",
            "message": f"Generated detailed report for '{query}'",
            "result": dataclasses.asdict(result),
        }

    async def _build_encoder_config(self):
        """Merged config (backend.profiles + inference_service_urls) the media
        agents hand to QueryEncoderFactory so query encoders route through the
        deployed sidecar. System-scoped: inference URLs are infra-level, not
        per-tenant. Built off the loop — the ConfigUtils ensure-chain reads Vespa.
        """
        from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
        from cogniverse_foundation.config.utils import get_config

        return await asyncio.to_thread(
            get_config, SYSTEM_TENANT_ID, self._config_manager
        )

    async def _execute_image_search_task(
        self, query: str, tenant_id: str, top_k: int
    ) -> Dict[str, Any]:
        from cogniverse_agents.image_search_agent import (
            ImageSearchAgent,
            ImageSearchDeps,
        )

        vespa_endpoint = self._get_vespa_endpoint(tenant_id)
        deps = ImageSearchDeps(
            vespa_endpoint=vespa_endpoint,
            tenant_id=tenant_id,
            encoder_config=await self._build_encoder_config(),
        )
        agent = ImageSearchAgent(deps=deps)

        results = await agent.search_images(query=query, limit=top_k)

        result_list = [r.model_dump() for r in results]
        return {
            "status": "success",
            "agent": "image_search_agent",
            "message": f"Found {len(result_list)} images for '{query}'",
            "results_count": len(result_list),
            "results": result_list,
        }

    async def _execute_audio_search_task(
        self, query: str, tenant_id: str, top_k: int
    ) -> Dict[str, Any]:
        from cogniverse_agents.audio_analysis_agent import (
            AudioAnalysisAgent,
            AudioAnalysisDeps,
        )

        vespa_endpoint = self._get_vespa_endpoint(tenant_id)
        sys_cfg = self._config_manager.get_system_config()
        deps = AudioAnalysisDeps(
            vespa_endpoint=vespa_endpoint,
            tenant_id=tenant_id,
            clap_endpoint=(sys_cfg.inference_service_urls or {}).get("clap_embed"),
        )
        agent = AudioAnalysisAgent(deps=deps)

        results = await agent.search_audio(query=query, limit=top_k)

        result_list = [r.model_dump() for r in results]
        return {
            "status": "success",
            "agent": "audio_analysis_agent",
            "message": f"Found {len(result_list)} audio results for '{query}'",
            "results_count": len(result_list),
            "results": result_list,
        }

    async def _execute_document_search_task(
        self, query: str, tenant_id: str, top_k: int
    ) -> Dict[str, Any]:
        from cogniverse_agents.document_agent import (
            DocumentAgent,
            DocumentAgentDeps,
        )

        vespa_endpoint = self._get_vespa_endpoint(tenant_id)
        deps = DocumentAgentDeps(
            vespa_endpoint=vespa_endpoint,
            tenant_id=tenant_id,
            encoder_config=await self._build_encoder_config(),
        )
        agent = DocumentAgent(deps=deps)
        await asyncio.to_thread(
            self._init_agent_memory, agent, "document_agent", tenant_id
        )

        results = await agent.search_documents(query=query, limit=top_k)

        result_list = [r.model_dump() for r in results]
        return {
            "status": "success",
            "agent": "document_agent",
            "message": f"Found {len(result_list)} documents for '{query}'",
            "results_count": len(result_list),
            "results": result_list,
        }

    async def _execute_deep_research_task(
        self, query: str, tenant_id: str
    ) -> Dict[str, Any]:
        from cogniverse_agents.deep_research_agent import (
            DeepResearchAgent,
            DeepResearchDeps,
            DeepResearchInput,
        )

        deps = DeepResearchDeps(tenant_id=tenant_id)

        async def search_fn(query: str, tenant_id: str):
            result = await self._execute_search_task(query, tenant_id, top_k=10)
            return result.get("results", [])

        agent = DeepResearchAgent(
            deps=deps, search_fn=search_fn, config_manager=self._config_manager
        )
        await asyncio.to_thread(
            self._init_agent_memory, agent, "deep_research_agent", tenant_id
        )

        input_data = DeepResearchInput(query=query, tenant_id=tenant_id)
        result = await agent.process(input_data)

        return {
            "status": "success",
            "agent": "deep_research_agent",
            "message": f"Research complete for '{query}'",
            "result": result.model_dump(),
        }

    async def _execute_coding_task(
        self,
        query: str,
        tenant_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import dspy

        from cogniverse_agents.coding_agent import (
            CodingAgent,
            CodingDeps,
            CodingInput,
        )
        from cogniverse_foundation.config.semantic_router import (
            create_routed_lm,
            resolve_semantic_router_config,
        )
        from cogniverse_foundation.config.utils import get_config

        # Offload the config ensure-chain (Vespa read on a cold/expired config)
        # off the API loop, as the sibling search/encoder paths do.
        config = await asyncio.to_thread(
            get_config, tenant_id=tenant_id, config_manager=self._config_manager
        )
        coding_lm = create_routed_lm(
            config.get_llm_config().resolve("coding_agent"),
            resolve_semantic_router_config(config),
            tenant_id,
        )

        deps = CodingDeps(
            tenant_id=tenant_id,
            sandbox_manager=self._sandbox_manager,
        )

        agent = CodingAgent(
            deps=deps,
            search_fn=self._code_search,
            sandbox_manager=self._sandbox_manager,
            config_manager=self._config_manager,
        )
        # Auto-init memory so the coding agent receives learned strategies
        # and tenant memories in inject_context_into_prompt.
        await asyncio.to_thread(
            self._init_agent_memory, agent, "coding_agent", tenant_id
        )

        ctx = context or {}
        input_data = CodingInput(
            task=query,
            codebase_path=ctx.get("codebase_path", ""),
            tenant_id=tenant_id,
            max_iterations=ctx.get("max_iterations", 5),
            language=ctx.get("language", "python"),
        )
        with dspy.context(lm=coding_lm):
            result = await agent.process(input_data)

        return {
            "status": "success",
            "agent": "coding_agent",
            "message": f"Coding task complete for '{query}'",
            "result": result.model_dump(),
        }
