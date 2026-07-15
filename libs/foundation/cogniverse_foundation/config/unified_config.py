"""Unified configuration schema with multi-tenant support.

Hosts ``SystemConfig`` (global deployment-level state) +
``LLMEndpointConfig`` / ``LLMConfig`` (LLM wiring) +
``RoutingConfigUnified`` (query-routing knobs) +
``AgentConfig`` / ``TelemetryConfig`` re-exports.
"""

import copy
import logging
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, Dict, List, Optional

from cogniverse_foundation.config.agent_config import (
    AgentConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMEndpointConfig:
    """
    Configuration for a single LLM endpoint.

    Provider prefix is always explicit in model string (e.g., "openai/gpt-4o",
    "openai/gpt-4o", "anthropic/claude-3-5-sonnet-20241022"). No auto-detection.
    Matches DSPy/LiteLLM convention.

    api_key=None means no key needed (e.g., local OAI-compat servers).
    """

    model: str
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 1000
    # Records which LoRA/fine-tuned artifact this endpoint corresponds to.
    # Bookkeeping only: LM construction never reads it — vLLM serves adapters
    # server-side, selected via the model name.
    adapter_path: Optional[str] = None
    extra_body: Optional[Dict[str, Any]] = None  # Provider-specific request params
    # Static HTTP headers attached to every request sent to ``api_base``.
    # Forwarded to dspy.LM/litellm as ``extra_headers``. Used to pass
    # per-endpoint routing metadata to an OpenAI-compatible semantic router sitting
    # in front of the backend (e.g. a semantic router keying on a tenant
    # tier via ``x-authz-user-groups`` or a task label via ``x-vsr-task``).
    # The dict is sent verbatim; the factory does not interpret it.
    extra_headers: Optional[Dict[str, str]] = None
    # vLLM sampling seed. When set, gets forwarded into extra_body so vLLM's
    # OpenAI-compat layer honors it on each request. With ``temperature=0``
    # + a fixed ``seed``, the LM produces bit-stable output across runs
    # against the same vLLM deployment (modulo vLLM batching state — see
    # the test docs for caveats). Default None preserves vanilla behavior.
    seed: Optional[int] = None
    # Fail fast on a down endpoint instead of litellm's ~600s x dspy retries.
    request_timeout: float = 120.0
    num_retries: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values for clean serialization."""
        result: Dict[str, Any] = {"model": self.model}
        if self.api_base is not None:
            result["api_base"] = self.api_base
        if self.api_key is not None:
            result["api_key"] = "***"  # Never serialize real keys
        result["temperature"] = self.temperature
        result["max_tokens"] = self.max_tokens
        if self.adapter_path is not None:
            result["adapter_path"] = self.adapter_path
        if self.extra_body is not None:
            result["extra_body"] = self.extra_body
        if self.extra_headers is not None:
            result["extra_headers"] = self.extra_headers
        if self.seed is not None:
            result["seed"] = self.seed
        result["request_timeout"] = self.request_timeout
        result["num_retries"] = self.num_retries
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMEndpointConfig":
        """Create from dictionary."""
        return cls(
            model=data["model"],
            api_base=data.get("api_base"),
            api_key=data.get("api_key"),
            temperature=data.get("temperature", 0.1),
            max_tokens=data.get("max_tokens", 1000),
            adapter_path=data.get("adapter_path"),
            extra_body=data.get("extra_body"),
            extra_headers=data.get("extra_headers"),
            seed=data.get("seed"),
            request_timeout=data.get("request_timeout", 120.0),
            num_retries=data.get("num_retries", 1),
        )


@dataclass
class LLMConfig:
    """
    Centralized LLM configuration with override resolution.

    - primary: Global default for ALL DSPy modules, agents, optimizers.
      Also serves as the student model during optimization.
    - teacher: Bootstrap-teacher endpoint for DSPy optimization —
      resolve_teacher() feeds BootstrapFewShot(teacher_settings={"lm": ...}).
    - overrides: Per-component partial dicts merged with primary.
      None = use primary unchanged. Only differing fields need to be specified.
    """

    primary: LLMEndpointConfig
    teacher: LLMEndpointConfig
    overrides: Dict[str, Optional[Dict[str, Any]]] = field(default_factory=dict)

    def resolve(self, component: str) -> LLMEndpointConfig:
        """
        Resolve the LLM config for a specific component.

        If the component has an override, merge it onto primary defaults
        (override fields take precedence). If no override or override is None,
        return a copy of primary.

        Args:
            component: Component name (e.g., "gateway_agent", "summarizer_agent")

        Returns:
            Fully resolved LLMEndpointConfig
        """
        override = self.overrides.get(component)
        resolved = copy.deepcopy(self.primary)
        if override is None:
            return resolved

        # Merge on the dataclass, never through to_dict() — it masks api_key
        # to "***", which must never leak into a resolved endpoint.
        valid_fields = {f.name for f in fields(resolved)}
        for key, value in override.items():
            if key in valid_fields:
                setattr(resolved, key, value)
        return resolved

    def resolve_teacher(self) -> LLMEndpointConfig:
        """Resolve the teacher endpoint for DSPy optimization.

        Mirrors ``resolve()``'s no-override path: an isolated copy with the
        real api_key preserved. The teacher drives bootstrap demo generation
        (``BootstrapFewShot(teacher_settings={"lm": ...})``) and annotation.
        """
        return copy.deepcopy(self.teacher)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        overrides_dict: Dict[str, Any] = {}
        for key, val in self.overrides.items():
            overrides_dict[key] = val if val is not None else None
        return {
            "primary": self.primary.to_dict(),
            "teacher": self.teacher.to_dict(),
            "overrides": overrides_dict,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        """Create from dictionary."""
        primary = LLMEndpointConfig.from_dict(data["primary"])
        teacher = LLMEndpointConfig.from_dict(data["teacher"])

        # Store overrides as raw dicts — they are partial and may lack "model"
        overrides: Dict[str, Optional[Dict[str, Any]]] = {}
        for key, val in data.get("overrides", {}).items():
            overrides[key] = val

        return cls(primary=primary, teacher=teacher, overrides=overrides)


@dataclass
class TenantConfig:
    """Tenant-specific configuration wrapper"""

    tenant_id: str
    tenant_name: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticRouterConfig:
    """Opt-in routing of LLM calls through the vLLM Semantic Router.

    When ``enabled``, LLM endpoint configs are rewritten to target
    ``semantic_router_url`` (the Envoy front-end for the semantic router)
    instead of the model backend, and two authz headers are attached per
    request: the tenant identity (``user_id_header``, the ``tenant_id``) and
    the tenant tier (``tier_header``, resolved from ``tenant_tiers`` with
    ``default_tier`` as fallback). The router's authz signal requires the
    identity header and refuses to evaluate role bindings without it (no silent
    bypass), then gates the tenant's allowed model set by tier and classifies
    the request content itself (domain/complexity) to pick the model +
    reasoning mode. The application helper lives in
    ``cogniverse_foundation.config.semantic_router``.

    Disabled by default: with ``enabled=False`` the endpoint config is
    passed through untouched, so the direct-to-backend path is unchanged.
    """

    enabled: bool = False
    semantic_router_url: str = ""
    tenant_tiers: Dict[str, str] = field(default_factory=dict)
    default_tier: str = "default"
    tier_header: str = "x-authz-user-groups"
    user_id_header: str = "x-authz-user-id"
    # Model name sent on routed requests (litellm provider prefix + the
    # router's auto alias). The router resolves models by its own catalog
    # names and rejects raw provider model ids, so the endpoint's model is
    # replaced, not forwarded.
    routed_model: str = "openai/auto"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "semantic_router_url": self.semantic_router_url,
            "tenant_tiers": dict(self.tenant_tiers),
            "default_tier": self.default_tier,
            "tier_header": self.tier_header,
            "user_id_header": self.user_id_header,
            "routed_model": self.routed_model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticRouterConfig":
        return cls(
            enabled=bool(data.get("enabled", False)),
            semantic_router_url=data.get("semantic_router_url", ""),
            tenant_tiers=dict(data.get("tenant_tiers") or {}),
            default_tier=data.get("default_tier", "default"),
            tier_header=data.get("tier_header", "x-authz-user-groups"),
            user_id_header=data.get("user_id_header", "x-authz-user-id"),
            routed_model=data.get("routed_model", "openai/auto"),
        )


@dataclass
class SystemConfig:
    """System-level infrastructure configuration.

    This is global — NOT per-tenant. There is one SystemConfig for the
    entire deployment. Tenants don't decide where Vespa or Phoenix run.
    """

    # Agent service URLs
    video_agent_url: str = "http://localhost:8002"
    summarizer_agent_url: str = "http://localhost:8004"

    # API service URLs
    ingestion_api_url: str = "http://localhost:8000"

    # Search backend configuration
    search_backend: str = "vespa"
    backend_url: str = "http://localhost"
    backend_port: int = 8080
    application_name: str = "cogniverse"  # Vespa application package name

    # LLM configuration
    # ``llm_model`` is the bare model id (e.g. "google/gemma-4-e4b-it" or
    # "Qwen/Qwen2.5-7B-Instruct"); ``llm_engine`` picks the DSPy/litellm
    # prefix at call time via ``cogniverse_foundation.llm.dspy_format``.
    # Both are populated from chart env (LLM_MODEL / LLM_ENGINE).
    llm_model: str = "google/gemma-4-e4b-it"
    llm_engine: str = "vllm"
    base_url: str = "http://localhost:8101/v1"
    llm_api_key: Optional[str] = None

    # Opt-in routing of LLM calls through an OpenAI-compatible semantic router
    # (e.g. an Envoy front-end for a semantic router). Disabled by
    # default — see SemanticRouterConfig.
    semantic_router: SemanticRouterConfig = field(default_factory=SemanticRouterConfig)

    # Phoenix/Telemetry
    telemetry_url: str = "http://localhost:6006"
    telemetry_collector_endpoint: str = "localhost:4317"

    # Video processing
    video_processing_profiles: List[str] = field(default_factory=list)

    # Agent Registry - structured config for all agents
    agents: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Agent Registry URL for Curated Registry pattern (A2A discovery)
    agent_registry_url: str = "http://localhost:8000"

    # ColPali stays on its own URL for now — distinct HTTP contract from
    # ColBERT-family services (image processing vs text pooling).
    colpali_inference_url: str = ""

    # Map of inference-service logical name -> URL. Keys match the Helm
    # ``inference.*`` block names (e.g., "vllm_colpali", "colbert_pylate",
    # "vllm_llm_student", "vllm_asr"). Profiles pick services via the
    # per-role ``inference_services`` map (``{"embedding": "...",
    # "transcription": "..."}``); the runtime resolves each role's URL
    # from this flat map. Populated from the JSON in the
    # ``INFERENCE_SERVICE_URLS`` env var at startup.
    inference_service_urls: Dict[str, str] = field(default_factory=dict)

    # Orchestrator iterative-retrieval-loop tuning. Bounded by these
    # three caps; the loop exits with ``exit_reason="max_iter"``,
    # ``"token_budget"``, or ``"wall_clock"`` when the corresponding
    # cap is hit. Populated from chart env (ITER_RETRIEVAL_MAX_ITER,
    # ITER_RETRIEVAL_TOKEN_BUDGET, ITER_RETRIEVAL_WALL_CLOCK_MS) so
    # ops can re-tune without a code change.
    iter_retrieval_max_iter: int = 3
    iter_retrieval_token_budget: int = 8000
    iter_retrieval_wall_clock_ms: int = 30000

    # Redis URL for cross-pod inbound-messaging routing + durability.
    # Empty string means "use the in-pod InboundQueueRegistry"; a real
    # URL ("redis://host:6379/0") activates the Redis-backed registry.
    # Populated from chart env REDIS_URL at the runtime startup
    # boundary.
    redis_url: str = ""

    # Local cache directory the finetuning adapter resolver downloads
    # to. Empty string means "no cache configured" — production code
    # paths that call ``resolve_adapter_path`` will raise rather than
    # silently fall back to ``/tmp/...``. Populated from chart env
    # COGNIVERSE_ADAPTER_CACHE at the runtime startup boundary.
    adapter_cache_dir: str = ""

    # MinIO object-store endpoint for the ingestion upload path
    # (``POST /ingestion/upload`` queues binary uploads to MinIO,
    # then the ingestion worker reads them back to feed Vespa).
    # Empty string means MinIO isn't configured and the upload route
    # responds with 503. Populated from chart env MINIO_ENDPOINT at
    # the runtime startup boundary.
    minio_endpoint: str = ""

    # Metadata
    environment: str = "development"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "video_agent_url": self.video_agent_url,
            "summarizer_agent_url": self.summarizer_agent_url,
            "ingestion_api_url": self.ingestion_api_url,
            "search_backend": self.search_backend,
            "backend_url": self.backend_url,
            "backend_port": self.backend_port,
            "application_name": self.application_name,
            "llm_model": self.llm_model,
            "llm_engine": self.llm_engine,
            "base_url": self.base_url,
            "llm_api_key": "***" if self.llm_api_key else None,
            "semantic_router": self.semantic_router.to_dict(),
            "telemetry_url": self.telemetry_url,
            "telemetry_collector_endpoint": self.telemetry_collector_endpoint,
            "video_processing_profiles": self.video_processing_profiles,
            "agents": self.agents,
            "agent_registry_url": self.agent_registry_url,
            "colpali_inference_url": self.colpali_inference_url,
            "inference_service_urls": dict(self.inference_service_urls),
            "iter_retrieval_max_iter": self.iter_retrieval_max_iter,
            "iter_retrieval_token_budget": self.iter_retrieval_token_budget,
            "iter_retrieval_wall_clock_ms": self.iter_retrieval_wall_clock_ms,
            "redis_url": self.redis_url,
            "adapter_cache_dir": self.adapter_cache_dir,
            "minio_endpoint": self.minio_endpoint,
            "environment": self.environment,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemConfig":
        """Create from dictionary"""
        return cls(
            video_agent_url=data.get("video_agent_url", "http://localhost:8002"),
            summarizer_agent_url=data.get(
                "summarizer_agent_url", "http://localhost:8004"
            ),
            ingestion_api_url=data.get("ingestion_api_url", "http://localhost:8000"),
            search_backend=data.get("search_backend", "vespa"),
            backend_url=data.get("backend_url", "http://localhost"),
            backend_port=data.get("backend_port", 8080),
            application_name=data.get("application_name", "cogniverse"),
            llm_model=data.get("llm_model", "google/gemma-4-e4b-it"),
            llm_engine=data.get("llm_engine", "vllm"),
            base_url=data.get("base_url", "http://localhost:8101/v1"),
            llm_api_key=data.get("llm_api_key"),
            semantic_router=SemanticRouterConfig.from_dict(
                data.get("semantic_router") or {}
            ),
            telemetry_url=data.get("telemetry_url", "http://localhost:6006"),
            telemetry_collector_endpoint=data.get(
                "telemetry_collector_endpoint", "localhost:4317"
            ),
            video_processing_profiles=data.get("video_processing_profiles", []),
            agents=data.get("agents", {}),
            agent_registry_url=data.get("agent_registry_url", "http://localhost:8000"),
            colpali_inference_url=data.get("colpali_inference_url", ""),
            inference_service_urls=dict(data.get("inference_service_urls") or {}),
            iter_retrieval_max_iter=int(data.get("iter_retrieval_max_iter", 3)),
            iter_retrieval_token_budget=int(
                data.get("iter_retrieval_token_budget", 8000)
            ),
            iter_retrieval_wall_clock_ms=int(
                data.get("iter_retrieval_wall_clock_ms", 30000)
            ),
            redis_url=data.get("redis_url", ""),
            adapter_cache_dir=data.get("adapter_cache_dir", ""),
            minio_endpoint=data.get("minio_endpoint", ""),
            environment=data.get("environment", "development"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RoutingConfigUnified:
    """Unified routing configuration with tenant support.

    ``tenant_id`` is required — constructors/parsers that omit it will
    raise ``ValueError`` via ``__post_init__``. The default value is
    ``None`` only to satisfy the dataclass ordering rule (required
    fields after defaulted fields is a syntax error); the runtime
    check enforces the real contract.
    """

    tenant_id: Optional[str] = None

    # Routing mode. Only "tiered" is implemented; other values are accepted by
    # the schema for forward-compat but produce no dispatch-time behavior change.
    routing_mode: str = "tiered"

    # Seeded into GatewayDeps by the dispatcher; when False the gateway routes
    # every query through orchestration (skips the fast path).
    enable_fast_path: bool = True
    # Default matches GatewayDeps.fast_path_confidence_threshold so seeding an
    # untouched tenant's config into the gateway is a no-op.
    fast_path_confidence_threshold: float = 0.4

    # GLiNER configuration — seeded into GatewayDeps by the dispatcher.
    gliner_model: str = "urchade/gliner_large-v2.1"
    gliner_threshold: float = 0.3
    gliner_device: str = "cpu"

    # Optimization settings — consumed by scripts/auto_optimization_trigger.py
    enable_auto_optimization: bool = True
    optimization_interval_seconds: int = 3600
    min_samples_for_optimization: int = 100

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from cogniverse_foundation.common.tenant_utils import require_tenant_id

        require_tenant_id(self.tenant_id, source="RoutingConfigUnified")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tenant_id": self.tenant_id,
            "routing_mode": self.routing_mode,
            "enable_fast_path": self.enable_fast_path,
            "fast_path_confidence_threshold": self.fast_path_confidence_threshold,
            "gliner_model": self.gliner_model,
            "gliner_threshold": self.gliner_threshold,
            "gliner_device": self.gliner_device,
            "enable_auto_optimization": self.enable_auto_optimization,
            "optimization_interval_seconds": self.optimization_interval_seconds,
            "min_samples_for_optimization": self.min_samples_for_optimization,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingConfigUnified":
        """Create from dictionary. Raises if tenant_id is absent."""
        from cogniverse_foundation.common.tenant_utils import require_tenant_id

        tenant_id = require_tenant_id(
            data.get("tenant_id"), source="RoutingConfigUnified.from_dict"
        )
        return cls(
            tenant_id=tenant_id,
            routing_mode=data.get("routing_mode", "tiered"),
            enable_fast_path=data.get("enable_fast_path", True),
            fast_path_confidence_threshold=data.get(
                "fast_path_confidence_threshold", 0.4
            ),
            gliner_model=data.get("gliner_model", "urchade/gliner_large-v2.1"),
            gliner_threshold=data.get("gliner_threshold", 0.3),
            gliner_device=data.get("gliner_device", "cpu"),
            enable_auto_optimization=data.get("enable_auto_optimization", True),
            optimization_interval_seconds=data.get(
                "optimization_interval_seconds", 3600
            ),
            min_samples_for_optimization=data.get("min_samples_for_optimization", 100),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DurableExecutionConfig:
    """Per-tenant durable-execution enablement for long-running workflows.

    Gates whether the optimization / auto-eval jobs checkpoint their progress
    so a killed Argo pod resumes from the last completed stage instead of
    re-running expensive compiles. Default off.
    """

    tenant_id: str = ""
    enabled: bool = False

    def to_dict(self) -> dict:
        return {"tenant_id": self.tenant_id, "enabled": self.enabled}

    @classmethod
    def from_dict(cls, data: dict) -> "DurableExecutionConfig":
        return cls(
            tenant_id=data.get("tenant_id", ""),
            enabled=bool(data.get("enabled", False)),
        )


@dataclass
class AgentConfigUnified:
    """
    Agent configuration with tenant support.
    Wraps existing AgentConfig with tenant_id.
    """

    tenant_id: str
    agent_config: AgentConfig

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        config_dict = self.agent_config.to_dict()
        config_dict["tenant_id"] = self.tenant_id
        return config_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfigUnified":
        """Create from dictionary. Raises if tenant_id is absent."""
        from cogniverse_foundation.common.tenant_utils import require_tenant_id

        data = dict(data)  # don't mutate caller's dict
        tenant_id = require_tenant_id(
            data.pop("tenant_id", None), source="AgentConfigUnified.from_dict"
        )
        agent_config = AgentConfig.from_dict(data)
        return cls(tenant_id=tenant_id, agent_config=agent_config)


@dataclass
class BackendProfileConfig:
    """
    Backend profile configuration for video processing.

    Represents a single processing profile with embedding model,
    pipeline configuration, and processing strategies.
    """

    profile_name: str
    type: str = "video"
    description: str = ""
    schema_name: str = ""
    embedding_model: str = ""
    pipeline_config: Dict[str, Any] = field(default_factory=dict)
    strategies: Dict[str, Any] = field(default_factory=dict)
    embedding_type: str = ""
    schema_config: Dict[str, Any] = field(default_factory=dict)
    model_specific: Dict[str, Any] = field(default_factory=dict)
    process_type: Optional[str] = None
    model_loader: str = ""
    extra_config: Dict[str, Any] = field(default_factory=dict)

    # Keys handled by named fields (everything else goes into extra_config)
    _KNOWN_KEYS = frozenset(
        {
            "type",
            "description",
            "schema_name",
            "embedding_model",
            "pipeline_config",
            "strategies",
            "embedding_type",
            "schema_config",
            "model_specific",
            "process_type",
            "model_loader",
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, preserving extra fields like semantic_model."""
        result = {
            "type": self.type,
            "description": self.description,
            "schema_name": self.schema_name,
            "embedding_model": self.embedding_model,
            "pipeline_config": self.pipeline_config,
            "strategies": self.strategies,
            "embedding_type": self.embedding_type,
            "schema_config": self.schema_config,
        }
        if self.model_specific:
            result["model_specific"] = self.model_specific
        if self.process_type:
            result["process_type"] = self.process_type
        if self.model_loader:
            result["model_loader"] = self.model_loader
        # Preserve extra fields (semantic_model, etc.) that downstream
        # consumers (EmbeddingGeneratorImpl) need from the profile config
        result.update(self.extra_config)
        return result

    @classmethod
    def from_dict(
        cls, profile_name: str, data: Dict[str, Any]
    ) -> "BackendProfileConfig":
        """Create from dictionary, capturing unknown keys in extra_config."""
        extra = {k: v for k, v in data.items() if k not in cls._KNOWN_KEYS}
        return cls(
            profile_name=profile_name,
            type=data.get("type", "video"),
            description=data.get("description", ""),
            schema_name=data.get("schema_name", ""),
            embedding_model=data.get("embedding_model", ""),
            pipeline_config=data.get("pipeline_config", {}),
            strategies=data.get("strategies", {}),
            embedding_type=data.get("embedding_type", ""),
            schema_config=data.get("schema_config", {}),
            model_specific=data.get("model_specific", {}),
            process_type=data.get("process_type"),
            model_loader=data.get("model_loader", ""),
            extra_config=extra,
        )


@dataclass
class BackendConfig:
    """Backend configuration with multi-tenant profile support.

    ``tenant_id`` is required; the ``None`` default is only a
    dataclass-ordering placeholder. ``__post_init__`` raises via
    ``require_tenant_id`` when callers omit it.
    """

    tenant_id: Optional[str] = None
    backend_type: str = "vespa"
    url: str = "http://localhost"
    port: int = 8080
    profiles: Dict[str, BackendProfileConfig] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from cogniverse_foundation.common.tenant_utils import require_tenant_id

        require_tenant_id(self.tenant_id, source="BackendConfig")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tenant_id": self.tenant_id,
            "type": self.backend_type,
            "url": self.url,
            "port": self.port,
            "profiles": {
                name: profile.to_dict() for name, profile in self.profiles.items()
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackendConfig":
        """Create from dictionary. Raises if tenant_id is absent."""
        from cogniverse_foundation.common.tenant_utils import require_tenant_id

        tenant_id = require_tenant_id(
            data.get("tenant_id"), source="BackendConfig.from_dict"
        )
        profiles_data = data.get("profiles", {})
        profiles = {
            name: BackendProfileConfig.from_dict(name, profile_data)
            for name, profile_data in profiles_data.items()
        }

        return cls(
            tenant_id=tenant_id,
            backend_type=data.get("type", "vespa"),
            url=data.get("url", "http://localhost"),
            port=data.get("port", 8080),
            profiles=profiles,
            metadata=data.get("metadata", {}),
        )

    def get_profile(self, profile_name: str) -> Optional[BackendProfileConfig]:
        """Get a specific profile by name"""
        return self.profiles.get(profile_name)

    def add_profile(self, profile: BackendProfileConfig) -> None:
        """Add or update a profile"""
        self.profiles[profile.profile_name] = profile

    def merge_profile(
        self, profile_name: str, overrides: Dict[str, Any]
    ) -> BackendProfileConfig:
        """
        Merge overrides into an existing profile.

        Supports partial updates - only specified fields are overridden.
        Useful for tenant-specific tweaks to system profiles.

        Args:
            profile_name: Name of base profile to merge with
            overrides: Dictionary of fields to override

        Returns:
            New BackendProfileConfig with merged values

        Raises:
            ValueError: If base profile doesn't exist
        """
        base_profile = self.profiles.get(profile_name)
        if not base_profile:
            raise ValueError(f"Base profile '{profile_name}' not found")

        # Deep copy base profile dict and merge overrides
        merged = base_profile.to_dict()
        self._deep_merge(merged, overrides)

        return BackendProfileConfig.from_dict(profile_name, merged)

    @staticmethod
    def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> None:
        """
        Deep merge overrides into base dict (in-place).

        Recursively merges nested dictionaries.
        """
        for key, value in overrides.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                BackendConfig._deep_merge(base[key], value)
            else:
                base[key] = value


@dataclass
class FieldMappingConfig:
    """
    Maps semantic field roles to actual backend field names.

    Allows synthetic data generator to work with any backend schema
    by defining which fields contain topics, descriptions, transcripts, etc.
    """

    topic_fields: List[str] = field(default_factory=lambda: ["video_title", "title"])
    description_fields: List[str] = field(
        default_factory=lambda: ["segment_description", "description"]
    )
    transcript_fields: List[str] = field(
        default_factory=lambda: ["audio_transcript", "transcript"]
    )
    entity_fields: List[str] = field(
        default_factory=lambda: ["video_title", "segment_description"]
    )
    temporal_fields: Dict[str, str] = field(
        default_factory=lambda: {"start": "start_time", "end": "end_time"}
    )
    metadata_fields: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "topic_fields": self.topic_fields,
            "description_fields": self.description_fields,
            "transcript_fields": self.transcript_fields,
            "entity_fields": self.entity_fields,
            "temporal_fields": self.temporal_fields,
            "metadata_fields": self.metadata_fields,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldMappingConfig":
        """Create from dictionary"""
        return cls(
            topic_fields=data.get("topic_fields", ["video_title", "title"]),
            description_fields=data.get(
                "description_fields", ["segment_description", "description"]
            ),
            transcript_fields=data.get(
                "transcript_fields", ["audio_transcript", "transcript"]
            ),
            entity_fields=data.get(
                "entity_fields", ["video_title", "segment_description"]
            ),
            temporal_fields=data.get(
                "temporal_fields", {"start": "start_time", "end": "end_time"}
            ),
            metadata_fields=data.get("metadata_fields", {}),
        )


@dataclass
class DSPyModuleConfig:
    """
    Configuration for DSPy-based query generation.

    Instead of hardcoded templates, uses DSPy signatures that can be optimized.
    """

    signature_class: str  # Fully qualified class name
    module_type: str = "ChainOfThought"  # DSPy module type
    lm_config: Dict[str, Any] = field(default_factory=dict)  # LLM settings
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "signature_class": self.signature_class,
            "module_type": self.module_type,
            "lm_config": self.lm_config,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DSPyModuleConfig":
        """Create from dictionary"""
        return cls(
            signature_class=data["signature_class"],
            module_type=data.get("module_type", "ChainOfThought"),
            lm_config=data.get("lm_config", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentMappingRule:
    """
    Rule for mapping modality/query type to agent.

    Replaces hardcoded agent selection logic with configuration.
    """

    modality: str
    agent_name: str
    confidence_threshold: float = 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "modality": self.modality,
            "agent_name": self.agent_name,
            "confidence_threshold": self.confidence_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMappingRule":
        """Create from dictionary"""
        return cls(
            modality=data["modality"],
            agent_name=data["agent_name"],
            confidence_threshold=data.get("confidence_threshold", 0.7),
        )


@dataclass
class ProfileScoringRule:
    """
    Scoring rule for profile selection.

    Defines conditions and score adjustments for selecting backend profiles
    during synthetic data generation.
    """

    condition: Dict[str, Any]
    score_adjustment: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "condition": self.condition,
            "score_adjustment": self.score_adjustment,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProfileScoringRule":
        """Create from dictionary"""
        return cls(
            condition=data["condition"],
            score_adjustment=data["score_adjustment"],
            reason=data["reason"],
        )


@dataclass
class OptimizerGenerationConfig:
    """
    Configuration for generating synthetic data for a specific optimizer module.

    Each optimizer (modality, cross_modal, routing, workflow, unified) can have
    its own DSPy modules for query generation, profile scoring rules, and agent mappings.
    """

    optimizer_type: str
    dspy_modules: Dict[str, DSPyModuleConfig] = field(
        default_factory=dict
    )  # e.g., {"query_generator": DSPyModuleConfig(...)}
    profile_scoring_rules: List[ProfileScoringRule] = field(default_factory=list)
    agent_mappings: List[AgentMappingRule] = field(default_factory=list)
    num_examples_target: int = 50
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "optimizer_type": self.optimizer_type,
            "dspy_modules": {
                key: module.to_dict() for key, module in self.dspy_modules.items()
            },
            "profile_scoring_rules": [
                rule.to_dict() for rule in self.profile_scoring_rules
            ],
            "agent_mappings": [mapping.to_dict() for mapping in self.agent_mappings],
            "num_examples_target": self.num_examples_target,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizerGenerationConfig":
        """Create from dictionary"""
        dspy_modules = {}
        for key, module_data in data.get("dspy_modules", {}).items():
            dspy_modules[key] = DSPyModuleConfig.from_dict(module_data)

        profile_scoring_rules = [
            ProfileScoringRule.from_dict(rule_data)
            for rule_data in data.get("profile_scoring_rules", [])
        ]

        agent_mappings = [
            AgentMappingRule.from_dict(mapping_data)
            for mapping_data in data.get("agent_mappings", [])
        ]

        return cls(
            optimizer_type=data["optimizer_type"],
            dspy_modules=dspy_modules,
            profile_scoring_rules=profile_scoring_rules,
            agent_mappings=agent_mappings,
            num_examples_target=data.get("num_examples_target", 50),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ApprovalConfig:
    """
    Configuration for human-in-the-loop approval system.

    Controls how AI-generated outputs are reviewed and approved before use.
    Supports confidence-based auto-approval and manual review workflows.
    """

    enabled: bool = False
    confidence_threshold: float = 0.85
    storage_backend: str = "phoenix"  # phoenix, database, file
    phoenix_project_name: str = "approval_system"
    max_regeneration_attempts: int = 2
    reviewer_email: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "enabled": self.enabled,
            "confidence_threshold": self.confidence_threshold,
            "storage_backend": self.storage_backend,
            "phoenix_project_name": self.phoenix_project_name,
            "max_regeneration_attempts": self.max_regeneration_attempts,
            "reviewer_email": self.reviewer_email,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApprovalConfig":
        """Create from dictionary"""
        return cls(
            enabled=data.get("enabled", False),
            confidence_threshold=data.get("confidence_threshold", 0.85),
            storage_backend=data.get("storage_backend", "phoenix"),
            phoenix_project_name=data.get("phoenix_project_name", "approval_system"),
            max_regeneration_attempts=data.get("max_regeneration_attempts", 2),
            reviewer_email=data.get("reviewer_email"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SyntheticGeneratorConfig:
    """
    Main configuration for synthetic data generation system.

    Provides configuration-driven approach to:
    - Field mapping (which fields contain topics, descriptions, etc.)
    - Query template generation per optimizer type
    - Agent mapping rules
    - Profile selection scoring
    - Sampling configuration

    ``tenant_id`` is required; the ``None`` default is only a
    dataclass-ordering placeholder. ``__post_init__`` raises via
    ``require_tenant_id`` when callers omit it.
    """

    tenant_id: Optional[str] = None
    field_mappings: FieldMappingConfig = field(default_factory=FieldMappingConfig)
    optimizer_configs: Dict[str, OptimizerGenerationConfig] = field(
        default_factory=dict
    )
    sampling_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from cogniverse_foundation.common.tenant_utils import require_tenant_id

        require_tenant_id(self.tenant_id, source="SyntheticGeneratorConfig")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tenant_id": self.tenant_id,
            "field_mappings": self.field_mappings.to_dict(),
            "optimizer_configs": {
                key: config.to_dict() for key, config in self.optimizer_configs.items()
            },
            "sampling_config": self.sampling_config,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyntheticGeneratorConfig":
        """Create from dictionary. Raises if tenant_id is absent."""
        from cogniverse_foundation.common.tenant_utils import require_tenant_id

        tenant_id = require_tenant_id(
            data.get("tenant_id"), source="SyntheticGeneratorConfig.from_dict"
        )
        field_mappings = FieldMappingConfig.from_dict(data.get("field_mappings", {}))

        optimizer_configs = {}
        for key, config_data in data.get("optimizer_configs", {}).items():
            optimizer_configs[key] = OptimizerGenerationConfig.from_dict(config_data)

        return cls(
            tenant_id=tenant_id,
            field_mappings=field_mappings,
            optimizer_configs=optimizer_configs,
            sampling_config=data.get("sampling_config", {}),
            metadata=data.get("metadata", {}),
        )

    def get_optimizer_config(
        self, optimizer_type: str
    ) -> Optional[OptimizerGenerationConfig]:
        """Get configuration for a specific optimizer"""
        return self.optimizer_configs.get(optimizer_type)
