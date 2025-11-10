"""
Unified configuration schema with multi-tenant support.
Consolidates AgentConfig, RoutingConfig, TelemetryConfig into single system.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from cogniverse_core.config.agent_config import (
    AgentConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class TenantConfig:
    """Tenant-specific configuration wrapper"""

    tenant_id: str
    tenant_name: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemConfig:
    """System-level configuration (agent URLs, backends, etc.)"""

    tenant_id: str = "default"

    # Agent service URLs
    routing_agent_url: str = "http://localhost:8001"
    video_agent_url: str = "http://localhost:8002"
    text_agent_url: str = "http://localhost:8003"
    summarizer_agent_url: str = "http://localhost:8004"
    text_analysis_agent_url: str = "http://localhost:8005"

    # API service URLs
    ingestion_api_url: str = "http://localhost:8000"

    # Search backend configuration
    search_backend: str = "vespa"
    backend_url: str = "http://localhost"
    backend_port: int = 8080
    application_name: str = "cogniverse"  # Vespa application package name
    elasticsearch_url: Optional[str] = None

    # LLM configuration
    llm_model: str = "ollama/gemma3:4b"
    base_url: str = "http://localhost:11434"
    llm_api_key: Optional[str] = None

    # Phoenix/Telemetry
    phoenix_url: str = "http://localhost:6006"
    phoenix_collector_endpoint: str = "localhost:4317"

    # Video processing
    video_processing_profiles: List[str] = field(default_factory=list)

    # Metadata
    environment: str = "development"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tenant_id": self.tenant_id,
            "routing_agent_url": self.routing_agent_url,
            "video_agent_url": self.video_agent_url,
            "text_agent_url": self.text_agent_url,
            "summarizer_agent_url": self.summarizer_agent_url,
            "text_analysis_agent_url": self.text_analysis_agent_url,
            "ingestion_api_url": self.ingestion_api_url,
            "search_backend": self.search_backend,
            "backend_url": self.backend_url,
            "backend_port": self.backend_port,
            "application_name": self.application_name,
            "elasticsearch_url": self.elasticsearch_url,
            "llm_model": self.llm_model,
            "base_url": self.base_url,
            "llm_api_key": "***" if self.llm_api_key else None,
            "phoenix_url": self.phoenix_url,
            "phoenix_collector_endpoint": self.phoenix_collector_endpoint,
            "video_processing_profiles": self.video_processing_profiles,
            "environment": self.environment,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemConfig":
        """Create from dictionary"""
        return cls(
            tenant_id=data.get("tenant_id", "default"),
            routing_agent_url=data.get("routing_agent_url", "http://localhost:8001"),
            video_agent_url=data.get("video_agent_url", "http://localhost:8002"),
            text_agent_url=data.get("text_agent_url", "http://localhost:8003"),
            summarizer_agent_url=data.get(
                "summarizer_agent_url", "http://localhost:8004"
            ),
            text_analysis_agent_url=data.get(
                "text_analysis_agent_url", "http://localhost:8005"
            ),
            ingestion_api_url=data.get("ingestion_api_url", "http://localhost:8000"),
            search_backend=data.get("search_backend", "vespa"),
            backend_url=data.get("backend_url") or data.get("backend", {}).get("url"),
            backend_port=data.get("backend_port") or data.get("backend", {}).get("port"),
            elasticsearch_url=data.get("elasticsearch_url"),
            llm_model=data.get("llm_model", "gpt-4"),
            base_url=data.get("base_url", "http://localhost:11434"),
            llm_api_key=data.get("llm_api_key"),
            phoenix_url=data.get("phoenix_url", "http://localhost:6006"),
            phoenix_collector_endpoint=data.get(
                "phoenix_collector_endpoint", "localhost:4317"
            ),
            video_processing_profiles=data.get("video_processing_profiles", []),
            environment=data.get("environment", "development"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class RoutingConfigUnified:
    """Unified routing configuration with tenant support"""

    tenant_id: str = "default"

    # Routing mode
    routing_mode: str = "tiered"  # "tiered", "ensemble", "hybrid"

    # Tier configuration
    enable_fast_path: bool = True
    enable_slow_path: bool = True
    enable_fallback: bool = True
    fast_path_confidence_threshold: float = 0.7
    slow_path_confidence_threshold: float = 0.6
    max_routing_time_ms: int = 1000

    # GLiNER configuration (Fast Path)
    gliner_model: str = "urchade/gliner_large-v2.1"
    gliner_threshold: float = 0.3
    gliner_device: str = "cpu"
    gliner_labels: List[str] = field(default_factory=list)

    # LLM configuration (Slow Path)
    llm_provider: str = "local"
    llm_routing_model: str = "ollama/gemma3:4b"
    llm_endpoint: str = "http://localhost:11434"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 150
    use_chain_of_thought: bool = True

    # Optimization settings
    enable_auto_optimization: bool = True
    optimization_interval_seconds: int = 3600
    min_samples_for_optimization: int = 100
    dspy_enabled: bool = True
    dspy_max_bootstrapped_demos: int = 10
    dspy_max_labeled_demos: int = 50

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    max_cache_size: int = 1000

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tenant_id": self.tenant_id,
            "routing_mode": self.routing_mode,
            "enable_fast_path": self.enable_fast_path,
            "enable_slow_path": self.enable_slow_path,
            "enable_fallback": self.enable_fallback,
            "fast_path_confidence_threshold": self.fast_path_confidence_threshold,
            "slow_path_confidence_threshold": self.slow_path_confidence_threshold,
            "max_routing_time_ms": self.max_routing_time_ms,
            "gliner_model": self.gliner_model,
            "gliner_threshold": self.gliner_threshold,
            "gliner_device": self.gliner_device,
            "gliner_labels": self.gliner_labels,
            "llm_provider": self.llm_provider,
            "llm_routing_model": self.llm_routing_model,
            "llm_endpoint": self.llm_endpoint,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "use_chain_of_thought": self.use_chain_of_thought,
            "enable_auto_optimization": self.enable_auto_optimization,
            "optimization_interval_seconds": self.optimization_interval_seconds,
            "min_samples_for_optimization": self.min_samples_for_optimization,
            "dspy_enabled": self.dspy_enabled,
            "dspy_max_bootstrapped_demos": self.dspy_max_bootstrapped_demos,
            "dspy_max_labeled_demos": self.dspy_max_labeled_demos,
            "enable_caching": self.enable_caching,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "max_cache_size": self.max_cache_size,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingConfigUnified":
        """Create from dictionary"""
        return cls(
            tenant_id=data.get("tenant_id", "default"),
            routing_mode=data.get("routing_mode", "tiered"),
            enable_fast_path=data.get("enable_fast_path", True),
            enable_slow_path=data.get("enable_slow_path", True),
            enable_fallback=data.get("enable_fallback", True),
            fast_path_confidence_threshold=data.get(
                "fast_path_confidence_threshold", 0.7
            ),
            slow_path_confidence_threshold=data.get(
                "slow_path_confidence_threshold", 0.6
            ),
            max_routing_time_ms=data.get("max_routing_time_ms", 1000),
            gliner_model=data.get("gliner_model", "urchade/gliner_large-v2.1"),
            gliner_threshold=data.get("gliner_threshold", 0.3),
            gliner_device=data.get("gliner_device", "cpu"),
            gliner_labels=data.get("gliner_labels", []),
            llm_provider=data.get("llm_provider", "local"),
            llm_routing_model=data.get("llm_routing_model", "ollama/gemma3:4b"),
            llm_endpoint=data.get("llm_endpoint", "http://localhost:11434"),
            llm_temperature=data.get("llm_temperature", 0.1),
            llm_max_tokens=data.get("llm_max_tokens", 150),
            use_chain_of_thought=data.get("use_chain_of_thought", True),
            enable_auto_optimization=data.get("enable_auto_optimization", True),
            optimization_interval_seconds=data.get(
                "optimization_interval_seconds", 3600
            ),
            min_samples_for_optimization=data.get("min_samples_for_optimization", 100),
            dspy_enabled=data.get("dspy_enabled", True),
            dspy_max_bootstrapped_demos=data.get("dspy_max_bootstrapped_demos", 10),
            dspy_max_labeled_demos=data.get("dspy_max_labeled_demos", 50),
            enable_caching=data.get("enable_caching", True),
            cache_ttl_seconds=data.get("cache_ttl_seconds", 300),
            max_cache_size=data.get("max_cache_size", 1000),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TelemetryConfigUnified:
    """Unified telemetry configuration with tenant support"""

    tenant_id: str = "default"

    # Core settings
    enabled: bool = True
    level: str = "detailed"  # "disabled", "basic", "detailed", "verbose"
    environment: str = "development"

    # Phoenix settings
    phoenix_enabled: bool = True
    phoenix_endpoint: str = "localhost:4317"
    phoenix_use_tls: bool = False

    # Multi-tenant settings
    tenant_project_template: str = "cogniverse-{tenant_id}-{service}"
    max_cached_tenants: int = 100
    tenant_cache_ttl_seconds: int = 3600

    # Batch export settings
    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    export_timeout_millis: int = 30000
    schedule_delay_millis: int = 500
    use_sync_export: bool = False

    # Service identification
    service_name: str = "video-search"
    service_version: str = "1.0.0"

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tenant_id": self.tenant_id,
            "enabled": self.enabled,
            "level": self.level,
            "environment": self.environment,
            "phoenix_enabled": self.phoenix_enabled,
            "phoenix_endpoint": self.phoenix_endpoint,
            "phoenix_use_tls": self.phoenix_use_tls,
            "tenant_project_template": self.tenant_project_template,
            "max_cached_tenants": self.max_cached_tenants,
            "tenant_cache_ttl_seconds": self.tenant_cache_ttl_seconds,
            "max_queue_size": self.max_queue_size,
            "max_export_batch_size": self.max_export_batch_size,
            "export_timeout_millis": self.export_timeout_millis,
            "schedule_delay_millis": self.schedule_delay_millis,
            "use_sync_export": self.use_sync_export,
            "service_name": self.service_name,
            "service_version": self.service_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelemetryConfigUnified":
        """Create from dictionary"""
        return cls(
            tenant_id=data.get("tenant_id", "default"),
            enabled=data.get("enabled", True),
            level=data.get("level", "detailed"),
            environment=data.get("environment", "development"),
            phoenix_enabled=data.get("phoenix_enabled", True),
            phoenix_endpoint=data.get("phoenix_endpoint", "localhost:4317"),
            phoenix_use_tls=data.get("phoenix_use_tls", False),
            tenant_project_template=data.get(
                "tenant_project_template", "cogniverse-{tenant_id}-{service}"
            ),
            max_cached_tenants=data.get("max_cached_tenants", 100),
            tenant_cache_ttl_seconds=data.get("tenant_cache_ttl_seconds", 3600),
            max_queue_size=data.get("max_queue_size", 2048),
            max_export_batch_size=data.get("max_export_batch_size", 512),
            export_timeout_millis=data.get("export_timeout_millis", 30000),
            schedule_delay_millis=data.get("schedule_delay_millis", 500),
            use_sync_export=data.get("use_sync_export", False),
            service_name=data.get("service_name", "video-search"),
            service_version=data.get("service_version", "1.0.0"),
            metadata=data.get("metadata", {}),
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
        """Create from dictionary"""
        tenant_id = data.pop("tenant_id", "default")
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
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
        return result

    @classmethod
    def from_dict(cls, profile_name: str, data: Dict[str, Any]) -> "BackendProfileConfig":
        """Create from dictionary"""
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
        )


@dataclass
class BackendConfig:
    """Backend configuration with multi-tenant profile support"""

    tenant_id: str = "default"
    backend_type: str = "vespa"
    url: str = "http://localhost"
    port: int = 8080
    profiles: Dict[str, BackendProfileConfig] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tenant_id": self.tenant_id,
            "type": self.backend_type,
            "url": self.url,
            "port": self.port,
            "profiles": {
                name: profile.to_dict()
                for name, profile in self.profiles.items()
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackendConfig":
        """Create from dictionary"""
        profiles_data = data.get("profiles", {})
        profiles = {
            name: BackendProfileConfig.from_dict(name, profile_data)
            for name, profile_data in profiles_data.items()
        }

        return cls(
            tenant_id=data.get("tenant_id", "default"),
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

    def merge_profile(self, profile_name: str, overrides: Dict[str, Any]) -> BackendProfileConfig:
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
    description_fields: List[str] = field(default_factory=lambda: ["segment_description", "description"])
    transcript_fields: List[str] = field(default_factory=lambda: ["audio_transcript", "transcript"])
    entity_fields: List[str] = field(default_factory=lambda: ["video_title", "segment_description"])
    temporal_fields: Dict[str, str] = field(default_factory=lambda: {
        "start": "start_time",
        "end": "end_time"
    })
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
            description_fields=data.get("description_fields", ["segment_description", "description"]),
            transcript_fields=data.get("transcript_fields", ["audio_transcript", "transcript"]),
            entity_fields=data.get("entity_fields", ["video_title", "segment_description"]),
            temporal_fields=data.get("temporal_fields", {"start": "start_time", "end": "end_time"}),
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
    compiled_path: Optional[str] = None  # Path to optimized module
    lm_config: Dict[str, Any] = field(default_factory=dict)  # LLM settings
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "signature_class": self.signature_class,
            "module_type": self.module_type,
            "compiled_path": self.compiled_path,
            "lm_config": self.lm_config,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DSPyModuleConfig":
        """Create from dictionary"""
        return cls(
            signature_class=data["signature_class"],
            module_type=data.get("module_type", "ChainOfThought"),
            compiled_path=data.get("compiled_path"),
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
    dspy_modules: Dict[str, DSPyModuleConfig] = field(default_factory=dict)  # e.g., {"query_generator": DSPyModuleConfig(...)}
    profile_scoring_rules: List[ProfileScoringRule] = field(default_factory=list)
    agent_mappings: List[AgentMappingRule] = field(default_factory=list)
    num_examples_target: int = 50
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "optimizer_type": self.optimizer_type,
            "dspy_modules": {
                key: module.to_dict()
                for key, module in self.dspy_modules.items()
            },
            "profile_scoring_rules": [
                rule.to_dict() for rule in self.profile_scoring_rules
            ],
            "agent_mappings": [
                mapping.to_dict() for mapping in self.agent_mappings
            ],
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
    """
    tenant_id: str = "default"
    field_mappings: FieldMappingConfig = field(default_factory=FieldMappingConfig)
    optimizer_configs: Dict[str, OptimizerGenerationConfig] = field(default_factory=dict)
    sampling_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tenant_id": self.tenant_id,
            "field_mappings": self.field_mappings.to_dict(),
            "optimizer_configs": {
                key: config.to_dict()
                for key, config in self.optimizer_configs.items()
            },
            "sampling_config": self.sampling_config,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyntheticGeneratorConfig":
        """Create from dictionary"""
        field_mappings = FieldMappingConfig.from_dict(
            data.get("field_mappings", {})
        )

        optimizer_configs = {}
        for key, config_data in data.get("optimizer_configs", {}).items():
            optimizer_configs[key] = OptimizerGenerationConfig.from_dict(config_data)

        return cls(
            tenant_id=data.get("tenant_id", "default"),
            field_mappings=field_mappings,
            optimizer_configs=optimizer_configs,
            sampling_config=data.get("sampling_config", {}),
            metadata=data.get("metadata", {}),
        )

    def get_optimizer_config(self, optimizer_type: str) -> Optional[OptimizerGenerationConfig]:
        """Get configuration for a specific optimizer"""
        return self.optimizer_configs.get(optimizer_type)


# ConfigEntry is now defined in config_store_interface.py
# Import it from there if needed
