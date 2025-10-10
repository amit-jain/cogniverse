"""
Unified configuration schema with multi-tenant support.
Consolidates AgentConfig, RoutingConfig, TelemetryConfig into single system.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from cogniverse_core.common.agent_config import (
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

    # Search backend configuration
    search_backend: str = "vespa"
    vespa_url: str = "http://localhost"
    vespa_port: int = 8080
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
            "search_backend": self.search_backend,
            "vespa_url": self.vespa_url,
            "vespa_port": self.vespa_port,
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
            search_backend=data.get("search_backend", "vespa"),
            vespa_url=data.get("vespa_url", "http://localhost"),
            vespa_port=data.get("vespa_port", 8080),
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


# ConfigEntry is now defined in config_store_interface.py
# Import it from there if needed
