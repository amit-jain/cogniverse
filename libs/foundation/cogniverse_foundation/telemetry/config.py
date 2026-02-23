"""
Configuration for telemetry system.

Note: Core config is generic - no provider-specific fields (Phoenix, LangSmith, etc.).
Provider-specific config goes in provider_config dict.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

# Span name constants
SPAN_NAME_REQUEST = "cogniverse.request"
SPAN_NAME_ROUTING = "cogniverse.routing"
SPAN_NAME_ORCHESTRATION = "cogniverse.orchestration"

# Service name constants
SERVICE_NAME_ORCHESTRATION = "cogniverse.orchestration"


class TelemetryLevel(Enum):
    """Telemetry collection levels."""

    DISABLED = "disabled"
    BASIC = "basic"  # Only search operations
    DETAILED = "detailed"  # Search + encoders + backend
    VERBOSE = "verbose"  # Everything including internal operations


@dataclass
class BatchExportConfig:
    """Configuration for batch span export."""

    max_queue_size: int = 2048
    max_export_batch_size: int = 512
    export_timeout_millis: int = 30_000
    schedule_delay_millis: int = 500
    drop_on_queue_full: bool = True
    log_dropped_spans: bool = True
    max_drop_log_rate_per_minute: int = 10
    use_sync_export: bool = False


@dataclass
class TelemetryConfig:
    """
    Generic telemetry configuration.

    Core config has ZERO knowledge of provider specifics (Phoenix, LangSmith, etc.).
    Provider-specific config goes in provider_config dict.
    """

    # Core settings
    enabled: bool = True
    level: TelemetryLevel = TelemetryLevel.DETAILED
    environment: str = "development"

    # OpenTelemetry span export (generic OTLP) - backend-agnostic
    otlp_enabled: bool = True
    otlp_endpoint: str = "localhost:4317"
    otlp_use_tls: bool = False

    # Provider selection (for querying spans/annotations/datasets)
    # Separate from span export (which uses OpenTelemetry OTLP)
    provider: Optional[str] = None  # "phoenix" | "langsmith" | None (auto-detect)

    # Generic provider configuration (dict - provider interprets)
    # Core doesn't know what keys providers expect
    # Examples:
    #   Phoenix: {"http_endpoint": "...", "grpc_endpoint": "..."}
    #   LangSmith: {"api_key": "...", "project": "..."}
    provider_config: Dict[str, Any] = field(default_factory=dict)

    # Multi-tenant settings
    tenant_project_template: str = "cogniverse-{tenant_id}"
    tenant_service_template: str = "cogniverse-{tenant_id}-{service}"
    default_tenant_id: str = "default"
    max_cached_tenants: int = 100  # LRU cache size
    tenant_cache_ttl_seconds: int = 3600  # 1 hour

    # Batch export settings
    batch_config: BatchExportConfig = field(default_factory=BatchExportConfig)

    # Service identification
    service_name: str = "video-search"
    service_version: str = "1.0.0"

    # Resource attributes
    extra_resource_attributes: Dict[str, str] = field(default_factory=dict)

    def get_project_name(self, tenant_id: str, service: Optional[str] = None) -> str:
        """
        Generate project name for a tenant.

        User operations: cogniverse-{tenant_id}
        Management operations: cogniverse-{tenant_id}-{service}
        """
        if service:
            return self.tenant_service_template.format(
                tenant_id=tenant_id, service=service
            )
        return self.tenant_project_template.format(tenant_id=tenant_id)

    def should_instrument_level(self, component: str) -> bool:
        """Check if a component should be instrumented based on level."""
        if not self.enabled:
            return False

        level_components = {
            TelemetryLevel.DISABLED: set(),
            TelemetryLevel.BASIC: {"search_service"},
            TelemetryLevel.DETAILED: {"search_service", "backend", "encoder"},
            TelemetryLevel.VERBOSE: {
                "search_service",
                "backend",
                "encoder",
                "pipeline",
                "agents",
            },
        }

        return component in level_components.get(self.level, set())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence."""
        return {
            "enabled": self.enabled,
            "level": self.level.value,
            "environment": self.environment,
            "otlp_enabled": self.otlp_enabled,
            "otlp_endpoint": self.otlp_endpoint,
            "otlp_use_tls": self.otlp_use_tls,
            "provider": self.provider,
            "provider_config": self.provider_config,
            "tenant_project_template": self.tenant_project_template,
            "tenant_service_template": self.tenant_service_template,
            "default_tenant_id": self.default_tenant_id,
            "max_cached_tenants": self.max_cached_tenants,
            "tenant_cache_ttl_seconds": self.tenant_cache_ttl_seconds,
            "max_queue_size": self.batch_config.max_queue_size,
            "max_export_batch_size": self.batch_config.max_export_batch_size,
            "export_timeout_millis": self.batch_config.export_timeout_millis,
            "schedule_delay_millis": self.batch_config.schedule_delay_millis,
            "drop_on_queue_full": self.batch_config.drop_on_queue_full,
            "log_dropped_spans": self.batch_config.log_dropped_spans,
            "max_drop_log_rate_per_minute": self.batch_config.max_drop_log_rate_per_minute,
            "use_sync_export": self.batch_config.use_sync_export,
            "service_name": self.service_name,
            "service_version": self.service_version,
            "extra_resource_attributes": self.extra_resource_attributes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelemetryConfig":
        """Deserialize from dictionary."""
        level_str = data.get("level", "detailed")
        try:
            level = TelemetryLevel(level_str)
        except ValueError:
            level = TelemetryLevel.DETAILED

        batch_config = BatchExportConfig(
            max_queue_size=data.get("max_queue_size", 2048),
            max_export_batch_size=data.get("max_export_batch_size", 512),
            export_timeout_millis=data.get("export_timeout_millis", 30_000),
            schedule_delay_millis=data.get("schedule_delay_millis", 500),
            drop_on_queue_full=data.get("drop_on_queue_full", True),
            log_dropped_spans=data.get("log_dropped_spans", True),
            max_drop_log_rate_per_minute=data.get("max_drop_log_rate_per_minute", 10),
            use_sync_export=data.get("use_sync_export", False),
        )

        return cls(
            enabled=data.get("enabled", True),
            level=level,
            environment=data.get("environment", "development"),
            otlp_enabled=data.get("otlp_enabled", True),
            otlp_endpoint=data.get("otlp_endpoint", "localhost:4317"),
            otlp_use_tls=data.get("otlp_use_tls", False),
            provider=data.get("provider"),
            provider_config=data.get("provider_config", {}),
            tenant_project_template=data.get(
                "tenant_project_template", "cogniverse-{tenant_id}"
            ),
            tenant_service_template=data.get(
                "tenant_service_template", "cogniverse-{tenant_id}-{service}"
            ),
            default_tenant_id=data.get("default_tenant_id", "default"),
            max_cached_tenants=data.get("max_cached_tenants", 100),
            tenant_cache_ttl_seconds=data.get("tenant_cache_ttl_seconds", 3600),
            batch_config=batch_config,
            service_name=data.get("service_name", "video-search"),
            service_version=data.get("service_version", "1.0.0"),
            extra_resource_attributes=data.get("extra_resource_attributes", {}),
        )

    def validate(self) -> None:
        """Validate configuration."""
        if self.enabled and self.otlp_enabled:
            if not self.otlp_endpoint:
                raise ValueError("otlp_endpoint required when OTLP span export enabled")

        if self.batch_config.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")

        if self.max_cached_tenants <= 0:
            raise ValueError("max_cached_tenants must be positive")
