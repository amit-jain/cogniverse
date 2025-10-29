"""
Configuration for telemetry system.

Note: Core config is generic - no provider-specific fields (Phoenix, LangSmith, etc.).
Provider-specific config goes in provider_config dict.
"""

import os
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
    tenant_project_template: str = "cogniverse-{tenant_id}-{service}"
    default_tenant_id: str = "default"
    max_cached_tenants: int = 100  # LRU cache size
    tenant_cache_ttl_seconds: int = 3600  # 1 hour

    # Batch export settings
    batch_config: BatchExportConfig = field(default_factory=BatchExportConfig)

    # Service identification
    service_name: str = "video-search"
    service_version: str = field(
        default_factory=lambda: os.getenv("SERVICE_VERSION", "1.0.0")
    )

    # Resource attributes
    extra_resource_attributes: Dict[str, str] = field(default_factory=dict)

    def get_project_name(self, tenant_id: str, service: Optional[str] = None) -> str:
        """Generate project name for a tenant."""
        service = service or self.service_name
        return self.tenant_project_template.format(tenant_id=tenant_id, service=service)

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

    @classmethod
    def from_env(cls) -> "TelemetryConfig":
        """Create config from environment variables."""
        return cls()

    def validate(self) -> None:
        """Validate configuration."""
        if self.enabled and self.otlp_enabled:
            if not self.otlp_endpoint:
                raise ValueError("otlp_endpoint required when OTLP span export enabled")

        if self.batch_config.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")

        if self.max_cached_tenants <= 0:
            raise ValueError("max_cached_tenants must be positive")
