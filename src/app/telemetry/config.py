"""
Configuration for telemetry system.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


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
    
    # Queue behavior when full
    drop_on_queue_full: bool = True  # Drop spans instead of blocking
    log_dropped_spans: bool = True
    max_drop_log_rate_per_minute: int = 10


@dataclass
class TelemetryConfig:
    """Telemetry configuration."""
    
    # Core settings
    enabled: bool = field(default_factory=lambda: os.getenv("TELEMETRY_ENABLED", "true").lower() == "true")
    level: TelemetryLevel = field(default_factory=lambda: TelemetryLevel(os.getenv("TELEMETRY_LEVEL", "detailed")))
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    
    # Phoenix settings
    phoenix_enabled: bool = field(default_factory=lambda: os.getenv("PHOENIX_ENABLED", "true").lower() == "true")
    phoenix_endpoint: str = field(default_factory=lambda: os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "localhost:4317"))
    phoenix_use_tls: bool = field(default_factory=lambda: os.getenv("PHOENIX_USE_TLS", "false").lower() == "true")
    
    # Multi-tenant settings
    tenant_project_template: str = "cogniverse-{tenant_id}-{service}"
    routing_optimization_template: str = "cogniverse-{tenant_id}-routing-optimization"
    default_tenant_id: str = "default"
    max_cached_tenants: int = 100  # LRU cache size
    tenant_cache_ttl_seconds: int = 3600  # 1 hour
    
    # Batch export settings
    batch_config: BatchExportConfig = field(default_factory=BatchExportConfig)
    
    # Service identification
    service_name: str = "video-search"
    service_version: str = field(default_factory=lambda: os.getenv("SERVICE_VERSION", "1.0.0"))
    
    # Resource attributes
    extra_resource_attributes: Dict[str, str] = field(default_factory=dict)
    
    def get_project_name(self, tenant_id: str, service: Optional[str] = None) -> str:
        """Generate project name for a tenant."""
        service = service or self.service_name
        return self.tenant_project_template.format(
            tenant_id=tenant_id,
            service=service
        )

    def get_routing_optimization_project_name(self, tenant_id: str) -> str:
        """Generate routing optimization project name for a tenant."""
        return self.routing_optimization_template.format(tenant_id=tenant_id)
    
    def should_instrument_level(self, component: str) -> bool:
        """Check if a component should be instrumented based on level."""
        if not self.enabled:
            return False
            
        level_components = {
            TelemetryLevel.DISABLED: set(),
            TelemetryLevel.BASIC: {"search_service"},
            TelemetryLevel.DETAILED: {"search_service", "backend", "encoder"},
            TelemetryLevel.VERBOSE: {"search_service", "backend", "encoder", "pipeline", "agents"}
        }
        
        return component in level_components.get(self.level, set())
    
    @classmethod
    def from_env(cls) -> "TelemetryConfig":
        """Create config from environment variables."""
        return cls()
    
    def validate(self) -> None:
        """Validate configuration."""
        if self.enabled and self.phoenix_enabled:
            if not self.phoenix_endpoint:
                raise ValueError("phoenix_endpoint required when Phoenix enabled")
        
        if self.batch_config.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        
        if self.max_cached_tenants <= 0:
            raise ValueError("max_cached_tenants must be positive")