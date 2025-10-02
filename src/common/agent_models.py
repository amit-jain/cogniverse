"""Shared agent data models"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class AgentEndpoint:
    """Agent endpoint configuration with health monitoring"""

    name: str
    url: str
    capabilities: List[str]
    health_endpoint: str = "/health"
    process_endpoint: str = "/process"
    timeout: int = 30
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"  # unknown, healthy, unhealthy, unreachable
    health_check_interval: int = 60  # seconds

    def is_healthy(self) -> bool:
        """Check if agent is considered healthy"""
        return self.health_status == "healthy"

    def needs_health_check(self) -> bool:
        """Check if agent needs a health check"""
        if not self.last_health_check:
            return True
        elapsed = (datetime.now() - self.last_health_check).total_seconds()
        return elapsed >= self.health_check_interval
