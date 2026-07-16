"""Shared agent data models"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

# Default per-agent call timeout (seconds). Matches the orchestrator's
# agent-call HTTP client so a config that omits ``timeout`` doesn't silently
# shorten slow agent calls (deep synthesis, long LM chains) — the previous 30s
# default was ignored by the caller, which hardcoded this value.
DEFAULT_AGENT_CALL_TIMEOUT_SECONDS = 240


@dataclass
class AgentEndpoint:
    """Agent endpoint configuration with health monitoring"""

    name: str
    url: str
    capabilities: List[str]
    health_endpoint: str = "/health"
    process_endpoint: str = "/process"
    timeout: int = DEFAULT_AGENT_CALL_TIMEOUT_SECONDS
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
        # Normalise stored stamp to aware UTC so a registry written before
        # the tz-aware switch doesn't crash the subtraction.
        stored = self.last_health_check
        if stored.tzinfo is None:
            stored = stored.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - stored).total_seconds()
        return elapsed >= self.health_check_interval
