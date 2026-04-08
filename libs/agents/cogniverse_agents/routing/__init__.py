# src/routing/__init__.py
"""
Routing subsystem — base types, config, and optimization.

ComprehensiveRouter/TieredRouter/strategies have been replaced by
GatewayAgent (fast GLiNER triage) + OrchestratorAgent (LLM-planned pipeline).
"""

from .base import RoutingDecision, RoutingMetrics, RoutingStrategy
from .config import AutomationRulesConfig, RoutingConfig, load_config
from .optimizer import AutoTuningOptimizer, RoutingOptimizer

__all__ = [
    "RoutingStrategy",
    "RoutingDecision",
    "RoutingMetrics",
    "RoutingOptimizer",
    "AutoTuningOptimizer",
    "AutomationRulesConfig",
    "RoutingConfig",
    "load_config",
]
