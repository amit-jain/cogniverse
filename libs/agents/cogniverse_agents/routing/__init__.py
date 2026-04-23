"""Routing subsystem.

The live routing path is served by:
  - ``cogniverse_agents.gateway_agent.GatewayAgent``     — fast GLiNER triage
  - ``cogniverse_agents.orchestrator_agent.OrchestratorAgent`` — LLM-planned pipeline

This package still holds:
  - ``cogniverse_agents.routing.contract`` — ``RoutingContext`` wire type
  - ``cogniverse_agents.routing.config`` — runtime config loaders
  - offline optimisation/analytics modules (``modality_*``, ``xgboost_meta_models``,
    ``advanced_optimizer``, ``annotation_*``) used by Argo CronWorkflows and the
    Phoenix dashboard.
"""

from .config import AutomationRulesConfig, RoutingConfig, load_config

__all__ = [
    "AutomationRulesConfig",
    "RoutingConfig",
    "load_config",
]
