"""Routing subsystem.

Live routing entry points:
  - ``cogniverse_agents.gateway_agent.GatewayAgent``     — GLiNER triage
  - ``cogniverse_agents.orchestrator_agent.OrchestratorAgent`` — DSPy planner

Preprocessing agents (``entity_extraction_agent``, ``query_enhancement_agent``,
``profile_selection_agent``) produce enrichment that the orchestrator threads
onto each execution agent's typed input fields (``enhanced_query``,
``entities``, ``relationships``, ``query_variants``, ``profiles``) via
``OrchestratorAgent._merge_enrichment``.

This package also holds:
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
