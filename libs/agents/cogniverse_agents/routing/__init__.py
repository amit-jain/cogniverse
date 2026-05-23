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
  - ``cogniverse_agents.routing.config`` — pydantic schemas for the
    annotation / online-evaluator config knobs
    (``AutomationRulesConfig`` + ``OnlineEvaluationConfig``)
  - offline optimisation/analytics modules (``modality_*``, ``xgboost_meta_models``,
    ``advanced_optimizer``, ``annotation_*``) used by Argo CronWorkflows and the
    Phoenix dashboard.

The routing-system configuration itself lives in
``cogniverse_foundation.config.unified_config.RoutingConfigUnified``.
"""

from .config import AutomationRulesConfig, OnlineEvaluationConfig

__all__ = [
    "AutomationRulesConfig",
    "OnlineEvaluationConfig",
]
