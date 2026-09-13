"""Every routed LM call site declares which router entry it takes.

The router bills a domain classification before it picks a decision for a
request on the ``auto`` alias. A bounded-output call - one whose answer is a
selection from a fixed vocabulary, a short structured record, or a one-line
transformation - has nothing to gain from that classification, so it names the
``cogniverse-classification`` entrypoint instead and the router chooses from
the tenant tier alone.

The two sets are the whole vocabulary: an agent in neither is an agent nobody
classified, and a call site naming something the chart does not serve reaches
the router as an unknown model. Both facts are checked against the shipped
files (``configs/config.json`` and the chart's router config), never against a
list restated here.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import yaml

from cogniverse_foundation.config.semantic_router import (
    CLASSIFICATION_CALL_SITES,
    FREE_FORM_CALL_SITES,
    routed_model_for,
)
from cogniverse_foundation.config.unified_config import SemanticRouterConfig
from tests.utils.semantic_router_stack import render_router_config

pytestmark = [pytest.mark.unit]

_REPO = Path(__file__).resolve().parents[3]
_SHIPPED_CONFIG = _REPO / "configs" / "config.json"
_LIBS = _REPO / "libs"

# Call sites that are not agents: a shared LM seam reached from many agents,
# so it has no agent name of its own.
_NON_AGENT_CALL_SITES = frozenset(
    {"dynamic_dspy_module", "rlm_inference", "vlm_interface"}
)


def _chart_router_config() -> dict:
    return yaml.safe_load(render_router_config())


def _shipped_agents() -> set[str]:
    return set(json.loads(_SHIPPED_CONFIG.read_text())["agents"])


def _literal_call_sites() -> dict[str, list[str]]:
    """Every ``call_site="..."`` literal in ``libs/``, keyed by value."""
    found: dict[str, list[str]] = {}
    for path in _LIBS.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if kw.arg != "call_site":
                    continue
                if isinstance(kw.value, ast.Constant) and isinstance(
                    kw.value.value, str
                ):
                    found.setdefault(kw.value.value, []).append(
                        f"{path.relative_to(_REPO)}:{node.lineno}"
                    )
    return found


class TestTheCatalogCoversTheShippedAgents:
    def test_every_shipped_agent_is_classified_exactly_once(self):
        classified = CLASSIFICATION_CALL_SITES | FREE_FORM_CALL_SITES
        assert _shipped_agents() - classified == set()
        assert CLASSIFICATION_CALL_SITES & FREE_FORM_CALL_SITES == frozenset()

    def test_the_catalog_names_nothing_the_deployment_does_not_run(self):
        """A name in neither configs/config.json nor the non-agent seam list is
        a dead entry the router will never be asked for."""
        classified = CLASSIFICATION_CALL_SITES | FREE_FORM_CALL_SITES
        assert classified - _shipped_agents() == set(_NON_AGENT_CALL_SITES)

    def test_the_bounded_output_agents_are_exactly_these(self):
        assert CLASSIFICATION_CALL_SITES == frozenset(
            {
                "entity_extraction_agent",
                "gateway_agent",
                "orchestrator_agent",
                "profile_selection_agent",
                "query_enhancement_agent",
                "search_agent",
            }
        )


class TestEveryCallSiteInLibsIsClassified:
    def test_every_literal_call_site_is_in_the_catalog(self):
        classified = CLASSIFICATION_CALL_SITES | FREE_FORM_CALL_SITES
        unknown = {
            value: where
            for value, where in _literal_call_sites().items()
            if value not in classified
        }
        assert unknown == {}

    def test_the_dispatcher_and_the_shared_seams_are_the_literal_call_sites(self):
        """The set of literals is small and named, so a new one is a visible
        edit here rather than a silent default onto the classifying alias."""
        assert set(_literal_call_sites()) == {
            "coding_agent",
            "dynamic_dspy_module",
            "rlm_inference",
            "vlm_interface",
        }


class TestTheModelNameMatchesTheChart:
    def test_the_configured_entries_name_the_charts_entrypoints(self):
        """The runtime's two virtual model names are exactly the chart's two
        entrypoints: bounded calls on classification, image-bearing calls on
        vision, both sent with litellm's openai/ prefix."""
        entrypoints = _chart_router_config()["entrypoints"]
        served = {name for e in entrypoints for name in e["model_names"]}
        config = SemanticRouterConfig()
        assert config.classification_model.split("/", 1) == [
            "openai",
            "cogniverse-classification",
        ]
        assert config.vision_model.split("/", 1) == ["openai", "cogniverse-vision"]
        assert served == {
            config.classification_model.split("/", 1)[1],
            config.vision_model.split("/", 1)[1],
        }

    def test_the_free_form_model_is_the_routers_auto_alias(self):
        assert SemanticRouterConfig().routed_model == "openai/auto"

    def test_a_bounded_call_site_takes_the_classification_entrypoint(self):
        config = SemanticRouterConfig()
        assert (
            routed_model_for(config, "search_agent")
            == "openai/cogniverse-classification"
        )

    def test_a_free_form_call_site_takes_the_auto_alias(self):
        config = SemanticRouterConfig()
        assert routed_model_for(config, "summarizer_agent") == "openai/auto"

    def test_an_unclassified_call_site_takes_the_auto_alias(self):
        """The classifying entry is the safe default: an agent added without a
        classification keeps today's behaviour instead of inheriting a bounded
        decision that may not fit it."""
        config = SemanticRouterConfig()
        assert routed_model_for(config, "an_agent_added_tomorrow") == "openai/auto"

    def test_the_configured_names_reach_the_serialized_form(self):
        assert SemanticRouterConfig().to_dict() == {
            "enabled": False,
            "semantic_router_url": "",
            "tier_header": "x-authz-user-groups",
            "user_id_header": "x-authz-user-id",
            "routed_model": "openai/auto",
            "response_cache_ttl_seconds": 3600,
            "response_cache_max_entries": 1024,
            "classification_model": "openai/cogniverse-classification",
            "vision_model": "openai/cogniverse-vision",
        }
