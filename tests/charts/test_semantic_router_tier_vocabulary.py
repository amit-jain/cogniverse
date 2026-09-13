"""The tier the runtime emits must name a group the router binds.

The runtime sends ``x-authz-user-groups: <tier>``; every routing decision is
gated on an ``authz`` condition naming a role, and roles are bound to Groups.
A tier naming no bound group matches no decision, so the router classifies the
request, discards the result, and falls through to
``providers.defaults.default_model`` - which is what the deployed cluster was
doing for every request, because the runtime's only shipped tier is
``default`` and nothing bound it.

The vocabulary lives once, in ``ROUTER_TIERS``; these pin the chart against it
rather than restating it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from cogniverse_foundation.config.unified_config import ROUTER_TIERS

pytestmark = [pytest.mark.unit]

_REPO = Path(__file__).resolve().parents[2]
_CHART_ROUTER_CONFIG = (
    _REPO / "charts" / "cogniverse" / "files" / "semantic-router" / "config.yaml"
)
_STACK_ROUTER_CONFIG = (
    _REPO / "tests" / "foundation" / "integration" / "_sr_stack" / "sr-config.yaml"
)

# Helm expressions are opaque to the YAML parser. A ``.Values`` lookup is
# resolved against values.yaml so the twin can be compared against the numbers
# the chart really renders; anything else is neutralised to load the file.
_TEMPLATE = re.compile(r"\{\{-?\s*(?P<body>.*?)\s*-?\}\}")
_VALUES_LOOKUP = re.compile(r"^(?:int\s+)?\.Values\.(?P<path>[\w.]+)$")
_CHART_VALUES = _REPO / "charts" / "cogniverse" / "values.yaml"


def _chart_value(dotted: str) -> str:
    node = yaml.safe_load(_CHART_VALUES.read_text())
    for key in dotted.split("."):
        node = node[key]
    return str(node)


def _resolve(match: re.Match) -> str:
    lookup = _VALUES_LOOKUP.match(match["body"])
    return _chart_value(lookup["path"]) if lookup else "templated"


def _load(path: Path) -> dict:
    return yaml.safe_load(_TEMPLATE.sub(_resolve, path.read_text()))


def _bound_groups(config: dict) -> set[str]:
    return {
        subject["name"]
        for binding in config["routing"]["signals"]["role_bindings"]
        for subject in binding["subjects"]
        if subject["kind"] == "Group"
    }


def _routing_profiles(config: dict) -> list[dict]:
    """The default profile and every recipe's, which the router evaluates
    independently: a tier bound only in the default profile matches nothing on
    a request that entered through a recipe."""
    return [config["routing"]] + [
        recipe["routing"] for recipe in config.get("recipes", [])
    ]


def _roles_with_a_decision(config: dict) -> set[str]:
    return {
        condition["name"]
        for decision in config["routing"]["decisions"]
        for condition in decision["rules"]["conditions"]
        if condition["type"] == "authz"
    }


class TestEveryRuntimeTierReachesADecision:
    def test_the_chart_binds_a_group_for_every_tier_the_runtime_emits(self):
        assert ROUTER_TIERS <= _bound_groups(_load(_CHART_ROUTER_CONFIG))

    def test_the_chart_binds_no_group_the_runtime_cannot_emit(self):
        """Containment both ways: a bound group nothing emits is dead config."""
        assert _bound_groups(_load(_CHART_ROUTER_CONFIG)) == set(ROUTER_TIERS)

    def test_every_bound_role_has_a_decision(self):
        """A group bound to a role no decision names still matches nothing."""
        config = _load(_CHART_ROUTER_CONFIG)
        bound_roles = {
            binding["role"] for binding in config["routing"]["signals"]["role_bindings"]
        }
        assert bound_roles <= _roles_with_a_decision(config)

    def test_every_tier_reaches_a_decision_in_every_routing_profile(self):
        """Including the classification recipe: a tier the recipe does not bind
        falls through to providers.defaults.default_model, which is the tier
        gate silently off for every bounded-output call that tenant makes."""
        config = _load(_CHART_ROUTER_CONFIG)
        for profile in _routing_profiles(config):
            groups = {
                subject["name"]
                for binding in profile["signals"]["role_bindings"]
                for subject in binding["subjects"]
                if subject["kind"] == "Group"
            }
            roles_with_decision = {
                condition["name"]
                for decision in profile["decisions"]
                for condition in decision["rules"]["conditions"]
                if condition["type"] == "authz"
            }
            bound_roles = {
                binding["role"] for binding in profile["signals"]["role_bindings"]
            }
            assert groups == set(ROUTER_TIERS)
            assert bound_roles == roles_with_decision


class TestTheClassificationRecipeRunsNoClassifier:
    """The entrypoint exists to remove the domain classifier from the request.
    A condition of any classifying type inside the recipe puts it back."""

    def test_the_recipe_conditions_are_authz_only(self):
        config = _load(_CHART_ROUTER_CONFIG)
        recipe = next(r for r in config["recipes"] if r["name"] == "classification")
        types = {
            condition["type"]
            for decision in recipe["routing"]["decisions"]
            for condition in decision["rules"]["conditions"]
        }
        assert types == {"authz"}

    def test_the_recipe_declares_no_classifying_signal(self):
        config = _load(_CHART_ROUTER_CONFIG)
        recipe = next(r for r in config["recipes"] if r["name"] == "classification")
        assert sorted(recipe["routing"]["signals"]) == ["role_bindings"]

    def test_the_entrypoint_model_is_not_a_catalog_model_or_the_auto_alias(self):
        """The router refuses an entrypoint name that collides with a model or
        a reserved alias, so a collision is a router that will not start."""
        config = _load(_CHART_ROUTER_CONFIG)
        served = {name for e in config["entrypoints"] for name in e["model_names"]}
        catalog = {model["name"] for model in config["providers"]["models"]}
        catalog |= {card["name"] for card in config["routing"]["modelCards"]}
        assert served == {"cogniverse-classification", "cogniverse-vision"}
        assert served & (catalog | {"auto", "vllm-sr/auto", "MoM"}) == set()

    def test_every_entrypoint_names_a_declared_recipe(self):
        config = _load(_CHART_ROUTER_CONFIG)
        declared = {recipe["name"] for recipe in config["recipes"]}
        assert {e["recipe"] for e in config["entrypoints"]} == declared


class TestTheTestStackRoutesTheShippedConfiguration:
    """The stack's router config is a committed twin of the chart's. A twin
    that drifts tests a configuration nothing deploys - which is how a
    semantic cache enabled only in the chart went unexercised."""

    def test_the_twin_routes_exactly_what_the_chart_routes(self):
        assert (
            _load(_STACK_ROUTER_CONFIG)["routing"]
            == (_load(_CHART_ROUTER_CONFIG)["routing"])
        )

    def test_the_twin_serves_exactly_the_charts_entrypoints_and_recipes(self):
        chart = _load(_CHART_ROUTER_CONFIG)
        twin = _load(_STACK_ROUTER_CONFIG)
        assert twin["entrypoints"] == chart["entrypoints"]
        assert twin["recipes"] == chart["recipes"]

    def test_the_twin_and_the_chart_agree_on_the_response_cache_store(self):
        """Including the bounds: the chart reads them from values.yaml, and a
        twin pinned only on the block's shape would absorb a changed TTL."""
        chart = _load(_CHART_ROUTER_CONFIG)["global"]["stores"]
        twin = _load(_STACK_ROUTER_CONFIG)["global"]["stores"]
        assert chart == twin
