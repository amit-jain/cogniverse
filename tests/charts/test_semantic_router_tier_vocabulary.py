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

# Helm expressions are opaque to the YAML parser; the routing section this
# module reads carries none, so neutralising them is enough to load the file.
_TEMPLATE = re.compile(r"\{\{-?.*?-?\}\}")


def _load(path: Path) -> dict:
    return yaml.safe_load(_TEMPLATE.sub("templated", path.read_text()))


def _bound_groups(config: dict) -> set[str]:
    return {
        subject["name"]
        for binding in config["routing"]["signals"]["role_bindings"]
        for subject in binding["subjects"]
        if subject["kind"] == "Group"
    }


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


class TestTheTestStackRoutesTheShippedConfiguration:
    """The stack's router config is a committed twin of the chart's. A twin
    that drifts tests a configuration nothing deploys - which is how a
    semantic cache enabled only in the chart went unexercised."""

    def test_the_twin_routes_exactly_what_the_chart_routes(self):
        assert (
            _load(_STACK_ROUTER_CONFIG)["routing"]
            == (_load(_CHART_ROUTER_CONFIG)["routing"])
        )

    def test_the_twin_and_the_chart_agree_on_the_semantic_cache(self):
        chart = _load(_CHART_ROUTER_CONFIG)["global"]["stores"]["semantic_cache"]
        twin = _load(_STACK_ROUTER_CONFIG)["global"]["stores"]["semantic_cache"]
        assert chart == twin
