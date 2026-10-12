"""Config sections: form values, schemas and submitted values."""

import pytest

from cogniverse_foundation.config.sections import (
    CONFIG_SECTIONS,
    ConfigValueError,
    section_for,
)
from cogniverse_sdk.interfaces.config_store import ConfigScope

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.mark.parametrize("name", list(CONFIG_SECTIONS))
def test_an_empty_submission_leaves_the_config_as_it_was(name):
    section = CONFIG_SECTIONS[name]
    config = section.default("acme:prod", "search_agent")
    stored = section.dump(config, "acme:prod")
    assert section.dump(section.from_form({}, config), "acme:prod") == stored
    assert section.load(stored) == section.from_form({}, config)


def test_unknown_keys_are_found_inside_optional_and_nested_objects():
    agent = CONFIG_SECTIONS["agent"]
    current = agent.default("acme:prod", "search_agent")
    with pytest.raises(ConfigValueError) as raised:
        agent.from_form(
            {
                "optimizer_config": {"optimizer_type": "mipro_v2", "budget": 3},
                "module_config": {"signature": "Q -> A", "depth": 2},
                "metadata": {"free": "form"},
            },
            current,
        )
    assert raised.value.errors == [
        "unknown field optimizer_config.budget",
        "unknown field module_config.depth",
    ]


def test_fixed_fields_are_neither_shown_nor_required_nor_settable():
    routing = CONFIG_SECTIONS["routing"]
    schema = routing.schema()
    assert "tenant_id" not in schema["properties"]
    assert "tenant_id" not in schema.get("required", [])
    current = routing.default("acme:prod", "gateway_agent")
    assert "tenant_id" not in routing.form_value(current)
    with pytest.raises(ConfigValueError) as raised:
        routing.from_form({"tenant_id": "other:tenant"}, current)
    assert raised.value.errors == ["unknown field tenant_id"]


def test_a_secret_is_kept_when_null_set_when_given_and_cleared_when_empty():
    system = CONFIG_SECTIONS["system"]
    current = system.from_form({"llm_api_key": "k1"}, system.default("", ""))
    assert system.form_value(current)["llm_api_key"] is None
    assert system.secrets_set(current) == {"llm_api_key": True}
    assert system.from_form({"llm_api_key": None}, current).llm_api_key == "k1"
    assert system.from_form({}, current).llm_api_key == "k1"
    assert system.from_form({"llm_api_key": "k2"}, current).llm_api_key == "k2"
    assert system.from_form({"llm_api_key": ""}, current).llm_api_key is None
    assert system.dump(current, "")["llm_api_key"] == "k1"


def test_sections_are_found_by_where_they_are_stored():
    assert section_for(ConfigScope.ROUTING, "routing_config").name == "routing"
    assert section_for(ConfigScope.AGENT, "agent_config").name == "agent"
    assert section_for(ConfigScope.BACKEND, "backend_config") is None


def test_the_agent_section_names_its_entries_by_service():
    agent = CONFIG_SECTIONS["agent"]
    assert agent.entry_service("search_agent") == "search_agent"
    with pytest.raises(ConfigValueError):
        agent.entry_service(None)
    routing = CONFIG_SECTIONS["routing"]
    assert routing.entry_service(None) == "gateway_agent"
    with pytest.raises(ConfigValueError) as raised:
        routing.entry_service("other")
    assert raised.value.errors == [
        "section routing is stored under service gateway_agent"
    ]


def test_the_routing_form_offers_the_tenants_promotion_gate_values():
    routing = CONFIG_SECTIONS["routing"]
    properties = routing.schema()["properties"]
    assert {
        name: (properties[name]["anyOf"], properties[name]["default"])
        for name in ("optimization_gap_share", "optimization_skip_above_score")
    } == {
        name: (
            [
                {"exclusiveMinimum": 0.0, "maximum": 1.0, "type": "number"},
                {"type": "null"},
            ],
            None,
        )
        for name in ("optimization_gap_share", "optimization_skip_above_score")
    }

    current = routing.default("acme:prod", "gateway_agent")
    assert (
        routing.form_value(current)["optimization_gap_share"],
        routing.form_value(current)["optimization_skip_above_score"],
    ) == (None, None)

    saved = routing.from_form(
        {"optimization_gap_share": 0.25, "optimization_skip_above_score": 1.0},
        current,
    )
    assert (saved.optimization_gap_share, saved.optimization_skip_above_score) == (
        0.25,
        1.0,
    )
    stored = routing.dump(saved, "acme:prod")
    assert (
        stored["optimization_gap_share"],
        stored["optimization_skip_above_score"],
    ) == (0.25, 1.0)
    assert routing.load(stored) == saved
    # A form that leaves them out keeps them; null hands them back to the
    # deployment.
    assert routing.from_form({"min_unique_queries": 4}, saved) == routing.load(
        {**stored, "min_unique_queries": 4}
    )
    cleared = routing.from_form(
        {"optimization_gap_share": None, "optimization_skip_above_score": None},
        saved,
    )
    assert (cleared.optimization_gap_share, cleared.optimization_skip_above_score) == (
        None,
        None,
    )


@pytest.mark.parametrize(
    ("value", "errors"),
    [
        (
            {"optimization_gap_share": 0},
            ["optimization_gap_share: Input should be greater than 0"],
        ),
        (
            {"optimization_gap_share": 1.01},
            ["optimization_gap_share: Input should be less than or equal to 1"],
        ),
        (
            {"optimization_skip_above_score": 0},
            ["optimization_skip_above_score: Input should be greater than 0"],
        ),
        (
            {"optimization_skip_above_score": 1.5},
            ["optimization_skip_above_score: Input should be less than or equal to 1"],
        ),
    ],
)
def test_the_routing_form_refuses_a_gate_value_outside_its_range(value, errors):
    routing = CONFIG_SECTIONS["routing"]
    with pytest.raises(ConfigValueError) as raised:
        routing.from_form(value, routing.default("acme:prod", "gateway_agent"))
    assert raised.value.errors == errors


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "optimization_gap_share",
            0.0,
            "optimization_gap_share must be in (0, 1.0]; got 0.0",
        ),
        (
            "optimization_gap_share",
            1.5,
            "optimization_gap_share must be in (0, 1.0]; got 1.5",
        ),
        (
            "optimization_skip_above_score",
            -0.1,
            "optimization_skip_above_score must be in (0, 1.0]; got -0.1",
        ),
        (
            "optimization_skip_above_score",
            1.01,
            "optimization_skip_above_score must be in (0, 1.0]; got 1.01",
        ),
    ],
)
def test_a_stored_gate_value_outside_its_range_does_not_load(field, value, message):
    from cogniverse_foundation.config.unified_config import RoutingConfigUnified

    with pytest.raises(ValueError) as raised:
        RoutingConfigUnified.from_dict({"tenant_id": "acme:prod", field: value})
    assert str(raised.value) == message
