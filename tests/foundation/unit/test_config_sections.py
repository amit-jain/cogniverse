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
