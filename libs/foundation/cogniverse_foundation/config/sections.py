"""The configuration sections an operator edits as forms.

Each section is one config dataclass and where it is stored. Its form is the
dataclass's JSON schema, so a field added to the dataclass appears in the form
without client changes. A section's value moves between three shapes: the
stored dict (what ``ConfigManager`` reads and writes), the dataclass, and the
form value (the dataclass dumped as JSON, without the fields the form must not
change and with secrets withheld).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import TypeAdapter, ValidationError

from cogniverse_foundation.config.agent_config import (
    AgentConfig,
    DSPyModuleType,
    ModuleConfig,
)
from cogniverse_foundation.config.unified_config import (
    AgentConfigUnified,
    DurableExecutionConfig,
    RoutingConfigUnified,
    SystemConfig,
)
from cogniverse_foundation.telemetry.config import TelemetryConfig
from cogniverse_sdk.interfaces.config_store import ConfigScope

# The tenant id system-wide configs are stored under.
SYSTEM_CONFIG_TENANT = "_system"

# The values a form offers for the fields that take one of a fixed set.
SEARCH_BACKENDS = ("vespa",)
ENVIRONMENTS = ("development", "staging", "production")
ROUTING_MODES = ("tiered", "direct", "adaptive")
PORT_RANGE = (1, 65535)


def _telemetry_providers() -> List[str]:
    from cogniverse_foundation.telemetry.registry import get_telemetry_registry

    return sorted(get_telemetry_registry().list_available())


class ConfigValueError(ValueError):
    """A form value the section's dataclass does not accept; ``errors`` names
    each problem by its field path."""

    def __init__(self, errors: List[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class ConfigSection:
    """One editable config: its dataclass and its storage location.

    ``service`` is the fixed service the config is stored under, or None
    when each entry of the section is its own service (one per agent).
    ``fixed_fields`` are set by the location, not the form (the tenant id a
    tenant's config carries). ``choices`` gives, per top-level field, the
    values it takes; ``ranges`` the inclusive bounds of an integer field. A
    stored value outside them is kept when a save leaves it unchanged.
    """

    name: str
    title: str
    scope: ConfigScope
    config_key: str
    service: Optional[str]
    tenant_scoped: bool
    model: type
    load: Callable[[Dict[str, Any]], Any]
    dump: Callable[[Any, str], Dict[str, Any]]
    default: Callable[[str, str], Any]
    secret_fields: frozenset = frozenset()
    fixed_fields: frozenset = frozenset()
    choices: Dict[str, Callable[[], Sequence[str]]] = field(default_factory=dict)
    ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    @property
    def _adapter(self) -> TypeAdapter:
        return TypeAdapter(self.model)

    def schema(self) -> Dict[str, Any]:
        """The form's JSON schema: the dataclass's, without the fixed fields,
        with secrets marked write-only."""
        schema = copy.deepcopy(self._adapter.json_schema())
        properties = schema.get("properties", {})
        for name in self.fixed_fields:
            properties.pop(name, None)
        if "required" in schema:
            schema["required"] = [
                name for name in schema["required"] if name not in self.fixed_fields
            ]
        for name in self.secret_fields:
            properties[name] = {**properties[name], "writeOnly": True}
        for name, options in self.choices.items():
            properties[name] = _with_enum(properties[name], list(options()))
        for name, (low, high) in self.ranges.items():
            properties[name] = {**properties[name], "minimum": low, "maximum": high}
        return schema

    def form_value(self, config: Any) -> Dict[str, Any]:
        """``config`` as the form shows it; secrets are withheld as null."""
        value = self._adapter.dump_python(config, mode="json")
        for name in self.fixed_fields:
            value.pop(name, None)
        for name in self.secret_fields:
            value[name] = None
        return value

    def secrets_set(self, config: Any) -> Dict[str, bool]:
        """Which of ``config``'s secrets hold a value."""
        return {
            name: bool(getattr(config, name)) for name in sorted(self.secret_fields)
        }

    def from_form(self, value: Dict[str, Any], current: Any) -> Any:
        """``current`` with a submitted form value's fields applied.

        A field the value leaves out keeps ``current``'s, so a form built
        from an older schema never resets the fields it does not show. A
        secret left null keeps ``current``'s; an empty string clears it.
        Fixed fields always come from ``current``. Raises ``ConfigValueError``
        naming every key the schema does not know and every value the
        dataclass refuses.
        """
        if not isinstance(value, dict):
            raise ConfigValueError(["the value must be a JSON object"])
        unknown = _unknown_keys(self.schema(), value, "")
        if unknown:
            raise ConfigValueError([f"unknown field {path}" for path in unknown])
        current_value = self._adapter.dump_python(current, mode="json")
        merged = {**current_value, **value}
        for name in self.secret_fields:
            submitted = value.get(name)
            if submitted is None:
                merged[name] = current_value[name]
            elif submitted == "":
                merged[name] = None
        for name in self.fixed_fields:
            merged[name] = current_value[name]
        refused = self._outside_choices_and_ranges(value, current_value)
        if refused:
            raise ConfigValueError(refused)
        try:
            return self._adapter.validate_python(merged)
        except ValidationError as exc:
            raise ConfigValueError(
                [
                    f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                    for error in exc.errors()
                ]
            ) from exc

    def _outside_choices_and_ranges(
        self, value: Dict[str, Any], current_value: Dict[str, Any]
    ) -> List[str]:
        """A problem per submitted field whose new value is not one of its
        choices or lies outside its range."""
        problems = []
        for name, options in self.choices.items():
            submitted = value.get(name, current_value.get(name))
            allowed = list(options())
            if (
                submitted is not None
                and submitted != current_value.get(name)
                and submitted not in allowed
            ):
                problems.append(f"{name}: must be one of {', '.join(allowed)}")
        for name, (low, high) in self.ranges.items():
            submitted = value.get(name, current_value.get(name))
            if (
                isinstance(submitted, int)
                and submitted != current_value.get(name)
                and not low <= submitted <= high
            ):
                problems.append(f"{name}: must be between {low} and {high}")
        return problems

    def entry_service(self, service: Optional[str]) -> str:
        """The service an entry of this section is stored under."""
        if self.service is not None:
            if service not in (None, self.service):
                raise ConfigValueError(
                    [f"section {self.name} is stored under service {self.service}"]
                )
            return self.service
        if not service:
            raise ConfigValueError([f"section {self.name} needs a service name"])
        return service


def _with_enum(property_schema: Dict[str, Any], options: List[str]) -> Dict[str, Any]:
    """``property_schema`` restricted to ``options``; a nullable string stays
    nullable."""
    branches = property_schema.get("anyOf")
    if branches:
        return {
            **property_schema,
            "anyOf": [
                {**branch, "enum": options}
                if branch.get("type") == "string"
                else branch
                for branch in branches
            ],
        }
    return {**property_schema, "enum": options}


def _unknown_keys(schema: Dict[str, Any], value: Any, path: str) -> List[str]:
    """Paths of ``value``'s object keys that ``schema`` declares no property
    for, following ``$ref``s, ``anyOf`` branches and object-valued fields."""
    defs = schema.get("$defs", {})

    def resolve(node: Dict[str, Any]) -> Dict[str, Any]:
        while "$ref" in node:
            node = defs[node["$ref"].rsplit("/", 1)[-1]]
        return node

    def walk(node: Dict[str, Any], item: Any, at: str) -> List[str]:
        node = resolve(node)
        if "anyOf" in node:
            objects = [resolve(b) for b in node["anyOf"] if "properties" in resolve(b)]
            return (
                walk(objects[0], item, at) if objects and isinstance(item, dict) else []
            )
        properties = node.get("properties")
        if properties is None or not isinstance(item, dict):
            return []
        found = []
        for key, child in item.items():
            child_path = f"{at}.{key}" if at else key
            if key not in properties:
                found.append(child_path)
            else:
                found.extend(walk(properties[key], child, child_path))
        return found

    return walk(schema, value, path)


def _agent_load(stored: Dict[str, Any]) -> AgentConfig:
    return AgentConfigUnified.from_dict(stored).agent_config


def _agent_dump(config: AgentConfig, tenant_id: str) -> Dict[str, Any]:
    return AgentConfigUnified(tenant_id=tenant_id, agent_config=config).to_dict(
        redact=False
    )


def _agent_default(tenant_id: str, service: str) -> AgentConfig:
    return AgentConfig(
        agent_name=service,
        agent_version="1.0.0",
        agent_description="",
        agent_url="",
        capabilities=[],
        skills=[],
        module_config=ModuleConfig(module_type=DSPyModuleType.PREDICT, signature=""),
    )


def _with_tenant(config: Any, tenant_id: str) -> Dict[str, Any]:
    config.tenant_id = tenant_id
    return config.to_dict()


CONFIG_SECTIONS: Dict[str, ConfigSection] = {
    section.name: section
    for section in (
        ConfigSection(
            name="system",
            title="System",
            scope=ConfigScope.SYSTEM,
            config_key="system_config",
            service="system",
            tenant_scoped=False,
            model=SystemConfig,
            load=SystemConfig.from_dict,
            dump=lambda config, tenant_id: config.to_dict(redact=False),
            default=lambda tenant_id, service: SystemConfig(),
            secret_fields=frozenset({"llm_api_key"}),
            choices={
                "search_backend": lambda: SEARCH_BACKENDS,
                "environment": lambda: ENVIRONMENTS,
            },
            ranges={"backend_port": PORT_RANGE},
        ),
        ConfigSection(
            name="routing",
            title="Routing",
            scope=ConfigScope.ROUTING,
            config_key="routing_config",
            service="gateway_agent",
            tenant_scoped=True,
            model=RoutingConfigUnified,
            load=RoutingConfigUnified.from_dict,
            dump=_with_tenant,
            default=lambda tenant_id, service: RoutingConfigUnified(
                tenant_id=tenant_id
            ),
            fixed_fields=frozenset({"tenant_id"}),
            choices={"routing_mode": lambda: ROUTING_MODES},
        ),
        ConfigSection(
            name="telemetry",
            title="Telemetry",
            scope=ConfigScope.TELEMETRY,
            config_key="telemetry_config",
            service="telemetry",
            tenant_scoped=True,
            model=TelemetryConfig,
            load=TelemetryConfig.from_dict,
            dump=lambda config, tenant_id: config.to_dict(),
            default=lambda tenant_id, service: TelemetryConfig(),
            choices={"provider": _telemetry_providers},
        ),
        ConfigSection(
            name="agent",
            title="Agents",
            scope=ConfigScope.AGENT,
            config_key="agent_config",
            service=None,
            tenant_scoped=True,
            model=AgentConfig,
            load=_agent_load,
            dump=_agent_dump,
            default=_agent_default,
            secret_fields=frozenset({"llm_api_key"}),
        ),
        ConfigSection(
            name="durable_execution",
            title="Durable execution",
            scope=ConfigScope.DURABLE,
            config_key="durable_execution_config",
            service="optimization",
            tenant_scoped=True,
            model=DurableExecutionConfig,
            load=DurableExecutionConfig.from_dict,
            dump=_with_tenant,
            default=lambda tenant_id, service: DurableExecutionConfig(
                tenant_id=tenant_id
            ),
            fixed_fields=frozenset({"tenant_id"}),
        ),
    )
}


def section_for(scope: ConfigScope, config_key: str) -> Optional[ConfigSection]:
    """The section stored at ``scope`` under ``config_key``, if any."""
    for section in CONFIG_SECTIONS.values():
        if section.scope == scope and section.config_key == config_key:
            return section
    return None
