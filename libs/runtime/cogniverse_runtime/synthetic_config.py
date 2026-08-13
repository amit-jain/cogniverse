"""Strict runtime configuration for synthetic data generation."""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from cogniverse_foundation.config.unified_config import (
    BackendConfig,
    SyntheticGeneratorConfig,
)

logger = logging.getLogger(__name__)

_MISSING = object()
_PROFILE_REQUIRED_FIELDS = ("type", "schema_name")
_PROFILE_TYPED_FIELDS: dict[str, type] = {
    "description": str,
    "embedding_model": str,
    "embedding_type": str,
    "model_loader": str,
    "model_specific": dict,
    "pipeline_config": dict,
    "process_type": str,
    "schema_config": dict,
    "schema_name": str,
    "strategies": dict,
    "type": str,
}
_AGENT_KEYS = {"capabilities", "enabled", "modalities", "timeout", "url"}


@dataclass(frozen=True)
class SyntheticRuntimeConfig:
    """Fully validated configuration passed to ``SyntheticDataService``."""

    backend_config: BackendConfig
    backend_default_profiles: dict[str, dict[str, str]]
    generator_config: SyntheticGeneratorConfig
    agents_config: dict[str, dict[str, Any]]


def _require_section(config: Any, name: str, tenant_id: str) -> dict[str, Any]:
    getter = getattr(config, "get", None)
    if not callable(getter):
        raise ValueError(
            f"Synthetic runtime configuration for tenant={tenant_id!r} "
            f"requires object section {name!r}"
        )
    value = getter(name, _MISSING)
    if value is _MISSING:
        raise ValueError(
            f"Synthetic runtime configuration for tenant={tenant_id!r} "
            f"requires object section {name!r}"
        )
    if not isinstance(value, dict):
        raise ValueError(
            f"Synthetic runtime configuration for tenant={tenant_id!r} "
            f"section {name!r} must be an object, got {type(value).__name__}"
        )
    if not value:
        raise ValueError(
            f"Synthetic runtime configuration for tenant={tenant_id!r} "
            f"section {name!r} must not be empty"
        )
    return copy.deepcopy(value)


def _require_keys(
    value: dict[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    source: str,
) -> None:
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing or unknown:
        raise ValueError(
            f"{source} has invalid keys: missing={missing} unknown={unknown}"
        )


def _require_object(value: Any, source: str, *, nonempty: bool = False) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{source} must be an object")
    if nonempty and not value:
        raise ValueError(f"{source} must be a non-empty object")
    return value


def _require_string(value: Any, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source} must be a non-empty string")
    return value


def _require_string_list(value: Any, source: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"{source} must be a list of non-empty strings")
    return value


def _validate_backend(
    raw: dict[str, Any], tenant_id: str
) -> tuple[BackendConfig, dict[str, dict[str, str]]]:
    _require_keys(
        raw,
        required={"type", "url", "port", "profiles"},
        optional={"metadata", "default_profiles", "tenant_id"},
        source="backend",
    )
    hydrated_tenant_id = raw.get("tenant_id")
    if hydrated_tenant_id is not None:
        _require_string(hydrated_tenant_id, "backend.tenant_id")
        if hydrated_tenant_id != tenant_id:
            raise ValueError(
                f"backend.tenant_id must equal {tenant_id!r}, "
                f"got {hydrated_tenant_id!r}"
            )
    _require_string(raw["type"], "backend.type")
    _require_string(raw["url"], "backend.url")
    port = raw["port"]
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("backend.port must be an integer between 1 and 65535")
    profiles = _require_object(raw["profiles"], "backend.profiles", nonempty=True)
    if "metadata" in raw:
        _require_object(raw["metadata"], "backend.metadata")
    for profile_name, profile in profiles.items():
        _require_string(profile_name, "backend profile name")
        profile = _require_object(
            profile,
            f"backend.profiles.{profile_name}",
            nonempty=True,
        )
        for field_name in _PROFILE_REQUIRED_FIELDS:
            if field_name not in profile:
                raise ValueError(
                    f"backend.profiles.{profile_name} is missing required "
                    f"key {field_name!r}"
                )
        for field_name, expected_type in _PROFILE_TYPED_FIELDS.items():
            field_value = profile.get(field_name, _MISSING)
            if field_value is _MISSING or field_value is None:
                continue
            if not isinstance(field_value, expected_type):
                raise ValueError(
                    f"backend.profiles.{profile_name}.{field_name} must be "
                    f"{expected_type.__name__}, got {type(field_value).__name__}"
                )
        _require_string(profile.get("type"), f"backend.profiles.{profile_name}.type")
        _require_string(
            profile.get("schema_name"),
            f"backend.profiles.{profile_name}.schema_name",
        )

    default_profiles = _require_object(
        raw.get("default_profiles", {}),
        "backend.default_profiles",
    )
    for modality, selection in default_profiles.items():
        _require_string(modality, "backend default-profile modality")
        selection = _require_object(
            selection,
            f"backend.default_profiles.{modality}",
            nonempty=True,
        )
        _require_keys(
            selection,
            required={"profile"},
            optional={"strategy"},
            source=f"backend.default_profiles.{modality}",
        )
        profile_name = _require_string(
            selection["profile"],
            f"backend.default_profiles.{modality}.profile",
        )
        if profile_name not in profiles:
            raise ValueError(
                f"backend.default_profiles.{modality}.profile references "
                f"unknown profile {profile_name!r}"
            )
        if "strategy" in selection:
            _require_string(
                selection["strategy"],
                f"backend.default_profiles.{modality}.strategy",
            )

    raw["tenant_id"] = tenant_id
    return BackendConfig.from_dict(raw), copy.deepcopy(default_profiles)


def _validate_field_mappings(raw: Any) -> None:
    fields = _require_object(raw, "synthetic.field_mappings")
    _require_keys(
        fields,
        required={
            "topic_fields",
            "description_fields",
            "transcript_fields",
            "entity_fields",
            "temporal_fields",
            "metadata_fields",
        },
        source="synthetic.field_mappings",
    )
    for name in (
        "topic_fields",
        "description_fields",
        "transcript_fields",
        "entity_fields",
    ):
        _require_string_list(fields[name], f"synthetic.field_mappings.{name}")
    for name in ("temporal_fields", "metadata_fields"):
        mapping = _require_object(
            fields[name],
            f"synthetic.field_mappings.{name}",
        )
        if not all(
            isinstance(key, str)
            and key.strip()
            and isinstance(value, str)
            and value.strip()
            for key, value in mapping.items()
        ):
            raise ValueError(
                f"synthetic.field_mappings.{name} must map non-empty strings"
            )


def _validate_dspy_modules(raw: Any, optimizer_name: str) -> None:
    modules = _require_object(
        raw,
        f"synthetic.optimizer_configs.{optimizer_name}.dspy_modules",
        nonempty=True,
    )
    if "query_generator" not in modules:
        raise ValueError(
            f"synthetic optimizer {optimizer_name!r} requires query_generator"
        )
    for module_name, module in modules.items():
        _require_string(module_name, "synthetic DSPy module name")
        module = _require_object(
            module,
            f"synthetic.optimizer_configs.{optimizer_name}.dspy_modules.{module_name}",
        )
        _require_keys(
            module,
            required={"signature_class", "module_type", "lm_config", "metadata"},
            source=(
                f"synthetic.optimizer_configs.{optimizer_name}."
                f"dspy_modules.{module_name}"
            ),
        )
        _require_string(module["signature_class"], f"{module_name}.signature_class")
        _require_string(module["module_type"], f"{module_name}.module_type")
        _require_object(module["lm_config"], f"{module_name}.lm_config")
        _require_object(module["metadata"], f"{module_name}.metadata")


def _validate_optimizer_configs(raw: Any) -> None:
    optimizers = _require_object(
        raw,
        "synthetic.optimizer_configs",
        nonempty=True,
    )
    _require_keys(
        optimizers,
        required={
            "cross_modal",
            "entity_extraction",
            "modality",
            "profile",
            "query_enhancement",
            "routing",
            "unified",
            "workflow",
        },
        source="synthetic.optimizer_configs",
    )
    for optimizer_name, optimizer in optimizers.items():
        _require_string(optimizer_name, "synthetic optimizer name")
        optimizer = _require_object(
            optimizer,
            f"synthetic.optimizer_configs.{optimizer_name}",
        )
        required_fields = {
            "modality": {"optimizer_type", "agent_mappings"},
            "routing": {
                "optimizer_type",
                "dspy_modules",
                "profile_scoring_rules",
            },
            "profile": {"optimizer_type", "profile_scoring_rules"},
            "entity_extraction": {"optimizer_type", "profile_scoring_rules"},
            "cross_modal": {"optimizer_type", "profile_scoring_rules"},
            "query_enhancement": {"optimizer_type", "profile_scoring_rules"},
            "unified": {"optimizer_type", "profile_scoring_rules"},
            "workflow": {"optimizer_type", "profile_scoring_rules"},
        }[optimizer_name]
        _require_keys(
            optimizer,
            required=required_fields,
            source=f"synthetic.optimizer_configs.{optimizer_name}",
        )
        if optimizer["optimizer_type"] != optimizer_name:
            raise ValueError(
                f"synthetic optimizer key {optimizer_name!r} does not match "
                f"optimizer_type {optimizer['optimizer_type']!r}"
            )
        if optimizer_name == "routing":
            _validate_dspy_modules(optimizer["dspy_modules"], optimizer_name)
        if optimizer_name != "modality" and (
            not isinstance(optimizer["profile_scoring_rules"], list)
            or not optimizer["profile_scoring_rules"]
        ):
            raise ValueError(
                f"synthetic optimizer {optimizer_name!r} "
                "profile_scoring_rules must be a non-empty list"
            )
        if optimizer_name != "modality":
            continue
        mappings = optimizer["agent_mappings"]
        if not isinstance(mappings, list) or not mappings:
            raise ValueError(
                "synthetic optimizer 'modality' agent_mappings must not be empty"
            )
        for index, mapping in enumerate(mappings):
            mapping = _require_object(
                mapping,
                f"synthetic.optimizer_configs.{optimizer_name}.agent_mappings[{index}]",
            )
            _require_keys(
                mapping,
                required={"modality", "agent_name"},
                source=(
                    f"synthetic.optimizer_configs.{optimizer_name}."
                    f"agent_mappings[{index}]"
                ),
            )
            _require_string(mapping["modality"], f"agent_mappings[{index}].modality")
            _require_string(
                mapping["agent_name"], f"agent_mappings[{index}].agent_name"
            )


def _validate_synthetic(
    raw: dict[str, Any], tenant_id: str
) -> SyntheticGeneratorConfig:
    _require_keys(
        raw,
        required={
            "field_mappings",
            "optimizer_configs",
            "synthetic_generation_timeout_seconds",
        },
        source="synthetic",
    )
    _validate_field_mappings(raw["field_mappings"])
    _validate_optimizer_configs(raw["optimizer_configs"])
    raw["tenant_id"] = tenant_id
    return SyntheticGeneratorConfig.from_dict(raw)


def _validate_agents(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    active = {}
    for name, agent in raw.items():
        _require_string(name, "agent name")
        agent = _require_object(agent, f"agents.{name}", nonempty=True)
        _require_keys(
            agent,
            required={"enabled", "url", "capabilities"},
            optional=_AGENT_KEYS - {"enabled", "url", "capabilities"},
            source=f"agents.{name}",
        )
        url = _require_string(agent["url"], f"agents.{name}.url")
        parsed_url = urlparse(url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError(f"agents.{name}.url must be an absolute HTTP(S) URL")
        _require_string_list(agent["capabilities"], f"agents.{name}.capabilities")
        if "modalities" in agent:
            _require_string_list(agent["modalities"], f"agents.{name}.modalities")
        if "timeout" in agent:
            timeout = agent["timeout"]
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout)
                or timeout <= 0
            ):
                raise ValueError(
                    f"agents.{name}.timeout must be a finite positive number"
                )
        enabled = agent.get("enabled", _MISSING)
        if not isinstance(enabled, bool):
            raise ValueError(f"agents.{name}.enabled must be a boolean")
        if enabled:
            active[name] = agent
    if not active:
        raise ValueError("agents configuration has no enabled agents")
    return active


def parse_synthetic_runtime_config(
    config: Any,
    *,
    tenant_id: str,
    loaded_agent_names: Collection[str] | None = None,
) -> SyntheticRuntimeConfig:
    """Parse and validate the three required synthetic runtime sections."""
    from cogniverse_foundation.common.tenant_utils import require_tenant_id

    tenant_id = require_tenant_id(tenant_id, source="synthetic runtime configuration")
    backend_raw = _require_section(config, "backend", tenant_id)
    synthetic_raw = _require_section(config, "synthetic", tenant_id)
    agents_raw = _require_section(config, "agents", tenant_id)

    try:
        agents_config = _validate_agents(agents_raw)
        if loaded_agent_names is not None:
            loaded = set(loaded_agent_names)
            unloaded = sorted(agents_config.keys() - loaded)
            if unloaded:
                raise ValueError(
                    "enabled agents were not loaded: " + ", ".join(unloaded)
                )

        backend_config, backend_default_profiles = _validate_backend(
            backend_raw, tenant_id
        )
        generator_config = _validate_synthetic(synthetic_raw, tenant_id)

        from cogniverse_synthetic.utils import (
            AgentInferrer,
            partition_profiles_by_sampleability,
            profile_modality,
        )

        modality_config = generator_config.get_optimizer_config("modality")
        if modality_config is None:
            raise ValueError("synthetic optimizer 'modality' was not hydrated")
        inferrer = AgentInferrer(
            agents_config=agents_raw,
            agent_mappings=modality_config.agent_mappings,
        )
        sampleable, internal = partition_profiles_by_sampleability(
            backend_config.profiles
        )
        if not sampleable:
            raise ValueError(
                "no backend profile has a synthetic-sampleable modality, got "
                + str(sorted({p.type for p in backend_config.profiles.values()}))
            )
        if internal:
            logger.warning(
                "Synthetic skips internal backend profiles: %s",
                ", ".join(sorted(internal)),
            )
        inferrer.require_mappings(
            {
                modality
                for modality in (
                    profile_modality(profile) for profile in sampleable.values()
                )
                if modality is not None
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid synthetic runtime configuration for tenant={tenant_id!r}: {exc}"
        ) from exc

    return SyntheticRuntimeConfig(
        backend_config=backend_config,
        backend_default_profiles=backend_default_profiles,
        generator_config=generator_config,
        agents_config=agents_config,
    )
