"""
Configuration validation schemas using JSON Schema.
Validates configuration before persistence to ensure data integrity.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


# JSON Schema definitions for each configuration type
SYSTEM_CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["tenant_id", "search_backend"],
    "properties": {
        "tenant_id": {"type": "string", "minLength": 1},
        "routing_agent_url": {"type": "string", "format": "uri"},
        "video_agent_url": {"type": "string", "format": "uri"},
        "text_agent_url": {"type": "string", "format": "uri"},
        "summarizer_agent_url": {"type": "string", "format": "uri"},
        "text_analysis_agent_url": {"type": "string", "format": "uri"},
        "search_backend": {"type": "string", "enum": ["vespa", "elasticsearch"]},
        "vespa_url": {"type": "string"},
        "vespa_port": {"type": "integer", "minimum": 1, "maximum": 65535},
        "elasticsearch_url": {"type": ["string", "null"], "format": "uri"},
        "llm_model": {"type": "string", "minLength": 1},
        "ollama_base_url": {"type": "string", "format": "uri"},
        "llm_api_key": {"type": ["string", "null"]},
        "phoenix_url": {"type": "string", "format": "uri"},
        "phoenix_collector_endpoint": {"type": "string"},
        "video_processing_profiles": {"type": "array", "items": {"type": "string"}},
        "environment": {
            "type": "string",
            "enum": ["development", "staging", "production"],
        },
        "metadata": {"type": "object"},
    },
}

ROUTING_CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["tenant_id", "routing_mode"],
    "properties": {
        "tenant_id": {"type": "string", "minLength": 1},
        "routing_mode": {
            "type": "string",
            "enum": ["tiered", "ensemble", "hybrid", "single"],
        },
        "enable_fast_path": {"type": "boolean"},
        "enable_slow_path": {"type": "boolean"},
        "enable_fallback": {"type": "boolean"},
        "fast_path_confidence_threshold": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "slow_path_confidence_threshold": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "max_routing_time_ms": {"type": "integer", "minimum": 1},
        "gliner_model": {"type": "string", "minLength": 1},
        "gliner_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "gliner_device": {"type": "string", "enum": ["cpu", "cuda", "mps"]},
        "gliner_labels": {"type": "array", "items": {"type": "string"}},
        "llm_provider": {"type": "string", "enum": ["local", "modal", "langextract"]},
        "llm_routing_model": {"type": "string", "minLength": 1},
        "llm_endpoint": {"type": "string", "format": "uri"},
        "llm_temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
        "llm_max_tokens": {"type": "integer", "minimum": 1},
        "use_chain_of_thought": {"type": "boolean"},
        "enable_auto_optimization": {"type": "boolean"},
        "optimization_interval_seconds": {"type": "integer", "minimum": 1},
        "min_samples_for_optimization": {"type": "integer", "minimum": 1},
        "dspy_enabled": {"type": "boolean"},
        "dspy_max_bootstrapped_demos": {"type": "integer", "minimum": 1},
        "dspy_max_labeled_demos": {"type": "integer", "minimum": 1},
        "enable_caching": {"type": "boolean"},
        "cache_ttl_seconds": {"type": "integer", "minimum": 0},
        "max_cache_size": {"type": "integer", "minimum": 1},
        "metadata": {"type": "object"},
    },
}

TELEMETRY_CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["tenant_id", "service_name"],
    "properties": {
        "tenant_id": {"type": "string", "minLength": 1},
        "enabled": {"type": "boolean"},
        "level": {
            "type": "string",
            "enum": ["disabled", "basic", "detailed", "verbose"],
        },
        "environment": {
            "type": "string",
            "enum": ["development", "staging", "production"],
        },
        "phoenix_enabled": {"type": "boolean"},
        "phoenix_endpoint": {"type": "string"},
        "phoenix_use_tls": {"type": "boolean"},
        "tenant_project_template": {"type": "string", "minLength": 1},
        "max_cached_tenants": {"type": "integer", "minimum": 1},
        "tenant_cache_ttl_seconds": {"type": "integer", "minimum": 1},
        "max_queue_size": {"type": "integer", "minimum": 1},
        "max_export_batch_size": {"type": "integer", "minimum": 1},
        "export_timeout_millis": {"type": "integer", "minimum": 1},
        "schedule_delay_millis": {"type": "integer", "minimum": 1},
        "use_sync_export": {"type": "boolean"},
        "service_name": {"type": "string", "minLength": 1},
        "service_version": {"type": "string", "minLength": 1},
        "metadata": {"type": "object"},
    },
}

AGENT_CONFIG_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": [
        "tenant_id",
        "agent_name",
        "agent_version",
        "agent_url",
        "capabilities",
        "module_config",
    ],
    "properties": {
        "tenant_id": {"type": "string", "minLength": 1},
        "agent_name": {"type": "string", "minLength": 1},
        "agent_version": {"type": "string", "minLength": 1},
        "agent_description": {"type": "string"},
        "agent_url": {"type": "string", "format": "uri"},
        "capabilities": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "skills": {"type": "array", "items": {"type": "object"}},
        "module_config": {
            "type": "object",
            "required": ["module_type", "signature"],
            "properties": {
                "module_type": {
                    "type": "string",
                    "enum": [
                        "predict",
                        "chain_of_thought",
                        "react",
                        "multi_chain_comparison",
                        "program_of_thought",
                    ],
                },
                "signature": {"type": "string", "minLength": 1},
                "max_retries": {"type": "integer", "minimum": 1},
                "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                "max_tokens": {"type": ["integer", "null"], "minimum": 1},
                "custom_params": {"type": "object"},
            },
        },
        "optimizer_config": {
            "type": ["object", "null"],
            "properties": {
                "optimizer_type": {
                    "type": "string",
                    "enum": [
                        "bootstrap_few_shot",
                        "labeled_few_shot",
                        "bootstrap_few_shot_with_random_search",
                        "copro",
                        "mipro_v2",
                    ],
                },
                "max_bootstrapped_demos": {"type": "integer", "minimum": 1},
                "max_labeled_demos": {"type": "integer", "minimum": 1},
                "num_trials": {"type": "integer", "minimum": 1},
                "metric": {"type": ["string", "null"]},
                "teacher_settings": {"type": "object"},
                "custom_params": {"type": "object"},
            },
        },
        "llm_model": {"type": "string", "minLength": 1},
        "llm_base_url": {"type": ["string", "null"], "format": "uri"},
        "llm_api_key": {"type": ["string", "null"]},
        "llm_temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
        "llm_max_tokens": {"type": ["integer", "null"], "minimum": 1},
        "thinking_enabled": {"type": "boolean"},
        "visual_analysis_enabled": {"type": "boolean"},
        "max_processing_time": {"type": "integer", "minimum": 1},
        "metadata": {"type": "object"},
    },
}


class ConfigValidator:
    """Configuration validator using JSON Schema"""

    SCHEMAS = {
        "system": SYSTEM_CONFIG_SCHEMA,
        "routing": ROUTING_CONFIG_SCHEMA,
        "telemetry": TELEMETRY_CONFIG_SCHEMA,
        "agent": AGENT_CONFIG_SCHEMA,
    }

    @staticmethod
    def validate(config_type: str, config_data: Dict[str, Any]) -> bool:
        """
        Validate configuration against schema.

        Args:
            config_type: Type of configuration (system, routing, telemetry, agent)
            config_data: Configuration data to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails with details
        """
        try:
            import jsonschema
        except ImportError:
            logger.warning(
                "jsonschema not installed, skipping validation. "
                "Install with: uv add jsonschema"
            )
            return True

        schema = ConfigValidator.SCHEMAS.get(config_type)
        if not schema:
            raise ValueError(f"Unknown config type: {config_type}")

        try:
            jsonschema.validate(instance=config_data, schema=schema)
            logger.debug(f"Config validation passed for {config_type}")
            return True
        except jsonschema.ValidationError as e:
            raise ValueError(
                f"Config validation failed for {config_type}: {e.message}\n"
                f"Path: {'.'.join(str(p) for p in e.path)}\n"
                f"Schema path: {'.'.join(str(p) for p in e.schema_path)}"
            )
        except jsonschema.SchemaError as e:
            raise ValueError(f"Invalid schema for {config_type}: {e.message}")

    @classmethod
    def validate_system_config(cls, config_data: Dict[str, Any]) -> bool:
        """Validate system configuration"""
        return cls.validate("system", config_data)

    @classmethod
    def validate_routing_config(cls, config_data: Dict[str, Any]) -> bool:
        """Validate routing configuration"""
        return cls.validate("routing", config_data)

    @classmethod
    def validate_telemetry_config(cls, config_data: Dict[str, Any]) -> bool:
        """Validate telemetry configuration"""
        return cls.validate("telemetry", config_data)

    @classmethod
    def validate_agent_config(cls, config_data: Dict[str, Any]) -> bool:
        """Validate agent configuration"""
        return cls.validate("agent", config_data)
