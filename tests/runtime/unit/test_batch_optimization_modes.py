"""Unit tests for optimization_cli batch modes: simba, workflow, gateway-thresholds, profile.

Tests:
1. CLI argument parser recognizes all new modes
2. Each optimization function handles empty span data gracefully
3. Each function produces expected artifact types when given mock span data
"""

import asyncio
import hashlib
import json
import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pandas as pd
import pytest

from cogniverse_runtime.optimization_cli import build_parser
from cogniverse_sdk.interfaces.workflow_store import WorkflowLearningState

# Patch targets: these are imported locally inside each function,
# so we patch at the source module.
_PATCH_CONFIG = "cogniverse_foundation.config.utils.create_default_config_manager"
_PATCH_TELEMETRY = "cogniverse_foundation.telemetry.manager.get_telemetry_manager"


def _selection_block(
    pool: int,
    deduped: int,
    *,
    cap: int = 300,
    mmr_applied: bool = False,
    decayed_count: int = 0,
    decayed_example_ids: list[str] | None = None,
) -> dict[str, dict[str, int | bool | list[str]]]:
    return {
        "selection": {
            "pool": pool,
            "deduped": deduped,
            "cap": cap,
            "mmr_applied": mmr_applied,
            "decayed_count": decayed_count,
            "decayed_example_ids": decayed_example_ids or [],
        }
    }


def _fake_bootstrap_block(trainset: int) -> dict:
    """The bootstrap report for a teleprompter that never calls the metric."""
    return {
        "trainset": trainset,
        "max_bootstrapped_demos": 4,
        "max_labeled_demos": 8,
        "max_rounds": 1,
        "metric_threshold": 1.0,
        "attempts": 0,
        "errors": 0,
        "examples_walked": 0,
        "accepted": 0,
        "bootstrapped_demos": 0,
        "labeled_demos": 0,
        "metric_values": [],
    }


def _training_selection_config_manager(
    tenant_id: str,
    training_selection: dict[str, dict[str, float]],
):
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import RoutingConfigUnified
    from tests.utils.memory_store import InMemoryConfigStore

    manager = ConfigManager(store=InMemoryConfigStore())
    manager.set_routing_config(
        RoutingConfigUnified(
            tenant_id=tenant_id,
            training_selection=training_selection,
        )
    )
    return manager


def _signed_approved_record(record: dict[str, Any]) -> dict[str, Any]:
    signed = {
        "confidence": 0.9,
        "created_at": "2026-08-05T01:00:00+00:00",
        "reviewed_at": "2026-08-05T01:01:00+00:00",
        **record,
    }
    decision = signed.get("metadata.decision")
    decision_intent = dict(decision) if isinstance(decision, dict) else decision
    if isinstance(decision_intent, dict):
        decision_intent.pop("timestamp", None)
    identity = {
        "item_id": signed.get("item_id"),
        "status": signed.get("status"),
        "decision": decision_intent,
    }
    signed["metadata.approval_decision_sha256"] = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    signed["metadata.approval_decision_timestamp"] = signed["reviewed_at"]
    canonical_json = json.dumps(
        signed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    signed["metadata.approval_record_json"] = canonical_json
    signed["metadata.approval_record_sha256"] = hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()
    return signed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeTelemetryConfig:
    """Minimal config with get_project_name."""

    def get_project_name(self, tenant_id: str, service: Optional[str] = None) -> str:
        if service:
            return f"cogniverse-{tenant_id}-{service}"
        return f"cogniverse-{tenant_id}"


class FakeTraceStore:
    """In-memory trace store returning canned DataFrames."""

    def __init__(self, spans_df: pd.DataFrame | None = None):
        self._spans_df = spans_df if spans_df is not None else pd.DataFrame()
        self.calls: List[Dict[str, Any]] = []

    async def get_all_spans(self, **kwargs) -> pd.DataFrame:
        self.calls.append(kwargs)
        return self._spans_df.copy(deep=True)


class FakeDatasetStore:
    """Records calls to create_dataset, delete_dataset, and get_dataset."""

    def __init__(self):
        self.created: List[Dict[str, Any]] = []
        self.deleted: List[str] = []
        self.datasets: Dict[str, pd.DataFrame] = {}

    async def replace_dataset(self, name, data, metadata=None):
        return await self.create_dataset(name=name, data=data, metadata=metadata)

    async def create_dataset(self, name, data, metadata=None):
        self.created.append({"name": name, "data": data, "metadata": metadata})
        self.datasets[name] = data.copy(deep=True)
        return f"dataset-{len(self.created)}"

    async def delete_dataset(self, name) -> bool:
        # Blobs are last-write-wins: the artifact store deletes before create.
        self.deleted.append(name)
        return self.datasets.pop(name, None) is not None

    async def get_dataset(self, name):
        from cogniverse_foundation.telemetry.providers.base import (
            DatasetNotFoundError,
        )

        if name not in self.datasets:
            raise DatasetNotFoundError(f"No dataset {name}")
        return self.datasets[name].copy(deep=True)


class FakeTelemetryProvider:
    """Minimal TelemetryProvider stand-in with trace + dataset stores."""

    def __init__(self, spans_df: pd.DataFrame | None = None):
        self._trace_store = FakeTraceStore(spans_df)
        self._dataset_store = FakeDatasetStore()

    @property
    def traces(self):
        return self._trace_store

    @property
    def datasets(self):
        return self._dataset_store


class FakeTelemetryManager:
    def __init__(self, provider):
        self._provider = provider
        self.config = FakeTelemetryConfig()

    def get_provider(self, tenant_id):
        return self._provider


class FakeWorkflowStore:
    """Canonical in-memory workflow-state boundary for CLI unit tests."""

    def __init__(self):
        self.states = {}

    async def replace_learning_state(
        self, tenant_id, executions, profiles, patterns, templates
    ):
        self.states[tenant_id] = {
            "executions": list(executions),
            "profiles": list(profiles),
            "patterns": dict(patterns),
            "templates": list(templates),
        }

    def _state(self, tenant_id):
        return self.states.get(
            tenant_id,
            {"executions": [], "profiles": [], "patterns": {}, "templates": []},
        )

    async def load_learning_state(self, tenant_id):
        state = self._state(tenant_id)
        return WorkflowLearningState(
            executions=list(state["executions"]),
            profiles=list(state["profiles"]),
            patterns=dict(state["patterns"]),
            templates=list(state["templates"]),
        )


@pytest.fixture
def empty_provider():
    return FakeTelemetryProvider(spans_df=pd.DataFrame())


@pytest.fixture
def fake_telemetry_manager(empty_provider):
    return FakeTelemetryManager(empty_provider)


@contextmanager
def _patch_telemetry(fake_mgr):
    """Patch get_telemetry_manager at BOTH lookup sites: the source module
    (optimization_cli imports it at call time) and the orchestration evaluator
    (which binds it at module import, so the source patch doesn't reach it)."""
    with (
        patch(_PATCH_TELEMETRY, return_value=fake_mgr),
        patch(
            "cogniverse_agents.routing.orchestration_evaluator.get_telemetry_manager",
            return_value=fake_mgr,
        ),
    ):
        yield


_TEACHER_SERVED_MODEL = "test/teacher-model"


def _teacher_endpoint():
    """The concrete type production hands to the teacher probe."""
    from cogniverse_foundation.config.unified_config import LLMEndpointConfig

    return LLMEndpointConfig(
        model=f"openai/{_TEACHER_SERVED_MODEL}",
        api_base="http://teacher-svc:8000/v1",
    )


@pytest.fixture(autouse=True)
def _teacher_service_reports_configured_model(monkeypatch):
    """Report the configured teacher model so unit tests never dial a service."""
    import cogniverse_runtime.inference_health_check as inference_health_check

    monkeypatch.setattr(
        inference_health_check,
        "probe_service_model",
        lambda url, **kwargs: _TEACHER_SERVED_MODEL,
    )


def _patch_infra(fake_mgr, *, config_manager=None):
    """Return a combined context manager patching config + telemetry."""
    config_patch = (
        patch(_PATCH_CONFIG, return_value=config_manager)
        if config_manager is not None
        else patch(_PATCH_CONFIG)
    )
    return (
        config_patch,
        _patch_telemetry(fake_mgr),
    )


def _synthetic_runtime_sections(*, marker: str = "fixture") -> dict[str, Any]:
    """Return a complete strict configuration for one VIDEO profile."""
    return {
        "backend": {
            "type": "vespa",
            "url": f"http://vespa-{marker}.test",
            "port": 8080,
            "profiles": {
                "video_fixture": {
                    "type": "video",
                    "description": f"{marker} video profile",
                    "schema_name": f"video_{marker}",
                    "embedding_model": "TomoroAI/tomoro-colqwen3-embed-4b",
                    "pipeline_config": {"extract_keyframes": True},
                    "strategies": {},
                    "embedding_type": "multi_vector",
                    "schema_config": {"embedding_dim": 320},
                }
            },
            "default_profiles": {
                "video": {"profile": "video_fixture", "strategy": "segmentation"}
            },
            "metadata": {"marker": marker},
        },
        "synthetic": {
            "field_mappings": {
                "topic_fields": ["video_title"],
                "description_fields": ["segment_description"],
                "transcript_fields": ["audio_transcript"],
                "entity_fields": ["video_title"],
                "temporal_fields": {"start": "start_time", "end": "end_time"},
                "metadata_fields": {"source": "source_uri"},
            },
            "synthetic_generation_timeout_seconds": 300.0,
            "synthetic_generation_floor_count": 1,
            "optimizer_configs": {
                "modality": {
                    "optimizer_type": "modality",
                    "agent_mappings": [
                        {
                            "modality": "VIDEO",
                            "agent_name": "search_agent",
                        }
                    ],
                },
                "cross_modal": {
                    "optimizer_type": "cross_modal",
                    "profile_scoring_rules": [
                        {
                            "condition": {"field": "type", "equals": "video"},
                            "score_adjustment": 1.0,
                            "reason": "video cross-modal source",
                        }
                    ],
                },
                "entity_extraction": {
                    "optimizer_type": "entity_extraction",
                    "profile_scoring_rules": [
                        {
                            "condition": {"field": "type", "equals": "video"},
                            "score_adjustment": 2.0,
                            "reason": "video entity source",
                        }
                    ],
                },
                "profile": {
                    "optimizer_type": "profile",
                    "profile_scoring_rules": [
                        {
                            "condition": {"field": "type", "equals": "video"},
                            "score_adjustment": 1.5,
                            "reason": "video profile",
                        }
                    ],
                },
                "query_enhancement": {
                    "optimizer_type": "query_enhancement",
                    "profile_scoring_rules": [
                        {
                            "condition": {"field": "type", "equals": "video"},
                            "score_adjustment": 1.0,
                            "reason": "video query source",
                        }
                    ],
                },
                "routing": {
                    "optimizer_type": "routing",
                    "dspy_modules": {
                        "query_generator": {
                            "signature_class": (
                                "cogniverse_synthetic.dspy_signatures."
                                "GenerateEntityQuery"
                            ),
                            "module_type": "ChainOfThought",
                            "lm_config": {},
                            "metadata": {},
                        }
                    },
                    "profile_scoring_rules": [
                        {
                            "condition": {"field": "type", "equals": "video"},
                            "score_adjustment": 3.0,
                            "reason": "video routing source",
                        }
                    ],
                },
                "unified": {
                    "optimizer_type": "unified",
                    "profile_scoring_rules": [
                        {
                            "condition": {"field": "type", "equals": "video"},
                            "score_adjustment": 1.0,
                            "reason": "video unified source",
                        }
                    ],
                },
                "workflow": {
                    "optimizer_type": "workflow",
                    "profile_scoring_rules": [
                        {
                            "condition": {"field": "type", "equals": "video"},
                            "score_adjustment": 1.0,
                            "reason": "video workflow source",
                        }
                    ],
                },
            },
        },
        "agents": {
            "search_agent": {
                "enabled": True,
                "url": f"http://search-{marker}.test",
                "capabilities": ["search", "video_search"],
                "modalities": ["VIDEO"],
            },
            "disabled_video_agent": {
                "enabled": False,
                "url": "http://disabled.test",
                "capabilities": ["video_search"],
                "modalities": ["VIDEO"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Test: CLI argument parser recognizes all new modes
# ---------------------------------------------------------------------------


_REAL_MODES = [
    "cleanup",
    "triggered",
    "simba",
    "workflow",
    "gateway-thresholds",
    "online-routing-eval",
    "profile",
    "entity-extraction",
    "synthetic",
    "rollback",
    "ab-compare",
    "egress-netpol",
    "monthly-reports",
]


class TestCliArgumentParser:
    """Drive the REAL CLI parser (build_parser) so the test can't drift from
    production the way the old hand-built parser had (it listed a phantom
    'routing' mode, omitted 5 real modes, and used a wrong tenant default)."""

    @pytest.fixture
    def parser(self):
        return build_parser()

    @pytest.mark.parametrize("mode", _REAL_MODES)
    def test_real_mode_accepted(self, parser, mode):
        assert parser.parse_args(["--mode", mode]).mode == mode

    def test_online_routing_eval_is_a_mode(self, parser):
        assert (
            parser.parse_args(["--mode", "online-routing-eval"]).mode
            == "online-routing-eval"
        )

    def test_routing_is_not_a_mode(self, parser):
        # 'routing' is the router family, NOT an optimization CLI mode.
        with pytest.raises(SystemExit):
            parser.parse_args(["--mode", "routing"])

    def test_cleanup_tenant_defaults_to_none(self, parser):
        # cleanup + monthly-reports run globally; tenant_id default is None so
        # the no-tenant CronWorkflows don't exit 2 on argparse.
        assert parser.parse_args(["--mode", "cleanup"]).tenant_id is None

    def test_tenant_and_lookback_hours(self, parser):
        args = parser.parse_args(
            ["--mode", "simba", "--tenant-id", "acme:prod", "--lookback-hours", "48"]
        )
        assert args.tenant_id == "acme:prod"
        assert args.lookback_hours == 48.0

    def test_lookback_hours_default(self, parser):
        assert parser.parse_args(["--mode", "simba"]).lookback_hours == 24.0

    def test_invalid_mode_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["--mode", "nonexistent"])


class TestSyntheticRuntimeConfig:
    def test_hydrates_exact_canonical_objects_and_enabled_agents(self):
        from cogniverse_foundation.config.unified_config import (
            BackendConfig,
            SyntheticGeneratorConfig,
        )
        from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config

        sections = _synthetic_runtime_sections(marker="alpha")

        parsed = parse_synthetic_runtime_config(
            sections,
            tenant_id="acme:alpha",
            loaded_agent_names={"search_agent"},
        )

        assert isinstance(parsed.backend_config, BackendConfig)
        assert (
            parsed.backend_config.tenant_id,
            parsed.backend_config.backend_type,
            parsed.backend_config.url,
            parsed.backend_config.port,
            parsed.backend_config.metadata,
        ) == (
            "acme:alpha",
            "vespa",
            "http://vespa-alpha.test",
            8080,
            {"marker": "alpha"},
        )
        profile = parsed.backend_config.profiles["video_fixture"]
        assert (
            profile.type,
            profile.description,
            profile.schema_name,
            profile.embedding_model,
            profile.pipeline_config,
            profile.embedding_type,
            profile.schema_config,
        ) == (
            "video",
            "alpha video profile",
            "video_alpha",
            "TomoroAI/tomoro-colqwen3-embed-4b",
            {"extract_keyframes": True},
            "multi_vector",
            {"embedding_dim": 320},
        )
        assert isinstance(parsed.generator_config, SyntheticGeneratorConfig)
        assert parsed.generator_config.tenant_id == "acme:alpha"
        assert set(parsed.generator_config.optimizer_configs) == {
            "cross_modal",
            "entity_extraction",
            "modality",
            "profile",
            "query_enhancement",
            "routing",
            "unified",
            "workflow",
        }
        modality = parsed.generator_config.optimizer_configs["modality"]
        assert (
            modality.agent_mappings[0].modality,
            modality.agent_mappings[0].agent_name,
        ) == ("VIDEO", "search_agent")
        assert parsed.agents_config == {
            "search_agent": sections["agents"]["search_agent"]
        }
        assert parsed.agents_config is not sections["agents"]
        assert parsed.backend_default_profiles == {
            "video": {"profile": "video_fixture", "strategy": "segmentation"}
        }

    @pytest.mark.parametrize(
        ("url", "detail"),
        [
            ("search.test", "must be an absolute HTTP(S) URL"),
            ("ftp://search.test", "must be an absolute HTTP(S) URL"),
            ("http:///search", "must be an absolute HTTP(S) URL"),
        ],
    )
    def test_rejects_non_http_absolute_agent_urls(self, url, detail):
        from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config

        sections = _synthetic_runtime_sections()
        sections["agents"]["search_agent"]["url"] = url

        with pytest.raises(ValueError) as error:
            parse_synthetic_runtime_config(sections, tenant_id="acme:invalid")

        assert str(error.value) == (
            "Invalid synthetic runtime configuration for tenant='acme:invalid': "
            f"agents.search_agent.url {detail}"
        )

    @pytest.mark.parametrize(
        ("mutate", "detail"),
        [
            (
                lambda sections: sections["synthetic"].update({"sampling_config": {}}),
                "synthetic has invalid keys: missing=[] unknown=['sampling_config']",
            ),
            (
                lambda sections: sections["synthetic"]["optimizer_configs"]["modality"][
                    "agent_mappings"
                ][0].update({"confidence_threshold": 0.8}),
                "synthetic.optimizer_configs.modality.agent_mappings[0] "
                "has invalid keys: missing=[] unknown=['confidence_threshold']",
            ),
            (
                lambda sections: sections["synthetic"]["optimizer_configs"][
                    "profile"
                ].update({"num_examples_target": 17}),
                "synthetic.optimizer_configs.profile has invalid keys: "
                "missing=[] unknown=['num_examples_target']",
            ),
        ],
    )
    def test_rejects_obsolete_synthetic_fields(self, mutate, detail):
        from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config

        sections = _synthetic_runtime_sections()
        mutate(sections)

        with pytest.raises(ValueError) as error:
            parse_synthetic_runtime_config(sections, tenant_id="acme:invalid")

        assert str(error.value) == (
            "Invalid synthetic runtime configuration for tenant='acme:invalid': "
            f"{detail}"
        )

    def test_rejects_unknown_agent_keys(self):
        from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config

        sections = _synthetic_runtime_sections()
        sections["agents"]["search_agent"]["silent_agent_typo"] = True

        with pytest.raises(ValueError) as error:
            parse_synthetic_runtime_config(sections, tenant_id="acme:invalid")

        assert str(error.value) == (
            "Invalid synthetic runtime configuration for tenant='acme:invalid': "
            "agents.search_agent has invalid keys: missing=[] "
            "unknown=['silent_agent_typo']"
        )

    @pytest.mark.parametrize("field_name", ["type", "schema_name"])
    def test_rejects_profile_missing_a_required_field(self, field_name):
        from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config

        sections = _synthetic_runtime_sections()
        del sections["backend"]["profiles"]["video_fixture"][field_name]

        with pytest.raises(ValueError) as error:
            parse_synthetic_runtime_config(sections, tenant_id="acme:invalid")

        assert str(error.value) == (
            "Invalid synthetic runtime configuration for tenant='acme:invalid': "
            f"backend.profiles.video_fixture is missing required key {field_name!r}"
        )

    @pytest.mark.parametrize(
        ("field_name", "bad_value", "expected_type", "got_type"),
        [
            ("embedding_model", 123, "str", "int"),
            ("pipeline_config", [1, 2], "dict", "list"),
            ("strategies", "segmentation", "dict", "str"),
            ("schema_config", 5, "dict", "int"),
        ],
    )
    def test_rejects_profile_field_of_the_wrong_type(
        self,
        field_name,
        bad_value,
        expected_type,
        got_type,
    ):
        from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config

        sections = _synthetic_runtime_sections()
        sections["backend"]["profiles"]["video_fixture"][field_name] = bad_value

        with pytest.raises(ValueError) as error:
            parse_synthetic_runtime_config(sections, tenant_id="acme:invalid")

        assert str(error.value) == (
            "Invalid synthetic runtime configuration for tenant='acme:invalid': "
            f"backend.profiles.video_fixture.{field_name} must be "
            f"{expected_type}, got {got_type}"
        )

    def test_accepts_profile_keys_outside_the_typed_set(self):
        """Profiles registered at runtime carry keys the parser never names.

        ``build_memory_profile`` registers ``agent_memories`` with ``model``,
        ``encoder``, ``strategy`` and ``embedding_dims``; the Vespa search
        backend then reads ``encoder`` and ``strategy`` back by name. The
        parser must hand those through to ``extra_config`` untouched.
        """
        from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config

        sections = _synthetic_runtime_sections()
        sections["backend"]["profiles"]["video_fixture"].update(
            {
                "model": "lightonai/DenseOn",
                "encoder": "denseon",
                "strategy": "semantic_search",
                "embedding_dims": 768,
            }
        )

        config = parse_synthetic_runtime_config(sections, tenant_id="acme:invalid")

        profile = config.backend_config.profiles["video_fixture"]
        assert profile.extra_config == {
            "model": "lightonai/DenseOn",
            "encoder": "denseon",
            "strategy": "semantic_search",
            "embedding_dims": 768,
        }
        assert profile.schema_name == "video_fixture"
        assert profile.type == "video"
        assert profile.embedding_model == "TomoroAI/tomoro-colqwen3-embed-4b"

    @pytest.mark.parametrize(
        ("mutate", "detail"),
        [
            (
                lambda sections: sections["backend"]["default_profiles"][
                    "video"
                ].update({"legacy_profile": "video_fixture"}),
                "backend.default_profiles.video has invalid keys: "
                "missing=[] unknown=['legacy_profile']",
            ),
            (
                lambda sections: sections["backend"]["default_profiles"][
                    "video"
                ].update({"profile": "missing_profile"}),
                "backend.default_profiles.video.profile references unknown profile "
                "'missing_profile'",
            ),
        ],
    )
    def test_rejects_invalid_default_profile_contract(self, mutate, detail):
        from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config

        sections = _synthetic_runtime_sections()
        mutate(sections)

        with pytest.raises(ValueError) as error:
            parse_synthetic_runtime_config(sections, tenant_id="acme:invalid")

        assert str(error.value) == (
            "Invalid synthetic runtime configuration for tenant='acme:invalid': "
            f"{detail}"
        )

    @pytest.mark.parametrize(
        ("section", "replacement", "expected"),
        [
            (
                "backend",
                None,
                "Synthetic runtime configuration for tenant='acme:invalid' "
                "requires object section 'backend'",
            ),
            (
                "synthetic",
                None,
                "Synthetic runtime configuration for tenant='acme:invalid' "
                "requires object section 'synthetic'",
            ),
            (
                "agents",
                None,
                "Synthetic runtime configuration for tenant='acme:invalid' "
                "requires object section 'agents'",
            ),
            (
                "backend",
                {},
                "Synthetic runtime configuration for tenant='acme:invalid' "
                "section 'backend' must not be empty",
            ),
            (
                "synthetic",
                {},
                "Synthetic runtime configuration for tenant='acme:invalid' "
                "section 'synthetic' must not be empty",
            ),
            (
                "agents",
                {},
                "Synthetic runtime configuration for tenant='acme:invalid' "
                "section 'agents' must not be empty",
            ),
            (
                "backend",
                [],
                "Synthetic runtime configuration for tenant='acme:invalid' "
                "section 'backend' must be an object, got list",
            ),
            (
                "synthetic",
                [],
                "Synthetic runtime configuration for tenant='acme:invalid' "
                "section 'synthetic' must be an object, got list",
            ),
            (
                "agents",
                [],
                "Synthetic runtime configuration for tenant='acme:invalid' "
                "section 'agents' must be an object, got list",
            ),
        ],
    )
    def test_rejects_missing_empty_and_non_object_sections(
        self,
        section,
        replacement,
        expected,
    ):
        from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config

        sections = _synthetic_runtime_sections()
        if replacement is None:
            sections.pop(section)
        else:
            sections[section] = replacement

        with pytest.raises(ValueError) as error:
            parse_synthetic_runtime_config(sections, tenant_id="acme:invalid")

        assert str(error.value) == expected

    @pytest.mark.parametrize(
        ("target", "agent_changes", "detail"),
        [
            (
                "missing_agent",
                {},
                "mapping for modality 'VIDEO' targets unknown agent 'missing_agent'",
            ),
            (
                "disabled_video_agent",
                {},
                "mapping for modality 'VIDEO' targets disabled agent "
                "'disabled_video_agent'",
            ),
            (
                "search_agent",
                {"modalities": ["DOCUMENT"]},
                "agent 'search_agent' does not declare mapped modality 'VIDEO'",
            ),
        ],
    )
    def test_mapping_errors_preserve_tenant_modality_and_agent_context(
        self,
        target,
        agent_changes,
        detail,
    ):
        from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config

        sections = _synthetic_runtime_sections()
        mapping = sections["synthetic"]["optimizer_configs"]["modality"][
            "agent_mappings"
        ][0]
        mapping["agent_name"] = target
        sections["agents"].get(target, {}).update(agent_changes)

        with pytest.raises(ValueError) as error:
            parse_synthetic_runtime_config(sections, tenant_id="acme:alpha")

        assert str(error.value) == (
            f"Invalid synthetic runtime configuration for tenant='acme:alpha': {detail}"
        )

    def test_loaded_agent_validation_precedes_backend_use(self):
        from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config

        with pytest.raises(ValueError) as error:
            parse_synthetic_runtime_config(
                _synthetic_runtime_sections(),
                tenant_id="acme:alpha",
                loaded_agent_names={"document_agent"},
            )

        assert str(error.value) == (
            "Invalid synthetic runtime configuration for tenant='acme:alpha': "
            "enabled agents were not loaded: search_agent"
        )

    @pytest.mark.asyncio
    async def test_concurrent_tenant_parses_do_not_share_nested_state(self):
        from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config

        sections = _synthetic_runtime_sections(marker="shared")
        barrier = threading.Barrier(2)

        def parse(tenant_id):
            barrier.wait(timeout=2)
            return parse_synthetic_runtime_config(sections, tenant_id=tenant_id)

        alpha, beta = await asyncio.gather(
            asyncio.to_thread(parse, "acme:alpha"),
            asyncio.to_thread(parse, "acme:beta"),
        )

        alpha.backend_config.metadata["marker"] = "alpha-only"
        alpha.generator_config.field_mappings.topic_fields.append("alpha-only")
        alpha.agents_config["search_agent"]["url"] = "http://alpha-only.test"

        assert (
            beta.backend_config.tenant_id,
            beta.backend_config.metadata,
            beta.generator_config.tenant_id,
            beta.generator_config.field_mappings.topic_fields,
            beta.agents_config["search_agent"]["url"],
        ) == (
            "acme:beta",
            {"marker": "shared"},
            "acme:beta",
            ["video_title"],
            "http://search-shared.test",
        )
        assert sections["backend"]["metadata"] == {"marker": "shared"}
        assert sections["synthetic"]["field_mappings"]["topic_fields"] == [
            "video_title"
        ]
        assert sections["agents"]["search_agent"]["url"] == (
            "http://search-shared.test"
        )


class TestSyntheticEntityExtractorWiring:
    @pytest.mark.asyncio
    async def test_runtime_extractor_dispatches_exact_production_agent_request(self):
        from cogniverse_runtime.main import _dispatcher_entity_extractor

        calls = []
        output = {
            "status": "success",
            "agent": "entity_extraction_agent",
            "query": "Marie Curie discovered radium",
            "entities": [
                {"text": "Marie Curie", "type": "PERSON", "confidence": 0.99},
                {"text": "radium", "type": "CONCEPT", "confidence": 0.97},
            ],
            "relationships": [],
        }

        class Dispatcher:
            async def dispatch(self, **kwargs):
                calls.append(kwargs)
                return output

        extractor = _dispatcher_entity_extractor(Dispatcher())

        assert (
            await extractor("Marie Curie discovered radium", "acme:science") is output
        )
        assert calls == [
            {
                "agent_name": "entity_extraction_agent",
                "query": "Marie Curie discovered radium",
                "context": {"tenant_id": "acme:science"},
            }
        ]

    @pytest.mark.asyncio
    async def test_runtime_extractor_keeps_concurrent_tenants_isolated(self):
        from cogniverse_runtime.main import _dispatcher_entity_extractor

        entered = asyncio.Event()
        calls = []

        class Dispatcher:
            async def dispatch(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 2:
                    entered.set()
                await asyncio.wait_for(entered.wait(), timeout=1)
                return {
                    "query": kwargs["query"],
                    "tenant": kwargs["context"]["tenant_id"],
                }

        extractor = _dispatcher_entity_extractor(Dispatcher())
        outputs = await asyncio.gather(
            extractor("Alpha source", "org:alpha"),
            extractor("Beta source", "org:beta"),
        )

        assert outputs == [
            {"query": "Alpha source", "tenant": "org:alpha"},
            {"query": "Beta source", "tenant": "org:beta"},
        ]
        assert calls == [
            {
                "agent_name": "entity_extraction_agent",
                "query": "Alpha source",
                "context": {"tenant_id": "org:alpha"},
            },
            {
                "agent_name": "entity_extraction_agent",
                "query": "Beta source",
                "context": {"tenant_id": "org:beta"},
            },
        ]

    @pytest.mark.asyncio
    async def test_runtime_extractor_preserves_dispatch_failure_context(self):
        from cogniverse_runtime.main import _dispatcher_entity_extractor

        failure = ConnectionError("dispatcher unavailable")

        class Dispatcher:
            async def dispatch(self, **kwargs):
                raise failure

        extractor = _dispatcher_entity_extractor(Dispatcher())
        with pytest.raises(RuntimeError) as error:
            await extractor("Alpha source", "org:alpha")

        assert str(error.value) == (
            "Entity extraction dispatch failed for tenant='org:alpha' "
            "source_text='Alpha source': dispatcher unavailable"
        )
        assert error.value.__cause__ is failure

    @pytest.mark.asyncio
    async def test_cli_extractor_calls_real_agent_process_contract(self):
        from cogniverse_agents.entity_extraction_agent import (
            EntityExtractionInput,
            EntityExtractionOutput,
        )
        from cogniverse_runtime.optimization_cli import _build_cli_entity_extractor

        process_inputs = []
        built_agents = []
        telemetry = object()
        config_manager = SimpleNamespace(
            get_system_config=lambda: SimpleNamespace(
                inference_service_urls={"gliner": "http://gliner.test:8010"}
            )
        )
        expected = EntityExtractionOutput(
            query="Marie Curie discovered radium",
            entities=[
                {
                    "text": "Marie Curie",
                    "type": "PERSON",
                    "confidence": 0.99,
                },
                {"text": "radium", "type": "CONCEPT", "confidence": 0.97},
            ],
            relationships=[],
            entity_count=2,
            has_entities=True,
            dominant_types=["PERSON", "CONCEPT"],
            path_used="gliner",
        )

        class RecordingAgent:
            def __init__(self, *, deps):
                self.deps = deps
                self.artifact_loads = 0
                built_agents.append(self)

            def _load_artifact(self):
                self.artifact_loads += 1

            async def process(self, value):
                process_inputs.append(value)
                return expected

        with patch(
            "cogniverse_agents.entity_extraction_agent.EntityExtractionAgent",
            RecordingAgent,
        ):
            extractor = await _build_cli_entity_extractor(
                config_manager=config_manager,
                telemetry_manager=telemetry,
                tenant_id="acme:science",
            )
            result = await extractor("Marie Curie discovered radium", "acme:science")

        assert result is expected
        assert len(built_agents) == 1
        agent = built_agents[0]
        assert agent.deps.gliner_inference_url == "http://gliner.test:8010"
        assert agent.telemetry_manager is telemetry
        assert agent._config_manager is config_manager
        assert agent._artifact_tenant_id == "acme:science"
        assert agent.artifact_loads == 1
        assert len(process_inputs) == 1
        assert isinstance(process_inputs[0], EntityExtractionInput)
        assert process_inputs[0].model_dump() == {
            "query": "Marie Curie discovered radium",
            "tenant_id": "acme:science",
        }

    @pytest.mark.asyncio
    async def test_cli_extractor_requires_configured_gliner_endpoint(self):
        from cogniverse_runtime.optimization_cli import _build_cli_entity_extractor

        config_manager = SimpleNamespace(
            get_system_config=lambda: SimpleNamespace(inference_service_urls={})
        )

        with pytest.raises(ValueError) as error:
            await _build_cli_entity_extractor(
                config_manager=config_manager,
                telemetry_manager=object(),
                tenant_id="acme:science",
            )

        assert str(error.value) == (
            "GLiNER inference endpoint is required for synthetic entity extraction "
            "for tenant='acme:science'"
        )

    @pytest.mark.asyncio
    async def test_cli_extractor_preserves_process_failure_context(self):
        from cogniverse_runtime.optimization_cli import _build_cli_entity_extractor

        failure = TimeoutError("GLiNER timed out")
        config_manager = SimpleNamespace(
            get_system_config=lambda: SimpleNamespace(
                inference_service_urls={"gliner": "http://gliner.test:8010"}
            )
        )

        class FailingAgent:
            def __init__(self, *, deps):
                pass

            def _load_artifact(self):
                pass

            async def process(self, value):
                raise failure

        with patch(
            "cogniverse_agents.entity_extraction_agent.EntityExtractionAgent",
            FailingAgent,
        ):
            extractor = await _build_cli_entity_extractor(
                config_manager=config_manager,
                telemetry_manager=object(),
                tenant_id="org:alpha",
            )
            with pytest.raises(RuntimeError) as error:
                await extractor("Alpha source", "org:alpha")

        assert str(error.value) == (
            "Entity extraction agent failed for tenant='org:alpha' "
            "source_text='Alpha source': GLiNER timed out"
        )
        assert error.value.__cause__ is failure

    @pytest.mark.asyncio
    async def test_cli_profile_labeler_calls_profile_agent_process_contract(self):
        from cogniverse_agents.profile_selection_agent import (
            ProfileSelectionInput,
            ProfileSelectionOutput,
        )
        from cogniverse_runtime.optimization_cli import _build_cli_profile_labeler

        process_inputs = []
        built_agents = []
        telemetry = object()
        config_manager = object()
        expected = ProfileSelectionOutput(
            query="quantum computing",
            selected_profile="document_semantic",
            confidence=0.95,
            reasoning="The production selector chose document retrieval.",
            query_intent="document_search",
            modality="document",
            complexity="medium",
        )

        class RecordingAgent:
            def __init__(self, *, deps):
                self.deps = deps
                self.artifact_loads = 0
                built_agents.append(self)

            def _load_artifact(self):
                self.artifact_loads += 1

            async def process(self, value):
                process_inputs.append(value)
                return expected

        with patch(
            "cogniverse_agents.profile_selection_agent.ProfileSelectionAgent",
            RecordingAgent,
        ):
            labeler = await _build_cli_profile_labeler(
                config_manager=config_manager,
                telemetry_manager=telemetry,
                tenant_id="acme:science",
            )
            result = await labeler(
                "quantum computing",
                ["audio_semantic", "document_semantic"],
                "acme:science",
            )

        assert result is expected
        assert len(built_agents) == 1
        agent = built_agents[0]
        assert agent.deps.available_profiles == []
        assert agent.telemetry_manager is telemetry
        assert agent._config_manager is config_manager
        assert agent._artifact_tenant_id == "acme:science"
        assert agent.artifact_loads == 1
        assert len(process_inputs) == 1
        assert isinstance(process_inputs[0], ProfileSelectionInput)
        assert process_inputs[0].model_dump() == {
            "query": "quantum computing",
            "available_profiles": ["audio_semantic", "document_semantic"],
            "tenant_id": "acme:science",
        }


# ---------------------------------------------------------------------------
# Test: each mode handles empty span data gracefully
# ---------------------------------------------------------------------------


class TestEmptySpanHandling:
    """Most optimization functions return no_data when Phoenix has no matching spans."""

    @pytest.mark.asyncio
    async def test_simba_no_data(self, fake_telemetry_manager):
        from cogniverse_runtime.optimization_cli import run_simba_optimization

        p1, p2 = _patch_infra(fake_telemetry_manager)
        with p1, p2:
            result = await run_simba_optimization(
                tenant_id="test:unit", lookback_hours=1
            )
        assert result["status"] == "no_data"
        assert result["spans_found"] == 0
        assert "selection" not in result

    @pytest.mark.asyncio
    async def test_workflow_no_data(self, fake_telemetry_manager):
        from cogniverse_runtime.optimization_cli import run_workflow_optimization

        p1, p2 = _patch_infra(fake_telemetry_manager)
        with p1, p2:
            result = await run_workflow_optimization(
                tenant_id="test:unit", lookback_hours=1
            )
        assert result["status"] == "no_data"
        assert result["spans_found"] == 0

    @pytest.mark.asyncio
    async def test_gateway_thresholds_no_data(self, fake_telemetry_manager):
        from cogniverse_runtime.optimization_cli import (
            run_gateway_thresholds_optimization,
        )

        p1, p2 = _patch_infra(fake_telemetry_manager)
        with p1, p2:
            result = await run_gateway_thresholds_optimization(
                tenant_id="test:unit", lookback_hours=1
            )
        assert result["status"] == "no_data"
        assert result["spans_found"] == 0

    @pytest.mark.asyncio
    async def test_profile_no_data(self, fake_telemetry_manager):
        from cogniverse_runtime.optimization_cli import (
            ProfileLabelDerivationResult,
            run_profile_optimization,
        )

        empty_label_source = ProfileLabelDerivationResult({}, [], [])

        p1, p2 = _patch_infra(fake_telemetry_manager)
        with (
            p1,
            p2,
            patch(
                "cogniverse_agents.profile_selection_agent.tenant_usable_profile_names",
                return_value=["video_colpali_smol500_mv_frame"],
            ),
            patch(
                "cogniverse_runtime.optimization_cli._profile_selection_label_source",
                return_value=empty_label_source,
            ),
        ):
            result = await run_profile_optimization(
                tenant_id="test:unit", lookback_hours=1
            )
        assert result == {
            "status": "profile_selection_ground_truth_missing",
            "retryable": False,
            "error": "profile_selection_ground_truth is not configured for tenant test:unit",
        }


class TestProfileSelectionTrainingExamples:
    @pytest.mark.asyncio
    async def test_profile_pairs_use_recorded_pool_and_live_fallback(self):
        from cogniverse_agents.profile_selection_agent import (
            tenant_usable_profile_names,
        )
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_foundation.config.unified_config import (
            BackendProfileConfig,
            SystemConfig,
        )
        from cogniverse_runtime.optimization_cli import _profile_selection_pairs
        from tests.utils.memory_store import InMemoryConfigStore

        store = InMemoryConfigStore()
        config_manager = ConfigManager(store=store)
        config_manager.set_system_config(
            SystemConfig(
                inference_service_urls={
                    "vllm_colpali": "http://localhost:8000",
                    "vllm_colqwen": "http://localhost:8001",
                }
            )
        )
        config_manager.add_backend_profile(
            BackendProfileConfig.from_dict(
                "video_colpali_smol500_mv_frame",
                {
                    "type": "video",
                    "schema_name": "video_colpali_smol500_mv_frame",
                    "embedding_model": "TomoroAI/tomoro-colqwen3-embed-4b",
                    "inference_services": {"embedding": "vllm_colpali"},
                },
            ),
            tenant_id="acme:docs",
        )
        config_manager.add_backend_profile(
            BackendProfileConfig.from_dict(
                "video_colqwen_omni_mv_chunk_30s",
                {
                    "type": "video",
                    "schema_name": "video_colqwen_omni_mv_chunk_30s",
                    "embedding_model": "TomoroAI/tomoro-colqwen3-embed-4b",
                    "inference_services": {"embedding": "vllm_colqwen"},
                },
            ),
            tenant_id="acme:docs",
        )
        config_manager.add_backend_profile(
            BackendProfileConfig.from_dict(
                "video_videoprism_base_mv_chunk_30s",
                {
                    "type": "video",
                    "schema_name": "video_videoprism_base_mv_chunk_30s",
                    "embedding_model": "videoprism_public_v1_base_hf",
                    "inference_services": {"embedding": "videoprism_jax"},
                },
            ),
            tenant_id="acme:docs",
        )

        expected_live = [
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
        ]
        assert tenant_usable_profile_names(config_manager, "acme:docs") == (
            expected_live
        )

        spans_df = _make_spans_df(
            "cogniverse.profile_selection",
            [
                {
                    "context.span_id": "ps-1",
                    "attributes.input.value": "find a clip about transformer architecture",
                    "attributes.output.value": json.dumps(
                        {
                            "selected_profile": "video_colqwen_omni_mv_chunk_30s",
                            "modality": "video",
                            "complexity": "medium",
                            "intent": "video_search",
                            "confidence": 0.9,
                        }
                    ),
                    "attributes.available_profiles": (
                        "video_colqwen_omni_mv_chunk_30s, "
                        "video_colpali_smol500_mv_frame"
                    ),
                },
                {
                    "context.span_id": "ps-2",
                    "attributes.input.value": "find a clip about transformer architecture",
                    "attributes.output.value": json.dumps(
                        {
                            "selected_profile": "video_colpali_smol500_mv_frame",
                            "modality": "video",
                            "complexity": "medium",
                            "intent": "video_search",
                            "confidence": 0.9,
                        }
                    ),
                },
            ],
        )

        pairs = _profile_selection_pairs(
            spans_df, config_manager=config_manager, tenant_id="acme:docs"
        )

        assert pairs == [
            {
                "query": "find a clip about transformer architecture",
                "available_profiles": (
                    "video_colqwen_omni_mv_chunk_30s, video_colpali_smol500_mv_frame"
                ),
                "selected_profile": "video_colqwen_omni_mv_chunk_30s",
                "modality": "video",
                "complexity": "medium",
                "intent": "video_search",
                "confidence": 0.9,
                "example_id": "span:ps-1",
            },
            {
                "query": "find a clip about transformer architecture",
                "available_profiles": ", ".join(expected_live),
                "selected_profile": "video_colpali_smol500_mv_frame",
                "modality": "video",
                "complexity": "medium",
                "intent": "video_search",
                "confidence": 0.9,
                "example_id": "span:ps-2",
            },
        ]

    def test_pair_builders_refuse_rows_without_span_id(self):
        """A record the ledger cannot attribute is an error, never ``span:None``."""
        from cogniverse_runtime.optimization_cli import (
            _entity_extraction_pairs,
            _profile_selection_pairs,
            _query_enhancement_pairs,
        )

        qe_df = _make_spans_df(
            "cogniverse.query_enhancement",
            [
                {
                    "attributes.input.value": "find tutorials",
                    "attributes.output.value": json.dumps(
                        {
                            "enhanced_query": "find ML tutorials",
                            "expansion_terms": ["ML"],
                        }
                    ),
                }
            ],
        )
        with pytest.raises(ValueError) as qe_err:
            _query_enhancement_pairs(qe_df)
        assert str(qe_err.value) == (
            "cogniverse.query_enhancement span row 0 has no context.span_id; "
            "the optimizer cannot record which example it consumed"
        )

        entity_df = _make_spans_df(
            "cogniverse.entity_extraction",
            [
                {
                    "attributes.input.value": "find PyTorch tutorials",
                    "attributes.output.value": json.dumps(
                        {"entities": [{"text": "PyTorch", "type": "TECHNOLOGY"}]}
                    ),
                }
            ],
        )
        with pytest.raises(ValueError) as entity_err:
            _entity_extraction_pairs(entity_df)
        assert str(entity_err.value) == (
            "cogniverse.entity_extraction span row 0 has no context.span_id; "
            "the optimizer cannot record which example it consumed"
        )

        profile_df = _make_spans_df(
            "cogniverse.profile_selection",
            [
                {
                    "attributes.input.value": "find a clip",
                    "attributes.output.value": json.dumps(
                        {
                            "selected_profile": "video_colpali_smol500_mv_frame",
                            "modality": "video",
                            "complexity": "simple",
                            "intent": "video_search",
                            "confidence": 0.9,
                        }
                    ),
                    "attributes.available_profiles": "video_colpali_smol500_mv_frame",
                }
            ],
        )
        with pytest.raises(ValueError) as profile_err:
            _profile_selection_pairs(profile_df, config_manager=None, tenant_id="acme")
        assert str(profile_err.value) == (
            "cogniverse.profile_selection span row 0 has no context.span_id; "
            "the optimizer cannot record which example it consumed"
        )

    def test_profile_optimizer_source_does_not_embed_retired_pool(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[3]
            / "libs"
            / "runtime"
            / "cogniverse_runtime"
            / "optimization_cli.py"
        )
        text = source.read_text()
        retired_pool = (
            "video_colpali_smol500_mv_frame,"
            "video_colqwen_omni_mv_chunk_30s,"
            "video_videoprism_base_mv_chunk_30s,"
            "video_videoprism_large_mv_chunk_30s"
        )
        assert retired_pool not in text


# ---------------------------------------------------------------------------
# Test: functions handle spans with no extractable training examples
# ---------------------------------------------------------------------------


def _make_spans_df(span_name: str, rows: list[dict]) -> pd.DataFrame:
    """Build a spans DataFrame with the given name and attribute columns."""
    df = pd.DataFrame(rows)
    df["name"] = span_name
    return df


def _gateway_row(complexity: str, confidence: float, status_code: str) -> dict:
    """A canonical cogniverse.gateway span row (decision on output.value).

    Only the calibration MATH needs controlled complexity/status inputs (a real
    gateway won't emit ERROR spans on demand); the real producer->reader
    contract is covered by the real-Phoenix gateway test.
    """
    return {
        "attributes.output.value": json.dumps(
            {
                "complexity": complexity,
                "confidence": confidence,
                "modality": "video",
                "generation_type": "raw_results",
                "routed_to": "search_agent"
                if complexity == "simple"
                else "orchestrator_agent",
            }
        ),
        "status_code": status_code,
    }


class TestSpansWithNoExamples:
    """Spans exist but contain no usable training data (missing attributes)."""

    @pytest.mark.asyncio
    async def test_simba_spans_missing_attributes(self):
        # Canonical span whose enhancement is empty -> no usable training pair.
        spans_df = _make_spans_df(
            "cogniverse.query_enhancement",
            [
                {
                    "attributes.input.value": "robots",
                    "attributes.output.value": json.dumps({"enhanced_query": ""}),
                }
            ],
        )
        provider = FakeTelemetryProvider(spans_df)
        mgr = FakeTelemetryManager(provider)

        from cogniverse_runtime.optimization_cli import run_simba_optimization

        p1, p2 = _patch_infra(mgr)
        with p1, p2:
            result = await run_simba_optimization(
                tenant_id="test:unit", lookback_hours=1
            )
        assert result["status"] == "no_data"
        assert result["spans_found"] == 1
        assert result["examples"] == 0

    @pytest.mark.asyncio
    async def test_profile_spans_low_confidence(self):
        """Profile optimization skips examples with confidence < 0.5."""
        spans_df = _make_spans_df(
            "cogniverse.profile_selection",
            [
                {
                    "attributes.input.value": "find videos",
                    "attributes.output.value": json.dumps(
                        {
                            "selected_profile": "video_colpali_smol500_mv_frame",
                            "modality": "video",
                            "complexity": "simple",
                            "intent": "video_search",
                            "confidence": 0.2,
                        }
                    ),
                },
            ],
        )
        provider = FakeTelemetryProvider(spans_df)
        mgr = FakeTelemetryManager(provider)

        from cogniverse_runtime.optimization_cli import (
            ProfileLabelDerivationResult,
            run_profile_optimization,
        )

        empty_label_source = ProfileLabelDerivationResult({}, [], [])

        p1, p2 = _patch_infra(mgr)
        with (
            p1,
            p2,
            patch(
                "cogniverse_agents.profile_selection_agent.tenant_usable_profile_names",
                return_value=["video_colpali_smol500_mv_frame"],
            ),
            patch(
                "cogniverse_runtime.optimization_cli._profile_selection_label_source",
                return_value=empty_label_source,
            ),
        ):
            result = await run_profile_optimization(
                tenant_id="test:unit", lookback_hours=1
            )
        assert result == {
            "status": "profile_selection_ground_truth_missing",
            "retryable": False,
            "error": "profile_selection_ground_truth is not configured for tenant test:unit",
        }


def _qe_span_row(
    query: str,
    enhanced: str,
    *,
    expansion_terms: list[str],
    grounding_context: str = "",
    source_text: str = "",
    confidence: float = 0.8,
    span_id: str = "span-0",
) -> dict:
    """A cogniverse.query_enhancement span row as the agent writes it."""
    return {
        "context.span_id": span_id,
        "attributes.input.value": query,
        "attributes.input.source_text": source_text,
        "attributes.input.grounding_context": grounding_context,
        "attributes.output.value": json.dumps(
            {
                "enhanced_query": enhanced,
                "expansion_terms": expansion_terms,
                "synonyms": ["s1"],
                "context_additions": ["c1"],
                "variant_count": 2,
                "confidence": confidence,
            }
        ),
    }


def _profile_span_row(
    query: str,
    *,
    span_id: str,
    available_profiles: list[str],
    selected_profile: str,
    confidence: float = 0.9,
) -> dict:
    return {
        "context.span_id": span_id,
        "attributes.input.value": query,
        "attributes.available_profiles": ", ".join(available_profiles),
        "attributes.output.value": json.dumps(
            {
                "selected_profile": selected_profile,
                "modality": "video",
                "complexity": "simple",
                "intent": "video_search",
                "confidence": confidence,
                "reasoning": f"Selected {selected_profile}",
            }
        ),
    }


def _profile_example(
    *, selected: str, available: list[str] | str, query: str = "find a clip"
):
    import dspy

    return dspy.Example(
        query=query,
        available_profiles=available,
        selected_profile=selected,
        confidence="0.9",
        reasoning="Selected profile for the query.",
        query_intent="video_search",
        modality="video",
        complexity="simple",
    ).with_inputs("query", "available_profiles")


def _sel(selected: str):
    import dspy

    return dspy.Prediction(selected_profile=selected)


def _profile_selection_query_record(
    query: str,
    *,
    expected_videos: list[str],
    ground_truth: str,
    query_type: str = "question",
    source: str = "synthetic.json",
) -> dict[str, Any]:
    return {
        "query": query,
        "expected_videos": expected_videos,
        "ground_truth": ground_truth,
        "query_type": query_type,
        "source": source,
    }


_CONTENT_HASH = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
_OTHER_CONTENT_HASH = "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae"
_VIDEO_TITLE_FIELDS = {
    "video_colpali_smol500_mv_frame": "video_title",
    "video_colqwen_omni_mv_chunk_30s": "video_title",
}


def _profile_hit(
    title: str | None,
    *,
    title_field: str = "video_title",
    content_hash: str = _CONTENT_HASH,
    segment: int = 0,
) -> dict[str, Any]:
    """One ``SearchResult.to_dict()`` row for a content-addressed segment."""
    metadata: dict[str, Any] = {"video_id": content_hash, "source_id": content_hash}
    if title is not None:
        metadata[title_field] = title
    return {
        "document_id": f"{content_hash}_seg_{segment}",
        "score": 0.5,
        "metadata": metadata,
        "highlights": {},
        "source_id": content_hash,
    }


_TENANT_GROUND_TRUTH_ROWS: list[dict[str, Any]] = [
    {
        "query": "tenant clip of a robot welding a seam",
        "expected_videos": ["tenant_v_weld_01"],
        "ground_truth": "A robot arm welds a seam on a steel frame.",
        "query_type": "question",
        "source": "tenant_upload.json",
    },
    {
        "query": "tenant clip of a kite surfer jumping",
        "expected_videos": ["tenant_v_kite_02", "tenant_v_kite_03"],
        "ground_truth": "A kite surfer jumps a wave and lands.",
        "query_type": "answer_phrase",
        "source": "tenant_upload.json",
    },
    {
        "query": "tenant clip of a drone landing on a roof",
        "expected_videos": ["tenant_v_drone_04"],
        "ground_truth": "A quadcopter lands on a flat rooftop.",
        "query_type": "question",
        "source": "tenant_upload.json",
    },
    {
        "query": "tenant clip nobody can recover",
        "expected_videos": ["tenant_v_missing_05"],
        "ground_truth": "No indexed video matches this row.",
        "query_type": "question",
        "source": "tenant_upload.json",
    },
]
_TENANT_RECOVERABLE_QUERIES = {
    row["query"]: list(row["expected_videos"]) for row in _TENANT_GROUND_TRUTH_ROWS[:3]
}


def _tenant_ground_truth_blob() -> str:
    return json.dumps(_TENANT_GROUND_TRUTH_ROWS, separators=(",", ":"))


def _titled_search_result(title: str, segment: int):
    """A real ``SearchResult`` whose document carries ``video_title``."""
    from cogniverse_sdk.document import (
        ContentType,
        Document,
        ProcessingStatus,
        SearchResult,
    )

    document = Document(
        id=f"{_CONTENT_HASH}_seg_{segment}",
        content_type=ContentType.VIDEO,
        status=ProcessingStatus.COMPLETED,
    )
    document.add_metadata("video_id", _CONTENT_HASH)
    document.add_metadata("video_title", title)
    document.add_metadata("source_id", _CONTENT_HASH)
    return SearchResult(document, 0.5)


def _search_service_recovering(
    expected_by_query: dict[str, list[str]],
    *,
    recovering_profile: str,
    calls: list[tuple[str, str, str, int]],
    constructed: list[dict[str, Any]],
):
    """A ``SearchService`` stand-in: ``recovering_profile`` returns every
    expected video of a known query as a titled result; anything else returns
    one unrelated title."""

    class FakeSearchService:
        def __init__(self, *, config, config_manager, schema_loader):
            del config
            constructed.append(
                {
                    "config_manager": config_manager,
                    "schema_loader": type(schema_loader).__name__,
                }
            )

        def search(self, *, query, profile, tenant_id, top_k):
            calls.append((query, profile, tenant_id, top_k))
            if profile == recovering_profile and query in expected_by_query:
                titles = [f"{video}.mp4" for video in expected_by_query[query]]
            else:
                titles = ["v_unrelated.mp4"]
            return [
                _titled_search_result(title, index)
                for index, title in enumerate(titles)
            ]

    return FakeSearchService


def _video_profile_config_manager(profiles: list[str]):
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import BackendProfileConfig
    from tests.utils.memory_store import InMemoryConfigStore

    config_manager = ConfigManager(store=InMemoryConfigStore())
    for profile in profiles:
        config_manager.add_backend_profile(
            BackendProfileConfig.from_dict(
                profile, {"type": "video", "schema_name": profile}
            ),
            tenant_id="test:unit",
        )
    return config_manager


class _ProfileSelectionByQuery:
    def __init__(self, labels: dict[str, str], *, default: str):
        self.labels = labels
        self.default = default

    def __call__(self, query: str, available_profiles: str):
        del available_profiles
        return _sel(self.labels.get(query, self.default))


def _entity_example(*, query: str = "find entities", entities: str = "[]"):
    import dspy

    return dspy.Example(
        query=query,
        entities=entities,
    ).with_inputs("query")


def _ents(entities: str):
    import dspy

    return dspy.Prediction(entities=entities)


def _example(**kwargs):
    import dspy

    fields = {
        "query": "",
        "source_text": "",
        "grounding_context": "",
    }
    fields.update(kwargs)
    return dspy.Example(**fields).with_inputs(
        "query", "source_text", "grounding_context"
    )


def _pred(enhanced: str, terms):
    import dspy

    return dspy.Prediction(enhanced_query=enhanced, expansion_terms=terms)


def _module_returning(prediction):
    class _Module:
        def __init__(self, value):
            self._value = value
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return self._value

    return _Module(prediction)


_TF_CONTEXT = (
    "Entities: TensorFlow (TECHNOLOGY), neural networks (CONCEPT); "
    "Relationships: TensorFlow -used_for-> neural networks"
)


class TestPopulationFloorConfig:
    def test_routing_config_round_trips_optimizer_floors(self):
        from cogniverse_foundation.config.unified_config import RoutingConfigUnified

        cfg = RoutingConfigUnified(
            tenant_id="acme:acme",
            min_samples_for_optimization=40,
            min_unique_queries=7,
            optimizer_floors={
                "profile_selection": {
                    "min_samples_for_optimization": 20,
                    "min_unique_queries": 6,
                },
                "entity_extraction": {
                    "min_samples_for_optimization": 58,
                    "min_unique_queries": 15,
                },
            },
        )
        restored = RoutingConfigUnified.from_dict(cfg.to_dict())
        assert restored == cfg

    def test_min_unique_queries_defaults_to_three(self):
        from cogniverse_foundation.config.unified_config import RoutingConfigUnified

        assert RoutingConfigUnified(tenant_id="acme:acme").min_unique_queries == 3

    def test_population_floor_reads_per_optimizer_mapping(self, tmp_path, monkeypatch):
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_foundation.config.unified_config import RoutingConfigUnified
        from cogniverse_runtime import optimization_cli
        from cogniverse_runtime.optimization_cli import _population_floor_from_config
        from tests.utils.memory_store import InMemoryConfigStore

        shipped_config = tmp_path / "config.json"
        shipped_config.write_text(
            json.dumps(
                {
                    "routing": {
                        "optimization_config": {
                            "optimizer_floors": {
                                "profile_selection": {
                                    "min_samples_for_optimization": 20,
                                    "min_unique_queries": 6,
                                },
                                "entity_extraction": {
                                    "min_samples_for_optimization": 58,
                                    "min_unique_queries": 15,
                                },
                            }
                        }
                    }
                }
            )
        )
        monkeypatch.setattr(optimization_cli, "SHIPPED_CONFIG_PATH", shipped_config)

        manager = ConfigManager(store=InMemoryConfigStore())
        manager.set_routing_config(
            RoutingConfigUnified(
                tenant_id="acme:acme",
            )
        )
        assert _population_floor_from_config(
            "acme:acme", manager, "profile_selection"
        ) == (20, 6)
        assert _population_floor_from_config(
            "acme:acme", manager, "entity_extraction"
        ) == (58, 15)
        assert _population_floor_from_config(
            "acme:acme", manager, "query_enhancement"
        ) == (100, 3)

    def test_profile_selection_max_label_share_reads_shipped_floor(self):
        from cogniverse_runtime.optimization_cli import (
            _shipped_profile_selection_max_label_share_from_config,
        )

        assert _shipped_profile_selection_max_label_share_from_config() == 0.8

    def test_population_floor_store_override_wins_over_shipped_floor(
        self, tmp_path, monkeypatch
    ):
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_foundation.config.unified_config import RoutingConfigUnified
        from cogniverse_runtime import optimization_cli
        from cogniverse_runtime.optimization_cli import _population_floor_from_config
        from tests.utils.memory_store import InMemoryConfigStore

        shipped_config = tmp_path / "config.json"
        shipped_config.write_text(
            json.dumps(
                {
                    "routing": {
                        "optimization_config": {
                            "optimizer_floors": {
                                "profile_selection": {
                                    "min_samples_for_optimization": 20,
                                    "min_unique_queries": 6,
                                },
                                "entity_extraction": {
                                    "min_samples_for_optimization": 58,
                                    "min_unique_queries": 15,
                                },
                            }
                        }
                    }
                }
            )
        )

        monkeypatch.setattr(optimization_cli, "SHIPPED_CONFIG_PATH", shipped_config)

        manager = ConfigManager(store=InMemoryConfigStore())
        manager.set_routing_config(
            RoutingConfigUnified(
                tenant_id="acme:acme",
                optimizer_floors={
                    "profile_selection": {
                        "min_samples_for_optimization": 33,
                        "min_unique_queries": 9,
                    }
                },
            )
        )
        assert _population_floor_from_config(
            "acme:acme", manager, "profile_selection"
        ) == (33, 9)
        assert _population_floor_from_config(
            "acme:acme", manager, "entity_extraction"
        ) == (58, 15)

    def test_unlisted_optimizer_falls_back_to_tenant_global_floor(self):
        """An optimizer absent from optimizer_floors inherits the tenant's
        CONFIGURED global floor, never a hardcoded literal."""
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_foundation.config.unified_config import RoutingConfigUnified
        from cogniverse_runtime.optimization_cli import _population_floor_from_config
        from tests.utils.memory_store import InMemoryConfigStore

        manager = ConfigManager(store=InMemoryConfigStore())
        manager.set_routing_config(
            RoutingConfigUnified(
                tenant_id="acme:acme",
                min_samples_for_optimization=40,
                min_unique_queries=7,
                optimizer_floors={
                    "profile_selection": {
                        "min_samples_for_optimization": 20,
                        "min_unique_queries": 6,
                    },
                },
            )
        )
        assert _population_floor_from_config(
            "acme:acme", manager, "query_enhancement"
        ) == (40, 7)
        assert _population_floor_from_config(
            "acme:acme", manager, "profile_selection"
        ) == (20, 6)

    def test_malformed_store_floor_warns_and_falls_through(self, caplog):
        """A malformed tenant override never silently vanishes: it is named in
        a warning and resolution falls through to the shipped mapping."""
        import logging

        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_foundation.config.unified_config import RoutingConfigUnified
        from cogniverse_runtime.optimization_cli import _population_floor_from_config
        from tests.utils.memory_store import InMemoryConfigStore

        manager = ConfigManager(store=InMemoryConfigStore())
        manager.set_routing_config(
            RoutingConfigUnified(
                tenant_id="acme:acme",
                optimizer_floors={
                    "profile_selection": {
                        "min_samples_for_optimization": "not-a-number",
                    },
                },
            )
        )
        with caplog.at_level(logging.WARNING):
            resolved = _population_floor_from_config(
                "acme:acme", manager, "profile_selection"
            )
        assert resolved == (20, 6)
        warning_lines = [
            record.getMessage()
            for record in caplog.records
            if "malformed optimizer floor" in record.getMessage()
        ]
        assert warning_lines == [
            "Ignoring malformed optimizer floor for tenant 'acme:acme' "
            "optimizer 'profile_selection': "
            "{'min_samples_for_optimization': 'not-a-number'}"
        ]

    def test_raw_store_malformed_floor_entry_falls_back(self, tmp_path, monkeypatch):
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_runtime import optimization_cli
        from cogniverse_runtime.optimization_cli import _population_floor_from_config
        from cogniverse_sdk.interfaces.config_store import ConfigScope
        from tests.utils.memory_store import InMemoryConfigStore

        shipped_config = tmp_path / "config.json"
        shipped_config.write_text(
            json.dumps(
                {
                    "routing": {
                        "optimization_config": {
                            "optimizer_floors": {
                                "profile_selection": {
                                    "min_samples_for_optimization": 20,
                                    "min_unique_queries": 6,
                                }
                            }
                        }
                    }
                }
            )
        )
        monkeypatch.setattr(optimization_cli, "SHIPPED_CONFIG_PATH", shipped_config)

        manager = ConfigManager(store=InMemoryConfigStore())
        manager.store.set_config(
            tenant_id="acme:acme",
            scope=ConfigScope.ROUTING,
            service="gateway_agent",
            config_key="routing_config",
            config_value={
                "tenant_id": "acme:acme",
                "optimizer_floors": {
                    "profile_selection": "nonsense",
                },
            },
        )

        assert _population_floor_from_config(
            "acme:acme", manager, "profile_selection"
        ) == (20, 6)


class TestTrainingSelectionConfig:
    def test_routing_config_round_trips_training_selection(self):
        from cogniverse_foundation.config.unified_config import RoutingConfigUnified

        cfg = RoutingConfigUnified(
            tenant_id="acme:acme",
            training_selection={
                "simba_query_enhancement": {
                    "trainset_cap": 12,
                    "mmr_lambda": 0.5,
                    "low_confirmation_threshold": 4,
                    "downweight_age_days": 21,
                    "downweight_factor": 0.25,
                }
            },
        )
        restored = RoutingConfigUnified.from_dict(cfg.to_dict())
        assert restored == cfg

    def test_training_selection_prefers_store_then_shipped_then_defaults(
        self, tmp_path, monkeypatch
    ):
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_foundation.config.unified_config import RoutingConfigUnified
        from cogniverse_runtime import optimization_cli
        from cogniverse_runtime.optimization_cli import (
            TrainingSelectionKnobs,
            _training_selection_from_config,
        )
        from tests.utils.memory_store import InMemoryConfigStore

        shipped_config = tmp_path / "config.json"
        shipped_config.write_text(json.dumps({"routing": {"optimization_config": {}}}))
        monkeypatch.setattr(optimization_cli, "SHIPPED_CONFIG_PATH", shipped_config)

        manager = ConfigManager(store=InMemoryConfigStore())
        manager.set_routing_config(
            RoutingConfigUnified(
                tenant_id="acme:acme",
                training_selection={
                    "simba_query_enhancement": {
                        "trainset_cap": 12,
                        "mmr_lambda": 0.5,
                    }
                },
            )
        )
        assert _training_selection_from_config(
            manager, "acme:acme", "simba_query_enhancement"
        ) == TrainingSelectionKnobs(12, 0.5, 3, 14, 0.5)

        shipped_config.write_text(
            json.dumps(
                {
                    "routing": {
                        "optimization_config": {
                            "training_selection": {
                                "simba_query_enhancement": {
                                    "trainset_cap": 24,
                                    "mmr_lambda": 0.9,
                                    "low_confirmation_threshold": 5,
                                    "downweight_age_days": 21,
                                    "downweight_factor": 0.25,
                                }
                            }
                        }
                    }
                }
            )
        )
        manager = ConfigManager(store=InMemoryConfigStore())
        manager.set_routing_config(RoutingConfigUnified(tenant_id="acme:acme"))
        assert _training_selection_from_config(
            manager, "acme:acme", "simba_query_enhancement"
        ) == TrainingSelectionKnobs(24, 0.9, 5, 21, 0.25)

        shipped_config.write_text(json.dumps({"routing": {"optimization_config": {}}}))
        manager = ConfigManager(store=InMemoryConfigStore())
        manager.set_routing_config(RoutingConfigUnified(tenant_id="acme:acme"))
        assert _training_selection_from_config(
            manager, "acme:acme", "simba_query_enhancement"
        ) == TrainingSelectionKnobs(300, 0.7, 3, 14, 0.5)

    def test_malformed_selection_entry_warns_and_falls_back(
        self, tmp_path, monkeypatch, caplog
    ):
        import logging

        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_runtime import optimization_cli
        from cogniverse_runtime.optimization_cli import (
            TrainingSelectionKnobs,
            _training_selection_from_config,
        )
        from cogniverse_sdk.interfaces.config_store import ConfigScope
        from tests.utils.memory_store import InMemoryConfigStore

        shipped_config = tmp_path / "config.json"
        shipped_config.write_text(
            json.dumps(
                {
                    "routing": {
                        "optimization_config": {
                            "training_selection": {
                                "simba_query_enhancement": {
                                    "trainset_cap": 24,
                                    "mmr_lambda": 0.9,
                                }
                            }
                        }
                    }
                }
            )
        )
        monkeypatch.setattr(optimization_cli, "SHIPPED_CONFIG_PATH", shipped_config)

        manager = ConfigManager(store=InMemoryConfigStore())
        manager.store.set_config(
            tenant_id="t:t",
            scope=ConfigScope.ROUTING,
            service="gateway_agent",
            config_key="routing_config",
            config_value={
                "tenant_id": "t:t",
                "training_selection": {
                    "simba_query_enhancement": "nonsense",
                },
            },
        )
        with caplog.at_level(logging.WARNING):
            resolved = _training_selection_from_config(
                manager, "t", "simba_query_enhancement"
            )

        assert resolved == TrainingSelectionKnobs(24, 0.9, 3, 14, 0.5)
        warning_lines = [
            record.getMessage()
            for record in caplog.records
            if "malformed training_selection entry" in record.getMessage()
        ]
        assert warning_lines == [
            "tenant='t' optimizer='simba_query_enhancement' has malformed "
            "training_selection entry: 'nonsense'"
        ]


class TestRoutingConfigSerialization:
    @pytest.mark.parametrize(
        "field_name,field_value,expected_serialized,expected_warning",
        [
            (
                "optimizer_floors",
                {
                    "profile_selection": {
                        "min_samples_for_optimization": 20,
                        "min_unique_queries": 6,
                    },
                    "entity_extraction": "nonsense",
                },
                {
                    "profile_selection": {
                        "min_samples_for_optimization": 20,
                        "min_unique_queries": 6,
                    }
                },
                "Dropping malformed optimizer_floors entry for tenant 'acme:acme' optimizer 'entity_extraction': 'nonsense'",
            ),
            (
                "training_selection",
                {
                    "simba_query_enhancement": {
                        "trainset_cap": 12,
                        "mmr_lambda": 0.5,
                        "low_confirmation_threshold": 4,
                        "downweight_age_days": 21,
                        "downweight_factor": 0.25,
                    },
                    "profile_selection": "nonsense",
                },
                {
                    "simba_query_enhancement": {
                        "trainset_cap": 12,
                        "mmr_lambda": 0.5,
                        "low_confirmation_threshold": 4,
                        "downweight_age_days": 21,
                        "downweight_factor": 0.25,
                    }
                },
                "Dropping malformed training_selection entry for tenant 'acme:acme' optimizer 'profile_selection': 'nonsense'",
            ),
        ],
    )
    def test_routing_config_to_dict_drops_malformed_entries(
        self,
        field_name: str,
        field_value: dict[str, Any],
        expected_serialized: dict[str, Any],
        expected_warning: str,
        caplog,
    ):
        import logging

        from cogniverse_foundation.config.unified_config import RoutingConfigUnified

        cfg = RoutingConfigUnified(tenant_id="acme:acme", **{field_name: field_value})

        with caplog.at_level(logging.WARNING):
            serialized = cfg.to_dict()

        assert serialized[field_name] == expected_serialized
        warning_lines = [
            record.getMessage()
            for record in caplog.records
            if f"malformed {field_name} entry" in record.getMessage()
        ]
        assert warning_lines == [expected_warning]


class TestSimbaQueryEnhancement:
    """The SIMBA path trains on served calls and only serves a module that
    scores at least as well as the base module on held-out inputs."""

    def test_pairs_read_the_full_served_call(self):
        from cogniverse_runtime.optimization_cli import _query_enhancement_pairs

        spans_df = _make_spans_df(
            "cogniverse.query_enhancement",
            [
                _qe_span_row(
                    "find tutorials",
                    "find TensorFlow tutorials",
                    expansion_terms=["TensorFlow", "neural networks"],
                    grounding_context=_TF_CONTEXT,
                    source_text="src",
                    span_id="s-tut",
                ),
                _qe_span_row(
                    "robots", "Robots", expansion_terms=["machines"], span_id="s-rob"
                ),
                _qe_span_row("cats", "cats and dogs", expansion_terms=[]),
                _qe_span_row("dogs", "", expansion_terms=["puppies"]),
            ],
        )

        assert _query_enhancement_pairs(spans_df) == [
            {
                "query": "find tutorials",
                "source_text": "src",
                "grounding_context": _TF_CONTEXT,
                "enhanced_query": "find TensorFlow tutorials",
                "expansion_terms": ["TensorFlow", "neural networks"],
                "synonyms": ["s1"],
                "context": ["c1"],
                "confidence": 0.8,
                "example_id": "span:s-tut",
                "trainable": True,
            },
            {
                "query": "robots",
                "source_text": "",
                "grounding_context": "",
                "enhanced_query": "Robots",
                "expansion_terms": ["machines"],
                "synonyms": ["s1"],
                "context": ["c1"],
                "confidence": 0.8,
                "example_id": "span:s-rob",
                "trainable": False,
            },
            {
                "query": "cats",
                "source_text": "",
                "grounding_context": "",
                "enhanced_query": "cats and dogs",
                "expansion_terms": [],
                "synonyms": ["s1"],
                "context": ["c1"],
                "confidence": 0.8,
                "example_id": "span:span-0",
                "trainable": False,
            },
        ]

    def test_pairs_carry_the_span_id_as_example_id(self):
        """Every production record carries example_id = span:<span_id>, the id
        the optimizer ledger records as the example's source."""
        from cogniverse_runtime.optimization_cli import _query_enhancement_pairs

        spans_df = _make_spans_df(
            "cogniverse.query_enhancement",
            [
                _qe_span_row(
                    "q0", "q0 enhanced", expansion_terms=["a"], span_id="sid-0"
                ),
                _qe_span_row(
                    "q1", "q1 enhanced", expansion_terms=["b"], span_id="sid-1"
                ),
            ],
        )
        pairs = _query_enhancement_pairs(spans_df)
        assert [p["example_id"] for p in pairs] == ["span:sid-0", "span:sid-1"]

    @pytest.mark.parametrize(
        ("grounding_context", "enhanced", "expansion_terms", "expected"),
        [
            ("", "find video tutorials", "guides, lessons", None),
            ("", "find tutorials", "guides", None),
            ("", "Find Tutorials ", "guides", None),
            ("", "find video tutorials", "", None),
            ("", "", "guides", None),
            (_TF_CONTEXT, "find tutorials locate", "locate, discover", 0.0),
            (_TF_CONTEXT, "find tensorflow tutorials", "guides", 1.0),
            (_TF_CONTEXT, "find tutorials online", "Neural Networks, guides", 1.0),
        ],
    )
    def test_quality_scores_the_module_output(
        self, grounding_context, enhanced, expansion_terms, expected
    ):
        import dspy

        from cogniverse_runtime.optimization_cli import (
            _query_enhancement_metric,
            _query_enhancement_quality,
        )

        example = dspy.Example(
            query="find tutorials",
            source_text="",
            grounding_context=grounding_context,
        ).with_inputs("query", "source_text", "grounding_context")
        prediction = dspy.Prediction(
            enhanced_query=enhanced, expansion_terms=expansion_terms
        )

        assert _query_enhancement_quality(prediction, example) == expected
        assert _query_enhancement_metric(example, prediction) is (expected == 1.0)

    def test_quality_scores_against_source_text_exact_table(self):
        from cogniverse_runtime.optimization_cli import _query_enhancement_quality

        source = (
            "The video begins with a man wearing a blue shirt pulling heavy "
            "logs placed against each other with a thick rope."
        )
        ex = _example(query="cats", source_text=source, grounding_context="")
        scores = {
            "grounded": _query_enhancement_quality(
                _pred("cats pulling heavy logs", "heavy logs, rope"), ex
            ),
            "junk": _query_enhancement_quality(
                _pred("find The video is of zzzzzz", "zzzzzz"), ex
            ),
            "video_id": _query_enhancement_quality(
                _pred("cats v_-6dz6tBH77I", "v_-6dz6tBH77I"), ex
            ),
            "off_topic": _query_enhancement_quality(
                _pred("cats quantum chromodynamics", "quantum chromodynamics"), ex
            ),
        }
        assert scores == {
            "grounded": 1.0,
            "junk": 0.0,
            "video_id": 0.0,
            "off_topic": 0.0,
        }

    def test_quality_returns_none_for_unscoreable_example(self):
        from cogniverse_runtime.optimization_cli import (
            _query_enhancement_quality,
            _query_enhancement_scores,
        )

        ex = _example(query="cats", source_text="", grounding_context="")
        prediction = _pred("cats playing piano", "piano")
        quality = _query_enhancement_quality(prediction, ex)

        assert [quality] == [None]
        assert _query_enhancement_scores(
            _module_returning(prediction),
            [ex],
        ) == (0.0, 0)

    @pytest.mark.parametrize(
        ("baseline", "current", "candidate", "min_improvement", "expected"),
        [
            (0.5, None, 1.0, 0.0, "promote"),
            (0.5, None, 0.5, 0.0, "promote"),
            (0.5, None, 0.5, 0.05, "reject"),
            (0.5, None, 0.25, 0.0, "reject"),
            (0.5, None, None, 0.0, "reject"),
            (0.5, 0.0, 0.25, 0.0, "rollback"),
            (0.5, 0.0, None, 0.0, "rollback"),
            (0.5, 0.5, 0.25, 0.0, "keep"),
            (0.5, 0.75, 0.75, 0.05, "keep"),
            (0.5, 0.75, 0.8, 0.05, "promote"),
            (1.0, 1.0, 1.0, 0.0, "promote"),
        ],
    )
    def test_select_simba_artifact(
        self, baseline, current, candidate, min_improvement, expected
    ):
        from cogniverse_runtime.optimization_cli import _select_simba_artifact

        assert (
            _select_simba_artifact(baseline, current, candidate, min_improvement)
            == expected
        )

    @staticmethod
    def _score_by_module(module, holdout) -> tuple[float, int]:
        """Base module 0.5, the persisted artifact 0.0, the compiled candidate 1.0."""
        scored_count = sum(
            bool(
                str(getattr(example, "source_text", "") or "").strip()
                or str(getattr(example, "grounding_context", "") or "").strip()
            )
            for example in holdout
        )
        if getattr(module, "_compiled_marker", False):
            return 1.0, scored_count
        if module.enhancer.predict.demos:
            return 0.0, scored_count
        return 0.5, scored_count

    @staticmethod
    def _fake_teleprompter(*args, **kwargs):
        from cogniverse_agents.query_enhancement_agent import QueryEnhancementModule

        class _Compiler:
            def compile(self, module, trainset):
                compiled = QueryEnhancementModule()
                compiled._compiled_marker = True
                compiled.enhancer.predict.demos = [
                    ex.toDict() if hasattr(ex, "toDict") else ex for ex in trainset
                ]
                return compiled

        return _Compiler()

    def _run(
        self,
        provider,
        *,
        min_improvement: float,
        scorer=None,
        floor=(1, 1),
        config_manager=None,
    ):
        from cogniverse_runtime.optimization_cli import run_simba_optimization

        mgr = FakeTelemetryManager(provider)
        p1, p2 = _patch_infra(mgr, config_manager=config_manager)
        llm_config = SimpleNamespace(
            resolve=lambda purpose: "student-endpoint",
            resolve_teacher=_teacher_endpoint,
        )
        with (
            p1,
            p2,
            patch(
                "cogniverse_foundation.config.utils.get_config",
                return_value=SimpleNamespace(get_llm_config=lambda: llm_config),
            ),
            patch(
                "cogniverse_foundation.config.llm_factory.create_dspy_lm",
                return_value=object(),
            ),
            patch(
                "cogniverse_runtime.optimization_cli._create_teleprompter",
                side_effect=self._fake_teleprompter,
            ),
            patch(
                "cogniverse_runtime.optimization_cli._query_enhancement_scores",
                side_effect=scorer or self._score_by_module,
            ),
            patch(
                "cogniverse_runtime.optimization_cli._min_improvement_from_config",
                return_value=min_improvement,
            ),
            patch(
                "cogniverse_runtime.optimization_cli._population_floor_from_config",
                return_value=floor,
            ),
        ):
            return asyncio.run(
                run_simba_optimization(tenant_id="test:unit", lookback_hours=1)
            )

    @staticmethod
    def _persisted_state(provider) -> dict:
        blob_df = provider.datasets.datasets[
            "dspy-model-test:unit-simba_query_enhancement"
        ]
        return json.loads(blob_df.iloc[-1]["content"])

    @staticmethod
    def _lineage(provider) -> list[dict]:
        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

        return asyncio.run(
            ArtifactManager(provider, "test:unit").get_version_lineage(
                "model", "simba_query_enhancement"
            )
        )

    @staticmethod
    def _active_version(provider, key):
        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager

        state = asyncio.run(
            ArtifactManager(provider, "test:unit").get_blob_state("model", key)
        )
        return state["active"]["version"] if state["active"] else None

    def test_promotes_a_candidate_that_beats_the_base_module(self):
        rows = [
            _qe_span_row(
                f"query {i}",
                f"query {i} expanded",
                expansion_terms=["expanded"],
                source_text="src",
                span_id=f"qe-{i}",
            )
            for i in range(4)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.query_enhancement", rows)
        )

        result = self._run(provider, min_improvement=0.0)

        assert result == {
            "status": "success",
            "spans_found": 4,
            "examples": 4,
            "served_examples": 4,
            "approved_examples": 0,
            "served_scoreable_examples": 4,
            "non_trainable_examples": 0,
            "unscoreable_examples": 0,
            "training_examples": 3,
            "holdout_examples": 1,
            "holdout_source": "served",
            **_selection_block(3, 3),
            "baseline_score": 0.5,
            "current_score": None,
            "candidate_score": 1.0,
            "decision": "promote",
            "version": 1,
            "consumed_example_ids": [
                "span:qe-0",
                "span:qe-1",
                "span:qe-2",
                "span:qe-3",
            ],
        }
        # promote activated v1: the served blob and the ledger both hold it.
        state = self._persisted_state(provider)
        assert [d["query"] for d in state["enhancer.predict"]["demos"]] == [
            "query 0",
            "query 1",
            "query 2",
        ]
        lineage = self._lineage(provider)
        assert [(e["version"], e["decision"]) for e in lineage] == [(1, "promote")]
        assert lineage[0]["consumed_example_ids"] == [
            "span:qe-0",
            "span:qe-1",
            "span:qe-2",
            "span:qe-3",
        ]
        assert lineage[0]["scored"] is True
        assert self._active_version(provider, "simba_query_enhancement") == 1

    def test_fallback_shaped_served_record_is_counted_and_excluded(self):
        rows = [
            _qe_span_row(
                "fallback query",
                "fallback query",
                expansion_terms=[],
                source_text="src",
                span_id="qe-fallback",
            )
        ] + [
            _qe_span_row(
                f"query {i}",
                f"query {i} expanded",
                expansion_terms=["expanded"],
                source_text="src",
                span_id=f"qe-{i}",
            )
            for i in range(4)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.query_enhancement", rows)
        )

        result = self._run(provider, min_improvement=0.0)

        assert result == {
            "status": "success",
            "spans_found": 5,
            "examples": 5,
            "served_examples": 5,
            "approved_examples": 0,
            "served_scoreable_examples": 5,
            "non_trainable_examples": 1,
            "unscoreable_examples": 0,
            "training_examples": 3,
            "holdout_examples": 1,
            "holdout_source": "served",
            **_selection_block(3, 3),
            "baseline_score": 0.5,
            "current_score": None,
            "candidate_score": 1.0,
            "decision": "promote",
            "version": 1,
            "consumed_example_ids": [
                "span:qe-fallback",
                "span:qe-0",
                "span:qe-1",
                "span:qe-2",
                "span:qe-3",
            ],
        }
        state = self._persisted_state(provider)
        assert [d["query"] for d in state["enhancer.predict"]["demos"]] == [
            "query 0",
            "query 1",
            "query 2",
        ]

    def test_rolls_a_worse_persisted_artifact_back_to_the_base_state(self):
        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
        from cogniverse_agents.query_enhancement_agent import QueryEnhancementModule

        # Only identity enhancements were served (nothing trainable), and the
        # persisted artifact is what produced them.
        rows = [
            _qe_span_row(
                f"query {i}",
                f"Query {i}",
                expansion_terms=[],
                source_text="src",
                span_id=f"qe-{i}",
            )
            for i in range(4)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.query_enhancement", rows)
        )
        degraded = QueryEnhancementModule()
        degraded.enhancer.predict.demos = [
            {"query": "old", "enhanced_query": "old related content"}
        ]
        asyncio.run(
            ArtifactManager(provider, "test:unit").save_blob(
                "model",
                "simba_query_enhancement",
                json.dumps(degraded.dump_state(), default=str),
            )
        )

        result = self._run(provider, min_improvement=0.0)

        assert result == {
            "status": "success",
            "spans_found": 4,
            "examples": 4,
            "served_examples": 4,
            "approved_examples": 0,
            "served_scoreable_examples": 4,
            "non_trainable_examples": 3,
            "unscoreable_examples": 0,
            "training_examples": 0,
            "holdout_examples": 1,
            "holdout_source": "served",
            **_selection_block(0, 0),
            "baseline_score": 0.5,
            "current_score": 0.0,
            "candidate_score": None,
            "decision": "rollback",
            "version": 1,
            "consumed_example_ids": [
                "span:qe-0",
                "span:qe-1",
                "span:qe-2",
                "span:qe-3",
            ],
        }
        # rollback activated a base-state version; the served blob is the base.
        assert self._persisted_state(provider) == json.loads(
            json.dumps(QueryEnhancementModule().dump_state(), default=str)
        )
        lineage = self._lineage(provider)
        assert [(e["version"], e["decision"]) for e in lineage] == [(1, "rollback")]
        assert lineage[0]["score"] is None
        assert self._active_version(provider, "simba_query_enhancement") == 1

    def test_keeps_a_persisted_artifact_the_candidate_does_not_beat(self):
        from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
        from cogniverse_agents.query_enhancement_agent import QueryEnhancementModule

        rows = [
            _qe_span_row(
                f"query {i}",
                f"query {i} expanded",
                expansion_terms=["expanded"],
                source_text="src",
                span_id=f"qe-{i}",
            )
            for i in range(4)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.query_enhancement", rows)
        )
        served = QueryEnhancementModule()
        served.enhancer.predict.demos = [{"query": "old", "enhanced_query": "old x"}]
        served_state = json.dumps(served.dump_state(), default=str)
        asyncio.run(
            ArtifactManager(provider, "test:unit").save_blob(
                "model", "simba_query_enhancement", served_state
            )
        )

        # Every module scores 1.0: the candidate cannot clear the 0.05 bar
        # over the served artifact, which is not worse than base, so it stays.
        result = self._run(
            provider,
            min_improvement=0.05,
            scorer=lambda module, holdout: (1.0, len(holdout)),
        )

        assert result == {
            "status": "success",
            "spans_found": 4,
            "examples": 4,
            "served_examples": 4,
            "approved_examples": 0,
            "served_scoreable_examples": 4,
            "non_trainable_examples": 0,
            "unscoreable_examples": 0,
            "training_examples": 3,
            "holdout_examples": 1,
            "holdout_source": "served",
            **_selection_block(3, 3),
            "baseline_score": 1.0,
            "current_score": 1.0,
            "candidate_score": 1.0,
            "decision": "keep",
            "version": 1,
            "consumed_example_ids": [
                "span:qe-0",
                "span:qe-1",
                "span:qe-2",
                "span:qe-3",
            ],
        }
        # keep records the candidate as a version but never activates it: the
        # incumbent blob is untouched and no active pointer is written.
        assert self._persisted_state(provider) == json.loads(served_state)
        lineage = self._lineage(provider)
        assert [(e["version"], e["decision"]) for e in lineage] == [(1, "keep")]
        assert lineage[0]["score"] == 1.0
        assert self._active_version(provider, "simba_query_enhancement") is None

    def test_score_failure_preserves_selection_block(self):
        rows = [
            _qe_span_row(
                f"query {i}",
                f"query {i} expanded",
                expansion_terms=["expanded"],
                source_text="src",
                span_id=f"qe-{i}",
            )
            for i in range(4)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.query_enhancement", rows)
        )

        def failing_score(module, holdout):
            raise RuntimeError("simba scorer failed")

        result = self._run(provider, min_improvement=0.0, scorer=failing_score)

        assert result == {
            "status": "failed",
            "error": "simba scorer failed",
            **_selection_block(3, 3),
        }

    def test_refuses_when_no_holdout_rows_are_scoreable(self):
        rows = [
            _qe_span_row(
                f"query {i}",
                f"query {i} expanded",
                expansion_terms=["expanded"],
                source_text="src",
                span_id=f"qe-{i}",
            )
            for i in range(4)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.query_enhancement", rows)
        )

        result = self._run(
            provider,
            min_improvement=0.0,
            scorer=lambda module, holdout: (0.0, 0),
        )

        assert result == {
            "status": "no_eval_material",
            "spans_found": 4,
            "examples": 4,
            "served_scoreable_examples": 4,
            "non_trainable_examples": 0,
            "unscoreable_examples": 0,
            "training_examples": 3,
            "holdout_examples": 0,
            "holdout_source": "served",
            **_selection_block(3, 3),
        }
        assert provider.datasets.created == []
        assert self._active_version(provider, "simba_query_enhancement") is None

    def test_below_count_floor_persists_version_without_activating(self):
        rows = [
            _qe_span_row(
                f"query {i}",
                f"query {i} expanded",
                expansion_terms=["expanded"],
                source_text="src",
                span_id=f"qe-{i}",
            )
            for i in range(4)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.query_enhancement", rows)
        )

        result = self._run(provider, min_improvement=0.0, floor=(100, 1))

        assert result == {
            "status": "insufficient_population",
            "spans_found": 4,
            "examples": 4,
            "distinct_queries": 4,
            "min_samples": 100,
            "min_unique_queries": 1,
            "version": 1,
        }
        assert "selection" not in result
        lineage = self._lineage(provider)
        assert [(e["version"], e["decision"]) for e in lineage] == [
            (1, "insufficient_population")
        ]
        assert lineage[0]["score"] is None
        assert self._active_version(provider, "simba_query_enhancement") is None

    def test_below_distinct_query_floor_is_insufficient(self):
        rows = [
            _qe_span_row(
                "same query",
                "same query expanded",
                expansion_terms=["expanded"],
                span_id=f"qe-{i}",
            )
            for i in range(100)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.query_enhancement", rows)
        )

        result = self._run(provider, min_improvement=0.0, floor=(1, 3))

        assert result == {
            "status": "insufficient_population",
            "spans_found": 100,
            "examples": 100,
            "distinct_queries": 1,
            "min_samples": 1,
            "min_unique_queries": 3,
            "version": 1,
        }
        assert "selection" not in result
        lineage = self._lineage(provider)
        assert [(e["version"], e["decision"]) for e in lineage] == [
            (1, "insufficient_population")
        ]
        assert lineage[0]["score"] is None
        assert self._active_version(provider, "simba_query_enhancement") is None

    def test_population_exactly_at_floor_proceeds_to_promotion(self):
        rows = [
            _qe_span_row(
                f"query {i}",
                f"query {i} expanded",
                expansion_terms=["expanded"],
                source_text="src",
                span_id=f"qe-{i}",
            )
            for i in range(4)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.query_enhancement", rows)
        )

        result = self._run(provider, min_improvement=0.0, floor=(4, 4))

        assert result == {
            "status": "success",
            "spans_found": 4,
            "examples": 4,
            "served_examples": 4,
            "approved_examples": 0,
            "served_scoreable_examples": 4,
            "non_trainable_examples": 0,
            "unscoreable_examples": 0,
            "training_examples": 3,
            "holdout_examples": 1,
            "holdout_source": "served",
            **_selection_block(3, 3),
            "baseline_score": 0.5,
            "current_score": None,
            "candidate_score": 1.0,
            "decision": "promote",
            "version": 1,
            "consumed_example_ids": [
                "span:qe-0",
                "span:qe-1",
                "span:qe-2",
                "span:qe-3",
            ],
        }
        lineage = self._lineage(provider)
        assert [(e["version"], e["decision"]) for e in lineage] == [(1, "promote")]
        assert lineage[0]["score"] == 1.0
        assert self._active_version(provider, "simba_query_enhancement") == 1

    def test_single_record_is_scoreable_and_rejects_without_activation(self):
        provider = FakeTelemetryProvider(
            _make_spans_df(
                "cogniverse.query_enhancement",
                [
                    _qe_span_row(
                        "q",
                        "q expanded",
                        expansion_terms=["expanded"],
                        source_text="src",
                    )
                ],
            )
        )

        result = self._run(provider, min_improvement=0.0)

        assert result == {
            "status": "success",
            "spans_found": 1,
            "examples": 1,
            "served_examples": 1,
            "approved_examples": 0,
            "served_scoreable_examples": 1,
            "non_trainable_examples": 0,
            "unscoreable_examples": 0,
            "training_examples": 0,
            "holdout_examples": 1,
            "holdout_source": "served",
            **_selection_block(0, 0),
            "baseline_score": 0.5,
            "current_score": None,
            "candidate_score": None,
            "decision": "reject",
            "version": 1,
            "consumed_example_ids": ["span:span-0"],
        }
        lineage = self._lineage(provider)
        assert [(e["version"], e["decision"]) for e in lineage] == [(1, "reject")]
        assert lineage[0]["consumed_example_ids"] == ["span:span-0"]
        assert lineage[0]["scored"] is False
        assert lineage[0]["score"] is None
        assert self._active_version(provider, "simba_query_enhancement") is None

    def test_training_selection_store_override_binds_simba_canonical_key(self):
        rows = [
            _qe_span_row(
                f"query {i}",
                f"query {i} expanded",
                expansion_terms=["expanded"],
                source_text="src",
                span_id=f"qe-{i}",
            )
            for i in range(4)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.query_enhancement", rows)
        )
        config_manager = _training_selection_config_manager(
            "test:unit",
            {"simba_query_enhancement": {"trainset_cap": 42}},
        )

        result = self._run(
            provider,
            min_improvement=0.0,
            config_manager=config_manager,
        )

        assert result["selection"]["cap"] == 42, result

    def test_unscoreable_records_train_last_and_are_counted(self):
        """A record without source text or grounding context can never pass
        the bootstrap metric, so every scoreable record precedes it in the
        trainset and the run reports how many there were."""
        rows = [
            _qe_span_row(
                "plain 0",
                "plain 0 expanded",
                expansion_terms=["expanded"],
                span_id="qe-u0",
            ),
            _qe_span_row(
                "plain 1",
                "plain 1 expanded",
                expansion_terms=["expanded"],
                span_id="qe-u1",
            ),
            _qe_span_row(
                "grounded 2",
                "grounded 2 expanded",
                expansion_terms=["expanded"],
                source_text="src",
                span_id="qe-s2",
            ),
            _qe_span_row(
                "grounded 3",
                "grounded 3 expanded",
                expansion_terms=["expanded"],
                source_text="src",
                span_id="qe-s3",
            ),
            _qe_span_row(
                "grounded 4",
                "grounded 4 expanded",
                expansion_terms=["expanded"],
                source_text="src",
                span_id="qe-s4",
            ),
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.query_enhancement", rows)
        )

        result = self._run(provider, min_improvement=0.0)

        assert result == {
            "status": "success",
            "spans_found": 5,
            "examples": 5,
            "served_examples": 5,
            "approved_examples": 0,
            "served_scoreable_examples": 3,
            "non_trainable_examples": 0,
            "unscoreable_examples": 2,
            "training_examples": 4,
            "holdout_examples": 1,
            "holdout_source": "served",
            **_selection_block(4, 4),
            "baseline_score": 0.5,
            "current_score": None,
            "candidate_score": 1.0,
            "decision": "promote",
            "version": 1,
            "consumed_example_ids": [
                "span:qe-u0",
                "span:qe-u1",
                "span:qe-s2",
                "span:qe-s3",
                "span:qe-s4",
            ],
        }
        demos = self._persisted_state(provider)["enhancer.predict"]["demos"]
        assert [demo["query"] for demo in demos] == [
            "grounded 2",
            "grounded 3",
            "plain 0",
            "plain 1",
        ]

    def test_scoreable_first_keeps_relative_order(self):
        from cogniverse_runtime.optimization_cli import _scoreable_first

        records = [
            {"query": "a", "source_text": "", "grounding_context": ""},
            {"query": "b", "source_text": "s", "grounding_context": ""},
            {"query": "c", "source_text": "", "grounding_context": "g"},
            {"query": "d", "source_text": "", "grounding_context": ""},
        ]

        assert _scoreable_first(records) == (
            [records[1], records[2], records[0], records[3]],
            2,
        )


class TestProfileSelectionOptimization:
    @staticmethod
    def _served_state() -> str:
        from cogniverse_agents.profile_selection_agent import ProfileSelectionModule

        state = json.loads(
            json.dumps(ProfileSelectionModule().dump_state(), default=str)
        )
        state["selector.predict"]["demos"] = [
            {
                "query": "find basketball highlights",
                "available_profiles": (
                    "video_colpali_smol500_mv_frame,video_colqwen_omni_mv_chunk_30s"
                ),
                "selected_profile": "video_colpali_smol500_mv_frame",
                "confidence": "0.9",
                "reasoning": "Short clip search works best with frame-level ColPali",
                "query_intent": "video_search",
                "modality": "video",
                "complexity": "simple",
            }
        ]
        return json.dumps(state, default=str)

    async def _run(
        self,
        provider,
        *,
        current_blob: str,
        floor: tuple[int, int],
        min_improvement: float = 0.05,
        score: float = 1.0,
        score_by_module=None,
        config_manager=None,
        ground_truth_blob: str | None = None,
        ground_truth_present: bool = True,
        ground_truth_error: Exception | None = None,
        scoreable_fields: bool = True,
        search_service_factory=None,
    ):
        from cogniverse_agents.optimizer.profile_selection_ground_truth import (
            canonicalize_profile_selection_ground_truth_rows,
        )
        from cogniverse_runtime.optimization_cli import (
            ProfileLabelDerivationResult,
            run_profile_optimization,
        )

        if ground_truth_blob is None:
            ground_truth_blob = json.dumps(
                [
                    {
                        "query": "find basketball highlights",
                        "expected_videos": ["video_colpali_smol500_mv_frame"],
                        "ground_truth": "basketball",
                        "query_type": "question",
                        "source": "fixture",
                    }
                ],
                separators=(",", ":"),
            )

        state = {
            "active_blob": current_blob,
            "versioned_saves": [],
            "activate_calls": [],
            "load_blob_calls": [],
            "query_spans_calls": [],
            "score_calls": [],
            "ground_truth_blob": ground_truth_blob,
            "ground_truth_present": ground_truth_present,
            "ground_truth_error": ground_truth_error,
        }

        spans_df = provider.traces._spans_df.copy(deep=True)
        span_rows = spans_df.to_dict("records")
        label_records = []
        label_map = {}
        candidate_profiles = []
        for position, row in enumerate(span_rows):
            query = str(row.get("attributes.input.value", "") or "").strip()
            available_raw = row.get("attributes.available_profiles", "")
            if isinstance(available_raw, list):
                available_profiles = [
                    str(profile).strip()
                    for profile in available_raw
                    if str(profile).strip()
                ]
            else:
                available_profiles = [
                    profile.strip()
                    for profile in str(available_raw or "").split(",")
                    if profile.strip()
                ]
            output_raw = row.get("attributes.output.value", "{}")
            if isinstance(output_raw, str):
                output = json.loads(output_raw)
            elif isinstance(output_raw, dict):
                output = output_raw
            else:
                output = {}
            selected_profile = str(output.get("selected_profile", "") or "").strip()
            if not candidate_profiles and available_profiles:
                candidate_profiles = list(available_profiles)
            if query and selected_profile:
                label_map[query] = selected_profile
                record = {
                    "query": query,
                    "available_profiles": ", ".join(available_profiles),
                    "selected_profile": selected_profile,
                    "confidence": float(output.get("confidence", 0.0) or 0.0),
                    "reasoning": str(
                        output.get("reasoning") or f"Selected {selected_profile}"
                    ),
                    "query_intent": str(output.get("intent") or "video_search"),
                    "modality": str(output.get("modality") or "video"),
                    "complexity": str(output.get("complexity") or "simple"),
                    "example_id": f"span:profile-label:{position}",
                }
                if scoreable_fields:
                    record["source_text"] = ", ".join(available_profiles)
                    record["grounding_context"] = selected_profile
                label_records.append(record)

        label_source = ProfileLabelDerivationResult(label_map, label_records, [])

        def _label_source_from_ground_truth(
            *,
            config,
            config_manager,
            tenant_id,
            candidate_profiles,
            schema_loader,
            ground_truth_rows,
        ):
            del config, config_manager, schema_loader
            assert tenant_id == "test:unit"
            assert list(candidate_profiles) == candidate_profiles_expected
            assert (
                ground_truth_rows
                == canonicalize_profile_selection_ground_truth_rows(
                    json.loads(state["ground_truth_blob"])
                )
            )
            state["label_source_ground_truth_rows"] = ground_truth_rows
            return label_source

        candidate_profiles_expected = list(candidate_profiles)

        class FakeArtifactManager:
            def __init__(self, received_provider, tenant_id):
                assert received_provider is provider
                assert tenant_id == "test:unit"
                self._tenant_id = tenant_id

            async def load_blob(self, kind, key):
                state["load_blob_calls"].append((kind, key))
                if (kind, key) == ("config", "profile_selection_ground_truth"):
                    error = state["ground_truth_error"]
                    if error is not None:
                        raise error
                    if not state["ground_truth_present"]:
                        return None
                    return state["ground_truth_blob"]
                assert (kind, key) == ("model", "profile_selection")
                return state["active_blob"]

            async def save_blob_versioned(
                self,
                kind,
                key,
                content,
                *,
                consumed_example_ids,
                decision,
                scored,
                score,
                base_score,
                candidate_score,
                extra_ledger_fields=None,
            ):
                state["versioned_saves"].append(
                    {
                        "kind": kind,
                        "key": key,
                        "content": content,
                        "consumed_example_ids": list(consumed_example_ids),
                        "decision": decision,
                        "scored": scored,
                        "score": score,
                        "base_score": base_score,
                        "candidate_score": candidate_score,
                        "extra_ledger_fields": dict(extra_ledger_fields or {}),
                    }
                )
                return f"artifact-{len(state['versioned_saves'])}", len(
                    state["versioned_saves"]
                )

            async def activate_version(self, kind, key, version):
                state["activate_calls"].append((kind, key, version))
                state["active_blob"] = state["versioned_saves"][version - 1]["content"]
                return {"active": {"version": version, "activated_at": "now"}}

            async def get_version_lineage(self, kind, key):
                assert (kind, key) == ("model", "profile_selection")
                return []

        class FakeTeleprompter:
            def compile(self, module, trainset):
                state["compiled_module"] = type(module).__name__
                state["trainset"] = [
                    example.toDict() if hasattr(example, "toDict") else example
                    for example in trainset
                ]

                class Compiled:
                    def dump_state(self):
                        return {"compiled": "profile"}

                return Compiled()

        class FakeLLMConfig:
            def resolve(self, purpose):
                return "student-endpoint"

            def resolve_teacher(self):
                return _teacher_endpoint()

        class FakeConfig:
            def get_llm_config(self):
                return FakeLLMConfig()

        mgr = FakeTelemetryManager(provider)

        async def _query_spans_by_name(
            telemetry_manager,
            telemetry_provider,
            tenant_id,
            span_name,
            lookback_hours,
        ):
            state["query_spans_calls"].append((tenant_id, span_name, lookback_hours))
            return provider.traces._spans_df.copy(deep=True)

        def _score(module, holdout):
            state["score_calls"].append(
                {
                    "module": type(module).__name__,
                    "holdout": len(holdout),
                }
            )
            if score_by_module is not None:
                return score_by_module(module, holdout)
            return score

        p1, p2 = _patch_infra(mgr, config_manager=config_manager)
        with (
            p1,
            p2,
            patch(
                "cogniverse_agents.profile_selection_agent.tenant_usable_profile_names",
                return_value=candidate_profiles,
            ),
            patch(
                "cogniverse_foundation.config.utils.get_config",
                return_value=FakeConfig(),
            ),
            patch(
                "cogniverse_foundation.config.llm_factory.create_dspy_lm",
                return_value=object(),
            ),
            patch(
                "cogniverse_runtime.optimization_cli._create_teleprompter",
                return_value=FakeTeleprompter(),
            ),
            (
                patch(
                    "cogniverse_agents.search.service.SearchService",
                    search_service_factory,
                )
                if search_service_factory is not None
                else patch(
                    "cogniverse_runtime.optimization_cli._profile_selection_label_source",
                    side_effect=_label_source_from_ground_truth,
                )
            ),
            patch(
                "cogniverse_runtime.optimization_cli._query_spans_by_name",
                side_effect=_query_spans_by_name,
            ),
            patch(
                "cogniverse_runtime.optimization_cli._profile_selection_scores",
                side_effect=_score,
            ),
            patch(
                "cogniverse_runtime.optimization_cli._min_improvement_from_config",
                return_value=min_improvement,
            ),
            patch(
                "cogniverse_runtime.optimization_cli._population_floor_from_config",
                return_value=floor,
            ),
            patch(
                "cogniverse_agents.optimizer.artifact_manager.ArtifactManager",
                FakeArtifactManager,
            ),
            patch("dspy.configure", lambda **kwargs: None),
        ):
            result = await run_profile_optimization(
                tenant_id="test:unit", lookback_hours=1
            )
        return state, result

    @staticmethod
    def _profile_selection_score_by_module(module, holdout) -> float:
        state = json.loads(json.dumps(module.dump_state(), default=str))
        if state.get("compiled") == "profile":
            return 0.0
        if state.get("selector.predict", {}).get("demos", []):
            return 0.0
        return 1.0

    def test_profile_selection_quality_exact_table(self):
        from cogniverse_runtime.optimization_cli import _profile_selection_quality

        ex = _profile_example(
            selected="video_colpali_smol500_mv_frame",
            available=[
                "video_colpali_smol500_mv_frame",
                "wiki_semantic",
            ],
        )
        assert {
            "exact": _profile_selection_quality(
                _sel("video_colpali_smol500_mv_frame"), ex
            ),
            "other_available": _profile_selection_quality(_sel("wiki_semantic"), ex),
            "not_in_pool": _profile_selection_quality(_sel("image_colpali_mv"), ex),
            "empty": _profile_selection_quality(_sel(""), ex),
        } == {
            "exact": 1.0,
            "other_available": 0.0,
            "not_in_pool": 0.0,
            "empty": 0.0,
        }

    def test_profile_selection_quality_rejects_out_of_pool_predictions(self):
        from cogniverse_runtime.optimization_cli import _profile_selection_quality

        ex = _profile_example(
            selected="video_colpali_smol500_mv_frame",
            available=[
                "video_colpali_smol500_mv_frame",
                "video_colqwen_omni_mv_chunk_30s",
            ],
        )
        assert _profile_selection_quality(_sel("video_videoprism_base"), ex) == 0.0

    def test_profile_selection_floor_gap_is_exact(self):
        from cogniverse_runtime.optimization_cli import (
            _profile_selection_scores,
            derive_profile_labels,
        )

        queries = [
            _profile_selection_query_record(
                "q1",
                expected_videos=["v1"],
                ground_truth="one",
            ),
            _profile_selection_query_record(
                "q2",
                expected_videos=["v2"],
                ground_truth="two",
            ),
            _profile_selection_query_record(
                "q3",
                expected_videos=["v3"],
                ground_truth="three",
            ),
            _profile_selection_query_record(
                "q4",
                expected_videos=["v4"],
                ground_truth="four",
            ),
        ]
        candidate_profiles = [
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
        ]
        corpus = {
            ("q1", "video_colpali_smol500_mv_frame"): [_profile_hit("v1.mp4")],
            ("q1", "video_colqwen_omni_mv_chunk_30s"): [_profile_hit("z1.mp4")],
            ("q2", "video_colpali_smol500_mv_frame"): [_profile_hit("z2.mp4")],
            ("q2", "video_colqwen_omni_mv_chunk_30s"): [_profile_hit("v2.mp4")],
            ("q3", "video_colpali_smol500_mv_frame"): [_profile_hit("v3.mp4")],
            ("q3", "video_colqwen_omni_mv_chunk_30s"): [_profile_hit("z3.mp4")],
            ("q4", "video_colpali_smol500_mv_frame"): [_profile_hit("z4.mp4")],
            ("q4", "video_colqwen_omni_mv_chunk_30s"): [_profile_hit("v4.mp4")],
        }

        def retrieve(query: str, profile: str) -> list[dict[str, Any]]:
            return corpus[(query, profile)]

        labels = derive_profile_labels(
            queries, candidate_profiles, retrieve, title_fields=_VIDEO_TITLE_FIELDS
        )
        assert labels == {
            "q1": "video_colpali_smol500_mv_frame",
            "q2": "video_colqwen_omni_mv_chunk_30s",
            "q3": "video_colpali_smol500_mv_frame",
            "q4": "video_colqwen_omni_mv_chunk_30s",
        }
        assert labels.excluded_count == 0
        assert labels.excluded_queries == ()

        holdout = [
            _profile_example(
                query=query["query"],
                available=candidate_profiles,
                selected=labels[query["query"]],
            )
            for query in queries
        ]
        correct_selector = _ProfileSelectionByQuery(
            labels, default="video_colpali_smol500_mv_frame"
        )

        def broken_selector(query: str, available_profiles: str):
            del query, available_profiles
            return _sel("video_colpali_smol500_mv_frame")

        scores = {
            "correct": _profile_selection_scores(correct_selector, holdout),
            "broken": _profile_selection_scores(broken_selector, holdout),
        }
        assert scores == {"correct": 1.0, "broken": 0.5}

    def test_profile_selection_labels_surface_explicit_exclusions(self):
        from cogniverse_runtime.optimization_cli import derive_profile_labels

        queries = [
            _profile_selection_query_record(
                "q1",
                expected_videos=["v1"],
                ground_truth="one",
            ),
            _profile_selection_query_record(
                "q2",
                expected_videos=["v2"],
                ground_truth="two",
            ),
            _profile_selection_query_record(
                "q3",
                expected_videos=["missing"],
                ground_truth="three",
            ),
        ]
        candidate_profiles = [
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
        ]
        corpus = {
            ("q1", "video_colpali_smol500_mv_frame"): [_profile_hit("v1.mp4")],
            ("q1", "video_colqwen_omni_mv_chunk_30s"): [_profile_hit("z1.mp4")],
            ("q2", "video_colpali_smol500_mv_frame"): [_profile_hit("z2.mp4")],
            ("q2", "video_colqwen_omni_mv_chunk_30s"): [_profile_hit("v2.mp4")],
            ("q3", "video_colpali_smol500_mv_frame"): [_profile_hit("z3.mp4")],
            ("q3", "video_colqwen_omni_mv_chunk_30s"): [_profile_hit("z4.mp4")],
        }

        def retrieve(query: str, profile: str) -> list[dict[str, Any]]:
            return corpus[(query, profile)]

        labels = derive_profile_labels(
            queries, candidate_profiles, retrieve, title_fields=_VIDEO_TITLE_FIELDS
        )
        assert labels == {
            "q1": "video_colpali_smol500_mv_frame",
            "q2": "video_colqwen_omni_mv_chunk_30s",
        }
        assert labels.excluded_count == 1
        assert labels.excluded_queries == ("q3",)

    def test_profile_selection_media_type_filter_keeps_video_candidates_only(self):
        from cogniverse_runtime.optimization_cli import derive_profile_labels

        video_profile = "video_colpali_smol500_mv_frame"
        document_profile = "document_text_semantic"
        queries = [
            _profile_selection_query_record(
                "q1",
                expected_videos=["v1"],
                ground_truth="one",
            ),
            _profile_selection_query_record(
                "q2",
                expected_videos=["v2"],
                ground_truth="two",
            ),
        ]
        candidate_profiles = [video_profile, document_profile]
        title_fields = {
            video_profile: "video_title",
            document_profile: "document_title",
        }
        profile_types = {
            video_profile: "video",
            document_profile: "document",
        }
        corpus = {
            ("q1", video_profile): [_profile_hit("v1.mp4")],
            ("q1", document_profile): [
                _profile_hit("v1.txt", title_field="document_title")
            ],
            ("q2", video_profile): [_profile_hit("v2.mp4")],
            ("q2", document_profile): [
                _profile_hit("v2.txt", title_field="document_title")
            ],
        }

        def retrieve(query: str, profile: str) -> list[dict[str, Any]]:
            return corpus[(query, profile)]

        labels = derive_profile_labels(
            queries,
            candidate_profiles,
            retrieve,
            title_fields=title_fields,
            profile_types=profile_types,
        )
        assert labels == {
            "q1": video_profile,
            "q2": video_profile,
        }
        assert labels.records == (
            {
                "query": "q1",
                "ground_truth": "one",
                "query_type": "question",
                "source": "synthetic.json",
                "expected_videos": ["v1"],
                "available_profiles": [video_profile],
                "selected_profile": video_profile,
                "confidence": 1.0,
                "reasoning": f"{video_profile} recovered v1",
                "example_id": "span:profile-label:0",
            },
            {
                "query": "q2",
                "ground_truth": "two",
                "query_type": "question",
                "source": "synthetic.json",
                "expected_videos": ["v2"],
                "available_profiles": [video_profile],
                "selected_profile": video_profile,
                "confidence": 1.0,
                "reasoning": f"{video_profile} recovered v2",
                "example_id": "span:profile-label:1",
            },
        )
        assert labels.exclusions == ()
        assert labels.labels_by_profile == {video_profile: 2}
        assert labels.exclusions_by_reason == {}

        def retrieve_no_video(query: str, profile: str) -> list[dict[str, Any]]:
            del query, profile
            raise AssertionError(
                "retrieval must not run without a serving video profile"
            )

        document_only = derive_profile_labels(
            [
                _profile_selection_query_record(
                    "q3",
                    expected_videos=["v3"],
                    ground_truth="three",
                )
            ],
            [document_profile],
            retrieve_no_video,
            title_fields={document_profile: "document_title"},
            profile_types={document_profile: "document"},
        )
        assert document_only == {}
        assert document_only.records == ()
        assert document_only.exclusions == (
            {
                "query": "q3",
                "reason": "no_profile_serves_media_type",
                "expected_videos": ["v3"],
                "position": 0,
                "required_media_type": "video",
                "candidate_profiles": [],
            },
        )
        assert document_only.labels_by_profile == {}
        assert document_only.exclusions_by_reason == {"no_profile_serves_media_type": 1}

    def test_profile_selection_backend_outage_raises_for_query(self):
        from cogniverse_runtime.optimization_cli import derive_profile_labels

        queries = [
            _profile_selection_query_record(
                "q-outage",
                expected_videos=["v1"],
                ground_truth="one",
            )
        ]
        candidate_profiles = [
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
        ]

        def retrieve(query: str, profile: str) -> list[dict[str, Any]]:
            del query, profile
            raise ConnectionError("backend unavailable")

        with pytest.raises(RuntimeError) as err:
            derive_profile_labels(
                queries,
                candidate_profiles,
                retrieve,
                title_fields=_VIDEO_TITLE_FIELDS,
            )

        assert (
            str(err.value) == "Profile selection retrieval failed for query 'q-outage'"
        )
        assert type(err.value.__cause__) is ConnectionError
        assert str(err.value.__cause__) == "backend unavailable"

    def test_profile_selection_record_contains_only_derivable_fields(self):
        from cogniverse_runtime.optimization_cli import derive_profile_labels

        queries = [
            _profile_selection_query_record(
                "man lifting",
                expected_videos=["v_-HpCLXdtcas"],
                ground_truth=(
                    "The man is lifting a barbell over his head. He is wearing a "
                    "black polo shirt and shorts."
                ),
                query_type="answer_phrase",
                source="generic_qa.json",
            )
        ]
        candidate_profiles = [
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
        ]
        corpus = {
            ("man lifting", "video_colpali_smol500_mv_frame"): [
                _profile_hit("v_-HpCLXdtcas.mp4")
            ],
            ("man lifting", "video_colqwen_omni_mv_chunk_30s"): [
                _profile_hit("z1.mp4")
            ],
        }

        def retrieve(query: str, profile: str) -> list[dict[str, Any]]:
            return corpus[(query, profile)]

        labels = derive_profile_labels(
            queries, candidate_profiles, retrieve, title_fields=_VIDEO_TITLE_FIELDS
        )
        assert labels == {
            "man lifting": "video_colpali_smol500_mv_frame",
        }
        assert labels.records == (
            {
                "query": "man lifting",
                "ground_truth": (
                    "The man is lifting a barbell over his head. He is wearing a "
                    "black polo shirt and shorts."
                ),
                "query_type": "answer_phrase",
                "source": "generic_qa.json",
                "expected_videos": ["v_-HpCLXdtcas"],
                "available_profiles": [
                    "video_colpali_smol500_mv_frame",
                    "video_colqwen_omni_mv_chunk_30s",
                ],
                "selected_profile": "video_colpali_smol500_mv_frame",
                "confidence": 1.0,
                "reasoning": "video_colpali_smol500_mv_frame recovered v_-HpCLXdtcas",
                "example_id": "span:profile-label:0",
            },
        )
        assert labels.excluded_count == 0
        assert labels.excluded_queries == ()

    def test_profile_selection_tie_is_explicitly_reported(self):
        from cogniverse_runtime.optimization_cli import derive_profile_labels

        queries = [
            _profile_selection_query_record(
                "ambiguous query",
                expected_videos=["v1"],
                ground_truth="one",
            )
        ]
        candidate_profiles = [
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
        ]

        def retrieve(query: str, profile: str) -> list[dict[str, Any]]:
            del query, profile
            return [_profile_hit("v1.mp4")]

        labels = derive_profile_labels(
            queries, candidate_profiles, retrieve, title_fields=_VIDEO_TITLE_FIELDS
        )
        assert labels == {}
        assert labels.records == ()
        assert labels.excluded_count == 1
        assert labels.excluded_queries == ("ambiguous query",)
        assert labels.exclusions == (
            {
                "query": "ambiguous query",
                "reason": "ambiguous_profile_tie",
                "expected_videos": ["v1"],
                "tied_profiles": [
                    "video_colpali_smol500_mv_frame",
                    "video_colqwen_omni_mv_chunk_30s",
                ],
                "best_score": 1.0,
                "position": 0,
            },
        )

    def test_profile_selection_label_matches_preserved_upload_basename(self):
        from cogniverse_runtime.optimization_cli import derive_profile_labels

        queries = [
            _profile_selection_query_record(
                "man lifting",
                expected_videos=["v_-6dz6tBH77I"],
                ground_truth="one",
            )
        ]
        candidate_profiles = [
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
        ]
        corpus = {
            ("man lifting", "video_colpali_smol500_mv_frame"): [
                _profile_hit("v_-6dz6tBH77I.mp4")
            ],
            ("man lifting", "video_colqwen_omni_mv_chunk_30s"): [
                _profile_hit("v_-cAcA8dO7kA.mp4")
            ],
        }

        def retrieve(query: str, profile: str) -> list[dict[str, Any]]:
            return corpus[(query, profile)]

        labels = derive_profile_labels(
            queries, candidate_profiles, retrieve, title_fields=_VIDEO_TITLE_FIELDS
        )
        assert labels == {"man lifting": "video_colpali_smol500_mv_frame"}
        assert labels.records == (
            {
                "query": "man lifting",
                "ground_truth": "one",
                "query_type": "question",
                "source": "synthetic.json",
                "expected_videos": ["v_-6dz6tBH77I"],
                "available_profiles": [
                    "video_colpali_smol500_mv_frame",
                    "video_colqwen_omni_mv_chunk_30s",
                ],
                "selected_profile": "video_colpali_smol500_mv_frame",
                "confidence": 1.0,
                "reasoning": "video_colpali_smol500_mv_frame recovered v_-6dz6tBH77I",
                "example_id": "span:profile-label:0",
            },
        )
        assert labels.exclusions == ()

    def test_profile_selection_unmatched_titles_are_excluded(self):
        from cogniverse_runtime.optimization_cli import derive_profile_labels

        queries = [
            _profile_selection_query_record(
                "man lifting",
                expected_videos=["v_-6dz6tBH77I"],
                ground_truth="one",
            )
        ]
        candidate_profiles = [
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
        ]
        corpus = {
            ("man lifting", "video_colpali_smol500_mv_frame"): [
                _profile_hit("v_-cAcA8dO7kA.mp4")
            ],
            ("man lifting", "video_colqwen_omni_mv_chunk_30s"): [
                _profile_hit("v_-IMXSEIabMM.mp4")
            ],
        }

        def retrieve(query: str, profile: str) -> list[dict[str, Any]]:
            return corpus[(query, profile)]

        labels = derive_profile_labels(
            queries, candidate_profiles, retrieve, title_fields=_VIDEO_TITLE_FIELDS
        )
        assert labels == {}
        assert labels.records == ()
        assert labels.exclusions == (
            {
                "query": "man lifting",
                "reason": "no_profile_recovered_expected_videos",
                "expected_videos": ["v_-6dz6tBH77I"],
                "position": 0,
                "best_score": 0.0,
                "candidate_profiles": [
                    "video_colpali_smol500_mv_frame",
                    "video_colqwen_omni_mv_chunk_30s",
                ],
            },
        )
        assert labels.excluded_count == 1
        assert labels.excluded_queries == ("man lifting",)

    def test_profile_selection_untitled_result_is_excluded(self):
        from cogniverse_runtime.optimization_cli import derive_profile_labels

        queries = [
            _profile_selection_query_record(
                "man lifting",
                expected_videos=["v_-6dz6tBH77I"],
                ground_truth="one",
            )
        ]
        candidate_profiles = [
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
        ]
        corpus = {
            ("man lifting", "video_colpali_smol500_mv_frame"): [
                _profile_hit("v_-6dz6tBH77I.mp4"),
                _profile_hit(None, content_hash=_OTHER_CONTENT_HASH, segment=4),
            ],
            ("man lifting", "video_colqwen_omni_mv_chunk_30s"): [
                _profile_hit("v_-cAcA8dO7kA.mp4")
            ],
        }

        def retrieve(query: str, profile: str) -> list[dict[str, Any]]:
            return corpus[(query, profile)]

        labels = derive_profile_labels(
            queries, candidate_profiles, retrieve, title_fields=_VIDEO_TITLE_FIELDS
        )
        assert labels == {}
        assert labels.records == ()
        assert labels.exclusions == (
            {
                "query": "man lifting",
                "reason": "result_missing_title",
                "expected_videos": ["v_-6dz6tBH77I"],
                "position": 0,
                "untitled_results": [
                    {
                        "profile": "video_colpali_smol500_mv_frame",
                        "title_field": "video_title",
                        "document_ids": [f"{_OTHER_CONTENT_HASH}_seg_4"],
                    }
                ],
            },
        )
        assert labels.excluded_count == 1
        assert labels.excluded_queries == ("man lifting",)

    def test_profile_selection_title_tie_across_media_types_is_excluded(self):
        from cogniverse_runtime.optimization_cli import derive_profile_labels

        queries = [
            _profile_selection_query_record(
                "keynote audio",
                expected_videos=["talk_01"],
                ground_truth="one",
            )
        ]
        candidate_profiles = ["video_colpali_smol500_mv_frame", "audio_clap_semantic"]
        title_fields = {
            "video_colpali_smol500_mv_frame": "video_title",
            "audio_clap_semantic": "audio_title",
        }
        corpus = {
            ("keynote audio", "video_colpali_smol500_mv_frame"): [
                _profile_hit("talk_01.mp4")
            ],
            ("keynote audio", "audio_clap_semantic"): [
                _profile_hit("uploads/talk_01.wav", title_field="audio_title")
            ],
        }

        def retrieve(query: str, profile: str) -> list[dict[str, Any]]:
            return corpus[(query, profile)]

        labels = derive_profile_labels(
            queries, candidate_profiles, retrieve, title_fields=title_fields
        )
        assert labels == {}
        assert labels.records == ()
        assert labels.exclusions == (
            {
                "query": "keynote audio",
                "reason": "ambiguous_profile_tie",
                "expected_videos": ["talk_01"],
                "tied_profiles": [
                    "video_colpali_smol500_mv_frame",
                    "audio_clap_semantic",
                ],
                "best_score": 1.0,
                "position": 0,
            },
        )

    def test_profile_selection_title_field_is_read_per_profile(self):
        from cogniverse_runtime.optimization_cli import derive_profile_labels

        queries = [
            _profile_selection_query_record(
                "keynote audio",
                expected_videos=["talk_01"],
                ground_truth="one",
            )
        ]
        candidate_profiles = ["video_colpali_smol500_mv_frame", "audio_clap_semantic"]
        title_fields = {
            "video_colpali_smol500_mv_frame": "video_title",
            "audio_clap_semantic": "audio_title",
        }
        corpus = {
            ("keynote audio", "video_colpali_smol500_mv_frame"): [
                _profile_hit("v_-cAcA8dO7kA.mp4")
            ],
            ("keynote audio", "audio_clap_semantic"): [
                _profile_hit("uploads/talk_01.wav", title_field="audio_title")
            ],
        }

        def retrieve(query: str, profile: str) -> list[dict[str, Any]]:
            return corpus[(query, profile)]

        labels = derive_profile_labels(
            queries, candidate_profiles, retrieve, title_fields=title_fields
        )
        assert labels == {"keynote audio": "audio_clap_semantic"}
        assert labels.records == (
            {
                "query": "keynote audio",
                "ground_truth": "one",
                "query_type": "question",
                "source": "synthetic.json",
                "expected_videos": ["talk_01"],
                "available_profiles": [
                    "video_colpali_smol500_mv_frame",
                    "audio_clap_semantic",
                ],
                "selected_profile": "audio_clap_semantic",
                "confidence": 1.0,
                "reasoning": "audio_clap_semantic recovered talk_01",
                "example_id": "span:profile-label:0",
            },
        )
        assert labels.exclusions == ()

    def test_profile_selection_requires_title_field_for_every_candidate(self):
        from cogniverse_runtime.optimization_cli import derive_profile_labels

        queries = [
            _profile_selection_query_record(
                "man lifting",
                expected_videos=["v_-6dz6tBH77I"],
                ground_truth="one",
            )
        ]
        candidate_profiles = [
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
        ]

        def retrieve(query: str, profile: str) -> list[dict[str, Any]]:
            raise AssertionError("retrieval must not run without title fields")

        with pytest.raises(ValueError) as err:
            derive_profile_labels(
                queries,
                candidate_profiles,
                retrieve,
                title_fields={"video_colpali_smol500_mv_frame": "video_title"},
            )

        assert str(err.value) == (
            "Profile selection derivation has no title field for profile "
            "'video_colqwen_omni_mv_chunk_30s'"
        )

    def test_profile_selection_retrieval_rows_must_be_mappings(self):
        from cogniverse_runtime.optimization_cli import derive_profile_labels

        queries = [
            _profile_selection_query_record(
                "man lifting",
                expected_videos=["v_-6dz6tBH77I"],
                ground_truth="one",
            )
        ]
        candidate_profiles = [
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
        ]

        def retrieve(query: str, profile: str) -> list[str]:
            del query, profile
            return ["v_-6dz6tBH77I"]

        with pytest.raises(TypeError) as err:
            derive_profile_labels(
                queries,
                candidate_profiles,
                retrieve,
                title_fields=_VIDEO_TITLE_FIELDS,
            )

        assert str(err.value) == (
            "Profile selection retrieval row 0 for profile "
            "'video_colpali_smol500_mv_frame' must be a mapping, got str"
        )

    @staticmethod
    def _title_field_config_manager(profile_name: str, schema_name: str):
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_foundation.config.unified_config import BackendProfileConfig
        from tests.utils.memory_store import InMemoryConfigStore

        config_manager = ConfigManager(store=InMemoryConfigStore())
        config_manager.add_backend_profile(
            BackendProfileConfig.from_dict(
                profile_name, {"type": "video", "schema_name": schema_name}
            ),
            tenant_id="acme:docs",
        )
        return config_manager

    def test_profile_selection_title_fields_derive_from_shipped_schemas(self):
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_foundation.config.unified_config import BackendProfileConfig
        from cogniverse_runtime.optimization_cli import (
            SHIPPED_CONFIG_PATH,
            _profile_selection_title_fields,
        )
        from tests.utils.memory_store import InMemoryConfigStore

        shipped_profiles = json.loads(SHIPPED_CONFIG_PATH.read_text())["backend"][
            "profiles"
        ]
        config_manager = ConfigManager(store=InMemoryConfigStore())
        for profile_name, profile in shipped_profiles.items():
            config_manager.add_backend_profile(
                BackendProfileConfig.from_dict(profile_name, profile),
                tenant_id="acme:docs",
            )

        title_fields = _profile_selection_title_fields(
            config_manager,
            "acme:docs",
            list(shipped_profiles),
            FilesystemSchemaLoader(SHIPPED_CONFIG_PATH.parent / "schemas"),
        )

        assert title_fields == {
            "video_colpali_smol500_mv_frame": "video_title",
            "image_colpali_mv": "image_title",
            "audio_clap_semantic": "audio_title",
            "document_text_semantic": "document_title",
            "document_visual_colpali": "document_title",
            "lateon_mv": "title",
            "video_colqwen_omni_mv_chunk_30s": "video_title",
            "video_videoprism_base_mv_chunk_30s": "video_title",
            "video_videoprism_large_mv_chunk_30s": "video_title",
            "video_videoprism_lvt_base_sv_chunk_6s": "video_title",
            "video_videoprism_lvt_large_sv_chunk_6s": "video_title",
            "code_lateon_mv": "chunk_name",
            "wiki_semantic": "title",
        }

    def test_profile_selection_title_field_requires_document_mapping(self, tmp_path):
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_runtime.optimization_cli import _profile_selection_title_fields

        (tmp_path / "video_unmapped_schema.json").write_text(
            json.dumps({"name": "video_unmapped", "document": {"fields": []}})
        )
        config_manager = self._title_field_config_manager(
            "video_unmapped_profile", "video_unmapped"
        )

        with pytest.raises(ValueError) as err:
            _profile_selection_title_fields(
                config_manager,
                "acme:docs",
                ["video_unmapped_profile"],
                FilesystemSchemaLoader(tmp_path),
            )

        assert str(err.value) == (
            "Profile selection derivation: schema 'video_unmapped' for profile "
            "'video_unmapped_profile' declares no document_mapping"
        )

    def test_profile_selection_title_field_requires_mapped_title(self, tmp_path):
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_runtime.optimization_cli import _profile_selection_title_fields

        (tmp_path / "video_untitled_schema.json").write_text(
            json.dumps(
                {
                    "name": "video_untitled",
                    "document": {"fields": []},
                    "document_mapping": {"id": "video_id"},
                }
            )
        )
        config_manager = self._title_field_config_manager(
            "video_untitled_profile", "video_untitled"
        )

        with pytest.raises(ValueError) as err:
            _profile_selection_title_fields(
                config_manager,
                "acme:docs",
                ["video_untitled_profile"],
                FilesystemSchemaLoader(tmp_path),
            )

        assert str(err.value) == (
            "Profile selection derivation: schema 'video_untitled' for profile "
            "'video_untitled_profile' declares no document_mapping.title"
        )

    def test_profile_selection_title_field_unknown_profile_raises(self, tmp_path):
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_runtime.optimization_cli import _profile_selection_title_fields

        config_manager = self._title_field_config_manager(
            "video_colpali_smol500_mv_frame", "video_colpali_smol500_mv_frame"
        )

        with pytest.raises(ValueError) as err:
            _profile_selection_title_fields(
                config_manager,
                "acme:docs",
                ["ghost_profile"],
                FilesystemSchemaLoader(tmp_path),
            )

        assert str(err.value) == (
            "Profile selection derivation: profile 'ghost_profile' is not "
            "configured for tenant 'acme:docs'"
        )

    def test_profile_selection_title_field_missing_schema_raises(self, tmp_path):
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_runtime.optimization_cli import _profile_selection_title_fields
        from cogniverse_sdk.interfaces.schema_loader import SchemaNotFoundException

        config_manager = self._title_field_config_manager(
            "video_orphan_profile", "video_orphan"
        )

        with pytest.raises(SchemaNotFoundException) as err:
            _profile_selection_title_fields(
                config_manager,
                "acme:docs",
                ["video_orphan_profile"],
                FilesystemSchemaLoader(tmp_path),
            )

        assert str(err.value) == (
            f"Schema 'video_orphan' not found at {tmp_path / 'video_orphan_schema.json'}"
        )

    def test_shipped_label_reader_stays_removed(self):
        """Tenant ground-truth blobs are the sole label source. The shipped-file
        reader was deleted; re-adding any of its symbols re-opens the path where
        every tenant is graded against one bundled fixture."""
        import cogniverse_runtime.optimization_cli as cli

        assert not hasattr(cli, "PROFILE_SELECTION_LABEL_SOURCE_PATH")
        assert not hasattr(cli, "_load_profile_selection_label_source")
        assert not hasattr(cli, "_load_profile_selection_labels")

    @pytest.mark.asyncio
    async def test_profile_selection_labels_derive_from_tenant_ground_truth_rows(
        self,
    ):
        """The labels come from the tenant's uploaded blob, row for row; the
        shipped label source is never opened."""
        import collections
        import io
        from pathlib import Path

        shipped_data_root = Path(__file__).resolve().parents[3] / "data"

        profiles = [
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
        ]
        config_manager = _video_profile_config_manager(profiles)
        calls: list[tuple[str, str, str, int]] = []
        constructed: list[dict[str, Any]] = []
        search_service = _search_service_recovering(
            _TENANT_RECOVERABLE_QUERIES,
            recovering_profile=profiles[0],
            calls=calls,
            constructed=constructed,
        )
        rows = [
            _profile_span_row(
                "find clip",
                span_id="profile-0",
                available_profiles=profiles,
                selected_profile=profiles[0],
            )
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.profile_selection", rows)
        )

        opened: list[str] = []
        real_open = io.open

        def spy_open(file, *args, **kwargs):
            opened.append(str(file))
            return real_open(file, *args, **kwargs)

        with patch("io.open", spy_open), patch("builtins.open", spy_open):
            state, result = await self._run(
                provider,
                current_blob=self._served_state(),
                floor=(1, 1),
                config_manager=config_manager,
                ground_truth_blob=_tenant_ground_truth_blob(),
                search_service_factory=search_service,
            )

        assert sorted({query for query, _profile, _tenant, _top_k in calls}) == sorted(
            row["query"] for row in _TENANT_GROUND_TRUTH_ROWS
        )
        assert [path for path in opened if str(shipped_data_root) in path] == []
        assert collections.Counter(calls) == collections.Counter(
            (row["query"], profile, "test:unit", 10)
            for row in _TENANT_GROUND_TRUTH_ROWS
            for profile in profiles
        )
        assert result == {
            "status": "profile_labels_degenerate",
            "spans_found": 1,
            "served_examples": 3,
            "approved_examples": 0,
            "served_scoreable_examples": 3,
            "label_exclusions": {
                "count": 1,
                "queries": ["tenant clip nobody can recover"],
            },
            "labels_by_profile": {profiles[0]: 3},
            "exclusions_by_reason": {"no_profile_recovered_expected_videos": 1},
        }
        assert state["load_blob_calls"] == [
            ("config", "profile_selection_ground_truth")
        ]
        assert state["versioned_saves"] == []
        assert state["activate_calls"] == []
        assert state["score_calls"] == []
        assert "compiled_module" not in state
        assert "trainset" not in state
        assert constructed == [
            {
                "config_manager": config_manager,
                "schema_loader": "FilesystemSchemaLoader",
            }
        ]

    @pytest.mark.asyncio
    async def test_profile_selection_label_source_receives_the_loaded_rows(self):
        """One seam, both ends real: the list the ground-truth loader returns
        is the list derive_profile_labels is handed."""
        import cogniverse_agents.optimizer.profile_selection_ground_truth as gt
        import cogniverse_runtime.optimization_cli as optimization_cli
        from cogniverse_agents.optimizer.profile_selection_ground_truth import (
            canonicalize_profile_selection_ground_truth_rows,
        )

        profiles = [
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
        ]
        config_manager = _video_profile_config_manager(profiles)
        search_service = _search_service_recovering(
            _TENANT_RECOVERABLE_QUERIES,
            recovering_profile=profiles[0],
            calls=[],
            constructed=[],
        )
        rows = [
            _profile_span_row(
                "find clip",
                span_id="profile-0",
                available_profiles=profiles,
                selected_profile=profiles[0],
            )
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.profile_selection", rows)
        )

        seen: dict[str, Any] = {}
        real_loader = gt.load_profile_selection_ground_truth_rows
        real_derive = optimization_cli.derive_profile_labels

        async def spy_loader(artifact_manager):
            loaded = await real_loader(artifact_manager)
            seen["loader_rows"] = loaded
            return loaded

        def spy_derive(
            queries, candidate_profiles, retrieve, *, title_fields, profile_types=None
        ):
            seen["derive_queries"] = queries
            seen["derive_candidate_profiles"] = list(candidate_profiles)
            seen["derive_title_fields"] = dict(title_fields)
            seen["derive_profile_types"] = dict(profile_types or {})
            return real_derive(
                queries,
                candidate_profiles,
                retrieve,
                title_fields=title_fields,
                profile_types=profile_types,
            )

        with (
            patch.object(gt, "load_profile_selection_ground_truth_rows", spy_loader),
            patch.object(optimization_cli, "derive_profile_labels", spy_derive),
        ):
            _state, result = await self._run(
                provider,
                current_blob=self._served_state(),
                floor=(1, 1),
                config_manager=config_manager,
                ground_truth_blob=_tenant_ground_truth_blob(),
                search_service_factory=search_service,
            )

        assert seen["derive_queries"] is seen["loader_rows"]
        assert seen["loader_rows"] == canonicalize_profile_selection_ground_truth_rows(
            _TENANT_GROUND_TRUTH_ROWS
        )
        assert seen["derive_candidate_profiles"] == profiles
        assert seen["derive_profile_types"] == {
            profiles[0]: "video",
            profiles[1]: "video",
        }
        assert seen["derive_title_fields"] == {
            profiles[0]: "video_title",
            profiles[1]: "video_title",
        }
        assert result == {
            "status": "profile_labels_degenerate",
            "spans_found": 1,
            "served_examples": 3,
            "approved_examples": 0,
            "served_scoreable_examples": 3,
            "label_exclusions": {
                "count": 1,
                "queries": ["tenant clip nobody can recover"],
            },
            "labels_by_profile": {profiles[0]: 3},
            "exclusions_by_reason": {"no_profile_recovered_expected_videos": 1},
        }
        assert _state["load_blob_calls"] == [
            ("config", "profile_selection_ground_truth")
        ]
        assert _state["versioned_saves"] == []
        assert _state["activate_calls"] == []
        assert _state["score_calls"] == []
        assert "compiled_module" not in _state

    def test_profile_selection_derivation_does_not_import_evaluation_package(
        self, monkeypatch
    ):
        import builtins

        from cogniverse_runtime.optimization_cli import derive_profile_labels

        queries = [
            _profile_selection_query_record(
                "clean import query",
                expected_videos=["v1"],
                ground_truth="one",
            )
        ]
        candidate_profiles = [
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
        ]

        def retrieve(query: str, profile: str) -> list[dict[str, Any]]:
            if (
                query == "clean import query"
                and profile == "video_colpali_smol500_mv_frame"
            ):
                return [_profile_hit("v1.mp4")]
            return [_profile_hit("z1.mp4")]

        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("cogniverse_evaluation"):
                raise ModuleNotFoundError(name)
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", guarded_import)

        labels = derive_profile_labels(
            queries, candidate_profiles, retrieve, title_fields=_VIDEO_TITLE_FIELDS
        )
        assert labels == {
            "clean import query": "video_colpali_smol500_mv_frame",
        }
        assert labels.records == (
            {
                "query": "clean import query",
                "ground_truth": "one",
                "query_type": "question",
                "source": "synthetic.json",
                "expected_videos": ["v1"],
                "available_profiles": [
                    "video_colpali_smol500_mv_frame",
                    "video_colqwen_omni_mv_chunk_30s",
                ],
                "selected_profile": "video_colpali_smol500_mv_frame",
                "confidence": 1.0,
                "reasoning": "video_colpali_smol500_mv_frame recovered v1",
                "example_id": "span:profile-label:0",
            },
        )

    @pytest.mark.asyncio
    async def test_profile_selection_label_based_holdout_is_non_empty(self):
        from cogniverse_runtime import optimization_cli

        rows = [
            _profile_span_row(
                f"find clip {index}",
                span_id=f"profile-{index}",
                available_profiles=[
                    "video_colpali_smol500_mv_frame",
                    "video_colqwen_omni_mv_chunk_30s",
                ],
                selected_profile=(
                    "video_colpali_smol500_mv_frame"
                    if index % 2 == 0
                    else "video_colqwen_omni_mv_chunk_30s"
                ),
            )
            for index in range(4)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.profile_selection", rows)
        )
        served_state = self._served_state()
        captured: dict[str, Any] = {}
        real_split = optimization_cli._split_served_holdout

        def wrapped_split(
            records, min_holdout, scoreable_predicate=optimization_cli.is_scoreable
        ):
            train_records, holdout_records = real_split(
                records,
                min_holdout,
                scoreable_predicate=scoreable_predicate,
            )
            captured["train_ids"] = tuple(
                record["example_id"] for record in train_records
            )
            captured["holdout_ids"] = tuple(
                record["example_id"] for record in holdout_records
            )
            captured["train_size"] = len(train_records)
            captured["holdout_size"] = len(holdout_records)
            return train_records, holdout_records

        with patch(
            "cogniverse_runtime.optimization_cli._split_served_holdout",
            side_effect=wrapped_split,
        ):
            await self._run(
                provider,
                current_blob=served_state,
                floor=(1, 1),
                min_improvement=0.05,
                score=1.0,
                scoreable_fields=False,
            )

        assert captured == {
            "train_ids": (
                "span:profile-label:0",
                "span:profile-label:1",
                "span:profile-label:2",
            ),
            "holdout_ids": ("span:profile-label:3",),
            "train_size": 3,
            "holdout_size": 1,
        }

    @pytest.mark.asyncio
    async def test_profile_selection_holdout_scores_are_exact(self):
        from cogniverse_runtime import optimization_cli
        from cogniverse_runtime.optimization_cli import (
            _profile_selection_example,
            _profile_selection_scores,
            derive_profile_labels,
        )

        queries = [
            _profile_selection_query_record(
                "q1",
                expected_videos=["v1"],
                ground_truth="one",
            ),
            _profile_selection_query_record(
                "q2",
                expected_videos=["v2"],
                ground_truth="two",
            ),
            _profile_selection_query_record(
                "q3",
                expected_videos=["v3"],
                ground_truth="three",
            ),
            _profile_selection_query_record(
                "q4",
                expected_videos=["v4"],
                ground_truth="four",
            ),
        ]
        candidate_profiles = [
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
        ]
        corpus = {
            ("q1", "video_colpali_smol500_mv_frame"): [_profile_hit("v1.mp4")],
            ("q1", "video_colqwen_omni_mv_chunk_30s"): [_profile_hit("z1.mp4")],
            ("q2", "video_colpali_smol500_mv_frame"): [_profile_hit("z2.mp4")],
            ("q2", "video_colqwen_omni_mv_chunk_30s"): [_profile_hit("v2.mp4")],
            ("q3", "video_colpali_smol500_mv_frame"): [_profile_hit("v3.mp4")],
            ("q3", "video_colqwen_omni_mv_chunk_30s"): [_profile_hit("z3.mp4")],
            ("q4", "video_colpali_smol500_mv_frame"): [_profile_hit("z4.mp4")],
            ("q4", "video_colqwen_omni_mv_chunk_30s"): [_profile_hit("v4.mp4")],
        }

        def retrieve(query: str, profile: str) -> list[dict[str, Any]]:
            return corpus[(query, profile)]

        labels = derive_profile_labels(
            queries, candidate_profiles, retrieve, title_fields=_VIDEO_TITLE_FIELDS
        )
        rows = [
            _profile_span_row(
                query["query"],
                span_id=f"profile-{index}",
                available_profiles=candidate_profiles,
                selected_profile=labels[query["query"]],
            )
            for index, query in enumerate(queries)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.profile_selection", rows)
        )
        served_state = self._served_state()
        captured: dict[str, Any] = {}
        real_split = optimization_cli._split_served_holdout

        def wrapped_split(
            records, min_holdout, scoreable_predicate=optimization_cli.is_scoreable
        ):
            train_records, holdout_records = real_split(
                records,
                min_holdout,
                scoreable_predicate=scoreable_predicate,
            )
            captured["train_ids"] = tuple(
                record["example_id"] for record in train_records
            )
            captured["holdout_ids"] = tuple(
                record["example_id"] for record in holdout_records
            )
            captured["holdout_records"] = [dict(record) for record in holdout_records]
            captured["train_size"] = len(train_records)
            captured["holdout_size"] = len(holdout_records)
            return train_records, holdout_records

        with patch(
            "cogniverse_runtime.optimization_cli._split_served_holdout",
            side_effect=wrapped_split,
        ):
            await self._run(
                provider,
                current_blob=served_state,
                floor=(1, 1),
                min_improvement=0.05,
                score=1.0,
                scoreable_fields=False,
            )

        assert captured == {
            "train_ids": (
                "span:profile-label:0",
                "span:profile-label:1",
                "span:profile-label:2",
            ),
            "holdout_ids": ("span:profile-label:3",),
            "train_size": 3,
            "holdout_size": 1,
            "holdout_records": [
                {
                    "query": "q4",
                    "available_profiles": (
                        "video_colpali_smol500_mv_frame, "
                        "video_colqwen_omni_mv_chunk_30s"
                    ),
                    "selected_profile": "video_colqwen_omni_mv_chunk_30s",
                    "confidence": 0.9,
                    "reasoning": "Selected video_colqwen_omni_mv_chunk_30s",
                    "query_intent": "video_search",
                    "modality": "video",
                    "complexity": "simple",
                    "example_id": "span:profile-label:3",
                }
            ],
        }

        holdout_examples = [
            _profile_selection_example(record) for record in captured["holdout_records"]
        ]
        correct_selector = _ProfileSelectionByQuery(
            labels, default="video_colpali_smol500_mv_frame"
        )

        def broken_selector(query: str, available_profiles: str):
            del query, available_profiles
            return _sel("video_colpali_smol500_mv_frame")

        scores = {
            "correct": _profile_selection_scores(correct_selector, holdout_examples),
            "broken": _profile_selection_scores(broken_selector, holdout_examples),
        }
        assert scores == {"correct": 1.0, "broken": 0.0}

    def test_query_enhancement_holdout_size_is_unchanged(self):
        from cogniverse_runtime.optimization_cli import _split_served_holdout

        records = [
            {
                "query": f"qe-{index}",
                "source_text": f"source-{index}",
                "grounding_context": f"context-{index}",
                "example_id": f"span:query-enhancement:{index}",
            }
            for index in range(4)
        ]

        train_records, holdout_records = _split_served_holdout(records, 1)

        assert {
            "train_ids": tuple(record["example_id"] for record in train_records),
            "holdout_ids": tuple(record["example_id"] for record in holdout_records),
            "train_size": len(train_records),
            "holdout_size": len(holdout_records),
        } == {
            "train_ids": (
                "span:query-enhancement:0",
                "span:query-enhancement:1",
                "span:query-enhancement:2",
            ),
            "holdout_ids": ("span:query-enhancement:3",),
            "train_size": 3,
            "holdout_size": 1,
        }

    @pytest.mark.asyncio
    async def test_profile_gate_persists_but_never_activates_a_losing_candidate(self):
        rows = [
            _profile_span_row(
                f"find clip {i}",
                span_id=f"profile-{i}",
                available_profiles=[
                    "video_colpali_smol500_mv_frame",
                    "video_colqwen_omni_mv_chunk_30s",
                ],
                selected_profile=(
                    "video_colpali_smol500_mv_frame"
                    if i % 2 == 0
                    else "video_colqwen_omni_mv_chunk_30s"
                ),
            )
            for i in range(4)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.profile_selection", rows)
        )
        served_state = self._served_state()

        state, result = await self._run(
            provider,
            current_blob=served_state,
            floor=(1, 1),
            min_improvement=0.05,
            score=1.0,
        )

        assert result == {
            "status": "success",
            "spans_found": 4,
            "served_examples": 4,
            "approved_examples": 0,
            "served_scoreable_examples": 4,
            "training_examples": 3,
            "holdout_examples": 1,
            "holdout_source": "derived_labels",
            "label_exclusions": {"count": 0, "queries": []},
            "labels_by_profile": {
                "video_colpali_smol500_mv_frame": 2,
                "video_colqwen_omni_mv_chunk_30s": 2,
            },
            "exclusions_by_reason": {},
            **_selection_block(3, 3),
            "baseline_score": 1.0,
            "current_score": 1.0,
            "candidate_score": 1.0,
            "decision": "keep",
            "version": 1,
            "consumed_example_ids": [
                "span:profile-label:0",
                "span:profile-label:1",
                "span:profile-label:2",
                "span:profile-label:3",
            ],
        }
        assert state["versioned_saves"] == [
            {
                "kind": "model",
                "key": "profile_selection",
                "content": '{"compiled": "profile"}',
                "consumed_example_ids": [
                    "span:profile-label:0",
                    "span:profile-label:1",
                    "span:profile-label:2",
                    "span:profile-label:3",
                ],
                "decision": "keep",
                "scored": True,
                "score": 1.0,
                "base_score": 1.0,
                "candidate_score": 1.0,
                "extra_ledger_fields": {
                    "labels_by_profile": {
                        "video_colpali_smol500_mv_frame": 2,
                        "video_colqwen_omni_mv_chunk_30s": 2,
                    },
                    "exclusions_by_reason": {},
                },
            }
        ]
        assert state["activate_calls"] == []
        assert state["active_blob"] == served_state

    @pytest.mark.asyncio
    async def test_profile_rollback_persists_and_activates_base_state(self):
        from cogniverse_agents.profile_selection_agent import ProfileSelectionModule

        rows = [
            _profile_span_row(
                f"find clip {i}",
                span_id=f"profile-{i}",
                available_profiles=[
                    "video_colpali_smol500_mv_frame",
                    "video_colqwen_omni_mv_chunk_30s",
                ],
                selected_profile=(
                    "video_colpali_smol500_mv_frame"
                    if i % 2 == 0
                    else "video_colqwen_omni_mv_chunk_30s"
                ),
            )
            for i in range(4)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.profile_selection", rows)
        )
        current_blob = self._served_state()
        base_state = json.dumps(ProfileSelectionModule().dump_state(), default=str)

        state, result = await self._run(
            provider,
            current_blob=current_blob,
            floor=(1, 1),
            min_improvement=0.05,
            score_by_module=self._profile_selection_score_by_module,
        )

        assert result == {
            "status": "success",
            "spans_found": 4,
            "served_examples": 4,
            "approved_examples": 0,
            "served_scoreable_examples": 4,
            "training_examples": 3,
            "holdout_examples": 1,
            "holdout_source": "derived_labels",
            "label_exclusions": {"count": 0, "queries": []},
            "labels_by_profile": {
                "video_colpali_smol500_mv_frame": 2,
                "video_colqwen_omni_mv_chunk_30s": 2,
            },
            "exclusions_by_reason": {},
            **_selection_block(3, 3),
            "baseline_score": 1.0,
            "current_score": 0.0,
            "candidate_score": 0.0,
            "decision": "rollback",
            "version": 1,
            "consumed_example_ids": [
                "span:profile-label:0",
                "span:profile-label:1",
                "span:profile-label:2",
                "span:profile-label:3",
            ],
        }
        assert state["versioned_saves"] == [
            {
                "kind": "model",
                "key": "profile_selection",
                "content": base_state,
                "consumed_example_ids": [
                    "span:profile-label:0",
                    "span:profile-label:1",
                    "span:profile-label:2",
                    "span:profile-label:3",
                ],
                "decision": "rollback",
                "scored": True,
                "score": 0.0,
                "base_score": 1.0,
                "candidate_score": 0.0,
                "extra_ledger_fields": {
                    "labels_by_profile": {
                        "video_colpali_smol500_mv_frame": 2,
                        "video_colqwen_omni_mv_chunk_30s": 2,
                    },
                    "exclusions_by_reason": {},
                },
            }
        ]
        assert state["activate_calls"] == [("model", "profile_selection", 1)]
        assert state["active_blob"] == base_state

    @pytest.mark.asyncio
    async def test_score_failure_preserves_selection_block(self):
        rows = [
            _profile_span_row(
                f"find clip {i}",
                span_id=f"profile-{i}",
                available_profiles=[
                    "video_colpali_smol500_mv_frame",
                    "video_colqwen_omni_mv_chunk_30s",
                ],
                selected_profile=(
                    "video_colpali_smol500_mv_frame"
                    if i % 2 == 0
                    else "video_colqwen_omni_mv_chunk_30s"
                ),
            )
            for i in range(4)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.profile_selection", rows)
        )
        served_state = self._served_state()

        def failing_score(module, holdout):
            raise RuntimeError("profile scorer failed")

        state, result = await self._run(
            provider,
            current_blob=served_state,
            floor=(1, 1),
            score_by_module=failing_score,
        )

        assert state["versioned_saves"] == []
        assert state["activate_calls"] == []
        assert state["active_blob"] == served_state
        assert result == {
            "status": "failed",
            "error": "profile scorer failed",
            "label_exclusions": {"count": 0, "queries": []},
            "labels_by_profile": {
                "video_colpali_smol500_mv_frame": 2,
                "video_colqwen_omni_mv_chunk_30s": 2,
            },
            "exclusions_by_reason": {},
            **_selection_block(3, 3),
        }

    @pytest.mark.asyncio
    async def test_training_selection_store_override_binds_profile_key(self):
        rows = [
            _profile_span_row(
                f"find clip {i}",
                span_id=f"profile-{i}",
                available_profiles=[
                    "video_colpali_smol500_mv_frame",
                    "video_colqwen_omni_mv_chunk_30s",
                ],
                selected_profile=(
                    "video_colpali_smol500_mv_frame"
                    if i % 2 == 0
                    else "video_colqwen_omni_mv_chunk_30s"
                ),
            )
            for i in range(4)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.profile_selection", rows)
        )
        served_state = self._served_state()
        config_manager = _training_selection_config_manager(
            "test:unit",
            {"profile_selection": {"trainset_cap": 42}},
        )

        _, result = await self._run(
            provider,
            current_blob=served_state,
            floor=(1, 1),
            config_manager=config_manager,
        )

        assert result["selection"]["cap"] == 42, result

    @pytest.mark.asyncio
    async def test_profile_ground_truth_missing_stops_before_scoring_or_compile(self):
        rows = [
            _profile_span_row(
                f"find clip {i}",
                span_id=f"profile-{i}",
                available_profiles=[
                    "video_colpali_smol500_mv_frame",
                    "video_colqwen_omni_mv_chunk_30s",
                ],
                selected_profile="video_colpali_smol500_mv_frame",
            )
            for i in range(4)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.profile_selection", rows)
        )
        served_state = self._served_state()

        state, result = await self._run(
            provider,
            current_blob=served_state,
            floor=(1, 1),
            ground_truth_present=False,
        )

        assert result == {
            "status": "profile_selection_ground_truth_missing",
            "retryable": False,
            "error": "profile_selection_ground_truth is not configured for tenant test:unit",
        }
        assert state["load_blob_calls"] == [
            ("config", "profile_selection_ground_truth")
        ]
        assert state["query_spans_calls"] == []
        assert state["score_calls"] == []
        assert state["versioned_saves"] == []
        assert state["activate_calls"] == []
        assert "compiled_module" not in state

    @pytest.mark.asyncio
    async def test_profile_ground_truth_store_unavailable_sets_retryable_true(self):
        rows = [
            _profile_span_row(
                f"find clip {i}",
                span_id=f"profile-{i}",
                available_profiles=[
                    "video_colpali_smol500_mv_frame",
                    "video_colqwen_omni_mv_chunk_30s",
                ],
                selected_profile="video_colpali_smol500_mv_frame",
            )
            for i in range(4)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.profile_selection", rows)
        )
        served_state = self._served_state()

        state, result = await self._run(
            provider,
            current_blob=served_state,
            floor=(1, 1),
            ground_truth_error=ConnectionError("blob store down"),
        )

        assert result == {
            "status": "profile_selection_ground_truth_store_unavailable",
            "retryable": True,
            "error": "profile_selection_ground_truth store unavailable",
            "cause": {
                "type": "ConnectionError",
                "message": "blob store down",
            },
        }
        assert state["load_blob_calls"] == [
            ("config", "profile_selection_ground_truth")
        ]
        assert state["query_spans_calls"] == []
        assert state["score_calls"] == []
        assert state["versioned_saves"] == []
        assert state["activate_calls"] == []
        assert "compiled_module" not in state

    @pytest.mark.asyncio
    async def test_profile_below_population_floor_persists_without_activating(self):
        profiles = [
            "video_colpali_smol500_mv_frame",
            "video_colqwen_omni_mv_chunk_30s",
        ]
        rows = [
            _profile_span_row(
                f"find clip {i}",
                span_id=f"profile-{i}",
                available_profiles=profiles,
                selected_profile=profiles[i % len(profiles)],
            )
            for i in range(4)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.profile_selection", rows)
        )
        served_state = self._served_state()

        state, result = await self._run(
            provider,
            current_blob=served_state,
            floor=(100, 1),
        )

        assert result == {
            "status": "insufficient_population",
            "spans_found": 4,
            "examples": 4,
            "distinct_queries": 4,
            "min_samples": 100,
            "min_unique_queries": 1,
            "version": 1,
            "label_exclusions": {"count": 0, "queries": []},
            "labels_by_profile": {
                "video_colpali_smol500_mv_frame": 2,
                "video_colqwen_omni_mv_chunk_30s": 2,
            },
            "exclusions_by_reason": {},
        }
        assert "selection" not in result
        assert state["versioned_saves"] == [
            {
                "kind": "model",
                "key": "profile_selection",
                "content": served_state,
                "consumed_example_ids": [
                    "span:profile-label:0",
                    "span:profile-label:1",
                    "span:profile-label:2",
                    "span:profile-label:3",
                ],
                "decision": "insufficient_population",
                "scored": False,
                "score": None,
                "base_score": None,
                "candidate_score": None,
                "extra_ledger_fields": {
                    "labels_by_profile": {
                        "video_colpali_smol500_mv_frame": 2,
                        "video_colqwen_omni_mv_chunk_30s": 2,
                    },
                    "exclusions_by_reason": {},
                },
            }
        ]
        assert state["activate_calls"] == []
        assert state["active_blob"] == served_state


# ---------------------------------------------------------------------------
# Test: gateway threshold analysis with mock span data
# ---------------------------------------------------------------------------


class TestGatewayThresholdAnalysis:
    """Verify threshold tuning logic with synthetic gateway spans."""

    @pytest.mark.asyncio
    async def test_high_simple_error_rate_raises_threshold(self):
        """When simple-routed queries fail often, threshold should increase."""
        rows = [
            # 5 simple queries, 3 with ERROR status
            _gateway_row("simple", 0.8, "ERROR" if i < 3 else "OK")
            for i in range(5)
        ]
        # 2 complex queries, both OK
        rows += [_gateway_row("complex", 0.4, "OK") for _ in range(2)]

        spans_df = _make_spans_df("cogniverse.gateway", rows)
        provider = FakeTelemetryProvider(spans_df)
        mgr = FakeTelemetryManager(provider)

        from cogniverse_runtime.optimization_cli import (
            run_gateway_thresholds_optimization,
        )

        p1, p2 = _patch_infra(mgr)
        with p1, p2:
            result = await run_gateway_thresholds_optimization(
                tenant_id="test:unit", lookback_hours=1
            )

        assert result["status"] == "success"
        thresholds = result["thresholds"]
        # Threshold should have been raised from default 0.4
        assert thresholds["fast_path_confidence_threshold"] > 0.4
        assert "artifact_id" in result

    @pytest.mark.asyncio
    async def test_all_ok_keeps_threshold_stable(self):
        """When error rates are low, threshold stays near default."""
        rows = [_gateway_row("simple", 0.75, "OK") for _ in range(10)]

        spans_df = _make_spans_df("cogniverse.gateway", rows)
        provider = FakeTelemetryProvider(spans_df)
        mgr = FakeTelemetryManager(provider)

        from cogniverse_runtime.optimization_cli import (
            run_gateway_thresholds_optimization,
        )

        p1, p2 = _patch_infra(mgr)
        with p1, p2:
            result = await run_gateway_thresholds_optimization(
                tenant_id="test:unit", lookback_hours=1
            )

        assert result["status"] == "success"
        # Threshold should stay at default (0.4) since no high error rates
        threshold = result["thresholds"]["fast_path_confidence_threshold"]
        assert 0.3 <= threshold <= 0.5

    @pytest.mark.asyncio
    async def test_non_numeric_confidence_dropped_not_fatal(self):
        """One span with a string confidence must not abort the recompute;
        thresholds come from the numeric rows only."""
        rows = [_gateway_row("simple", c, "OK") for c in (0.5, 0.6, 0.7, 0.8)]
        rows.append(_gateway_row("simple", "high", "OK"))

        spans_df = _make_spans_df("cogniverse.gateway", rows)
        provider = FakeTelemetryProvider(spans_df)
        mgr = FakeTelemetryManager(provider)

        from cogniverse_runtime.optimization_cli import (
            run_gateway_thresholds_optimization,
        )

        p1, p2 = _patch_infra(mgr)
        with p1, p2:
            result = await run_gateway_thresholds_optimization(
                tenant_id="test:unit", lookback_hours=1
            )

        assert result["status"] == "success"
        thresholds = result["thresholds"]
        # Numeric rows [0.5, 0.6, 0.7, 0.8]: mean 0.65 keeps the default
        # fast path; p25 0.575 -> gliner 0.575 * 0.8 = 0.46.
        assert thresholds["fast_path_confidence_threshold"] == 0.4
        assert thresholds["gliner_threshold"] == 0.46
        analysis = thresholds["analysis"]
        assert analysis["total_spans"] == 5
        assert analysis["mean_confidence"] == 0.65
        assert analysis["p25_confidence"] == 0.575


# ---------------------------------------------------------------------------
# Test: workflow optimization with mock orchestration spans
# ---------------------------------------------------------------------------


class TestWorkflowOptimization:
    """Verify workflow optimization extracts executions and saves artifacts."""

    @pytest.mark.asyncio
    async def test_workflow_with_orchestration_spans(self):
        """Workflow mode processes orchestration spans through the evaluator.

        OrchestrationEvaluator._extract_workflow_execution reads the workflow
        off the canonical input.value (query) and output.value (the decision).
        """
        rows = [
            {
                "name": "cogniverse.orchestration",
                "context.span_id": f"span-{i}",
                "start_time": datetime.now(timezone.utc) - timedelta(seconds=3 - i),
                "attributes.input.value": f"test query {i}",
                "attributes.output.value": json.dumps(
                    {
                        "workflow_id": f"wf-{i}",
                        "pattern": "sequential",
                        "agent_sequence": ["search_agent", "summarizer_agent"],
                        "execution_order": ["search_agent", "summarizer_agent"],
                        "execution_time": 2.5,
                        "success": True,
                        "tasks_completed": 2,
                        "confidence": 0.8,
                        "agent_observations": [
                            {
                                "agent_name": "search_agent",
                                "execution_time": 1.0,
                                "success": True,
                                "confidence": 0.9,
                            },
                            {
                                "agent_name": "summarizer_agent",
                                "execution_time": 1.5,
                                "success": True,
                                "confidence": 0.7,
                            },
                        ],
                    }
                ),
                "status_code": "OK",
                "status_message": None,
            }
            for i in range(3)
        ]
        spans_df = pd.DataFrame(rows)
        provider = FakeTelemetryProvider(spans_df)
        mgr = FakeTelemetryManager(provider)

        from cogniverse_runtime.optimization_cli import run_workflow_optimization

        workflow_store = FakeWorkflowStore()
        p1, p2 = _patch_infra(mgr)
        workflow_store_patch = patch(
            "cogniverse_core.registries.WorkflowStoreRegistry.get",
            return_value=workflow_store,
        )
        with p1, p2, workflow_store_patch:
            result = await run_workflow_optimization(
                tenant_id="test:unit", lookback_hours=1
            )

        assert result["status"] == "success"
        assert result["spans_found"] == 3
        assert result["workflows_extracted"] == 3

    @pytest.mark.asyncio
    async def test_drains_more_than_one_batch_and_persists_serving_artifacts(self):
        query = "find exact aurora video"
        evaluation_base = datetime.now(timezone.utc) - timedelta(minutes=2)
        rows = [
            {
                "name": "cogniverse.orchestration",
                "context.span_id": f"span-page-{index:02d}",
                "start_time": evaluation_base + timedelta(milliseconds=index),
                "attributes.input.value": query,
                "attributes.output.value": json.dumps(
                    {
                        "workflow_id": f"wf-page-{index:02d}",
                        "pattern": "sequential",
                        "agent_sequence": ["search_agent", "summarizer_agent"],
                        "execution_order": ["search_agent", "summarizer_agent"],
                        "execution_time": 2.5,
                        "success": True,
                        "tasks_completed": 2,
                        "confidence": 0.8,
                        "agent_observations": [
                            {
                                "agent_name": "search_agent",
                                "execution_time": 1.0,
                                "success": True,
                                "confidence": 0.9,
                            },
                            {
                                "agent_name": "summarizer_agent",
                                "execution_time": 1.5,
                                "success": True,
                                "confidence": 0.7,
                            },
                        ],
                    }
                ),
                "status_code": "OK",
                "status_message": None,
            }
            for index in range(55)
        ]
        provider = FakeTelemetryProvider(pd.DataFrame(rows))
        manager = FakeTelemetryManager(provider)

        from cogniverse_agents.workflow.intelligence import WorkflowIntelligence
        from cogniverse_runtime.optimization_cli import run_workflow_optimization

        workflow_store = FakeWorkflowStore()
        config_patch, telemetry_patch = _patch_infra(manager)
        workflow_store_patch = patch(
            "cogniverse_core.registries.WorkflowStoreRegistry.get",
            return_value=workflow_store,
        )
        with config_patch, telemetry_patch, workflow_store_patch:
            result = await run_workflow_optimization(
                tenant_id="test:workflow-pagination",
                lookback_hours=1,
            )
            fresh_intelligence = WorkflowIntelligence(
                tenant_id="test:workflow-pagination"
            )
            await fresh_intelligence.load_historical_data()

        assert result == {
            "status": "success",
            "spans_found": 55,
            "workflows_extracted": 55,
            "execution_demos_saved": 55,
            "agent_profiles_saved": 2,
            "workflow_templates_saved": 1,
        }
        assert len(provider.traces.calls) == 2
        assert {call["end_time"] for call in provider.traces.calls} == {
            provider.traces.calls[0]["end_time"]
        }
        assert len(fresh_intelligence.workflow_history) == 55
        assert set(fresh_intelligence.agent_performance) == {
            "search_agent",
            "summarizer_agent",
        }
        search_profile = fresh_intelligence.agent_performance["search_agent"]
        assert (
            search_profile.total_executions,
            search_profile.successful_executions,
            search_profile.average_execution_time,
            search_profile.average_confidence,
            search_profile.error_rate,
            search_profile.preferred_query_types,
        ) == (55, 55, 1.0, 0.9, 0.0, ["sequential_query"])

        summarizer_profile = fresh_intelligence.agent_performance["summarizer_agent"]
        assert (
            summarizer_profile.total_executions,
            summarizer_profile.successful_executions,
            summarizer_profile.average_execution_time,
            summarizer_profile.average_confidence,
            summarizer_profile.error_rate,
            summarizer_profile.preferred_query_types,
        ) == (55, 55, 1.5, 0.7, 0.0, ["sequential_query"])

        template = fresh_intelligence._find_matching_template(query)
        assert template is not None
        assert template.query_patterns == [query]
        assert template.task_sequence == [
            {"agent": "search_agent", "task": "process", "dependencies": []},
            {
                "agent": "summarizer_agent",
                "task": "process",
                "dependencies": ["template_task_0"],
            },
        ]
        assert template.expected_execution_time == 2.5
        assert template.success_rate == 1.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure_point", ["template", "corpus"])
    async def test_learning_state_helper_forwards_exact_state_and_store_failure(
        self, failure_point
    ):
        from cogniverse_runtime.optimization_cli import (
            _save_workflow_learning_state,
        )
        from cogniverse_sdk.interfaces.workflow_store import WorkflowTemplate

        def template(template_id, agent):
            return WorkflowTemplate(
                template_id=template_id,
                name=template_id,
                description=f"template for {agent}",
                query_patterns=[f"query for {agent}"],
                task_sequence=[{"agent": agent, "task": "process", "dependencies": []}],
                expected_execution_time=1.0,
                success_rate=1.0,
            )

        previous = template("previous", "search_agent")
        replacements = [
            template("replacement-a", "search_agent"),
            template("replacement-b", "summarizer_agent"),
        ]
        failure = ConnectionError(f"{failure_point} store unavailable")

        class Store:
            def __init__(self):
                self.templates = {previous.template_id: previous}
                self.calls = []

            async def replace_learning_state(
                self, tenant_id, executions, profiles, patterns, templates
            ):
                self.calls.append(
                    (tenant_id, executions, profiles, patterns, templates)
                )
                raise failure

        store = Store()

        with pytest.raises(ConnectionError) as exc_info:
            await _save_workflow_learning_state(
                store,
                tenant_id="acme:prod",
                executions=[],
                profiles=[],
                patterns={},
                templates=replacements,
            )

        assert exc_info.value is failure
        assert store.templates == {"previous": previous}
        assert store.calls == [("acme:prod", [], [], {}, replacements)]

    @pytest.mark.asyncio
    async def test_learning_state_helper_awaits_store_owned_serialization(self):
        from cogniverse_runtime.optimization_cli import (
            _save_workflow_learning_state,
        )

        class Store:
            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.calls = []
                self.lock = asyncio.Lock()

            async def replace_learning_state(
                self, tenant_id, executions, profiles, patterns, templates
            ):
                self.calls.append(
                    (tenant_id, executions, profiles, patterns, templates)
                )
                async with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                    await asyncio.sleep(0.02)
                    self.active -= 1

        store = Store()

        await asyncio.gather(
            *(
                _save_workflow_learning_state(
                    store,
                    tenant_id="acme:prod",
                    executions=[],
                    profiles=[],
                    patterns={},
                    templates=[],
                )
                for _ in range(2)
            )
        )

        assert store.max_active == 1
        assert store.active == 0
        assert store.calls == [
            ("acme:prod", [], [], {}, []),
            ("acme:prod", [], [], {}, []),
        ]


class TestEntityExtractionOptimization:
    """Entity extraction optimization handles missing/empty span data."""

    @staticmethod
    def _base_state() -> str:
        from cogniverse_agents.entity_extraction_agent import EntityExtractionModule

        return json.dumps(
            EntityExtractionModule().dump_state(),
            default=str,
        )

    @staticmethod
    def _served_state() -> str:
        from cogniverse_agents.entity_extraction_agent import EntityExtractionModule

        state = json.loads(
            json.dumps(EntityExtractionModule().dump_state(), default=str)
        )
        state["extractor.predict"]["demos"] = [
            {
                "query": "find PyTorch tutorials",
                "entities": "PyTorch|TECHNOLOGY|1.0",
            }
        ]
        return json.dumps(state, default=str)

    @staticmethod
    def _entity_score_by_module(module, holdout) -> float:
        state = json.loads(json.dumps(module.dump_state(), default=str))
        if state.get("extractor.predict", {}).get("demos", []):
            return 0.0
        return 1.0

    async def _run(
        self,
        provider,
        *,
        current_blob: str | None,
        floor: tuple[int, int],
        min_improvement: float = 0.05,
        score: float = 1.0,
        score_by_module=None,
        config_manager=None,
        truth_rows: list[dict[str, Any]] | None = None,
        approved_data: list[dict[str, Any]] | None = None,
        ground_truth_missing: bool = False,
        ground_truth_error: Exception | None = None,
    ):
        from cogniverse_runtime.optimization_cli import (
            run_entity_extraction_optimization,
        )

        state = {
            "active_blob": current_blob,
            "versioned_saves": [],
            "activate_calls": [],
            "load_blob_calls": [],
            "trainset_queries": [],
        }

        spans_df = provider._trace_store._spans_df.copy(deep=True)
        if truth_rows is None:
            truth_rows = []
            for _, row in spans_df.iterrows():
                query = str(row.get("attributes.input.value") or "").strip()
                output = row.get("attributes.output.value", "{}")
                if isinstance(output, str):
                    try:
                        output = json.loads(output)
                    except Exception:
                        output = {}
                if not isinstance(output, dict):
                    output = {}
                entities = output.get("entities", [])
                if query and entities:
                    truth_rows.append({"query": query, "entities": entities})

        class FakeArtifactManager:
            def __init__(self, received_provider, tenant_id):
                assert received_provider is provider
                assert tenant_id == "test:unit"
                self._tenant_id = tenant_id

            async def load_blob(self, kind, key):
                state["load_blob_calls"].append((kind, key))
                if (kind, key) == (
                    "config",
                    "entity_extraction_ground_truth",
                ):
                    if ground_truth_error is not None:
                        raise ground_truth_error
                    if ground_truth_missing:
                        return None
                    return json.dumps(truth_rows, separators=(",", ":"))
                assert (kind, key) == ("model", "entity_extraction")
                return state["active_blob"]

            async def save_blob_versioned(
                self,
                kind,
                key,
                content,
                *,
                consumed_example_ids,
                decision,
                scored,
                score,
                base_score,
                candidate_score,
                extra_ledger_fields=None,
            ):
                state["versioned_saves"].append(
                    {
                        "kind": kind,
                        "key": key,
                        "content": content,
                        "consumed_example_ids": list(consumed_example_ids),
                        "decision": decision,
                        "scored": scored,
                        "score": score,
                        "base_score": base_score,
                        "candidate_score": candidate_score,
                        "extra_ledger_fields": dict(extra_ledger_fields or {}),
                    }
                )
                return f"artifact-{len(state['versioned_saves'])}", len(
                    state["versioned_saves"]
                )

            async def activate_version(self, kind, key, version):
                state["activate_calls"].append((kind, key, version))
                state["active_blob"] = state["versioned_saves"][version - 1]["content"]
                return {"active": {"version": version, "activated_at": "now"}}

            async def get_version_lineage(self, kind, key):
                assert (kind, key) == ("model", "entity_extraction")
                return []

        class FakePredictor:
            demos: list = []

        class FakeTeleprompter:
            max_bootstrapped_demos = 4
            max_labeled_demos = 8
            max_rounds = 1
            metric_threshold = 1.0
            error_count = 0

            def compile(self, module, trainset):
                state["compiled_module"] = type(module).__name__
                state["trainset"] = [
                    example.toDict() if hasattr(example, "toDict") else example
                    for example in trainset
                ]
                state["trainset_queries"] = [
                    example.query if hasattr(example, "query") else example["query"]
                    for example in trainset
                ]

                class Compiled:
                    def dump_state(self):
                        return {"compiled": "entity_extraction"}

                    def named_predictors(self):
                        return [("extractor.predict", FakePredictor())]

                return Compiled()

        class FakeLLMConfig:
            def resolve(self, purpose):
                return "student-endpoint"

            def resolve_teacher(self):
                return _teacher_endpoint()

        class FakeConfig:
            def get_llm_config(self):
                return FakeLLMConfig()

        mgr = FakeTelemetryManager(provider)
        approved_rows = approved_data or []
        effective_config_manager = config_manager or _training_selection_config_manager(
            "test:unit",
            {"entity_extraction": {}},
        )
        p1, p2 = _patch_infra(mgr, config_manager=effective_config_manager)
        with (
            p1,
            p2,
            patch(
                "cogniverse_foundation.config.utils.get_config",
                return_value=FakeConfig(),
            ),
            patch(
                "cogniverse_foundation.config.llm_factory.create_dspy_lm",
                return_value=object(),
            ),
            patch(
                "cogniverse_runtime.optimization_cli._load_approved_synthetic_data",
                side_effect=lambda received_provider, tenant_id, optimizer_type: (
                    approved_rows
                ),
            ),
            patch(
                "cogniverse_runtime.optimization_cli._create_teleprompter",
                return_value=FakeTeleprompter(),
            ),
            patch(
                "cogniverse_runtime.optimization_cli._entity_extraction_scores",
                side_effect=score_by_module or (lambda module, holdout: score),
            ),
            patch(
                "cogniverse_runtime.optimization_cli._min_improvement_from_config",
                return_value=min_improvement,
            ),
            patch(
                "cogniverse_runtime.optimization_cli._population_floor_from_config",
                return_value=floor,
            ),
            patch(
                "cogniverse_agents.optimizer.artifact_manager.ArtifactManager",
                FakeArtifactManager,
            ),
            patch("dspy.configure", lambda **kwargs: None),
        ):
            result = await run_entity_extraction_optimization(
                tenant_id="test:unit",
                lookback_hours=1,
            )
        return state, result

    @pytest.mark.asyncio
    async def test_entity_extraction_missing_ground_truth_returns_named_status(
        self, fake_telemetry_manager
    ):
        spans = _make_spans_df(
            "cogniverse.entity_extraction",
            [
                {
                    "attributes.input.value": "served query one",
                    "attributes.output.value": json.dumps(
                        {"entities": [{"text": "served", "type": "CONCEPT"}]}
                    ),
                }
            ],
        )
        provider = FakeTelemetryProvider(spans)

        state, result = await self._run(
            provider,
            current_blob=None,
            floor=(1, 1),
            ground_truth_missing=True,
        )

        assert result == {
            "status": "entity_extraction_ground_truth_missing",
            "retryable": False,
            "error": "entity_extraction_ground_truth is not configured for tenant test:unit",
        }
        assert state["versioned_saves"] == []

    @pytest.mark.asyncio
    async def test_entity_extraction_uses_truth_and_approved_rows_not_served_spans(
        self, monkeypatch, tmp_path
    ):
        from dspy.utils.dummies import DummyLM

        import cogniverse_runtime.optimization_cli as optimization_cli
        from cogniverse_runtime.optimization_cli import (
            run_entity_extraction_optimization,
        )

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

        truth_rows = [
            {
                "query": f"truth{i} query",
                "entities": [{"text": f"truth{i}", "type": "CONCEPT"}],
            }
            for i in range(8)
        ]
        approved_rows = [
            {
                "query": "approved0 query",
                "entities": [{"text": "approved0", "type": "CONCEPT"}],
                "example_id": "approved:entity-approved-0",
            },
            {
                "query": "approved1 query",
                "entities": [{"text": "approved1", "type": "CONCEPT"}],
                "example_id": "approved:entity-approved-1",
            },
        ]
        served_rows = [
            {
                "context.span_id": f"served-{i}",
                "attributes.input.value": f"served{i} query",
                "attributes.output.value": json.dumps(
                    {"entities": [{"text": f"served{i}", "type": "CONCEPT"}]}
                ),
            }
            for i in range(6)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.entity_extraction", served_rows)
        )
        manager = FakeTelemetryManager(provider)
        state = {"holdout_queries": []}

        class FakeLLMConfig:
            def resolve(self, purpose):
                return SimpleNamespace(
                    api_base="http://student:8000/v1",
                    model="student-model",
                )

            def resolve_teacher(self):
                return SimpleNamespace(
                    api_base="http://teacher:8000/v1",
                    model="test/teacher-model",
                )

        class FakeConfig:
            def get_llm_config(self):
                return FakeLLMConfig()

        teacher_answers = [
            {
                "reasoning": f"truth reasoning {i}",
                "entities": f"truth{i}|CONCEPT|1.0",
            }
            for i in range(8)
        ]
        student_answers = [teacher_answers[0]] * 20
        real_create_teleprompter = optimization_cli._create_teleprompter

        class RecordingTeleprompter:
            def __init__(self, inner):
                self._inner = inner
                self.max_bootstrapped_demos = inner.max_bootstrapped_demos
                self.max_labeled_demos = inner.max_labeled_demos
                self.max_rounds = inner.max_rounds
                self.metric_threshold = inner.metric_threshold
                self.error_count = inner.error_count

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def compile(self, module, trainset):
                state["trainset_queries"] = [example.query for example in trainset]
                compiled = self._inner.compile(module, trainset=trainset)
                state["compiled_state"] = compiled.dump_state()
                return compiled

        def fake_create_dspy_lm(endpoint):
            if getattr(endpoint, "api_base", "") == "http://teacher:8000/v1":
                return DummyLM(list(teacher_answers))
            return DummyLM(list(student_answers))

        def wrapped_create_teleprompter(*args, **kwargs):
            return RecordingTeleprompter(real_create_teleprompter(*args, **kwargs))

        def score_by_module(module, holdout):
            state["holdout_queries"] = [example.query for example in holdout]
            dump = json.loads(json.dumps(module.dump_state(), default=str))
            demos = dump.get("extractor.predict", {}).get("demos", [])
            return 1.0 if demos else 0.0

        async def approved_data(received_provider, tenant_id, optimizer_type):
            assert received_provider is provider
            assert tenant_id == "test:unit"
            assert optimizer_type == "entity_extraction"
            return approved_rows

        class FakeArtifactManager:
            def __init__(self, received_provider, tenant_id):
                assert received_provider is provider
                assert tenant_id == "test:unit"
                self._tenant_id = tenant_id

            async def load_blob(self, kind, key):
                if (kind, key) == ("config", "entity_extraction_ground_truth"):
                    return json.dumps(truth_rows, separators=(",", ":"))
                if (kind, key) == ("model", "entity_extraction"):
                    return None
                raise AssertionError((kind, key))

            async def save_blob_versioned(self, *args, **kwargs):
                state.setdefault("versioned_saves", []).append(
                    {"args": args, "kwargs": kwargs}
                )
                return "artifact-1", 1

            async def activate_version(self, kind, key, version):
                state.setdefault("activate_calls", []).append((kind, key, version))
                return {"active": {"version": version, "activated_at": "now"}}

            async def get_version_lineage(self, kind, key):
                return []

        async def identity_training_selection(**kwargs):
            train_records = kwargs["train_records"]
            return train_records, SimpleNamespace(
                pool=len(train_records),
                deduped=len(train_records),
                cap=300,
                mmr_applied=False,
                decayed_count=0,
                decayed_example_ids=[],
            )

        p1, p2 = _patch_infra(manager)
        with (
            p1,
            p2,
            patch(
                "cogniverse_foundation.config.utils.create_default_config_manager",
                return_value=SimpleNamespace(),
            ),
            patch(
                "cogniverse_foundation.config.utils.get_config",
                return_value=FakeConfig(),
            ),
            patch(
                "cogniverse_foundation.config.llm_factory.create_dspy_lm",
                side_effect=fake_create_dspy_lm,
            ),
            patch(
                "cogniverse_runtime.optimization_cli._create_teleprompter",
                side_effect=wrapped_create_teleprompter,
            ),
            patch(
                "cogniverse_runtime.optimization_cli._entity_extraction_scores",
                side_effect=score_by_module,
            ),
            patch(
                "cogniverse_runtime.optimization_cli._load_approved_synthetic_data",
                side_effect=approved_data,
            ),
            patch(
                "cogniverse_runtime.optimization_cli._population_floor_from_config",
                return_value=(1, 1),
            ),
            patch(
                "cogniverse_runtime.optimization_cli._min_improvement_from_config",
                return_value=0.05,
            ),
            patch(
                "cogniverse_runtime.optimization_cli._apply_training_selection",
                side_effect=identity_training_selection,
            ),
            patch(
                "cogniverse_agents.optimizer.artifact_manager.ArtifactManager",
                FakeArtifactManager,
            ),
            patch("dspy.configure", lambda **kwargs: None),
        ):
            result = await run_entity_extraction_optimization(
                tenant_id="test:unit", lookback_hours=1
            )

        assert result == {
            "status": "success",
            "spans_found": 6,
            "served_examples": 6,
            "served_scoreable_examples": 6,
            "label_rows": 10,
            "truth_rows": 8,
            "approved_rows": 2,
            "training_examples": 8,
            "holdout_examples": 2,
            "holdout_source": "ground_truth",
            "selection": {
                "pool": 8,
                "deduped": 8,
                "cap": 300,
                "mmr_applied": False,
                "decayed_count": 0,
                "decayed_example_ids": [],
            },
            "bootstrap": {
                "trainset": 8,
                "max_bootstrapped_demos": 4,
                "max_labeled_demos": 8,
                "max_rounds": 1,
                "metric_threshold": 1.0,
                "attempts": 4,
                "errors": 0,
                "examples_walked": 4,
                "accepted": 4,
                "bootstrapped_demos": 4,
                "labeled_demos": 4,
                "metric_values": [1.0, 1.0, 1.0, 1.0],
            },
            "baseline_score": 0.0,
            "current_score": None,
            "candidate_score": 1.0,
            "decision": "promote",
            "version": 1,
            "consumed_example_ids": [
                "truth:0",
                "truth:1",
                "truth:2",
                "truth:3",
                "truth:4",
                "truth:5",
                "truth:6",
                "truth:7",
                "approved:entity-approved-0",
                "approved:entity-approved-1",
            ],
        }
        assert state["trainset_queries"] == [
            "truth0 query",
            "truth1 query",
            "truth2 query",
            "truth3 query",
            "truth4 query",
            "truth5 query",
            "truth6 query",
            "truth7 query",
        ]
        assert state["holdout_queries"] == ["approved0 query", "approved1 query"]
        assert set(state["trainset_queries"]).isdisjoint(
            {
                "served0 query",
                "served1 query",
                "served2 query",
                "served3 query",
                "served4 query",
                "served5 query",
            }
        )
        assert set(state["holdout_queries"]).isdisjoint(
            {
                "served0 query",
                "served1 query",
                "served2 query",
                "served3 query",
                "served4 query",
                "served5 query",
            }
        )

    def test_token_f1_casefolds_whitespace_tokens(self):
        from cogniverse_runtime.optimization_cli import _token_f1

        assert _token_f1("Marie Curie", "marie curie") == 1.0

    def test_entity_quality_pair_f1_table(self):
        from cogniverse_runtime.optimization_cli import _entity_extraction_quality

        ex = _entity_example(entities=("Marie Curie|PERSON|1.0\nradium|CONCEPT|1.0"))
        assert {
            "exact": _entity_extraction_quality(
                _ents("Marie Curie|PERSON|0.9\nradium|CONCEPT|0.8"), ex
            ),
            "wrong_type": _entity_extraction_quality(
                _ents("Marie Curie|CONCEPT|0.9"), ex
            ),
            "partial": _entity_extraction_quality(
                _ents("Marie Curie|PERSON|0.9\nwrong|CONCEPT|0.5"), ex
            ),
            "extra": _entity_extraction_quality(
                _ents("Marie Curie|PERSON|0.9\nwrong1|CONCEPT|0.5\nwrong2|PERSON|0.5"),
                ex,
            ),
            "case": _entity_extraction_quality(
                _ents("marie curie|person|0.9\nRADIUM|concept|0.8"), ex
            ),
            "empty": _entity_extraction_quality(_ents(""), ex),
        } == {
            "exact": 1.0,
            "wrong_type": 0.0,
            "partial": 0.5,
            "extra": 0.4,
            "case": 1.0,
            "empty": 0.0,
        }
        with pytest.raises(ValueError, match="carries no recorded entities"):
            _entity_extraction_quality(_ents("x|T|1.0"), _entity_example(entities=""))

    def test_served_entities_become_pipe_lines_that_serving_parses(self):
        """A served GLiNER record trains in the signature's pipe format, and
        the demo text parses through the agent's own parser to the same
        entities. A JSON-array demo would parse to nothing at serve time."""
        from cogniverse_agents.entity_extraction_agent import (
            Entity,
            EntityExtractionAgent,
            EntityExtractionDeps,
        )
        from cogniverse_runtime.optimization_cli import _entity_extraction_example

        query = "The video begins with a man riding a dirt bike in a dirt field"
        record = {
            "query": query,
            "entities": [
                {
                    "text": "man",
                    "type": "PERSON",
                    "confidence": 0.9509217143058777,
                    "context": "The video begins with a man riding a dirt bike in a dirt",
                },
                {
                    "text": "dirt bike",
                    "type": "CONCEPT",
                    "confidence": 0.9843305945396423,
                    "context": "eo begins with a man riding a dirt bike in a dirt field",
                },
                {"text": "houses", "type": "PLACE"},
            ],
            "example_id": "span:ee-1",
        }

        example = _entity_extraction_example(record)

        assert list(example.toDict()) == ["query", "entities"]
        assert example.toDict() == {
            "query": query,
            "entities": "man|PERSON|0.95\ndirt bike|CONCEPT|0.98\nhouses|PLACE|1.0",
        }
        assert list(example.inputs().toDict()) == ["query"]

        agent = EntityExtractionAgent(deps=EntityExtractionDeps(), port=8010)
        assert agent._parse_entities(example.entities, query) == [
            Entity(
                text="man",
                type="PERSON",
                confidence=0.95,
                context="The video begins with a man riding a dirt bike in a dirt",
            ),
            Entity(
                text="dirt bike",
                type="CONCEPT",
                confidence=0.98,
                context="eo begins with a man riding a dirt bike in a dirt field",
            ),
        ]

    def test_approved_pipe_lines_pass_through_unchanged(self):
        from cogniverse_runtime.optimization_cli import _entity_extraction_example

        example = _entity_extraction_example(
            {
                "query": "find PyTorch tutorials",
                "entities": "PyTorch|TECHNOLOGY|1.0",
                "example_id": "approved:1",
            }
        )

        assert list(example.toDict()) == ["query", "entities"]
        assert example.toDict() == {
            "query": "find PyTorch tutorials",
            "entities": "PyTorch|TECHNOLOGY|1.0",
        }

    def test_entity_extraction_pairs_carry_span_ids(self):
        """Every entity record names the served span it came from."""
        from cogniverse_runtime.optimization_cli import _entity_extraction_pairs

        spans_df = _make_spans_df(
            "cogniverse.entity_extraction",
            [
                {
                    "context.span_id": "ee-1",
                    "attributes.input.value": "find PyTorch tutorials",
                    "attributes.output.value": json.dumps(
                        {"entities": [{"text": "PyTorch", "type": "TECHNOLOGY"}]}
                    ),
                },
                {
                    "context.span_id": "ee-skip",
                    "attributes.input.value": "nothing here",
                    "attributes.output.value": json.dumps({"entities": []}),
                },
                {
                    "context.span_id": "ee-2",
                    "attributes.input.value": "compare JAX and TensorFlow",
                    "attributes.output.value": json.dumps(
                        {
                            "entities": [
                                {"text": "JAX", "type": "TECHNOLOGY"},
                                {"text": "TensorFlow", "type": "TECHNOLOGY"},
                            ]
                        }
                    ),
                },
            ],
        )

        assert _entity_extraction_pairs(spans_df) == [
            {
                "query": "find PyTorch tutorials",
                "entities": [{"text": "PyTorch", "type": "TECHNOLOGY"}],
                "example_id": "span:ee-1",
            },
            {
                "query": "compare JAX and TensorFlow",
                "entities": [
                    {"text": "JAX", "type": "TECHNOLOGY"},
                    {"text": "TensorFlow", "type": "TECHNOLOGY"},
                ],
                "example_id": "span:ee-2",
            },
        ]

    @pytest.mark.asyncio
    async def test_entity_extraction_ground_truth_store_outage_raises(self):
        from cogniverse_agents.optimizer.entity_extraction_ground_truth import (
            EntityExtractionGroundTruthStoreUnavailableError,
        )

        spans_df = _make_spans_df(
            "cogniverse.entity_extraction",
            [
                {
                    "attributes.input.value": "find something",
                    "attributes.output.value": json.dumps({"entities": []}),
                }
            ],
        )
        provider = FakeTelemetryProvider(spans_df)

        with pytest.raises(
            EntityExtractionGroundTruthStoreUnavailableError,
            match="entity_extraction_ground_truth store unavailable",
        ):
            await self._run(
                provider,
                current_blob=None,
                floor=(1, 1),
                ground_truth_error=ConnectionError(
                    "Phoenix refused the dataset request"
                ),
            )

    @pytest.mark.asyncio
    async def test_training_selection_store_override_binds_entity_key(self):
        rows = [
            {
                "context.span_id": f"ee-{i}",
                "attributes.input.value": f"find entity {i}",
                "attributes.output.value": json.dumps(
                    {"entities": [{"text": f"Entity {i}", "type": "CONCEPT"}]}
                ),
            }
            for i in range(2)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.entity_extraction", rows)
        )
        config_manager = _training_selection_config_manager(
            "test:unit",
            {"entity_extraction": {"trainset_cap": 42}},
        )

        _, result = await self._run(
            provider,
            current_blob=None,
            floor=(1, 1),
            config_manager=config_manager,
        )

        assert result["selection"]["cap"] == 42, result

    @pytest.mark.asyncio
    async def test_entity_extraction_promote_persists_and_activates_candidate(self):
        rows = [
            {
                "context.span_id": f"ee-{i}",
                "attributes.input.value": f"find entity {i}",
                "attributes.output.value": json.dumps(
                    {"entities": [{"text": f"Entity {i}", "type": "CONCEPT"}]}
                ),
            }
            for i in range(2)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.entity_extraction", rows)
        )

        state, result = await self._run(
            provider,
            current_blob=None,
            floor=(1, 1),
            score_by_module=lambda module, holdout: (
                1.0
                if json.loads(json.dumps(module.dump_state(), default=str)).get(
                    "compiled"
                )
                == "entity_extraction"
                else 0.0
            ),
        )

        assert result == {
            "status": "success",
            "spans_found": 2,
            "served_examples": 2,
            "served_scoreable_examples": 2,
            "label_rows": 2,
            "truth_rows": 2,
            "approved_rows": 0,
            "training_examples": 1,
            "holdout_examples": 1,
            "holdout_source": "ground_truth",
            **_selection_block(1, 1),
            "bootstrap": _fake_bootstrap_block(1),
            "baseline_score": 0.0,
            "current_score": None,
            "candidate_score": 1.0,
            "decision": "promote",
            "version": 1,
            "consumed_example_ids": ["truth:0", "truth:1"],
        }
        assert state["versioned_saves"] == [
            {
                "kind": "model",
                "key": "entity_extraction",
                "content": '{"compiled": "entity_extraction"}',
                "consumed_example_ids": ["truth:0", "truth:1"],
                "decision": "promote",
                "scored": True,
                "score": 1.0,
                "base_score": 0.0,
                "candidate_score": 1.0,
                "extra_ledger_fields": {},
            }
        ]
        assert state["activate_calls"] == [("model", "entity_extraction", 1)]
        assert state["active_blob"] == '{"compiled": "entity_extraction"}'

    @pytest.mark.asyncio
    async def test_entity_extraction_keeps_persisted_state_without_activating(self):
        rows = [
            {
                "context.span_id": f"ee-{i}",
                "attributes.input.value": f"find entity {i}",
                "attributes.output.value": json.dumps(
                    {"entities": [{"text": f"Entity {i}", "type": "CONCEPT"}]}
                ),
            }
            for i in range(2)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.entity_extraction", rows)
        )
        current_blob = self._base_state()

        state, result = await self._run(
            provider,
            current_blob=current_blob,
            floor=(1, 1),
            score_by_module=self._entity_score_by_module,
        )

        assert result == {
            "status": "success",
            "spans_found": 2,
            "served_examples": 2,
            "served_scoreable_examples": 2,
            "label_rows": 2,
            "truth_rows": 2,
            "approved_rows": 0,
            "training_examples": 1,
            "holdout_examples": 1,
            "holdout_source": "ground_truth",
            **_selection_block(1, 1),
            "bootstrap": _fake_bootstrap_block(1),
            "baseline_score": 1.0,
            "current_score": 1.0,
            "candidate_score": 1.0,
            "decision": "keep",
            "version": 1,
            "consumed_example_ids": ["truth:0", "truth:1"],
        }
        assert state["versioned_saves"] == [
            {
                "kind": "model",
                "key": "entity_extraction",
                "content": '{"compiled": "entity_extraction"}',
                "consumed_example_ids": ["truth:0", "truth:1"],
                "decision": "keep",
                "scored": True,
                "score": 1.0,
                "base_score": 1.0,
                "candidate_score": 1.0,
                "extra_ledger_fields": {},
            }
        ]
        assert state["activate_calls"] == []
        assert state["active_blob"] == current_blob

    @pytest.mark.asyncio
    async def test_entity_extraction_rollback_persists_and_activates_base_state(self):
        rows = [
            {
                "context.span_id": f"ee-{i}",
                "attributes.input.value": f"find entity {i}",
                "attributes.output.value": json.dumps(
                    {"entities": [{"text": f"Entity {i}", "type": "CONCEPT"}]}
                ),
            }
            for i in range(2)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.entity_extraction", rows)
        )
        current_blob = self._served_state()
        base_state = self._base_state()

        state, result = await self._run(
            provider,
            current_blob=current_blob,
            floor=(1, 1),
            score_by_module=self._entity_score_by_module,
        )

        assert result == {
            "status": "success",
            "spans_found": 2,
            "served_examples": 2,
            "served_scoreable_examples": 2,
            "label_rows": 2,
            "truth_rows": 2,
            "approved_rows": 0,
            "training_examples": 1,
            "holdout_examples": 1,
            "holdout_source": "ground_truth",
            **_selection_block(1, 1),
            "bootstrap": _fake_bootstrap_block(1),
            "baseline_score": 1.0,
            "current_score": 0.0,
            "candidate_score": 1.0,
            "decision": "rollback",
            "version": 1,
            "consumed_example_ids": ["truth:0", "truth:1"],
        }
        assert state["versioned_saves"] == [
            {
                "kind": "model",
                "key": "entity_extraction",
                "content": base_state,
                "consumed_example_ids": ["truth:0", "truth:1"],
                "decision": "rollback",
                "scored": True,
                "score": 1.0,
                "base_score": 1.0,
                "candidate_score": 1.0,
                "extra_ledger_fields": {},
            }
        ]
        assert state["activate_calls"] == [("model", "entity_extraction", 1)]
        assert state["active_blob"] == base_state

    @pytest.mark.asyncio
    async def test_score_failure_preserves_selection_block(self):
        rows = [
            {
                "context.span_id": f"ee-{i}",
                "attributes.input.value": f"find entity {i}",
                "attributes.output.value": json.dumps(
                    {"entities": [{"text": f"Entity {i}", "type": "CONCEPT"}]}
                ),
            }
            for i in range(2)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.entity_extraction", rows)
        )
        current_blob = self._served_state()

        def failing_score(module, holdout):
            raise RuntimeError("entity scorer failed")

        state, result = await self._run(
            provider,
            current_blob=current_blob,
            floor=(1, 1),
            score_by_module=failing_score,
        )

        assert state["versioned_saves"] == []
        assert state["activate_calls"] == []
        assert state["active_blob"] == current_blob
        assert result == {
            "status": "failed",
            "error": "entity scorer failed",
            **_selection_block(1, 1),
        }

    @pytest.mark.asyncio
    async def test_entity_extraction_below_population_floor_persists_without_activating(
        self,
    ):
        rows = [
            {
                "context.span_id": f"ee-{i}",
                "attributes.input.value": f"find entity {i}",
                "attributes.output.value": json.dumps(
                    {"entities": [{"text": f"Entity {i}", "type": "CONCEPT"}]}
                ),
            }
            for i in range(29)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.entity_extraction", rows)
        )
        current_blob = self._base_state()

        state, result = await self._run(
            provider,
            current_blob=current_blob,
            floor=(30, 15),
        )

        assert result == {
            "status": "insufficient_population",
            "spans_found": 29,
            "served_examples": 29,
            "served_scoreable_examples": 29,
            "label_rows": 29,
            "truth_rows": 29,
            "approved_rows": 0,
            "examples": 29,
            "distinct_queries": 29,
            "min_samples": 30,
            "min_unique_queries": 15,
            "version": 1,
        }
        assert "selection" not in result
        assert state["versioned_saves"] == [
            {
                "kind": "model",
                "key": "entity_extraction",
                "content": current_blob,
                "consumed_example_ids": [f"truth:{i}" for i in range(29)],
                "decision": "insufficient_population",
                "scored": False,
                "score": None,
                "base_score": None,
                "candidate_score": None,
                "extra_ledger_fields": {},
            }
        ]
        assert state["activate_calls"] == []
        assert state["active_blob"] == current_blob

    @pytest.mark.asyncio
    async def test_entity_extraction_at_population_floor_compiles_and_persists(
        self, monkeypatch
    ):
        rows = [
            {
                "context.span_id": f"ee-{i}",
                "attributes.input.value": f"find entity {i}",
                "attributes.output.value": json.dumps(
                    {"entities": [{"text": f"Entity {i}", "type": "CONCEPT"}]}
                ),
            }
            for i in range(30)
        ]
        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.entity_extraction", rows)
        )
        current_blob = self._base_state()

        async def identity_training_selection(**kwargs):
            train_records = kwargs["train_records"]
            return train_records, SimpleNamespace(
                pool=len(train_records),
                deduped=len(train_records),
                cap=300,
                mmr_applied=False,
                decayed_count=0,
                decayed_example_ids=[],
            )

        monkeypatch.setattr(
            "cogniverse_runtime.optimization_cli._apply_training_selection",
            identity_training_selection,
        )

        state, result = await self._run(
            provider,
            current_blob=current_blob,
            floor=(30, 15),
            score_by_module=self._entity_score_by_module,
        )

        assert result == {
            "status": "success",
            "spans_found": 30,
            "served_examples": 30,
            "served_scoreable_examples": 30,
            "label_rows": 30,
            "truth_rows": 30,
            "approved_rows": 0,
            "training_examples": 23,
            "holdout_examples": 7,
            "holdout_source": "ground_truth",
            **_selection_block(23, 23),
            "bootstrap": _fake_bootstrap_block(23),
            "baseline_score": 1.0,
            "current_score": 1.0,
            "candidate_score": 1.0,
            "decision": "keep",
            "version": 1,
            "consumed_example_ids": [f"truth:{i}" for i in range(30)],
        }
        assert state["versioned_saves"] == [
            {
                "kind": "model",
                "key": "entity_extraction",
                "content": '{"compiled": "entity_extraction"}',
                "consumed_example_ids": [f"truth:{i}" for i in range(30)],
                "decision": "keep",
                "scored": True,
                "score": 1.0,
                "base_score": 1.0,
                "candidate_score": 1.0,
                "extra_ledger_fields": {},
            }
        ]
        assert state["activate_calls"] == []
        assert state["active_blob"] == current_blob


# ---------------------------------------------------------------------------
# Test: routing mode
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test: synthetic data merge helper
# ---------------------------------------------------------------------------


class TestEntityBootstrapThreshold:
    """The entity bootstrap accepts a teacher trace by token-set F1 against
    ``metric_threshold`` and records every attempt's score, so one run yields
    the acceptance histogram."""

    _ANSWERS = [
        {
            "reasoning": "r1",
            "entities": "Marie Curie|PERSON|0.9",
        },
        {
            "reasoning": "r2",
            "entities": "Alan Turing|PERSON|0.9",
        },
        {
            "reasoning": "r3",
            "entities": "Ada Lovelace|PERSON|0.9",
        },
    ]

    @staticmethod
    def _trainset(tag: str) -> list:
        import dspy

        rows = [
            (f"{tag} curie", "Marie Curie|PERSON|1.0"),
            (f"{tag} turing", "Alan Turing|PERSON|1.0\nEnigma|CONCEPT|1.0"),
            (f"{tag} lovelace", "Ada Lovelace|PERSON|1.0"),
        ]
        return [
            dspy.Example(query=query, entities=entities).with_inputs("query")
            for query, entities in rows
        ]

    def _compile(self, tag: str, threshold: float):
        import dspy
        from dspy.utils.dummies import DummyLM

        from cogniverse_agents.entity_extraction_agent import EntityExtractionModule
        from cogniverse_runtime.optimization_cli import (
            BootstrapMetricRecorder,
            _bootstrap_report,
            _create_teleprompter,
            _entity_extraction_quality,
        )

        trainset = self._trainset(tag)
        recorder = BootstrapMetricRecorder(
            _entity_extraction_quality,
            tenant="test:unit",
            threshold=threshold,
        )
        with dspy.context(lm=DummyLM(list(self._ANSWERS))):
            teleprompter = _create_teleprompter(
                len(trainset), metric=recorder, metric_threshold=threshold
            )
            compiled = teleprompter.compile(EntityExtractionModule(), trainset=trainset)
        report = _bootstrap_report(recorder, teleprompter, compiled, len(trainset))
        ((_, predictor),) = compiled.named_predictors()
        demos = [
            (demo.get("augmented", False), demo["entities"]) for demo in predictor.demos
        ]
        return recorder, report, demos

    def test_exact_bar_keeps_only_exact_traces_and_records_every_attempt(self, caplog):
        caplog.set_level(logging.INFO, logger="cogniverse_runtime.optimization_cli")

        recorder, report, demos = self._compile("t1", 1.0)

        assert recorder.attempts == [
            ("t1 curie", 1.0),
            ("t1 turing", 0.6666666666666666),
            ("t1 lovelace", 1.0),
        ]
        assert report == {
            "trainset": 3,
            "max_bootstrapped_demos": 4,
            "max_labeled_demos": 8,
            "max_rounds": 1,
            "metric_threshold": 1.0,
            "attempts": 3,
            "errors": 0,
            "examples_walked": 3,
            "accepted": 2,
            "bootstrapped_demos": 2,
            "labeled_demos": 1,
            "metric_values": [0.6666666666666666, 1.0, 1.0],
        }
        assert demos == [
            (True, "Marie Curie|PERSON|0.9"),
            (True, "Ada Lovelace|PERSON|0.9"),
            (False, "Alan Turing|PERSON|1.0\nEnigma|CONCEPT|1.0"),
        ]
        assert [
            record.getMessage()
            for record in caplog.records
            if record.getMessage().startswith("bootstrap attempt")
        ] == [
            "bootstrap attempt 1 query='t1 curie' metric=1.000 accepted=True",
            "bootstrap attempt 2 query='t1 turing' metric=0.667 accepted=False",
            "bootstrap attempt 3 query='t1 lovelace' metric=1.000 accepted=True",
        ]

    def test_lower_bar_accepts_partial_traces(self):
        recorder, report, demos = self._compile("t2", 0.75)

        assert [score for _, score in recorder.attempts] == [
            1.0,
            0.6666666666666666,
            1.0,
        ]
        assert (
            report["accepted"],
            report["bootstrapped_demos"],
            report["labeled_demos"],
            report["metric_threshold"],
        ) == (2, 2, 1, 0.75)
        assert demos == [
            (True, "Marie Curie|PERSON|0.9"),
            (True, "Ada Lovelace|PERSON|0.9"),
            (False, "Alan Turing|PERSON|1.0\nEnigma|CONCEPT|1.0"),
        ]

    def test_bar_never_drops_below_the_served_score(self):
        from cogniverse_runtime.optimization_cli import (
            ENTITY_BOOTSTRAP_METRIC_THRESHOLD,
            _entity_bootstrap_threshold,
        )

        assert ENTITY_BOOTSTRAP_METRIC_THRESHOLD == 1.0
        assert {
            "bar_above_served": _entity_bootstrap_threshold(0.392, 0.621, bar=0.9),
            "served_above_bar": _entity_bootstrap_threshold(0.392, 0.621, bar=0.5),
            "no_current_artifact": _entity_bootstrap_threshold(0.7, None, bar=0.5),
            "default_bar": _entity_bootstrap_threshold(0.392, 0.621),
        } == {
            "bar_above_served": 0.9,
            "served_above_bar": 0.621,
            "no_current_artifact": 0.7,
            "default_bar": 1.0,
        }

    def test_the_default_bar_stays_at_one(self):
        from cogniverse_runtime.optimization_cli import (
            ENTITY_BOOTSTRAP_METRIC_THRESHOLD as bar,
        )
        from cogniverse_runtime.optimization_cli import (
            _entity_bootstrap_threshold,
        )

        assert bar == 1.0
        assert _entity_bootstrap_threshold(0.392, 0.621) == 1.0
        assert _entity_bootstrap_threshold(0.392, 0.621, bar=0.5) == 0.621

    def test_recorder_appends_one_jsonl_row_per_attempt(self, monkeypatch, tmp_path):
        from datetime import datetime as _datetime
        from datetime import timezone as _timezone

        from cogniverse_runtime import optimization_cli
        from cogniverse_runtime.optimization_cli import BootstrapMetricRecorder

        class FixedDatetime(_datetime):
            @classmethod
            def now(cls, tz=None):
                return _datetime(2026, 8, 26, 12, 34, 56, tzinfo=_timezone.utc)

        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        monkeypatch.setattr(optimization_cli, "datetime", FixedDatetime)

        recorder = BootstrapMetricRecorder(
            lambda prediction, example: (
                1.0 if prediction.entities == example.entities else 0.0
            ),
            tenant="acme:production",
            threshold=0.75,
        )
        example_one = SimpleNamespace(query="alpha", entities="alpha|CONCEPT|1.0")
        example_two = SimpleNamespace(query="beta", entities="beta|CONCEPT|1.0")

        assert (
            recorder(example_one, SimpleNamespace(entities="alpha|CONCEPT|1.0")) == 1.0
        )
        assert recorder(example_two, SimpleNamespace(entities="wrong")) == 0.0
        assert recorder.attempts == [("alpha", 1.0), ("beta", 0.0)]

        path = tmp_path / "cogniverse" / "bootstrap_attempts.jsonl"
        assert path.read_text(encoding="utf-8").splitlines() == [
            (
                '{"optimizer":"entity_extraction","tenant":"acme:production",'
                '"query":"alpha","metric":1.0,"threshold":0.75,"accepted":true,'
                '"ts":"2026-08-26T12:34:56+00:00"}'
            ),
            (
                '{"optimizer":"entity_extraction","tenant":"acme:production",'
                '"query":"beta","metric":0.0,"threshold":0.75,"accepted":false,'
                '"ts":"2026-08-26T12:34:56+00:00"}'
            ),
        ]


class TestSyntheticDataMerge:
    def test_approved_dataset_name_canonicalizes_tenant(self):
        from cogniverse_core.approval.interfaces import (
            approved_synthetic_dataset_name,
        )

        assert (
            approved_synthetic_dataset_name("acme")
            == "approved_synthetic_data-acme:acme"
        )
        assert (
            approved_synthetic_dataset_name("acme:production")
            == "approved_synthetic_data-acme:production"
        )
        with pytest.raises(ValueError, match="tenant_id is required"):
            approved_synthetic_dataset_name("")

    @pytest.mark.asyncio
    async def test_load_approved_synthetic_no_data(self):
        """Returns empty list when no synthetic data exists."""
        from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data

        provider = FakeTelemetryProvider()
        result = await _load_approved_synthetic_data(
            provider, "default", "query_enhancement"
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_load_approved_synthetic_isolates_tenant_rows_and_order(self):
        """Concurrent optimizers consume only their tenant's ordered records."""
        from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data

        class ApprovedDatasetStore:
            def __init__(self):
                self.names = []

            async def get_dataset(self, name):
                self.names.append(name)
                frames = {
                    "approved_synthetic_data-acme:alpha": pd.DataFrame(
                        [
                            {
                                "input": {
                                    "item_id": "alpha-approved-1",
                                    "confidence": 0.91,
                                    "status": "approved",
                                    "created_at": "2026-08-05T01:00:00+00:00",
                                    "reviewed_at": "2026-08-05T01:01:00+00:00",
                                    "query": "Find exact PyTorch tutorials",
                                    "enhanced_query": "Find exact PyTorch framework tutorials",
                                    "expansion_terms": ["framework"],
                                    "synonyms": ["torch"],
                                    "context": "document_text",
                                    "reasoning": "Framework disambiguates the library.",
                                    "metadata.batch_id": "batch-a",
                                    "metadata.agent_type": "query_enhancement",
                                    "context.optimizer": "query_enhancement",
                                    "context.purpose": "optimizer training",
                                }
                            },
                            {
                                "input": {
                                    "item_id": "pending",
                                    "status": "pending_review",
                                    "query": "Do not consume",
                                    "context.optimizer": "query_enhancement",
                                }
                            },
                            {
                                "input": {
                                    "item_id": "alpha-approved-2",
                                    "confidence": 0.88,
                                    "status": "approved",
                                    "created_at": "2026-08-05T01:02:00+00:00",
                                    "reviewed_at": "2026-08-05T01:03:00+00:00",
                                    "query": "Find exact JAX tutorials",
                                    "enhanced_query": "Find exact JAX framework tutorials",
                                    "expansion_terms": ["framework"],
                                    "synonyms": [],
                                    "context": "document_text",
                                    "reasoning": "Framework distinguishes JAX documentation.",
                                    "metadata.batch_id": "batch-b",
                                    "metadata.agent_type": "query_enhancement",
                                    "context.optimizer": "query_enhancement",
                                    "context.purpose": "optimizer training",
                                }
                            },
                        ]
                    ),
                    "approved_synthetic_data-acme:beta": pd.DataFrame(
                        [
                            {
                                "input": {
                                    "item_id": "beta-approved-1",
                                    "confidence": 0.95,
                                    "status": "approved",
                                    "query": "Find exact Vespa tutorials",
                                    "enhanced_query": "Find exact Vespa search tutorials",
                                    "expansion_terms": ["search"],
                                    "synonyms": [],
                                    "context": "document_text",
                                    "reasoning": "Search identifies the Vespa platform.",
                                    "metadata.agent_type": "query_enhancement",
                                    "context.optimizer": "query_enhancement",
                                }
                            }
                        ]
                    ),
                }
                frame = frames[name]
                frame["input"] = frame["input"].map(_signed_approved_record)
                return frame

        provider = FakeTelemetryProvider()
        dataset_store = ApprovedDatasetStore()
        provider._dataset_store = dataset_store

        alpha, beta = await asyncio.gather(
            _load_approved_synthetic_data(provider, "acme:alpha", "query_enhancement"),
            _load_approved_synthetic_data(provider, "acme:beta", "query_enhancement"),
        )

        assert alpha == [
            {
                "query": "Find exact PyTorch tutorials",
                "enhanced_query": "Find exact PyTorch framework tutorials",
                "expansion_terms": ["framework"],
                "synonyms": ["torch"],
                "context": "document_text",
                "reasoning": "Framework disambiguates the library.",
                "example_id": "approved:alpha-approved-1",
            },
            {
                "query": "Find exact JAX tutorials",
                "enhanced_query": "Find exact JAX framework tutorials",
                "expansion_terms": ["framework"],
                "synonyms": [],
                "context": "document_text",
                "reasoning": "Framework distinguishes JAX documentation.",
                "example_id": "approved:alpha-approved-2",
            },
        ]
        assert beta == [
            {
                "query": "Find exact Vespa tutorials",
                "enhanced_query": "Find exact Vespa search tutorials",
                "expansion_terms": ["search"],
                "synonyms": [],
                "context": "document_text",
                "reasoning": "Search identifies the Vespa platform.",
                "example_id": "approved:beta-approved-1",
            }
        ]
        assert dataset_store.names == [
            "approved_synthetic_data-acme:alpha",
            "approved_synthetic_data-acme:beta",
        ]

    @pytest.mark.asyncio
    async def test_load_approved_synthetic_filters_nonapproved_and_other_optimizer(
        self,
    ):
        from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data

        class ApprovedDatasetStore:
            async def get_dataset(self, name):
                assert name == "approved_synthetic_data-acme:alpha"
                records = [
                    {
                        "item_id": "pending",
                        "status": "pending_review",
                        "query": "Do not consume",
                        "context.optimizer": "query_enhancement",
                    },
                    {
                        "item_id": "other-optimizer",
                        "status": "approved",
                        "query": "Wrong optimizer",
                        "context.optimizer": "profile",
                    },
                ]
                return pd.DataFrame(
                    [{"input": _signed_approved_record(record)} for record in records]
                )

        provider = FakeTelemetryProvider()
        provider._dataset_store = ApprovedDatasetStore()

        result = await _load_approved_synthetic_data(
            provider, "acme:alpha", "query_enhancement"
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_load_approved_synthetic_requires_canonical_agent_owner(self):
        from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data

        class ApprovedDatasetStore:
            async def get_dataset(self, name):
                record = {
                    "item_id": "wrong-owner",
                    "status": "approved",
                    "query": "Find exact PyTorch tutorials",
                    "enhanced_query": "Find exact PyTorch framework tutorials",
                    "expansion_terms": ["framework"],
                    "synonyms": [],
                    "context": "document_text",
                    "reasoning": "Framework disambiguates the library.",
                    "metadata.agent_type": "profile_selection",
                    "context.optimizer": "query_enhancement",
                }
                return pd.DataFrame([{"input": _signed_approved_record(record)}])

        provider = FakeTelemetryProvider()
        provider._dataset_store = ApprovedDatasetStore()

        with pytest.raises(
            ValueError,
            match=(
                "Approved synthetic dataset row 0 for optimizer=query_enhancement "
                "requires metadata.agent_type='query_enhancement', got "
                "'profile_selection'"
            ),
        ):
            await _load_approved_synthetic_data(
                provider, "acme:alpha", "query_enhancement"
            )

    @pytest.mark.asyncio
    async def test_load_approved_synthetic_validates_values_before_returning_them(self):
        from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data

        class ApprovedDatasetStore:
            async def get_dataset(self, name):
                record = {
                    "item_id": "invalid-profile",
                    "status": "approved",
                    "query": "find transformer lectures",
                    "available_profiles": "video_colpali,document_colpali",
                    "selected_profile": "missing_profile",
                    "reasoning": "A selector response with an invalid target.",
                    "query_intent": "document_search",
                    "modality": "document",
                    "complexity": "complex",
                    "metadata.agent_type": "profile_selection",
                    "context.optimizer": "profile",
                }
                return pd.DataFrame([{"input": _signed_approved_record(record)}])

        provider = FakeTelemetryProvider()
        provider._dataset_store = ApprovedDatasetStore()

        with pytest.raises(
            ValueError,
            match=(
                "Approved synthetic dataset row 0 for optimizer=profile "
                "selected_profile 'missing_profile' is absent from "
                "available_profiles"
            ),
        ):
            await _load_approved_synthetic_data(provider, "acme:alpha", "profile")

    @pytest.mark.asyncio
    async def test_load_approved_synthetic_surfaces_dataset_outage(self):
        """A Phoenix outage cannot masquerade as an empty approved dataset."""
        from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data

        class UnavailableDatasetStore:
            async def get_dataset(self, name):
                raise ConnectionError("Phoenix refused the dataset request")

        provider = FakeTelemetryProvider()
        provider._dataset_store = UnavailableDatasetStore()

        with pytest.raises(
            RuntimeError,
            match=(
                "Failed to load approved synthetic data for "
                "tenant=acme:production optimizer=query_enhancement "
                "dataset=approved_synthetic_data-acme:production"
            ),
        ) as error:
            await _load_approved_synthetic_data(
                provider, "acme:production", "query_enhancement"
            )

        assert isinstance(error.value.__cause__, ConnectionError)
        assert str(error.value.__cause__) == "Phoenix refused the dataset request"

    @pytest.mark.asyncio
    async def test_load_approved_synthetic_rejects_missing_provider_frame(self):
        """Only the provider's typed not-found result represents absence."""
        from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data

        class MissingFrameDatasetStore:
            async def get_dataset(self, name):
                assert name == "approved_synthetic_data-acme:production"
                return None

        provider = FakeTelemetryProvider()
        provider._dataset_store = MissingFrameDatasetStore()

        with pytest.raises(
            RuntimeError,
            match=(
                "Approved synthetic dataset provider returned no frame for "
                "tenant=acme:production optimizer=query_enhancement "
                "dataset=approved_synthetic_data-acme:production"
            ),
        ):
            await _load_approved_synthetic_data(
                provider, "acme:production", "query_enhancement"
            )


# ---------------------------------------------------------------------------
# Test: _create_teleprompter optimizer selection
# ---------------------------------------------------------------------------


class TestCreateTeleprompter:
    """Verify optimizer selection based on training set size."""

    def test_bootstrap_accepts_only_exact_approved_labels(self):
        import dspy

        from cogniverse_runtime.optimization_cli import (
            _approved_example_exact_metric,
            _create_teleprompter,
        )

        example = dspy.Example(
            query="transformer architecture",
            enhanced_query="transformer architecture attention mechanism",
            reasoning="Added the reviewed attention term.",
        ).with_inputs("query")

        assert (
            _approved_example_exact_metric(
                example,
                dspy.Prediction(
                    enhanced_query="transformer architecture attention mechanism",
                    reasoning="Added the reviewed attention term.",
                ),
            )
            is True
        )
        assert (
            _approved_example_exact_metric(
                example,
                dspy.Prediction(
                    enhanced_query="Explain transformer architecture",
                    reasoning="Replaced the approved labels.",
                ),
            )
            is False
        )
        assert _create_teleprompter(1).metric is _approved_example_exact_metric

    def test_small_trainset_uses_bootstrap(self):
        """< 50 examples should use BootstrapFewShot."""
        from dspy.teleprompt import BootstrapFewShot

        from cogniverse_runtime.optimization_cli import _create_teleprompter

        tp = _create_teleprompter(10)
        assert isinstance(tp, BootstrapFewShot), (
            f"Expected BootstrapFewShot for 10 examples, got {type(tp).__name__}"
        )

    def test_teacher_settings_forwarded_to_bootstrap(self):
        """The configured teacher LM must reach BootstrapFewShot — DSPy runs
        the bootstrap teacher inside dspy.context(**teacher_settings), so an
        unforwarded teacher means the student silently teaches itself."""
        from cogniverse_runtime.optimization_cli import _create_teleprompter

        sentinel = object()
        small = _create_teleprompter(10, teacher_settings={"lm": sentinel})
        assert small.teacher_settings == {"lm": sentinel}
        assert small.max_bootstrapped_demos == 4

        scaled = _create_teleprompter(50, teacher_settings={"lm": sentinel})
        assert scaled.teacher_settings == {"lm": sentinel}
        assert scaled.max_bootstrapped_demos == 8

    def test_teacher_settings_default_empty(self):
        from cogniverse_runtime.optimization_cli import _create_teleprompter

        tp = _create_teleprompter(10)
        assert tp.teacher_settings == {}

    def test_49_uses_bootstrap(self):
        """Boundary: 49 examples should still use BootstrapFewShot."""
        from dspy.teleprompt import BootstrapFewShot

        from cogniverse_runtime.optimization_cli import _create_teleprompter

        tp = _create_teleprompter(49)
        assert isinstance(tp, BootstrapFewShot), (
            f"Expected BootstrapFewShot for 49 examples, got {type(tp).__name__}"
        )

    def test_50_uses_scaled_bootstrap(self):
        """Boundary: >= 50 examples should use scaled BootstrapFewShot."""
        from dspy.teleprompt import BootstrapFewShot

        from cogniverse_runtime.optimization_cli import _create_teleprompter

        tp = _create_teleprompter(50)
        assert isinstance(tp, BootstrapFewShot)
        assert tp.max_bootstrapped_demos == 8
        assert tp.max_labeled_demos == 16

    def test_large_trainset_uses_scaled_bootstrap(self):
        """200 examples should use scaled BootstrapFewShot with more demos."""
        from dspy.teleprompt import BootstrapFewShot

        from cogniverse_runtime.optimization_cli import _create_teleprompter

        tp = _create_teleprompter(200)
        assert isinstance(tp, BootstrapFewShot)
        assert tp.max_bootstrapped_demos == 8
        assert tp.max_labeled_demos == 16

    def test_zero_uses_bootstrap(self):
        """Edge case: 0 examples should use BootstrapFewShot."""
        from dspy.teleprompt import BootstrapFewShot

        from cogniverse_runtime.optimization_cli import _create_teleprompter

        tp = _create_teleprompter(0)
        assert isinstance(tp, BootstrapFewShot)

    def test_metric_threshold_forwarded_to_bootstrap(self):
        from cogniverse_runtime.optimization_cli import (
            _create_teleprompter,
            _entity_extraction_quality,
        )

        assert _create_teleprompter(10).metric_threshold is None
        scaled = _create_teleprompter(
            50, metric=_entity_extraction_quality, metric_threshold=0.7
        )
        assert (scaled.metric, scaled.metric_threshold) == (
            _entity_extraction_quality,
            0.7,
        )


# ---------------------------------------------------------------------------
# Test: synthetic generation mode
# ---------------------------------------------------------------------------


def _gateway_spans(rows: list[dict]) -> pd.DataFrame:
    """Build a ``cogniverse.gateway`` spans DataFrame with the canonical
    ``output.value`` decision populated from ``rows``. Each row is
    ``{"complexity": ..., "confidence": ..., "status_code": ...}``;
    ``status_code`` defaults to ``OK`` if absent. The DataFrame shape matches
    what Phoenix's ``get_all_spans`` returns."""
    records = []
    for r in rows:
        records.append(
            {
                "attributes.output.value": json.dumps(
                    {
                        "complexity": r.get("complexity"),
                        "confidence": r.get("confidence"),
                    }
                ),
                "status_code": r.get("status_code", "OK"),
            }
        )
    df = pd.DataFrame(records)
    df["name"] = "cogniverse.gateway"
    return df


class TestComputeGatewayThresholdsAlgorithm:
    """Tight assertions on every output field of ``_compute_gateway_thresholds``.

    The calibration has three branches:
      (1) simple_error_rate > 0.2        → optimized = min(0.4 + 0.1, 0.95) = 0.5
      (2) complex_err < 0.05 AND mean > 0.8 → optimized = max(0.4 - 0.05, 0.3) = 0.35
      (3) otherwise                       → optimized = 0.4 (default)

    ``gliner_threshold`` is always ``round(max(0.15, min(p25 * 0.8, 0.5)), 3)``.
    Tests cover each branch plus degenerate inputs.
    """

    def test_empty_df_reports_no_data(self):
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        result = _compute_gateway_thresholds(pd.DataFrame())
        assert result == {"status": "no_data", "spans_found": 0}

    def test_missing_attributes_gateway_column(self):
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        df = pd.DataFrame([{"name": "cogniverse.gateway", "status_code": "OK"}])
        result = _compute_gateway_thresholds(df)
        assert result == {
            "status": "no_data",
            "spans_found": 1,
            "reason": "no_gateway_attributes",
        }

    def test_no_confidence_values_across_spans(self):
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        df = _gateway_spans(
            [
                {"complexity": "simple", "confidence": None},
                {"complexity": "complex", "confidence": None},
            ]
        )
        result = _compute_gateway_thresholds(df)
        assert result == {
            "status": "no_data",
            "spans_found": 2,
            "reason": "no_confidence_data",
        }

    def test_high_simple_error_rate_raises_threshold(self):
        """Branch (1): 5 of 10 simple spans are errors → rate = 0.5 > 0.2.
        Optimizer raises fast_path threshold from 0.4 → 0.5."""
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        rows = []
        # 10 simple spans: 5 with status=ERROR (high error rate), all conf=0.5.
        for i in range(10):
            rows.append(
                {
                    "complexity": "simple",
                    "confidence": 0.5,
                    "status_code": "ERROR" if i < 5 else "OK",
                }
            )
        # 2 complex spans, no errors.
        rows += [{"complexity": "complex", "confidence": 0.5} for _ in range(2)]

        result = _compute_gateway_thresholds(_gateway_spans(rows))
        assert result["status"] == "ready"
        assert result["spans_found"] == 12

        t = result["thresholds"]
        assert t["fast_path_confidence_threshold"] == 0.5
        # All confidences = 0.5 → p25 = 0.5 → gliner = round(min(0.5*0.8, 0.5), 3)
        assert t["gliner_threshold"] == 0.4

        a = t["analysis"]
        assert a["total_spans"] == 12
        assert a["simple_count"] == 10
        assert a["complex_count"] == 2
        assert a["simple_error_rate"] == 0.5
        assert a["complex_error_rate"] == 0.0
        assert a["mean_confidence"] == 0.5
        assert a["p25_confidence"] == 0.5

    def test_high_confidence_low_complex_errors_lowers_threshold(self):
        """Branch (2): complex_error_rate = 0, mean_confidence = 0.9 > 0.8,
        simple_error_rate = 0 (not > 0.2). Optimizer lowers the threshold from
        0.4 → max(0.35, 0.3) = 0.35 so MORE queries stay on the fast path — the
        floor must be below the 0.4 default, not above it."""
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        rows = [{"complexity": "simple", "confidence": 0.9} for _ in range(10)] + [
            {"complexity": "complex", "confidence": 0.9} for _ in range(5)
        ]

        result = _compute_gateway_thresholds(_gateway_spans(rows))
        assert result["status"] == "ready"

        t = result["thresholds"]
        # Genuinely lowered from the 0.4 default (the pre-fix 0.5 floor RAISED it).
        assert t["fast_path_confidence_threshold"] == pytest.approx(0.35)
        assert t["fast_path_confidence_threshold"] < 0.4
        # p25 = 0.9 → gliner = round(max(0.15, min(0.72, 0.5)), 3) = 0.5
        assert t["gliner_threshold"] == 0.5

        a = t["analysis"]
        assert a["mean_confidence"] == 0.9
        assert a["p25_confidence"] == 0.9
        assert a["simple_error_rate"] == 0.0
        assert a["complex_error_rate"] == 0.0

    def test_moderate_signal_keeps_default_threshold(self):
        """Branch (3): simple_error_rate = 0.1 (not > 0.2), mean_confidence =
        0.55 (not > 0.8). Neither branch fires; threshold stays at 0.4."""
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        rows = []
        for i in range(10):
            rows.append(
                {
                    "complexity": "simple",
                    "confidence": 0.6 if i < 5 else 0.5,
                    "status_code": "ERROR" if i == 0 else "OK",
                }
            )
        rows += [{"complexity": "complex", "confidence": 0.5} for _ in range(2)]

        result = _compute_gateway_thresholds(_gateway_spans(rows))
        t = result["thresholds"]
        assert t["fast_path_confidence_threshold"] == 0.4

        a = t["analysis"]
        # 1 of 10 simple = 0.1; doesn't trigger branch 1.
        assert a["simple_error_rate"] == 0.1
        # Mean of 5x 0.6 + 5x 0.5 + 2x 0.5 over 12 = 6.5 / 12 ≈ 0.5417
        assert a["mean_confidence"] == 0.5417

    def test_gliner_floor_at_0_15(self):
        """When p25 * 0.8 < 0.15, gliner_threshold floors at 0.15 (prevents
        the GLiNER model from being effectively disabled by a near-zero
        threshold derived from low-confidence training data)."""
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        rows = [{"complexity": "simple", "confidence": 0.05} for _ in range(4)]
        result = _compute_gateway_thresholds(_gateway_spans(rows))
        t = result["thresholds"]
        # p25 = 0.05, p25*0.8 = 0.04, below the 0.15 floor.
        assert t["gliner_threshold"] == 0.15

    def test_gliner_ceiling_at_0_5(self):
        """When p25 * 0.8 > 0.5, gliner_threshold caps at 0.5 (preserves
        recall — too high a threshold means GLiNER misses valid entities)."""
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        rows = [{"complexity": "simple", "confidence": 0.95} for _ in range(4)]
        result = _compute_gateway_thresholds(_gateway_spans(rows))
        t = result["thresholds"]
        # p25 = 0.95, p25*0.8 = 0.76, caps at 0.5.
        assert t["gliner_threshold"] == 0.5

    def test_status_col_absent_means_zero_error_rate(self):
        """Spans without a ``status_code`` column count as all-OK — the
        optimizer must not crash on minimal Phoenix schemas that lack it."""
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        df = _gateway_spans([{"complexity": "simple", "confidence": 0.5}])
        df = df.drop(columns=["status_code"])
        result = _compute_gateway_thresholds(df)
        a = result["thresholds"]["analysis"]
        assert a["simple_error_rate"] == 0.0
        assert a["complex_error_rate"] == 0.0

    def test_malformed_attributes_dict_treated_as_missing(self):
        """Defensive: an ``output.value`` that parses to a non-dict (e.g. a
        stray string from a malformed write) must not crash the compute."""
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        df = pd.DataFrame(
            [
                {
                    "name": "cogniverse.gateway",
                    "attributes.output.value": json.dumps("not-a-dict"),
                    "status_code": "OK",
                }
            ]
        )
        result = _compute_gateway_thresholds(df)
        # No decision dict extractable → treated as missing, no crash.
        assert result["status"] == "no_data"
        assert result["reason"] == "no_gateway_attributes"


class TestSyntheticGeneration:
    """Verify synthetic generation CLI mode."""

    @pytest.mark.asyncio
    async def test_rejects_optimizer_without_approved_training_consumer(self):
        from cogniverse_runtime.optimization_cli import run_synthetic_generation

        with pytest.raises(
            ValueError,
            match=(
                r"synthetic optimizer types have no approved training-data "
                r"consumer: \['workflow'\]"
            ),
        ):
            await run_synthetic_generation(
                tenant_id="acme:production",
                optimizer_types=["profile", "workflow"],
            )

    @pytest.mark.parametrize(
        ("results", "expected_status"),
        [
            (
                {
                    "profile": {"status": "success"},
                    "routing": {"status": "success"},
                },
                "success",
            ),
            (
                {
                    "profile": {"status": "success"},
                    "routing": {"status": "no_data"},
                },
                "success",
            ),
            (
                {
                    "profile": {"status": "no_data"},
                    "routing": {"status": "no_data"},
                },
                "no_data",
            ),
            (
                {
                    "profile": {"status": "success"},
                    "routing": {"status": "failed", "error": "backend down"},
                },
                "failed",
            ),
            (
                {
                    "profile": {"status": "no_data"},
                    "routing": {"status": "error", "error": "invalid result"},
                },
                "failed",
            ),
        ],
        ids=[
            "all-success",
            "success-and-no-data",
            "all-no-data",
            "success-and-failed",
            "no-data-and-error",
        ],
    )
    def test_aggregate_status_preserves_requested_result_contract(
        self,
        results,
        expected_status,
    ):
        from cogniverse_runtime.optimization_cli import _synthetic_aggregate_status

        assert _synthetic_aggregate_status(results) == expected_status

    @pytest.mark.asyncio
    @pytest.mark.parametrize("missing_section", ["backend", "synthetic", "agents"])
    async def test_missing_config_fails_before_backend_access(
        self,
        missing_section,
        fake_telemetry_manager,
    ):
        from cogniverse_runtime.optimization_cli import run_synthetic_generation

        sections = _synthetic_runtime_sections()
        sections.pop(missing_section)
        tenant_config = SimpleNamespace(
            get_llm_config=lambda: SimpleNamespace(primary="test-lm"),
            get=lambda key, default=None: sections.get(key, default),
        )
        backend_accesses = 0

        def record_backend_access(*args, **kwargs):
            nonlocal backend_accesses
            backend_accesses += 1
            return object()

        with (
            patch(_PATCH_CONFIG),
            _patch_telemetry(fake_telemetry_manager),
            patch(
                "cogniverse_foundation.config.utils.get_config",
                return_value=tenant_config,
            ),
            patch(
                "cogniverse_foundation.config.llm_factory.create_dspy_lm",
                return_value=object(),
            ),
            patch(
                "cogniverse_core.registries.backend_registry."
                "BackendRegistry.get_search_backend",
                side_effect=record_backend_access,
            ),
        ):
            result = await run_synthetic_generation(
                tenant_id="acme:invalid",
                optimizer_types=["query_enhancement"],
                count=1,
            )

        error = (
            "Synthetic runtime configuration for tenant='acme:invalid' "
            f"requires object section '{missing_section}'"
        )
        assert result == {
            "status": "failed",
            "results": {"query_enhancement": {"status": "failed", "error": error}},
        }
        assert backend_accesses == 0

    @pytest.mark.asyncio
    async def test_routing_generation_receives_production_entity_extractor(
        self,
        fake_telemetry_manager,
    ):
        from cogniverse_runtime.optimization_cli import run_synthetic_generation
        from cogniverse_synthetic.schemas import SyntheticDataResponse

        sections = _synthetic_runtime_sections(marker="routing")
        tenant_config = SimpleNamespace(
            get_llm_config=lambda: SimpleNamespace(primary="test-lm"),
            get=lambda key, default=None: sections.get(key, default),
        )
        config_manager = object()
        backend = object()
        build_calls = []
        extraction_calls = []

        async def extract_entities(source_text, tenant_id):
            extraction_calls.append((source_text, tenant_id))
            return {
                "query": source_text,
                "entities": [
                    {"text": "Marie Curie", "type": "PERSON"},
                    {"text": "radium", "type": "CONCEPT"},
                ],
                "relationships": [],
            }

        async def build_extractor(**kwargs):
            build_calls.append(kwargs)
            return extract_entities

        async def route_query(query, tenant_id):
            raise AssertionError(
                f"fixture router must not be invoked directly: {query} {tenant_id}"
            )

        async def build_router(**kwargs):
            assert kwargs == {
                "config_manager": config_manager,
                "telemetry_manager": fake_telemetry_manager,
                "tenant_id": "acme:science",
            }
            return route_query

        class RecordingSyntheticService:
            def __init__(self, **kwargs):
                assert kwargs["backend"] is backend
                assert kwargs["entity_extractor"] is extract_entities
                self.entity_extractor = kwargs["entity_extractor"]

            async def generate(self, request):
                labelled = await self.entity_extractor(
                    "Marie Curie discovered radium", request.tenant_id
                )
                assert labelled == {
                    "query": "Marie Curie discovered radium",
                    "entities": [
                        {"text": "Marie Curie", "type": "PERSON"},
                        {"text": "radium", "type": "CONCEPT"},
                    ],
                    "relationships": [],
                }
                return SyntheticDataResponse(
                    optimizer=request.optimizer,
                    schema_name="RoutingExperienceSchema",
                    count=0,
                    selected_profiles=["video_fixture"],
                    profile_selection_reasoning="No source rows in unit fixture",
                    data=[],
                    metadata={},
                )

        with (
            patch(_PATCH_CONFIG, return_value=config_manager),
            _patch_telemetry(fake_telemetry_manager),
            patch(
                "cogniverse_foundation.config.utils.get_config",
                return_value=tenant_config,
            ),
            patch(
                "cogniverse_foundation.config.llm_factory.create_dspy_lm",
                return_value=object(),
            ),
            patch(
                "cogniverse_runtime.optimization_cli._build_cli_entity_extractor",
                side_effect=build_extractor,
            ),
            patch(
                "cogniverse_runtime.optimization_cli._build_cli_routing_decider",
                side_effect=build_router,
            ),
            patch(
                "cogniverse_core.registries.backend_registry."
                "BackendRegistry.get_search_backend",
                return_value=backend,
            ),
            patch(
                "cogniverse_synthetic.service.SyntheticDataService",
                RecordingSyntheticService,
            ),
        ):
            result = await run_synthetic_generation(
                tenant_id="acme:science",
                optimizer_types=["routing"],
                count=1,
            )

        assert result == {
            "status": "no_data",
            "results": {"routing": {"status": "no_data", "examples_generated": 0}},
        }
        assert build_calls == [
            {
                "config_manager": config_manager,
                "telemetry_manager": fake_telemetry_manager,
                "tenant_id": "acme:science",
            }
        ]
        assert extraction_calls == [("Marie Curie discovered radium", "acme:science")]

    @pytest.mark.asyncio
    async def test_entity_agent_failure_does_not_block_independent_optimizer(
        self,
        fake_telemetry_manager,
    ):
        from cogniverse_runtime.optimization_cli import run_synthetic_generation
        from cogniverse_synthetic.schemas import SyntheticDataResponse

        sections = _synthetic_runtime_sections(marker="independent")
        tenant_config = SimpleNamespace(
            get_llm_config=lambda: SimpleNamespace(primary="test-lm"),
            get=lambda key, default=None: sections.get(key, default),
        )
        config_manager = object()
        backend = object()
        generated = []

        class RecordingSyntheticService:
            def __init__(self, **kwargs):
                assert kwargs["backend"] is backend
                assert kwargs["entity_extractor"] is None

            async def generate(self, request):
                generated.append(request.optimizer)
                return SyntheticDataResponse(
                    optimizer=request.optimizer,
                    schema_name="ProfileSelectionExampleSchema",
                    count=0,
                    selected_profiles=["video_fixture"],
                    profile_selection_reasoning="No source rows in unit fixture",
                    data=[],
                    metadata={},
                )

        with (
            patch(_PATCH_CONFIG, return_value=config_manager),
            _patch_telemetry(fake_telemetry_manager),
            patch(
                "cogniverse_foundation.config.utils.get_config",
                return_value=tenant_config,
            ),
            patch(
                "cogniverse_foundation.config.llm_factory.create_dspy_lm",
                return_value=object(),
            ),
            patch(
                "cogniverse_runtime.optimization_cli._build_cli_entity_extractor",
                side_effect=RuntimeError("GLiNER health check failed"),
            ),
            patch(
                "cogniverse_core.registries.backend_registry."
                "BackendRegistry.get_search_backend",
                return_value=backend,
            ),
            patch(
                "cogniverse_synthetic.service.SyntheticDataService",
                RecordingSyntheticService,
            ),
        ):
            result = await run_synthetic_generation(
                tenant_id="acme:science",
                optimizer_types=["entity_extraction", "profile"],
                count=1,
            )

        assert result == {
            "status": "failed",
            "results": {
                "entity_extraction": {
                    "status": "failed",
                    "error": "GLiNER health check failed",
                },
                "profile": {"status": "no_data", "examples_generated": 0},
            },
        }
        assert generated == ["profile"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        (
            "optimizer_type",
            "expected_agent_type",
            "schema_name",
            "example",
            "expected_confidence",
        ),
        [
            (
                "query_enhancement",
                "query_enhancement",
                "QueryEnhancementExampleSchema",
                {
                    "query": "Find exact PyTorch tutorials",
                    "enhanced_query": "Find exact PyTorch framework tutorials",
                    "expansion_terms": ["framework"],
                    "synonyms": ["machine learning library"],
                    "context": "PyTorch education",
                    "reasoning": (
                        "Added the framework category to focus tutorial retrieval"
                    ),
                },
                0.0,
            ),
            (
                "entity_extraction",
                "entity_extraction",
                "EntityExtractionExampleSchema",
                {
                    "query": "Marie Curie discovered radium",
                    "entities": [
                        {"text": "Marie Curie", "type": "PERSON"},
                        {"text": "radium", "type": "CONCEPT"},
                    ],
                    "relationships": [],
                },
                0.0,
            ),
            (
                "profile",
                "profile_selection",
                "ProfileSelectionExampleSchema",
                {
                    "query": "find transformer lectures",
                    "available_profiles": "video_fixture,document_fixture",
                    "selected_profile": "video_fixture",
                    "reasoning": "Video retrieval matches the requested lectures.",
                    "query_intent": "video_search",
                    "modality": "video",
                    "complexity": "medium",
                },
                0.0,
            ),
        ],
    )
    async def test_generated_examples_enter_the_human_approval_queue(
        self,
        fake_telemetry_manager,
        optimizer_type,
        expected_agent_type,
        schema_name,
        example,
        expected_confidence,
    ):
        from cogniverse_core.approval.interfaces import ApprovalStatus
        from cogniverse_foundation.config.unified_config import (
            BackendConfig,
            SyntheticGeneratorConfig,
        )
        from cogniverse_runtime.optimization_cli import run_synthetic_generation
        from cogniverse_synthetic.schemas import SyntheticDataResponse

        sections = _synthetic_runtime_sections(marker="approval")
        tenant_config = SimpleNamespace(
            get_llm_config=lambda: SimpleNamespace(primary="test-lm"),
            get=lambda key, default=None: sections.get(key, default),
        )
        system_config = SimpleNamespace(
            telemetry_url="http://phoenix.test:6006",
            telemetry_collector_endpoint="phoenix.test:4317",
            redis_url="redis://redis.test:6379/0",
        )
        config_manager = SimpleNamespace(get_system_config=lambda: system_config)
        saved_batches = []
        storage_inits = []
        service_inits = []
        backend = object()

        async def entity_extractor(source_text, tenant_id):
            raise AssertionError(
                f"fixture extractor must not be invoked directly: {source_text} {tenant_id}"
            )

        async def build_entity_extractor(**kwargs):
            assert kwargs == {
                "config_manager": config_manager,
                "telemetry_manager": fake_telemetry_manager,
                "tenant_id": "acme:production",
            }
            return entity_extractor

        async def profile_labeler(query, profiles, tenant_id):
            raise AssertionError(
                f"fixture labeler must not be invoked directly: "
                f"{query} {profiles} {tenant_id}"
            )

        async def build_profile_labeler(**kwargs):
            assert kwargs == {
                "config_manager": config_manager,
                "telemetry_manager": fake_telemetry_manager,
                "tenant_id": "acme:production",
            }
            return profile_labeler

        async def query_enhancer(query, tenant_id, source_text):
            raise AssertionError(
                "fixture enhancer must not be invoked directly: "
                f"{query} {tenant_id} {source_text}"
            )

        async def build_query_enhancer(**kwargs):
            assert kwargs == {
                "config_manager": config_manager,
                "telemetry_manager": fake_telemetry_manager,
                "tenant_id": "acme:production",
            }
            return query_enhancer

        class RecordingStorage:
            def __init__(self, **kwargs):
                storage_inits.append(kwargs)

            async def save_batch(self, batch):
                saved_batches.append(batch)
                return batch.batch_id

        class RecordingSyntheticService:
            def __init__(self, **kwargs):
                service_inits.append(kwargs)

            async def generate(self, request):
                assert request.tenant_id == "acme:production"
                assert request.optimizer == optimizer_type
                assert request.count == 1
                return SyntheticDataResponse(
                    optimizer=request.optimizer,
                    schema_name=schema_name,
                    count=1,
                    selected_profiles=[],
                    profile_selection_reasoning="Direct unit fixture",
                    data=[example],
                    metadata={},
                )

        with (
            patch(_PATCH_CONFIG, return_value=config_manager),
            _patch_telemetry(fake_telemetry_manager),
            patch(
                "cogniverse_foundation.config.utils.get_config",
                return_value=tenant_config,
            ),
            patch(
                "cogniverse_foundation.config.llm_factory.create_dspy_lm",
                return_value=object(),
            ),
            patch(
                "cogniverse_runtime.optimization_cli._build_cli_entity_extractor",
                side_effect=build_entity_extractor,
            ),
            patch(
                "cogniverse_runtime.optimization_cli._build_cli_profile_labeler",
                side_effect=build_profile_labeler,
            ),
            patch(
                "cogniverse_runtime.optimization_cli._build_cli_query_enhancer",
                side_effect=build_query_enhancer,
            ),
            patch(
                "cogniverse_core.registries.backend_registry."
                "BackendRegistry.get_search_backend",
                return_value=backend,
            ),
            patch(
                "cogniverse_synthetic.service.SyntheticDataService",
                RecordingSyntheticService,
            ),
            patch(
                "cogniverse_agents.approval.approval_storage.ApprovalStorageImpl",
                RecordingStorage,
            ),
        ):
            result = await run_synthetic_generation(
                tenant_id="acme:production",
                optimizer_types=[optimizer_type],
                count=1,
            )

        assert result == {
            "status": "success",
            "results": {
                optimizer_type: {
                    "status": "success",
                    "examples_generated": 1,
                    "batch_id": saved_batches[0].batch_id,
                    "pending_review": 1,
                }
            },
        }
        assert storage_inits == [
            {
                "grpc_endpoint": "http://phoenix.test:4317",
                "http_endpoint": "http://phoenix.test:6006",
                "tenant_id": "acme:production",
                "telemetry_manager": fake_telemetry_manager,
                "redis_url": "redis://redis.test:6379/0",
            }
        ]
        assert len(service_inits) == 1
        service_init = service_inits[0]
        assert service_init["backend"] is backend
        assert isinstance(service_init["backend_config"], BackendConfig)
        assert (
            service_init["backend_config"].tenant_id,
            service_init["backend_config"].url,
            service_init["backend_config"].metadata,
            service_init["backend_config"].profiles["video_fixture"].schema_name,
        ) == (
            "acme:production",
            "http://vespa-approval.test",
            {"marker": "approval"},
            "video_approval",
        )
        assert isinstance(service_init["generator_config"], SyntheticGeneratorConfig)
        assert service_init["generator_config"].tenant_id == "acme:production"
        assert set(service_init["generator_config"].optimizer_configs) == set(
            sections["synthetic"]["optimizer_configs"]
        )
        assert (
            service_init["generator_config"]
            .optimizer_configs[optimizer_type]
            .optimizer_type
            == optimizer_type
        )
        assert service_init["agents_config"] == {
            "search_agent": sections["agents"]["search_agent"]
        }
        assert service_init["entity_extractor"] is (
            entity_extractor if optimizer_type == "entity_extraction" else None
        )
        batch = saved_batches[0]
        assert batch.batch_id.startswith(f"synthetic_{optimizer_type}_")
        assert batch.context == {
            "tenant_id": "acme:production",
            "agent_type": expected_agent_type,
            "optimizer": optimizer_type,
            "purpose": "optimizer_training",
        }
        assert [
            (
                item.item_id,
                item.data,
                item.confidence,
                item.status,
                item.metadata,
            )
            for item in batch.items
        ] == [
            (
                f"{batch.batch_id}_0",
                example,
                expected_confidence,
                ApprovalStatus.PENDING_REVIEW,
                {
                    "agent_type": expected_agent_type,
                    "optimizer_type": optimizer_type,
                    "synthetic": True,
                },
            )
        ]
        assert fake_telemetry_manager._provider.datasets.created == []

    @pytest.mark.asyncio
    async def test_synthetic_generation_does_not_reconfigure_global_dspy(
        self,
        fake_telemetry_manager,
        monkeypatch,
    ):
        """Synthetic generation keeps its LM binding local to the async task."""
        import dspy

        from cogniverse_runtime.optimization_cli import run_synthetic_generation

        def reject_global_configuration(self, **kwargs):
            raise AssertionError(f"global DSPy configuration attempted: {kwargs}")

        monkeypatch.setattr(
            type(dspy.settings),
            "configure",
            reject_global_configuration,
        )

        sections = _synthetic_runtime_sections()
        tenant_config = SimpleNamespace(
            get_llm_config=lambda: SimpleNamespace(primary="test-lm"),
            get=lambda key, default=None: sections.get(key, default),
        )
        p1, p2 = _patch_infra(fake_telemetry_manager)

        async def profile_labeler(query, profiles, tenant_id):
            raise AssertionError(
                f"fixture labeler must not be invoked: {query} {profiles} {tenant_id}"
            )

        with (
            p1,
            p2,
            patch(
                "cogniverse_foundation.config.utils.get_config",
                return_value=tenant_config,
            ),
            patch(
                "cogniverse_foundation.config.llm_factory.create_dspy_lm",
                return_value=object(),
            ),
            patch(
                "cogniverse_runtime.optimization_cli._build_cli_profile_labeler",
                return_value=profile_labeler,
            ),
            patch(
                "cogniverse_core.registries.backend_registry."
                "BackendRegistry.get_search_backend",
                side_effect=RuntimeError("synthetic backend unavailable"),
            ),
        ):
            result = await run_synthetic_generation(
                tenant_id="test:unit",
                optimizer_types=["profile"],
                count=5,
            )

        assert result == {
            "status": "failed",
            "results": {
                "profile": {
                    "status": "failed",
                    "error": (
                        "Synthetic backend access failed for tenant='test:unit' "
                        "backend='vespa': synthetic backend unavailable"
                    ),
                }
            },
        }

    @pytest.mark.asyncio
    async def test_synthetic_backend_outage_returns_exact_failure(
        self,
        fake_telemetry_manager,
    ):
        """Synthetic generation reports the backend failure without masking it."""
        from cogniverse_runtime.optimization_cli import run_synthetic_generation

        sections = _synthetic_runtime_sections()
        tenant_config = SimpleNamespace(
            get_llm_config=lambda: SimpleNamespace(primary="test-lm"),
            get=lambda key, default=None: sections.get(key, default),
        )
        p1, p2 = _patch_infra(fake_telemetry_manager)

        async def profile_labeler(query, profiles, tenant_id):
            raise AssertionError(
                f"fixture labeler must not be invoked: {query} {profiles} {tenant_id}"
            )

        with (
            p1,
            p2,
            patch(
                "cogniverse_foundation.config.utils.get_config",
                return_value=tenant_config,
            ),
            patch(
                "cogniverse_foundation.config.llm_factory.create_dspy_lm",
                return_value=object(),
            ),
            patch(
                "cogniverse_runtime.optimization_cli._build_cli_profile_labeler",
                return_value=profile_labeler,
            ),
            patch(
                "cogniverse_core.registries.backend_registry."
                "BackendRegistry.get_search_backend",
                side_effect=RuntimeError("synthetic backend unavailable"),
            ),
        ):
            result = await run_synthetic_generation(
                tenant_id="test:unit",
                optimizer_types=["profile"],
                count=5,
            )

        assert result == {
            "status": "failed",
            "results": {
                "profile": {
                    "status": "failed",
                    "error": (
                        "Synthetic backend access failed for tenant='test:unit' "
                        "backend='vespa': synthetic backend unavailable"
                    ),
                }
            },
        }

    @pytest.mark.asyncio
    async def test_concurrent_synthetic_failures_keep_context_and_result_names(
        self,
        fake_telemetry_manager,
    ):
        import dspy

        from cogniverse_runtime.optimization_cli import run_synthetic_generation

        class TenantConfig:
            def __init__(self, tenant_id):
                self._sections = _synthetic_runtime_sections(
                    marker=tenant_id.replace(":", "-")
                )
                self._llm_config = type(
                    "LLMConfig",
                    (),
                    {"primary": f"endpoint-{tenant_id}"},
                )()

            def get_llm_config(self):
                return self._llm_config

            def get(self, key, default=None):
                return self._sections.get(key, default)

        tenants = ("org:one", "org:two")
        optimizer_by_tenant = {
            "org:one": "profile",
            "org:two": "query_enhancement",
        }
        configs = {tenant: TenantConfig(tenant) for tenant in tenants}
        tenant_lms = {f"endpoint-{tenant}": object() for tenant in tenants}
        observed = {}
        observed_configs = {}
        both_entered = asyncio.Event()

        class RecordingSyntheticService:
            def __init__(self, **kwargs):
                tenant = kwargs["backend_config"].tenant_id
                observed_configs[tenant] = (
                    kwargs["backend_config"].metadata,
                    kwargs["generator_config"].field_mappings.to_dict(),
                    kwargs["agents_config"]["search_agent"]["url"],
                )

            async def generate(self, request):
                lm_before_await = dspy.settings.lm
                observed[request.tenant_id] = [lm_before_await]
                if len(observed) == len(tenants):
                    both_entered.set()
                await asyncio.wait_for(both_entered.wait(), timeout=1)
                observed[request.tenant_id].append(dspy.settings.lm)
                raise RuntimeError(f"{request.tenant_id} {request.optimizer} failed")

        with (
            patch(_PATCH_CONFIG),
            _patch_telemetry(fake_telemetry_manager),
            patch(
                "cogniverse_foundation.config.utils.get_config",
                side_effect=lambda tenant_id, **kwargs: configs[tenant_id],
            ),
            patch(
                "cogniverse_foundation.config.llm_factory.create_dspy_lm",
                side_effect=lambda endpoint: tenant_lms[endpoint],
            ),
            patch(
                "cogniverse_runtime.optimization_cli._build_cli_profile_labeler",
                return_value=lambda *args: None,
            ),
            patch(
                "cogniverse_runtime.optimization_cli._build_cli_query_enhancer",
                return_value=lambda *args: None,
            ),
            patch(
                "cogniverse_core.registries.backend_registry."
                "BackendRegistry.get_search_backend",
                return_value=object(),
            ),
            patch(
                "cogniverse_synthetic.service.SyntheticDataService",
                RecordingSyntheticService,
            ),
        ):
            results = await asyncio.gather(
                *(
                    run_synthetic_generation(
                        tenant_id=tenant,
                        optimizer_types=[optimizer_by_tenant[tenant]],
                        count=1,
                    )
                    for tenant in tenants
                )
            )

        assert results == [
            {
                "status": "failed",
                "results": {
                    "profile": {
                        "status": "failed",
                        "error": "org:one profile failed",
                    }
                },
            },
            {
                "status": "failed",
                "results": {
                    "query_enhancement": {
                        "status": "failed",
                        "error": "org:two query_enhancement failed",
                    }
                },
            },
        ]
        assert observed == {
            tenant: [tenant_lms[f"endpoint-{tenant}"]] * 2 for tenant in tenants
        }
        assert observed_configs == {
            "org:one": (
                {"marker": "org-one"},
                configs["org:one"]._sections["synthetic"]["field_mappings"],
                "http://search-org-one.test",
            ),
            "org:two": (
                {"marker": "org-two"},
                configs["org:two"]._sections["synthetic"]["field_mappings"],
                "http://search-org-two.test",
            ),
        }


class TestOptimizeAgentPersistence:
    """_optimize_agent must construct ArtifactManager(provider, tenant_id) and
    persist the compiled module via save_blob(kind="model", ...). The prior code
    called ArtifactManager(telemetry_provider=...) (missing the required
    tenant_id) and a non-existent store_artifact() — so every triggered
    optimization failed. The fake ArtifactManager below enforces the real
    interface, so the old code would raise (TypeError / AttributeError) here."""

    @pytest.mark.asyncio
    async def test_optimize_agent_persists_compiled_module(self):
        from unittest.mock import MagicMock

        from cogniverse_runtime.optimization_cli import _optimize_agent

        captured: Dict[str, Any] = {}

        class _FakeArtifactManager:
            def __init__(self, telemetry_provider, tenant_id):  # both REQUIRED
                captured["tenant_id"] = tenant_id

            async def save_blob(self, kind, key, content):
                captured["kind"] = kind
                captured["key"] = key
                return "artifact-xyz"

        class _FakeOptimizer:
            optimization_settings = {
                "max_bootstrapped_demos": 1,
                "max_labeled_demos": 1,
                "max_rounds": 1,
                "max_errors": 1,
                "teacher_settings": {},
            }

            def initialize_language_model(self, endpoint, teacher_endpoint_config=None):
                self.lm = MagicMock()  # consumed by dspy.context(lm=optimizer.lm)

            def create_query_analysis_signature(self):
                return object()

        class _FakeCompiled:
            def dump_state(self):
                return {"demos": []}

        class _FakeTeleprompter:
            def __init__(self, *a, **k):
                pass

            def compile(self, module, trainset=None):
                return _FakeCompiled()

        high_df = pd.DataFrame([{"query": "find cats", "output": "{}", "score": 0.9}])

        with (
            patch(
                "cogniverse_agents.optimizer.dspy_agent_optimizer.DSPyAgentPromptOptimizer",
                _FakeOptimizer,
            ),
            patch("dspy.ChainOfThought", lambda sig: object()),
            patch("dspy.teleprompt.BootstrapFewShot", _FakeTeleprompter),
            patch(
                "cogniverse_agents.optimizer.artifact_manager.ArtifactManager",
                _FakeArtifactManager,
            ),
        ):
            result = await _optimize_agent(
                "search",
                pd.DataFrame([]),
                high_df,
                "http://lm",
                config_manager=MagicMock(),
                telemetry_provider=MagicMock(),
                tenant_id="acme:prod",
            )

        assert result["status"] == "success"
        assert result["training_examples"] == 1
        assert captured["tenant_id"] == "acme:prod"
        # The compile reaches traffic through the versioned-prompts serving
        # path only; no side blob is written and no artifact id is reported.
        assert "key" not in captured, captured
        assert "artifact_id" not in result, result

    @pytest.mark.asyncio
    async def test_optimize_agent_threads_teacher_into_bootstrap(self):
        """_optimize_agent must hand the teacher endpoint to the real optimizer
        and forward the resulting teacher_settings into BootstrapFewShot —
        DSPy runs the bootstrap teacher inside dspy.context(**teacher_settings)."""
        from unittest.mock import MagicMock

        from cogniverse_foundation.config.unified_config import LLMEndpointConfig
        from cogniverse_runtime.optimization_cli import _optimize_agent

        captured: Dict[str, Any] = {}

        class _FakeArtifactManager:
            def __init__(self, telemetry_provider, tenant_id):
                pass

            async def save_blob(self, kind, key, content):
                return "artifact-teacher"

        class _FakeCompiled:
            def dump_state(self):
                return {"demos": []}

        class _CapturingTeleprompter:
            def __init__(self, *a, **k):
                captured["teleprompter_kwargs"] = k

            def compile(self, module, trainset=None):
                return _FakeCompiled()

        student = LLMEndpointConfig(
            model="hosted_vllm/org/Student", api_base="http://student:8000/v1"
        )
        teacher = LLMEndpointConfig(
            model="hosted_vllm/org/Teacher", api_base="http://teacher:9000/v1"
        )
        high_df = pd.DataFrame([{"query": "find cats", "output": "{}", "score": 0.9}])

        with (
            patch("dspy.teleprompt.BootstrapFewShot", _CapturingTeleprompter),
            patch(
                "cogniverse_agents.optimizer.artifact_manager.ArtifactManager",
                _FakeArtifactManager,
            ),
        ):
            result = await _optimize_agent(
                "search",
                pd.DataFrame([]),
                high_df,
                student,
                config_manager=MagicMock(),
                telemetry_provider=MagicMock(),
                tenant_id="acme:prod",
                teacher_endpoint=teacher,
            )

        assert result["status"] == "success"
        teacher_settings = captured["teleprompter_kwargs"]["teacher_settings"]
        assert teacher_settings["lm"].model == "hosted_vllm/org/Teacher"
        assert teacher_settings["lm"].kwargs["api_base"] == "http://teacher:9000/v1"


class FailingTraceStore:
    """Trace store whose get_all_spans always raises (Phoenix down/slow)."""

    def __init__(self):
        self.calls = 0

    async def get_all_spans(self, **kwargs) -> pd.DataFrame:
        self.calls += 1
        raise TimeoutError("phoenix query timed out")


class TestQuerySpansFailureIsNotNoData:
    """A failed Phoenix query must raise, not return an empty frame.

    Flattening the exception to an empty DataFrame made every batch mode
    report status=no_data during a Phoenix timeout — indistinguishable
    from a genuinely empty optimization window. The retry budget is bounded:
    2 attempts with a 60s per-attempt timeout, so a persistently down or
    hung Phoenix costs at most ~125s per call site (this runs in a per-agent
    loop; the previous 3x120s budget hung a cycle for 370s per agent).
    """

    def test_retry_budget_constants(self):
        from cogniverse_runtime import optimization_cli as cli

        assert cli._SPAN_QUERY_ATTEMPTS == 2
        assert cli._SPAN_QUERY_TIMEOUT_S == 60

    @pytest.mark.asyncio
    async def test_query_failure_raises_after_exactly_two_attempts(self, monkeypatch):
        import asyncio as _asyncio

        from cogniverse_runtime import optimization_cli as cli

        provider = FakeTelemetryProvider()
        store = FailingTraceStore()
        provider._trace_store = store
        manager = FakeTelemetryManager(provider)

        monkeypatch.setattr(_asyncio, "sleep", _instant_sleep)
        with patch(_PATCH_TELEMETRY, return_value=manager):
            with pytest.raises(RuntimeError, match="after 2 attempts"):
                await cli._query_spans_by_name(
                    manager,
                    provider,
                    "acme:prod",
                    "cogniverse.entity_extraction",
                    1.0,
                )
        assert store.calls == 2

    @pytest.mark.asyncio
    async def test_transient_failure_recovers_on_retry(self, monkeypatch):
        import asyncio as _asyncio

        from cogniverse_runtime import optimization_cli as cli

        df = pd.DataFrame([{"name": "cogniverse.entity_extraction", "x": 1}])

        class FlakyStore:
            def __init__(self):
                self.calls = 0

            async def get_all_spans(self, **kwargs):
                self.calls += 1
                assert set(kwargs) == {
                    "project",
                    "start_time",
                    "end_time",
                    "filters",
                }
                if self.calls == 1:
                    raise TimeoutError("first attempt times out")
                return df

        provider = FakeTelemetryProvider()
        store = FlakyStore()
        provider._trace_store = store
        manager = FakeTelemetryManager(provider)

        monkeypatch.setattr(_asyncio, "sleep", _instant_sleep)
        with patch(_PATCH_TELEMETRY, return_value=manager):
            out = await cli._query_spans_by_name(
                manager,
                provider,
                "acme:prod",
                "cogniverse.entity_extraction",
                1.0,
            )
        assert store.calls == 2
        assert len(out) == 1


async def _instant_sleep(_seconds):
    return None


class TestQuerySpansHungPhoenixIsCancelled:
    """A get_all_spans call that hangs forever must be cancelled per attempt.

    A dead Phoenix raises promptly; a hung one never returns — only
    asyncio.wait_for's cancellation bounds the retry budget. The wall-clock
    band proves each attempt was cut at the per-attempt timeout instead of
    hanging the cycle.
    """

    @pytest.mark.asyncio
    async def test_hung_query_cancelled_each_attempt_then_raises(self, monkeypatch):
        import asyncio as _asyncio
        import time

        from cogniverse_runtime import optimization_cli as cli

        class HangingTraceStore:
            def __init__(self):
                self.calls = 0
                self.cancelled = 0

            async def get_all_spans(self, **kwargs):
                self.calls += 1
                try:
                    await _asyncio.Event().wait()
                except _asyncio.CancelledError:
                    self.cancelled += 1
                    raise

        provider = FakeTelemetryProvider()
        store = HangingTraceStore()
        provider._trace_store = store
        manager = FakeTelemetryManager(provider)

        monkeypatch.setattr(cli, "_SPAN_QUERY_TIMEOUT_S", 0.2)
        monkeypatch.setattr(cli, "_SPAN_QUERY_ATTEMPTS", 2)
        monkeypatch.setattr(_asyncio, "sleep", _instant_sleep)

        start = time.monotonic()
        with patch(_PATCH_TELEMETRY, return_value=manager):
            with pytest.raises(RuntimeError, match="after 2 attempts"):
                await cli._query_spans_by_name(
                    manager,
                    provider,
                    "acme:prod",
                    "cogniverse.entity_extraction",
                    1.0,
                )
        elapsed = time.monotonic() - start

        assert store.calls == 2
        assert store.cancelled == 2
        # Two 0.2s attempt timeouts must have elapsed; anything near 2s or
        # beyond means a hung attempt was not cancelled.
        assert 0.35 <= elapsed < 2.0, elapsed


class TestGoldenSetCandidates:
    """Golden-set growth skips rows whose score cannot coerce to float."""

    def test_junk_score_row_skipped_valid_rows_survive(self):
        from cogniverse_runtime.optimization_cli import _golden_set_candidates

        df = pd.DataFrame(
            [
                {"category": "high_scoring", "query": "good one", "score": 0.9},
                {"category": "high_scoring", "query": "junk score", "score": "great"},
                {"category": "high_scoring", "query": "good two", "score": 0.85},
                {"category": "high_scoring", "query": "none score", "score": None},
                {"category": "high_scoring", "query": "below cut", "score": 0.5},
                {"category": "low_scoring", "query": "wrong category", "score": 0.95},
            ]
        )

        candidate = {
            "expected_videos": [],
            "ground_truth": "",
            "query_type": "live_traffic",
            "source": "quality_monitor",
        }
        assert _golden_set_candidates(df) == [
            {"query": "good one", **candidate},
            {"query": "good two", **candidate},
        ]


class TestRunFailed:
    """_run_failed maps a mode result to the failed/ok exit decision."""

    def test_top_level_failed(self):
        from cogniverse_runtime.optimization_cli import _run_failed

        assert _run_failed({"status": "failed", "error": "phoenix down"}) is True

    def test_top_level_error(self):
        from cogniverse_runtime.optimization_cli import _run_failed

        assert _run_failed({"status": "error"}) is True

    @pytest.mark.parametrize("nested_status", ["failed", "error"])
    def test_top_level_success_does_not_mask_nested_failure(self, nested_status):
        from cogniverse_runtime.optimization_cli import _run_failed

        assert (
            _run_failed(
                {
                    "status": "success",
                    "results": {"a": {"status": nested_status}},
                }
            )
            is True
        )

    def test_batch_shape_nested_failure(self):
        from cogniverse_runtime.optimization_cli import _run_failed

        assert (
            _run_failed(
                {
                    "search": {"status": "failed", "error": "lm down"},
                    "summary": {"status": "success"},
                }
            )
            is True
        )

    def test_batch_shape_skips_and_nonfatal_eval_error_ok(self):
        from cogniverse_runtime.optimization_cli import _run_failed

        assert (
            _run_failed(
                {
                    "search": {"status": "skipped", "reason": "no_data"},
                    "post_optimization_eval": {"error": "eval unavailable"},
                    "baseline_updated": True,
                }
            )
            is False
        )

    def test_no_data_is_ok(self):
        from cogniverse_runtime.optimization_cli import _run_failed

        assert _run_failed({"status": "no_data"}) is False

    def test_non_dict_is_ok(self):
        from cogniverse_runtime.optimization_cli import _run_failed

        assert _run_failed(None) is False

    def test_failed_string_marker_fails(self):
        from cogniverse_runtime.optimization_cli import _run_failed

        assert _run_failed("failed: Vespa connection refused") is True
        assert _run_failed("error: boom") is True

    def test_completed_string_marker_ok(self):
        from cogniverse_runtime.optimization_cli import _run_failed

        assert _run_failed("completed: {'fact': 3}") is False
        assert _run_failed("skipped: path /logs is not a directory") is False

    def test_failed_key_dict_fails(self):
        from cogniverse_runtime.optimization_cli import _run_failed

        # config_vacuum encodes an outage as {"failed": <exc>}, no status key.
        assert _run_failed({"config_vacuum": {"failed": "Vespa refused"}}) is True
        # A zero failed-count is not a failure.
        assert _run_failed({"failed": 0, "succeeded": 5}) is False

    def test_cleanup_total_outage_shape_fails(self):
        """The exact run_cleanup result under a total mem0/Vespa outage: the
        per-tenant memory_cleanup entry is a 'failed: ...' string and
        config_vacuum is {'failed': ...}, neither carrying a top-level status.
        The old .get('status')-only check returned False here → exit 0 =
        SUCCESS while the cron did nothing."""
        from cogniverse_runtime.optimization_cli import _run_failed

        outage_result = {
            "log_retention_days": 7,
            "memory_retention_days": 30,
            "memory_cleanup": {"acme:acme": "failed: Vespa connection refused"},
            "tenants_processed": 1,
            "log_cleanup": {"path": "/logs", "scanned": 0, "deleted": 0, "errors": []},
            "temp_cleanup": {"path": "/tmp", "scanned": 0, "deleted": 0, "errors": []},
            "config_vacuum": {"failed": "Vespa connection refused"},
        }
        assert _run_failed(outage_result) is True

    def test_cleanup_healthy_shape_ok(self):
        """The happy run_cleanup result — completed per-tenant strings, a
        dropped-count vacuum, empty prune errors — must NOT trip the exit."""
        from cogniverse_runtime.optimization_cli import _run_failed

        healthy_result = {
            "log_retention_days": 7,
            "memory_retention_days": 30,
            "memory_cleanup": {"acme:acme": "completed: {'fact': 3}"},
            "tenants_processed": 1,
            "log_cleanup": {"path": "/logs", "scanned": 5, "deleted": 2, "errors": []},
            "temp_cleanup": {"path": "/tmp", "scanned": 0, "deleted": 0, "errors": []},
            "config_vacuum": {"dropped": 4, "keep_versions": 10},
        }
        assert _run_failed(healthy_result) is False


class TestMainExitCode:
    """The exit code is the only success signal Argo sees for a workflow
    step — a failed run must exit non-zero, not print-and-exit-0."""

    def _run_main(self, monkeypatch, mode_result) -> int:
        import sys

        from cogniverse_runtime import optimization_cli as cli

        async def fake_run(*args, **kwargs):
            return mode_result

        monkeypatch.setattr(cli, "run_triggered_optimization", fake_run)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "optimization_cli",
                "--mode",
                "triggered",
                "--tenant-id",
                "t1",
                "--agents",
                "search",
                "--trigger-dataset",
                "trigger-ds",
            ],
        )
        with pytest.raises(SystemExit) as exc:
            cli.main()
        return exc.value.code

    def test_failed_result_exits_nonzero(self, monkeypatch):
        code = self._run_main(
            monkeypatch, {"status": "failed", "error": "phoenix down"}
        )
        assert code == 1

    def test_success_result_exits_zero(self, monkeypatch):
        code = self._run_main(
            monkeypatch, {"status": "success", "training_examples": 3}
        )
        assert code == 0


@pytest.mark.unit
class TestTeacherEndpointReachability:
    """BootstrapFewShot asks the teacher for every demonstration. When the
    teacher endpoint is declared but nothing serves it, each request fails,
    litellm retries, and the caller swallows it into a fallback -- so the job
    walks the whole trainset collecting no demos and hits its timeout. Refuse
    at job start instead, naming the endpoint.
    """

    def _endpoint(self, model="openai/cyankiwi/Qwen3.6-27B-AWQ-INT4"):
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        return LLMEndpointConfig(model=model, api_base="http://teacher-svc:8000/v1")

    def test_raises_naming_the_endpoint_when_nothing_serves_it(self):
        from cogniverse_runtime.optimization_cli import teacher_lm_or_raise

        cfg = SimpleNamespace(resolve_teacher=self._endpoint)
        with pytest.raises(RuntimeError) as exc:
            teacher_lm_or_raise(cfg, probe=lambda url: None)
        message = str(exc.value)
        assert "http://teacher-svc:8000/v1" in message
        assert "unreachable" in message

    def test_probes_the_service_root_and_returns_the_lm_when_served(self):
        from cogniverse_runtime.optimization_cli import teacher_lm_or_raise

        cfg = SimpleNamespace(resolve_teacher=self._endpoint)
        probed = []

        def probe(url):
            probed.append(url)
            return "cyankiwi/Qwen3.6-27B-AWQ-INT4"

        with patch(
            "cogniverse_foundation.config.llm_factory.create_dspy_lm",
            return_value="TEACHER_LM",
        ):
            built = teacher_lm_or_raise(cfg, probe=probe)
        assert built == "TEACHER_LM"
        assert probed == ["http://teacher-svc:8000"]

    def test_raises_when_the_service_serves_a_different_model(self):
        from cogniverse_runtime.optimization_cli import teacher_lm_or_raise

        cfg = SimpleNamespace(resolve_teacher=self._endpoint)
        with pytest.raises(RuntimeError) as exc:
            teacher_lm_or_raise(cfg, probe=lambda url: "google/gemma-4-26b-a4b-it")
        message = str(exc.value)
        assert "cyankiwi/Qwen3.6-27B-AWQ-INT4" in message
        assert "google/gemma-4-26b-a4b-it" in message
