"""The runtime lifespan must configure the synthetic-data service with a real
search backend.

``/synthetic/generate`` serves the process-global service configured at
startup. The service requires both a live backend and a non-empty configured
profile map so requests can resolve tenant schemas and sample the tenant's
Vespa corpus.

This boots the real ``main.py`` lifespan and asserts the global service holds
the canonical configured Vespa profile.
"""

from __future__ import annotations

import asyncio
import copy
import json
import threading
from pathlib import Path

import pytest
from fastapi import FastAPI

pytestmark = pytest.mark.integration


def _configured_synthetic_section() -> dict:
    root = Path(__file__).resolve().parents[3]
    return json.loads((root / "configs/config.json").read_text())["synthetic"]


def _configured_runtime_sections() -> dict:
    root = Path(__file__).resolve().parents[3]
    return json.loads((root / "configs/config.json").read_text())


def test_deployable_synthetic_config_has_exact_routing_contract():
    section = _configured_synthetic_section()

    assert set(section) == {
        "field_mappings",
        "synthetic_generation_timeout_seconds",
        "synthetic_generation_floor_count",
        "optimizer_configs",
    }
    assert "tenant_id" not in section
    assert section["synthetic_generation_timeout_seconds"] == 300.0
    assert section["synthetic_generation_floor_count"] == 1
    assert section["optimizer_configs"]["routing"]["dspy_modules"] == {
        "query_generator": {
            "signature_class": (
                "cogniverse_synthetic.dspy_signatures.GenerateEntityQuery"
            ),
            "module_type": "ChainOfThought",
            "lm_config": {},
            "metadata": {},
        }
    }
    assert {
        mapping["modality"]: mapping["agent_name"]
        for mapping in section["optimizer_configs"]["modality"]["agent_mappings"]
    } == {
        "VIDEO": "search_agent",
        "DOCUMENT": "document_agent",
        "IMAGE": "image_search_agent",
        "AUDIO": "audio_analysis_agent",
        "CODE": "coding_agent",
        "WIKI": "document_agent",
    }


def test_runtime_synthetic_config_rejects_missing_and_unknown_agent():
    from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config

    with pytest.raises(
        ValueError,
        match=(
            "Synthetic runtime configuration for tenant='acme:alpha' "
            "requires object section 'backend'"
        ),
    ):
        parse_synthetic_runtime_config({}, tenant_id="acme:alpha")

    sections = _configured_runtime_sections()
    sections["synthetic"]["optimizer_configs"]["modality"]["agent_mappings"][0][
        "agent_name"
    ] = "unloaded_agent"
    with pytest.raises(
        ValueError,
        match=(
            "Invalid synthetic runtime configuration for tenant='acme:alpha': "
            "mapping for modality 'VIDEO' targets unknown agent 'unloaded_agent'"
        ),
    ):
        parse_synthetic_runtime_config(sections, tenant_id="acme:alpha")


@pytest.mark.asyncio
async def test_runtime_synthetic_config_is_isolated_across_concurrent_tenants():
    from cogniverse_runtime.synthetic_config import parse_synthetic_runtime_config

    sections = _configured_runtime_sections()
    barrier = threading.Barrier(2)

    def resolve(tenant_id: str, marker: str):
        tenant_sections = copy.deepcopy(sections)
        tenant_sections["synthetic"]["optimizer_configs"]["profile"][
            "profile_scoring_rules"
        ][0]["reason"] = marker
        barrier.wait()
        return parse_synthetic_runtime_config(
            tenant_sections,
            tenant_id=tenant_id,
        )

    alpha, beta = await asyncio.gather(
        asyncio.to_thread(resolve, "acme:alpha", "alpha-only"),
        asyncio.to_thread(resolve, "acme:beta", "beta-only"),
    )

    assert (
        alpha.generator_config.tenant_id,
        alpha.generator_config.optimizer_configs["profile"]
        .profile_scoring_rules[0]
        .reason,
    ) == (
        "acme:alpha",
        "alpha-only",
    )
    assert (
        beta.generator_config.tenant_id,
        beta.generator_config.optimizer_configs["profile"]
        .profile_scoring_rules[0]
        .reason,
    ) == (
        "acme:beta",
        "beta-only",
    )


class TestLifespanWiresSyntheticBackend:
    @pytest.mark.asyncio
    async def test_synthetic_service_configured_with_backend(self, monkeypatch):
        # Keep the boot light: skip the sandbox connect and the memory
        # lifecycle scheduler; neither is needed for the synthetic wiring.
        monkeypatch.setenv("COGNIVERSE_SANDBOX_POLICY", "disabled")
        monkeypatch.setenv("COGNIVERSE_MEMORY_LIFECYCLE_DISABLED", "1")

        from cogniverse_synthetic import api as synthetic_api
        from cogniverse_vespa.backend import VespaBackend

        app = FastAPI()
        from cogniverse_runtime.main import lifespan

        async with lifespan(app):
            service = synthetic_api._service
            assert service is not None, "lifespan did not configure the service"
            assert type(service.backend) is VespaBackend
            assert service.backend_config.backend_type == "vespa"
            profile = service.backend_config.profiles["video_colpali_smol500_mv_frame"]
            assert profile.type == "video"
            assert profile.schema_name == "video_colpali_smol500_mv_frame"
            assert profile.embedding_type == "multi_vector"
            assert profile.pipeline_config["extract_keyframes"] is True
            modality_config = service.generator_config.optimizer_configs["modality"]
            assert {
                mapping.modality: mapping.agent_name
                for mapping in modality_config.agent_mappings
            } == {
                "VIDEO": "search_agent",
                "DOCUMENT": "document_agent",
                "IMAGE": "image_search_agent",
                "AUDIO": "audio_analysis_agent",
                "CODE": "coding_agent",
                "WIKI": "document_agent",
            }
            assert {
                modality: service.agent_inferrer.infer_from_modality(modality)
                for modality in (
                    "VIDEO",
                    "DOCUMENT",
                    "IMAGE",
                    "AUDIO",
                    "CODE",
                    "WIKI",
                )
            } == {
                "VIDEO": "search_agent",
                "DOCUMENT": "document_agent",
                "IMAGE": "image_search_agent",
                "AUDIO": "audio_analysis_agent",
                "CODE": "coding_agent",
                "WIKI": "document_agent",
            }
            extraction = await service.entity_extractor(
                "Marie Curie discovered radium",
                "acme:science",
            )
            assert extraction["query"] == "Marie Curie discovered radium"
            assert [
                {"text": entity["text"], "type": entity["type"]}
                for entity in extraction["entities"]
            ] == [
                {"text": "Marie Curie", "type": "PERSON"},
                {"text": "discovered", "type": "EVENT"},
                {"text": "radium", "type": "CONCEPT"},
            ]

    @pytest.fixture
    def _dspy_ambient_state(self):
        """Snapshot dspy's ambient-config globals and restore them on exit.

        The test below claims the ambient ownership slot and binds a dead LM;
        without a restore, both would leak into every later test in the
        session (the exact leak class the production fix addresses).
        """
        import importlib

        # ``dspy.dsp.utils`` re-exports ``settings`` as the Settings
        # singleton; the ownership globals live on the module itself.
        dspy_settings = importlib.import_module("dspy.dsp.utils.settings")

        saved_config = dict(dspy_settings.main_thread_config)
        saved_thread = dspy_settings.config_owner_thread_id
        saved_task = dspy_settings.config_owner_async_task
        # Release any ownership an earlier test's lifespan claimed so the
        # claim task below becomes the first (and thus owning) configurer.
        dspy_settings.config_owner_async_task = None
        yield
        dspy_settings.main_thread_config.clear()
        dspy_settings.main_thread_config.update(saved_config)
        dspy_settings.config_owner_thread_id = saved_thread
        dspy_settings.config_owner_async_task = saved_task

    @pytest.mark.asyncio
    async def test_boot_completes_when_another_task_owns_dspy_ambient(
        self, monkeypatch, _dspy_ambient_state
    ):
        """The lifespan must boot when dspy's ambient slot is already claimed.

        ``dspy.configure`` grants ambient-binding ownership to the first async
        task that calls it. In a process that runs several event-loop tasks
        (multiple test lifespans, a worker job before the API), the lifespan's
        own configure call is not the first — it must fall back to
        first-writer-wins instead of aborting the boot, leaving the already
        bound ambient LM in place and still wiring the synthetic service.
        """
        import asyncio

        import dspy

        monkeypatch.setenv("COGNIVERSE_SANDBOX_POLICY", "disabled")
        monkeypatch.setenv("COGNIVERSE_MEMORY_LIFECYCLE_DISABLED", "1")

        import cogniverse_runtime.main as runtime_main
        from cogniverse_synthetic import api as synthetic_api

        # Claim dspy's async-task ownership from a sibling task, the state a
        # prior lifespan or worker job leaves behind in this process.
        claimed_lm = dspy.LM(
            "openai/ambient-owner", api_base="http://127.0.0.1:29071/v1"
        )

        async def claim_ambient() -> None:
            dspy.configure(lm=claimed_lm)

        await asyncio.create_task(claim_ambient())
        monkeypatch.setattr(runtime_main, "_DSPY_AMBIENT_CONFIGURED", False)

        app = FastAPI()
        async with runtime_main.lifespan(app):
            assert dspy.settings.lm is claimed_lm, (
                "the ambient LM bound by the owning task must survive the boot"
            )
            service = synthetic_api._service
            assert service is not None, "lifespan did not configure the service"
            from cogniverse_vespa.backend import VespaBackend

            assert type(service.backend) is VespaBackend
            assert service.backend_config.backend_type == "vespa"
