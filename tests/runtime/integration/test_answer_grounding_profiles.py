"""Answer-agent grounding follows the tenant's servable profiles.

The dispatcher grounded every summary / detailed report in the system-level
``active_video_profile``, so a tenant that had ingested only documents was
grounded in a video profile it never deployed: zero hits, and an answer LLM
reporting that no content was provided. These tests drive the real dispatcher
against a real ConfigManager over a test-owned Vespa config store and pin which
profiles each tenant's grounding searches, what the envelope reports when there
is nothing to search, and that a backend outage stays distinguishable from it.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import pytest

from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
    BackendProfileConfig,
    SystemConfig,
)
from cogniverse_runtime.agent_dispatcher import (
    GROUNDING_NO_PROFILE_FOR_MODALITY,
    GROUNDING_SEARCH_UNAVAILABLE,
    GROUNDING_SEARCHED,
    AgentDispatcher,
)
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.vespa_docker import VespaDockerManager

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHIPPED_PROFILE_DATA = json.loads(
    (_REPO_ROOT / "configs" / "config.json").read_text()
)["backend"]["profiles"]

DOCUMENT_QUERY = "summarize the documents about robotics"
VIDEO_QUERY = "summarize the videos about robotics"

TENANT_DOCUMENTS = "grounding_documents"
TENANT_VIDEO = "grounding_video"
TENANT_BOTH = "grounding_both"


def _shipped_names_of_type(*types: str) -> list[str]:
    """Shipped profile names of these declared types, in servable order."""
    from cogniverse_agents.profile_selection_agent import _PROFILE_TYPE_ORDER

    return [
        name
        for name, data in sorted(
            _SHIPPED_PROFILE_DATA.items(),
            key=lambda item: (
                _PROFILE_TYPE_ORDER.get(
                    str(item[1].get("type") or "").lower(), len(_PROFILE_TYPE_ORDER)
                ),
                item[0],
            ),
        )
        if str(data.get("type") or "").lower() in types
    ]


DOCUMENT_PROFILES = _shipped_names_of_type("document")
VIDEO_PROFILES = _shipped_names_of_type("video")
DOCUMENT_MODALITY_PROFILES = _shipped_names_of_type("document", "wiki")
ALL_EMBEDDING_SERVICES = {
    service: "http://inference.invalid"
    for data in _SHIPPED_PROFILE_DATA.values()
    for service in [(data.get("inference_services") or {}).get("embedding")]
    if service
}


def _profile(name: str) -> BackendProfileConfig:
    return BackendProfileConfig.from_dict(name, _SHIPPED_PROFILE_DATA[name])


@pytest.fixture(scope="module")
def grounding_vespa():
    """A Vespa container this module owns, so it may be paused safely."""
    from cogniverse_vespa.metadata_schemas import (
        create_config_metadata_schema,
        create_organization_metadata_schema,
        create_tenant_metadata_schema,
    )
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
    from tests.conftest import _shared_vespa_application_package
    from tests.utils.vllm_sidecar import OWNER_LABEL

    manager = VespaDockerManager()
    info = manager.start_container(f"grounding-{uuid.uuid4().hex}")
    try:
        manager.wait_for_config_ready(info)
        owner = subprocess.check_output(
            [
                "docker",
                "inspect",
                "-f",
                '{{ index .Config.Labels "' + OWNER_LABEL + '" }}',
                info["container_name"],
            ],
            text=True,
        ).strip()
        assert owner == str(os.getpid())
        package = _shared_vespa_application_package(
            [
                create_config_metadata_schema(),
                create_organization_metadata_schema(),
                create_tenant_metadata_schema(),
            ]
        )
        VespaSchemaManager(
            backend_endpoint="http://localhost", backend_port=info["config_port"]
        )._deploy_package(package)
        manager.wait_for_application_ready(info)
        yield info
    finally:
        subprocess.run(
            ["docker", "unpause", info["container_name"]], capture_output=True
        )
        manager.stop_container(info)


def _build_config_manager(http_port: int, manager_cls=ConfigManager) -> ConfigManager:
    store = VespaConfigStore(backend_url="http://localhost", backend_port=http_port)
    config_manager = manager_cls(store=store)
    config_manager.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=http_port,
            inference_service_urls=dict(ALL_EMBEDDING_SERVICES),
        )
    )
    return config_manager


def _seed_tenants(config_manager: ConfigManager) -> None:
    for tenant_id, names in (
        (TENANT_DOCUMENTS, DOCUMENT_PROFILES),
        (TENANT_VIDEO, VIDEO_PROFILES),
        (TENANT_BOTH, DOCUMENT_PROFILES + VIDEO_PROFILES),
    ):
        for name in names:
            config_manager.add_backend_profile(_profile(name), tenant_id=tenant_id)


@pytest.fixture(scope="module")
def config_manager(grounding_vespa):
    config_manager = _build_config_manager(grounding_vespa["http_port"])
    _seed_tenants(config_manager)
    return config_manager


def _dispatcher(config_manager: ConfigManager) -> AgentDispatcher:
    return AgentDispatcher(
        agent_registry=AgentRegistry(
            tenant_id=TENANT_DOCUMENTS, config_manager=config_manager
        ),
        config_manager=config_manager,
        schema_loader=FilesystemSchemaLoader(_REPO_ROOT / "configs" / "schemas"),
    )


@pytest.fixture(scope="module")
def dispatcher(config_manager):
    return _dispatcher(config_manager)


@pytest.mark.asyncio
class TestGroundingProfilesComeFromTheTenant:
    async def test_persisted_document_profiles_are_what_a_document_query_grounds_on(
        self, dispatcher
    ):
        modalities, profiles, state = await dispatcher._grounding_plan(
            DOCUMENT_QUERY, TENANT_DOCUMENTS, {}, None
        )

        assert state == GROUNDING_SEARCHED
        assert modalities == ["document"]
        assert profiles == DOCUMENT_PROFILES

    async def test_document_tenant_has_no_profile_for_a_video_query(self, dispatcher):
        modalities, profiles, state = await dispatcher._grounding_plan(
            VIDEO_QUERY, TENANT_DOCUMENTS, {}, None
        )

        assert state == GROUNDING_NO_PROFILE_FOR_MODALITY
        assert modalities == ["video"]
        assert profiles == []

    async def test_video_tenant_grounds_on_its_video_profiles(self, dispatcher):
        modalities, profiles, state = await dispatcher._grounding_plan(
            VIDEO_QUERY, TENANT_VIDEO, {}, None
        )

        assert state == GROUNDING_SEARCHED
        assert modalities == ["video"]
        assert profiles == VIDEO_PROFILES

    async def test_tenant_serving_both_picks_the_queried_modality(self, dispatcher):
        _, document_profiles, document_state = await dispatcher._grounding_plan(
            DOCUMENT_QUERY, TENANT_BOTH, {}, None
        )
        _, video_profiles, video_state = await dispatcher._grounding_plan(
            VIDEO_QUERY, TENANT_BOTH, {}, None
        )

        assert (document_state, document_profiles) == (
            GROUNDING_SEARCHED,
            DOCUMENT_PROFILES,
        )
        assert (video_state, video_profiles) == (GROUNDING_SEARCHED, VIDEO_PROFILES)

    async def test_router_modality_decision_selects_the_profiles(self, dispatcher):
        modalities, profiles, state = await dispatcher._grounding_plan(
            VIDEO_QUERY,
            TENANT_BOTH,
            {},
            {"detected_modalities": ["document"]},
        )

        assert state == GROUNDING_SEARCHED
        assert modalities == ["document"]
        assert profiles == DOCUMENT_PROFILES


@pytest.mark.asyncio
class TestAnswerEnvelopeStatesItsGrounding:
    async def test_nothing_to_search_answers_that_without_calling_the_model(
        self, dispatcher
    ):
        result = await dispatcher._execute_summarization_task(
            VIDEO_QUERY, TENANT_DOCUMENTS
        )

        expected = (
            f"Tenant {TENANT_DOCUMENTS} serves no video content, so there is "
            "nothing to search for this request."
        )
        assert result["status"] == "success"
        assert result["agent"] == "summarizer_agent"
        assert result["message"] == expected
        assert result["result"]["summary"] == expected
        assert result["result"]["key_points"] == []
        assert result["grounding"] == {
            "state": GROUNDING_NO_PROFILE_FOR_MODALITY,
            "modalities": ["video"],
            "profiles": [],
            "result_count": 0,
        }


@pytest.mark.asyncio
class TestPausedBackendKeepsTheOutageDistinguishable:
    async def test_paused_config_store_reports_the_outage_not_an_empty_tenant(
        self, grounding_vespa, config_manager
    ):
        cold = _dispatcher(_build_config_manager(grounding_vespa["http_port"]))
        subprocess.run(
            ["docker", "pause", grounding_vespa["container_name"]], check=True
        )
        try:
            grounding = await cold._resolve_answer_search_results(
                DOCUMENT_QUERY, TENANT_DOCUMENTS, None, top_k=10
            )
        finally:
            subprocess.run(
                ["docker", "unpause", grounding_vespa["container_name"]], check=True
            )

        assert grounding.hits == []
        assert grounding.state == GROUNDING_SEARCH_UNAVAILABLE
        assert grounding.nothing_to_search is False, (
            "an outage must never read as a tenant with nothing to search"
        )

        recovered = await _dispatcher(config_manager)._grounding_plan(
            DOCUMENT_QUERY, TENANT_DOCUMENTS, {}, None
        )
        assert recovered == (["document"], DOCUMENT_PROFILES, GROUNDING_SEARCHED)


@pytest.mark.asyncio
class TestGroundingResolutionLeavesTheLoopFree:
    async def test_twenty_concurrent_resolutions_keep_the_loop_responsive(
        self, grounding_vespa
    ):
        class SlowProfileReads(ConfigManager):
            """The real manager with a backend round-trip's latency on the
            profile read the grounding resolution makes."""

            def list_backend_profiles(self, *args, **kwargs):
                time.sleep(0.03)
                return super().list_backend_profiles(*args, **kwargs)

        slow = _build_config_manager(
            grounding_vespa["http_port"], manager_cls=SlowProfileReads
        )
        for name in DOCUMENT_PROFILES:
            slow.add_backend_profile(_profile(name), tenant_id=TENANT_DOCUMENTS)
        dispatcher = _dispatcher(slow)
        await dispatcher._grounding_plan(DOCUMENT_QUERY, TENANT_DOCUMENTS, {}, None)

        gaps: list[float] = []

        async def ticker():
            previous = time.perf_counter()
            while True:
                await asyncio.sleep(0.005)
                now = time.perf_counter()
                gaps.append(now - previous)
                previous = now

        tick = asyncio.create_task(ticker())
        started = time.perf_counter()
        try:
            plans = await asyncio.gather(
                *(
                    dispatcher._grounding_plan(
                        DOCUMENT_QUERY, TENANT_DOCUMENTS, {}, None
                    )
                    for _ in range(20)
                )
            )
            elapsed = time.perf_counter() - started
        finally:
            tick.cancel()
            await asyncio.gather(tick, return_exceptions=True)

        assert plans == [(["document"], DOCUMENT_PROFILES, GROUNDING_SEARCHED)] * 20
        # Run on the loop these 20 reads would serialize into 20 x 30ms of dead
        # loop and the ticker would not tick at all; offloaded they overlap.
        assert gaps and max(gaps) < 0.05, (
            f"loop ticked {len(gaps)} times, max gap "
            f"{max(gaps, default=float('inf')):.4f}s"
        )
        assert elapsed < 0.3, f"20 concurrent resolutions took {elapsed:.3f}s"
