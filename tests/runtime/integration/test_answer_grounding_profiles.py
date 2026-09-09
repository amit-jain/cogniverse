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
import collections
import contextlib
import dataclasses
import http.server
import json
import os
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

import dspy
import pytest

from cogniverse_agents.search_agent import (
    QUERY_REWRITE_FAILED,
    QUERY_REWRITE_TIMED_OUT,
)
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
    BackendProfileConfig,
    SystemConfig,
)
from cogniverse_runtime.agent_dispatcher import (
    GROUNDING_NO_PROFILE_FOR_MODALITY,
    GROUNDING_SEARCH_RESERVE_S,
    GROUNDING_SEARCH_TIMEOUT_KEY,
    GROUNDING_SEARCH_UNAVAILABLE,
    GROUNDING_SEARCHED,
    GROUNDING_SEARCHED_DEGRADED,
    AgentDispatcher,
    GroundingPlan,
)
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.memory_store import register_deployed_schema
from tests.utils.vespa_docker import VespaDockerManager
from tests.utils.vespa_test_helpers import feed_text_documents

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
    """Each tenant's profiles, with the schema registry rows their deploy writes.

    Servability requires both halves; these tenants pin WHICH profiles a
    request grounds on, so every one of their schemas is registered.
    """
    for tenant_id, names in (
        (TENANT_DOCUMENTS, DOCUMENT_PROFILES),
        (TENANT_VIDEO, VIDEO_PROFILES),
        (TENANT_BOTH, DOCUMENT_PROFILES + VIDEO_PROFILES),
    ):
        for name in names:
            profile = _profile(name)
            config_manager.add_backend_profile(profile, tenant_id=tenant_id)
            register_deployed_schema(config_manager, tenant_id, profile.schema_name)


@pytest.fixture(scope="module")
def config_manager(grounding_vespa):
    config_manager = _build_config_manager(grounding_vespa["http_port"])
    _seed_tenants(config_manager)
    return config_manager


def _dispatcher(
    config_manager: ConfigManager, dispatcher_cls=AgentDispatcher
) -> AgentDispatcher:
    return dispatcher_cls(
        agent_registry=AgentRegistry(
            tenant_id=TENANT_DOCUMENTS, config_manager=config_manager
        ),
        config_manager=config_manager,
        schema_loader=FilesystemSchemaLoader(_REPO_ROOT / "configs" / "schemas"),
    )


@pytest.fixture(scope="module")
def _grounding_lm_instance(ensure_host_ollama):
    """The provisioned test LM, built once for this module."""
    from tests.fixtures.llm import make_dspy_lm

    return make_dspy_lm()


@pytest.fixture(autouse=True)
def grounding_lm(_grounding_lm_instance):
    """The LM the grounding search rewrites its query through.

    A served runtime binds one at startup. Every test here binds the
    provisioned endpoint so the rewrite runs rather than degrading on a
    missing LM; the root conftest clears ``dspy.settings.lm`` after each test,
    so the binding is per test.
    """
    dspy.configure(lm=_grounding_lm_instance)
    return _grounding_lm_instance


@pytest.fixture(scope="module")
def dispatcher(config_manager):
    return _dispatcher(config_manager)


@pytest.mark.asyncio
class TestGroundingProfilesComeFromTheTenant:
    async def test_persisted_document_profiles_are_what_a_document_query_grounds_on(
        self, dispatcher
    ):
        plan = await dispatcher._grounding_plan(
            DOCUMENT_QUERY, TENANT_DOCUMENTS, {}, None
        )

        assert plan.state == GROUNDING_SEARCHED
        assert plan.modalities == ("document",)
        assert plan.profiles == tuple(DOCUMENT_PROFILES)
        assert plan.undeployed_profiles == ()

    async def test_document_tenant_has_no_profile_for_a_video_query(self, dispatcher):
        plan = await dispatcher._grounding_plan(VIDEO_QUERY, TENANT_DOCUMENTS, {}, None)

        assert plan.state == GROUNDING_NO_PROFILE_FOR_MODALITY
        assert plan.modalities == ("video",)
        assert plan.profiles == ()
        assert plan.undeployed_profiles == ()

    async def test_video_tenant_grounds_on_its_video_profiles(self, dispatcher):
        plan = await dispatcher._grounding_plan(VIDEO_QUERY, TENANT_VIDEO, {}, None)

        assert plan.state == GROUNDING_SEARCHED
        assert plan.modalities == ("video",)
        assert plan.profiles == tuple(VIDEO_PROFILES)
        assert plan.undeployed_profiles == ()

    async def test_tenant_serving_both_picks_the_queried_modality(self, dispatcher):
        document_plan = await dispatcher._grounding_plan(
            DOCUMENT_QUERY, TENANT_BOTH, {}, None
        )
        video_plan = await dispatcher._grounding_plan(
            VIDEO_QUERY, TENANT_BOTH, {}, None
        )

        assert document_plan.state == GROUNDING_SEARCHED
        assert document_plan.profiles == tuple(DOCUMENT_PROFILES)
        assert document_plan.undeployed_profiles == ()
        assert video_plan.state == GROUNDING_SEARCHED
        assert video_plan.profiles == tuple(VIDEO_PROFILES)
        assert video_plan.undeployed_profiles == ()

    async def test_router_modality_decision_selects_the_profiles(self, dispatcher):
        plan = await dispatcher._grounding_plan(
            VIDEO_QUERY,
            TENANT_BOTH,
            {},
            {"detected_modalities": ["document"]},
        )

        assert plan.state == GROUNDING_SEARCHED
        assert plan.modalities == ("document",)
        assert plan.profiles == tuple(DOCUMENT_PROFILES)
        assert plan.undeployed_profiles == ()


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
            "degraded_profiles": [],
            "degraded_query_rewrite": None,
            "undeployed_profiles": [],
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
        assert recovered.state == GROUNDING_SEARCHED
        assert recovered.modalities == ("document",)
        assert recovered.profiles == tuple(DOCUMENT_PROFILES)
        assert recovered.undeployed_profiles == ()


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

        assert (
            plans
            == [
                GroundingPlan(
                    ("document",), tuple(DOCUMENT_PROFILES), GROUNDING_SEARCHED
                )
            ]
            * 20
        )
        # Run on the loop these 20 reads would serialize into 20 x 30ms of dead
        # loop and the ticker would not tick at all; offloaded they overlap.
        assert gaps and max(gaps) < 0.05, (
            f"loop ticked {len(gaps)} times, max gap "
            f"{max(gaps, default=float('inf')):.4f}s"
        )
        assert elapsed < 0.3, f"20 concurrent resolutions took {elapsed:.3f}s"


# --- Grounding against a real corpus -----------------------------------------
#
# The tests above pin WHICH profiles a tenant grounds on. These pin what the
# search returns: a second Vespa this module owns, carrying the document
# content schemas as well as the config store, with the tenant's LateOn
# profiles encoding through the served PyLate sidecar.

TENANT_RETRIEVAL = "grounding_retrieval"
TENANT_FANOUT = "grounding_fanout"

RETRIEVAL_PROFILES = [
    name
    for name in DOCUMENT_PROFILES
    if _SHIPPED_PROFILE_DATA[name]["schema_name"] in ("document_text", "lateon_mv")
]
FANOUT_PROFILES = DOCUMENT_PROFILES
UNSERVED_ENCODER_PROFILE = next(
    name
    for name in DOCUMENT_PROFILES
    if (_SHIPPED_PROFILE_DATA[name].get("inference_services") or {}).get("embedding")
    != "colbert_pylate"
)
UNSERVED_ENCODER_SCHEMA = _SHIPPED_PROFILE_DATA[UNSERVED_ENCODER_PROFILE]["schema_name"]
COLBERT_MODEL = _SHIPPED_PROFILE_DATA[RETRIEVAL_PROFILES[0]]["embedding_model"]

# Two documents with no vocabulary in common, so every profile's ranking of
# them agrees and an order pin fails when the query changes rather than
# resolving a near-tie by fusion tie-break.
CORPUS = (
    {
        "id": "grounding_harbour_dredging",
        "title": "Harbour Dredging Survey",
        "text": (
            "The dredging survey recorded silt accumulation across the tidal "
            "basin and recommends removing sediment from the northern berth "
            "before the winter shipping season begins."
        ),
    },
    {
        "id": "grounding_beekeeping_winter",
        "title": "Winter Beekeeping Manual",
        "text": (
            "Overwintering colonies need ventilated hives and candy board "
            "feeding, because the cluster warms itself and opening a hive "
            "during frost chills the brood."
        ),
    },
)
CORPUS_ID = "grounding_corpus"
# The pipeline ids a fed segment "<corpus>_<document id>"; the search returns
# that id, so the pins name what ingestion actually wrote.
HARBOUR_ID, BEEKEEPING_ID = (f"{CORPUS_ID}_{entry['id']}" for entry in CORPUS)
HARBOUR_QUERY = "silt accumulation across the tidal basin"
BEEKEEPING_QUERY = "candy board feeding for overwintering colonies"

SHIPPED_GROUNDING_BUDGET_S = json.loads(
    (_REPO_ROOT / "configs" / "config.json").read_text()
)[GROUNDING_SEARCH_TIMEOUT_KEY]


@pytest.fixture(scope="module")
def retrieval_vespa():
    """A Vespa carrying this module's config store AND document content."""
    from cogniverse_vespa.metadata_schemas import (
        create_config_metadata_schema,
        create_organization_metadata_schema,
        create_tenant_metadata_schema,
    )
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
    from tests.conftest import _shared_vespa_application_package

    manager = VespaDockerManager()
    info = manager.start_container(f"grounding-corpus-{uuid.uuid4().hex}")
    try:
        manager.wait_for_config_ready(info)
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
        manager.stop_container(info)


def _retrieval_config_manager(info, pylate_url: str) -> ConfigManager:
    """Config store on the corpus Vespa, LateOn served, everything else not."""
    config_manager = ConfigManager(
        store=VespaConfigStore(
            backend_url="http://localhost", backend_port=info["http_port"]
        )
    )
    service_urls = dict(ALL_EMBEDDING_SERVICES)
    service_urls["colbert_pylate"] = pylate_url
    config_manager.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=info["http_port"],
            inference_service_urls=service_urls,
        )
    )
    for tenant_id, names in (
        (TENANT_RETRIEVAL, RETRIEVAL_PROFILES),
        (TENANT_FANOUT, FANOUT_PROFILES),
    ):
        for name in names:
            config_manager.add_backend_profile(_profile(name), tenant_id=tenant_id)
    return config_manager


CORPUS_SCHEMAS = ("document_text", "lateon_mv")


def _tenant_backend(config_manager: ConfigManager, tenant_id: str):
    """The ingestion backend the worker builds for a tenant."""
    from cogniverse_runtime.ingestion.processors.embedding_generator.backend_factory import (  # noqa: E501
        BackendFactory,
    )

    return BackendFactory.create(
        "vespa",
        tenant_id,
        {},
        config_manager=config_manager,
        schema_loader=FilesystemSchemaLoader(_REPO_ROOT / "configs" / "schemas"),
    )


@pytest.fixture(scope="module")
def retrieval_config_manager(retrieval_vespa, pylate_server):
    return _retrieval_config_manager(retrieval_vespa, pylate_server)


@pytest.fixture(scope="module")
def seeded_corpus(retrieval_config_manager, pylate_server):
    """The two documents, in every schema this module's tenants search.

    Fed through the tenant's own ingestion backend, so the tenant-scoped
    schemas are deployed by the same path first ingest uses in production and
    the search reads what ingestion wrote. Document ids are the corpus's own,
    so a re-run overwrites in place instead of accumulating.
    """
    for tenant_id, schemas in (
        (TENANT_RETRIEVAL, CORPUS_SCHEMAS),
        (TENANT_FANOUT, CORPUS_SCHEMAS + (UNSERVED_ENCODER_SCHEMA,)),
    ):
        backend = _tenant_backend(retrieval_config_manager, tenant_id)
        backend.schema_registry.deploy_schemas(tenant_id, list(schemas))
        for schema in CORPUS_SCHEMAS:
            result = feed_text_documents(
                backend_client=backend,
                schema_name=schema,
                inference_url=pylate_server,
                model_name=COLBERT_MODEL,
                documents=CORPUS,
                corpus_id=CORPUS_ID,
            )
            assert (
                result.documents_fed,
                result.documents_processed,
                result.errors,
            ) == (len(CORPUS), len(CORPUS), [])
    time.sleep(3)
    return CORPUS


@pytest.fixture(scope="module")
def retrieval_dispatcher(retrieval_config_manager, seeded_corpus):
    return _dispatcher(retrieval_config_manager)


def _hit_ids(grounding) -> list[str]:
    return [hit["id"] for hit in grounding.hits]


@pytest.mark.asyncio
class TestGroundingSearchReturnsTheTenantsDocuments:
    """The grounding search returns THIS tenant's documents, ranked."""

    async def test_the_queried_document_ranks_first(self, retrieval_dispatcher):
        grounding = await retrieval_dispatcher._resolve_answer_search_results(
            HARBOUR_QUERY, TENANT_RETRIEVAL, None, top_k=10
        )

        assert grounding.state == GROUNDING_SEARCHED
        assert list(grounding.profiles) == RETRIEVAL_PROFILES
        assert _hit_ids(grounding) == [HARBOUR_ID, BEEKEEPING_ID]

    async def test_querying_the_other_document_flips_the_order(
        self, retrieval_dispatcher
    ):
        """The order pin has teeth: the same corpus, the other query."""
        grounding = await retrieval_dispatcher._resolve_answer_search_results(
            BEEKEEPING_QUERY, TENANT_RETRIEVAL, None, top_k=10
        )

        assert grounding.state == GROUNDING_SEARCHED
        assert _hit_ids(grounding) == [BEEKEEPING_ID, HARBOUR_ID]

    async def test_summarizer_envelope_reports_the_grounding_it_searched(
        self, retrieval_dispatcher, ensure_host_ollama
    ):
        result = await retrieval_dispatcher._execute_summarization_task(
            HARBOUR_QUERY, TENANT_RETRIEVAL
        )

        assert result["grounding"] == {
            "state": GROUNDING_SEARCHED,
            "modalities": [],
            "profiles": RETRIEVAL_PROFILES,
            "degraded_profiles": [],
            "degraded_query_rewrite": None,
            "undeployed_profiles": [],
            "result_count": len(CORPUS),
        }


def _reset_query_encoder_cache() -> None:
    """Drop the process-wide encoder cache.

    ``QueryEncoderFactory`` keys encoders by (model, service name, dim) and not
    by the resolved URL, so a service pointed at a different endpoint between
    tests would otherwise reuse the first endpoint's client.
    """
    from cogniverse_core.query.encoders import QueryEncoderFactory

    QueryEncoderFactory._encoder_cache.clear()
    QueryEncoderFactory._encoder_key_locks.clear()


class _StubEncoderService(http.server.BaseHTTPRequestHandler):
    """Answers every encode request with ``delay_s`` then a 503."""

    delay_s = 0.0

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        time.sleep(type(self).delay_s)
        self.send_response(503)
        self.end_headers()
        self.wfile.write(b"encoder unavailable")

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        self.do_POST()

    def log_message(self, *args):
        return


@contextlib.contextmanager
def _stub_encoder_service(delay_s: float):
    """A served endpoint that stalls ``delay_s`` and then fails."""
    handler = type("_Stub", (_StubEncoderService,), {"delay_s": delay_s})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _dispatcher_with_colpali_at(retrieval_vespa, pylate_url: str, colpali_url: str):
    """A dispatcher whose visual document profile encodes at ``colpali_url``."""
    config_manager = _retrieval_config_manager(retrieval_vespa, pylate_url)
    system_config = config_manager.get_system_config()
    service_urls = dict(system_config.inference_service_urls)
    service_urls[
        (_SHIPPED_PROFILE_DATA[UNSERVED_ENCODER_PROFILE]["inference_services"])[
            "embedding"
        ]
    ] = colpali_url
    config_manager.set_system_config(
        dataclasses.replace(system_config, inference_service_urls=service_urls)
    )
    _reset_query_encoder_cache()
    return _dispatcher(config_manager)


@pytest.mark.asyncio
class TestOneProfilesEncoderDownDegradesTheGrounding:
    """A fan-out leg whose encoder is unreachable is named, not erased."""

    async def test_healthy_profiles_still_rank_and_the_failed_one_is_named(
        self, retrieval_config_manager, seeded_corpus
    ):
        dispatcher = _dispatcher(retrieval_config_manager)

        grounding = await dispatcher._resolve_answer_search_results(
            HARBOUR_QUERY, TENANT_FANOUT, None, top_k=10
        )

        assert grounding.state == GROUNDING_SEARCHED_DEGRADED
        assert list(grounding.profiles) == RETRIEVAL_PROFILES
        assert grounding.degraded_profiles == (
            (UNSERVED_ENCODER_PROFILE, "encode_failed"),
        )
        assert _hit_ids(grounding) == [HARBOUR_ID, BEEKEEPING_ID]
        assert grounding.envelope() == {
            "state": GROUNDING_SEARCHED_DEGRADED,
            "modalities": [],
            "profiles": RETRIEVAL_PROFILES,
            "degraded_profiles": [
                {"profile": UNSERVED_ENCODER_PROFILE, "reason": "encode_failed"}
            ],
            "degraded_query_rewrite": None,
            "undeployed_profiles": [],
            "result_count": len(CORPUS),
        }


@pytest.mark.asyncio
class TestGroundingSearchIsBounded:
    """A leg that never answers must not hold the answer open."""

    async def test_a_hung_leg_returns_at_the_configured_budget(
        self, retrieval_vespa, pylate_server, seeded_corpus
    ):
        with _stub_encoder_service(delay_s=SHIPPED_GROUNDING_BUDGET_S * 20) as stub:
            dispatcher = _dispatcher_with_colpali_at(
                retrieval_vespa, pylate_server, stub
            )
            started = time.perf_counter()
            grounding = await dispatcher._resolve_answer_search_results(
                HARBOUR_QUERY, TENANT_FANOUT, None, top_k=10
            )
            elapsed = time.perf_counter() - started
        _reset_query_encoder_cache()

        assert grounding.state == GROUNDING_SEARCH_UNAVAILABLE
        assert grounding.hits == []
        assert list(grounding.profiles) == FANOUT_PROFILES
        assert SHIPPED_GROUNDING_BUDGET_S <= elapsed < SHIPPED_GROUNDING_BUDGET_S + 5, (
            f"budget {SHIPPED_GROUNDING_BUDGET_S}s, returned in {elapsed:.2f}s"
        )

    async def test_concurrent_dispatches_all_return_at_the_budget(
        self, retrieval_vespa, pylate_server, seeded_corpus
    ):
        concurrency = 4
        with _stub_encoder_service(delay_s=SHIPPED_GROUNDING_BUDGET_S * 20) as stub:
            dispatcher = _dispatcher_with_colpali_at(
                retrieval_vespa, pylate_server, stub
            )
            started = time.perf_counter()
            groundings = await asyncio.gather(
                *(
                    dispatcher._resolve_answer_search_results(
                        HARBOUR_QUERY, TENANT_FANOUT, None, top_k=10
                    )
                    for _ in range(concurrency)
                )
            )
            elapsed = time.perf_counter() - started
        _reset_query_encoder_cache()

        assert [g.state for g in groundings] == [
            GROUNDING_SEARCH_UNAVAILABLE
        ] * concurrency
        assert [g.hits for g in groundings] == [[]] * concurrency
        assert SHIPPED_GROUNDING_BUDGET_S <= elapsed < SHIPPED_GROUNDING_BUDGET_S + 5, (
            f"{concurrency} concurrent dispatches took {elapsed:.2f}s against a "
            f"{SHIPPED_GROUNDING_BUDGET_S}s budget"
        )


@pytest.mark.asyncio
class TestConcurrentDispatchesDeriveProfilesPerRequest:
    """Concurrent grounding derives each request's own tenant's profiles."""

    async def test_every_in_flight_request_derives_its_own_tenants_profiles(
        self, retrieval_config_manager, seeded_corpus
    ):
        per_tenant = 6
        tenants = [TENANT_RETRIEVAL, TENANT_FANOUT] * per_tenant
        # Every derivation blocks until all of them are in flight, so a
        # serialized implementation breaks the barrier instead of passing.
        barrier = threading.Barrier(len(tenants), timeout=60)
        counts: collections.Counter = collections.Counter()
        counts_lock = threading.Lock()

        class CountingDispatcher(AgentDispatcher):
            def _servable_grounding_profiles(self, tenant_id, modalities):
                with counts_lock:
                    counts[tenant_id] += 1
                barrier.wait()
                return super()._servable_grounding_profiles(tenant_id, modalities)

        dispatcher = _dispatcher(retrieval_config_manager, CountingDispatcher)

        plans = await asyncio.gather(
            *(
                dispatcher._grounding_plan(HARBOUR_QUERY, tenant_id, {}, None)
                for tenant_id in tenants
            )
        )

        assert dict(counts) == {
            TENANT_RETRIEVAL: per_tenant,
            TENANT_FANOUT: per_tenant,
        }
        assert [list(plan.profiles) for plan in plans] == [
            RETRIEVAL_PROFILES if tenant_id == TENANT_RETRIEVAL else FANOUT_PROFILES
            for tenant_id in tenants
        ]
        assert {plan.state for plan in plans} == {GROUNDING_SEARCHED}
        assert {plan.undeployed_profiles for plan in plans} == {()}


@pytest.mark.asyncio
class TestEnsembleFanOutCost:
    """What the fan-out costs, and that it is paid in parallel."""

    RUNS = 5
    THIRD_LEG_DELAY_S = 4.0

    async def _time_search(self, dispatcher, profiles, runs):
        durations = []
        for _ in range(runs):
            started = time.perf_counter()
            search = await dispatcher._execute_search_task(
                HARBOUR_QUERY,
                TENANT_RETRIEVAL,
                top_k=10,
                enrichment={"profiles": list(profiles)},
            )
            durations.append(time.perf_counter() - started)
            assert [hit["id"] for hit in search["results"]] == [
                HARBOUR_ID,
                BEEKEEPING_ID,
            ]
        return durations

    async def test_fan_out_cost_is_paid_in_parallel_and_fits_the_budget(
        self, retrieval_vespa, retrieval_config_manager, pylate_server, seeded_corpus
    ):
        dispatcher = _dispatcher(retrieval_config_manager)

        plan_durations = []
        for _ in range(self.RUNS):
            started = time.perf_counter()
            plan = await dispatcher._grounding_plan(
                HARBOUR_QUERY, TENANT_RETRIEVAL, {}, None
            )
            plan_durations.append(time.perf_counter() - started)
            assert plan.state == GROUNDING_SEARCHED
            assert plan.profiles == tuple(RETRIEVAL_PROFILES)
            assert plan.undeployed_profiles == ()

        # Cold: no cached SearchAgent and no cached encoder, the state a pod
        # is in for its first grounded answer.
        dispatcher._search_agent_cache.clear()
        _reset_query_encoder_cache()
        cold_two_legs = await self._time_search(dispatcher, RETRIEVAL_PROFILES, 1)
        dispatcher._search_agent_cache.clear()
        _reset_query_encoder_cache()
        cold_one_leg = await self._time_search(dispatcher, RETRIEVAL_PROFILES[:1], 1)

        one_leg = await self._time_search(dispatcher, RETRIEVAL_PROFILES[:1], self.RUNS)
        two_legs = await self._time_search(dispatcher, RETRIEVAL_PROFILES, self.RUNS)

        with _stub_encoder_service(delay_s=self.THIRD_LEG_DELAY_S) as stub:
            delayed = _dispatcher_with_colpali_at(retrieval_vespa, pylate_server, stub)
            started = time.perf_counter()
            grounding = await delayed._resolve_answer_search_results(
                HARBOUR_QUERY, TENANT_FANOUT, None, top_k=10
            )
            three_legs = time.perf_counter() - started
        _reset_query_encoder_cache()

        print(
            "\ngrounding fan-out latency (seconds)\n"
            f"  _grounding_plan            n={self.RUNS} "
            f"min={min(plan_durations):.3f} max={max(plan_durations):.3f}\n"
            f"  1 profile  warm            n={self.RUNS} "
            f"min={min(one_leg):.3f} max={max(one_leg):.3f}\n"
            f"  1 profile  cold            {cold_one_leg[0]:.3f}\n"
            f"  2 profiles warm (ensemble) n={self.RUNS} "
            f"min={min(two_legs):.3f} max={max(two_legs):.3f}\n"
            f"  2 profiles cold (ensemble) {cold_two_legs[0]:.3f}\n"
            f"  3 profiles, third leg stalled {self.THIRD_LEG_DELAY_S}s: "
            f"{three_legs:.3f}\n"
            f"  profiles: {', '.join(FANOUT_PROFILES)}\n"
            f"  shipped budget: {SHIPPED_GROUNDING_BUDGET_S}s"
        )

        # A leg stalled for THIRD_LEG_DELAY_S delays the ensemble by that
        # stall and not by the stall plus the healthy legs' work: the fan-out
        # is paid in parallel. The lower bound proves the stalled leg was
        # awaited rather than skipped.
        assert (
            self.THIRD_LEG_DELAY_S
            <= three_legs
            < self.THIRD_LEG_DELAY_S + 2 * max(two_legs)
        ), (
            f"three legs {three_legs:.3f}s with a "
            f"{self.THIRD_LEG_DELAY_S}s stall and two legs at {max(two_legs):.3f}s"
        )
        assert grounding.degraded_profiles == (
            (UNSERVED_ENCODER_PROFILE, "encode_failed"),
        )
        assert _hit_ids(grounding) == [HARBOUR_ID, BEEKEEPING_ID]
        # Every measured path, cold included, fits the shipped budget.
        assert (
            max([*one_leg, *two_legs, cold_one_leg[0], cold_two_legs[0]])
            < SHIPPED_GROUNDING_BUDGET_S
        )


# --- One query rewrite, whatever the profile count ----------------------------
#
# The single-profile path ran the SearchAgent's DSPy rewrite and the ensemble
# path skipped it, so the same question reached Vespa as two different queries
# depending only on how many profiles the tenant serves. These pin that both
# paths rewrite once, through the tenant's LM, inside the grounding budget, and
# that a rewrite that fails or hangs still searches — on the original query,
# named in the outcome.

REWRITE_QUERY = "what did the survey say about silt in the tidal basin"
REWRITE_QUERY_SINGLE_COST = "which berth needs its sediment removed before winter"
REWRITE_QUERY_ENSEMBLE_COST = "how much silt did the basin survey record"
REWRITE_QUERY_NO_LM = "silt in the northern berth before the winter season"
GROUNDING_REWRITE_BUDGET_S = SHIPPED_GROUNDING_BUDGET_S - GROUNDING_SEARCH_RESERVE_S


def _counting_test_lm():
    """The provisioned test LM, counting every round trip it serves."""
    from tests.fixtures.llm import (
        resolve_api_key,
        resolve_base_url,
        resolve_prefixed_model,
    )

    class _CountingLM(dspy.LM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.calls = 0

        def __call__(self, *args, **kwargs):
            self.calls += 1
            return super().__call__(*args, **kwargs)

    return _CountingLM(
        model=resolve_prefixed_model(),
        api_base=resolve_base_url(),
        api_key=resolve_api_key(),
    )


def _dead_port() -> int:
    """A port nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class _StubLLMService(http.server.BaseHTTPRequestHandler):
    """An OpenAI-compatible endpoint that stalls ``delay_s`` then fails."""

    delay_s = 0.0

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        time.sleep(type(self).delay_s)
        self.send_response(503)
        self.end_headers()
        self.wfile.write(b"llm unavailable")

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        self.do_POST()

    def log_message(self, *args):
        return


@contextlib.contextmanager
def _stub_llm_service(delay_s: float):
    handler = type("_StubLLM", (_StubLLMService,), {"delay_s": delay_s})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


async def _search(dispatcher, query, profiles, enhanced_query=None):
    enrichment = {"profiles": list(profiles)}
    if enhanced_query is not None:
        enrichment["enhanced_query"] = enhanced_query
    return await dispatcher._execute_search_task(
        query,
        TENANT_RETRIEVAL,
        top_k=10,
        enrichment=enrichment,
    )


@pytest.mark.asyncio
class TestBothSearchPathsRewriteTheQueryOnce:
    async def test_one_profile_and_an_ensemble_search_the_same_rewritten_query(
        self, retrieval_dispatcher, ensure_host_ollama
    ):
        lm = _counting_test_lm()

        with dspy.context(lm=lm):
            single = await _search(
                retrieval_dispatcher, REWRITE_QUERY, RETRIEVAL_PROFILES[:1]
            )
            after_single = lm.calls
            ensemble = await _search(
                retrieval_dispatcher, REWRITE_QUERY, RETRIEVAL_PROFILES
            )

        assert (single["search_mode"], ensemble["search_mode"]) == (
            "single_profile",
            "ensemble",
        )
        assert (after_single, lm.calls - after_single) == (1, 1), (
            "one rewrite per search — never one per ensemble leg"
        )
        rewritten = single["enhanced_query"]
        assert rewritten != REWRITE_QUERY, "the rewrite left the query unchanged"
        assert rewritten == rewritten.strip() != ""
        assert ensemble["enhanced_query"] == rewritten, (
            "the profile count must not change which query is searched"
        )
        assert (
            single["degraded_query_rewrite"],
            ensemble["degraded_query_rewrite"],
        ) == (None, None)
        assert [hit["id"] for hit in single["results"]] == [HARBOUR_ID, BEEKEEPING_ID]
        assert [hit["id"] for hit in ensemble["results"]] == [
            HARBOUR_ID,
            BEEKEEPING_ID,
        ]

    async def test_the_rewrite_costs_one_round_trip_on_either_path(
        self, retrieval_dispatcher, ensure_host_ollama
    ):
        """What the rewrite costs each path, measured on this fixture."""
        timings = {}

        async def _timed(name, query, profiles, **kwargs):
            started = time.perf_counter()
            search = await _search(retrieval_dispatcher, query, profiles, **kwargs)
            timings[name] = time.perf_counter() - started
            assert [hit["id"] for hit in search["results"]] == [
                HARBOUR_ID,
                BEEKEEPING_ID,
            ]
            return search

        # A rewrite already made upstream is the cost of the search alone —
        # the ensemble's behaviour before it rewrote.
        await _timed(
            "one profile, rewrite skipped",
            REWRITE_QUERY_SINGLE_COST,
            RETRIEVAL_PROFILES[:1],
            enhanced_query=REWRITE_QUERY_SINGLE_COST,
        )
        await _timed(
            "ensemble, rewrite skipped",
            REWRITE_QUERY_ENSEMBLE_COST,
            RETRIEVAL_PROFILES,
            enhanced_query=REWRITE_QUERY_ENSEMBLE_COST,
        )
        # No LM to rewrite through: the rewrite path's cost without its round
        # trip (the tenant-instruction read it makes before the call).
        with dspy.context(lm=None):
            await _timed(
                "one profile, no rewrite LM",
                REWRITE_QUERY_NO_LM,
                RETRIEVAL_PROFILES[:1],
            )
        with dspy.context(lm=_counting_test_lm()):
            single = await _timed(
                "one profile, rewritten",
                REWRITE_QUERY_SINGLE_COST,
                RETRIEVAL_PROFILES[:1],
            )
            ensemble = await _timed(
                "ensemble, rewritten",
                REWRITE_QUERY_ENSEMBLE_COST,
                RETRIEVAL_PROFILES,
            )

        print(
            "\ngrounding search latency with the query rewrite (seconds)\n"
            + "\n".join(f"  {name:<28} {value:.3f}" for name, value in timings.items())
            + f"\n  rewrite budget: {GROUNDING_REWRITE_BUDGET_S}s of a "
            f"{SHIPPED_GROUNDING_BUDGET_S}s grounding budget"
        )

        assert single["enhanced_query"] != REWRITE_QUERY_SINGLE_COST
        assert ensemble["enhanced_query"] != REWRITE_QUERY_ENSEMBLE_COST
        assert max(timings.values()) < SHIPPED_GROUNDING_BUDGET_S, (
            f"every path fits the shipped budget: {timings}"
        )


@pytest.mark.asyncio
class TestAFailedRewriteStillSearches:
    """The rewrite is best-effort: the search runs on the original query."""

    async def test_a_dead_rewrite_lm_searches_the_original_query(
        self, retrieval_dispatcher, seeded_corpus
    ):
        dead = dspy.LM(
            "openai/rewrite-lm",
            api_base=f"http://127.0.0.1:{_dead_port()}/v1",
            api_key="not-required",
            num_retries=0,
        )

        started = time.perf_counter()
        with dspy.context(lm=dead):
            grounding = await retrieval_dispatcher._resolve_answer_search_results(
                HARBOUR_QUERY, TENANT_RETRIEVAL, None, top_k=10
            )
        elapsed = time.perf_counter() - started

        assert grounding.degraded_query_rewrite == QUERY_REWRITE_FAILED
        assert grounding.state == GROUNDING_SEARCHED_DEGRADED
        assert grounding.degraded_profiles == ()
        assert _hit_ids(grounding) == [HARBOUR_ID, BEEKEEPING_ID]
        assert grounding.envelope() == {
            "state": GROUNDING_SEARCHED_DEGRADED,
            "modalities": [],
            "profiles": RETRIEVAL_PROFILES,
            "degraded_profiles": [],
            "degraded_query_rewrite": QUERY_REWRITE_FAILED,
            "undeployed_profiles": [],
            "result_count": len(CORPUS),
        }
        assert elapsed < SHIPPED_GROUNDING_BUDGET_S, (
            f"a dead rewrite LM took {elapsed:.2f}s of a "
            f"{SHIPPED_GROUNDING_BUDGET_S}s budget"
        )

    async def test_a_hung_rewrite_lm_returns_within_the_grounding_budget(
        self, retrieval_dispatcher, seeded_corpus
    ):
        with _stub_llm_service(delay_s=GROUNDING_REWRITE_BUDGET_S + 3) as stub:
            hung = dspy.LM(
                "openai/rewrite-lm",
                api_base=stub,
                api_key="not-required",
                num_retries=0,
            )
            started = time.perf_counter()
            with dspy.context(lm=hung):
                grounding = await retrieval_dispatcher._resolve_answer_search_results(
                    BEEKEEPING_QUERY, TENANT_RETRIEVAL, None, top_k=10
                )
            elapsed = time.perf_counter() - started

        assert grounding.degraded_query_rewrite == QUERY_REWRITE_TIMED_OUT
        assert grounding.state == GROUNDING_SEARCHED_DEGRADED
        assert _hit_ids(grounding) == [BEEKEEPING_ID, HARBOUR_ID]
        assert GROUNDING_REWRITE_BUDGET_S <= elapsed < SHIPPED_GROUNDING_BUDGET_S, (
            f"a hung rewrite returned in {elapsed:.2f}s against a "
            f"{GROUNDING_REWRITE_BUDGET_S}s rewrite budget inside a "
            f"{SHIPPED_GROUNDING_BUDGET_S}s grounding budget"
        )
