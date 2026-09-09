"""A profile is grounded on only when this tenant's schema for it is deployed.

Servability was derived from the embedding service URL alone, so a tenant whose
schema had never been deployed (or had been reconciled away) had that profile
advertised and searched: Vespa answered from an application that does not carry
its documents and the empty result was reported as a clean "no results". These
tests drive the real dispatcher against a real Vespa this module owns, with the
second schema deployed through the production deploy path mid-test, and pin
what the envelope reports on each side of that deploy.
"""

from __future__ import annotations

import json
import socket
import time
import uuid
from pathlib import Path

import dspy
import pytest

from cogniverse_agents.profile_selection_agent import (
    servable_tenant_profiles,
    tenant_profile_servability,
)
from cogniverse_agents.search_agent import QUERY_REWRITE_FAILED
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import (
    PROFILE_SCHEMA_NOT_DEPLOYED,
    PROFILE_SERVABLE,
    BackendProfileConfig,
    SystemConfig,
)
from cogniverse_runtime.agent_dispatcher import (
    GROUNDING_NO_DEPLOYED_SCHEMA_FOR_PROFILE,
    GROUNDING_SEARCHED,
    GROUNDING_SEARCHED_DEGRADED,
    AgentDispatcher,
    GroundingPlan,
)
from cogniverse_sdk.interfaces.backend import SchemaNotDeployedError
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.vespa_docker import VespaDockerManager
from tests.utils.vespa_test_helpers import feed_text_documents

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHIPPED_PROFILE_DATA = json.loads(
    (_REPO_ROOT / "configs" / "config.json").read_text()
)["backend"]["profiles"]

# Two shipped document profiles that share one embedding service, so the only
# difference between them in this module is whether their schema is deployed.
DEPLOYED_PROFILE, UNDEPLOYED_PROFILE = sorted(
    name
    for name, data in _SHIPPED_PROFILE_DATA.items()
    if str(data.get("type") or "").lower() == "document"
    and (data.get("inference_services") or {}).get("embedding") == "colbert_pylate"
)
DEPLOYED_SCHEMA = _SHIPPED_PROFILE_DATA[DEPLOYED_PROFILE]["schema_name"]
UNDEPLOYED_SCHEMA = _SHIPPED_PROFILE_DATA[UNDEPLOYED_PROFILE]["schema_name"]
COLBERT_MODEL = _SHIPPED_PROFILE_DATA[DEPLOYED_PROFILE]["embedding_model"]

TENANT_PARTIAL = "grounding_deploy_partial"
TENANT_NONE = "grounding_deploy_none"
TENANT_REWRITE = "grounding_deploy_rewrite"

CORPUS = (
    {
        "id": "deploy_kiln_firing",
        "title": "Stoneware Kiln Firing Log",
        "text": (
            "The bisque firing held at cone six overnight and the glaze "
            "crazing disappeared once the kiln cooled slowly through quartz "
            "inversion."
        ),
    },
    {
        "id": "deploy_canal_locks",
        "title": "Canal Lock Maintenance",
        "text": (
            "Lock gates were regreased and the paddle gearing replaced, so "
            "narrowboats can descend the flight without leaking chambers."
        ),
    },
)
CORPUS_ID = "deploy_corpus"
KILN_ID, CANAL_ID = (f"{CORPUS_ID}_{entry['id']}" for entry in CORPUS)
KILN_QUERY = "glaze crazing after a slow cone six firing"
# The rewrite an orchestrator already made. SearchAgent uses it as it stands
# (search_agent.py _rewrite_query_for_search), so the grounding search these
# tests read makes no LM round trip of its own.
KILN_CONTEXT = {"enhanced_query": KILN_QUERY}


def _dead_port() -> int:
    """A port nothing listens on."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def deploy_vespa():
    """A Vespa this module owns, carrying its config store and its content."""
    from cogniverse_vespa.metadata_schemas import (
        create_config_metadata_schema,
        create_organization_metadata_schema,
        create_tenant_metadata_schema,
    )
    from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
    from tests.conftest import _shared_vespa_application_package

    manager = VespaDockerManager()
    info = manager.start_container(f"grounding-deploy-{uuid.uuid4().hex}")
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


def _config_manager(info, pylate_url: str) -> ConfigManager:
    """Both profiles configured, both embedding services resolved."""
    config_manager = ConfigManager(
        store=VespaConfigStore(
            backend_url="http://localhost", backend_port=info["http_port"]
        )
    )
    config_manager.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=info["http_port"],
            inference_service_urls={"colbert_pylate": pylate_url},
        )
    )
    for tenant_id in (TENANT_PARTIAL, TENANT_NONE, TENANT_REWRITE):
        for name in (DEPLOYED_PROFILE, UNDEPLOYED_PROFILE):
            config_manager.add_backend_profile(
                BackendProfileConfig.from_dict(name, _SHIPPED_PROFILE_DATA[name]),
                tenant_id=tenant_id,
            )
    return config_manager


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


def _deploy_and_feed(config_manager, tenant_id: str, schema: str, pylate_url: str):
    """Deploy one schema through the production path and feed the corpus."""
    backend = _tenant_backend(config_manager, tenant_id)
    backend.schema_registry.deploy_schemas(tenant_id, [schema])
    result = feed_text_documents(
        backend_client=backend,
        schema_name=schema,
        inference_url=pylate_url,
        model_name=COLBERT_MODEL,
        documents=CORPUS,
        corpus_id=CORPUS_ID,
    )
    assert (result.documents_fed, result.documents_processed, result.errors) == (
        len(CORPUS),
        len(CORPUS),
        [],
    )
    time.sleep(3)


@pytest.fixture(scope="module")
def partially_deployed(deploy_vespa, pylate_server):
    """One of each tenant's two profiles has its schema deployed and fed.

    ``TENANT_REWRITE`` is seeded the same way and never deployed further, so
    the test that reads it owns its state whatever the other tests do to
    ``TENANT_PARTIAL``.
    """
    config_manager = _config_manager(deploy_vespa, pylate_server)
    _deploy_and_feed(config_manager, TENANT_PARTIAL, DEPLOYED_SCHEMA, pylate_server)
    _deploy_and_feed(config_manager, TENANT_REWRITE, DEPLOYED_SCHEMA, pylate_server)
    return config_manager


def _dispatcher(config_manager: ConfigManager) -> AgentDispatcher:
    return AgentDispatcher(
        agent_registry=AgentRegistry(
            tenant_id=TENANT_PARTIAL, config_manager=config_manager
        ),
        config_manager=config_manager,
        schema_loader=FilesystemSchemaLoader(_REPO_ROOT / "configs" / "schemas"),
    )


@pytest.mark.asyncio
class TestGroundingSkipsProfilesWithoutADeployedSchema:
    async def test_only_the_deployed_profile_is_servable_and_searched(
        self, partially_deployed, pylate_server
    ):
        """Both profiles resolve their encoder; only one has a schema.

        Deploying the second schema through the production path is the only
        change between the two halves of this test, and it moves that profile
        from "reported, not searched" to "searched".
        """
        dispatcher = _dispatcher(partially_deployed)

        assert [
            (row.name, row.state)
            for row in tenant_profile_servability(partially_deployed, TENANT_PARTIAL)
        ] == [
            (DEPLOYED_PROFILE, PROFILE_SERVABLE),
            (UNDEPLOYED_PROFILE, PROFILE_SCHEMA_NOT_DEPLOYED),
        ]
        assert [
            name
            for name, _profile in servable_tenant_profiles(
                partially_deployed, TENANT_PARTIAL
            )
        ] == [DEPLOYED_PROFILE]

        plan = await dispatcher._grounding_plan(KILN_QUERY, TENANT_PARTIAL, {}, None)
        assert plan == GroundingPlan(
            (), (DEPLOYED_PROFILE,), GROUNDING_SEARCHED, (UNDEPLOYED_PROFILE,)
        )

        grounding = await dispatcher._resolve_answer_search_results(
            KILN_QUERY, TENANT_PARTIAL, KILN_CONTEXT, top_k=10
        )
        assert grounding.envelope() == {
            "state": GROUNDING_SEARCHED,
            "modalities": [],
            "profiles": [DEPLOYED_PROFILE],
            "degraded_profiles": [],
            "degraded_query_rewrite": None,
            "undeployed_profiles": [UNDEPLOYED_PROFILE],
            "result_count": len(CORPUS),
        }
        assert [hit["id"] for hit in grounding.hits] == [KILN_ID, CANAL_ID]

        _deploy_and_feed(
            partially_deployed, TENANT_PARTIAL, UNDEPLOYED_SCHEMA, pylate_server
        )

        assert [
            (row.name, row.state)
            for row in tenant_profile_servability(partially_deployed, TENANT_PARTIAL)
        ] == [
            (DEPLOYED_PROFILE, PROFILE_SERVABLE),
            (UNDEPLOYED_PROFILE, PROFILE_SERVABLE),
        ]

        plan = await dispatcher._grounding_plan(KILN_QUERY, TENANT_PARTIAL, {}, None)
        assert plan == GroundingPlan(
            (), (DEPLOYED_PROFILE, UNDEPLOYED_PROFILE), GROUNDING_SEARCHED
        )

        grounding = await dispatcher._resolve_answer_search_results(
            KILN_QUERY, TENANT_PARTIAL, KILN_CONTEXT, top_k=10
        )
        assert grounding.envelope() == {
            "state": GROUNDING_SEARCHED,
            "modalities": [],
            "profiles": [DEPLOYED_PROFILE, UNDEPLOYED_PROFILE],
            "degraded_profiles": [],
            "degraded_query_rewrite": None,
            "undeployed_profiles": [],
            "result_count": len(CORPUS),
        }
        assert [hit["id"] for hit in grounding.hits] == [KILN_ID, CANAL_ID]

    async def test_a_tenant_with_no_deployed_schema_says_so_and_searches_nothing(
        self, partially_deployed
    ):
        """Distinct from "no servable profile": the profiles ARE configured."""
        dispatcher = _dispatcher(partially_deployed)

        plan = await dispatcher._grounding_plan(KILN_QUERY, TENANT_NONE, {}, None)
        assert plan == GroundingPlan(
            (),
            (),
            GROUNDING_NO_DEPLOYED_SCHEMA_FOR_PROFILE,
            (DEPLOYED_PROFILE, UNDEPLOYED_PROFILE),
        )

        grounding = await dispatcher._resolve_answer_search_results(
            KILN_QUERY, TENANT_NONE, None, top_k=10
        )
        assert grounding.envelope() == {
            "state": GROUNDING_NO_DEPLOYED_SCHEMA_FOR_PROFILE,
            "modalities": [],
            "profiles": [],
            "degraded_profiles": [],
            "degraded_query_rewrite": None,
            "undeployed_profiles": [DEPLOYED_PROFILE, UNDEPLOYED_PROFILE],
            "result_count": 0,
        }
        assert grounding.nothing_to_search is True
        assert grounding.unanswerable_text(TENANT_NONE) == (
            f"Tenant {TENANT_NONE} has no deployed search schema for its "
            f"profiles ({DEPLOYED_PROFILE}, {UNDEPLOYED_PROFILE}), so there is "
            "nothing to search for this request."
        )

    async def test_an_undeployed_profile_and_a_dead_rewrite_lm_are_both_named(
        self, partially_deployed
    ):
        """Two independent degradations on one grounding search.

        One profile was left out because this tenant never deployed its schema;
        the query the other was searched with is the original, because the
        rewrite LM is unreachable. The state reports the rewrite - a search did
        run - and ``undeployed_profiles`` still names what it left out, so
        neither cause is hidden behind the other and the hits are kept.
        """
        dispatcher = _dispatcher(partially_deployed)
        dead = dspy.LM(
            "openai/rewrite-lm",
            api_base=f"http://127.0.0.1:{_dead_port()}/v1",
            api_key="not-required",
            num_retries=0,
        )

        with dspy.context(lm=dead):
            grounding = await dispatcher._resolve_answer_search_results(
                KILN_QUERY, TENANT_REWRITE, None, top_k=10
            )

        assert grounding.envelope() == {
            "state": GROUNDING_SEARCHED_DEGRADED,
            "modalities": [],
            "profiles": [DEPLOYED_PROFILE],
            "degraded_profiles": [],
            "degraded_query_rewrite": QUERY_REWRITE_FAILED,
            "undeployed_profiles": [UNDEPLOYED_PROFILE],
            "result_count": len(CORPUS),
        }
        assert [hit["id"] for hit in grounding.hits] == [KILN_ID, CANAL_ID]
        assert grounding.nothing_to_search is False


class TestDirectSearchOnAnUndeployedSchemaRaises:
    def test_search_names_the_tenant_and_the_schema_instead_of_returning_nothing(
        self, partially_deployed
    ):
        """The empty batch this used to return read as "no matching documents"."""
        backend = _tenant_backend(partially_deployed, TENANT_NONE)

        with pytest.raises(SchemaNotDeployedError) as failure:
            backend.search(
                {
                    "query": KILN_QUERY,
                    "type": "document",
                    "profile": DEPLOYED_PROFILE,
                    "tenant_id": TENANT_NONE,
                    "top_k": 10,
                }
            )

        assert str(failure.value) == (
            f"Tenant '{TENANT_NONE}' has no deployed schema "
            f"'{DEPLOYED_SCHEMA}_{TENANT_NONE}_{TENANT_NONE}' "
            f"(base schema '{DEPLOYED_SCHEMA}', profile '{DEPLOYED_PROFILE}'); "
            "deploy it before searching this profile"
        )

    def test_the_deployed_profile_of_the_same_backend_still_answers(
        self, partially_deployed
    ):
        """The raise is scoped to the undeployed tenant/schema, not the backend."""
        backend = _tenant_backend(partially_deployed, TENANT_PARTIAL)

        results = backend.search(
            {
                "query": KILN_QUERY,
                "type": "document",
                "profile": DEPLOYED_PROFILE,
                "tenant_id": TENANT_PARTIAL,
                "top_k": 10,
            }
        )

        assert [result.document.id for result in results] == [KILN_ID, CANAL_ID]
