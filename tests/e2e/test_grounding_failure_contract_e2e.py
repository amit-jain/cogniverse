"""An answer agent whose retrieval cannot run fails the turn.

The summary and report paths ground their answer in a real search over the
tenant's profiles. When that search cannot be made — the profile it was told
to ground on is no longer one this tenant serves — the turn must come back as
a declared failure naming the grounding class, never as an answer written from
the query alone and reported as a success.

The module owns every tenant it touches: it mints an org and a tenant, deploys
the document profile into it, seeds the two committed caption documents, and
then removes that profile and its schema itself.
"""

from __future__ import annotations

import time
import uuid

import httpx
import pytest

from cogniverse_agents.wiki.wiki_manager import _AUTO_FILE_AGENTS, _WIKI_BASE_SCHEMA
from cogniverse_runtime.agent_dispatcher import (
    GROUNDING_SEARCHED,
    AnswerGroundingUnavailable,
)
from tests.e2e.conftest import (
    RUNTIME,
    SAMPLE_DOCUMENT_TITLES,
    TENANT_DEPLOY_TIMEOUT_S,
    _deployed_schema_names_strict,
    _ingest_sample_documents,
    _tenant_schema_name,
    _tenant_schema_names_in_vespa,
    register_tenant_and_wait,
    runtime_available,
    unique_id,
)
from tests.e2e.test_api_e2e import DOCUMENT_PROFILE, _deploy_profile_for_tenant
from tests.e2e.test_pi_harness_e2e import document_schema_for

pytestmark = pytest.mark.e2e

DOCUMENT_QUERY = "find PDF documents about washing dishes"
SEEDED_DOCUMENT_COUNT = len(SAMPLE_DOCUMENT_TITLES)

# The two answer agents that ground before they generate. Each one's dispatch
# envelope names itself, so the failure body is agent-specific.
ANSWER_AGENTS = ("summarizer_agent", "detailed_report_agent")

# The line each agent's dispatch envelope carries once it has answered from a
# search that ran. It is not the answer — that distinction is what a failed
# grounding must not be allowed to blur.
GROUNDED_MESSAGES = {
    "summarizer_agent": f"Generated summary for '{DOCUMENT_QUERY}'",
    "detailed_report_agent": f"Generated detailed report for '{DOCUMENT_QUERY}'",
}


@pytest.fixture(scope="module")
def grounded_tenant():
    """A tenant serving the committed caption documents, and nothing else."""
    assert runtime_available(), (
        f"the runtime is not reachable at {RUNTIME}; the e2e suite runs "
        "against a deployed k3d cluster"
    )
    org_id = unique_id("grounding")
    tenant_id = f"{org_id}:t1"
    with httpx.Client(base_url=RUNTIME, timeout=TENANT_DEPLOY_TIMEOUT_S) as client:
        created = client.post(
            "/admin/organizations",
            json={
                "org_id": org_id,
                "org_name": org_id.replace("_", "-"),
                "created_by": "e2e",
            },
        )
        assert created.status_code in (200, 201), created.text
        register_tenant_and_wait(tenant_id, created_by="e2e", timeout_s=600.0)
        _deploy_profile_for_tenant(client, DOCUMENT_PROFILE, tenant_id)
        seeded = _ingest_sample_documents(tenant_id=tenant_id)
        assert set(seeded) == set(SAMPLE_DOCUMENT_TITLES), seeded
        yield tenant_id


def _answer_turn(agent_name: str, tenant_id: str, session_id: str) -> httpx.Response:
    """One dispatch that grounds on the document profile by name."""
    with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
        return client.post(
            f"/agents/{agent_name}/process",
            json={
                "agent_name": agent_name,
                "query": DOCUMENT_QUERY,
                "context": {"tenant_id": tenant_id},
                "session_id": session_id,
                "profiles": [DOCUMENT_PROFILE],
            },
        )


def _expected_failure_detail(agent_name: str, session_id: str) -> dict:
    return {
        "detail": (
            f"Agent '{agent_name}' failed with "
            f"{AnswerGroundingUnavailable.__name__} "
            f"(request_id={session_id}). See runtime logs for detail."
        )
    }


class TestAnswerAgentsFailWhenRetrievalCannotRun:
    """Grounded first, then ungroundable: the same request, two outcomes."""

    def test_a_failed_retrieval_fails_the_turn_instead_of_answering(
        self, grounded_tenant
    ):
        # 1. While the profile is deployed, both agents ground in it and say so.
        for agent_name in ANSWER_AGENTS:
            session_id = f"e2e-grounded-{uuid.uuid4().hex}"
            response = _answer_turn(agent_name, grounded_tenant, session_id)
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["status"] == "success", body
            assert body["agent"] == agent_name, body
            assert body["grounding"] == {
                "state": GROUNDING_SEARCHED,
                "modalities": [],
                "profiles": [DOCUMENT_PROFILE],
                "degraded_profiles": [],
                "degraded_query_rewrite": None,
                "undeployed_profiles": [],
                "result_count": SEEDED_DOCUMENT_COUNT,
            }, body["grounding"]
            assert body["message"] == GROUNDED_MESSAGES[agent_name], body["message"]

        # 2. Take the profile and its schema away. Nothing else changes.
        #    First wait for the wiki schema the background wiki filing deploys.
        assert "detailed_report_agent" in _AUTO_FILE_AGENTS
        wiki_schema = _tenant_schema_name(_WIKI_BASE_SCHEMA, grounded_tenant)
        deadline = time.monotonic() + TENANT_DEPLOY_TIMEOUT_S
        while wiki_schema not in _tenant_schema_names_in_vespa(
            grounded_tenant, _deployed_schema_names_strict()
        ):
            assert time.monotonic() < deadline, f"{wiki_schema} never deployed"
            time.sleep(2)
        document_schema = document_schema_for(grounded_tenant)
        before = _tenant_schema_names_in_vespa(
            grounded_tenant, _deployed_schema_names_strict()
        )
        assert document_schema in before, before
        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            deleted = client.delete(
                f"/admin/profiles/{DOCUMENT_PROFILE}",
                params={"tenant_id": grounded_tenant, "delete_schema": True},
            )
        assert deleted.status_code == 200, deleted.text
        removed = deleted.json()
        assert removed["profile_name"] == DOCUMENT_PROFILE, removed
        assert removed["schema_deleted"] is True, removed
        after = _tenant_schema_names_in_vespa(
            grounded_tenant, _deployed_schema_names_strict()
        )
        assert after == before - {document_schema}, (before, after)
        listed = httpx.get(
            f"{RUNTIME}/admin/profiles",
            params={"tenant_id": grounded_tenant},
            timeout=120.0,
        )
        assert listed.status_code == 200, listed.text
        assert [row["profile_name"] for row in listed.json()["profiles"]] == [], (
            listed.json()
        )

        # 3. The identical request now names a profile the tenant cannot
        #    search. Every answer agent reports the failure and answers
        #    nothing; the body carries the agent and the failure class only.
        for agent_name in ANSWER_AGENTS:
            session_id = f"e2e-ungrounded-{uuid.uuid4().hex}"
            response = _answer_turn(agent_name, grounded_tenant, session_id)
            assert response.status_code == 500, response.text
            assert response.json() == _expected_failure_detail(
                agent_name, session_id
            ), response.json()
