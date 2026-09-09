"""The /v1 harness surface on the deployed runtime.

The router is mounted unconditionally, so a 404 from ``GET /v1/models`` says
the deployed image predates the surface — the exact regression this file
exists to catch, and a failure naming the image rather than a skip.

The module owns everything it asserts on: it mints an org and tenant,
deploys the document profile into it, seeds the two annotated caption
documents and mints a harness key bound to that tenant. Nothing here reads
the shared seeded corpus, so the assertions cannot pass on what an earlier
run left behind.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest

from tests.e2e.conftest import (
    RUNTIME,
    TENANT_DEPLOY_TIMEOUT_S,
    _ingest_sample_documents,
    register_tenant_and_wait,
    runtime_available,
    unique_id,
)
from tests.e2e.test_api_e2e import DOCUMENT_PROFILE, _deploy_profile_for_tenant

pytestmark = pytest.mark.e2e

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.json"

# The captions the tenant is seeded with, in the order the document route
# ranks them for this query (same corpus and query as
# tests/e2e/test_a2a_gateway_e2e.py::TestGatewaySeededSearchContract).
DOCUMENT_QUERY = "find PDF documents about washing dishes"
EXPECTED_TITLES = ["v_0BtHd6dvm78.txt", "v_-nl4G-00PtA.txt"]

SUMMARY_QUERY = "what are the seeded documents about?"
KEY_HASH = re.compile(r"[0-9a-f]{64}")
COMPLETION_ID = re.compile(r"chatcmpl-[0-9a-f]{32}")


def harness_models() -> dict[str, str]:
    """The model -> agent map the runtime serves, from the shipped config."""
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    rendered = re.sub(r"\{\{[^}]*\}\}", "http://rendered.invalid", raw)
    return json.loads(rendered)["harness"]["models"]


def require_runtime() -> None:
    assert runtime_available(), (
        f"the runtime is not reachable at {RUNTIME}; the e2e suite runs "
        "against a deployed k3d cluster (`cogniverse up`)"
    )


@pytest.fixture(scope="module")
def harness_tenant():
    """An owned tenant seeded with the caption documents, plus its key."""
    require_runtime()
    org_id = unique_id("pi_harness")
    tenant_id = f"{org_id}:t1"

    with httpx.Client(base_url=RUNTIME, timeout=TENANT_DEPLOY_TIMEOUT_S) as client:
        created_org = client.post(
            "/admin/organizations",
            json={
                "org_id": org_id,
                "org_name": org_id.replace("_", "-"),
                "created_by": "e2e",
            },
        )
        assert created_org.status_code in (200, 201), created_org.text

        register_tenant_and_wait(tenant_id, created_by="e2e")
        _deploy_profile_for_tenant(client, DOCUMENT_PROFILE, tenant_id)
        seeded = _ingest_sample_documents(tenant_id=tenant_id)

        minted = client.post(
            "/admin/harness/keys",
            json={"tenant_id": tenant_id, "name": "pi-harness-e2e"},
        )
        assert minted.status_code == 200, minted.text
        record = minted.json()
        assert set(record) == {
            "key",
            "key_hash",
            "key_prefix",
            "tenant_id",
            "name",
            "created_at",
            "revoked",
        }
        assert record["tenant_id"] == tenant_id
        assert record["name"] == "pi-harness-e2e"
        assert record["revoked"] is False
        assert KEY_HASH.fullmatch(record["key_hash"])
        assert record["key_prefix"] == record["key_hash"][:12]

        try:
            yield tenant_id, seeded, record["key"]
        finally:
            revoked = client.delete(f"/admin/harness/keys/{record['key_hash']}")
            assert revoked.status_code == 200, revoked.text
            assert revoked.json() == {
                "revoked": True,
                "key_hash": record["key_hash"],
            }


def test_models_needs_a_key_and_lists_the_configured_catalogue(harness_tenant):
    _, _, key = harness_tenant

    with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
        unauthorized = client.get("/v1/models")
        listed = client.get("/v1/models", headers={"Authorization": f"Bearer {key}"})

    assert unauthorized.status_code == 401
    assert unauthorized.headers["www-authenticate"] == "Bearer"
    assert unauthorized.json()["error"]["code"] == "invalid_api_key"

    assert listed.status_code == 200, (
        f"GET /v1/models returned {listed.status_code}; the deployed runtime "
        f"image at {RUNTIME} does not serve the harness surface"
    )
    body = listed.json()
    assert body["object"] == "list"
    assert [model["id"] for model in body["data"]] == list(harness_models())
    assert {(model["object"], model["owned_by"]) for model in body["data"]} == {
        ("model", "cogniverse")
    }


def test_the_seeded_captions_are_the_documents_this_tenant_serves(harness_tenant):
    tenant_id, seeded, _ = harness_tenant

    with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
        response = client.post(
            "/search/",
            json={
                "query": DOCUMENT_QUERY,
                "profile": DOCUMENT_PROFILE,
                "top_k": 10,
                "tenant_id": tenant_id,
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    results = body["results"]
    assert body["results_count"] == len(results)
    assert [result["title"] for result in results] == EXPECTED_TITLES
    assert [result["document_id"] for result in results] == [
        seeded[title] for title in EXPECTED_TITLES
    ]
    scores = [result["relevance_score"] for result in results]
    assert scores == sorted(scores, reverse=True)


def test_a_chat_completion_answers_from_the_seeded_documents(harness_tenant):
    tenant_id, seeded, key = harness_tenant

    with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "cogniverse/summarizer",
                "messages": [{"role": "user", "content": SUMMARY_QUERY}],
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"id", "object", "created", "model", "choices", "usage"}
    assert COMPLETION_ID.fullmatch(body["id"])
    assert body["object"] == "chat.completion"
    assert body["model"] == "cogniverse/summarizer"
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert set(choice) == {"index", "message", "finish_reason"}
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    usage = body["usage"]
    assert set(usage) == {"prompt_tokens", "completion_tokens", "total_tokens"}
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    # The summary is LM free text, so what is pinned is its grounding: the
    # tenant owns only the two dish-washing captions, which is the single
    # subject any faithful answer can name. The key resolved to this tenant,
    # so a summary of anything else means the turn read another corpus.
    assert "dish" in choice["message"]["content"].lower(), choice["message"]["content"]
    assert set(seeded) == set(EXPECTED_TITLES)
