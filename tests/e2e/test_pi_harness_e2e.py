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
    _CAPTION_CORPUS_DIR,
    RUNTIME,
    SAMPLE_DOCUMENT_TITLES,
    TENANT_DEPLOY_TIMEOUT_S,
    _ingest_sample_documents,
    register_tenant_and_wait,
    runtime_available,
    unique_id,
)
from tests.e2e.test_api_e2e import DOCUMENT_PROFILE, _deploy_profile_for_tenant

pytestmark = pytest.mark.e2e

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.json"

# The captions the tenant is seeded with, in the order the document route ranks
# them for this query.
DOCUMENT_QUERY = "find PDF documents about washing dishes"
EXPECTED_TITLE_ORDER = ("v_0BtHd6dvm78.txt", "v_-nl4G-00PtA.txt")

# The row and metadata shapes ``POST /search/`` serves for a document profile.
DOCUMENT_ROW_KEYS = {"document_id", "score", "metadata", "highlights", "source_id"}
DOCUMENT_METADATA_KEYS = {
    "creation_timestamp",
    "document_id",
    "document_path",
    "document_title",
    "document_type",
    "documentid",
    "full_text",
    "page_count",
    "sddocname",
    "source_id",
}

KEY_HASH = re.compile(r"[0-9a-f]{64}")
COMPLETION_ID = re.compile(r"chatcmpl-[0-9a-f]{32}")


def rendered_config() -> dict:
    """The shipped config with its deployment placeholders filled in."""
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    rendered = re.sub(r"\{\{[^}]*\}\}", "http://rendered.invalid", raw)
    return json.loads(rendered)


def harness_models() -> dict[str, str]:
    """The model -> agent map the runtime serves, from the shipped config."""
    return rendered_config()["harness"]["models"]


def document_schema_for(tenant_id: str) -> str:
    """The tenant-scoped Vespa schema the document profile writes into."""
    base = rendered_config()["backend"]["profiles"][DOCUMENT_PROFILE]["schema_name"]
    return f"{base}_{tenant_id.replace(':', '_')}"


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
        # Returns only once every seeded caption answers a search for its own
        # content id, so nothing below races indexing.
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
    assert set(seeded) == set(SAMPLE_DOCUMENT_TITLES)
    assert set(EXPECTED_TITLE_ORDER) == set(SAMPLE_DOCUMENT_TITLES)
    org_id, tenant_name = tenant_id.split(":", 1)

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
    assert set(body) == {
        "query",
        "profile",
        "strategy",
        "results_count",
        "results",
        "session_id",
    }
    assert body["query"] == DOCUMENT_QUERY
    assert body["profile"] == DOCUMENT_PROFILE
    assert body["strategy"] == "default"
    assert body["session_id"] is None

    results = body["results"]
    assert body["results_count"] == len(EXPECTED_TITLE_ORDER)
    assert len(results) == len(EXPECTED_TITLE_ORDER)
    assert [set(result) for result in results] == [DOCUMENT_ROW_KEYS] * len(
        EXPECTED_TITLE_ORDER
    )
    assert [set(result["metadata"]) for result in results] == [
        DOCUMENT_METADATA_KEYS
    ] * len(EXPECTED_TITLE_ORDER)

    # A document row carries the seeded content id as source_id and one chunk
    # per caption as document_id; the caption's own text and title ride in
    # metadata. There is no top-level "title" — that shape belongs to the
    # document agent's rows (tests/e2e/test_a2a_gateway_e2e.py), not to this
    # route.
    schema = document_schema_for(tenant_id)
    assert [result["source_id"] for result in results] == [
        seeded[title] for title in EXPECTED_TITLE_ORDER
    ]
    assert [result["document_id"] for result in results] == [
        f"{seeded[title]}_{seeded[title]}" for title in EXPECTED_TITLE_ORDER
    ]
    assert [result["highlights"] for result in results] == [{}, {}]
    assert [result["metadata"]["document_title"] for result in results] == list(
        EXPECTED_TITLE_ORDER
    )
    assert [result["metadata"]["document_type"] for result in results] == ["txt", "txt"]
    assert [result["metadata"]["page_count"] for result in results] == [1, 1]
    assert [result["metadata"]["document_id"] for result in results] == [
        seeded[title] for title in EXPECTED_TITLE_ORDER
    ]
    assert [result["metadata"]["source_id"] for result in results] == [
        seeded[title] for title in EXPECTED_TITLE_ORDER
    ]
    assert [result["metadata"]["sddocname"] for result in results] == [schema, schema]
    assert [result["metadata"]["documentid"] for result in results] == [
        f"id:content:{schema}::{seeded[title]}_{seeded[title]}"
        for title in EXPECTED_TITLE_ORDER
    ]
    assert [result["metadata"]["full_text"] for result in results] == [
        (_CAPTION_CORPUS_DIR / title).read_text(encoding="utf-8")
        for title in EXPECTED_TITLE_ORDER
    ]
    for result, title in zip(results, EXPECTED_TITLE_ORDER):
        path = result["metadata"]["document_path"]
        assert f"/{org_id}/{tenant_name}/media/" in path, path
        assert path.endswith(f"/{seeded[title]}.txt"), path

    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)


def test_a_chat_completion_answers_from_the_seeded_documents(harness_tenant):
    tenant_id, seeded, key = harness_tenant
    assert set(seeded) == set(SAMPLE_DOCUMENT_TITLES)

    # "cogniverse" is the model that routes by modality, so a document query
    # reaches the document route and the answer is the tenant's own hits.
    with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "cogniverse",
                "messages": [{"role": "user", "content": DOCUMENT_QUERY}],
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"id", "object", "created", "model", "choices", "usage"}
    assert COMPLETION_ID.fullmatch(body["id"])
    assert body["object"] == "chat.completion"
    assert body["model"] == "cogniverse"
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert set(choice) == {"index", "message", "finish_reason"}
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    usage = body["usage"]
    assert set(usage) == {"prompt_tokens", "completion_tokens", "total_tokens"}
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    # The answer is the rendered hit list, so grounding is pinned by identity:
    # the tenant's own content ids and caption titles, in rank order, and
    # nothing else.
    lines = choice["message"]["content"].split("\n")
    assert lines[0] == (
        f"Found {len(EXPECTED_TITLE_ORDER)} documents for '{DOCUMENT_QUERY}'"
    ), choice["message"]["content"]
    assert len(lines) == len(EXPECTED_TITLE_ORDER) + 1, choice["message"]["content"]
    rendered = [line.split(" · score ", 1) for line in lines[1:]]
    assert [part[0] for part in rendered] == [
        f"- {seeded[title]}" for title in EXPECTED_TITLE_ORDER
    ]
    assert [part[1].split(": ", 1)[1] for part in rendered] == list(
        EXPECTED_TITLE_ORDER
    )
    scores = [float(part[1].split(": ", 1)[0]) for part in rendered]
    assert scores == sorted(scores, reverse=True)


def test_the_summarizer_grounds_in_this_tenants_document_profile(harness_tenant):
    """A summary dispatch grounds in the profiles this tenant serves.

    The tenant deploys only the document profile, so the grounding search runs
    there and the envelope names it. Grounding used to read the system-level
    ``active_video_profile``, which this tenant never deployed: zero hits and a
    summary reporting that no content was provided.
    """
    tenant_id, seeded, _key = harness_tenant

    with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
        response = client.post(
            "/agents/summarizer_agent/process",
            json={"query": DOCUMENT_QUERY, "tenant_id": tenant_id},
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    assert body["agent"] == "summarizer_agent"
    assert body["grounding"] == {
        "state": "searched_servable_profiles",
        "modalities": ["document"],
        "profiles": [DOCUMENT_PROFILE],
        "degraded_profiles": [],
        "degraded_query_rewrite": None,
        "undeployed_profiles": [],
        "result_count": len(EXPECTED_TITLE_ORDER),
    }
    assert body["result"]["metadata"]["results_analyzed"] == len(EXPECTED_TITLE_ORDER)
