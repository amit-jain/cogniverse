"""Profile labels require complete, retryable comparisons against real Vespa."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from collections import Counter

import pytest
import requests
from vespa.application import Vespa
from vespa.package import Document, Field, FirstPhaseRanking, RankProfile, Schema

from cogniverse_runtime.optimization_cli import derive_profile_labels
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager
from tests.conftest import _shared_vespa_application_package
from tests.utils.http_fault_proxy import InterceptFaultProxy
from tests.utils.vespa_docker import VespaDockerManager

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]

PROFILES = ["video_a", "video_b"]
SCHEMA = "profile_comparison"


@pytest.fixture(scope="module")
def comparison_vespa():
    manager = VespaDockerManager()
    info = manager.start_container(f"profile-comparison-{uuid.uuid4().hex}")
    try:
        manager.wait_for_config_ready(info)
        schema = Schema(
            name=SCHEMA,
            document=Document(
                fields=[
                    Field(name="title", type="string", indexing=["summary"]),
                    Field(
                        name="group", type="string", indexing=["attribute", "summary"]
                    ),
                    Field(name="a_score", type="float", indexing=["attribute"]),
                    Field(name="b_score", type="float", indexing=["attribute"]),
                ]
            ),
            rank_profiles=[
                RankProfile(
                    name="video_a", first_phase=FirstPhaseRanking("attribute(a_score)")
                ),
                RankProfile(
                    name="video_b", first_phase=FirstPhaseRanking("attribute(b_score)")
                ),
            ],
        )
        VespaSchemaManager(
            backend_endpoint="http://localhost", backend_port=info["config_port"]
        )._deploy_package(_shared_vespa_application_package([schema]))
        manager.wait_for_application_ready(info)
        app = Vespa(url=info["base_url"])
        documents = [
            {
                "id": f"{group}-{kind}",
                "fields": {
                    "title": f"{group}-{kind}.mp4",
                    "group": group,
                    "a_score": 10 if (kind == "target") != (group == "later") else 1,
                    "b_score": 1 if (kind == "target") != (group == "later") else 10,
                },
            }
            for group in ("early", "affected", "later")
            for kind in ("target", "decoy")
        ]
        feeds = []
        app.feed_iterable(
            iter=documents,
            schema=SCHEMA,
            callback=lambda response, identifier: feeds.append(
                (identifier, response.status_code)
            ),
        )
        assert sorted(feeds) == sorted((doc["id"], 200) for doc in documents)
        deadline = time.monotonic() + 30
        while True:
            response = app.query(yql=f"select * from {SCHEMA} where true", hits=10)
            ids = {hit["fields"]["title"] for hit in response.hits}
            if ids == {doc["fields"]["title"] for doc in documents}:
                break
            if time.monotonic() >= deadline:
                pytest.fail(f"Vespa did not index exact comparison corpus: {ids!r}")
            time.sleep(0.1)
        yield info["base_url"]
    finally:
        manager.stop_container(info)


def _derive(endpoint, queries):
    def retrieve(query, profile):
        response = requests.post(
            endpoint + "/search/",
            json={
                "yql": f'select * from {SCHEMA} where group contains "{query}"',
                "ranking": profile,
                "hits": 2,
            },
            timeout=10,
        )
        response.raise_for_status()
        return [
            {"document_id": hit["id"], "metadata": {"title": hit["fields"]["title"]}}
            for hit in response.json()["root"]["children"]
        ]

    return derive_profile_labels(
        [{"query": query, "expected_videos": [f"{query}-target"]} for query in queries],
        PROFILES,
        retrieve,
        title_fields={profile: "title" for profile in PROFILES},
        profile_types={profile: "video" for profile in PROFILES},
    )


def _attempts(proxy):
    return Counter(
        (json.loads(body)["yql"].split('"')[1], json.loads(body)["ranking"])
        for method, _, body in proxy.requests
        if method == "POST"
    )


def test_transient_retrieval_retries_before_selecting_winner(comparison_vespa):
    calls = 0

    def fail_once(_method, _path, body):
        nonlocal calls
        if json.loads(body)["ranking"] == "video_a":
            calls += 1
            if calls == 1:
                return 503, b'{"detail":"encoder unavailable"}'
        return None

    with InterceptFaultProxy(comparison_vespa, fail_once) as proxy:
        labels = _derive(proxy.url, ["affected"])
        assert labels == {"affected": "video_a"}
        assert labels.records[0]["confidence"] == 1.0
        assert labels.exclusions == ()
        assert _attempts(proxy) == {
            ("affected", "video_a"): 2,
            ("affected", "video_b"): 1,
        }


def test_exhausted_retrieval_excludes_only_incomplete_row(comparison_vespa):
    def fail_affected(_method, _path, body):
        payload = json.loads(body)
        if payload["ranking"] == "video_a" and '"affected"' in payload["yql"]:
            return 503, b'{"detail":"Vespa unavailable"}'
        return None

    with InterceptFaultProxy(comparison_vespa, fail_affected) as proxy:
        labels = _derive(proxy.url, ["early", "affected", "later"])
        assert labels == {"early": "video_a", "later": "video_b"}
        assert [
            (row["query"], row["selected_profile"], row["confidence"])
            for row in labels.records
        ] == [("early", "video_a", 1.0), ("later", "video_b", 1.0)]
        assert labels.exclusions == (
            {
                "query": "affected",
                "reason": "incomplete_comparison",
                "position": 1,
                "expected_videos": ["affected-target"],
                "candidate_profiles": PROFILES,
                "failed_profiles": [
                    {
                        "profile": "video_a",
                        "attempts": 3,
                        "cause": {
                            "type": "HTTPError",
                            "message": f"503 Server Error: Service Unavailable for url: {proxy.url}/search/",
                        },
                    }
                ],
            },
        )
        assert labels.exclusions_by_reason == {"incomplete_comparison": 1}
        assert _attempts(proxy) == {
            ("early", "video_a"): 1,
            ("early", "video_b"): 1,
            ("affected", "video_a"): 3,
            ("affected", "video_b"): 1,
            ("later", "video_a"): 1,
            ("later", "video_b"): 1,
        }

    recovered = _derive(comparison_vespa, ["affected"])
    assert recovered == {"affected": "video_a"}
    assert recovered.records[0]["confidence"] == 1.0
    assert recovered.exclusions == ()


@pytest.mark.asyncio
async def test_concurrent_comparisons_isolate_retries(comparison_vespa):
    reached = threading.Event()
    release = threading.Event()
    attempts = 0

    def block_first_affected(_method, _path, body):
        nonlocal attempts
        payload = json.loads(body)
        if payload["ranking"] == "video_a" and '"affected"' in payload["yql"]:
            attempts += 1
            if attempts == 1:
                reached.set()
                if not release.wait(timeout=30):
                    raise AssertionError("comparison barrier was not released")
                return 503, b'{"detail":"encoder unavailable"}'
        return None

    with InterceptFaultProxy(comparison_vespa, block_first_affected) as proxy:
        blocked = asyncio.create_task(
            asyncio.to_thread(_derive, proxy.url, ["affected"])
        )
        try:
            assert await asyncio.to_thread(reached.wait, 10) is True
            sibling = await asyncio.wait_for(
                asyncio.to_thread(_derive, proxy.url, ["later"]), 10
            )
            assert sibling == {"later": "video_b"}
            assert blocked.done() is False
            release.set()
            recovered = await blocked
            assert recovered == {"affected": "video_a"}
            assert recovered.exclusions == ()
            assert _attempts(proxy) == {
                ("affected", "video_a"): 2,
                ("affected", "video_b"): 1,
                ("later", "video_a"): 1,
                ("later", "video_b"): 1,
            }
        finally:
            release.set()
            if not blocked.done():
                await blocked
