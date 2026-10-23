"""Acoustic search on an EmbeddingGemma 2 audio profile.

Test-owned HTTP servers stand in for ``vllm_omni_embed`` and Vespa's search
API, so the test pins the exact query text sent to the embedder and the exact
search posted to the profile's tenant schema.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from cogniverse_agents.audio_analysis_agent import (
    AudioAnalysisAgent,
    AudioAnalysisDeps,
    AudioSearchInput,
)
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _JsonServer:
    def __init__(self, answer) -> None:
        self.posts: list[tuple[str, dict]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.posts.append((self.path, body))
                data = json.dumps(answer(body)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "_JsonServer":
        self.thread.start()
        return self

    def __exit__(self, *args) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


QUERY_VECTOR = [0.0] * 768
QUERY_VECTOR[7] = 2.0

HIT = {
    "id": "id:audio:audio_embeddinggemma2_acme_prod::rain_rain_w0001",
    "relevance": 0.91,
    "fields": {
        "audio_id": "rain",
        "source_url": "s3://bucket/rain.wav",
        "audio_title": "rain.wav",
        "audio_transcript": "",
        "chunk_start": 10.0,
        "chunk_end": 20.0,
    },
}


def _agent(
    omni_url: str, vespa_url: str, schema_loader=None, audio_schema=None
) -> AudioAnalysisAgent:
    return AudioAnalysisAgent(
        deps=AudioAnalysisDeps(
            tenant_id="acme:prod",
            vespa_endpoint=vespa_url,
            audio_profile="audio_embeddinggemma2",
            audio_schema=audio_schema or "audio_embeddinggemma2",
            acoustic_service="vllm_omni_embed",
            acoustic_endpoint=omni_url,
            deployed_audio_schema=True,
            config_manager=ConfigManager(store=InMemoryConfigStore()),
            schema_loader=schema_loader,
        )
    )


def _acoustic_search(limit: int) -> dict:
    expected = np.zeros(768)
    expected[7] = 1.0
    return {
        "yql": (
            "select * from audio_embeddinggemma2_acme_prod where "
            f"{{targetHits: {limit}, hnsw.exploreAdditionalHits: 200}}"
            "nearestNeighbor(acoustic_embedding, acoustic_query)"
        ),
        "hits": limit,
        "ranking.profile": "acoustic_similarity",
        "input.query(acoustic_query)": expected.tolist(),
    }


async def test_acoustic_search_embeds_with_the_profile_service_and_its_schema():
    with (
        _JsonServer(
            lambda body: {"data": [{"index": 0, "embedding": QUERY_VECTOR}]}
        ) as omni,
        _JsonServer(lambda body: {"root": {"children": [HIT]}}) as vespa,
    ):
        results = await _agent(omni.url, vespa.url).search_audio(
            "rain on a tin roof", search_mode="acoustic", limit=3
        )

    [(_, embed_body)] = omni.posts
    assert embed_body["messages"][0]["content"] == [
        {"type": "text", "text": "task: search result | query: rain on a tin roof"}
    ]
    [(path, search)] = vespa.posts
    assert path == "/search/"
    assert search == _acoustic_search(3)
    assert [(r.audio_id, r.audio_url) for r in results] == [
        ("rain", "s3://bucket/rain.wav")
    ]


async def test_a_search_naming_no_mode_ranks_the_omni_profile_acoustically():
    # The omni audio schema has no transcript-embedding rank profile, so the
    # mode a caller leaves unset is the acoustic one its schema ranks with.
    with (
        _JsonServer(
            lambda body: {"data": [{"index": 0, "embedding": QUERY_VECTOR}]}
        ) as omni,
        _JsonServer(lambda body: {"root": {"children": [HIT]}}) as vespa,
    ):
        agent = _agent(
            omni.url, vespa.url, FilesystemSchemaLoader(Path("configs/schemas"))
        )
        output = await agent._process_impl(
            AudioSearchInput(query="rain on a tin roof", limit=3)
        )

    [(_, search)] = vespa.posts
    assert search == _acoustic_search(3)
    assert [r.audio_id for r in output.results] == ["rain"]


async def test_a_profile_ranking_with_no_default_mode_is_refused(tmp_path):
    (tmp_path / "audio_bare_schema.json").write_text(
        json.dumps({"name": "audio_bare", "rank_profiles": [{"name": "default"}]})
    )
    agent = _agent(
        "http://127.0.0.1:1",
        "http://127.0.0.1:1",
        FilesystemSchemaLoader(tmp_path),
        audio_schema="audio_bare",
    )

    with pytest.raises(ValueError) as caught:
        await agent.search_audio("rain", limit=3)

    assert str(caught.value) == (
        "Audio profile 'audio_embeddinggemma2' (schema 'audio_bare') ranks with "
        "none of ['phased_semantic', 'acoustic_similarity']; name a search mode"
    )


async def test_a_search_naming_no_mode_needs_a_schema_loader():
    agent = _agent("http://127.0.0.1:1", "http://127.0.0.1:1")

    with pytest.raises(ValueError) as caught:
        await agent.search_audio("rain", limit=3)

    assert str(caught.value) == (
        "Audio search on profile 'audio_embeddinggemma2' names no mode and the "
        "agent has no schema loader to choose one"
    )


def test_an_unknown_acoustic_service_is_refused():
    with pytest.raises(ValueError) as caught:
        AudioAnalysisDeps(tenant_id="acme:prod", acoustic_service="face_embed")

    assert (
        "acoustic_service must be one of ['clap_embed', 'vllm_omni_embed'], "
        "got 'face_embed'"
    ) in str(caught.value)


def test_concurrent_first_omni_access_builds_one_client():
    agent = _agent("http://127.0.0.1:9", "http://127.0.0.1:9")
    barrier = threading.Barrier(8)
    seen: list[int] = []

    def touch() -> None:
        barrier.wait()
        seen.append(id(agent.omni_embedder))

    threads = [threading.Thread(target=touch) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(seen) == 8
    assert len(set(seen)) == 1


async def test_an_unreachable_omni_service_raises_naming_it():
    import socket

    from cogniverse_foundation.config.inference_service import (
        InferenceServiceUnavailableError,
    )

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    omni_url = f"http://127.0.0.1:{port}"
    with _JsonServer(lambda body: {"root": {"children": [HIT]}}) as vespa:
        with pytest.raises(InferenceServiceUnavailableError) as caught:
            await _agent(omni_url, vespa.url).search_audio(
                "thunder", search_mode="acoustic", limit=3
            )

    assert caught.value.service == "vllm_omni_embed"
    assert vespa.posts == []
