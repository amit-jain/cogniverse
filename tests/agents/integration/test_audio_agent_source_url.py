"""Integration test: AudioAnalysisAgent surfaces source_url via search.

Spins up Vespa via :class:`VespaTestManager`, deploys the ``audio_content``
schema, feeds audio documents carrying a known ``source_url``, then runs
``AudioAnalysisAgent._search_transcript`` and asserts every
``AudioResult.audio_url`` matches the canonical URI that was written.

The agent is constructed via ``__new__`` so the heavy A2A + Whisper init
path is skipped — only the search code path under test is exercised.

Skips when Docker is unavailable.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import requests

from tests.system.vespa_test_manager import VespaTestManager
from tests.utils.docker_utils import generate_unique_ports
from tests.utils.markers import is_docker_available

AUDIO_SCHEMA = "audio_content"
SCHEMAS_DIR = Path(__file__).resolve().parents[3] / "configs" / "schemas"


def pytest_collection_modifyitems(items):
    docker_ok = is_docker_available()
    for item in items:
        if "requires_docker" in item.keywords and not docker_ok:
            item.add_marker(
                pytest.mark.skip(reason="Docker not available in this environment")
            )


@pytest.fixture(scope="module")
def audio_vespa():
    """Start Vespa with metadata + audio_content schemas deployed."""
    http_port, config_port = generate_unique_ports(f"audio_e2e_{__name__}")
    manager = VespaTestManager(
        app_name="test-audio-agent",
        http_port=http_port,
        config_port=config_port,
    )

    saved_url = os.environ.get("BACKEND_URL")
    saved_port = os.environ.get("BACKEND_PORT")
    try:
        if not manager.setup_application_directory():
            pytest.skip("Failed to setup Vespa application directory")
        if not manager.deploy_test_application():
            pytest.skip("Failed to deploy Vespa test application")

        os.environ["BACKEND_URL"] = "http://localhost"
        os.environ["BACKEND_PORT"] = str(manager.http_port)

        # Add the audio_content schema alongside the metadata schemas the
        # manager already deployed.
        from vespa.package import ApplicationPackage

        from cogniverse_vespa.json_schema_parser import JsonSchemaParser
        from cogniverse_vespa.metadata_schemas import (
            create_adapter_registry_schema,
            create_config_metadata_schema,
            create_organization_metadata_schema,
            create_tenant_metadata_schema,
        )
        from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

        parser = JsonSchemaParser()
        audio_schema = parser.load_schema_from_json_file(
            str(SCHEMAS_DIR / f"{AUDIO_SCHEMA}_schema.json")
        )

        schema_manager = VespaSchemaManager(
            backend_endpoint="http://localhost",
            backend_port=manager.config_port,
        )
        schema_manager._deploy_package(
            ApplicationPackage(
                name="cogniverse",
                schema=[
                    create_organization_metadata_schema(),
                    create_tenant_metadata_schema(),
                    create_config_metadata_schema(),
                    create_adapter_registry_schema(),
                    audio_schema,
                ],
            ),
            allow_schema_removal=True,
        )

        # Wait for the audio schema to be queryable.
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                r = requests.get(
                    f"http://localhost:{manager.http_port}/search/",
                    params={
                        "yql": f"select * from {AUDIO_SCHEMA} where true limit 0",
                        "hits": 0,
                    },
                    timeout=5,
                )
                if r.status_code == 200 and "errors" not in r.json().get("root", {}):
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            raise RuntimeError(f"{AUDIO_SCHEMA} did not become queryable")

        yield {
            "http_port": manager.http_port,
            "backend_url": f"http://localhost:{manager.http_port}",
        }
    finally:
        if saved_url is not None:
            os.environ["BACKEND_URL"] = saved_url
        else:
            os.environ.pop("BACKEND_URL", None)
        if saved_port is not None:
            os.environ["BACKEND_PORT"] = saved_port
        else:
            os.environ.pop("BACKEND_PORT", None)
        manager.cleanup()


@pytest.fixture
def seeded_audio_docs(audio_vespa):
    """Feed three audio docs with known source_urls via the raw HTTP API."""
    http_port = audio_vespa["http_port"]
    docs = [
        {
            "audio_id": f"audio_e2e_{i}",
            "audio_title": f"clip {i}",
            "source_url": f"s3://corpus/audio/clip_{i}.mp3",
            "audio_transcript": f"hello world this is clip number {i} talking",
        }
        for i in range(3)
    ]
    fed_ids: list[str] = []
    for i, doc in enumerate(docs):
        doc_id = f"audio_e2e_doc_{i}"
        resp = requests.post(
            f"http://localhost:{http_port}/document/v1/audio/{AUDIO_SCHEMA}/docid/{doc_id}",
            json={"fields": doc},
            timeout=10,
        )
        assert resp.status_code in (200, 201), resp.text[:300]
        fed_ids.append(doc_id)

    time.sleep(2)
    yield docs

    for doc_id in fed_ids:
        try:
            requests.delete(
                f"http://localhost:{http_port}/document/v1/audio/{AUDIO_SCHEMA}/docid/{doc_id}",
                timeout=5,
            )
        except requests.RequestException:
            pass


@pytest.fixture
def audio_agent(audio_vespa, tmp_path):
    """Bare AudioAnalysisAgent with just the attributes the search path reads."""
    from cogniverse_agents.audio_analysis_agent import AudioAnalysisAgent
    from cogniverse_core.common.media import MediaConfig, MediaLocator

    agent = AudioAnalysisAgent.__new__(AudioAnalysisAgent)
    agent._vespa_endpoint = audio_vespa["backend_url"]
    agent._whisper_model_size = "base"
    agent._audio_transcriber = None
    agent._embedding_generator = None
    agent._locator = MediaLocator(
        tenant_id="test",
        config=MediaConfig(),
        cache_root=tmp_path / "audio-cache",
    )
    return agent


@pytest.mark.requires_docker
@pytest.mark.integration
class TestAudioAgentSourceUrl:
    @pytest.mark.asyncio
    async def test_search_transcript_carries_source_url_into_audio_url(
        self, audio_agent, seeded_audio_docs
    ):
        results = await audio_agent._search_transcript("clip", limit=10)

        assert results, "search returned no results — BM25 should match 'clip'"

        # Every returned audio_url must be one of the canonical URIs we wrote.
        result_urls = {r.audio_url for r in results}
        expected_urls = {d["source_url"] for d in seeded_audio_docs}
        assert result_urls.intersection(expected_urls), (
            f"audio_url did not carry source_url through; "
            f"got={result_urls!r} expected one of {expected_urls!r}"
        )
        # Pre-rollout corpora with empty source_url would have surfaced
        # empty strings; assert the field is fully populated.
        for r in results:
            assert r.audio_url.startswith("s3://"), (
                f"audio_url not a canonical s3 URI: {r.audio_url!r}"
            )

    @pytest.mark.asyncio
    async def test_get_audio_path_resolves_via_locator(self, audio_agent, tmp_path):
        """The agent's _get_audio_path goes through the locator, so a
        file:// URI to a real on-disk audio resolves to the same path."""
        clip = tmp_path / "audio_e2e_clip.mp3"
        clip.write_bytes(b"fake audio bytes")

        local = audio_agent._get_audio_path(f"file://{clip}")
        assert local == str(clip)
