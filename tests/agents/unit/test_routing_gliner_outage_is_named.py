"""A GLiNER outage reaches the routing consumers as a named degrade.

GLiNERRelationshipExtractor raises GLiNEREntityExtractionUnavailableError when
it cannot answer. The two consumers that keep serving on their own fallback --
ComposableQueryAnalysisModule and RelationshipExtractorTool -- carry the
entity-extraction reason vocabulary on what they return, so an outage is told
apart from a query that genuinely holds no entities.

The boundary is real: the extractor talks to an inference URL that is either a
port nothing listens on or a paused HTTP peer.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time

import pytest

from cogniverse_agents.routing.dspy_relationship_router import (
    ComposableQueryAnalysisModule,
)
from cogniverse_agents.routing.relationship_extraction_tools import (
    GLiNEREntityExtractionUnavailableError,
    GLiNERRelationshipExtractor,
    RelationshipExtractorTool,
    SpaCyDependencyAnalyzer,
)
from cogniverse_core.common.models.model_loaders import RemoteGlinerClient
from cogniverse_foundation.telemetry.span_contract import (
    ENTITY_EXTRACTION_FALLBACK_EXTRACTOR_UNAVAILABLE,
)

pytestmark = pytest.mark.unit

DEAD_PORT = 29071
MODEL = "urchade/gliner_small-v2.1"
QUERY = "Barack Obama in Chicago"
DEAD_URL = f"http://127.0.0.1:{DEAD_PORT}"


def _dead_port_extractor() -> GLiNERRelationshipExtractor:
    """An extractor whose inference service is not listening."""
    return GLiNERRelationshipExtractor(model_name=MODEL, inference_url=DEAD_URL)


class _PlainFailureExtractor:
    """An extractor that fails for a reason that is not an outage."""

    gliner_model = object()

    def extract_entities(self, text, labels=None):
        del text, labels
        raise RuntimeError("planned non-outage failure")

    def infer_relationships_from_entities(self, text, entities):
        del text, entities
        return []


class TestExtractorRaisesAtTheDeadPort:
    def test_dead_inference_port_raises_naming_the_endpoint(self):
        with pytest.raises(GLiNEREntityExtractionUnavailableError) as excinfo:
            _dead_port_extractor().extract_entities(QUERY)

        assert excinfo.value.model_name == MODEL
        assert excinfo.value.inference_url == DEAD_URL
        assert MODEL in str(excinfo.value)
        assert DEAD_URL in str(excinfo.value)


class TestQueryAnalysisNamesTheOutage:
    def _module(self, extractor):
        return ComposableQueryAnalysisModule(
            gliner_extractor=extractor, spacy_analyzer=SpaCyDependencyAnalyzer()
        )

    def test_outage_is_carried_on_the_prediction(self):
        prediction = self._module(_dead_port_extractor())(query=QUERY)

        assert prediction.fallback_reason == (
            ENTITY_EXTRACTION_FALLBACK_EXTRACTOR_UNAVAILABLE
        )
        assert prediction.fallback_model == MODEL
        assert prediction.fallback_inference_url == DEAD_URL
        assert prediction.path_used == "fallback"
        assert prediction.entities == []
        assert prediction.enhanced_query == QUERY

    def test_a_non_outage_failure_is_not_reported_as_one(self):
        """The control: the broad fallback leaves the reason unset."""
        prediction = self._module(_PlainFailureExtractor())(query=QUERY)

        assert prediction.fallback_reason is None
        assert prediction.fallback_model is None
        assert prediction.fallback_inference_url is None
        assert prediction.path_used == "fallback"


class TestRelationshipToolNamesTheOutage:
    def _tool(self, extractor):
        tool = RelationshipExtractorTool()
        tool.gliner_extractor = extractor
        return tool

    async def _run(self, extractor):
        return await self._tool(extractor).extract_comprehensive_relationships(QUERY)

    @pytest.mark.asyncio
    async def test_outage_is_carried_on_the_result(self):
        result = await self._run(_dead_port_extractor())

        assert result["fallback_reason"] == (
            ENTITY_EXTRACTION_FALLBACK_EXTRACTOR_UNAVAILABLE
        )
        assert result["fallback_model"] == MODEL
        assert result["fallback_inference_url"] == DEAD_URL
        assert result["entities"] == []
        assert result["relationships"] == []
        assert result["query_structure"] == "unknown"

    @pytest.mark.asyncio
    async def test_a_non_outage_failure_is_not_reported_as_one(self):
        result = await self._run(_PlainFailureExtractor())

        assert result["fallback_reason"] is None
        assert result["fallback_model"] is None
        assert result["query_structure"] == "unknown"


_STUB_SERVER = r"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

ENTITY = {"text": "Barack Obama", "label": "PERSON", "score": 0.9,
          "start": 0, "end": 12}

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps({"entities": [ENTITY]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

HTTPServer(("0.0.0.0", 8080), H).serve_forever()
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def paused_inference_peer():
    """A real HTTP peer for the inference URL, pausable mid-request.

    The contract under test is the client's, not the model's: a peer that
    accepts the connection and then stops answering must raise the typed
    outage rather than return an empty extraction.
    """
    port = _free_port()
    name = f"cv-gliner-peer-{os.getpid()}"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "-p",
            f"{port}:8080",
            "python:3.12-slim",
            "python",
            "-c",
            _STUB_SERVER,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    url = f"http://127.0.0.1:{port}"
    # Readiness is a served request, not an accepted connection: docker's port
    # proxy accepts before the server inside binds.
    probe = RemoteGlinerClient(url=url, model_name=MODEL, timeout=2.0)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            probe.predict_entities(QUERY, ["person"])
            break
        except Exception:
            time.sleep(0.5)
    else:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
        pytest.fail(f"inference peer never served a request on {url}")
    try:
        yield name, url
    finally:
        subprocess.run(["docker", "unpause", name], capture_output=True, check=False)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


class TestPausedInferencePeerIsAnOutage:
    TIMEOUT_S = 3.0

    def _extractor(self, url):
        extractor = GLiNERRelationshipExtractor(model_name=MODEL, inference_url=url)
        extractor.gliner_model = RemoteGlinerClient(
            url=url, model_name=MODEL, timeout=self.TIMEOUT_S
        )
        return extractor

    def test_live_peer_answers_then_paused_peer_raises(self, paused_inference_peer):
        name, url = paused_inference_peer
        extractor = self._extractor(url)

        # Control: the same client against the same live peer extracts.
        assert extractor.extract_entities(QUERY) == [
            {
                "text": "Barack Obama",
                "label": "PERSON",
                "confidence": 0.9,
                "start_pos": 0,
                "end_pos": 12,
            }
        ]

        subprocess.run(["docker", "pause", name], capture_output=True, check=True)

        started = time.monotonic()
        with pytest.raises(GLiNEREntityExtractionUnavailableError) as excinfo:
            extractor.extract_entities(QUERY)
        elapsed = time.monotonic() - started

        assert excinfo.value.inference_url == url
        assert excinfo.value.model_name == MODEL
        # The injected budget bounds the hang; the shipped default does not
        # apply to this call.
        assert elapsed < self.TIMEOUT_S * 3

    @pytest.mark.asyncio
    async def test_paused_peer_reaches_both_consumers_as_a_named_degrade(
        self, paused_inference_peer
    ):
        name, url = paused_inference_peer
        extractor = self._extractor(url)
        subprocess.run(["docker", "pause", name], capture_output=True, check=True)

        module = ComposableQueryAnalysisModule(
            gliner_extractor=extractor, spacy_analyzer=SpaCyDependencyAnalyzer()
        )
        prediction = module(query=QUERY)

        tool = RelationshipExtractorTool()
        tool.gliner_extractor = extractor
        result = await tool.extract_comprehensive_relationships(QUERY)

        assert prediction.fallback_reason == (
            ENTITY_EXTRACTION_FALLBACK_EXTRACTOR_UNAVAILABLE
        )
        assert prediction.fallback_inference_url == url
        assert result["fallback_reason"] == (
            ENTITY_EXTRACTION_FALLBACK_EXTRACTOR_UNAVAILABLE
        )
        assert result["fallback_inference_url"] == url
        assert json.loads(json.dumps(result))["query_structure"] == "unknown"
