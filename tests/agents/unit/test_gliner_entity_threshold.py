"""One entity threshold, whichever way GLiNER is reached.

``GLiNERRelationshipExtractor`` sends ``GLINER_ENTITY_THRESHOLD`` to the
in-process model and to the inference service alike; the service's own request
default and ``RemoteGlinerClient``'s are the same value.
"""

from __future__ import annotations

import inspect
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from cogniverse_agents.routing.relationship_extraction_tools import (
    DEFAULT_GLINER_MODEL,
    GLiNERRelationshipExtractor,
    RelationshipExtractorTool,
    create_relationship_extractor,
)
from cogniverse_core.common import models
from cogniverse_core.common.models import GLINER_ENTITY_THRESHOLD
from cogniverse_core.common.models.model_loaders import RemoteGlinerClient

pytestmark = pytest.mark.unit

TEXT = "Barack Obama in Chicago"
_UNSET = object()


@contextmanager
def _gliner_service():
    """A real HTTP GLiNER endpoint that records each request body."""
    bodies: list[dict] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                bodies.append(body)
            payload = json.dumps(
                {
                    "entities": [
                        {
                            "text": "Barack Obama",
                            "label": "PERSON",
                            "score": 0.99,
                            "start": 0,
                            "end": 12,
                        }
                    ],
                    "model": DEFAULT_GLINER_MODEL,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", bodies
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _InProcessGLiNER:
    """Stands in for ``gliner.GLiNER``; records the threshold it was given."""

    def __init__(self):
        self.thresholds: list[object] = []

    def predict_entities(self, text, labels, threshold=_UNSET):
        self.thresholds.append(threshold)
        return [
            {
                "text": "Chicago",
                "label": "LOCATION",
                "score": 0.99,
                "start": 16,
                "end": 23,
            }
        ]


@pytest.fixture
def in_process_gliner(monkeypatch):
    model = _InProcessGLiNER()

    served = models.get_or_load_gliner

    def load(model_name, logger=None, inference_url=None, **kwargs):
        if inference_url is not None:
            return served(model_name, logger=logger, inference_url=inference_url)
        return model

    monkeypatch.setattr(models, "get_or_load_gliner", load)
    return model


def test_the_threshold_is_the_one_the_service_serves():
    from cogniverse_cli.modal_inference.servers.gliner import PredictRequest

    assert GLINER_ENTITY_THRESHOLD == 0.4
    assert PredictRequest.model_fields["threshold"].default == GLINER_ENTITY_THRESHOLD
    assert (
        inspect.signature(RemoteGlinerClient.predict_entities)
        .parameters["threshold"]
        .default
        == GLINER_ENTITY_THRESHOLD
    )


def test_in_process_and_served_paths_use_the_same_threshold(in_process_gliner):
    with _gliner_service() as (url, bodies):
        served = GLiNERRelationshipExtractor(inference_url=url).extract_entities(TEXT)
    in_process = GLiNERRelationshipExtractor().extract_entities(TEXT)

    assert [body["threshold"] for body in bodies] == [GLINER_ENTITY_THRESHOLD]
    assert in_process_gliner.thresholds == [GLINER_ENTITY_THRESHOLD]
    assert [entity["text"] for entity in served + in_process] == [
        "Barack Obama",
        "Chicago",
    ]


def test_a_caller_threshold_reaches_both_paths(in_process_gliner):
    with _gliner_service() as (url, bodies):
        GLiNERRelationshipExtractor(inference_url=url, threshold=0.25).extract_entities(
            TEXT
        )
    GLiNERRelationshipExtractor(threshold=0.25).extract_entities(TEXT)

    assert [body["threshold"] for body in bodies] == [0.25]
    assert in_process_gliner.thresholds == [0.25]


def test_the_relationship_tool_reaches_gliner_through_the_service():
    with _gliner_service() as (url, bodies):
        tool = RelationshipExtractorTool(gliner_inference_url=url)
        entities = tool.gliner_extractor.extract_entities(TEXT)
        made = create_relationship_extractor(gliner_inference_url=url)

    assert tool.gliner_extractor.inference_url == url
    assert made.gliner_extractor.inference_url == url
    assert [(body["text"], body["threshold"]) for body in bodies] == [
        (TEXT, GLINER_ENTITY_THRESHOLD)
    ]
    assert [entity["text"] for entity in entities] == ["Barack Obama"]
