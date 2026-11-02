"""The real GLiNER2 inference server with a stand-in model, for tests.

Tests never load model weights on this host (``tests/fixtures/
no_local_models.py``); the served checkpoint's behaviour belongs to the
cluster. :class:`ExactGliner2Model` answers in the shape ``gliner2``'s
``extract(..., include_confidence=True, include_spans=True)`` returns, for
the exact schema the gateway's shadow builds, and records every call. Its
answers include spans in the ``.`` GLiNER2 appends to text without closing
punctuation, as the real model emits them.
"""

from __future__ import annotations

import copy
import socket
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator

import uvicorn
from cogniverse_cli.modal_inference.servers import gliner2 as server

from cogniverse_agents.gateway_agent import SHADOW_DECISIONS, SHADOW_ENTITY_TYPES


def _span(text: str, confidence: float, start: int, end: int) -> dict:
    return {"text": text, "confidence": confidence, "start": start, "end": end}


def _choice(label: str, confidence: float) -> dict:
    return {"label": label, "confidence": confidence}


def _raw(entities: dict[str, list[dict]], **decisions: dict) -> dict:
    return {
        "entities": {label: entities.get(label, []) for label in SHADOW_ENTITY_TYPES},
        **decisions,
    }


# The model's answer per query, as gliner2 formats it.
RAW_ANSWERS: dict[str, dict] = {
    "show me cat videos": _raw(
        {"CONCEPT": [_span("cat", 0.71, 8, 11)]},
        complexity=_choice("simple", 0.93),
        modality=_choice("video", 0.88),
        generation_type=_choice("raw_results", 0.9),
    ),
    "find Messi videos in Barcelona": _raw(
        {
            "PERSON": [_span("Messi", 0.97, 5, 10)],
            # The span GLiNER2 reads with its appended period, and the period
            # alone: the server maps the first onto "Barcelona" and drops the
            # second.
            "PLACE": [
                _span("Barcelona.", 0.94, 21, 31),
                _span("Barcelona", 0.92, 21, 30),
                _span(".", 0.51, 30, 31),
            ],
        },
        complexity=_choice("simple", 0.81),
        modality=_choice("video", 0.95),
        generation_type=_choice("raw_results", 0.77),
    ),
    "compare reports on rockets from NASA and SpaceX": _raw(
        {
            "ORGANIZATION": [
                _span("NASA", 0.96, 32, 36),
                _span("SpaceX", 0.95, 41, 47),
            ],
            "TECHNOLOGY": [_span("rockets", 0.66, 19, 26)],
        },
        complexity=_choice("complex", 0.89),
        modality=_choice("document", 0.62),
        generation_type=_choice("detailed_report", 0.84),
    ),
    "summarize the podcast": _raw(
        {},
        complexity=_choice("simple", 0.74),
        modality=_choice("audio", 0.91),
        generation_type=_choice("summary", 0.96),
    ),
}

QUERIES = tuple(RAW_ANSWERS)

# The /extract answer the server owes for each query.
SERVED_ANSWERS: dict[str, dict] = {
    "show me cat videos": {
        "entities": [
            {"text": "cat", "label": "CONCEPT", "score": 0.71, "start": 8, "end": 11}
        ],
        "classifications": {
            "complexity": {"label": "simple", "score": 0.93},
            "modality": {"label": "video", "score": 0.88},
            "generation_type": {"label": "raw_results", "score": 0.9},
        },
    },
    "find Messi videos in Barcelona": {
        "entities": [
            {"text": "Messi", "label": "PERSON", "score": 0.97, "start": 5, "end": 10},
            {"text": "Barcelona", "label": "PLACE", "score": 0.94, "start": 21,
             "end": 30},
        ],
        "classifications": {
            "complexity": {"label": "simple", "score": 0.81},
            "modality": {"label": "video", "score": 0.95},
            "generation_type": {"label": "raw_results", "score": 0.77},
        },
    },
    "compare reports on rockets from NASA and SpaceX": {
        "entities": [
            {"text": "rockets", "label": "TECHNOLOGY", "score": 0.66, "start": 19,
             "end": 26},
            {"text": "NASA", "label": "ORGANIZATION", "score": 0.96, "start": 32,
             "end": 36},
            {"text": "SpaceX", "label": "ORGANIZATION", "score": 0.95, "start": 41,
             "end": 47},
        ],
        "classifications": {
            "complexity": {"label": "complex", "score": 0.89},
            "modality": {"label": "document", "score": 0.62},
            "generation_type": {"label": "detailed_report", "score": 0.84},
        },
    },
    "summarize the podcast": {
        "entities": [],
        "classifications": {
            "complexity": {"label": "simple", "score": 0.74},
            "modality": {"label": "audio", "score": 0.91},
            "generation_type": {"label": "summary", "score": 0.96},
        },
    },
}  # fmt: skip


def shadow_schema() -> dict:
    """The gliner2 schema the shadow's request builds, as ``Schema.build()``."""
    from gliner2 import Schema

    schema = Schema().entities(SHADOW_ENTITY_TYPES)
    for task, labels in SHADOW_DECISIONS.items():
        schema = schema.classification(task, labels)
    return schema.build()


class ExactGliner2Model:
    """Answers :data:`RAW_ANSWERS` for the shadow's exact schema."""

    def __init__(self, answers: dict[str, dict] = RAW_ANSWERS) -> None:
        self._answers = answers
        self._schema = shadow_schema()
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def create_schema(self):
        from gliner2 import Schema

        return Schema()

    def extract(self, text, schema, **options):
        with self._lock:
            self.calls.append({"text": text, "options": options})
        assert schema.build() == self._schema
        assert options == {
            "threshold": 0.5,
            "include_confidence": True,
            "include_spans": True,
        }
        return copy.deepcopy(self._answers[text])


@contextmanager
def gliner2_server(model: Any) -> Iterator[str]:
    """The real GLiNER2 server app, serving ``model``, on a local port."""
    previous = server._model
    server._model = model
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    uv = uvicorn.Server(
        uvicorn.Config(
            server.app, host="127.0.0.1", port=port, log_level="error", ws="none"
        )
    )
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not uv.started:
        time.sleep(0.01)
    try:
        if not uv.started:
            raise RuntimeError("GLiNER2 server did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        uv.should_exit = True
        thread.join(timeout=5)
        server._model = previous
        if thread.is_alive():
            raise RuntimeError("GLiNER2 server did not stop")
