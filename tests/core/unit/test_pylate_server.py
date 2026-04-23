"""Contract test for the PyLate remote inference FastAPI server.

Loads the server module from ``deploy/pylate/``, stubs the PyLate ColBERT
class, and verifies the ``/pooling`` response shape matches what
``cogniverse_core.common.models.model_loaders.RemoteColBERTLoader`` parses
when calling a vLLM-compatible endpoint.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

SERVER_DIR = Path(__file__).resolve().parents[3] / "deploy" / "pylate"
SERVER_PATH = SERVER_DIR / "server.py"


class _FakeColBERT:
    """Stand-in for pylate.models.ColBERT used only during tests."""

    def __init__(self, model_name_or_path: str, device: str = "cpu") -> None:
        self.model_name = model_name_or_path
        self.device = device

    def encode(
        self,
        texts: list[str],
        is_query: bool = False,
        show_progress_bar: bool = False,
    ) -> list[np.ndarray]:
        token_count = 3 if is_query else 5
        return [
            np.full((token_count, 128), float(i) + (0.1 if is_query else 0.2), dtype=np.float32)
            for i, _ in enumerate(texts)
        ]


@pytest.fixture
def server_app(monkeypatch):
    """Load server.py with pylate.models mocked, return (app, fake_model_name)."""
    fake_pylate = types.ModuleType("pylate")
    fake_models = types.ModuleType("pylate.models")
    fake_models.ColBERT = _FakeColBERT
    fake_pylate.models = fake_models
    monkeypatch.setitem(sys.modules, "pylate", fake_pylate)
    monkeypatch.setitem(sys.modules, "pylate.models", fake_models)

    spec = importlib.util.spec_from_file_location("pylate_inference_server", SERVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    app = module.build_app("lightonai/LateOn", "cpu")
    return app, "lightonai/LateOn"


def test_pooling_response_matches_remote_colbert_loader_expectation(server_app):
    app, model_name = server_app
    client = TestClient(app)

    resp = client.post(
        "/pooling",
        json={"input": ["hello world", "second text"], "model": model_name, "is_query": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["model"] == model_name
    assert len(body["data"]) == 2

    for idx, item in enumerate(body["data"]):
        assert item["index"] == idx
        embedding = item["data"]
        assert len(embedding) == 5
        assert all(len(row) == 128 for row in embedding)


def test_pooling_honours_is_query_flag(server_app):
    app, _ = server_app
    client = TestClient(app)

    doc_resp = client.post("/pooling", json={"input": ["x"], "is_query": False})
    query_resp = client.post("/pooling", json={"input": ["x"], "is_query": True})

    assert len(doc_resp.json()["data"][0]["data"]) == 5
    assert len(query_resp.json()["data"][0]["data"]) == 3


def test_pooling_rejects_empty_input(server_app):
    app, _ = server_app
    client = TestClient(app)
    resp = client.post("/pooling", json={"input": []})
    assert resp.status_code == 400


def test_health_endpoint_reports_model(server_app):
    app, model_name = server_app
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": model_name}
