"""Query classifier server and client contract over live HTTP.

The server is the real ``servers/query_classifier.py`` app in a child process
serving a checkpoint trained by the production trainer; the client is the
production ``RemoteQueryClassifierClient``.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
import requests
from cogniverse_cli.modal_inference.servers import query_classifier as server

from cogniverse_agents.gateway_agent import CLASSIFIER_HEADS
from cogniverse_core.common.models import query_classifier as loader
from cogniverse_core.common.models.query_classifier import (
    RemoteQueryClassifierClient,
    get_or_load_query_classifier,
)
from tests.utils.query_classifiers import GATEWAY_EXAMPLES, classify_in_child

pytestmark = [pytest.mark.unit]

FULL = [e for e in GATEWAY_EXAMPLES if set(e.labels) == set(CLASSIFIER_HEADS)]


@pytest.fixture(autouse=True)
def _fresh_loader_cache():
    loader._classifier_cache.clear()
    yield
    loader._classifier_cache.clear()


def test_classify_answers_the_trained_labels(query_classifier_service):
    url, model = query_classifier_service
    for example in FULL:
        response = httpx.post(
            f"{url}/classify", json={"text": example.text, "model": model}, timeout=30
        )
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"model", "heads", "predictions"}
        assert (body["model"], body["heads"]) == (model, CLASSIFIER_HEADS)
        assert {h: p["label"] for h, p in body["predictions"].items()} == example.labels

    health = httpx.get(f"{url}/health", timeout=10).json()
    assert health == {
        "status": "ready",
        "models": ["base", model],
        "loaded_models": [model],
    }


def test_remote_client_matches_the_in_process_classifier(
    query_classifier_service, gateway_classifier_checkpoint
):
    url, model = query_classifier_service
    client = RemoteQueryClassifierClient(url, model, timeout=30)
    texts = [e.text for e in FULL]

    remote = [client.classify(text) for text in texts]
    local = classify_in_child(gateway_classifier_checkpoint, texts)

    assert client.heads == CLASSIFIER_HEADS
    for got, want in zip(remote, local):
        assert {h: p["label"] for h, p in got.items()} == {
            h: p["label"] for h, p in want.items()
        }
        for head in CLASSIFIER_HEADS:
            assert got[head]["score"] == pytest.approx(want[head]["score"], abs=1e-6)


def test_unknown_checkpoint_is_a_503_naming_it(query_classifier_service):
    url, _ = query_classifier_service
    response = httpx.post(
        f"{url}/classify", json={"text": "cats", "model": "missing"}, timeout=10
    )
    assert response.status_code == 503
    assert response.json()["detail"].startswith(
        "query classifier missing load failed (FileNotFoundError):"
    )

    client = RemoteQueryClassifierClient(url, "missing", timeout=10)
    with pytest.raises(requests.HTTPError) as raised:
        client.classify("cats")
    assert raised.value.response.status_code == 503


@pytest.mark.parametrize("model", ["../gateway-classifier", "a/b", ".hidden"])
def test_a_model_outside_the_root_is_refused(query_classifier_service, model):
    url, _ = query_classifier_service
    response = httpx.post(
        f"{url}/classify", json={"text": "cats", "model": model}, timeout=10
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": f"model {model!r} must be a checkpoint directory name"
    }


def test_a_dead_service_raises_connection_error():
    client = RemoteQueryClassifierClient("http://127.0.0.1:9", "m", timeout=2)
    with pytest.raises(requests.ConnectionError):
        client.classify("cats")


class _Handler(BaseHTTPRequestHandler):
    body: dict = {}

    def do_POST(self) -> None:
        payload = json.dumps(self.body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args) -> None:
        return


@contextmanager
def _answering(body: dict):
    handler = type("Answer", (_Handler,), {"body": body})
    stub = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=stub.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{stub.server_port}"
    finally:
        stub.shutdown()
        stub.server_close()
        thread.join(timeout=5)


PREDICTIONS = {"complexity": {"label": "simple", "score": 0.9}}
HEADS = {"complexity": ["simple", "complex"]}


@pytest.mark.parametrize(
    "body, message",
    [
        ({"model": "m", "predictions": PREDICTIONS}, r"must be \{model, heads"),
        (
            {"model": "other", "heads": HEADS, "predictions": PREDICTIONS},
            "answered for model 'other', not 'm'",
        ),
        (
            {
                "model": "m",
                "heads": HEADS,
                "predictions": {"complexity": {"label": "odd", "score": 0.9}},
            },
            "label 'odd' is not one of",
        ),
    ],
)
def test_a_malformed_200_raises(body, message):
    with _answering(body) as url:
        client = RemoteQueryClassifierClient(url, "m", timeout=5)
        with pytest.raises(ValueError, match=message):
            client.classify("cats")


def test_concurrent_first_loads_build_one_client(monkeypatch):
    builds = []
    barrier = threading.Barrier(12)
    real_init = RemoteQueryClassifierClient.__init__

    def counting_init(self, *args, **kwargs):
        time.sleep(0.05)  # hold the build open so racing first touches overlap
        builds.append(threading.get_ident())
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(RemoteQueryClassifierClient, "__init__", counting_init)

    def first_touch(_):
        barrier.wait()
        return get_or_load_query_classifier(
            "gateway-classifier", inference_url="http://127.0.0.1:9", timeout=2.0
        )

    with ThreadPoolExecutor(max_workers=12) as pool:
        clients = list(pool.map(first_touch, range(12)))

    assert len(builds) == 1
    assert {id(client) for client in clients} == {id(clients[0])}


def test_a_missing_local_checkpoint_raises_with_its_path(tmp_path):
    with pytest.raises(RuntimeError) as raised:
        get_or_load_query_classifier(str(tmp_path / "nope"))
    assert str(raised.value).startswith(
        f"Failed to load query classifier {str(tmp_path / 'nope')!r} on cpu:"
    )
    assert loader._classifier_cache == {}


def test_server_loads_a_checkpoint_once_under_concurrent_first_requests(
    monkeypatch, tmp_path
):
    loads = []
    barrier = threading.Barrier(10)
    monkeypatch.setenv("MODEL_ROOT", str(tmp_path))
    monkeypatch.setattr(server, "_models", {})

    class _Loaded:
        heads = {"complexity": ["simple", "complex"]}

    def counting_load(path, device="cpu"):
        time.sleep(0.05)
        loads.append(str(path))
        return _Loaded()

    monkeypatch.setattr(server.QueryClassifier, "load", staticmethod(counting_load))

    def first_request(_):
        barrier.wait()
        return server._get_model("ckpt")

    with ThreadPoolExecutor(max_workers=10) as pool:
        models = list(pool.map(first_request, range(10)))

    assert loads == [str(tmp_path / "ckpt")]
    assert {id(m) for m in models} == {id(models[0])}
