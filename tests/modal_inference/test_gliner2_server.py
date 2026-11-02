"""The GLiNER2 inference server and its client, against a tiny real checkpoint.

The server runs in a child process (``tests/fixtures/gliner2.py``) serving the
production boundary architecture; what it must answer is what the checkpoint
computes when loaded directly in another child.
"""

from __future__ import annotations

import threading

import httpx
import pytest
import requests
from cogniverse_cli.modal_inference.servers import gliner2 as server

from cogniverse_agents.gateway_agent import SHADOW_DECISIONS, SHADOW_ENTITY_TYPES
from cogniverse_core.common.models import gliner2 as loader
from cogniverse_foundation.inference_specs import get_inference_service_spec
from tests.utils.gliner2_models import QUERIES, extract_in_child, gliner2_server

pytestmark = [pytest.mark.unit]


def _body(model: str, text: str, **overrides) -> dict:
    return {
        "model": model,
        "text": text,
        "entity_types": SHADOW_ENTITY_TYPES,
        "classifications": SHADOW_DECISIONS,
        "threshold": 0.5,
        **overrides,
    }


@pytest.fixture(scope="module")
def computed(gliner2_service, tiny_gliner2_checkpoint):
    _, model = gliner2_service
    bodies = [_body(model, query) for query in QUERIES]
    return dict(zip(QUERIES, extract_in_child(tiny_gliner2_checkpoint, bodies)))


@pytest.fixture(autouse=True)
def _fresh_client_cache():
    loader._client_cache.clear()
    yield
    loader._client_cache.clear()


def test_the_served_model_is_the_pinned_spec_checkpoint():
    spec = get_inference_service_spec("gliner2")
    assert (server.MODEL_ID, server.MODEL_REVISION) == (
        "fastino/gliner2.5-base-v1",
        "54785d51df8d86a0d8f2eb212fec15871aad2c4d",
    )
    assert (spec.model_id, spec.model_revision) == (
        server.MODEL_ID,
        server.MODEL_REVISION,
    )
    assert server.checkpoint_dir("/models/local-gliner2") == "/models/local-gliner2"


def test_health_names_the_served_checkpoint(gliner2_service):
    url, model = gliner2_service
    assert httpx.get(f"{url}/health", timeout=30).json() == {
        "status": "ready",
        "model": model,
        "model_revision": None,
    }


def test_extract_answers_what_the_checkpoint_computes(gliner2_service, computed):
    url, model = gliner2_service
    for query in QUERIES:
        resp = httpx.post(f"{url}/extract", json=_body(model, query), timeout=30)
        assert resp.status_code == 200
        assert resp.json() == computed[query]


def test_the_client_returns_the_server_answer_without_the_model_key(
    gliner2_service, computed
):
    url, model = gliner2_service
    client = loader.get_or_load_gliner2(model, inference_url=url, timeout=30)
    for query in QUERIES:
        result = client.extract(
            query, SHADOW_ENTITY_TYPES, SHADOW_DECISIONS, threshold=0.5
        )
        assert result == {
            "entities": computed[query]["entities"],
            "classifications": computed[query]["classifications"],
        }
        assert all(
            query[e["start"] : e["end"]] == e["text"] for e in result["entities"]
        )


def test_spans_in_the_appended_period_map_onto_the_request_text():
    """GLiNER2 appends '.' to text without closing punctuation; spans reaching
    into it are clipped to the request text, and one alone is dropped."""
    req = server.ExtractRequest(**_body(server._SERVED_MODEL, "summarize the podcast"))
    raw = {
        "entities": {
            "EVENT": [
                {"text": "podcast.", "confidence": 0.76, "start": 14, "end": 22},
                {"text": "podcast", "confidence": 0.61, "start": 14, "end": 21},
            ],
            "PERSON": [{"text": ".", "confidence": 0.55, "start": 21, "end": 22}],
            "PLACE": [{"text": "podcast", "confidence": 0.65, "start": 14, "end": 21}],
        },
        "complexity": {"label": "simple", "confidence": 0.7},
        "modality": {"label": "audio", "confidence": 0.6},
        "generation_type": {"label": "summary", "confidence": 0.8},
    }

    assert server.to_response(raw, req).model_dump() == {
        "model": server._SERVED_MODEL,
        "entities": [
            {"text": "podcast", "label": "EVENT", "score": 0.76, "start": 14, "end": 21},
            {"text": "podcast", "label": "PLACE", "score": 0.65, "start": 14, "end": 21},
        ],
        "classifications": {
            "complexity": {"label": "simple", "score": 0.7},
            "modality": {"label": "audio", "score": 0.6},
            "generation_type": {"label": "summary", "score": 0.8},
        },
    }  # fmt: skip


def test_a_span_that_is_not_in_the_text_fails_the_request():
    req = server.ExtractRequest(**_body(server._SERVED_MODEL, "find cat videos"))
    raw = {
        "entities": {"PLACE": [{"text": "dog", "confidence": 0.9, "start": 5, "end": 8}]},
        **{task: {"label": next(iter(labels)), "confidence": 0.5}
           for task, labels in SHADOW_DECISIONS.items()},
    }  # fmt: skip
    with pytest.raises(ValueError) as excinfo:
        server.to_response(raw, req)
    assert str(excinfo.value) == ("span 'dog' at [5, 8) is not in the request text")


def test_a_classification_without_a_requested_label_fails_the_request():
    req = server.ExtractRequest(**_body(server._SERVED_MODEL, "find cat videos"))
    raw = {
        "entities": {},
        "complexity": {"label": "simple", "confidence": 0.7},
        "modality": None,
        "generation_type": {"label": "summary", "confidence": 0.8},
    }
    with pytest.raises(ValueError) as excinfo:
        server.to_response(raw, req)
    assert str(excinfo.value) == "classification 'modality' returned None"


def test_requests_for_another_model_or_a_one_label_task_are_rejected(
    gliner2_service,
):
    url, model = gliner2_service
    wrong = httpx.post(
        f"{url}/extract", json=_body("fastino/other", "find cats"), timeout=30
    )
    assert wrong.status_code == 422
    assert wrong.json()["detail"][0]["msg"] == (
        f"Value error, model must equal {model}"
    )
    one_label = httpx.post(
        f"{url}/extract",
        json=_body(model, "find cats", classifications={"complexity": {"simple": ""}}),
        timeout=30,
    )
    assert one_label.status_code == 422
    assert one_label.json()["detail"][0]["msg"] == (
        "Value error, classification 'complexity' needs at least two labels"
    )


def test_a_checkpoint_that_cannot_load_answers_503_with_the_cause(tmp_path):
    missing = tmp_path / "missing-checkpoint"
    tmp_path.joinpath("missing-checkpoint").mkdir()
    with gliner2_server(missing, log_dir=tmp_path, wait_healthy=False) as url:
        health = httpx.get(f"{url}/health", timeout=60)
        extract = httpx.post(
            f"{url}/extract", json=_body(str(missing), "find cats"), timeout=60
        )

    assert health.status_code == 503
    assert health.json()["detail"].startswith(f"gliner2: model {missing} load failed (")
    assert extract.status_code == 503
    assert extract.json()["detail"].startswith(
        f"gliner2: model {missing} load failed ("
    )


def test_a_server_error_raises_with_the_endpoint(tmp_path):
    missing = tmp_path / "missing-checkpoint"
    missing.mkdir()
    with gliner2_server(missing, log_dir=tmp_path, wait_healthy=False) as url:
        client = loader.RemoteGliner2Client(url, str(missing), timeout=60)
        with pytest.raises(requests.HTTPError) as excinfo:
            client.extract(
                "find cats", SHADOW_ENTITY_TYPES, SHADOW_DECISIONS, threshold=0.5
            )
    assert str(excinfo.value).startswith(
        f"503 Server Error: Service Unavailable for url: {url}/extract"
    )


def test_concurrent_requests_each_get_their_own_answer(gliner2_service, computed):
    url, model = gliner2_service
    client = loader.get_or_load_gliner2(model, inference_url=url, timeout=30)
    queries = list(QUERIES) * 4
    barrier = threading.Barrier(len(queries))
    results: dict[int, dict] = {}

    def run(index: int, query: str) -> None:
        barrier.wait()
        results[index] = client.extract(
            query, SHADOW_ENTITY_TYPES, SHADOW_DECISIONS, threshold=0.5
        )

    threads = [threading.Thread(target=run, args=(i, q)) for i, q in enumerate(queries)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert [results[i] for i in range(len(queries))] == [
        {
            "entities": computed[q]["entities"],
            "classifications": computed[q]["classifications"],
        }
        for q in queries
    ]


def test_concurrent_first_calls_build_one_client(monkeypatch):
    builds = []
    real_init = loader.RemoteGliner2Client.__init__

    def counting_init(self, *args, **kwargs):
        threading.Event().wait(0.05)  # hold the build open so first calls overlap
        builds.append(args)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(loader.RemoteGliner2Client, "__init__", counting_init)
    barrier = threading.Barrier(8)
    clients = []

    def first_call() -> None:
        barrier.wait()
        clients.append(
            loader.get_or_load_gliner2("m", inference_url="http://gliner2:8080")
        )

    threads = [threading.Thread(target=first_call) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(builds) == 1
    assert len({id(c) for c in clients}) == 1
    assert len(clients) == 8


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.pop("classifications"),
         "R must be {model, entities, classifications}"),
        (lambda d: d["entities"][0].update(label="ANIMAL"),
         "R entity 0 label 'ANIMAL' was not requested"),
        (lambda d: d["entities"][0].update(end=99),
         "R entity 0 text 'cat' is not the query at [8, 99)"),
        (lambda d: d["entities"][0].update(score=1.5),
         "R entity 0 score 1.5 must be in [0, 1]"),
        (lambda d: d["classifications"].pop("modality"),
         "R classifications must answer exactly "
         "['complexity', 'generation_type', 'modality']"),
        (lambda d: d["classifications"]["modality"].update(label="smell"),
         "R classification 'modality' label 'smell' is not one of "
         "['video', 'text', 'audio', 'image', 'document', 'both']"),
    ],
)  # fmt: skip
def test_a_malformed_response_is_rejected_with_what_is_wrong(mutate, message):
    data = {
        "model": "m",
        "entities": [
            {"text": "cat", "label": "CONCEPT", "score": 0.8, "start": 8, "end": 11}
        ],
        "classifications": {
            "complexity": {"label": "simple", "score": 0.9},
            "modality": {"label": "video", "score": 0.8},
            "generation_type": {"label": "raw_results", "score": 0.7},
        },
    }
    assert loader.validate_gliner2_result(
        data, "show me cat videos", SHADOW_ENTITY_TYPES, SHADOW_DECISIONS, "R"
    ) == {"entities": data["entities"], "classifications": data["classifications"]}
    mutate(data)
    with pytest.raises(ValueError) as excinfo:
        loader.validate_gliner2_result(
            data, "show me cat videos", SHADOW_ENTITY_TYPES, SHADOW_DECISIONS, "R"
        )
    assert str(excinfo.value) == message
