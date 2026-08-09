"""Remote CLAP calls must reuse one httpx.Client per generator instance.

A bare httpx.post per segment rebuilds the connection pool (TCP + TLS
handshake) for every audio segment in a batch; the generator owns one
lazily created client instead and closes it via close().
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Condition, Event, Lock
from types import MappingProxyType, ModuleType

import numpy as np
import pytest

from cogniverse_runtime.ingestion.processors.audio_embedding_generator import (
    AudioEmbeddingGenerator,
)


class _FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"vec": [0.25] * 512}


class _FakeClient:
    instances: list["_FakeClient"] = []

    def __init__(self, *args, **kwargs):
        type(self).instances.append(self)
        self.headers = dict(kwargs.get("headers", {}))
        self.post_calls: list[tuple] = []
        self.closed = False

    def post(self, url, json=None, **kwargs):
        assert not self.closed
        self.post_calls.append((url, json))
        return _FakeResponse()

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def fake_httpx_client(monkeypatch):
    _FakeClient.instances = []
    monkeypatch.setattr("httpx.Client", _FakeClient)


def test_one_client_reused_across_segment_and_text_calls():
    gen = AudioEmbeddingGenerator(
        clap_endpoint_url="http://127.0.0.1:9",
        clap_headers={"Authorization": "Bearer modal-clap-key"},
    )

    tone = np.zeros(1000, dtype=np.float32)
    for _ in range(3):
        vec = gen.generate_acoustic_embedding(audio_array=tone, sample_rate=48000)
        assert vec.dtype == np.float32
        assert vec.tolist() == [0.25] * 512
    text_vec = gen.generate_acoustic_text_embedding("rain on a tin roof")
    assert text_vec.tolist() == [0.25] * 512

    assert len(_FakeClient.instances) == 1
    client = _FakeClient.instances[0]
    assert client.headers == {"Authorization": "Bearer modal-clap-key"}
    assert len(client.post_calls) == 4
    urls = [url for url, _ in client.post_calls]
    assert urls == ["http://127.0.0.1:9/embed/audio"] * 3 + [
        "http://127.0.0.1:9/embed/text"
    ]
    for _, payload in client.post_calls[:3]:
        assert set(payload) == {"audio_b64"}
    assert client.post_calls[3][1] == {"text": "rain on a tin roof"}


def test_close_shuts_client_and_next_call_builds_a_fresh_one():
    gen = AudioEmbeddingGenerator(clap_endpoint_url="http://127.0.0.1:9")

    gen.generate_acoustic_text_embedding("a dog barking")
    gen.close()
    assert len(_FakeClient.instances) == 1
    assert _FakeClient.instances[0].closed is True

    # close() is idempotent and the client rebuilds lazily on next use.
    gen.close()
    gen.generate_acoustic_text_embedding("a dog barking")
    assert len(_FakeClient.instances) == 2
    assert _FakeClient.instances[1].closed is False
    assert len(_FakeClient.instances[1].post_calls) == 1


@pytest.mark.parametrize(
    "headers",
    [
        {"Modal-Key": "wrong-scheme"},
        {"Authorization": "modal-clap-key"},
        {"Authorization": "Bearer "},
        {"Authorization": "Bearer modal-clap-key", "X-Extra": "rejected"},
    ],
)
def test_clap_headers_reject_every_noncanonical_shape(headers):
    with pytest.raises(ValueError, match="clap_headers.*Authorization"):
        AudioEmbeddingGenerator(
            clap_endpoint_url="https://clap.modal.run",
            clap_headers=headers,
        )


def test_clap_headers_require_remote_endpoint():
    with pytest.raises(ValueError, match="clap_headers requires clap_endpoint_url"):
        AudioEmbeddingGenerator(clap_headers={"Authorization": "Bearer modal-clap-key"})


def test_modal_endpoint_requires_environment_credential(monkeypatch):
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)

    with pytest.raises(
        RuntimeError,
        match="Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY",
    ):
        AudioEmbeddingGenerator(clap_endpoint_url="https://clap.modal.run")


def test_modal_endpoint_rejects_caller_supplied_headers(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "shared-production-key")

    for headers in ({"Authorization": "Bearer caller-specific-key"}, {}):
        with pytest.raises(ValueError, match="clap_headers.*Modal"):
            AudioEmbeddingGenerator(
                clap_endpoint_url="https://clap.modal.run",
                clap_headers=headers,
            )


def test_resolved_modal_headers_do_not_read_rotated_environment(monkeypatch):
    initial_headers = MappingProxyType(
        {"Authorization": "Bearer initial-production-key"}
    )
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "rotated-production-key")

    generator = AudioEmbeddingGenerator(
        clap_endpoint_url="https://clap.modal.run",
        _resolved_headers=initial_headers,
    )
    generator.generate_acoustic_text_embedding("rain on a tin roof")

    assert _FakeClient.instances[0].headers == {
        "Authorization": "Bearer initial-production-key"
    }


def test_resolved_headers_and_public_headers_are_mutually_exclusive():
    with pytest.raises(
        ValueError, match="clap_headers and _resolved_headers are mutually exclusive"
    ):
        AudioEmbeddingGenerator(
            clap_endpoint_url="http://127.0.0.1:9",
            clap_headers={"Authorization": "Bearer caller-key"},
            _resolved_headers=MappingProxyType(
                {"Authorization": "Bearer resolved-key"}
            ),
        )


def test_concurrent_local_clap_access_builds_one_model_processor_pair(monkeypatch):
    caller_count = 12
    start = Barrier(caller_count + 1)
    release_model = Event()
    calls_changed = Condition(Lock())
    model_calls = 0
    processor_calls = 0
    eval_calls = 0

    class LoadedModel:
        def eval(self):
            nonlocal eval_calls
            eval_calls += 1

    class ClapModel:
        @staticmethod
        def from_pretrained(model_name):
            nonlocal model_calls
            assert model_name == "laion/clap-htsat-unfused"
            with calls_changed:
                model_calls += 1
                calls_changed.notify_all()
            assert release_model.wait(timeout=5)
            return LoadedModel()

    class ClapProcessor:
        @staticmethod
        def from_pretrained(model_name):
            nonlocal processor_calls
            assert model_name == "laion/clap-htsat-unfused"
            processor_calls += 1
            return object()

    transformers = ModuleType("transformers")
    transformers.ClapModel = ClapModel
    transformers.ClapProcessor = ClapProcessor
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    generator = AudioEmbeddingGenerator()

    def load_component(index):
        start.wait()
        if index % 2:
            return "processor", generator.clap_processor
        return "model", generator.clap_model

    with ThreadPoolExecutor(max_workers=caller_count) as executor:
        futures = [
            executor.submit(load_component, index) for index in range(caller_count)
        ]
        start.wait()
        with calls_changed:
            calls_changed.wait_for(lambda: model_calls == caller_count, timeout=0.5)
        release_model.set()
        results = [future.result(timeout=5) for future in futures]

    models = [value for kind, value in results if kind == "model"]
    processors = [value for kind, value in results if kind == "processor"]
    assert model_calls == 1
    assert processor_calls == 1
    assert eval_calls == 1
    assert all(model is models[0] for model in models)
    assert all(processor is processors[0] for processor in processors)


def test_failed_clap_processor_build_publishes_neither_half(monkeypatch):
    model_calls = 0
    processor_calls = 0
    loaded_models = []
    loaded_processors = []

    class LoadedModel:
        def __init__(self, generation):
            self.generation = generation
            self.evaluated = False

        def eval(self):
            self.evaluated = True

    class ClapModel:
        @staticmethod
        def from_pretrained(model_name):
            nonlocal model_calls
            assert model_name == "laion/clap-htsat-unfused"
            model_calls += 1
            model = LoadedModel(model_calls)
            loaded_models.append(model)
            return model

    class ClapProcessor:
        @staticmethod
        def from_pretrained(model_name):
            nonlocal processor_calls
            assert model_name == "laion/clap-htsat-unfused"
            processor_calls += 1
            if processor_calls == 1:
                raise RuntimeError("processor build failed")
            processor = object()
            loaded_processors.append(processor)
            return processor

    transformers = ModuleType("transformers")
    transformers.ClapModel = ClapModel
    transformers.ClapProcessor = ClapProcessor
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    generator = AudioEmbeddingGenerator()
    with pytest.raises(RuntimeError, match="processor build failed"):
        _ = generator.clap_model

    recovered_model = generator.clap_model
    recovered_processor = generator.clap_processor

    assert model_calls == 2
    assert processor_calls == 2
    assert recovered_model is loaded_models[1]
    assert recovered_model.generation == 2
    assert recovered_model.evaluated is True
    assert recovered_processor is loaded_processors[0]


def test_concurrent_semantic_access_builds_one_embedder(monkeypatch):
    caller_count = 12
    start = Barrier(caller_count + 1)
    release_factory = Event()
    calls_changed = Condition(Lock())
    factory_calls = 0

    def build_embedder(*, model_name):
        nonlocal factory_calls
        assert model_name == "sentence-transformers/all-mpnet-base-v2"
        with calls_changed:
            factory_calls += 1
            calls_changed.notify_all()
        assert release_factory.wait(timeout=5)
        return object()

    monkeypatch.setattr(
        "cogniverse_runtime.ingestion.processors.audio_embedding_generator.get_semantic_embedder",
        build_embedder,
    )
    generator = AudioEmbeddingGenerator(
        semantic_model="sentence-transformers/all-mpnet-base-v2"
    )

    def load_embedder():
        start.wait()
        return generator.semantic_model

    with ThreadPoolExecutor(max_workers=caller_count) as executor:
        futures = [executor.submit(load_embedder) for _ in range(caller_count)]
        start.wait()
        with calls_changed:
            calls_changed.wait_for(lambda: factory_calls == caller_count, timeout=0.5)
        release_factory.set()
        embedders = [future.result(timeout=5) for future in futures]

    assert factory_calls == 1
    assert all(embedder is embedders[0] for embedder in embedders)


def test_failed_concurrent_semantic_build_retries_once(monkeypatch):
    caller_count = 12
    start = Barrier(caller_count + 1)
    first_call_entered = Event()
    release_first_call = Event()
    calls_lock = Lock()
    factory_calls = 0
    successful_embedders = []

    def build_embedder(*, model_name):
        nonlocal factory_calls
        assert model_name is None
        with calls_lock:
            factory_calls += 1
            call_number = factory_calls
        if call_number == 1:
            first_call_entered.set()
            assert release_first_call.wait(timeout=5)
            raise RuntimeError("semantic build failed")
        embedder = object()
        successful_embedders.append(embedder)
        return embedder

    monkeypatch.setattr(
        "cogniverse_runtime.ingestion.processors.audio_embedding_generator.get_semantic_embedder",
        build_embedder,
    )
    generator = AudioEmbeddingGenerator()

    def load_embedder():
        start.wait()
        return generator.semantic_model

    with ThreadPoolExecutor(max_workers=caller_count) as executor:
        futures = [executor.submit(load_embedder) for _ in range(caller_count)]
        start.wait()
        assert first_call_entered.wait(timeout=5)
        release_first_call.set()
        successes = []
        failures = []
        for future in futures:
            try:
                successes.append(future.result(timeout=5))
            except RuntimeError as exc:
                failures.append(str(exc))

    assert failures == ["semantic build failed"]
    assert len(successes) == caller_count - 1
    assert factory_calls == 2
    assert len(successful_embedders) == 1
    assert all(embedder is successful_embedders[0] for embedder in successes)
