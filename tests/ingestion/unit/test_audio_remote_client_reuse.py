"""Remote CLAP calls must reuse one httpx.Client per generator instance.

A bare httpx.post per segment rebuilds the connection pool (TCP + TLS
handshake) for every audio segment in a batch; the generator owns one
lazily created client instead and closes it via close().
"""

from __future__ import annotations

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
    gen = AudioEmbeddingGenerator(clap_endpoint_url="http://127.0.0.1:9")

    tone = np.zeros(1000, dtype=np.float32)
    for _ in range(3):
        vec = gen.generate_acoustic_embedding(audio_array=tone, sample_rate=48000)
        assert vec.dtype == np.float32
        assert vec.tolist() == [0.25] * 512
    text_vec = gen.generate_acoustic_text_embedding("rain on a tin roof")
    assert text_vec.tolist() == [0.25] * 512

    assert len(_FakeClient.instances) == 1
    client = _FakeClient.instances[0]
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
