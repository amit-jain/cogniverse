"""Tests for the pluggable semantic embedder factory."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cogniverse_core.common.models.semantic_embedder import (
    LocalSentenceTransformerEmbedder,
    RemoteOpenAIEmbedder,
    get_semantic_embedder,
    reset_semantic_embedder_cache,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    reset_semantic_embedder_cache()
    yield
    reset_semantic_embedder_cache()


@pytest.mark.unit
@pytest.mark.ci_fast
def test_remote_url_arg_selects_remote_backend():
    embedder = get_semantic_embedder(remote_url="http://fake.invalid:11434")
    assert isinstance(embedder, RemoteOpenAIEmbedder)


def test_env_var_selects_remote_backend(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_SEMANTIC_EMBED_URL", "http://fake.invalid:11434")
    embedder = get_semantic_embedder()
    assert isinstance(embedder, RemoteOpenAIEmbedder)


def test_no_url_falls_back_to_local(monkeypatch):
    monkeypatch.delenv("COGNIVERSE_SEMANTIC_EMBED_URL", raising=False)
    with patch("cogniverse_core.common.models.semantic_embedder.SemanticEmbedder"):
        with patch("sentence_transformers.SentenceTransformer") as MockST:
            MockST.return_value = MagicMock(name="local-st")
            embedder = get_semantic_embedder()
    assert isinstance(embedder, LocalSentenceTransformerEmbedder)
    MockST.assert_called_once()


def test_instances_cached_by_backend_and_model():
    a = get_semantic_embedder(remote_url="http://fake.invalid:11434")
    b = get_semantic_embedder(remote_url="http://fake.invalid:11434")
    assert a is b

    c = get_semantic_embedder(
        remote_url="http://fake.invalid:11434", model_name="other-model"
    )
    assert c is not a


def _openai_embed_response(vectors: list[list[float]]):
    """Build a MagicMock response matching the OpenAI /v1/embeddings shape."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"embedding": v, "index": i, "object": "embedding"}
            for i, v in enumerate(vectors)
        ],
        "model": "test-model",
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }
    mock_response.raise_for_status.return_value = None
    return mock_response


@pytest.mark.unit
@pytest.mark.ci_fast
def test_remote_encode_hits_v1_embeddings():
    embedder = RemoteOpenAIEmbedder("http://fake.invalid:8000/", "lightonai/DenseOn")

    mock_response = _openai_embed_response([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    with patch.object(
        embedder._session, "post", return_value=mock_response
    ) as mock_post:
        result = embedder.encode(["hello", "world"])

    # Trailing slash stripped, OpenAI-compatible endpoint
    called_url = mock_post.call_args.args[0]
    assert called_url == "http://fake.invalid:8000/v1/embeddings"

    payload = mock_post.call_args.kwargs["json"]
    assert payload["model"] == "lightonai/DenseOn"
    # Documents (default) carry the DenseOn "document: " prompt prefix.
    assert payload["input"] == ["document: hello", "document: world"]

    assert result.shape == (2, 3)
    assert result.dtype == np.float32


def test_remote_encode_preserves_order_when_backend_reorders():
    """Some backends return out-of-order rows; we sort by index."""
    embedder = RemoteOpenAIEmbedder("http://fake.invalid:8000", "lightonai/DenseOn")
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.4, 0.5, 0.6], "index": 1},
            {"embedding": [0.1, 0.2, 0.3], "index": 0},
        ],
    }
    mock_response.raise_for_status.return_value = None
    with patch.object(embedder._session, "post", return_value=mock_response):
        result = embedder.encode(["first", "second"])

    # Row 0 corresponds to "first" input regardless of backend ordering;
    # vectors come back L2-normalized.
    np.testing.assert_allclose(
        result[0],
        np.array([0.1, 0.2, 0.3]) / np.linalg.norm([0.1, 0.2, 0.3]),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        result[1],
        np.array([0.4, 0.5, 0.6]) / np.linalg.norm([0.4, 0.5, 0.6]),
        rtol=1e-5,
    )


@pytest.mark.unit
@pytest.mark.ci_fast
def test_remote_encode_empty_input_returns_empty_array():
    embedder = RemoteOpenAIEmbedder("http://fake.invalid:8000", "lightonai/DenseOn")
    with patch.object(embedder._session, "post") as mock_post:
        result = embedder.encode([])
    mock_post.assert_not_called()
    assert result.shape == (0, 0)


def test_remote_encode_returns_1d_for_single_string_input():
    """Match SentenceTransformer: str input -> (D,), list input -> (N, D)."""
    embedder = RemoteOpenAIEmbedder("http://fake.invalid:8000", "lightonai/DenseOn")
    mock_response = _openai_embed_response([[0.6, 0.8]])
    with patch.object(embedder._session, "post", return_value=mock_response):
        single = embedder.encode("hello")
        batch = embedder.encode(["hello"])

    assert single.shape == (2,)
    assert batch.shape == (1, 2)


def test_remote_encode_normalizes_when_requested():
    embedder = RemoteOpenAIEmbedder("http://fake.invalid:8000", "lightonai/DenseOn")
    mock_response = _openai_embed_response([[3.0, 4.0]])
    with patch.object(embedder._session, "post", return_value=mock_response):
        norm = embedder.encode("hello", normalize_embeddings=True)
    # normalized [3,4] = [0.6, 0.8]
    assert norm.shape == (2,)
    np.testing.assert_allclose(norm, [0.6, 0.8], rtol=1e-5)


def test_remote_encode_always_normalizes():
    # DenseOn was always served normalize_embeddings=True; the client
    # restores it unconditionally so vectors are unit-norm even without
    # an explicit normalize_embeddings kwarg.
    embedder = RemoteOpenAIEmbedder("http://fake.invalid:8000", "lightonai/DenseOn")
    mock_response = _openai_embed_response([[3.0, 4.0]])
    with patch.object(embedder._session, "post", return_value=mock_response):
        vec = embedder.encode("hello")
    np.testing.assert_allclose(vec, [0.6, 0.8], rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(vec), 1.0, rtol=1e-5)


def test_remote_encode_accepts_sentence_transformer_compat_kwargs():
    """convert_to_numpy / normalize_embeddings stay accepted (call sites like
    audio_embedding_generator pass them); output is normalized ndarray."""
    embedder = RemoteOpenAIEmbedder("http://fake.invalid:8000", "lightonai/DenseOn")
    mock_response = _openai_embed_response([[3.0, 4.0]])
    with patch.object(embedder._session, "post", return_value=mock_response):
        vec = embedder.encode("hello", convert_to_numpy=True, normalize_embeddings=True)
    assert isinstance(vec, np.ndarray)
    np.testing.assert_allclose(vec, [0.6, 0.8], rtol=1e-5)


@pytest.mark.unit
@pytest.mark.ci_fast
def test_remote_encode_rejects_unknown_kwargs():
    """Unknown kwargs must raise TypeError naming the offending keys instead
    of being silently dropped (the local sibling forwards its kwargs to
    SentenceTransformer, so a dropped kwarg here would diverge silently)."""
    embedder = RemoteOpenAIEmbedder("http://fake.invalid:8000", "lightonai/DenseOn")
    with patch.object(embedder._session, "post") as mock_post:
        with pytest.raises(
            TypeError,
            match=r"unexpected keyword arguments: \['batch_size', 'device'\]",
        ):
            embedder.encode("hello", device="cuda", batch_size=8)
    mock_post.assert_not_called()


def test_remote_encode_raises_when_backend_returns_no_embeddings():
    embedder = RemoteOpenAIEmbedder("http://fake.invalid:8000", "lightonai/DenseOn")
    mock_response = MagicMock()
    mock_response.json.return_value = {"error": "model not found"}
    mock_response.raise_for_status.return_value = None
    with patch.object(embedder._session, "post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="no embeddings"):
            embedder.encode(["hi"])


class _EchoEmbeddingHandler(BaseHTTPRequestHandler):
    """vLLM /v1/embeddings stub: records each received input string and
    returns a deterministic non-unit vector ([3, 4], norm 5) per input."""

    received_inputs: list[list[str]] = []

    def log_message(self, *args):  # silence stderr noise
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        inputs = body.get("input", [])
        type(self).received_inputs.append(list(inputs))
        data = [
            {"embedding": [3.0, 4.0], "index": i, "object": "embedding"}
            for i in range(len(inputs))
        ]
        payload = json.dumps({"data": data, "model": body.get("model", "")}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def echo_embed_server():
    _EchoEmbeddingHandler.received_inputs = []
    server = HTTPServer(("127.0.0.1", 0), _EchoEmbeddingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", _EchoEmbeddingHandler
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_remote_query_encode_prefixes_query_prompt(echo_embed_server):
    base_url, handler = echo_embed_server
    embedder = RemoteOpenAIEmbedder(base_url, "lightonai/DenseOn")

    vec = embedder.encode("how tall is the tower", is_query=True)

    assert handler.received_inputs == [["query: how tall is the tower"]]
    # [3, 4] echoed back -> L2-normalized to [0.6, 0.8], unit norm.
    np.testing.assert_allclose(vec, [0.6, 0.8], rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(vec), 1.0, rtol=1e-6)


def test_remote_document_encode_prefixes_document_prompt(echo_embed_server):
    base_url, handler = echo_embed_server
    embedder = RemoteOpenAIEmbedder(base_url, "lightonai/DenseOn")

    vecs = embedder.encode(["paris is in france", "the eiffel tower"])

    assert handler.received_inputs == [
        ["document: paris is in france", "document: the eiffel tower"]
    ]
    for v in vecs:
        np.testing.assert_allclose(v, [0.6, 0.8], rtol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, rtol=1e-6)


def test_mem0_adapter_search_uses_query_prompt(echo_embed_server):
    base_url, handler = echo_embed_server
    from mem0.configs.embeddings.base import BaseEmbedderConfig

    from cogniverse_core.memory.mem0_embedder import DenseOnMem0Embedder

    cfg = BaseEmbedderConfig(
        model="lightonai/DenseOn",
        openai_base_url=f"{base_url}/v1",
        api_key="denseon",
    )
    adapter = DenseOnMem0Embedder(cfg)

    out = adapter.embed("what is the capital of france", memory_action="search")

    assert handler.received_inputs == [["query: what is the capital of france"]]
    np.testing.assert_allclose(out, [0.6, 0.8], rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(out), 1.0, rtol=1e-6)


def test_mem0_adapter_add_and_update_use_document_prompt(echo_embed_server):
    base_url, handler = echo_embed_server
    from mem0.configs.embeddings.base import BaseEmbedderConfig

    from cogniverse_core.memory.mem0_embedder import DenseOnMem0Embedder

    cfg = BaseEmbedderConfig(
        model="lightonai/DenseOn",
        openai_base_url=f"{base_url}/v1",
        api_key="denseon",
    )
    adapter = DenseOnMem0Embedder(cfg)

    adapter.embed("user prefers dark mode", memory_action="add")
    adapter.embed("user now prefers light mode", memory_action="update")
    # Mem0 also calls embed() with no action in some code paths -> document.
    adapter.embed("a bare memory fact")

    assert handler.received_inputs == [
        ["document: user prefers dark mode"],
        ["document: user now prefers light mode"],
        ["document: a bare memory fact"],
    ]


def test_mem0_adapter_registered_provider_resolves(echo_embed_server):
    base_url, _ = echo_embed_server
    from mem0.utils.factory import EmbedderFactory

    from cogniverse_core.memory.mem0_embedder import (
        DENSEON_PROVIDER,
        DenseOnMem0Embedder,
        register_denseon_provider,
    )

    register_denseon_provider()
    cfg = {
        "model": "lightonai/DenseOn",
        "openai_base_url": f"{base_url}/v1",
        "api_key": "denseon",
    }
    embedder = EmbedderFactory.create(DENSEON_PROVIDER, cfg, vector_config=None)
    assert isinstance(embedder, DenseOnMem0Embedder)


def test_model_name_resolution_prefers_explicit_then_env_then_default(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_SEMANTIC_EMBED_URL", "http://fake.invalid:11434")
    monkeypatch.setenv("COGNIVERSE_SEMANTIC_EMBED_MODEL", "from-env")

    # Explicit arg wins
    e1 = get_semantic_embedder(model_name="explicit-arg")
    assert e1._model == "explicit-arg"  # type: ignore[attr-defined]

    # Env var used when no explicit arg
    reset_semantic_embedder_cache()
    e2 = get_semantic_embedder()
    assert e2._model == "from-env"  # type: ignore[attr-defined]

    # Default used when neither
    reset_semantic_embedder_cache()
    monkeypatch.delenv("COGNIVERSE_SEMANTIC_EMBED_MODEL")
    e3 = get_semantic_embedder()
    assert e3._model == "lightonai/DenseOn"  # type: ignore[attr-defined]
