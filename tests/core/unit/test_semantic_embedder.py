"""Tests for the pluggable semantic embedder factory."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cogniverse_core.common.models.semantic_embedder import (
    LocalSentenceTransformerEmbedder,
    RemoteOpenAIEmbedder,
    configure_semantic_embedder_defaults,
    get_semantic_embedder,
    reset_semantic_embedder_cache,
)


@pytest.fixture(autouse=True)
def _reset_cache():
    configure_semantic_embedder_defaults(remote_url=None, model_name=None)
    reset_semantic_embedder_cache()
    yield
    reset_semantic_embedder_cache()
    configure_semantic_embedder_defaults(remote_url=None, model_name=None)


@pytest.mark.unit
@pytest.mark.ci_fast
def test_remote_url_arg_selects_remote_backend():
    embedder = get_semantic_embedder(remote_url="http://fake.invalid:11434")
    assert isinstance(embedder, RemoteOpenAIEmbedder)


def test_configured_defaults_select_remote_backend():
    configure_semantic_embedder_defaults(
        remote_url="http://fake.invalid:11434",
        model_name="configured-remote-model",
    )
    embedder = get_semantic_embedder()
    assert isinstance(embedder, RemoteOpenAIEmbedder)
    assert embedder._base_url == "http://fake.invalid:11434"
    assert embedder._model == "configured-remote-model"


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


def _repeat_to_length(fragment: str, length: int) -> str:
    repeated = fragment * (length // len(fragment) + 1)
    return repeated[:length]


@lru_cache(maxsize=1)
def _denseon_tokenizer():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot = snapshot_download("lightonai/DenseOn", local_files_only=True)
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)


def _truncate_to_denseon_budget(text: str) -> str:
    tokenizer = _denseon_tokenizer()
    max_length = tokenizer.model_max_length
    token_count = len(tokenizer(text, add_special_tokens=True).input_ids)
    if token_count <= max_length:
        return text

    lo = 0
    hi = len(text)
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        candidate_count = len(tokenizer(candidate, add_special_tokens=True).input_ids)
        if candidate_count <= max_length:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return text[:best]


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


@pytest.mark.unit
@pytest.mark.ci_fast
@pytest.mark.parametrize(
    ("label", "content"),
    [
        (
            "code-punctuation",
            _repeat_to_length(
                "def solve(x): return x + 1  # []{}()<>:=,.;!?/\\\\|`~\n",
                2048,
            ),
        ),
        (
            "cjk",
            _repeat_to_length("中文測試漢字かなカナ、。！？；：『』「」", 2048),
        ),
    ],
)
def test_remote_encode_truncates_token_dense_inputs_to_denseon_budget(
    label, content, denseon_overflow_server, caplog
):
    import logging

    base_url, handler = denseon_overflow_server
    embedder = RemoteOpenAIEmbedder(base_url, "lightonai/DenseOn")
    prefixed = f"document: {content}"
    expected = _truncate_to_denseon_budget(prefixed)
    tokenizer = _denseon_tokenizer()

    with caplog.at_level(
        logging.WARNING,
        logger="cogniverse_core.common.models.semantic_embedder",
    ):
        vector = embedder.encode(content)

    assert label in {"code-punctuation", "cjk"}
    assert handler.received_inputs == [[prefixed], [expected]]
    assert handler.received_statuses == [400, 200]
    assert len(expected) < len(prefixed)
    assert (
        sum(
            1
            for record in caplog.records
            if "DenseOn input truncated" in record.message
        )
        == 1
    )
    warning = next(
        record
        for record in caplog.records
        if record.message.startswith("DenseOn input truncated")
    )
    assert warning.args == (
        len(prefixed),
        len(tokenizer(prefixed, add_special_tokens=True).input_ids),
        len(expected),
        len(tokenizer(expected, add_special_tokens=True).input_ids),
        tokenizer.model_max_length,
    )
    np.testing.assert_allclose(vector, [0.6, 0.8], rtol=1e-6)


def test_authenticated_remote_encode_sends_exact_bearer_header(echo_embed_server):
    base_url, handler = echo_embed_server
    token = "denseon-modal-secret"
    embedder = RemoteOpenAIEmbedder(
        base_url,
        "lightonai/DenseOn",
        headers={"Authorization": f"Bearer {token}"},
    )

    vector = embedder.encode("exact authenticated content")

    assert handler.received_authorizations == [f"Bearer {token}"]
    np.testing.assert_allclose(vector, [0.6, 0.8], rtol=1e-6)
    assert token not in repr(embedder)


def test_authenticated_remote_outage_propagates_without_leaking_key():
    import requests

    token = "denseon-modal-secret"
    embedder = RemoteOpenAIEmbedder(
        "http://denseon.invalid",
        "lightonai/DenseOn",
        headers={"Authorization": f"Bearer {token}"},
    )
    with patch.object(
        embedder._session,
        "post",
        side_effect=requests.ConnectionError("DenseOn connection refused"),
    ):
        with pytest.raises(
            requests.ConnectionError, match="DenseOn connection refused"
        ) as caught:
            embedder.encode("content")

    assert "lightonai/DenseOn" in str(caught.value)
    assert "http://denseon.invalid/v1/embeddings" in str(caught.value)
    assert token not in str(caught.value)


def test_modal_cache_uses_one_environment_credential_concurrently(monkeypatch):
    import cogniverse_core.common.models.semantic_embedder as semantic_embedder

    token = "shared-production-key"
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", token)
    barrier = threading.Barrier(8)

    def resolve(_: int):
        barrier.wait(timeout=5)
        return get_semantic_embedder(
            remote_url="https://denseon.modal.run",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        resolved = list(executor.map(resolve, range(8)))

    assert len({id(item) for item in resolved}) == 1
    assert resolved[0]._headers == {"Authorization": f"Bearer {token}"}
    with pytest.raises(TypeError):
        resolved[0]._headers["Authorization"] = "Bearer replacement"
    cache_keys = " ".join(semantic_embedder._cache)
    assert token not in cache_keys


def test_modal_remote_rejects_caller_supplied_credentials(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_INFERENCE_API_KEY", "shared-production-key")
    with pytest.raises(ValueError, match="headers.*Modal"):
        RemoteOpenAIEmbedder(
            "https://denseon.modal.run",
            "lightonai/DenseOn",
            headers={"Authorization": "Bearer caller-specific-key"},
        )


def test_explicit_headers_require_remote_url():
    with patch(
        "cogniverse_core.common.models.semantic_embedder.LocalSentenceTransformerEmbedder"
    ):
        with pytest.raises(ValueError, match="headers require a remote_url"):
            get_semantic_embedder(
                headers={"Authorization": "Bearer custom-endpoint-key"}
            )


def test_modal_remote_requires_environment_credential(monkeypatch):
    monkeypatch.delenv("COGNIVERSE_INFERENCE_API_KEY", raising=False)

    with pytest.raises(
        RuntimeError,
        match="Modal inference endpoint requires COGNIVERSE_INFERENCE_API_KEY",
    ):
        RemoteOpenAIEmbedder(
            "https://denseon.modal.run",
            "lightonai/DenseOn",
        )


def test_modal_cache_and_client_share_one_credential_snapshot(monkeypatch):
    import cogniverse_core.common.models.semantic_embedder as semantic_embedder

    resolved = iter(
        [
            {"Authorization": "Bearer first-key"},
            {"Authorization": "Bearer second-key"},
        ]
    )
    monkeypatch.setattr(
        semantic_embedder,
        "_resolved_inference_headers",
        lambda base_url, headers: next(resolved),
    )

    embedder = get_semantic_embedder(
        remote_url="https://denseon.modal.run",
    )

    assert embedder._headers == {"Authorization": "Bearer first-key"}
    assert next(resolved) == {"Authorization": "Bearer second-key"}


def test_remote_cache_is_bounded_lru_and_closes_evicted_sessions(monkeypatch):
    import cogniverse_core.common.models.semantic_embedder as semantic_embedder

    monkeypatch.setattr(semantic_embedder, "_CACHE_CAPACITY", 2)
    first = get_semantic_embedder(remote_url="http://denseon-first:8000")
    second = get_semantic_embedder(remote_url="http://denseon-second:8000")
    first_close = MagicMock(wraps=first._close)
    second_close = MagicMock(wraps=second._close)
    monkeypatch.setattr(first, "_close", first_close)
    monkeypatch.setattr(second, "_close", second_close)

    assert get_semantic_embedder(remote_url="http://denseon-first:8000") is first
    third = get_semantic_embedder(remote_url="http://denseon-third:8000")

    assert len(semantic_embedder._cache) == 2
    assert list(semantic_embedder._cache.values()) == [first, third]
    first_close.assert_not_called()
    second_close.assert_called_once_with()

    third_close = MagicMock(wraps=third._close)
    monkeypatch.setattr(third, "_close", third_close)
    reset_semantic_embedder_cache()
    first_close.assert_called_once_with()
    third_close.assert_called_once_with()
    assert semantic_embedder._cache == {}


def test_remote_cache_restores_old_entry_when_eviction_close_fails(monkeypatch):
    import cogniverse_core.common.models.semantic_embedder as semantic_embedder

    monkeypatch.setattr(semantic_embedder, "_CACHE_CAPACITY", 1)
    first = get_semantic_embedder(remote_url="http://denseon-first:8000")
    first_close = MagicMock(side_effect=OSError("controlled close failure"))
    monkeypatch.setattr(first, "_close", first_close)

    with pytest.raises(
        RuntimeError,
        match="semantic embedder cache eviction failed.*controlled close failure",
    ):
        get_semantic_embedder(remote_url="http://denseon-second:8000")

    assert list(semantic_embedder._cache.values()) == [first]
    first_close.assert_called_once_with()
    first_close.side_effect = None


def test_local_embedder_participates_in_cache_shutdown(monkeypatch):
    import cogniverse_core.common.models.semantic_embedder as semantic_embedder

    monkeypatch.delenv("COGNIVERSE_SEMANTIC_EMBED_URL", raising=False)
    with patch.object(LocalSentenceTransformerEmbedder, "__init__", return_value=None):
        local = get_semantic_embedder(model_name="local-test-model")
    close = MagicMock(wraps=local._close)
    monkeypatch.setattr(local, "_close", close)

    reset_semantic_embedder_cache()

    close.assert_called_once_with()
    assert semantic_embedder._cache == {}


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


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"data": [{"index": 0, "embedding": [1.0, 0.0]}]}, "expected 2 rows"),
        (
            {
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 0, "embedding": [0.0, 1.0]},
                ]
            },
            r"indices must be exactly \[0, 1\]",
        ),
        (
            {
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 1, "embedding": [1.0]},
                ]
            },
            "same non-zero dimension",
        ),
        (
            {
                "data": [
                    {"index": 0, "embedding": [1.0, float("nan")]},
                    {"index": 1, "embedding": [0.0, 1.0]},
                ]
            },
            "finite",
        ),
        (
            {
                "data": [
                    {"index": 0, "embedding": [0.0, 0.0]},
                    {"index": 1, "embedding": [0.0, 1.0]},
                ]
            },
            "non-zero norm",
        ),
    ],
)
def test_remote_encode_rejects_malformed_embedding_contract(payload, error):
    embedder = RemoteOpenAIEmbedder("http://fake.invalid:8000", "lightonai/DenseOn")
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None

    with patch.object(embedder._session, "post", return_value=response):
        with pytest.raises(RuntimeError, match=error) as caught:
            embedder.encode(["first", "second"])

    assert "lightonai/DenseOn" in str(caught.value)
    assert "http://fake.invalid:8000/v1/embeddings" in str(caught.value)


class _EchoEmbeddingHandler(BaseHTTPRequestHandler):
    """vLLM /v1/embeddings stub: records each received input string and
    returns a deterministic non-unit vector ([3, 4], norm 5) per input."""

    received_inputs: list[list[str]] = []
    received_authorizations: list[str | None] = []

    def log_message(self, *args):  # silence stderr noise
        pass

    def do_POST(self):
        type(self).received_authorizations.append(self.headers.get("Authorization"))
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


class _DenseOnOverflowHandler(BaseHTTPRequestHandler):
    """DenseOn stub that rejects over-budget inputs and accepts retries."""

    received_inputs: list[list[str]] = []
    received_statuses: list[int] = []

    def log_message(self, *args):  # silence stderr noise
        pass

    def do_POST(self):
        tokenizer = _denseon_tokenizer()
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        inputs = body.get("input", [])
        type(self).received_inputs.append(list(inputs))
        first_input = inputs[0] if inputs else ""
        token_count = len(tokenizer(first_input, add_special_tokens=True).input_ids)
        if token_count > tokenizer.model_max_length:
            payload = {
                "error": {
                    "message": (
                        "This model's maximum context length is 512 tokens. "
                        f"However, you requested 0 output tokens and your prompt "
                        f"contains at least {token_count} input tokens, for a "
                        f"total of at least {token_count} tokens. Please reduce "
                        "the length of the input prompt or the number of "
                        "requested output tokens. (parameter=input_tokens, "
                        f"value={token_count})"
                    ),
                    "type": "BadRequestError",
                    "param": "input_tokens",
                    "code": 400,
                }
            }
            data = json.dumps(payload).encode()
            type(self).received_statuses.append(400)
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        data = json.dumps(
            {
                "data": [
                    {"embedding": [3.0, 4.0], "index": i, "object": "embedding"}
                    for i in range(len(inputs))
                ],
                "model": body.get("model", ""),
            }
        ).encode()
        type(self).received_statuses.append(200)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def echo_embed_server():
    _EchoEmbeddingHandler.received_inputs = []
    _EchoEmbeddingHandler.received_authorizations = []
    server = HTTPServer(("127.0.0.1", 0), _EchoEmbeddingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", _EchoEmbeddingHandler
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.fixture
def denseon_overflow_server():
    _DenseOnOverflowHandler.received_inputs = []
    _DenseOnOverflowHandler.received_statuses = []
    server = HTTPServer(("127.0.0.1", 0), _DenseOnOverflowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", _DenseOnOverflowHandler
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


def test_model_name_resolution_prefers_explicit_then_config_then_default():
    configure_semantic_embedder_defaults(
        remote_url="http://fake.invalid:11434",
        model_name="from-config",
    )

    # Explicit arg wins
    e1 = get_semantic_embedder(model_name="explicit-arg")
    assert e1._model == "explicit-arg"  # type: ignore[attr-defined]

    # Configured default used when no explicit arg
    reset_semantic_embedder_cache()
    e2 = get_semantic_embedder()
    assert e2._model == "from-config"  # type: ignore[attr-defined]

    # Default used when no configured remote model remains.
    reset_semantic_embedder_cache()
    configure_semantic_embedder_defaults(
        remote_url="http://fake.invalid:11434",
        model_name=None,
    )
    e3 = get_semantic_embedder()
    assert e3._model == "lightonai/DenseOn"  # type: ignore[attr-defined]
