"""Tests for the pluggable semantic embedder factory."""

from __future__ import annotations

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


def test_remote_url_arg_selects_remote_backend():
    embedder = get_semantic_embedder(remote_url="http://fake.invalid:11434")
    assert isinstance(embedder, RemoteOpenAIEmbedder)


def test_env_var_selects_remote_backend(monkeypatch):
    monkeypatch.setenv("COGNIVERSE_SEMANTIC_EMBED_URL", "http://fake.invalid:11434")
    embedder = get_semantic_embedder()
    assert isinstance(embedder, RemoteOpenAIEmbedder)


def test_no_url_falls_back_to_local(monkeypatch):
    monkeypatch.delenv("COGNIVERSE_SEMANTIC_EMBED_URL", raising=False)
    with patch(
        "cogniverse_core.common.models.semantic_embedder.SemanticEmbedder"
    ):
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


def test_remote_encode_hits_v1_embeddings():
    embedder = RemoteOpenAIEmbedder("http://fake.invalid:8000/", "nomic-embed-text")

    mock_response = _openai_embed_response([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    with patch.object(embedder._session, "post", return_value=mock_response) as mock_post:
        result = embedder.encode(["hello", "world"])

    # Trailing slash stripped, OpenAI-compatible endpoint
    called_url = mock_post.call_args.args[0]
    assert called_url == "http://fake.invalid:8000/v1/embeddings"

    payload = mock_post.call_args.kwargs["json"]
    assert payload["model"] == "nomic-embed-text"
    assert payload["input"] == ["hello", "world"]

    assert result.shape == (2, 3)
    assert result.dtype == np.float32


def test_remote_encode_preserves_order_when_backend_reorders():
    """Some backends return out-of-order rows; we sort by index."""
    embedder = RemoteOpenAIEmbedder("http://fake.invalid:8000", "nomic-embed-text")
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

    # Row 0 corresponds to "first" input regardless of backend ordering
    np.testing.assert_allclose(result[0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(result[1], [0.4, 0.5, 0.6])


def test_remote_encode_empty_input_returns_empty_array():
    embedder = RemoteOpenAIEmbedder("http://fake.invalid:8000", "nomic-embed-text")
    with patch.object(embedder._session, "post") as mock_post:
        result = embedder.encode([])
    mock_post.assert_not_called()
    assert result.shape == (0, 0)


def test_remote_encode_returns_1d_for_single_string_input():
    """Match SentenceTransformer: str input -> (D,), list input -> (N, D)."""
    embedder = RemoteOpenAIEmbedder("http://fake.invalid:8000", "nomic-embed-text")
    mock_response = _openai_embed_response([[0.6, 0.8]])
    with patch.object(embedder._session, "post", return_value=mock_response):
        single = embedder.encode("hello")
        batch = embedder.encode(["hello"])

    assert single.shape == (2,)
    assert batch.shape == (1, 2)


def test_remote_encode_normalizes_when_requested():
    embedder = RemoteOpenAIEmbedder("http://fake.invalid:8000", "nomic-embed-text")
    mock_response = _openai_embed_response([[3.0, 4.0]])
    with patch.object(embedder._session, "post", return_value=mock_response):
        norm = embedder.encode("hello", normalize_embeddings=True)
    # L2-normalized [3,4] = [0.6, 0.8]
    assert norm.shape == (2,)
    np.testing.assert_allclose(norm, [0.6, 0.8], rtol=1e-5)


def test_remote_encode_skips_normalize_when_not_requested():
    embedder = RemoteOpenAIEmbedder("http://fake.invalid:8000", "nomic-embed-text")
    mock_response = _openai_embed_response([[3.0, 4.0]])
    with patch.object(embedder._session, "post", return_value=mock_response):
        raw = embedder.encode("hello")
    np.testing.assert_allclose(raw, [3.0, 4.0], rtol=1e-5)


def test_remote_encode_raises_when_backend_returns_no_embeddings():
    embedder = RemoteOpenAIEmbedder("http://fake.invalid:8000", "nomic-embed-text")
    mock_response = MagicMock()
    mock_response.json.return_value = {"error": "model not found"}
    mock_response.raise_for_status.return_value = None
    with patch.object(embedder._session, "post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="no embeddings"):
            embedder.encode(["hi"])


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
    assert e3._model == "nomic-embed-text"  # type: ignore[attr-defined]
