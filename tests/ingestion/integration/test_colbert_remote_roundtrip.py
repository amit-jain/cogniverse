"""End-to-end round-trip of the remote ColBERT path.

Spins up a real HTTP stub that mimics the pylate sidecar's ``/pooling``
endpoint, routes a text through ``RemoteColBERTLoader`` → wrapper →
``VespaEmbeddingProcessor``, and asserts the resulting Vespa-feed payload
contains both the bfloat16 ``embedding`` and the 16-byte ``embedding_binary``
tensors the ``lateon_mv`` schema expects.

This is the wiring-correctness test required by CLAUDE.md — it would catch:
- A client that silently drops ``is_query`` when talking to the sidecar.
- A response parser that loses per-token structure.
- A binarization step that emits the wrong byte count per token.
"""

from __future__ import annotations

import json
import socket
import threading
from binascii import unhexlify
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from cogniverse_core.common.models.model_loaders import RemoteColBERTLoader
from cogniverse_vespa.embedding_processor import VespaEmbeddingProcessor


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _PoolingStub(BaseHTTPRequestHandler):
    """Records received requests, responds with deterministic per-token floats."""

    captured_requests: list[dict] = []  # class-level, reset per test

    def log_message(self, format, *args):  # silence stderr
        return

    def do_POST(self) -> None:
        if self.path != "/pooling":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length))
        _PoolingStub.captured_requests.append(body)

        # Produce per-text (N_tokens, 128) embeddings. is_query=True gives 3 tokens,
        # is_query=False gives 5 — exercising the flag propagation path.
        is_query = bool(body.get("is_query", False))
        n_tokens = 3 if is_query else 5
        data = []
        for i, _text in enumerate(body["input"]):
            tokens = []
            for tok in range(n_tokens):
                # Alternating signs so binarization has both 1s and 0s
                base = 0.5 if (tok % 2 == 0) else -0.5
                row = [base + (i * 0.01) for _ in range(128)]
                tokens.append(row)
            data.append({"object": "pooling", "index": i, "data": tokens})
        resp = {"object": "list", "data": data, "model": body.get("model", "stub")}
        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def stub_sidecar():
    _PoolingStub.captured_requests = []
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _PoolingStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}", _PoolingStub
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_roundtrip_query_side_produces_vespa_feed_payload(stub_sidecar):
    url, stub = stub_sidecar
    loader = RemoteColBERTLoader(
        model_name="lightonai/LateOn",
        config={"remote_inference_url": url},
    )
    model, _ = loader.load_model()

    embeddings = model.encode(["what is vector search"], is_query=True)

    # 1. The wrapper forwarded is_query=True to the sidecar.
    assert len(stub.captured_requests) == 1
    assert stub.captured_requests[0]["is_query"] is True
    assert stub.captured_requests[0]["input"] == ["what is vector search"]
    assert stub.captured_requests[0]["model"] == "lightonai/LateOn"

    # 2. The wrapper produced a per-token array for the single input text.
    assert len(embeddings) == 1
    tokens = np.asarray(embeddings[0], dtype=np.float32)
    assert tokens.shape == (3, 128), f"query-side expected (3, 128), got {tokens.shape}"

    # 3. VespaEmbeddingProcessor emits both embedding fields in the schema's format.
    processor = VespaEmbeddingProcessor(schema_name="lateon_mv")
    feed = processor.process_embeddings(tokens)
    assert set(feed.keys()) == {"embedding", "embedding_binary"}
    assert isinstance(feed["embedding"], dict) and len(feed["embedding"]) == 3
    assert isinstance(feed["embedding_binary"], dict) and len(feed["embedding_binary"]) == 3
    for token_idx in ("0", "1", "2"):
        # bfloat16 hex is 4 chars per value → 4 * 128 = 512 chars per token
        assert len(feed["embedding"][token_idx]) == 512
        # binary is 128 bits → 16 bytes → 32 hex chars per token
        binary_bytes = unhexlify(feed["embedding_binary"][token_idx])
        assert len(binary_bytes) == 16


def test_roundtrip_document_side_passes_is_query_false(stub_sidecar):
    url, stub = stub_sidecar
    loader = RemoteColBERTLoader(
        model_name="lightonai/LateOn",
        config={"remote_inference_url": url},
    )
    model, _ = loader.load_model()

    embeddings = model.encode(
        ["Vespa stores token embeddings as tensor<bfloat16>(token{}, v[128])."],
        is_query=False,
    )

    assert stub.captured_requests[0]["is_query"] is False
    tokens = np.asarray(embeddings[0], dtype=np.float32)
    assert tokens.shape == (5, 128), f"doc-side expected (5, 128), got {tokens.shape}"


def test_roundtrip_batched_inputs_preserve_order(stub_sidecar):
    """Multiple texts in one call must retain per-text per-token structure."""
    url, _ = stub_sidecar
    loader = RemoteColBERTLoader(
        model_name="lightonai/LateOn",
        config={"remote_inference_url": url},
    )
    model, _ = loader.load_model()

    embeddings = model.encode(
        ["text zero", "text one", "text two"],
        is_query=False,
    )
    assert len(embeddings) == 3
    for i, emb in enumerate(embeddings):
        tokens = np.asarray(emb, dtype=np.float32)
        assert tokens.shape == (5, 128)
        # stub encoded `i * 0.01` into each token's first element; verify order
        assert tokens[0, 0] == pytest.approx(0.5 + i * 0.01, abs=1e-6)


def test_roundtrip_binary_bits_reflect_actual_sign_pattern(stub_sidecar):
    """Final sanity: the binary bytes in the Vespa payload are derived from
    the actual HTTP response, not cached or zeroed out."""
    url, _ = stub_sidecar
    loader = RemoteColBERTLoader(
        model_name="lightonai/LateOn",
        config={"remote_inference_url": url},
    )
    model, _ = loader.load_model()

    tokens = np.asarray(model.encode(["x"], is_query=False)[0], dtype=np.float32)
    processor = VespaEmbeddingProcessor(schema_name="lateon_mv")
    feed = processor.process_embeddings(tokens)

    # Stub alternates: even tokens all +0.5, odd tokens all -0.5.
    # → even tokens binarize to all-1s = 0xFF * 16
    # → odd tokens binarize to all-0s = 0x00 * 16
    assert unhexlify(feed["embedding_binary"]["0"]) == b"\xff" * 16
    assert unhexlify(feed["embedding_binary"]["1"]) == b"\x00" * 16
    assert unhexlify(feed["embedding_binary"]["2"]) == b"\xff" * 16
