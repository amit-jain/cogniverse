"""End-to-end round-trip of the KG → colbert_pylate → Vespa write path.

Spins up an HTTP stub that mimics both:
  - the colbert_pylate sidecar's ``POST /pooling`` (returns canonical
    (N, 128) per-token embeddings)
  - Vespa's Document v1 ``PUT /document/v1/...`` (records the payload)

Routes a node upsert through GraphManager and asserts the wire-format
the runtime sends to Vespa is correct: per-token keyed maps for both
``embedding`` (bfloat16-hex per token) and ``embedding_binary``
(1-bit packed, 16 bytes per token).

Catches the bug where multi-vector encode shape, VespaEmbeddingProcessor
output shape, or Document-v1 payload assembly drift apart silently —
the kind of regression that can't be reached by mocking any single
layer in isolation.
"""

from __future__ import annotations

import json
import socket
import threading
from binascii import unhexlify
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock

import pytest

from cogniverse_agents.graph.graph_manager import GraphManager
from cogniverse_agents.graph.graph_schema import ExtractionResult, Node


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _StubHandler(BaseHTTPRequestHandler):
    """Two-faced stub: /pooling for colbert_pylate, /document/v1 for Vespa."""

    pooling_requests: list[dict] = []
    feed_payloads: list[tuple[str, dict]] = []  # (path, payload)
    pooling_n_tokens: int = 4

    def log_message(self, format, *args):  # silence stderr
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length))

        if self.path == "/pooling":
            _StubHandler.pooling_requests.append(body)
            n = _StubHandler.pooling_n_tokens
            data = []
            for i, _text in enumerate(body["input"]):
                tokens = []
                for tok in range(n):
                    base = 0.4 if (tok % 2 == 0) else -0.4
                    tokens.append([base + tok * 0.05 + i * 0.001 for _ in range(128)])
                data.append({"object": "pooling", "index": i, "data": tokens})
            payload = json.dumps(
                {"object": "list", "data": data, "model": body.get("model", "stub")}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path.startswith("/document/v1/"):
            _StubHandler.feed_payloads.append((self.path, body))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"id": "ok"}')
            return

        self.send_response(404)
        self.end_headers()


@pytest.fixture
def stub():
    _StubHandler.pooling_requests = []
    _StubHandler.feed_payloads = []
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port, _StubHandler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _make_manager(port: int) -> GraphManager:
    backend = MagicMock()
    backend._url = "http://127.0.0.1"
    backend._port = port
    return GraphManager(
        backend=backend,
        tenant_id="test_tenant",
        schema_name="knowledge_graph_test_tenant",
        colbert_endpoint_url=f"http://127.0.0.1:{port}",
    )


def test_node_upsert_writes_both_tensor_fields_in_vespa_wire_format(stub):
    port, capture = stub
    manager = _make_manager(port)

    result = ExtractionResult(
        source_doc_id="src1.py",
        nodes=[
            Node(
                tenant_id="test_tenant",
                name="Alpha",
                kind="entity",
                description="First node under test",
            )
        ],
        edges=[],
    )

    counts = manager.upsert(result)
    assert counts == {"nodes_upserted": 1, "edges_upserted": 0}

    # 1. Stub saw a /pooling request with is_query=false (document side).
    assert len(capture.pooling_requests) == 1
    pool_req = capture.pooling_requests[0]
    assert pool_req["is_query"] is False
    assert pool_req["input"] == ["Alpha\nFirst node under test"]

    # 2. Stub saw exactly one PUT to /document/v1/... for the node.
    assert len(capture.feed_payloads) == 1
    path, payload = capture.feed_payloads[0]
    assert "/graph_content/knowledge_graph_test_tenant/docid/" in path
    fields = payload["fields"]

    # 3. Both tensor fields landed in the payload.
    assert "embedding" in fields
    assert "embedding_binary" in fields

    # 4. Wire format: mapped tensor keyed by token index. Stub returned
    # 4 tokens, so both maps must have 4 entries.
    n_tokens = capture.pooling_n_tokens
    assert isinstance(fields["embedding"], dict)
    assert set(fields["embedding"].keys()) == {str(i) for i in range(n_tokens)}
    assert set(fields["embedding_binary"].keys()) == {str(i) for i in range(n_tokens)}

    # 5. embedding values are bfloat16 hex — 4 hex chars per dim, 128 dims = 512 chars.
    for token_idx in range(n_tokens):
        hex_str = fields["embedding"][str(token_idx)]
        assert isinstance(hex_str, str)
        assert len(hex_str) == 512, (
            f"token {token_idx}: expected 512 hex chars, got {len(hex_str)}"
        )

    # 6. embedding_binary is 1-bit packed: 128 bits → 16 bytes → 32 hex chars.
    for token_idx in range(n_tokens):
        bin_hex = fields["embedding_binary"][str(token_idx)]
        assert isinstance(bin_hex, str)
        assert len(unhexlify(bin_hex)) == 16, (
            f"token {token_idx}: expected 16-byte binary embedding"
        )


def test_query_encoding_sets_is_query_true_and_builds_block_inputs(stub):
    port, capture = stub
    manager = _make_manager(port)

    # search_nodes encodes the query with is_query=True and POSTs to /search/.
    # Our stub doesn't implement /search/ — it'll 404, which is fine for
    # this test: we only care that the encoder was called with is_query=True
    # and that no exception escaped (search falls back to YQL visit on
    # transport errors).
    manager.search_nodes("find me alpha", top_k=5)

    # The stub saw at least one /pooling call for the query side.
    assert any(req["is_query"] is True for req in capture.pooling_requests), (
        "search_nodes must encode the query with is_query=True"
    )
    query_req = next(req for req in capture.pooling_requests if req["is_query"] is True)
    assert query_req["input"] == ["find me alpha"]


def test_edge_upsert_omits_embedding_fields(stub):
    """Edges aren't semantically searchable — they must not carry the
    mapped embedding fields. Vespa attribute tensors handle absence."""
    port, capture = stub
    manager = _make_manager(port)

    from cogniverse_agents.graph.graph_schema import Edge

    result = ExtractionResult(
        source_doc_id="src.py",
        nodes=[],
        edges=[
            Edge(
                tenant_id="test_tenant",
                source="alpha",
                target="beta",
                relation="depends_on",
                source_doc_id="src.py",
            )
        ],
    )
    counts = manager.upsert(result)
    assert counts == {"nodes_upserted": 0, "edges_upserted": 1}

    assert len(capture.feed_payloads) == 1
    _, payload = capture.feed_payloads[0]
    fields = payload["fields"]
    assert "embedding" not in fields
    assert "embedding_binary" not in fields
