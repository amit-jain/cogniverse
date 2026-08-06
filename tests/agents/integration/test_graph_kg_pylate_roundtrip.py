"""End-to-end round-trip of the KG → PyLate /pooling → Vespa write path.

Spins up an HTTP stub that mimics both:
  - the PyLate service (LateOn) ``POST /pooling`` endpoint (returns
    canonical (N, 128) per-token embeddings); query vs document is
    carried by the ``is_query`` request field — the service applies the
    ``[Q] ``/``[D] `` markers and query expansion itself, so the client
    sends raw text
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

import pytest
import requests

from cogniverse_agents.graph.graph_manager import GraphManager
from cogniverse_agents.graph.graph_schema import ExtractionResult, Node

pytestmark = pytest.mark.integration


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _StubHandler(BaseHTTPRequestHandler):
    """Two-faced stub: /pooling for the PyLate service, /document/v1 for Vespa."""

    pooling_requests: list[dict] = []
    feed_payloads: list[tuple[str, dict]] = []  # (path, payload)
    search_requests: list[dict] = []
    pooling_n_tokens: int = 4

    def log_message(self, format, *args):  # silence stderr
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length))

        if self.path == "/pooling":
            _StubHandler.pooling_requests.append(body)
            # The PyLate service owns query vs document encoding — the
            # client must send raw text plus an explicit is_query flag.
            assert isinstance(body.get("is_query"), bool), (
                f"/pooling requires a boolean is_query field; got keys: "
                f"{list(body.keys())}"
            )
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

        if self.path == "/search/":
            _StubHandler.search_requests.append(body)
            payload = json.dumps(
                {"root": {"fields": {"totalCount": 0}, "children": []}}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(404)
        self.end_headers()


@pytest.fixture
def stub():
    _StubHandler.pooling_requests = []
    _StubHandler.feed_payloads = []
    _StubHandler.search_requests = []
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


class _DocumentV1Backend:
    """Backend seam driving the same ``POST /document/v1/...`` pyvespa
    issues, so the stub records the true Vespa wire payload."""

    def __init__(self, port: int):
        self._url = "http://127.0.0.1"
        self._port = port

    def put_document_fields(
        self, document_id, fields, schema_name=None, namespace=None
    ):
        response = requests.post(
            f"{self._url}:{self._port}/document/v1/{namespace}/{schema_name}"
            f"/docid/{document_id}",
            json={"fields": fields},
            timeout=5,
        )
        response.raise_for_status()


def _make_manager(port: int) -> GraphManager:
    return GraphManager(
        backend=_DocumentV1Backend(port),
        tenant_id="test_tenant",
        schema_name="knowledge_graph_test_tenant",
        colbert_endpoint_url=f"http://127.0.0.1:{port}",
    )


def test_node_upsert_writes_both_tensor_fields_in_vespa_wire_format(stub):
    port, capture = stub
    manager = _make_manager(port)

    from cogniverse_agents.graph.graph_schema import Mention

    anchor = Mention(
        source_doc_id="src1.py",
        segment_id="function:alpha_handler",
        ts_start=0.0,
        ts_end=0.0,
        modality="code",
        evidence_span="def alpha_handler(): ...",
    )

    result = ExtractionResult(
        source_doc_id="src1.py",
        nodes=[
            Node(
                tenant_id="test_tenant",
                name="Alpha",
                mentions=[anchor],
                kind="entity",
                description="First node under test",
            )
        ],
        edges=[],
    )

    counts = manager.upsert(result)
    assert counts == {"nodes_upserted": 1, "edges_upserted": 0, "failed_ids": []}

    # 1. Stub saw a /pooling request carrying the raw node text with
    #    is_query false (the service applies the document marker itself).
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


def test_query_encoding_sends_raw_text_with_is_query(stub):
    port, capture = stub
    manager = _make_manager(port)

    results = manager.search_nodes("find me alpha", top_k=5)
    assert results == []

    # The encoder sent the raw query text with is_query true — the service
    # applies the [Q] marker and query expansion itself.
    assert len(capture.pooling_requests) == 1
    query_req = capture.pooling_requests[0]
    assert query_req["input"] == ["find me alpha"]
    assert query_req["is_query"] is True

    # The encoded query reached /search/ as per-token MaxSim inputs: one
    # block per stub token for both the bfloat16 and binary query tensors.
    assert len(capture.search_requests) == 1
    search_req = capture.search_requests[0]
    n_tokens = capture.pooling_n_tokens
    assert search_req["yql"] == (
        "select * from sources knowledge_graph_test_tenant "
        'where tenant_id contains "test_tenant" '
        'and doc_type contains "node" and userQuery() limit 5'
    )
    assert search_req["query"] == "find me alpha"
    assert search_req["ranking.profile"] == "hybrid_binary_bm25"
    assert search_req["model.restrict"] == "knowledge_graph_test_tenant"
    qt_blocks = search_req["input.query(qt)"]["blocks"]
    qtb_blocks = search_req["input.query(qtb)"]["blocks"]
    assert set(qt_blocks.keys()) == {str(i) for i in range(n_tokens)}
    assert all(len(block) == 128 for block in qt_blocks.values())
    assert qt_blocks["0"] == pytest.approx([0.4] * 128)
    # Sign-packed binary of the stub embeddings: positive tokens (0, 2) pack
    # to 0xFF bytes (int8 -1), negative tokens (1, 3) to 0x00.
    assert qtb_blocks == {
        "0": [-1] * 16,
        "1": [0] * 16,
        "2": [-1] * 16,
        "3": [0] * 16,
    }


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
                evidence_span="from beta import alpha_handler",
                segment_id="function:alpha_handler",
                ts_start=0.0,
                ts_end=0.0,
                modality="code",
                source_doc_id="src.py",
            )
        ],
    )
    counts = manager.upsert(result)
    assert counts == {"nodes_upserted": 0, "edges_upserted": 1, "failed_ids": []}

    assert len(capture.feed_payloads) == 1
    _, payload = capture.feed_payloads[0]
    fields = payload["fields"]
    assert "embedding" not in fields
    assert "embedding_binary" not in fields
