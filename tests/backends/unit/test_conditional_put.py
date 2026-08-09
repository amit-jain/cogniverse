"""VespaBackend conditional (test-and-set) writes speak Vespa's contract.

Exercised through the real pyvespa client against a local emulator of the
Document v1 API, so the create/condition/412 handling is validated against the
same httpx request path production uses — not a mock of the backend method.

The emulator models the slice of Document v1 the backend calls:
  * GET  -> 200 {"fields": {...}} or 404
  * PUT  (partial ``assign`` update) honouring ``?create=<bool>`` and
    ``?condition=<type>.update_count==N``:
      - missing + create=true -> insert (condition ignored, per Vespa's
        create+condition contract),
      - present + condition matches stored update_count -> apply,
      - present + condition mismatch -> 412.
Writes are serialized on one lock, mirroring Vespa's per-document ordering.
"""

from __future__ import annotations

import http.server
import json
import re
import socket
import threading
from urllib.parse import parse_qs, urlparse

import pytest

from cogniverse_vespa.backend import VespaBackend
from tests.utils.vespa_test_helpers import schema_full_name

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

_NAMESPACE = "wiki_content"
_SCHEMA = schema_full_name("wiki_pages", "test_tenant")


def _dead_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class _DocV1Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _doc_id(self) -> str:
        return urlparse(self.path).path.split("/docid/", 1)[1]

    def _send(self, code: int, body: dict | None = None):
        payload = json.dumps(body or {"id": self._doc_id()}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def do_GET(self):
        with self.server.lock:
            doc = self.server.store.get(self._doc_id())
        if doc is None:
            self._send(404, {"id": self._doc_id()})
        else:
            self._send(200, {"fields": dict(doc)})

    def do_POST(self):
        fields = self._body().get("fields", {})
        with self.server.lock:
            self.server.store[self._doc_id()] = dict(fields)
        self._send(200)

    def do_PUT(self):
        assigns = {
            name: spec.get("assign") if isinstance(spec, dict) else spec
            for name, spec in self._body().get("fields", {}).items()
        }
        query = parse_qs(urlparse(self.path).query)
        create = query.get("create", ["false"])[0] == "true"
        condition = query.get("condition", [None])[0]
        expected = None
        if condition:
            m = re.search(r"update_count==(\d+)", condition)
            if m:
                expected = int(m.group(1))

        doc_id = self._doc_id()
        with self.server.lock:
            existing = self.server.store.get(doc_id)
            if existing is None:
                if create:
                    self.server.store[doc_id] = dict(assigns)
                    self._send(200)
                else:
                    self._send(400, {"message": "no such document"})
                return
            if (
                expected is not None
                and int(existing.get("update_count", -1)) != expected
            ):
                self._send(412, {"message": "condition not met"})
                return
            existing.update(assigns)
            self._send(200)


class _DocV1Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _DocV1Handler)
        self.store: dict = {}
        self.lock = threading.Lock()


def _start_emulator() -> _DocV1Server:
    server = _DocV1Server()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _backend(port: int) -> VespaBackend:
    backend = object.__new__(VespaBackend)
    backend._url = "http://127.0.0.1"
    backend._port = port
    backend._metadata_app = None
    backend._metadata_app_key = None
    backend._metadata_app_lock = threading.Lock()
    return backend


def _cond(expected: int) -> str:
    return f"{_SCHEMA}.update_count=={expected}"


def _read(backend: VespaBackend, doc_id: str):
    return backend.get_document_fields(
        doc_id, schema_name=_SCHEMA, namespace=_NAMESPACE
    )


def test_conditional_update_creates_when_absent():
    server = _start_emulator()
    try:
        backend = _backend(server.server_address[1])
        applied = backend._conditional_update_fields(
            "topic_a",
            {"content": "first", "update_count": 1},
            condition=_cond(0),
            schema_name=_SCHEMA,
            namespace=_NAMESPACE,
            create=True,
        )
        assert applied is True
        stored = _read(backend, "topic_a")
        assert stored["content"] == "first"
        assert int(stored["update_count"]) == 1
    finally:
        server.shutdown()


def test_conditional_update_applies_when_condition_matches():
    server = _start_emulator()
    try:
        server.store["topic_a"] = {"content": "old", "update_count": 1}
        backend = _backend(server.server_address[1])
        applied = backend._conditional_update_fields(
            "topic_a",
            {"content": "old and new", "update_count": 2},
            condition=_cond(1),
            schema_name=_SCHEMA,
            namespace=_NAMESPACE,
            create=True,
        )
        assert applied is True
        stored = _read(backend, "topic_a")
        assert stored["content"] == "old and new"
        assert int(stored["update_count"]) == 2
    finally:
        server.shutdown()


def test_conditional_update_returns_false_on_condition_miss():
    server = _start_emulator()
    try:
        server.store["topic_a"] = {"content": "current", "update_count": 5}
        backend = _backend(server.server_address[1])
        applied = backend._conditional_update_fields(
            "topic_a",
            {"content": "stale write", "update_count": 2},
            condition=_cond(1),
            schema_name=_SCHEMA,
            namespace=_NAMESPACE,
            create=True,
        )
        assert applied is False
        stored = _read(backend, "topic_a")
        # The rejected write must not have touched the document.
        assert stored["content"] == "current"
        assert int(stored["update_count"]) == 5
    finally:
        server.shutdown()


def test_conditional_update_raises_on_transport_error():
    backend = _backend(_dead_port())
    with pytest.raises(Exception) as excinfo:
        backend._conditional_update_fields(
            "topic_a",
            {"content": "x", "update_count": 1},
            condition=_cond(0),
            schema_name=_SCHEMA,
            namespace=_NAMESPACE,
            create=True,
        )
    # A transport failure must surface (not be swallowed as a False "condition
    # miss"); it is a connection error, never an HTTP 412.
    assert "412" not in str(excinfo.value)


def test_conditional_put_document_maps_and_writes():
    server = _start_emulator()
    try:
        from cogniverse_sdk.document import ContentType, Document

        backend = _backend(server.server_address[1])
        backend.config = {}
        backend._schema_loader_instance = _StubSchemaLoader()

        doc = Document(
            id="topic_a",
            content_type=ContentType.TEXT,
            text_content="body text",
            metadata={"update_count": 1, "page_type": "topic"},
        )
        applied = backend.conditional_put_document(
            doc,
            condition=_cond(0),
            schema_name=_SCHEMA,
            base_schema_name="wiki_pages",
            namespace=_NAMESPACE,
            create=True,
        )
        assert applied is True
        stored = _read(backend, "topic_a")
        assert stored["content"] == "body text"
        assert int(stored["update_count"]) == 1
        assert stored["page_type"] == "topic"
    finally:
        server.shutdown()


def test_conditional_put_document_raises_without_mapping():
    backend = _backend(_dead_port())
    backend.config = {}
    backend._schema_loader_instance = _StubSchemaLoader(mapping=None)

    from cogniverse_sdk.document import ContentType, Document

    doc = Document(id="topic_a", content_type=ContentType.TEXT, text_content="x")
    with pytest.raises(ValueError, match="no document_mapping"):
        backend.conditional_put_document(
            doc,
            condition=_cond(0),
            schema_name=_SCHEMA,
            base_schema_name="wiki_pages",
            namespace=_NAMESPACE,
        )


class _StubSchemaLoader:
    def __init__(self, mapping: dict | None = ...):
        if mapping is ...:
            mapping = {
                "id": "doc_id",
                "text_content": "content",
                "include_metadata": True,
            }
        self._mapping = mapping

    def load_schema(self, base_name: str) -> dict:
        return {"document_mapping": self._mapping} if self._mapping else {}
