"""`cogniverse index` pipeline against the HTTP boundary via MockTransport.

Covers the upload → poll-status → graph-upsert flow and the graph-error
surfacing: a broken extractor previously read as "no graph in any file"
(zero nodes, no error anywhere), because every extraction exception was
swallowed per-file.
"""

from __future__ import annotations

import io
from pathlib import Path

import cogniverse_cli.index as index_cli
import httpx
import pytest
from rich.console import Console


@pytest.fixture(autouse=True)
def capture_console(monkeypatch: pytest.MonkeyPatch):
    buf = io.StringIO()
    test_console = Console(file=buf, width=200, force_terminal=False, color_system=None)
    monkeypatch.setattr(index_cli, "console", test_console)
    return buf


def _mount_httpx(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _factory)


def _runtime_handler(graph_response: httpx.Response):
    """A stub runtime: upload accepts, status completes, upsert configurable."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ingestion/upload":
            return httpx.Response(202, json={"ingest_id": "i1", "state": "queued"})
        if request.url.path == "/ingestion/i1/status":
            return httpx.Response(
                200,
                json={
                    "state": "complete",
                    "latest": {"result": {"chunks": 1, "documents_fed": 1}},
                },
            )
        if request.url.path == "/graph/upsert":
            return graph_response
        return httpx.Response(404, text=f"unexpected path {request.url.path}")

    return handler


def _invalid_document_count_handler(documents_fed):
    """Runtime reports terminal success with an invalid feed count."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ingestion/upload":
            return httpx.Response(202, json={"ingest_id": "i-empty", "state": "queued"})
        if request.url.path == "/ingestion/i-empty/status":
            return httpx.Response(
                200,
                json={
                    "state": "complete",
                    "latest": {"result": {"chunks": 0, "documents_fed": documents_fed}},
                },
            )
        if request.url.path == "/graph/upsert":
            return httpx.Response(200, json={"nodes_upserted": 0, "edges_upserted": 0})
        return httpx.Response(404, text=f"unexpected path {request.url.path}")

    return handler


class _BoomExtractor:
    def extract(self, *args, **kwargs):
        raise RuntimeError("tree-sitter exploded")


def test_extractor_failure_is_surfaced_in_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture_console: io.StringIO,
) -> None:
    (tmp_path / "a.py").write_text("def f():\n    return 1\n")

    import cogniverse_agents.graph.code_extractor as code_extractor_mod

    monkeypatch.setattr(code_extractor_mod, "CodeExtractor", _BoomExtractor)
    _mount_httpx(
        monkeypatch,
        _runtime_handler(httpx.Response(200, json={"nodes_upserted": 0})),
    )

    summary = index_cli.index_files(
        root=tmp_path,
        content_type="code",
        tenant_id="acme:acme",
        runtime_url="http://runtime.test",
    )

    assert summary["files_indexed"] == 1
    assert summary["graph_errors"] == 1
    assert summary["graph_nodes"] == 0
    out = capture_console.getvalue()
    assert "Graph errors: 1" in out
    assert "tree-sitter exploded" in out


def test_upload_and_graph_counts_in_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture_console: io.StringIO,
) -> None:
    (tmp_path / "a.py").write_text("def f():\n    return 1\n")

    _mount_httpx(
        monkeypatch,
        _runtime_handler(
            httpx.Response(200, json={"nodes_upserted": 3, "edges_upserted": 2})
        ),
    )

    summary = index_cli.index_files(
        root=tmp_path,
        content_type="code",
        tenant_id="acme:acme",
        runtime_url="http://runtime.test",
    )

    assert summary["files_found"] == 1
    assert summary["files_indexed"] == 1
    assert summary["chunks_created"] == 1
    assert summary["graph_errors"] == 0
    out = capture_console.getvalue()
    assert "Indexed 1/1 files" in out


@pytest.mark.parametrize(
    ("documents_fed", "error_fragment"),
    [
        (0, "completed without feeding any documents"),
        (-1, "completed without feeding any documents"),
        ("zero", "invalid documents_fed='zero'"),
        (True, "invalid documents_fed=True"),
    ],
)
def test_terminal_ingest_with_invalid_document_count_is_not_reported_as_indexed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture_console: io.StringIO,
    documents_fed,
    error_fragment: str,
) -> None:
    (tmp_path / "empty.py").write_text("")
    _mount_httpx(monkeypatch, _invalid_document_count_handler(documents_fed))

    summary = index_cli.index_files(
        root=tmp_path,
        content_type="code",
        tenant_id="acme:acme",
        runtime_url="http://runtime.test",
    )

    assert summary["files_indexed"] == 0
    assert summary["documents_fed"] == 0
    assert summary["errors"] == 1
    out = capture_console.getvalue()
    assert "Indexed 0/1 files" in out
    assert error_fragment in out


def test_upload_failure_lands_in_errors_not_silence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capture_console: io.StringIO,
) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/ingestion/upload":
            return httpx.Response(503, text="ingestion unavailable")
        if request.url.path == "/graph/upsert":
            return httpx.Response(200, json={"nodes_upserted": 0})
        return httpx.Response(404)

    _mount_httpx(monkeypatch, handler)

    summary = index_cli.index_files(
        root=tmp_path,
        content_type="code",
        tenant_id="acme:acme",
        runtime_url="http://runtime.test",
    )

    assert summary["files_indexed"] == 0
    assert summary["errors"] == 1
    assert "503" in capture_console.getvalue()
