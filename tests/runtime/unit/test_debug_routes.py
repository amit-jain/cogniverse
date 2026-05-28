"""HTTP-level coverage for /admin/debug/{memsnap, memreset}.

Both routes are env-gated on ``COGNIVERSE_DEBUG_MEM``. Without the gate
the surface returned 500 silently until a real ASGI test caught it.
"""

from __future__ import annotations

import tracemalloc

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_runtime.routers import debug as debug_router


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Mount the debug router on a minimal app at /admin/debug."""
    app = FastAPI()
    app.include_router(debug_router.router, prefix="/admin/debug")
    # Reset module-level state between tests.
    debug_router._prev_snapshot = None
    yield TestClient(app)
    # Stop tracing after each test so state does not leak.
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    debug_router._prev_snapshot = None


def test_memsnap_returns_403_without_env_var(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COGNIVERSE_DEBUG_MEM", raising=False)
    r = client.post("/admin/debug/memsnap")
    assert r.status_code == 403
    assert "COGNIVERSE_DEBUG_MEM" in r.json()["detail"]


def test_memreset_returns_403_without_env_var(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COGNIVERSE_DEBUG_MEM", raising=False)
    r = client.post("/admin/debug/memreset")
    assert r.status_code == 403


def test_memsnap_returns_diff_shape_with_env_var(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COGNIVERSE_DEBUG_MEM", "1")

    r1 = client.post("/admin/debug/memsnap?top_n=5")
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["started"] is True  # first call started tracing
    assert "total_mb" in body1
    assert isinstance(body1["total_mb"], (int, float))
    assert "top_current" in body1 and isinstance(body1["top_current"], list)
    assert len(body1["top_current"]) <= 5
    # First call has no previous snapshot — growth list is empty.
    assert body1["top_growth"] == []

    # Allocate something between snapshots so the diff has content.
    _retained = [bytearray(1024 * 16) for _ in range(50)]
    r2 = client.post("/admin/debug/memsnap?top_n=5")
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["started"] is False  # tracing already on
    # Each growth entry has the documented shape.
    for entry in body2["top_growth"]:
        assert set(entry.keys()) == {
            "file",
            "line",
            "size_bytes",
            "size_diff_bytes",
            "count",
            "count_diff",
        }
        assert entry["size_diff_bytes"] > 0
    # ensure the test allocation is reachable (silence unused-var warning).
    assert len(_retained) == 50


def test_memreset_stops_tracing_and_clears_baseline(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COGNIVERSE_DEBUG_MEM", "1")
    # Prime tracing.
    client.post("/admin/debug/memsnap")
    assert tracemalloc.is_tracing() is True
    assert debug_router._prev_snapshot is not None

    r = client.post("/admin/debug/memreset")
    assert r.status_code == 200
    body = r.json()
    assert body["was_tracing"] is True
    assert tracemalloc.is_tracing() is False
    assert debug_router._prev_snapshot is None


def test_memreset_when_not_tracing_is_idempotent(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("COGNIVERSE_DEBUG_MEM", "1")
    assert tracemalloc.is_tracing() is False
    r = client.post("/admin/debug/memreset")
    assert r.status_code == 200
    assert r.json()["was_tracing"] is False
