"""HTTP-level coverage for the synthetic-data FastAPI routes.

Routes are tested through ``httpx.ASGITransport`` so the FastAPI
``response_model`` validation, the ``HTTPException`` ladder
(400 → ValueError, 422 → ValidationError, 500 → other), the query-param
validators, and the response envelope all execute. Without these tests
the router shipped untested end-to-end.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_synthetic import api as synthetic_api


@pytest.fixture
def app() -> FastAPI:
    a = FastAPI()
    a.include_router(synthetic_api.router)
    return a


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_service(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(synthetic_api, "_service", None)
    yield


def test_health_returns_200_with_known_shape(client: TestClient) -> None:
    r = client.get("/synthetic/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert body["service"] == "synthetic-data-generation"
    assert "generators" in body and isinstance(body["generators"], int)
    assert "optimizers" in body and isinstance(body["optimizers"], int)


def test_list_optimizers_returns_mapping(client: TestClient) -> None:
    r = client.get("/synthetic/optimizers")
    assert r.status_code == 200
    body = r.json()
    # Must be a non-empty dict with at least the well-known optimizers.
    assert isinstance(body, dict)
    assert len(body) >= 1


def test_optimizer_details_404_on_unknown_name(client: TestClient) -> None:
    r = client.get("/synthetic/optimizers/no-such-optimizer")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


def test_generate_400_on_value_error(client: TestClient, monkeypatch) -> None:
    """Service-side ``ValueError`` must surface as 400, not 500."""
    fake = MagicMock()
    fake.generate = AsyncMock(side_effect=ValueError("optimizer not registered"))
    monkeypatch.setattr(synthetic_api, "_service", fake)
    r = client.post(
        "/synthetic/generate",
        json={
            "optimizer": "ROUTING_GEPA",
            "count": 5,
            "tenant_id": "acme",
        },
    )
    assert r.status_code == 400
    assert "optimizer not registered" in r.json()["detail"]


def test_generate_422_on_pydantic_request_error(client: TestClient) -> None:
    """Pydantic validation of the request body must produce 422 from FastAPI."""
    r = client.post(
        "/synthetic/generate",
        json={
            # missing required `optimizer` and `tenant_id`
            "count": 5,
        },
    )
    assert r.status_code == 422


def test_batch_generate_query_param_bounds_enforced(client: TestClient) -> None:
    """count_per_batch is bounded [1, 1000]; out-of-range → 422 from FastAPI."""
    r = client.post(
        "/synthetic/batch/generate",
        params={
            "optimizer": "ROUTING_GEPA",
            "count_per_batch": 10_000,  # exceeds le=1000
            "num_batches": 1,
            "tenant_id": "acme",
        },
    )
    assert r.status_code == 422


def test_batch_generate_unknown_optimizer_returns_400(client: TestClient) -> None:
    r = client.post(
        "/synthetic/batch/generate",
        params={
            "optimizer": "no-such-optimizer",
            "count_per_batch": 1,
            "num_batches": 1,
            "tenant_id": "acme",
        },
    )
    assert r.status_code == 400
    assert "Unknown optimizer" in r.json()["detail"]


def test_batch_generate_requires_tenant_id(client: TestClient) -> None:
    r = client.post(
        "/synthetic/batch/generate",
        params={
            "optimizer": "ROUTING_GEPA",
            "count_per_batch": 1,
            "num_batches": 1,
        },
    )
    assert r.status_code == 422  # tenant_id is required by FastAPI Query(...)


def test_batch_generate_service_value_error_returns_400(
    client: TestClient, monkeypatch
) -> None:
    """A ``ValueError`` raised mid-loop surfaces as 400, not 500."""
    fake = MagicMock()
    fake.generate = AsyncMock(side_effect=ValueError("no profiles available"))
    monkeypatch.setattr(synthetic_api, "_service", fake)

    r = client.post(
        "/synthetic/batch/generate",
        params={
            "optimizer": "routing",
            "count_per_batch": 2,
            "num_batches": 3,
            "tenant_id": "acme",
        },
    )
    assert r.status_code == 400
    assert "no profiles available" in r.json()["detail"]
