"""HTTP-level coverage for the synthetic-data FastAPI routes.

Routes are tested through ``httpx.ASGITransport`` so the FastAPI
``response_model`` validation, the ``HTTPException`` ladder
(400 → ValueError, 422 → ValidationError, 500 → other), the query-param
validators, and the response envelope all execute. Without these tests
the router shipped untested end-to-end.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

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


def test_health_requires_explicit_service_configuration(client: TestClient) -> None:
    r = client.get("/synthetic/health")
    assert r.status_code == 500
    assert r.json() == {"detail": "Internal server error"}


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


@pytest.mark.parametrize("field", ["count", "vespa_sample_size", "max_profiles"])
@pytest.mark.parametrize("value", [True, False])
def test_generate_rejects_boolean_integer_fields_before_service(
    client: TestClient, monkeypatch, field: str, value: bool
) -> None:
    fake = MagicMock()
    fake.generate = AsyncMock()
    monkeypatch.setattr(synthetic_api, "_service", fake)
    payload = {
        "optimizer": "profile",
        "count": 1,
        "vespa_sample_size": 1,
        "max_profiles": 1,
        "tenant_id": "acme",
    }
    payload[field] = value

    response = client.post("/synthetic/generate", json=payload)

    assert response.status_code == 422
    fake.generate.assert_not_awaited()


@pytest.mark.parametrize(
    "strategy",
    [["diverse"], {"name": "diverse"}],
    ids=["list", "object"],
)
def test_generate_rejects_non_string_strategy_before_service(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    strategy: object,
) -> None:
    fake = MagicMock()
    fake.generate = AsyncMock()
    monkeypatch.setattr(synthetic_api, "_service", fake)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/synthetic/generate",
        json={
            "optimizer": "profile",
            "count": 1,
            "tenant_id": "acme",
            "strategy": strategy,
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "type": "value_error",
                "loc": ["body", "strategy"],
                "msg": "Value error, strategy must be a string",
                "input": strategy,
                "ctx": {"error": {}},
            }
        ]
    }
    fake.generate.assert_not_awaited()


def test_generate_422_on_service_pydantic_error(
    client: TestClient, monkeypatch
) -> None:
    class _RequiredOutput(BaseModel):
        selected_profile: str

    with pytest.raises(ValueError) as captured:
        _RequiredOutput.model_validate({})

    fake = MagicMock()
    fake.generate = AsyncMock(side_effect=captured.value)
    monkeypatch.setattr(synthetic_api, "_service", fake)
    response = client.post(
        "/synthetic/generate",
        json={"optimizer": "profile", "count": 1, "tenant_id": "acme"},
    )

    assert response.status_code == 422
    assert "selected_profile" in response.json()["detail"]


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


def test_batch_generate_rejects_total_above_request_limit_before_service(
    client: TestClient, monkeypatch
) -> None:
    fake = MagicMock()
    fake.generate = AsyncMock()
    monkeypatch.setattr(synthetic_api, "_service", fake)

    response = client.post(
        "/synthetic/batch/generate",
        params={
            "optimizer": "profile",
            "count_per_batch": 1000,
            "num_batches": 11,
            "tenant_id": "acme",
        },
    )

    assert response.status_code == 422
    fake.generate.assert_not_awaited()


@pytest.mark.parametrize("field", ["count_per_batch", "num_batches"])
@pytest.mark.parametrize("value", [True, False])
def test_batch_generate_rejects_boolean_integer_fields_before_service(
    client: TestClient, monkeypatch, field: str, value: bool
) -> None:
    fake = MagicMock()
    fake.generate = AsyncMock()
    monkeypatch.setattr(synthetic_api, "_service", fake)
    params = {
        "optimizer": "routing",
        "count_per_batch": 1,
        "num_batches": 1,
        "tenant_id": "acme",
    }
    params[field] = value

    response = client.post("/synthetic/batch/generate", params=params)

    assert response.status_code == 422
    fake.generate.assert_not_awaited()


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


def test_batch_generate_rejects_obsolete_plural_strategy_before_service(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = MagicMock()
    fake.generate = AsyncMock()
    monkeypatch.setattr(synthetic_api, "_service", fake)

    response = client.post(
        "/synthetic/batch/generate",
        params={
            "optimizer": "routing",
            "count_per_batch": 1,
            "num_batches": 1,
            "tenant_id": "acme",
            "strategies": "diverse",
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "type": "extra_forbidden",
                "loc": ["query", "strategies"],
                "msg": "Extra inputs are not permitted",
                "input": "diverse",
            }
        ]
    }
    fake.generate.assert_not_awaited()


@pytest.mark.parametrize(
    "strategies",
    [
        ["diverse", "entity_rich"],
        ["diverse", "unsupported"],
        ["unsupported", "diverse"],
    ],
    ids=["valid-values", "valid-then-invalid", "invalid-then-valid"],
)
def test_batch_generate_rejects_repeated_strategy_before_service(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    strategies: list[str],
) -> None:
    service = MagicMock()
    service.generate = AsyncMock()
    get_service = MagicMock(return_value=service)
    monkeypatch.setattr(synthetic_api, "get_service", get_service)

    response = client.post(
        "/synthetic/batch/generate",
        params=[
            ("optimizer", "routing"),
            ("count_per_batch", "1"),
            ("num_batches", "1"),
            ("tenant_id", "acme"),
            *(("strategy", strategy) for strategy in strategies),
        ],
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "type": "multiple_argument_values",
                "loc": ["query", "strategy"],
                "msg": "Query parameter 'strategy' must be provided at most once",
                "input": strategies,
            }
        ]
    }
    get_service.assert_not_called()
    service.generate.assert_not_awaited()


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("optimizer", ["routing", "profile"]),
        ("count_per_batch", ["1", "2"]),
        ("num_batches", ["1", "2"]),
        ("vespa_sample_size", ["1", "2"]),
        ("max_profiles", ["1", "2"]),
        ("tenant_id", ["acme", "other"]),
    ],
)
def test_batch_generate_rejects_every_other_repeated_query_field_before_service(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    values: list[str],
) -> None:
    service = MagicMock()
    service.generate = AsyncMock()
    get_service = MagicMock(return_value=service)
    monkeypatch.setattr(synthetic_api, "get_service", get_service)
    base = {
        "optimizer": "routing",
        "count_per_batch": "1",
        "num_batches": "1",
        "vespa_sample_size": "1",
        "max_profiles": "1",
        "tenant_id": "acme",
    }
    params = [(key, value) for key, value in base.items() if key != field]
    params.extend((field, value) for value in values)

    response = client.post("/synthetic/batch/generate", params=params)

    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "type": "multiple_argument_values",
                "loc": ["query", field],
                "msg": f"Query parameter '{field}' must be provided at most once",
                "input": values,
            }
        ]
    }
    get_service.assert_not_called()
    service.generate.assert_not_awaited()


@pytest.mark.parametrize(
    "strategy",
    ["", "unsupported", "null"],
    ids=["empty", "unsupported", "literal-null"],
)
def test_batch_generate_rejects_invalid_strategy_before_service(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
) -> None:
    get_service = MagicMock(side_effect=AssertionError("service lookup was called"))
    monkeypatch.setattr(synthetic_api, "get_service", get_service)

    response = client.post(
        "/synthetic/batch/generate",
        params={
            "optimizer": "routing",
            "count_per_batch": 1,
            "num_batches": 1,
            "tenant_id": "acme",
            "strategy": strategy,
        },
    )

    allowed = "diverse, entity_rich, multi_modal_sequences, temporal_recent"
    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "type": "value_error",
                "loc": ["query", "strategy"],
                "msg": (
                    f"Value error, Unsupported sampling strategy: {strategy}. "
                    f"Allowed: {allowed}"
                ),
                "input": strategy,
                "ctx": {"error": {}},
            }
        ]
    }
    get_service.assert_not_called()


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


def test_batch_generate_builds_one_unique_pool_then_splits_batches(
    client: TestClient, monkeypatch
) -> None:
    examples = [{"query": f"grounded query {index}"} for index in range(6)]
    fake = MagicMock()
    fake.generate = AsyncMock(
        return_value=SimpleNamespace(
            data=examples,
            count=6,
            selected_profiles=["video_profile"],
        )
    )
    monkeypatch.setattr(synthetic_api, "_service", fake)

    response = client.post(
        "/synthetic/batch/generate",
        params={
            "optimizer": "profile",
            "count_per_batch": 2,
            "num_batches": 3,
            "tenant_id": "acme",
        },
    )

    assert response.status_code == 200
    assert response.json()["data"] == examples
    assert response.json()["batches"] == [
        {"batch_index": 0, "count": 2, "profiles": ["video_profile"]},
        {"batch_index": 1, "count": 2, "profiles": ["video_profile"]},
        {"batch_index": 2, "count": 2, "profiles": ["video_profile"]},
    ]
    fake.generate.assert_awaited_once()
    generated_request = fake.generate.await_args.args[0]
    assert generated_request.count == 6


@pytest.mark.parametrize(
    ("examples", "detail"),
    [
        (
            [
                {
                    "query": "same grounded query",
                    "selected_profile": "video_profile",
                    "timestamp": "2026-08-05T10:00:00Z",
                    "metadata": {"request_id": "first"},
                },
                {
                    "query": "same grounded query",
                    "selected_profile": "video_profile",
                    "timestamp": "2026-08-05T10:00:01Z",
                    "metadata": {"request_id": "second"},
                },
            ],
            "Batch generation returned duplicate query 'same grounded query'",
        ),
        (
            [
                {
                    "query": "same grounded query",
                    "selected_profile": "video_profile",
                },
                {
                    "query": "same grounded query",
                    "selected_profile": "document_profile",
                },
            ],
            "Batch generation returned conflicting outputs for query "
            "'same grounded query'",
        ),
    ],
    ids=["volatile-fields-cannot-hide-duplicate", "conflicting-output"],
)
def test_batch_generate_rejects_repeated_query_identity_across_batches(
    client: TestClient, monkeypatch, examples: list[dict], detail: str
) -> None:
    fake = MagicMock()
    fake.generate = AsyncMock(
        return_value=SimpleNamespace(
            data=examples,
            count=2,
            selected_profiles=["video_profile"],
        )
    )
    monkeypatch.setattr(synthetic_api, "_service", fake)

    response = client.post(
        "/synthetic/batch/generate",
        params={
            "optimizer": "profile",
            "count_per_batch": 1,
            "num_batches": 2,
            "tenant_id": "acme",
        },
    )

    assert response.status_code == 400
    assert response.json() == {"detail": detail}
    fake.generate.assert_awaited_once()


def test_get_service_rejects_unconfigured_access() -> None:
    with pytest.raises(
        RuntimeError,
        match="^SyntheticDataService is not configured$",
    ):
        synthetic_api.get_service()


def test_configure_service_publishes_one_instance_to_concurrent_readers(
    monkeypatch,
) -> None:
    worker_count = 8
    start = threading.Barrier(worker_count)
    count_lock = threading.Lock()
    constructor_calls = 0
    constructor_kwargs = None

    class _CountingService:
        def __init__(self, *args, **kwargs) -> None:
            nonlocal constructor_calls, constructor_kwargs
            with count_lock:
                constructor_calls += 1
                constructor_kwargs = kwargs
            time.sleep(0.05)

    monkeypatch.setattr(synthetic_api, "SyntheticDataService", _CountingService)

    async def extract_entities(text: str, tenant_id: str):
        raise AssertionError("configuration does not invoke the extractor")

    async def label_profile(query: str, available_profiles: list[str], tenant_id: str):
        raise AssertionError("configuration does not invoke the profile labeler")

    synthetic_api.configure_service(
        object(),
        object(),
        object(),
        object(),
        entity_extractor=extract_entities,
        profile_labeler=label_profile,
    )

    def get_service():
        start.wait()
        return synthetic_api.get_service()

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        services = list(pool.map(lambda _: get_service(), range(worker_count)))

    assert constructor_calls == 1
    assert len({id(service) for service in services}) == 1
    assert constructor_kwargs["entity_extractor"] is extract_entities
    assert constructor_kwargs["profile_labeler"] is label_profile


def test_generate_500_does_not_expose_internal_exception(
    client: TestClient, monkeypatch
) -> None:
    fake = MagicMock()
    fake.generate = AsyncMock(side_effect=RuntimeError("postgres password=top-secret"))
    monkeypatch.setattr(synthetic_api, "_service", fake)

    response = client.post(
        "/synthetic/generate",
        json={
            "optimizer": "profile",
            "count": 1,
            "tenant_id": "acme",
        },
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "top-secret" not in response.text


@pytest.mark.parametrize(
    ("method", "path", "request_kwargs"),
    [
        (
            "post",
            "/synthetic/generate",
            {
                "json": {
                    "optimizer": "routing",
                    "count": 1,
                    "tenant_id": "acme",
                }
            },
        ),
        ("get", "/synthetic/optimizers/routing", {}),
        ("get", "/synthetic/health", {}),
        (
            "post",
            "/synthetic/batch/generate",
            {
                "params": {
                    "optimizer": "routing",
                    "count_per_batch": 1,
                    "num_batches": 1,
                    "tenant_id": "acme",
                }
            },
        ),
    ],
)
def test_service_construction_failure_has_one_sanitized_http_contract(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    path: str,
    request_kwargs: dict,
) -> None:
    class _BrokenService:
        def __init__(self) -> None:
            raise RuntimeError("redis password=top-secret")

    monkeypatch.setattr(synthetic_api, "SyntheticDataService", _BrokenService)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.request(method, path, **request_kwargs)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}
    assert "top-secret" not in response.text
