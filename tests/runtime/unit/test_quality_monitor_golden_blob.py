"""QualityMonitor uses tenant golden-set blobs, not the shipped file."""

from __future__ import annotations

import json

import pytest

from cogniverse_agents.optimizer.golden_set_ground_truth import (
    GoldenSetGroundTruthStoreUnavailableError,
)
from cogniverse_evaluation.quality_monitor import (
    QualityMonitor,
)
from tests.evaluation.fakes import (
    InMemoryDatasetStore,
    StubArtifactManager,
    StubTelemetryProvider,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


SHIPPED_GOLDEN_PATH = (
    "/home/amitjain/source/cogniverse/data/testset/evaluation/"
    "sample_videos_retrieval_queries.json"
)


class _StubSearchResponse:
    def __init__(self, results):
        self.status_code = 200
        self._results = results

    def json(self):
        return {"results": self._results}


class _StubSearchClient:
    def __init__(self, responses_by_query):
        self.calls: list[str] = []
        self._responses_by_query = responses_by_query

    async def post(self, url, json=None):
        query = json["query"]
        self.calls.append(query)
        return _StubSearchResponse(self._responses_by_query[query])


@pytest.mark.asyncio
async def test_dataset_store_comes_from_telemetry_provider_identity():
    store = InMemoryDatasetStore()
    provider = StubTelemetryProvider(store)
    monitor = QualityMonitor(
        tenant_id="test_tenant",
        runtime_url="http://runtime",
        phoenix_http_endpoint="http://phoenix:6006",
        llm_base_url="http://llm",
        llm_model="test-model",
        golden_dataset_path=SHIPPED_GOLDEN_PATH,
        telemetry_provider=provider,
    )

    assert monitor._get_dataset_store() is store

    import cogniverse_evaluation.quality_monitor as qm

    assert "PhoenixDatasetStore" not in vars(qm)


@pytest.mark.asyncio
async def test_evaluate_golden_set_iterates_loaded_blob_queries_exactly(monkeypatch):
    rows = [
        {
            "query": "find basketball highlights",
            "expected_videos": ["video-a"],
            "ground_truth": "basketball",
            "query_type": "question",
            "source": "tenant_upload",
        },
        {
            "query": "find ocean waves",
            "expected_videos": ["video-b"],
            "ground_truth": "ocean",
            "query_type": "question",
            "source": "tenant_upload",
        },
    ]
    stub_manager = StubArtifactManager(raw=json.dumps(rows))
    monkeypatch.setattr(
        "cogniverse_agents.optimizer.artifact_manager.ArtifactManager",
        lambda *args, **kwargs: stub_manager,
    )
    provider = StubTelemetryProvider(InMemoryDatasetStore())
    monitor = QualityMonitor(
        tenant_id="test_tenant",
        runtime_url="http://runtime",
        phoenix_http_endpoint="http://phoenix:6006",
        llm_base_url="http://llm",
        llm_model="test-model",
        golden_dataset_path=SHIPPED_GOLDEN_PATH,
        telemetry_provider=provider,
    )
    monitor._http_client = _StubSearchClient(
        {
            "find basketball highlights": [{"source_id": "video-a"}],
            "find ocean waves": [{"source_id": "video-b"}],
        }
    )

    result = await monitor.evaluate_golden_set()

    assert stub_manager.load_calls == [("config", "golden_set_ground_truth")]
    assert monitor._http_client.calls == [
        "find basketball highlights",
        "find ocean waves",
    ]
    assert [entry["query"] for entry in result.per_query_scores] == [
        "find basketball highlights",
        "find ocean waves",
    ]
    assert result.query_count == 2
    assert result.per_query_scores[0]["retrieved_videos"] == ["video-a"]
    assert result.per_query_scores[1]["retrieved_videos"] == ["video-b"]


@pytest.mark.asyncio
async def test_missing_blob_returns_status_without_opening_shipped_file(monkeypatch):
    stub_manager = StubArtifactManager(raw=None)
    monkeypatch.setattr(
        "cogniverse_agents.optimizer.artifact_manager.ArtifactManager",
        lambda *args, **kwargs: stub_manager,
    )
    provider = StubTelemetryProvider(InMemoryDatasetStore())
    monitor = QualityMonitor(
        tenant_id="test_tenant",
        runtime_url="http://runtime",
        phoenix_http_endpoint="http://phoenix:6006",
        llm_base_url="http://llm",
        llm_model="test-model",
        golden_dataset_path=SHIPPED_GOLDEN_PATH,
        telemetry_provider=provider,
    )

    def fail_open(*args, **kwargs):
        raise AssertionError("the shipped golden dataset path was opened")

    monkeypatch.setattr("builtins.open", fail_open)

    result = await monitor.force_optimization_cycle()

    assert result == {
        "status": "golden_set_missing",
        "retryable": False,
        "error": "golden_set_ground_truth is not configured for tenant "
        "test_tenant:test_tenant",
    }
    assert stub_manager.load_calls == [("config", "golden_set_ground_truth")]
    assert stub_manager.save_calls == []
    assert stub_manager.activate_calls == []


@pytest.mark.asyncio
async def test_store_unavailable_propagates_chained(monkeypatch):
    load_exc = ConnectionError("blob store down")
    stub_manager = StubArtifactManager(load_exc=load_exc)
    monkeypatch.setattr(
        "cogniverse_agents.optimizer.artifact_manager.ArtifactManager",
        lambda *args, **kwargs: stub_manager,
    )
    provider = StubTelemetryProvider(InMemoryDatasetStore())
    monitor = QualityMonitor(
        tenant_id="test_tenant",
        runtime_url="http://runtime",
        phoenix_http_endpoint="http://phoenix:6006",
        llm_base_url="http://llm",
        llm_model="test-model",
        golden_dataset_path=SHIPPED_GOLDEN_PATH,
        telemetry_provider=provider,
    )

    with pytest.raises(GoldenSetGroundTruthStoreUnavailableError) as excinfo:
        await monitor.evaluate_golden_set()

    assert str(excinfo.value) == "golden_set_ground_truth store unavailable"
    assert excinfo.value.__cause__ is load_exc
    assert stub_manager.load_calls == [("config", "golden_set_ground_truth")]


@pytest.mark.asyncio
async def test_golden_set_update_uses_versioned_blob(monkeypatch):
    rows = [
        {
            "query": "find basketball highlights",
            "expected_videos": ["video-a"],
            "ground_truth": "basketball",
            "query_type": "question",
            "source": "tenant_upload",
        }
    ]
    stub_manager = StubArtifactManager(raw=json.dumps(rows))
    monkeypatch.setattr(
        "cogniverse_agents.optimizer.artifact_manager.ArtifactManager",
        lambda *args, **kwargs: stub_manager,
    )
    provider = StubTelemetryProvider(InMemoryDatasetStore())
    monitor = QualityMonitor(
        tenant_id="test_tenant",
        runtime_url="http://runtime",
        phoenix_http_endpoint="http://phoenix:6006",
        llm_base_url="http://llm",
        llm_model="test-model",
        golden_dataset_path=SHIPPED_GOLDEN_PATH,
        telemetry_provider=provider,
    )

    await monitor.grow_golden_set(
        [
            {
                "query": "find ocean waves",
                "expected_videos": ["video-b"],
                "ground_truth": "ocean",
                "query_type": "question",
                "source": "live",
            }
        ]
    )

    assert stub_manager.save_calls == [
        {
            "kind": "config",
            "key": "golden_set_ground_truth",
            "content": json.dumps(
                [
                    {
                        "query": "find basketball highlights",
                        "expected_videos": ["video-a"],
                        "ground_truth": "basketball",
                        "query_type": "question",
                        "source": "tenant_upload",
                    },
                    {
                        "query": "find ocean waves",
                        "expected_videos": ["video-b"],
                        "ground_truth": "ocean",
                        "query_type": "question",
                        "source": "live",
                    },
                ],
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            "consumed_example_ids": ["quality_monitor:golden_set_ground_truth"],
            "decision": "promote",
            "scored": False,
            "score": None,
            "base_score": None,
            "candidate_score": None,
        }
    ]
    assert stub_manager.activate_calls == [("config", "golden_set_ground_truth", 1)]
