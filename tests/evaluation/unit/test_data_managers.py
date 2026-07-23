"""
Unit tests for dataset and trace managers.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, mock_open, patch

import pandas as pd
import pytest

from cogniverse_evaluation.data.datasets import DatasetManager
from cogniverse_evaluation.data.traces import TraceManager
from tests.evaluation.fakes import FailingDatasetStore, InMemoryDatasetStore


class TestDatasetManager:
    """DatasetManager against a real in-memory DatasetStore."""

    @pytest.fixture
    def store(self):
        return InMemoryDatasetStore()

    @pytest.fixture
    def manager(self, store):
        return DatasetManager(tenant_id="acme:unit", dataset_store=store)

    @pytest.mark.unit
    def test_create_from_queries(self, manager, store):
        queries = [
            {"query": "red car", "expected_videos": ["v1", "v2"], "category": "visual"},
            {
                "query": "meeting discussion",
                "expected_videos": ["v3"],
                "category": "audio",
            },
        ]

        dataset_id = manager.create_from_queries(
            queries=queries, dataset_name="test_dataset", description="unit ds"
        )

        assert dataset_id == "ds-test_dataset"
        df = store._frames["test_dataset"]
        assert list(df.columns) == ["query", "category", "expected_videos"]
        assert df["query"].tolist() == ["red car", "meeting discussion"]
        assert df["expected_videos"].tolist() == ["v1,v2", "v3"]
        meta = store.metadata["test_dataset"]
        assert meta["input_keys"] == ["query", "category"]
        assert meta["output_keys"] == ["expected_videos"]
        assert meta["description"] == "unit ds"

    @pytest.mark.unit
    def test_create_from_queries_requires_query_field(self, manager):
        with pytest.raises(ValueError, match="query"):
            manager.create_from_queries(
                queries=[{"expected_videos": []}], dataset_name="bad"
            )

    @pytest.mark.unit
    def test_create_from_csv(self, manager, store, temp_csv_file):
        dataset_id = manager.create_from_csv(
            csv_path=temp_csv_file, dataset_name="csv_dataset"
        )

        assert dataset_id == "ds-csv_dataset"
        df = store._frames["csv_dataset"]
        assert len(df) == 3
        assert df["query"].tolist() == [
            "person wearing red shirt",
            "what happened after the meeting",
            "dog playing in the park",
        ]
        assert df["expected_videos"].tolist() == [
            "video1,video2",
            "video3",
            "video4,video5",
        ]
        assert df["category"].tolist() == ["visual", "temporal", "activity"]

    @pytest.mark.unit
    def test_create_from_csv_missing_query_column_raises(self, manager, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("name,value\na,1\n")

        with pytest.raises(ValueError, match="query"):
            manager.create_from_csv(csv_path=str(bad_csv), dataset_name="bad")

    @pytest.mark.unit
    def test_create_from_json(self, manager, store, temp_json_file):
        dataset_id = manager.create_from_json(
            json_path=temp_json_file, dataset_name="json_dataset"
        )

        assert dataset_id == "ds-json_dataset"
        assert "json_dataset" in store._frames

    @pytest.mark.unit
    def test_create_from_json_dict_form(self, manager, store, tmp_path):
        json_path = tmp_path / "wrapped.json"
        json_path.write_text(
            json.dumps({"queries": [{"query": "q1", "expected_videos": ["v"]}]})
        )

        manager.create_from_json(str(json_path), "wrapped")

        assert store._frames["wrapped"]["query"].tolist() == ["q1"]

    @pytest.mark.unit
    def test_get_dataset_returns_dataframe_and_caches(self, manager, store):
        manager.create_from_queries([{"query": "q", "expected_videos": []}], "cached")

        info = manager.get_dataset("cached")

        assert info["id"] == "ds-cached"
        assert isinstance(info["dataframe"], pd.DataFrame)
        assert len(info["dataframe"]) == 1
        # Cached: a second call must not hit the store again
        store._frames.clear()
        assert manager.get_dataset("cached") is info

    @pytest.mark.unit
    def test_get_dataset_not_found_returns_none(self, manager):
        assert manager.get_dataset("nonexistent") is None

    @pytest.mark.unit
    def test_get_dataset_outage_raises(self, store):
        manager = DatasetManager(
            tenant_id="acme:unit", dataset_store=FailingDatasetStore()
        )

        with pytest.raises(ConnectionError, match="connection refused"):
            manager.get_dataset("anything")

    @pytest.mark.unit
    def test_create_raises_on_store_failure(self):
        manager = DatasetManager(
            tenant_id="acme:unit", dataset_store=FailingDatasetStore()
        )

        with pytest.raises(ConnectionError, match="connection refused"):
            manager.create_from_queries(
                [{"query": "q", "expected_videos": []}], "doomed"
            )

    @pytest.mark.unit
    def test_list_datasets(self, manager):
        manager.create_from_queries([{"query": "a", "expected_videos": []}], "ds_a")
        manager.create_from_queries([{"query": "b", "expected_videos": []}], "ds_b")

        assert sorted(manager.list_datasets()) == ["ds_a", "ds_b"]

    @pytest.mark.unit
    def test_create_test_dataset(self, manager, store):
        dataset_id = manager.create_test_dataset()

        assert dataset_id.startswith("ds-test_dataset_")
        name = dataset_id[len("ds-") :]
        df = store._frames[name]
        assert len(df) == 3
        assert df["query"].iloc[0] == "person wearing red shirt"
        assert df["expected_videos"].iloc[0] == "video1,video2"

    @pytest.mark.unit
    def test_update_dataset_appends(self, manager, store):
        manager.create_from_queries(
            [{"query": "old", "expected_videos": ["v0"]}], "upd"
        )

        assert manager.update_dataset(
            "upd", [{"query": "new", "expected_videos": ["v10"]}]
        )

        df = store._frames["upd"]
        assert df["query"].tolist() == ["old", "new"]
        assert df["expected_videos"].tolist() == ["v0", "v10"]

    @pytest.mark.unit
    def test_update_missing_dataset_raises(self, manager):
        with pytest.raises(ValueError, match="does not exist"):
            manager.update_dataset("ghost", [{"query": "q", "expected_videos": []}])

    @pytest.mark.unit
    def test_delete_dataset(self, manager, store):
        manager.create_from_queries([{"query": "q", "expected_videos": []}], "gone")
        assert manager.get_dataset("gone") is not None

        assert manager.delete_dataset("gone") is True
        assert "gone" not in store._frames
        assert "gone" not in manager.datasets
        assert manager.delete_dataset("gone") is False

    @pytest.mark.unit
    def test_export_dataset(self, manager, tmp_path):
        manager.create_from_queries(
            [{"query": "exported query", "expected_videos": ["v1"], "category": "c"}],
            "exp",
        )
        out = tmp_path / "export.json"

        assert manager.export_dataset("exp", str(out)) is True

        data = json.loads(out.read_text())
        assert data["name"] == "exp"
        assert data["queries"] == [
            {"query": "exported query", "category": "c", "expected_videos": "v1"}
        ]

    @pytest.mark.unit
    def test_export_missing_dataset_raises(self, manager, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            manager.export_dataset("ghost", str(tmp_path / "x.json"))

    @pytest.mark.unit
    def test_cached_timestamps_are_utc_aware(self, manager):
        manager.create_from_queries([{"query": "q", "expected_videos": []}], "tz")

        created_at = manager.datasets["tz"]["created_at"]
        assert created_at.tzinfo is not None
        assert created_at.utcoffset().total_seconds() == 0


class TestTraceManager:
    """Test trace manager functionality."""

    @pytest.fixture
    def manager(self, mock_phoenix_client):
        """Create trace manager with mocked storage."""
        from unittest.mock import AsyncMock

        with patch(
            "cogniverse_evaluation.data.storage.TelemetryStorage"
        ) as mock_storage_class:
            # Create mock storage instance
            mock_storage = Mock()
            # Create mock provider structure
            mock_provider = Mock()
            mock_provider.telemetry.traces.get_spans = AsyncMock(
                return_value=mock_phoenix_client.spans.get_spans_dataframe()
            )
            mock_storage.provider = mock_provider
            from cogniverse_evaluation.data.storage import ConnectionState

            mock_storage.connection_state = ConnectionState.CONNECTED
            mock_storage_class.return_value = mock_storage

            manager = TraceManager()
            # Mock get_traces_for_evaluation to return the mock dataframe directly
            manager.storage.get_traces_for_evaluation = Mock(
                return_value=mock_phoenix_client.spans.get_spans_dataframe()
            )
            # Mock the removed methods that TraceManager still expects
            manager.storage.update_trace_metadata = Mock(return_value=True)
            return manager

    @pytest.mark.unit
    def test_get_recent_traces(self, manager):
        """Test getting recent traces."""
        df = manager.get_recent_traces(hours_back=2, limit=10)

        assert df is not None
        assert not df.empty

        # Check that get_traces_for_evaluation was called with time filter
        call_kwargs = manager.storage.get_traces_for_evaluation.call_args[1]
        assert "start_time" in call_kwargs

        # Check time is roughly 2 hours ago. TraceManager now emits UTC-aware
        # start_time (post-audit naive-datetime sweep); compare against an
        # aware reference so the subtraction does not raise.
        start_time = call_kwargs["start_time"]
        assert start_time.tzinfo is not None
        time_diff = datetime.now(timezone.utc) - start_time
        assert (
            timedelta(hours=1, minutes=30) < time_diff < timedelta(hours=2, minutes=30)
        )

    @pytest.mark.unit
    def test_get_traces_by_ids(self, manager):
        """Test getting specific traces by IDs."""
        trace_ids = ["trace1", "trace2", "trace3"]

        df = manager.get_traces_by_ids(trace_ids)

        assert df is not None
        # Check that get_traces_for_evaluation was called once per trace_id
        assert manager.storage.get_traces_for_evaluation.call_count == 3
        # Verify each trace_id was requested
        all_calls = manager.storage.get_traces_for_evaluation.call_args_list
        requested_ids = [call[1]["trace_ids"][0] for call in all_calls]
        assert set(requested_ids) == set(trace_ids)

    @pytest.mark.unit
    def test_extract_trace_data(self, manager):
        """Test extracting trace data from dataframe."""
        df = pd.DataFrame(
            [
                {
                    "trace_id": "trace1",
                    "attributes.input.value": "query 1",
                    "attributes.output.value": [{"video_id": "v1"}],
                    "attributes.metadata.profile": "profile1",
                    "attributes.metadata.strategy": "strategy1",
                    "timestamp": datetime.now(),
                    "duration_ms": 100,
                },
                {
                    "trace_id": "trace2",
                    "attributes.input.value": "query 2",
                    "attributes.output.value": [{"video_id": "v2"}],
                    "attributes.metadata.profile": "profile2",
                    "attributes.metadata.strategy": "strategy2",
                    "timestamp": datetime.now(),
                    "duration_ms": 200,
                },
            ]
        )

        traces = manager.extract_trace_data(df)

        assert len(traces) == 2
        assert traces[0]["trace_id"] == "trace1"
        assert traces[0]["query"] == "query 1"
        assert traces[0]["profile"] == "profile1"
        assert traces[1]["duration_ms"] == 200

    @pytest.mark.unit
    def test_extract_trace_data_missing_fields(self, manager):
        """Test extracting trace data with missing fields."""
        df = pd.DataFrame(
            [
                {
                    "trace_id": "trace1",
                    "attributes.input.value": "query 1",
                    # Missing output and metadata
                }
            ]
        )

        traces = manager.extract_trace_data(df)

        assert len(traces) == 1
        assert traces[0]["results"] == []
        assert traces[0]["profile"] == "unknown"
        assert traces[0]["strategy"] == "unknown"

    @pytest.mark.unit
    def test_get_traces_by_experiment(self, manager):
        """Filtering happens client-side on the returned frame — the storage
        layer takes no filter expression at all."""
        manager.storage.get_traces_for_evaluation = Mock(
            return_value=pd.DataFrame(
                [
                    {
                        "trace_id": "keep",
                        "attributes.metadata.profile": "test_profile",
                        "attributes.metadata.strategy": "test_strategy",
                    },
                    {
                        "trace_id": "drop",
                        "attributes.metadata.profile": "other",
                        "attributes.metadata.strategy": "test_strategy",
                    },
                ]
            )
        )

        traces = manager.get_traces_by_experiment(
            profile="test_profile", strategy="test_strategy", hours_back=24
        )

        assert traces["trace_id"].tolist() == ["keep"]
        call_kwargs = manager.storage.get_traces_for_evaluation.call_args[1]
        assert "filter_condition" not in call_kwargs

    @pytest.mark.unit
    def test_get_trace_statistics(self, manager):
        """Test getting trace statistics."""
        # Mock dataframe with various traces
        df = pd.DataFrame(
            [
                {
                    "trace_id": "t1",
                    "attributes.metadata.profile": "p1",
                    "duration_ms": 100,
                },
                {
                    "trace_id": "t2",
                    "attributes.metadata.profile": "p1",
                    "duration_ms": 200,
                },
                {
                    "trace_id": "t3",
                    "attributes.metadata.profile": "p2",
                    "duration_ms": 150,
                },
            ]
        )
        manager.storage.get_traces_for_evaluation.return_value = df

        stats = manager.get_trace_statistics(hours_back=1)

        assert stats["total_traces"] == 3
        assert stats["average_duration_ms"] == 150
        assert "profiles" in stats
        assert stats["profiles"]["p1"] == 2
        assert stats["profiles"]["p2"] == 1

    @pytest.mark.unit
    def test_export_traces(self, manager):
        """Test exporting traces to file."""
        traces = [
            {"trace_id": "t1", "query": "test1"},
            {"trace_id": "t2", "query": "test2"},
        ]

        with patch("builtins.open", mock_open()) as mock_file:
            with patch.object(manager, "get_recent_traces"):
                with patch.object(manager, "extract_trace_data", return_value=traces):
                    success = manager.export_traces(
                        output_path="traces.json", hours_back=1
                    )

                    assert success is True
                    mock_file.assert_called_with("traces.json", "w")

                    # Verify JSON was written
                    handle = mock_file()
                    assert handle.write.called
