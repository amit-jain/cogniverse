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
    def test_create_test_dataset_returns_the_written_name(self, manager, store):
        # Returns the dataset NAME (what callers evaluate by), not the id.
        name = manager.create_test_dataset()

        assert name.startswith("test_dataset_")
        assert name in store._frames
        df = store._frames[name]
        assert len(df) == 3
        assert df["query"].iloc[0] == "person wearing red shirt"
        assert df["expected_videos"].iloc[0] == "video1,video2"

    @pytest.mark.unit
    def test_join_expected_coerces_hostile_values(self):
        # Scalars, dicts and None must never raise or repr-leak.
        assert DatasetManager._join_expected(["v1", "v2"]) == "v1,v2"
        assert DatasetManager._join_expected("v1,v2") == "v1,v2"
        assert DatasetManager._join_expected(42) == "42"
        assert DatasetManager._join_expected(None) == ""
        assert DatasetManager._join_expected({"v1": True}) == ""
        assert DatasetManager._join_expected(("a", "b")) == "a,b"

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
        fetched = manager.get_dataset("gone")
        assert fetched["id"] == "ds-gone"
        assert fetched["dataframe"]["query"].tolist() == ["q"]

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
        assert created_at.tzinfo == timezone.utc
        assert created_at.utcoffset().total_seconds() == 0


class TestTraceManager:
    """Test trace manager functionality."""

    @pytest.fixture
    def manager(self, mock_phoenix_client):
        """Create a trace manager through its storage injection contract."""
        storage = Mock()
        storage.get_traces_for_evaluation = Mock(
            return_value=mock_phoenix_client.spans.get_spans_dataframe()
        )
        storage.update_trace_metadata = Mock(return_value=True)
        return TraceManager(tenant_id="acme:acme", storage=storage)

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
        """Extracts trace dicts from the columns get_spans_dataframe emits:
        context.trace_id for identity, start_time/end_time for timing, and
        output.value as a JSON string."""
        import json

        df = pd.DataFrame(
            [
                {
                    "context.span_id": "span1",
                    "context.trace_id": "trace1",
                    "start_time": pd.Timestamp("2026-01-01T00:00:00Z"),
                    "end_time": pd.Timestamp("2026-01-01T00:00:00.100Z"),
                    "attributes.input.value": "query 1",
                    "attributes.output.value": json.dumps([{"video_id": "v1"}]),
                    "attributes.metadata.profile": "profile1",
                    "attributes.metadata.strategy": "strategy1",
                },
                {
                    "context.span_id": "span2",
                    "context.trace_id": "trace2",
                    "start_time": pd.Timestamp("2026-01-01T00:00:10Z"),
                    "end_time": pd.Timestamp("2026-01-01T00:00:10.200Z"),
                    "attributes.input.value": "query 2",
                    "attributes.output.value": json.dumps([{"video_id": "v2"}]),
                    "attributes.metadata.profile": "profile2",
                    "attributes.metadata.strategy": "strategy2",
                },
            ]
        )

        traces = manager.extract_trace_data(df)

        assert len(traces) == 2
        assert traces[0]["trace_id"] == "trace1"
        assert traces[0]["query"] == "query 1"
        assert traces[0]["profile"] == "profile1"
        assert traces[0]["results"] == [{"video_id": "v1"}]
        assert traces[0]["duration_ms"] == 100.0
        assert str(traces[0]["timestamp"]) == "2026-01-01 00:00:00+00:00"
        assert traces[1]["trace_id"] == "trace2"
        assert traces[1]["duration_ms"] == 200.0
        assert traces[1]["metadata"] == {
            "profile": "profile2",
            "strategy": "strategy2",
        }

    @pytest.mark.unit
    def test_extract_trace_data_missing_fields(self, manager):
        """Missing output/metadata/timing columns yield explicit defaults —
        no fabricated zero duration, no crash."""
        df = pd.DataFrame(
            [
                {
                    "context.trace_id": "trace1",
                    "attributes.input.value": "query 1",
                    # Missing output, metadata, and timing columns
                }
            ]
        )

        traces = manager.extract_trace_data(df)

        assert len(traces) == 1
        assert traces[0]["trace_id"] == "trace1"
        assert traces[0]["results"] == []
        assert traces[0]["profile"] == "unknown"
        assert traces[0]["strategy"] == "unknown"
        assert traces[0]["duration_ms"] is None
        assert traces[0]["timestamp"] is None

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
        """Average duration is derived from the frame's start_time/end_time
        bounds; a row with a missing bound is excluded, not averaged as 0."""
        base = pd.Timestamp("2026-01-01T00:00:00Z")
        df = pd.DataFrame(
            [
                {
                    "context.trace_id": "t1",
                    "attributes.metadata.profile": "p1",
                    "start_time": base,
                    "end_time": base + pd.Timedelta(milliseconds=100),
                },
                {
                    "context.trace_id": "t2",
                    "attributes.metadata.profile": "p1",
                    "start_time": base,
                    "end_time": base + pd.Timedelta(milliseconds=200),
                },
                {
                    "context.trace_id": "t3",
                    "attributes.metadata.profile": "p2",
                    "start_time": base,
                    "end_time": pd.NaT,
                },
            ]
        )
        manager.storage.get_traces_for_evaluation.return_value = df

        stats = manager.get_trace_statistics(hours_back=1)

        assert stats["total_traces"] == 3
        assert stats["average_duration_ms"] == 150.0
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


class TestReaderProjectDerivation:
    """Batch/live trace readers must derive the span project from the LOADED
    telemetry config — the same template the span writers use — so that a
    ``tenant_project_template`` override does not make the readers query a
    project nothing writes to."""

    @pytest.mark.unit
    def test_readers_match_writer_project_under_template_override(self):
        import cogniverse_foundation.telemetry.manager as telemetry_manager_module
        from cogniverse_evaluation.core.solvers import _resolve_project
        from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
        from cogniverse_foundation.telemetry.config import TelemetryConfig
        from cogniverse_foundation.telemetry.manager import (
            TelemetryManager,
            get_telemetry_manager,
        )

        override = "acme-spans-{tenant_id}-v2"
        TelemetryManager.reset()
        telemetry_manager_module._telemetry_manager = TelemetryManager(
            TelemetryConfig(tenant_project_template=override)
        )
        try:
            tenant = "acme:acme"
            # Writer-side derivation — identical to the SpanEvaluator wiring in
            # quality_monitor._make_span_evaluator.
            writer_project = get_telemetry_manager().config.get_project_name(
                canonical_tenant_id(tenant)
            )
            assert writer_project == "acme-spans-acme:acme-v2"

            trace_manager = TraceManager(tenant_id=tenant, storage=Mock())
            assert trace_manager.project_name == writer_project

            assert _resolve_project({"tenant_id": tenant}) == writer_project
        finally:
            TelemetryManager.reset()
            telemetry_manager_module._telemetry_manager = TelemetryManager(
                TelemetryConfig()
            )
