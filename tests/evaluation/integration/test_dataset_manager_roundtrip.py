"""Round-trip tests for DatasetManager against a real Phoenix dataset store.

DatasetManager is the sync facade used by the eval CLI, the dashboard
optimization tab, and scripts/manage_datasets.py. These tests pin its
contract at the real boundary: datasets created through the manager are
readable back from Phoenix with the exact example shape, missing datasets
map to None, and a dead backend raises instead of masquerading as no-data.
"""

import json
import threading
import uuid

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

TENANT = "acme:dm-roundtrip"


def _unique(name: str) -> str:
    return f"{name}-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def manager(phoenix_container):
    from cogniverse_evaluation.data.datasets import DatasetManager
    from cogniverse_telemetry_phoenix.provider import PhoenixDatasetStore

    store = PhoenixDatasetStore(http_endpoint=phoenix_container["http_endpoint"])
    return DatasetManager(tenant_id=TENANT, dataset_store=store)


class TestDatasetManagerRoundTrip:
    def test_create_from_csv_roundtrip(self, manager, phoenix_container, tmp_path):
        csv_path = tmp_path / "queries.csv"
        csv_path.write_text(
            "query,expected_videos,category\n"
            'person in red shirt,"v1,v2",visual\n'
            "dog in park,v3,activity\n"
        )
        name = _unique("csv-roundtrip")

        dataset_id = manager.create_from_csv(str(csv_path), name)

        assert isinstance(dataset_id, str) and dataset_id

        # The manager's own read-back
        loaded = manager.get_dataset(name)
        assert loaded is not None
        df = loaded["dataframe"]
        assert len(df) == 2

        # The raw boundary: input/output key classification must round-trip
        from phoenix.client import Client

        raw = Client(base_url=phoenix_container["http_endpoint"]).datasets.get_dataset(
            dataset=name
        )
        examples = list(raw.examples)
        assert len(examples) == 2
        by_query = {ex["input"]["query"]: ex for ex in examples}
        assert set(by_query) == {"person in red shirt", "dog in park"}
        assert by_query["person in red shirt"]["input"]["category"] == "visual"
        # Lists persist comma-joined — the form core.ground_truth and
        # core.task split back into item lists.
        assert by_query["person in red shirt"]["output"]["expected_videos"] == "v1,v2"
        assert by_query["dog in park"]["output"]["expected_videos"] == "v3"

    def test_export_flattens_phoenix_example_shape(self, manager, tmp_path):
        name = _unique("export")
        manager.create_from_queries(
            [{"query": "exported", "expected_videos": ["v1", "v2"], "category": "c"}],
            name,
        )
        # Force a real read-back so export sees the store's nested shape
        manager.datasets.clear()
        out = tmp_path / "export.json"

        assert manager.export_dataset(name, str(out)) is True

        data = json.loads(out.read_text())
        assert data["name"] == name
        assert data["queries"] == [
            {"query": "exported", "category": "c", "expected_videos": "v1,v2"}
        ]

    def test_create_from_queries_and_json(self, manager, tmp_path):
        queries = [
            {"query": "sunset over water", "expected_videos": ["v9"], "category": "x"}
        ]
        json_path = tmp_path / "queries.json"
        json_path.write_text(json.dumps({"queries": queries}))
        name = _unique("json-roundtrip")

        dataset_id = manager.create_from_json(str(json_path), name)

        assert isinstance(dataset_id, str) and dataset_id
        df = manager.get_dataset(name)["dataframe"]
        assert len(df) == 1

    def test_get_dataset_missing_returns_none(self, manager):
        assert manager.get_dataset(_unique("never-created")) is None

    def test_update_dataset_appends(self, manager):
        name = _unique("update")
        manager.create_from_queries(
            [
                {"query": "q1", "expected_videos": ["a"], "category": "c"},
                {"query": "q2", "expected_videos": ["b"], "category": "c"},
            ],
            name,
        )

        assert manager.update_dataset(
            name, [{"query": "q3", "expected_videos": ["c"], "category": "c"}]
        )

        df = manager.get_dataset(name)["dataframe"]
        assert len(df) == 3

    def test_update_missing_dataset_raises(self, manager):
        with pytest.raises(ValueError):
            manager.update_dataset(
                _unique("missing"), [{"query": "q", "expected_videos": []}]
            )

    def test_delete_dataset(self, manager):
        name = _unique("delete")
        manager.create_from_queries([{"query": "q", "expected_videos": []}], name)

        assert manager.delete_dataset(name) is True
        assert manager.get_dataset(name) is None
        assert manager.delete_dataset(name) is False

    def test_concurrent_creates_distinct_datasets(self, manager):
        """N threads creating distinct datasets through ONE manager must all
        succeed with no cross-talk (each sync call runs its own event loop)."""
        names = [_unique(f"conc-{i}") for i in range(4)]
        barrier = threading.Barrier(4)
        errors = []

        def create(name):
            try:
                barrier.wait(timeout=10)
                manager.create_from_queries(
                    [{"query": f"query for {name}", "expected_videos": ["v"]}], name
                )
            except Exception as e:  # collected for assertion in the main thread
                errors.append((name, e))

        threads = [threading.Thread(target=create, args=(n,)) for n in names]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert errors == []
        for name in names:
            df = manager.get_dataset(name)["dataframe"]
            assert len(df) == 1
            assert df.iloc[0]["input"]["query"] == f"query for {name}"


class TestDatasetManagerFaultContract:
    """A dead telemetry backend must raise — never read as 'no datasets'."""

    @pytest.fixture
    def dead_manager(self):
        from cogniverse_evaluation.data.datasets import DatasetManager
        from cogniverse_telemetry_phoenix.provider import PhoenixDatasetStore

        store = PhoenixDatasetStore(http_endpoint="http://127.0.0.1:29071")
        return DatasetManager(tenant_id=TENANT, dataset_store=store)

    def test_create_raises_on_outage(self, dead_manager):
        with pytest.raises(Exception, match="[Cc]onnect|refused|29071"):
            dead_manager.create_from_queries(
                [{"query": "q", "expected_videos": []}], _unique("outage")
            )

    def test_get_raises_on_outage(self, dead_manager):
        with pytest.raises(Exception, match="[Cc]onnect|refused|29071"):
            dead_manager.get_dataset(_unique("outage"))
