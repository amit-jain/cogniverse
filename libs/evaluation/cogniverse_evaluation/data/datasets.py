"""
Dataset management for evaluation framework.

Sync facade over the telemetry provider's async :class:`DatasetStore`,
used by the eval CLI, the dashboard optimization tab, and
``scripts/manage_datasets.py``. Reads distinguish a genuinely missing
dataset (``None``) from a backend outage (raises).
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Coroutine, Dict, List, Optional, TypeVar

import pandas as pd

from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.providers.base import (
    DatasetNotFoundError,
    DatasetStore,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

INPUT_KEYS = ["query", "category"]
OUTPUT_KEYS = ["expected_videos"]


class DatasetManager:
    """
    Manages evaluation datasets through the telemetry provider's dataset store.
    """

    def __init__(self, tenant_id: str, dataset_store: Optional[DatasetStore] = None):
        """
        Initialize dataset manager.

        Args:
            tenant_id: Tenant whose dataset store to use (canonicalized here).
            dataset_store: Explicit store; resolved from the tenant's
                evaluation provider when omitted.
        """
        self.tenant_id = canonical_tenant_id(tenant_id)
        self._store = dataset_store or self._resolve_store()
        self.datasets: Dict[str, Dict[str, Any]] = {}  # Cache of loaded datasets

    def _resolve_store(self) -> DatasetStore:
        from cogniverse_evaluation.providers import get_evaluation_provider

        provider = get_evaluation_provider(tenant_id=self.tenant_id)
        telemetry = provider.telemetry
        if telemetry is None:
            raise RuntimeError(
                f"Evaluation provider for tenant '{self.tenant_id}' has no "
                "telemetry provider — dataset operations unavailable"
            )
        return telemetry.datasets

    @staticmethod
    def _run(coro: Coroutine[Any, Any, T]) -> T:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError(
            "DatasetManager is a sync facade and cannot be driven from a "
            "running event loop; call the DatasetStore directly instead"
        )

    @staticmethod
    def _queries_to_dataframe(queries: List[Dict[str, Any]]) -> pd.DataFrame:
        for q in queries:
            if "query" not in q:
                raise ValueError("Each query must have 'query' field")
        # expected_videos is persisted comma-joined: Phoenix stringifies list
        # cells to their Python repr, which downstream parsers
        # (core.ground_truth._resolve_expected_items, core.task) cannot split.
        return pd.DataFrame(
            [
                {
                    "query": q["query"],
                    "category": q.get("category", "general"),
                    "expected_videos": DatasetManager._join_expected(
                        q.get("expected_videos")
                    ),
                }
                for q in queries
            ]
        )

    @staticmethod
    def _join_expected(value: Any) -> str:
        """Comma-join expected ids into the persisted string form.

        Phoenix serializes list cells to their Python repr, so lists are
        joined; downstream (core.ground_truth._resolve_expected_items,
        core.task) splits them back. A scalar is wrapped as a single id; a
        mapping/None becomes empty — never a raw TypeError or a repr-leaked
        nested structure.
        """
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return ""
        if isinstance(value, (list, tuple, set)):
            return ",".join(str(v) for v in value)
        # A bare scalar (int/float/bool) is a single expected id.
        return str(value)

    def create_from_queries(
        self,
        queries: List[Dict[str, Any]],
        dataset_name: str,
        description: Optional[str] = None,
    ) -> str:
        """
        Create dataset from list of queries.

        Args:
            queries: List of query dictionaries
            dataset_name: Name for the dataset
            description: Dataset description

        Returns:
            Dataset ID
        """
        df = self._queries_to_dataframe(queries)
        dataset_id = self._run(
            self._store.create_dataset(
                name=dataset_name,
                data=df,
                metadata={
                    "description": description or "",
                    "input_keys": INPUT_KEYS,
                    "output_keys": OUTPUT_KEYS,
                },
            )
        )

        self.datasets[dataset_name] = {
            "id": dataset_id,
            "queries": queries,
            "created_at": datetime.now(timezone.utc),
        }
        logger.info(f"Created dataset '{dataset_name}' with {len(queries)} queries")
        return dataset_id

    def create_from_csv(
        self, csv_path: str, dataset_name: str, description: Optional[str] = None
    ) -> str:
        """
        Create dataset from CSV file.

        Expected CSV columns:
        - query: Search query
        - expected_videos: Comma-separated list of expected video IDs
        - category: Query category (optional)

        Args:
            csv_path: Path to CSV file
            dataset_name: Name for the dataset
            description: Dataset description

        Returns:
            Dataset ID
        """
        df = pd.read_csv(csv_path)
        if "query" not in df.columns:
            raise ValueError("CSV must have 'query' column")

        queries = []
        for _, row in df.iterrows():
            query_data = {
                "query": row["query"],
                "category": row.get("category", "general"),
            }
            expected = row.get("expected_videos")
            if isinstance(expected, str):
                query_data["expected_videos"] = [v.strip() for v in expected.split(",")]
            else:
                query_data["expected_videos"] = []
            queries.append(query_data)

        dataset_id = self.create_from_queries(queries, dataset_name, description)
        logger.info(
            f"Created dataset '{dataset_name}' from {csv_path} "
            f"with {len(queries)} queries"
        )
        return dataset_id

    def create_from_json(
        self, json_path: str, dataset_name: str, description: Optional[str] = None
    ) -> str:
        """
        Create dataset from JSON file.

        Args:
            json_path: Path to JSON file — either a list of queries or a
                dict with a 'queries' key.
            dataset_name: Name for the dataset
            description: Dataset description

        Returns:
            Dataset ID
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            queries = data
        elif isinstance(data, dict) and "queries" in data:
            queries = data["queries"]
        else:
            raise ValueError(
                "JSON must be a list of queries or dict with 'queries' key"
            )

        return self.create_from_queries(queries, dataset_name, description)

    def get_dataset(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get dataset by name.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dict with ``id``, ``dataframe`` and ``loaded_at``, or None if no
            dataset by that name exists. A backend outage raises.
        """
        cached = self.datasets.get(dataset_name)
        if cached is not None and "dataframe" in cached:
            return cached

        try:
            df = self._run(self._store.get_dataset(dataset_name))
        except DatasetNotFoundError:
            return None

        entry = {
            "id": (cached or {}).get("id", dataset_name),
            "dataframe": df,
            "loaded_at": datetime.now(timezone.utc),
        }
        self.datasets[dataset_name] = entry
        return entry

    def list_datasets(self) -> List[str]:
        """
        List datasets this manager has created or loaded.

        The DatasetStore interface has no enumeration API, so this reflects
        the local cache only.
        """
        return list(self.datasets.keys())

    def update_dataset(
        self, dataset_name: str, new_queries: List[Dict[str, Any]]
    ) -> bool:
        """
        Append new queries to an existing dataset.

        Args:
            dataset_name: Name of dataset to update
            new_queries: New queries to add

        Returns:
            True on success.

        Raises:
            ValueError: If the dataset does not exist.
        """
        df = self._queries_to_dataframe(new_queries)
        self._run(
            self._store.append_to_dataset(
                name=dataset_name,
                data=df,
                metadata={"input_keys": INPUT_KEYS, "output_keys": OUTPUT_KEYS},
            )
        )
        # Drop the stale cache entry so the next get_dataset re-reads
        self.datasets.pop(dataset_name, None)
        logger.info(f"Appended {len(new_queries)} queries to dataset '{dataset_name}'")
        return True

    def delete_dataset(self, dataset_name: str) -> bool:
        """
        Delete a dataset.

        Args:
            dataset_name: Name of dataset to delete

        Returns:
            True if a dataset was deleted, False if none existed.
        """
        self.datasets.pop(dataset_name, None)
        return self._run(self._store.delete_dataset(dataset_name))

    def export_dataset(self, dataset_name: str, output_path: str) -> bool:
        """
        Export dataset to JSON file.

        Args:
            dataset_name: Name of dataset to export
            output_path: Path for output file

        Returns:
            True on success.

        Raises:
            ValueError: If the dataset does not exist.
        """
        dataset = self.get_dataset(dataset_name)
        if not dataset:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        df = dataset["dataframe"]
        records = df.to_dict(orient="records")
        if records and "input" in records[0] and "output" in records[0]:
            # Phoenix example shape (input/output/metadata dicts) — flatten
            records = [
                {**(r.get("input") or {}), **(r.get("output") or {})} for r in records
            ]
        export_data = {"name": dataset_name, "queries": records}
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported dataset '{dataset_name}' to {output_path}")
        return True

    def create_test_dataset(self) -> str:
        """
        Create a test dataset with sample queries.

        Returns:
            Dataset ID
        """
        test_queries = [
            {
                "query": "person wearing red shirt",
                "expected_videos": ["video1", "video2"],
                "category": "visual",
            },
            {
                "query": "what happened after the meeting",
                "expected_videos": ["video3"],
                "category": "temporal",
            },
            {
                "query": "dog playing in the park",
                "expected_videos": ["video4", "video5"],
                "category": "activity",
            },
        ]

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dataset_name = f"test_dataset_{timestamp}"
        self.create_from_queries(
            queries=test_queries,
            dataset_name=dataset_name,
            description="Test dataset for evaluation framework",
        )
        # Return the NAME (not the backend id): callers evaluate by name, and
        # re-deriving it from a second clock read races the UTC timestamp.
        return dataset_name
