"""
Dataset management for evaluation framework.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .storage import TelemetryStorage

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Manages evaluation datasets in Phoenix.
    """

    def __init__(self, storage: Optional[TelemetryStorage] = None):
        """
        Initialize dataset manager.

        Args:
            storage: Phoenix storage instance
        """
        self.storage = storage or TelemetryStorage()
        self.datasets = {}  # Cache of loaded datasets

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
        try:
            # Load CSV
            df = pd.read_csv(csv_path)

            # Validate required columns
            if "query" not in df.columns:
                raise ValueError("CSV must have 'query' column")

            # Convert to query list
            queries = []
            for _, row in df.iterrows():
                query_data = {
                    "query": row["query"],
                    "category": row.get("category", "general"),
                }

                # Parse expected videos
                if "expected_videos" in row and pd.notna(row["expected_videos"]):
                    if isinstance(row["expected_videos"], str):
                        # Split comma-separated values
                        query_data["expected_videos"] = [
                            v.strip() for v in row["expected_videos"].split(",")
                        ]
                    else:
                        query_data["expected_videos"] = []
                else:
                    query_data["expected_videos"] = []

                queries.append(query_data)

            # Create dataset in Phoenix
            dataset_id = self.storage.create_dataset(
                name=dataset_name, queries=queries, description=description
            )

            # Cache dataset info
            self.datasets[dataset_name] = {
                "id": dataset_id,
                "queries": queries,
                "created_at": datetime.now(),
            }

            logger.info(
                f"Created dataset '{dataset_name}' from {csv_path} with {len(queries)} queries"
            )
            return dataset_id

        except Exception as e:
            logger.error(f"Failed to create dataset from CSV: {e}")
            raise

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
        try:
            # Validate queries
            for q in queries:
                if "query" not in q:
                    raise ValueError("Each query must have 'query' field")

            # Create dataset in Phoenix
            dataset_id = self.storage.create_dataset(
                name=dataset_name, queries=queries, description=description
            )

            # Cache dataset info
            self.datasets[dataset_name] = {
                "id": dataset_id,
                "queries": queries,
                "created_at": datetime.now(),
            }

            logger.info(f"Created dataset '{dataset_name}' with {len(queries)} queries")
            return dataset_id

        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            raise

    def get_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get dataset by name.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dataset information
        """
        # Check cache first
        if dataset_name in self.datasets:
            return self.datasets[dataset_name]

        # Load from Phoenix
        dataset = self.storage.get_dataset(dataset_name)
        if dataset:
            # Cache for future use
            self.datasets[dataset_name] = {
                "id": dataset.id,
                "dataset": dataset,
                "loaded_at": datetime.now(),
            }
            return self.datasets[dataset_name]

        return None

    def list_datasets(self) -> List[str]:
        """
        List available datasets.

        Returns:
            List of dataset names
        """
        # This would query Phoenix for available datasets
        # For now, return cached datasets
        return list(self.datasets.keys())

    def create_from_json(
        self, json_path: str, dataset_name: str, description: Optional[str] = None
    ) -> str:
        """
        Create dataset from JSON file.

        Args:
            json_path: Path to JSON file
            dataset_name: Name for the dataset
            description: Dataset description

        Returns:
            Dataset ID
        """
        try:
            with open(json_path, "r") as f:
                data = json.load(f)

            # Handle different JSON formats
            if isinstance(data, list):
                queries = data
            elif isinstance(data, dict) and "queries" in data:
                queries = data["queries"]
            else:
                raise ValueError(
                    "JSON must be a list of queries or dict with 'queries' key"
                )

            return self.create_from_queries(queries, dataset_name, description)

        except Exception as e:
            logger.error(f"Failed to create dataset from JSON: {e}")
            raise

    def update_dataset(
        self, dataset_name: str, new_queries: List[Dict[str, Any]]
    ) -> bool:
        """
        Update existing dataset with new queries.

        Args:
            dataset_name: Name of dataset to update
            new_queries: New queries to add

        Returns:
            True if successful
        """
        try:
            # Get existing dataset
            existing = self.get_dataset(dataset_name)
            if not existing:
                raise ValueError(f"Dataset '{dataset_name}' not found")

            # Combine queries
            dataset_obj = existing.get("dataset")
            if dataset_obj and hasattr(dataset_obj, "examples"):
                # Extract existing queries
                existing_queries = []
                for example in dataset_obj.examples:
                    query_dict = {
                        "query": example.input.get("query"),
                        "expected_items": example.output.get("expected_items", []),
                    }
                    if "category" in example.input:
                        query_dict["category"] = example.input["category"]
                    existing_queries.append(query_dict)

                # Add new queries
                all_queries = existing_queries + new_queries
            else:
                all_queries = new_queries

            # Re-create dataset with all queries
            self.create_from_queries(all_queries, dataset_name)
            return True

        except Exception as e:
            logger.error(f"Failed to update dataset: {e}")
            return False

    def delete_dataset(self, dataset_name: str) -> bool:
        """
        Delete a dataset.

        Args:
            dataset_name: Name of dataset to delete

        Returns:
            True if successful
        """
        try:
            # Remove from cache
            if dataset_name in self.datasets:
                del self.datasets[dataset_name]

            # Note: Phoenix client doesn't have delete_dataset method
            # This would need to be implemented in Phoenix client
            # For now, just remove from cache
            logger.warning(
                f"Dataset '{dataset_name}' removed from cache (Phoenix deletion not implemented)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to delete dataset: {e}")
            return False

    def export_dataset(self, dataset_name: str, output_path: str) -> bool:
        """
        Export dataset to JSON file.

        Args:
            dataset_name: Name of dataset to export
            output_path: Path for output file

        Returns:
            True if successful
        """
        try:
            dataset = self.get_dataset(dataset_name)
            if not dataset:
                raise ValueError(f"Dataset '{dataset_name}' not found")

            # Extract queries from dataset
            dataset_obj = dataset.get("dataset")
            export_data = {"name": dataset_name, "queries": []}

            if dataset_obj and hasattr(dataset_obj, "examples"):
                for example in dataset_obj.examples:
                    query_dict = {
                        "query": example.input.get("query"),
                        "expected_items": example.output.get("expected_items", []),
                    }
                    if "category" in example.input:
                        query_dict["category"] = example.input["category"]
                    export_data["queries"].append(query_dict)

            # Write to file
            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info(f"Exported dataset '{dataset_name}' to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export dataset: {e}")
            return False

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

        return self.create_from_queries(
            queries=test_queries,
            dataset_name=f"test_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description="Test dataset for evaluation framework",
        )
