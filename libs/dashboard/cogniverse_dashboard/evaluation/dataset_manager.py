"""
Dataset manager for experiments

Handles:
- Loading datasets from CSV files
- Maintaining persistent dataset registry
- Reusing existing datasets
- Future: Automatic golden dataset creation from low-scoring traces

Uses telemetry provider abstraction for backend-agnostic dataset storage.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from cogniverse_core.telemetry.manager import TelemetryManager
from cogniverse_core.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


class DatasetManager:
    """Manages datasets for experiments using telemetry provider abstraction"""

    def __init__(
        self,
        registry_path: Optional[Path] = None,
        provider: Optional[TelemetryProvider] = None
    ):
        """
        Initialize dataset manager

        Args:
            registry_path: Path to dataset registry JSON file
            provider: Telemetry provider (if None, uses TelemetryManager's provider)
        """
        # Get provider from TelemetryManager if not provided
        if provider is None:
            telemetry_manager = TelemetryManager()
            provider = telemetry_manager.provider

        self.provider = provider
        self.registry_path = registry_path or Path("configs/dataset_registry.json")
        self.registry = self._load_registry()
        
    def _load_registry(self) -> Dict[str, Any]:
        """Load dataset registry from file"""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {
            "datasets": {},
            "last_updated": None
        }
    
    def _save_registry(self):
        """Save dataset registry to file"""
        self.registry["last_updated"] = datetime.now().isoformat()
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
            
    async def get_or_create_dataset(
        self,
        name: str,
        csv_path: Optional[Union[str, Path]] = None,
        dataframe: Optional[pd.DataFrame] = None,
        description: Optional[str] = None,
        force_new: bool = False
    ) -> str:
        """
        Get existing dataset or create new one

        Args:
            name: Dataset name/identifier
            csv_path: Path to CSV file with queries and expected results
            dataframe: DataFrame with queries and expected results
            description: Dataset description
            force_new: Force creation of new dataset even if one exists

        Returns:
            Backend dataset ID
        """
        # Check registry for existing dataset
        if not force_new and name in self.registry["datasets"]:
            dataset_info = self.registry["datasets"][name]

            # Support both old (phoenix_id) and new (backend_id) registry formats
            dataset_id = dataset_info.get("backend_id") or dataset_info.get("phoenix_id")

            # If CSV provided but dataset exists, warn user
            if csv_path:
                logger.warning(
                    f"Dataset '{name}' already exists (ID: {dataset_id}). "
                    f"CSV file '{csv_path}' will be ignored. Use --force-new to create new dataset from CSV."
                )

            logger.info(f"Using existing dataset '{name}' (ID: {dataset_id})")
            return dataset_id
        
        # Load data
        if csv_path:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} examples from {csv_path}")
        elif dataframe is not None:
            df = dataframe
        else:
            raise ValueError(
                f"Dataset '{name}' does not exist. "
                "Must provide either csv_path or dataframe to create new dataset."
            )
        
        # Validate required columns
        required_cols = ["query"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        # Determine column types
        input_keys = ["query"]
        output_keys = []
        metadata_keys = []
        
        # Check for expected results columns
        if "expected_videos" in df.columns:
            output_keys.append("expected_videos")
        if "expected_video_ids" in df.columns:
            output_keys.append("expected_video_ids")
            
        # All other columns become metadata
        for col in df.columns:
            if col not in input_keys + output_keys:
                metadata_keys.append(col)
        
        # Create unique dataset name with timestamp
        backend_name = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Upload to telemetry backend via provider
        # Pack Phoenix-specific parameters into metadata for backend compatibility
        dataset_metadata = {
            "description": description or f"Dataset: {name}",
            "input_keys": input_keys,
            "output_keys": output_keys,
            "metadata_keys": metadata_keys,
        }

        dataset_id = await self.provider.datasets.create_dataset(
            name=backend_name,
            data=df,
            metadata=dataset_metadata
        )

        # Update registry
        self.registry["datasets"][name] = {
            "backend_id": dataset_id,
            "backend_name": backend_name,
            "created_at": datetime.now().isoformat(),
            "num_examples": len(df),
            "csv_path": str(csv_path) if csv_path else None,
            "description": description,
            "columns": {
                "input": input_keys,
                "output": output_keys,
                "metadata": metadata_keys
            }
        }
        self._save_registry()

        logger.info(f"Created new dataset '{name}' with {len(df)} examples (ID: {dataset_id})")
        return dataset_id
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all registered datasets"""
        datasets = []
        for name, info in self.registry["datasets"].items():
            # Support both old (phoenix_id) and new (backend_id) registry formats
            dataset_id = info.get("backend_id") or info.get("phoenix_id")
            datasets.append({
                "name": name,
                "dataset_id": dataset_id,
                "created_at": info["created_at"],
                "num_examples": info["num_examples"],
                "description": info.get("description", "")
            })
        return datasets
    
    def get_dataset_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific dataset"""
        return self.registry["datasets"].get(name)
    
    def create_golden_dataset_from_traces(
        self,
        hours: int = 24,
        score_threshold: float = 0.3,
        min_examples: int = 10,
        dataset_name: str = "auto_golden"
    ) -> Optional[str]:
        """
        Create golden dataset from low-scoring traces (FUTURE IMPLEMENTATION)
        
        Args:
            hours: Look back hours for traces
            score_threshold: Maximum score to include in golden dataset
            min_examples: Minimum examples needed
            dataset_name: Name for the dataset
            
        Returns:
            Dataset ID if created, None otherwise
        """
        logger.info("Automatic golden dataset creation from traces not yet implemented")
        # TODO: Implement this functionality
        # 1. Query Phoenix for recent traces with evaluation scores
        # 2. Filter for low-scoring queries
        # 3. Extract query and actual results
        # 4. Create dataset with these as "expected" results for testing
        # 5. Upload as new dataset
        return None


def create_sample_dataset_csv():
    """Create a sample dataset CSV file for testing"""
    sample_data = {
        "query": [
            "person wearing winter clothes",
            "outdoor sports activity", 
            "cooking in kitchen",
            "abstract art patterns",
            "nighttime city scene"
        ],
        "expected_videos": [
            "v_-IMXSEIabMM,v_HWFrgou1LD2Q",
            "v_gkSMwfO1q1I,v_0NIKVT3kmT4",
            "v_J0nA4VgnoCo",
            "",  # No expected results for abstract query
            "v_WFrgou1LD2Q"
        ],
        "difficulty": [
            "easy",
            "medium",
            "easy",
            "hard",
            "medium"
        ],
        "category": [
            "people",
            "sports",
            "activities",
            "abstract",
            "scenes"
        ]
    }
    
    df = pd.DataFrame(sample_data)
    output_path = Path("datasets/sample_evaluation_dataset.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Created sample dataset at {output_path}")
    return output_path
