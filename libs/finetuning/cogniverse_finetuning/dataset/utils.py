"""
Shared dataset utilities for LLM and embedding fine-tuning.

Provides:
- Train/val/test splitting (works for all dataset types)
- Upload to Hugging Face Hub
- Upload to S3 (JSONL or Parquet format)
"""

import json
import logging
from io import BytesIO
from typing import Dict, List, Literal, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class DatasetUtils:
    """Utilities for dataset processing (works for both LLM and embedding)."""

    @staticmethod
    def split_dataset(
        data: List[Dict],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split dataset into train/val/test.

        Args:
            data: List of examples (any format: instruction, preference, triplet)
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            shuffle: Whether to shuffle before splitting
            seed: Random seed for reproducibility

        Returns:
            (train, val, test) tuple of lists

        Raises:
            ValueError: If ratios don't sum to 1.0
        """
        import random

        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"Ratios must sum to 1.0, got {total_ratio:.3f} "
                f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
            )

        # Shuffle if requested
        data_copy = data.copy()
        if shuffle:
            random.seed(seed)
            random.shuffle(data_copy)

        # Calculate split indices
        n = len(data_copy)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Split
        train = data_copy[:train_end]
        val = data_copy[train_end:val_end]
        test = data_copy[val_end:]

        logger.info(
            f"Split {n} examples into train={len(train)}, val={len(val)}, test={len(test)}"
        )

        return train, val, test

    @staticmethod
    def upload_to_hf_hub(
        data: List[Dict],
        repo_id: str,
        split: str = "train",
        token: Optional[str] = None,
    ):
        """
        Upload dataset to Hugging Face Hub.

        Args:
            data: List of examples
            repo_id: HF Hub repository ID (e.g., "org/dataset-name")
            split: Split name ("train", "validation", "test")
            token: HF Hub token (optional if logged in via `huggingface-cli login`)

        Example:
            >>> upload_to_hf_hub(
            ...     data=train_examples,
            ...     repo_id="cogniverse/routing-agent-sft",
            ...     split="train",
            ...     token=os.getenv("HF_TOKEN")
            ... )
        """
        from datasets import Dataset

        dataset = Dataset.from_list(data)

        logger.info(f"Uploading {len(data)} examples to {repo_id} (split={split})...")

        dataset.push_to_hub(repo_id, split=split, token=token)

        logger.info(f"Upload complete: https://huggingface.co/datasets/{repo_id}")

    @staticmethod
    def upload_to_s3(
        data: List[Dict],
        bucket: str,
        key: str,
        format: Literal["jsonl", "parquet"] = "jsonl",
    ):
        """
        Upload dataset to S3.

        Args:
            data: List of examples
            bucket: S3 bucket name
            key: S3 key (path within bucket)
            format: Output format ("jsonl" or "parquet")

        Example:
            >>> upload_to_s3(
            ...     data=train_examples,
            ...     bucket="cogniverse-datasets",
            ...     key="finetuning/routing-agent/train.jsonl",
            ...     format="jsonl"
            ... )
        """
        import boto3

        s3 = boto3.client("s3")

        logger.info(
            f"Uploading {len(data)} examples to s3://{bucket}/{key} (format={format})..."
        )

        if format == "jsonl":
            # JSONL: One JSON object per line
            content = "\n".join(json.dumps(item) for item in data)
            s3.put_object(Bucket=bucket, Key=key, Body=content.encode("utf-8"))
        elif format == "parquet":
            # Parquet: Efficient columnar format
            df = pd.DataFrame(data)
            buffer = BytesIO()
            df.to_parquet(buffer, index=False)
            s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        else:
            raise ValueError(f"Unknown format: {format}. Supported: jsonl, parquet")

        logger.info(f"Upload complete: s3://{bucket}/{key}")


# Convenience functions for common workflows
def prepare_dataset_splits(
    data: List[Dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Split dataset and return as dictionary.

    Args:
        data: List of examples
        train_ratio: Training set proportion
        val_ratio: Validation set proportion
        test_ratio: Test set proportion
        shuffle: Whether to shuffle
        seed: Random seed

    Returns:
        Dictionary with "train", "validation", "test" keys

    Example:
        >>> splits = prepare_dataset_splits(examples)
        >>> train_data = splits["train"]
        >>> val_data = splits["validation"]
        >>> test_data = splits["test"]
    """
    train, val, test = DatasetUtils.split_dataset(
        data, train_ratio, val_ratio, test_ratio, shuffle, seed
    )

    return {"train": train, "validation": val, "test": test}


def upload_splits_to_hf_hub(
    splits: Dict[str, List[Dict]],
    repo_id: str,
    token: Optional[str] = None,
):
    """
    Upload all splits to HF Hub.

    Args:
        splits: Dictionary with "train", "validation", "test" keys
        repo_id: HF Hub repository ID
        token: HF Hub token (optional)

    Example:
        >>> splits = prepare_dataset_splits(examples)
        >>> upload_splits_to_hf_hub(splits, "cogniverse/routing-agent-sft")
    """
    utils = DatasetUtils()

    for split_name, split_data in splits.items():
        if split_data:  # Only upload non-empty splits
            utils.upload_to_hf_hub(
                data=split_data, repo_id=repo_id, split=split_name, token=token
            )
