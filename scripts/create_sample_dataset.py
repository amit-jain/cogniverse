#!/usr/bin/env python3
"""
Create sample evaluation datasets
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.dataset_manager import create_sample_dataset_csv

if __name__ == "__main__":
    csv_path = create_sample_dataset_csv()
    print(f"Sample dataset created at: {csv_path}")
    print("\nTo use this dataset, run:")
    print(f"  uv run python scripts/run_experiments_with_visualization.py --dataset-name sample_eval --csv-path {csv_path}")
