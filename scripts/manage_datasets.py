#!/usr/bin/env python3
"""
Dataset management utility
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from cogniverse_evaluation.data import DatasetManager


def main():
    parser = argparse.ArgumentParser(description="Manage evaluation datasets")
    parser.add_argument("--list", action="store_true", help="List all datasets")
    parser.add_argument("--create", help="Create dataset with name")
    parser.add_argument("--csv", help="CSV file path")
    parser.add_argument("--info", help="Get info about specific dataset")
    args = parser.parse_args()

    dm = DatasetManager()

    if args.list:
        dataset_names = dm.list_datasets()
        if dataset_names:
            print("\nRegistered datasets:")
            for ds_name in dataset_names:
                info = dm.get_dataset(ds_name)
                print(f"\nName: {ds_name}")
                if info:
                    for key, value in info.items():
                        print(f"  {key}: {value}")
        else:
            print("\nNo datasets registered yet")

    elif args.create and args.csv:
        dataset_id = dm.create_from_csv(
            csv_path=args.csv,
            dataset_name=args.create,
            description=f"Created from {args.csv}",
        )
        print(f"\nDataset '{args.create}' created with ID: {dataset_id}")

    elif args.info:
        info = dm.get_dataset(args.info)
        if info:
            print(f"\nDataset: {args.info}")
            for key, value in info.items():
                print(f"  {key}: {value}")
        else:
            print(f"\nDataset '{args.info}' not found")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
