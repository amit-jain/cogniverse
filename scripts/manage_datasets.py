#!/usr/bin/env python3
"""
Dataset management utility
"""

import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.dataset_manager import DatasetManager

def main():
    parser = argparse.ArgumentParser(description="Manage evaluation datasets")
    parser.add_argument("--list", action="store_true", help="List all datasets")
    parser.add_argument("--create", help="Create dataset with name")
    parser.add_argument("--csv", help="CSV file path")
    parser.add_argument("--info", help="Get info about specific dataset")
    args = parser.parse_args()
    
    dm = DatasetManager()
    
    if args.list:
        datasets = dm.list_datasets()
        if datasets:
            print("\nRegistered datasets:")
            for ds in datasets:
                print(f"\nName: {ds['name']}")
                print(f"  Phoenix ID: {ds['phoenix_id']}")
                print(f"  Created: {ds['created_at']}")
                print(f"  Examples: {ds['num_examples']}")
                if ds['description']:
                    print(f"  Description: {ds['description']}")
        else:
            print("\nNo datasets registered yet")
            
    elif args.create and args.csv:
        dataset_id = dm.get_or_create_dataset(
            name=args.create,
            csv_path=args.csv,
            description=f"Created from {args.csv}"
        )
        print(f"\nDataset '{args.create}' created with ID: {dataset_id}")
        
    elif args.info:
        info = dm.get_dataset_info(args.info)
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