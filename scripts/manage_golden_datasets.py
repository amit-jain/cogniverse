#!/usr/bin/env python3
"""
Manage golden datasets - merge, update, and export
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))


class GoldenDatasetManager:
    """
    Manages golden datasets - loading, merging, updating
    """
    
    def __init__(self):
        self.datasets_dir = Path("data/golden_datasets")
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
    
    def list_datasets(self) -> List[Path]:
        """List all golden dataset files"""
        json_files = list(self.datasets_dir.glob("*.json"))
        csv_files = list(self.datasets_dir.glob("*.csv"))
        return sorted(json_files + csv_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    def load_dataset(self, path: Path) -> Dict[str, Dict]:
        """Load a golden dataset from file"""
        if path.suffix == ".json":
            with open(path, 'r') as f:
                return json.load(f)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
            dataset = {}
            for _, row in df.iterrows():
                query = row["query"]
                expected_videos = row["expected_videos"].split(",") if row["expected_videos"] else []
                dataset[query] = {
                    "expected_videos": expected_videos,
                    "relevance_scores": {},
                    "difficulty": row.get("difficulty", "medium"),
                    "source": str(path)
                }
            return dataset
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def merge_datasets(self, datasets: List[Dict[str, Dict]], strategy: str = "union") -> Dict[str, Dict]:
        """
        Merge multiple datasets
        
        Args:
            datasets: List of dataset dictionaries
            strategy: 'union' (combine all), 'intersection' (common queries), 
                     'latest' (prefer newer data)
                     
        Returns:
            Merged dataset
        """
        if not datasets:
            return {}
        
        if strategy == "union":
            # Combine all queries
            merged = {}
            for dataset in datasets:
                for query, data in dataset.items():
                    if query not in merged:
                        merged[query] = data
                    else:
                        # Merge expected videos
                        existing_videos = set(merged[query]["expected_videos"])
                        new_videos = set(data["expected_videos"])
                        merged[query]["expected_videos"] = list(existing_videos | new_videos)
                        
                        # Merge relevance scores
                        merged[query]["relevance_scores"].update(data.get("relevance_scores", {}))
                        
                        # Track sources
                        sources = merged[query].get("sources", [])
                        if "source" in data:
                            sources.append(data["source"])
                        merged[query]["sources"] = sources
                        
        elif strategy == "intersection":
            # Only keep queries that appear in all datasets
            if len(datasets) == 1:
                return datasets[0]
            
            common_queries = set(datasets[0].keys())
            for dataset in datasets[1:]:
                common_queries &= set(dataset.keys())
            
            merged = {}
            for query in common_queries:
                # Take from first dataset
                merged[query] = datasets[0][query].copy()
                
        elif strategy == "latest":
            # Simply use the last dataset (assumed to be newest)
            merged = datasets[-1] if datasets else {}
            
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")
        
        return merged
    
    def update_dataset(self, 
                      base_dataset: Dict[str, Dict],
                      updates: Dict[str, Dict],
                      overwrite: bool = False) -> Dict[str, Dict]:
        """
        Update a dataset with new data
        
        Args:
            base_dataset: Base dataset to update
            updates: New data to add/update
            overwrite: Whether to overwrite existing entries
            
        Returns:
            Updated dataset
        """
        updated = base_dataset.copy()
        
        for query, data in updates.items():
            if query not in updated or overwrite:
                updated[query] = data
            else:
                # Merge the data
                existing = updated[query]
                
                # Merge expected videos
                existing_videos = set(existing.get("expected_videos", []))
                new_videos = set(data.get("expected_videos", []))
                existing["expected_videos"] = list(existing_videos | new_videos)
                
                # Merge relevance scores
                existing.setdefault("relevance_scores", {}).update(
                    data.get("relevance_scores", {})
                )
                
                # Update metadata
                if "avg_score" in data:
                    existing["avg_score"] = data["avg_score"]
                if "difficulty" in data:
                    existing["difficulty"] = data["difficulty"]
        
        return updated
    
    def filter_dataset(self,
                      dataset: Dict[str, Dict],
                      min_score: Optional[float] = None,
                      max_score: Optional[float] = None,
                      difficulty: Optional[str] = None,
                      min_videos: Optional[int] = None) -> Dict[str, Dict]:
        """
        Filter dataset based on criteria
        
        Args:
            dataset: Dataset to filter
            min_score: Minimum average score
            max_score: Maximum average score
            difficulty: Difficulty level
            min_videos: Minimum number of expected videos
            
        Returns:
            Filtered dataset
        """
        filtered = {}
        
        for query, data in dataset.items():
            # Apply filters
            if min_score and data.get("avg_score", 1.0) < min_score:
                continue
            if max_score and data.get("avg_score", 0.0) > max_score:
                continue
            if difficulty and data.get("difficulty") != difficulty:
                continue
            if min_videos and len(data.get("expected_videos", [])) < min_videos:
                continue
            
            filtered[query] = data
        
        return filtered
    
    def export_to_code(self, dataset: Dict[str, Dict]) -> str:
        """
        Export dataset as Python code
        
        Args:
            dataset: Dataset to export
            
        Returns:
            Python code string
        """
        code = []
        code.append('"""')
        code.append('Auto-generated golden dataset')
        code.append(f'Generated: {datetime.now().isoformat()}')
        code.append('"""')
        code.append('')
        code.append('GOLDEN_DATASET = {')
        
        for query, data in dataset.items():
            code.append(f'    "{query}": {{')
            code.append(f'        "expected_videos": {data["expected_videos"]},')
            
            if data.get("relevance_scores"):
                code.append('        "relevance_scores": {')
                for video, score in data["relevance_scores"].items():
                    code.append(f'            "{video}": {score},')
                code.append('        },')
            else:
                code.append('        "relevance_scores": {},')
            
            if "difficulty" in data:
                code.append(f'        "difficulty": "{data["difficulty"]}",')
            
            code.append('    },')
        
        code.append('}')
        
        return '\n'.join(code)
    
    def get_statistics(self, dataset: Dict[str, Dict]) -> Dict[str, any]:
        """Get statistics about a dataset"""
        if not dataset:
            return {"total_queries": 0}
        
        stats = {
            "total_queries": len(dataset),
            "total_videos": sum(len(d.get("expected_videos", [])) for d in dataset.values()),
            "queries_with_scores": sum(1 for d in dataset.values() if d.get("relevance_scores")),
            "difficulty_distribution": {},
            "avg_expected_videos": 0
        }
        
        # Count difficulties
        for data in dataset.values():
            difficulty = data.get("difficulty", "unknown")
            stats["difficulty_distribution"][difficulty] = stats["difficulty_distribution"].get(difficulty, 0) + 1
        
        # Average expected videos
        if dataset:
            stats["avg_expected_videos"] = stats["total_videos"] / len(dataset)
        
        # Score statistics if available
        scores = [d.get("avg_score") for d in dataset.values() if "avg_score" in d]
        if scores:
            stats["avg_score"] = sum(scores) / len(scores)
            stats["min_score"] = min(scores)
            stats["max_score"] = max(scores)
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Manage golden datasets")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available datasets")
    
    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge multiple datasets")
    merge_parser.add_argument("files", nargs="+", help="Dataset files to merge")
    merge_parser.add_argument("--strategy", choices=["union", "intersection", "latest"],
                            default="union", help="Merge strategy")
    merge_parser.add_argument("--output", help="Output path")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update dataset with new data")
    update_parser.add_argument("base", help="Base dataset file")
    update_parser.add_argument("updates", help="Updates dataset file")
    update_parser.add_argument("--overwrite", action="store_true", 
                              help="Overwrite existing entries")
    update_parser.add_argument("--output", help="Output path")
    
    # Filter command
    filter_parser = subparsers.add_parser("filter", help="Filter dataset")
    filter_parser.add_argument("file", help="Dataset file to filter")
    filter_parser.add_argument("--min-score", type=float, help="Minimum average score")
    filter_parser.add_argument("--max-score", type=float, help="Maximum average score")
    filter_parser.add_argument("--difficulty", help="Difficulty level")
    filter_parser.add_argument("--min-videos", type=int, help="Minimum expected videos")
    filter_parser.add_argument("--output", help="Output path")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument("file", help="Dataset file")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export dataset as Python code")
    export_parser.add_argument("file", help="Dataset file")
    export_parser.add_argument("--output", help="Output Python file")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    manager = GoldenDatasetManager()
    
    if args.command == "list":
        datasets = manager.list_datasets()
        if datasets:
            print("Available golden datasets:")
            for path in datasets:
                size = path.stat().st_size / 1024
                modified = datetime.fromtimestamp(path.stat().st_mtime)
                print(f"  {path.name:40} {size:8.1f}KB  {modified:%Y-%m-%d %H:%M}")
        else:
            print("No golden datasets found")
    
    elif args.command == "merge":
        # Load datasets
        datasets = []
        for file_path in args.files:
            path = Path(file_path)
            if path.exists():
                dataset = manager.load_dataset(path)
                datasets.append(dataset)
                print(f"Loaded {len(dataset)} queries from {path.name}")
        
        # Merge
        merged = manager.merge_datasets(datasets, args.strategy)
        print(f"\nMerged dataset contains {len(merged)} queries")
        
        # Save
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = manager.datasets_dir / f"merged_dataset_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(merged, f, indent=2)
        print(f"Saved to {output_path}")
    
    elif args.command == "update":
        base = manager.load_dataset(Path(args.base))
        updates = manager.load_dataset(Path(args.updates))
        
        print(f"Base dataset: {len(base)} queries")
        print(f"Updates: {len(updates)} queries")
        
        updated = manager.update_dataset(base, updates, args.overwrite)
        print(f"Updated dataset: {len(updated)} queries")
        
        # Save
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = manager.datasets_dir / f"updated_dataset_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(updated, f, indent=2)
        print(f"Saved to {output_path}")
    
    elif args.command == "filter":
        dataset = manager.load_dataset(Path(args.file))
        print(f"Original dataset: {len(dataset)} queries")
        
        filtered = manager.filter_dataset(
            dataset,
            min_score=args.min_score,
            max_score=args.max_score,
            difficulty=args.difficulty,
            min_videos=args.min_videos
        )
        print(f"Filtered dataset: {len(filtered)} queries")
        
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(filtered, f, indent=2)
            print(f"Saved to {output_path}")
    
    elif args.command == "stats":
        dataset = manager.load_dataset(Path(args.file))
        stats = manager.get_statistics(dataset)
        
        print(f"\nDataset Statistics for {Path(args.file).name}")
        print("=" * 50)
        print(f"Total queries: {stats['total_queries']}")
        print(f"Total expected videos: {stats['total_videos']}")
        print(f"Avg videos per query: {stats['avg_expected_videos']:.1f}")
        print(f"Queries with scores: {stats['queries_with_scores']}")
        
        if "avg_score" in stats:
            print("\nScore Statistics:")
            print(f"  Average: {stats['avg_score']:.3f}")
            print(f"  Min: {stats['min_score']:.3f}")
            print(f"  Max: {stats['max_score']:.3f}")
        
        if stats["difficulty_distribution"]:
            print("\nDifficulty Distribution:")
            for difficulty, count in stats["difficulty_distribution"].items():
                print(f"  {difficulty}: {count}")
    
    elif args.command == "export":
        dataset = manager.load_dataset(Path(args.file))
        code = manager.export_to_code(dataset)
        
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                f.write(code)
            print(f"Exported to {output_path}")
        else:
            print(code)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
