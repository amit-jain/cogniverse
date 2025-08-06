#!/usr/bin/env python3
"""
Automatically create golden dataset from low-scoring traces in Phoenix

This script analyzes historical traces to identify queries that consistently
receive low evaluation scores, making them good candidates for a challenging
golden dataset.
"""

import sys
from pathlib import Path
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json
import pandas as pd
from collections import defaultdict
import argparse

sys.path.append(str(Path(__file__).parent.parent))

import phoenix as px

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GoldenDatasetGenerator:
    """
    Generates golden dataset from low-scoring traces
    """
    
    def __init__(self, 
                 hours_back: int = 48,
                 min_occurrences: int = 2,
                 score_threshold: float = 0.5,
                 top_n_queries: int = 20):
        """
        Initialize the generator
        
        Args:
            hours_back: How many hours back to analyze
            min_occurrences: Minimum times a query must appear
            score_threshold: Maximum avg score to be considered "low-scoring"
            top_n_queries: Number of queries to include in dataset
        """
        self.hours_back = hours_back
        self.min_occurrences = min_occurrences
        self.score_threshold = score_threshold
        self.top_n_queries = top_n_queries
        self.client = px.Client()
    
    def fetch_traces_with_evaluations(self) -> pd.DataFrame:
        """
        Fetch traces with evaluation scores from Phoenix
        
        Returns:
            DataFrame with traces and their evaluations
        """
        logger.info(f"Fetching traces from last {self.hours_back} hours...")
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=self.hours_back)
        
        try:
            # Get spans from Phoenix
            spans_df = self.client.get_spans_dataframe(
                start_time=start_time,
                end_time=end_time
            )
            
            if spans_df is None or spans_df.empty:
                logger.warning("No spans found in the specified time range")
                return pd.DataFrame()
            
            logger.info(f"Retrieved {len(spans_df)} spans")
            
            # Get evaluations
            try:
                evaluations_df = self.client.get_evaluations_dataframe()
                
                if evaluations_df is not None and not evaluations_df.empty:
                    # Merge spans with evaluations
                    merged_df = pd.merge(
                        spans_df,
                        evaluations_df,
                        left_index=True,
                        right_on="context.span_id",
                        how="left"
                    )
                    logger.info(f"Found {len(evaluations_df)} evaluations")
                    return merged_df
                else:
                    logger.warning("No evaluations found")
                    return spans_df
            except Exception as e:
                logger.warning(f"Could not fetch evaluations: {e}")
                return spans_df
                
        except Exception as e:
            logger.error(f"Failed to fetch traces: {e}")
            return pd.DataFrame()
    
    def analyze_query_performance(self, traces_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyze query performance from traces
        
        Args:
            traces_df: DataFrame with traces and evaluations
            
        Returns:
            Dictionary mapping queries to performance metrics
        """
        query_stats = defaultdict(lambda: {
            "occurrences": 0,
            "scores": [],
            "results": [],
            "timestamps": [],
            "profiles": set(),
            "strategies": set()
        })
        
        for _, row in traces_df.iterrows():
            try:
                # Extract query from input
                input_data = row.get("input", {})
                if isinstance(input_data, str):
                    try:
                        input_data = json.loads(input_data)
                    except:
                        continue
                
                query = input_data.get("query", "")
                if not query:
                    continue
                
                # Extract evaluation score
                score = row.get("score")
                if score is None:
                    # Try to get from evaluation columns
                    score = row.get("evaluation_score")
                
                if score is not None:
                    query_stats[query]["scores"].append(float(score))
                
                # Extract results
                output_data = row.get("output", {})
                if isinstance(output_data, str):
                    try:
                        output_data = json.loads(output_data)
                    except:
                        pass
                
                if isinstance(output_data, dict):
                    results = output_data.get("results", [])
                    if results:
                        # Store top results for this query
                        top_videos = []
                        for r in results[:5]:
                            if isinstance(r, dict):
                                video_id = r.get("video_id") or r.get("source_id")
                                if video_id:
                                    top_videos.append(video_id)
                        
                        if top_videos:
                            query_stats[query]["results"].append(top_videos)
                
                # Track metadata
                query_stats[query]["occurrences"] += 1
                query_stats[query]["timestamps"].append(row.get("start_time"))
                
                # Extract profile and strategy from attributes
                attrs = row.get("attributes", {})
                if isinstance(attrs, str):
                    try:
                        attrs = json.loads(attrs)
                    except:
                        attrs = {}
                
                if attrs.get("profile"):
                    query_stats[query]["profiles"].add(attrs["profile"])
                if attrs.get("ranking_strategy"):
                    query_stats[query]["strategies"].add(attrs["ranking_strategy"])
                    
            except Exception as e:
                logger.debug(f"Error processing row: {e}")
                continue
        
        # Calculate average scores and filter
        processed_stats = {}
        for query, stats in query_stats.items():
            if stats["occurrences"] >= self.min_occurrences and stats["scores"]:
                avg_score = sum(stats["scores"]) / len(stats["scores"])
                
                # Find most common results (majority vote)
                if stats["results"]:
                    video_counts = defaultdict(int)
                    for result_list in stats["results"]:
                        for video in result_list:
                            video_counts[video] += 1
                    
                    # Get videos that appear in majority of results
                    threshold = len(stats["results"]) / 2
                    expected_videos = [
                        video for video, count in video_counts.items()
                        if count >= threshold
                    ]
                else:
                    expected_videos = []
                
                processed_stats[query] = {
                    "avg_score": avg_score,
                    "min_score": min(stats["scores"]),
                    "max_score": max(stats["scores"]),
                    "occurrences": stats["occurrences"],
                    "expected_videos": expected_videos,
                    "profiles_tested": list(stats["profiles"]),
                    "strategies_tested": list(stats["strategies"])
                }
        
        return processed_stats
    
    def identify_challenging_queries(self, query_stats: Dict[str, Dict]) -> List[Tuple[str, Dict]]:
        """
        Identify challenging queries based on low scores
        
        Args:
            query_stats: Query performance statistics
            
        Returns:
            List of (query, stats) tuples for challenging queries
        """
        # Filter for low-scoring queries
        low_scoring = [
            (query, stats) for query, stats in query_stats.items()
            if stats["avg_score"] <= self.score_threshold
        ]
        
        # Sort by average score (lowest first)
        low_scoring.sort(key=lambda x: x[1]["avg_score"])
        
        # Take top N
        return low_scoring[:self.top_n_queries]
    
    def create_golden_dataset(self, challenging_queries: List[Tuple[str, Dict]]) -> Dict[str, Dict]:
        """
        Create golden dataset from challenging queries
        
        Args:
            challenging_queries: List of challenging query tuples
            
        Returns:
            Golden dataset dictionary
        """
        golden_dataset = {}
        
        for query, stats in challenging_queries:
            golden_dataset[query] = {
                "expected_videos": stats["expected_videos"],
                "relevance_scores": {},  # Can be populated manually later
                "difficulty": "challenging",
                "avg_score": stats["avg_score"],
                "occurrences": stats["occurrences"],
                "profiles_tested": stats["profiles_tested"],
                "strategies_tested": stats["strategies_tested"],
                "auto_generated": True,
                "generation_date": datetime.now().isoformat()
            }
            
            # Add relevance scores for expected videos (default to 1.0)
            for video in stats["expected_videos"]:
                golden_dataset[query]["relevance_scores"][video] = 1.0
        
        return golden_dataset
    
    def save_golden_dataset(self, dataset: Dict[str, Dict], output_path: Optional[Path] = None) -> Path:
        """
        Save golden dataset to file
        
        Args:
            dataset: Golden dataset dictionary
            output_path: Optional output path
            
        Returns:
            Path where dataset was saved
        """
        if output_path is None:
            output_dir = Path("data/golden_datasets")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"auto_golden_dataset_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Saved golden dataset to {output_path}")
        return output_path
    
    def save_as_csv(self, dataset: Dict[str, Dict], output_path: Optional[Path] = None) -> Path:
        """
        Save golden dataset as CSV for easier editing
        
        Args:
            dataset: Golden dataset dictionary
            output_path: Optional output path
            
        Returns:
            Path where CSV was saved
        """
        if output_path is None:
            output_dir = Path("data/golden_datasets")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"auto_golden_dataset_{timestamp}.csv"
        
        # Convert to DataFrame
        rows = []
        for query, data in dataset.items():
            rows.append({
                "query": query,
                "expected_videos": ",".join(data["expected_videos"]),
                "difficulty": data["difficulty"],
                "avg_score": data["avg_score"],
                "occurrences": data["occurrences"],
                "auto_generated": data["auto_generated"]
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved golden dataset CSV to {output_path}")
        return output_path
    
    def generate_report(self, dataset: Dict[str, Dict]) -> str:
        """
        Generate a report about the created dataset
        
        Args:
            dataset: Golden dataset dictionary
            
        Returns:
            Report string
        """
        report = []
        report.append("=" * 60)
        report.append("GOLDEN DATASET GENERATION REPORT")
        report.append("=" * 60)
        report.append(f"\nGeneration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Analysis Period: Last {self.hours_back} hours")
        report.append(f"Score Threshold: ≤ {self.score_threshold}")
        report.append(f"Min Occurrences: {self.min_occurrences}")
        report.append(f"\nDataset Statistics:")
        report.append(f"  Total Queries: {len(dataset)}")
        
        if dataset:
            avg_scores = [d["avg_score"] for d in dataset.values()]
            report.append(f"  Average Score: {sum(avg_scores)/len(avg_scores):.3f}")
            report.append(f"  Score Range: {min(avg_scores):.3f} - {max(avg_scores):.3f}")
            
            total_videos = sum(len(d["expected_videos"]) for d in dataset.values())
            report.append(f"  Total Expected Videos: {total_videos}")
            
            report.append(f"\nTop 5 Most Challenging Queries:")
            sorted_queries = sorted(dataset.items(), key=lambda x: x[1]["avg_score"])
            for i, (query, data) in enumerate(sorted_queries[:5], 1):
                report.append(f"  {i}. \"{query}\" (score: {data['avg_score']:.3f}, seen: {data['occurrences']}x)")
            
            # Profile coverage
            all_profiles = set()
            for data in dataset.values():
                all_profiles.update(data.get("profiles_tested", []))
            
            if all_profiles:
                report.append(f"\nProfiles Covered: {', '.join(all_profiles)}")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)
    
    def run(self) -> Tuple[Dict[str, Dict], str]:
        """
        Run the complete golden dataset generation process
        
        Returns:
            Tuple of (golden_dataset, report)
        """
        # Fetch traces
        traces_df = self.fetch_traces_with_evaluations()
        if traces_df.empty:
            logger.warning("No traces found")
            return {}, "No traces found to analyze"
        
        # Analyze performance
        logger.info("Analyzing query performance...")
        query_stats = self.analyze_query_performance(traces_df)
        logger.info(f"Analyzed {len(query_stats)} unique queries")
        
        # Identify challenging queries
        logger.info("Identifying challenging queries...")
        challenging = self.identify_challenging_queries(query_stats)
        logger.info(f"Found {len(challenging)} challenging queries")
        
        # Create golden dataset
        logger.info("Creating golden dataset...")
        golden_dataset = self.create_golden_dataset(challenging)
        
        # Generate report
        report = self.generate_report(golden_dataset)
        
        return golden_dataset, report


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden dataset from low-scoring traces"
    )
    parser.add_argument(
        "--hours", 
        type=int, 
        default=48,
        help="Hours back to analyze (default: 48)"
    )
    parser.add_argument(
        "--min-occurrences",
        type=int,
        default=2,
        help="Minimum times a query must appear (default: 2)"
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Maximum avg score for challenging queries (default: 0.5)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of queries to include (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for golden dataset"
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also save as CSV"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't save files, just print report"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = GoldenDatasetGenerator(
        hours_back=args.hours,
        min_occurrences=args.min_occurrences,
        score_threshold=args.score_threshold,
        top_n_queries=args.top_n
    )
    
    # Run generation
    dataset, report = generator.run()
    
    # Print report
    print(report)
    
    if dataset and not args.dry_run:
        # Save dataset
        output_path = Path(args.output) if args.output else None
        json_path = generator.save_golden_dataset(dataset, output_path)
        print(f"\n✅ Saved JSON dataset to: {json_path}")
        
        if args.csv:
            csv_path = generator.save_as_csv(dataset)
            print(f"✅ Saved CSV dataset to: {csv_path}")
        
        # Update golden dataset in code if desired
        print("\n📝 To use this dataset in experiments:")
        print(f"  1. Copy the dataset to src/evaluation/evaluators/golden_dataset.py")
        print(f"  2. Or load from file: pd.read_json('{json_path}')")
    elif not dataset:
        print("\n⚠️  No challenging queries found. Try:")
        print("  - Increasing --hours to analyze more data")
        print("  - Decreasing --min-occurrences")
        print("  - Increasing --score-threshold")
    
    return 0 if dataset else 1


if __name__ == "__main__":
    sys.exit(main())