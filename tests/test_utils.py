"""
Test utilities for formatting and exporting results
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from tabulate import tabulate


class TestResultsFormatter:
    """Format and export test results in various formats"""
    
    def __init__(self, test_name: str, output_dir: Optional[str] = None):
        from src.common.utils.output_manager import get_output_manager
        
        self.test_name = test_name
        self.output_manager = get_output_manager()
        
        # Use provided directory or default to test_results under output manager
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.output_manager.get_test_results_dir()
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def format_table(self, results: List[Dict[str, Any]], 
                    headers: Optional[List[str]] = None,
                    tablefmt: str = "grid") -> str:
        """Format results as a human-readable table
        
        Args:
            results: List of result dictionaries
            headers: Optional custom headers. If None, uses dict keys
            tablefmt: Table format (grid, simple, pretty, html, etc.)
            
        Returns:
            Formatted table string
        """
        if not results:
            return "No results to display"
            
        # Extract headers from first result if not provided
        if headers is None:
            headers = list(results[0].keys())
            
        # Create table data
        table_data = []
        for result in results:
            row = [result.get(h, "") for h in headers]
            table_data.append(row)
            
        return tabulate(table_data, headers=headers, tablefmt=tablefmt)
    
    def save_csv(self, results: List[Dict[str, Any]], 
                 filename: Optional[str] = None) -> Path:
        """Save results to CSV file
        
        Args:
            results: List of result dictionaries
            filename: Optional filename. If None, generates from test name
            
        Returns:
            Path to saved CSV file
        """
        if not results:
            return None
            
        if filename is None:
            filename = f"{self.test_name}_{self.timestamp}.csv"
            
        filepath = self.output_dir / filename
        
        # Get all unique keys from results
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        headers = sorted(list(all_keys))
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
            
        return filepath
    
    def save_json(self, results: List[Dict[str, Any]], 
                  filename: Optional[str] = None) -> Path:
        """Save results to JSON file
        
        Args:
            results: List of result dictionaries
            filename: Optional filename. If None, generates from test name
            
        Returns:
            Path to saved JSON file
        """
        if filename is None:
            filename = f"{self.test_name}_{self.timestamp}.json"
            
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "test_name": self.test_name,
                "timestamp": self.timestamp,
                "results": results
            }, f, indent=2, ensure_ascii=False)
            
        return filepath
    
    def format_comparison_table(self, 
                               results1: List[Dict[str, Any]], 
                               results2: List[Dict[str, Any]],
                               label1: str = "Method 1",
                               label2: str = "Method 2",
                               comparison_fields: List[str] = None) -> str:
        """Format a comparison table between two result sets
        
        Args:
            results1: First set of results
            results2: Second set of results
            label1: Label for first method
            label2: Label for second method
            comparison_fields: Fields to compare. If None, uses common fields
            
        Returns:
            Formatted comparison table
        """
        if not results1 or not results2:
            return "Insufficient results for comparison"
            
        # Determine fields to compare
        if comparison_fields is None:
            fields1 = set(results1[0].keys())
            fields2 = set(results2[0].keys())
            comparison_fields = sorted(list(fields1.intersection(fields2)))
            
        # Create comparison data
        comparison_data = []
        max_len = max(len(results1), len(results2))
        
        for i in range(max_len):
            row = {"Index": i + 1}
            
            # Add data from first result set
            if i < len(results1):
                for field in comparison_fields:
                    row[f"{label1}_{field}"] = results1[i].get(field, "")
            else:
                for field in comparison_fields:
                    row[f"{label1}_{field}"] = "N/A"
                    
            # Add data from second result set
            if i < len(results2):
                for field in comparison_fields:
                    row[f"{label2}_{field}"] = results2[i].get(field, "")
            else:
                for field in comparison_fields:
                    row[f"{label2}_{field}"] = "N/A"
                    
            comparison_data.append(row)
            
        # Format as table
        headers = list(comparison_data[0].keys())
        table_data = [[row.get(h, "") for h in headers] for row in comparison_data]
        
        return tabulate(table_data, headers=headers, tablefmt="grid")
    
    def print_summary(self, results: List[Dict[str, Any]], 
                     summary_stats: Dict[str, Any] = None):
        """Print a summary of results with optional statistics
        
        Args:
            results: List of result dictionaries
            summary_stats: Optional dictionary of summary statistics
        """
        print(f"\n{'='*60}")
        print(f"Test: {self.test_name}")
        print(f"Timestamp: {self.timestamp}")
        print(f"Total Results: {len(results)}")
        
        if summary_stats:
            print(f"\nSummary Statistics:")
            for key, value in summary_stats.items():
                print(f"  {key}: {value}")
                
        print(f"{'='*60}\n")
    
    def format_ranking_strategy_table(self, 
                                    strategy_performance: Dict[str, Dict],
                                    show_details: bool = True,
                                    top_n: int = 3) -> str:
        """Format ranking strategy performance table
        
        Args:
            strategy_performance: Dict mapping strategy names to their performance metrics
            show_details: Whether to show detailed results for top strategies
            top_n: Number of top strategies to show details for
            
        Returns:
            Formatted table string
        """
        # Calculate overall scores for each strategy
        strategy_scores = {}
        for strategy, results in strategy_performance.items():
            if isinstance(results, list):
                # Handle list of results per strategy
                total_score = sum(r.get('top_score', 0) for r in results)
                avg_score = total_score / len(results) if results else 0
                found_count = sum(1 for r in results if r.get('expected_found', False))
                accuracy = found_count / len(results) * 100 if results else 0
                
                strategy_scores[strategy] = {
                    'avg_score': avg_score,
                    'accuracy': accuracy,
                    'found_count': found_count,
                    'total_queries': len(results),
                    'results': results
                }
            else:
                # Handle single result dict per strategy
                strategy_scores[strategy] = results
        
        # Sort strategies by accuracy then by average score
        sorted_strategies = sorted(strategy_scores.items(), 
                                 key=lambda x: (x[1].get('accuracy', 0), x[1].get('avg_score', 0)), 
                                 reverse=True)
        
        # Create summary table
        summary_data = []
        for rank, (strategy, scores) in enumerate(sorted_strategies, 1):
            summary_data.append({
                'Rank': rank,
                'Strategy': strategy,
                'Accuracy': f"{scores.get('accuracy', 0):.1f}%",
                'Found': f"{scores.get('found_count', 0)}/{scores.get('total_queries', 0)}",
                'Avg Score': f"{scores.get('avg_score', 0):.4f}"
            })
        
        output = []
        
        # Main performance table
        output.append("\n" + "="*120)
        output.append("RANKING STRATEGY PERFORMANCE - SORTED BY ACCURACY")
        output.append("="*120)
        output.append(self.format_table(summary_data, tablefmt="grid"))
        
        # Detailed results for top strategies
        if show_details and top_n > 0:
            output.append("\n" + "="*120)
            output.append(f"TOP {min(top_n, len(sorted_strategies))} STRATEGIES - DETAILED PERFORMANCE")
            output.append("="*120)
            
            for strategy, scores in sorted_strategies[:top_n]:
                if 'results' in scores and scores['results']:
                    output.append(f"\n{strategy}:")
                    detail_data = []
                    
                    for result in scores['results'][:5]:  # Show first 5 queries
                        query_display = result.get('query_key', 'N/A')
                        if len(query_display) > 30:
                            query_display = query_display[:30] + "..."
                            
                        detail_data.append({
                            'Query': query_display,
                            'Top Result': result.get('top_video', 'N/A'),
                            'Score': f"{result.get('top_score', 0):.4f}",
                            'Found': "✅" if result.get('expected_found', False) else "❌",
                            'Rank': result.get('expected_rank', 'N/A')
                        })
                    
                    output.append(self.format_table(detail_data, tablefmt="simple"))
        
        return "\n".join(output)
    
    def format_performance_sorted_table(self,
                                      results: List[Dict[str, Any]],
                                      sort_by: str = "score",
                                      descending: bool = True,
                                      group_by: Optional[str] = None) -> str:
        """Format results sorted by performance metrics
        
        Args:
            results: List of result dictionaries
            sort_by: Field to sort by (score, accuracy, etc.)
            descending: Sort in descending order
            group_by: Optional field to group results by
            
        Returns:
            Formatted table string
        """
        if not results:
            return "No results to display"
        
        # Sort results
        sorted_results = sorted(results, 
                              key=lambda x: x.get(sort_by, 0), 
                              reverse=descending)
        
        if group_by and group_by in results[0]:
            # Group results
            grouped = {}
            for result in sorted_results:
                group_key = result.get(group_by, "Unknown")
                if group_key not in grouped:
                    grouped[group_key] = []
                grouped[group_key].append(result)
            
            # Format each group
            output = []
            for group_key, group_results in grouped.items():
                output.append(f"\n{group_by}: {group_key}")
                output.append("-" * 50)
                output.append(self.format_table(group_results))
            
            return "\n".join(output)
        else:
            # Simple sorted table
            return self.format_table(sorted_results)