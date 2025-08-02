#!/usr/bin/env python3
"""
Generate comprehensive HTML report from the new comprehensive test v2 format
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import argparse

def generate_html_report(test_results_file: str = None):
    """Generate comprehensive HTML report from test results"""
    
    # Load the test results
    if not test_results_file:
        results_dir = Path("outputs/test_results")
        json_files = sorted(results_dir.glob("comprehensive_v2_*.json"))
        if not json_files:
            print("No test results found!")
            return
        test_results_file = json_files[-1]
        print(f"Using latest results file: {test_results_file}")
    
    with open(test_results_file, 'r') as f:
        data = json.load(f)
    
    # Generate HTML
    html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Comprehensive Video Search Results - Detailed Report</title>
    <style>
        body {
            font-family: -apple-system, Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1, h2 {
            color: #2c3e50;
        }
        .profile-section {
            margin: 40px 0;
            border: 2px solid #3498db;
            border-radius: 10px;
            padding: 20px;
        }
        .profile-header {
            background: #3498db;
            color: white;
            padding: 15px;
            margin: -20px -20px 20px -20px;
            border-radius: 8px 8px 0 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th {
            background: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        td {
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .correct {
            color: #27ae60;
            font-weight: bold;
        }
        .incorrect {
            color: #e74c3c;
        }
        .rank {
            font-size: 0.9em;
            color: #7f8c8d;
        }
        .metrics-table {
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #3498db;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }
        .query-group {
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .video-tag {
            display: inline-block;
            padding: 4px 8px;
            margin: 2px;
            background: #3498db;
            color: white;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .strategy-results {
            margin: 10px 0;
        }
        .strategy-name {
            font-weight: bold;
            color: #34495e;
        }
        .error-section {
            background: #fee;
            border: 2px solid #fcc;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }
        .metric-box {
            text-align: center;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }
        .metric-label {
            font-size: 0.9em;
            color: #666;
        }
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #3498db;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 Comprehensive Video Search Results Report</h1>
        <p><em>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</em></p>
"""
    
    # Process each profile's results
    for profile_result in data['results']:
        profile = profile_result['profile']
        
        # Map profile names to display names
        profile_names = {
            'frame_based_colpali': 'ColPali (Frame-based)',
            'direct_video_colqwen': 'ColQwen-Omni (Direct Video)',
            'direct_video_global': 'VideoPrism Global Base',
            'direct_video_global_large': 'VideoPrism Global Large'
        }
        profile_display = profile_names.get(profile, profile)
        
        # Check if profile had an error
        if 'error' in profile_result:
            html += f"""
        <div class="error-section">
            <h2>❌ {profile_display}</h2>
            <p>Error: {profile_result['error']}</p>
        </div>
"""
            continue
        
        html += f"""
        <div class="profile-section">
            <div class="profile-header">
                <h2>{profile_display}</h2>
            </div>
"""
        
        # If testing multiple strategies, show strategy comparison
        if profile_result.get('strategies'):
            html += """
            <h3>Strategy Comparison</h3>
            <table>
                <thead>
                    <tr>
                        <th width="30%">Query</th>
                        <th width="15%">Expected Videos</th>
                        <th width="55%">Strategy Results</th>
                    </tr>
                </thead>
                <tbody>
"""
            
            # Get all queries from the first strategy (they should all have the same queries)
            first_strategy = list(profile_result['strategies'].values())[0]
            
            for query_result in first_strategy['queries']:
                query = query_result['query']
                expected = query_result['expected']
                
                html += f"""
                    <tr>
                        <td>{query}</td>
                        <td>
"""
                for vid in expected:
                    html += f'<span class="video-tag">{vid}</span>'
                
                html += """
                        </td>
                        <td>
"""
                
                # Show results for each strategy
                for strategy_key, strategy_data in profile_result['strategies'].items():
                    strategy_name = strategy_data['name']
                    
                    # Find this query in strategy results
                    for sq in strategy_data['queries']:
                        if sq['query'] == query:
                            top_results = sq['results'][:3]  # Top 3
                            correct = any(vid in expected for vid in top_results)
                            
                            html += f'<div class="strategy-results">'
                            html += f'<span class="strategy-name">{strategy_name}:</span> '
                            
                            if correct:
                                # Find which result was correct
                                for i, vid in enumerate(top_results):
                                    if vid in expected:
                                        html += f'<span class="correct">✅ {vid} (rank {i+1})</span>'
                                        break
                            else:
                                html += f'<span class="incorrect">❌ {top_results[0] if top_results else "No results"}</span>'
                            
                            html += '</div>'
                            break
                
                html += """
                        </td>
                    </tr>
"""
            
            html += """
                </tbody>
            </table>
"""
            
            # Show aggregate metrics for each strategy
            html += """
            <h3>Strategy Metrics</h3>
            <div class="metrics-table">
"""
            
            for strategy_key, strategy_data in profile_result['strategies'].items():
                strategy_name = strategy_data['name']
                metrics = strategy_data.get('aggregate_metrics', {})
                
                html += f"""
                <h4>{strategy_name}</h4>
                <div class="metrics-grid">
"""
                
                for metric_name in ['mrr', 'recall@1', 'recall@5', 'recall@10']:
                    if metric_name in metrics:
                        value = metrics[metric_name]['mean']
                        html += f"""
                    <div class="metric-box">
                        <div class="metric-label">{metric_name.upper()}</div>
                        <div class="metric-value">{value:.3f}</div>
                    </div>
"""
                
                html += """
                </div>
"""
            
            html += """
            </div>
"""
        
        else:
            # Single strategy results
            html += """
            <table>
                <thead>
                    <tr>
                        <th width="35%">Query</th>
                        <th width="15%">Expected Videos</th>
                        <th width="25%">Top Results</th>
                        <th width="25%">Metrics</th>
                    </tr>
                </thead>
                <tbody>
"""
            
            for query_result in profile_result['queries']:
                query = query_result['query']
                expected = query_result['expected']
                results = query_result['results']
                metrics = query_result['metrics']
                
                html += f"""
                    <tr>
                        <td>{query}</td>
                        <td>
"""
                for vid in expected:
                    html += f'<span class="video-tag">{vid}</span>'
                
                html += """
                        </td>
                        <td>
"""
                
                # Show top 3 results
                for i, vid in enumerate(results[:3]):
                    if vid in expected:
                        html += f'<span class="correct">✅ {vid}</span><br>'
                    else:
                        html += f'<span class="incorrect">{vid}</span><br>'
                
                html += """
                        </td>
                        <td>
                            <div class="metrics-grid">
"""
                
                # Show key metrics
                for metric in ['mrr', 'recall@5']:
                    if metric in metrics:
                        html += f"""
                                <div class="metric-box">
                                    <div class="metric-label">{metric.upper()}</div>
                                    <div class="metric-value">{metrics[metric]:.3f}</div>
                                </div>
"""
                
                html += """
                            </div>
                        </td>
                    </tr>
"""
            
            html += """
                </tbody>
            </table>
"""
            
            # Show aggregate metrics
            if 'aggregate_metrics' in profile_result:
                html += """
            <h3>Aggregate Metrics</h3>
            <div class="summary-stats">
"""
                
                for metric_name, stats in profile_result['aggregate_metrics'].items():
                    if isinstance(stats, dict) and 'mean' in stats:
                        html += f"""
                <div class="stat-card">
                    <div class="stat-value">{stats['mean']:.3f}</div>
                    <div>{metric_name.upper()}</div>
                    <small>σ = {stats['std']:.3f}</small>
                </div>
"""
                
                html += """
            </div>
"""
        
        html += """
        </div>
"""
    
    # Add overall summary
    html += """
        <div class="metrics-table">
            <h2>📊 Overall Performance Summary</h2>
            <div class="summary-stats">
"""
    
    # Calculate overall stats
    total_queries = 0
    total_correct = 0
    
    for profile_result in data['results']:
        if 'error' in profile_result:
            continue
            
        for query_result in profile_result.get('queries', []):
            total_queries += 1
            if query_result.get('top_result_correct', False):
                total_correct += 1
    
    if total_queries > 0:
        accuracy = total_correct / total_queries
        html += f"""
                <div class="stat-card">
                    <div class="stat-value">{accuracy:.1%}</div>
                    <div>Top-1 Accuracy</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_queries}</div>
                    <div>Total Queries</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(data['results'])}</div>
                    <div>Profiles Tested</div>
                </div>
"""
    
    html += """
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    # Save HTML report
    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"comprehensive_report_v2_{timestamp}.html"
    
    with open(report_path, 'w') as f:
        f.write(html)
    
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML report from comprehensive test results")
    parser.add_argument("file", nargs="?", help="JSON results file (uses latest if not specified)")
    args = parser.parse_args()
    
    report_path = generate_html_report(args.file)
    if report_path:
        print(f"✅ HTML report generated: {report_path}")
        print(f"   Open in browser: file://{report_path.absolute()}")