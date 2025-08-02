#!/usr/bin/env python3
"""
Generate comprehensive HTML report in the EXACT original format
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
    
    # Get the test queries from the data
    test_queries = [(q['query'], q['expected_videos']) for q in data['queries']]
    
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 Comprehensive Video Search Results Report</h1>
        <p><em>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</em></p>
"""
    
    # Define profiles to show with their display names and default strategies
    profiles = [
        ("frame_based_colpali", "ColPali (ColSmol)", ["bm25_only", "float_float", "binary_binary", "phased"]),
        ("direct_video_colqwen", "ColQwen-Omni", ["float_float", "binary_binary", "float_binary", "phased"]),
        ("direct_video_global", "VideoPrism Global Base", ["float_float", "binary_binary", "phased"]),
        ("direct_video_global_large", "VideoPrism Global Large", ["float_float", "binary_binary", "phased"])
    ]
    
    # Process each profile
    for profile_key, profile_name, default_strategies in profiles:
        # Find this profile's results
        profile_data = None
        for result in data['results']:
            if result['profile'] == profile_key:
                profile_data = result
                break
        
        if not profile_data:
            continue
            
        html += f"""
        <div class="profile-section">
            <div class="profile-header">
                <h2>{profile_name}</h2>
            </div>
"""
        
        # Check if this profile had an error
        if 'error' in profile_data:
            html += f"""
            <p style="color: #e74c3c; font-weight: bold;">Error: {profile_data['error']}</p>
        </div>
"""
            continue
        
        html += """
            <table>
                <thead>
                    <tr>
                        <th width="35%">Query</th>
                        <th width="15%">Expected Video</th>
                        <th width="50%">Results (Strategy → Top Video)</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Get strategies used (either from multi-strategy test or default)
        if profile_data.get('strategies'):
            # Multi-strategy results
            strategies_data = profile_data['strategies']
        else:
            # Single strategy results - create fake strategies data
            strategies_data = {'default': {
                'name': 'Default',
                'queries': profile_data['queries']
            }}
        
        # For each query
        for query, expected_videos in test_queries:
            html += f"""
                    <tr>
                        <td>{query}</td>
                        <td>
"""
            # Show all expected videos
            for vid in expected_videos:
                html += f'<span class="video-tag">{vid}</span>'
            
            html += """
                        </td>
                        <td>
"""
            
            # Show results for each strategy
            results_html = []
            
            if profile_data.get('strategies'):
                # Multi-strategy results
                for strategy_key, strategy_data in strategies_data.items():
                    strategy_name = strategy_data['name']
                    
                    # Find this query's results
                    query_result = None
                    for qr in strategy_data['queries']:
                        if qr['query'] == query:
                            query_result = qr
                            break
                    
                    if query_result and query_result['results']:
                        top_video = query_result['results'][0]
                        if top_video in expected_videos:
                            # Find rank
                            rank = query_result['results'].index(top_video) + 1
                            status = f'<span class="correct">✅ {top_video}</span>'
                            if rank > 1:
                                status += f' <span class="rank">(rank {rank})</span>'
                        else:
                            status = f'<span class="incorrect">❌ {top_video}</span>'
                        
                        results_html.append(f"<strong>{strategy_key}:</strong> {status}")
                    else:
                        results_html.append(f"<strong>{strategy_key}:</strong> <span class='incorrect'>No results</span>")
            else:
                # Single strategy results
                query_result = None
                for qr in profile_data['queries']:
                    if qr['query'] == query:
                        query_result = qr
                        break
                
                if query_result and query_result['results']:
                    top_video = query_result['results'][0]
                    if top_video in expected_videos:
                        status = f'<span class="correct">✅ {top_video}</span>'
                    else:
                        status = f'<span class="incorrect">❌ {top_video}</span>'
                    
                    results_html.append(f"<strong>Default:</strong> {status}")
                else:
                    results_html.append(f"<strong>Default:</strong> <span class='incorrect'>No results</span>")
            
            html += "<br>".join(results_html)
            html += """
                        </td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
"""
        
        # Add metrics for this profile
        if 'aggregate_metrics' in profile_data and profile_data['aggregate_metrics']:
            html += """
            <div class="metrics-table">
                <h3>Metrics Summary</h3>
                <div class="summary-stats">
"""
            
            for metric_name, stats in profile_data['aggregate_metrics'].items():
                if isinstance(stats, dict) and 'mean' in stats:
                    html += f"""
                    <div class="stat-card">
                        <div class="stat-value">{stats['mean']:.3f}</div>
                        <div>{metric_name.upper()}</div>
                    </div>
"""
            
            html += """
                </div>
            </div>
"""
        
        html += """
        </div>
"""
    
    # Add overall metrics summary
    html += """
        <div class="metrics-table">
            <h2>📊 Evaluation Metrics Summary</h2>
            <table style="margin: 20px auto;">
                <thead>
                    <tr>
                        <th>Profile</th>
                        <th>MRR</th>
                        <th>Recall@1</th>
                        <th>Recall@5</th>
                        <th>Recall@10</th>
                        <th>NDCG@5</th>
                        <th>NDCG@10</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    for result in data['results']:
        profile = result['profile']
        profile_display = next((p[1] for p in profiles if p[0] == profile), profile)
        
        html += f"""
                    <tr>
                        <td>{profile_display}</td>
"""
        
        if 'error' in result:
            html += """
                        <td colspan="6" style="color: #e74c3c;">Error</td>
"""
        elif 'aggregate_metrics' in result:
            metrics = result['aggregate_metrics']
            for metric in ['mrr', 'recall@1', 'recall@5', 'recall@10', 'ndcg@5', 'ndcg@10']:
                if metric in metrics and isinstance(metrics[metric], dict):
                    html += f"""
                        <td>{metrics[metric]['mean']:.3f}</td>
"""
                else:
                    html += """
                        <td>N/A</td>
"""
        else:
            html += """
                        <td colspan="6">No data</td>
"""
        
        html += """
                    </tr>
"""
    
    html += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
    
    # Save HTML report
    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"comprehensive_report_exact_{timestamp}.html"
    
    with open(report_path, 'w') as f:
        f.write(html)
    
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML report in exact original format")
    parser.add_argument("file", nargs="?", help="JSON results file (uses latest if not specified)")
    args = parser.parse_args()
    
    report_path = generate_html_report(args.file)
    if report_path:
        print(f"✅ HTML report generated: {report_path}")
        print(f"   Open in browser: file://{report_path.absolute()}")