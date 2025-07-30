#!/usr/bin/env python3
"""
Generate comprehensive HTML report with all results in requested format
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

def generate_html_report(test_results_file: str = None):
    """Generate comprehensive HTML report from test results"""
    
    # Load the latest test results
    if not test_results_file:
        results_dir = Path("outputs/test_results")
        json_files = sorted(results_dir.glob("comprehensive_video_query_*.json"))
        if not json_files:
            print("No test results found!")
            return
        test_results_file = json_files[-1]
    
    with open(test_results_file, 'r') as f:
        data = json.load(f)
    
    # Define all test queries (from our comprehensive test)
    test_queries = [
        ("person shoveling snow in winter", "v_-IMXSEIabMM"),
        ("snow covered driveway", "v_-IMXSEIabMM"),
        ("winter storm snow removal", "v_-IMXSEIabMM"),
        ("shovel clearing snowy ground", "v_-IMXSEIabMM"),
        ("dark textured wall with tiles", "elephant_dream_clip"),
        ("cracked stone wall surface", "elephant_dream_clip"),
        ("worn brown tiles on wall", "elephant_dream_clip"),
        ("aged tile pattern with cracks", "elephant_dream_clip"),
        ("smartphone on wooden table", "for_bigger_blazes"),
        ("tablet displaying yellow flower", "for_bigger_blazes"),
        ("phone screen showing sunflower", "for_bigger_blazes"),
        ("green cup on wooden surface", "for_bigger_blazes")
    ]
    
    # Process results by profile and strategy
    profile_results = {}
    
    for result in data['results']:
        strategy = result['strategy']
        query_key = result['query']
        
        # Extract clean query text
        query_text = query_key.split('_', 2)[-1] if '_' in query_key else query_key
        
        # Find matching test query
        for test_query, expected in test_queries:
            if query_text in test_query or test_query.replace(' ', '') in query_text:
                if expected not in profile_results:
                    profile_results[expected] = {}
                if query_text not in profile_results[expected]:
                    profile_results[expected][query_text] = {}
                
                profile_results[expected][query_text][strategy] = {
                    'top_video': result['top_video'],
                    'score': result['top_score'],
                    'found': result['expected_found'],
                    'rank': result.get('expected_rank')
                }
                break
    
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
    
    # Define profiles to show
    profiles = [
        ("frame_based_colpali", "ColPali (ColSmol)", ["bm25_only", "float_float", "binary_binary", "phased"]),
        ("direct_video_colqwen", "ColQwen-Omni", ["float_float", "binary_binary", "float_binary", "phased"]),
        ("direct_video_global", "VideoPrism Global Base", ["float_float", "binary_binary", "phased"]),
        ("direct_video_global_large", "VideoPrism Global Large", ["float_float", "binary_binary", "phased"])
    ]
    
    for profile_key, profile_name, strategies in profiles:
        html += f"""
        <div class="profile-section">
            <div class="profile-header">
                <h2>{profile_name}</h2>
            </div>
            
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
        
        # For each query
        for query, expected_video in test_queries:
            html += f"""
                    <tr>
                        <td>{query}</td>
                        <td><span class="video-tag">{expected_video}</span></td>
                        <td>
"""
            
            # Show results for each strategy
            results_html = []
            for strategy in strategies:
                # Find result for this query/strategy/profile combination
                found_result = False
                for result in data['results']:
                    if (result['strategy'] == strategy and 
                        result['expected_video'] == expected_video and
                        (query.replace(' ', '') in result['query'] or 
                         result['query'].replace('_', ' ').endswith(query))):
                        
                        top_video = result['top_video']
                        rank = result.get('expected_rank')
                        
                        if result['expected_found']:
                            status = f'<span class="correct">✅ {top_video}</span>'
                            if rank:
                                status += f' <span class="rank">(rank {rank})</span>'
                        else:
                            status = f'<span class="incorrect">❌ {top_video}</span>'
                        
                        results_html.append(f"<strong>{strategy}:</strong> {status}")
                        found_result = True
                        break
                
                if not found_result:
                    results_html.append(f"<strong>{strategy}:</strong> <span class='incorrect'>N/A</span>")
            
            html += "<br>".join(results_html)
            html += """
                        </td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
        </div>
"""
    
    # Add metrics summary
    html += """
        <div class="metrics-table">
            <h2>📊 Evaluation Metrics Summary</h2>
            <p>Coming soon: MRR@5, NDCG@10, Recall@k metrics for each profile</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Save HTML
    output_path = Path("outputs/test_results/comprehensive_report.html")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Report generated: {output_path}")
    return output_path

if __name__ == "__main__":
    import sys
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    report_path = generate_html_report(file_path)
    
    # Open in browser
    import subprocess
    subprocess.run(["open", str(report_path)])