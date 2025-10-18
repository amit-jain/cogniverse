#!/usr/bin/env python3
"""
Generate HTML report with the EXACT original tabbed format
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List


def format_video_tag(video_id: str, expected_videos: List[str], position: int = 0) -> str:
    """Format a video tag with appropriate styling"""
    if video_id in expected_videos:
        return f'<span class="video-tag correct-video">✓ {video_id}</span>'
    else:
        return f'<span class="video-tag incorrect-video">✗ {video_id}</span>'

def format_position(position: int) -> str:
    """Format position indicator"""
    if position == 1:
        return f'<span class="position first">{position}</span>'
    else:
        return f'<span class="position">{position}</span>'

def format_metric_badge(mrr: float) -> str:
    """Format MRR metric badge"""
    if mrr >= 0.7:
        css_class = "metric-good"
    elif mrr >= 0.3:
        css_class = "metric-medium"
    else:
        css_class = "metric-poor"
    return f'<span class="metric-badge {css_class}">MRR: {mrr:.3f}</span>'

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
    
    # Extract test queries
    test_queries = [(q['query'], q['expected_videos']) for q in data['queries']]
    
    # Define profiles with display names
    profiles = [
        ("frame_based_colpali", "ColPali"),
        ("colqwen_chunks", "ColQwen Chunks"), 
        ("direct_video_global", "VideoPrism Base"),
        ("direct_video_global_large", "VideoPrism Large")
    ]
    
    # Define strategies for each profile
    profile_strategies = {
        "frame_based_colpali": [
            ("binary_binary", "Visual Only"),
            ("hybrid_binary_bm25_no_description", "Hybrid (No Desc)"),
            ("hybrid_binary_bm25", "Hybrid + Desc"),
            ("bm25_only", "Text Only"),
            ("float_float", "Float"),
            ("phased", "Phased")
        ],
        "colqwen_chunks": [
            ("float_float", "Float"),
            ("binary_binary", "Binary"),
            ("hybrid_binary_bm25", "Hybrid"),
            ("phased", "Phased")
        ],
        "direct_video_global": [
            ("float_float", "Float"),
            ("binary_binary", "Binary"),
            ("phased", "Phased")
        ],
        "direct_video_global_large": [
            ("float_float", "Float"),
            ("binary_binary", "Binary"),
            ("phased", "Phased")
        ]
    }
    
    # Generate HTML
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Comprehensive Query Results - All Strategies</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        
        /* Tab Navigation */
        .profile-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #dee2e6;
            padding-bottom: 10px;
        }
        .profile-tab {
            padding: 10px 20px;
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-bottom: none;
            border-radius: 8px 8px 0 0;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 500;
        }
        .profile-tab:hover {
            background: #e9ecef;
        }
        .profile-tab.active {
            background: #007bff;
            color: white;
            border-color: #007bff;
        }
        
        /* Strategy Tabs */
        .strategy-tabs {
            display: flex;
            gap: 5px;
            margin: 20px 0;
            background: #e9ecef;
            padding: 5px;
            border-radius: 8px;
        }
        .strategy-tab {
            padding: 8px 16px;
            background: transparent;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s;
            font-size: 14px;
        }
        .strategy-tab:hover {
            background: #dee2e6;
        }
        .strategy-tab.active {
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Content Areas */
        .profile-content {
            display: none;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .profile-content.active {
            display: block;
        }
        .strategy-content {
            display: none;
        }
        .strategy-content.active {
            display: block;
        }
        
        /* Query Results Table */
        .query-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        .query-table th {
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #dee2e6;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        .query-table td {
            padding: 12px;
            border-bottom: 1px solid #e9ecef;
            vertical-align: top;
        }
        .query-table tr:hover {
            background: #f8f9fa;
        }
        
        /* Video Tags */
        .video-tag {
            display: inline-block;
            padding: 4px 8px;
            margin: 2px;
            border-radius: 4px;
            font-size: 13px;
            font-family: monospace;
        }
        .expected-video {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .returned-video {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
        }
        .correct-video {
            background: #28a745;
            color: white;
            font-weight: bold;
        }
        .incorrect-video {
            background: #dc3545;
            color: white;
        }
        
        /* Position Numbers */
        .position {
            display: inline-block;
            width: 20px;
            height: 20px;
            line-height: 20px;
            text-align: center;
            background: #6c757d;
            color: white;
            border-radius: 50%;
            font-size: 11px;
            margin-right: 5px;
        }
        .position.first {
            background: #007bff;
        }
        
        /* Metrics */
        .metric-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 10px;
        }
        .metric-good {
            background: #d4edda;
            color: #155724;
        }
        .metric-medium {
            background: #fff3cd;
            color: #856404;
        }
        .metric-poor {
            background: #f8d7da;
            color: #721c24;
        }
        
        /* Summary Stats */
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .stat-box {
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        .stat-label {
            font-size: 14px;
            color: #6c757d;
            margin-top: 5px;
        }
        
        /* Error Message */
        .error-message {
            padding: 20px;
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
            border-radius: 4px;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Comprehensive Query Results - All Strategies</h1>
        <p style="text-align: center; color: #6c757d;">Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        
        <!-- Profile Tabs -->
        <div class="profile-tabs">
"""
    
    # Add profile tabs
    for i, (profile_key, profile_name) in enumerate(profiles):
        active_class = "active" if i == 0 else ""
        html += f'            <div class="profile-tab {active_class}" onclick="showProfile(\'{profile_key}\')">{profile_name}</div>\n'
    
    html += """        </div>
        
        <!-- Profile Contents -->
"""
    
    # Generate content for each profile
    for prof_idx, (profile_key, profile_name) in enumerate(profiles):
        active_class = "active" if prof_idx == 0 else ""
        
        # Find this profile's results
        profile_data = None
        for result in data['results']:
            if result['profile'] == profile_key:
                profile_data = result
                break
        
        html += f"""        <div id="{profile_key}" class="profile-content {active_class}">
"""
        
        if profile_data and 'error' in profile_data:
            # Show error message
            html += f"""            <div class="error-message">
                Error: {profile_data['error']}
            </div>
"""
        elif profile_data:
            # Add strategy tabs for this profile
            html += """            <div class="strategy-tabs">
"""
            
            strategies = profile_strategies.get(profile_key, [])
            for strat_idx, (strat_key, strat_name) in enumerate(strategies):
                active_class = "active" if strat_idx == 0 else ""
                html += f'                <div class="strategy-tab {active_class}" onclick="showStrategy(\'{profile_key}\', \'{strat_key}\')">{strat_name}</div>\n'
            
            html += """            </div>
            
"""
            
            # Generate content for each strategy
            for strat_idx, (strat_key, strat_name) in enumerate(strategies):
                active_class = "active" if strat_idx == 0 else ""
                
                html += f"""            <div id="{profile_key}-{strat_key}" class="strategy-content {active_class}">
"""
                
                # Check if this is multi-strategy or single strategy result
                if profile_data.get('strategies'):
                    # Multi-strategy results
                    if strat_key in profile_data['strategies']:
                        strategy_data = profile_data['strategies'][strat_key]
                        queries_data = strategy_data['queries']
                        metrics = strategy_data.get('aggregate_metrics', {})
                    else:
                        queries_data = []
                        metrics = {}
                else:
                    # Single strategy results - only show for first strategy tab
                    if strat_idx == 0:
                        queries_data = profile_data.get('queries', [])
                        metrics = profile_data.get('aggregate_metrics', {})
                    else:
                        queries_data = []
                        metrics = {}
                
                # Add summary stats
                mrr = metrics.get('mrr', {}).get('mean', 0) if metrics else 0
                recall1 = metrics.get('recall@1', {}).get('mean', 0) if metrics else 0
                recall5 = metrics.get('recall@5', {}).get('mean', 0) if metrics else 0
                
                html += f"""                <div class="summary-stats">
                    <div class="stat-box">
                        <div class="stat-value">{mrr*100:.1f}%</div>
                        <div class="stat-label">MRR Score</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{recall1*100:.1f}%</div>
                        <div class="stat-label">Recall@1</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{recall5*100:.1f}%</div>
                        <div class="stat-label">Recall@5</div>
                    </div>
                </div>
                
                <table class="query-table">
                    <thead>
                        <tr>
                            <th width="35%">Query</th>
                            <th width="15%">Expected Videos</th>
                            <th width="50%">Retrieved Videos (Top 5)</th>
                        </tr>
                    </thead>
                    <tbody>
"""
                
                # Add rows for each query
                for query_data in queries_data:
                    query = query_data['query']
                    expected = query_data.get('expected', [])
                    results = query_data.get('results', [])[:5]  # Top 5
                    query_mrr = query_data.get('metrics', {}).get('mrr', 0)
                    
                    # Format expected videos
                    expected_html = ' '.join([
                        f'<span class="video-tag expected-video">{vid}</span>' 
                        for vid in expected
                    ])
                    
                    # Format results
                    results_html_parts = []
                    for i, vid in enumerate(results):
                        pos_html = format_position(i + 1)
                        vid_html = format_video_tag(vid, expected, i + 1)
                        results_html_parts.append(pos_html + vid_html)
                    
                    if results_html_parts:
                        results_html_parts.append(format_metric_badge(query_mrr))
                        results_html = ' '.join(results_html_parts)
                    else:
                        results_html = '<span style="color: #6c757d;">No results</span>'
                    
                    html += f"""                        <tr>
                            <td>{query}</td>
                            <td>{expected_html}</td>
                            <td>{results_html}</td>
                        </tr>
"""
                
                html += """                    </tbody>
                </table>
            </div>
"""
        else:
            html += """            <div class="error-message">
                No data available for this profile
            </div>
"""
        
        html += """        </div>
"""
    
    # Add JavaScript for tab switching
    html += """    </div>
    
    <script>
        function showProfile(profileId) {
            // Hide all profile contents
            const profileContents = document.querySelectorAll('.profile-content');
            profileContents.forEach(content => content.classList.remove('active'));
            
            // Remove active from all profile tabs
            const profileTabs = document.querySelectorAll('.profile-tab');
            profileTabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected profile
            document.getElementById(profileId).classList.add('active');
            
            // Set active tab
            event.target.classList.add('active');
            
            // Show first strategy of this profile
            const firstStrategyTab = document.querySelector(`#${profileId} .strategy-tab`);
            if (firstStrategyTab) {
                firstStrategyTab.click();
            }
        }
        
        function showStrategy(profileId, strategyId) {
            // Hide all strategy contents for this profile
            const strategyContents = document.querySelectorAll(`#${profileId} .strategy-content`);
            strategyContents.forEach(content => content.classList.remove('active'));
            
            // Remove active from all strategy tabs in this profile
            const strategyTabs = document.querySelectorAll(`#${profileId} .strategy-tab`);
            strategyTabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected strategy
            const selectedContent = document.getElementById(`${profileId}-${strategyId}`);
            if (selectedContent) {
                selectedContent.classList.add('active');
            }
            
            // Set active tab
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
"""
    
    # Save HTML report
    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"comprehensive_tabbed_report_{timestamp}.html"
    
    with open(report_path, 'w') as f:
        f.write(html)
    
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HTML report with original tabbed format")
    parser.add_argument("file", nargs="?", help="JSON results file (uses latest if not specified)")
    args = parser.parse_args()
    
    report_path = generate_html_report(args.file)
    if report_path:
        print(f"✅ HTML report generated: {report_path}")
        print(f"   Open in browser: file://{report_path.absolute()}")
