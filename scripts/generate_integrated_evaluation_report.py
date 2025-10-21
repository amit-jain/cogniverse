#!/usr/bin/env python3
"""
Generate integrated HTML report combining quantitative test results and evaluation experiments
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def load_latest_files(results_dir: Path, experiments_dir: Path) -> tuple:
    """Load latest test results and experiment results"""
    # Get latest comprehensive test results
    test_files = sorted(results_dir.glob("comprehensive_v2_*.json"))
    test_data = None
    if test_files:
        try:
            with open(test_files[-1], 'r') as f:
                test_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load test results: {e}")
    
    # Get latest experiment results
    experiment_files = sorted(experiments_dir.glob("experiment_details_*.json"))
    experiment_data = None
    if experiment_files:
        try:
            with open(experiment_files[-1], 'r') as f:
                experiment_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load experiment results: {e}")
    
    # Get latest experiment summary CSV
    summary_files = sorted(experiments_dir.glob("experiment_summary_*.csv"))
    summary_df = None
    if summary_files:
        try:
            summary_df = pd.read_csv(summary_files[-1])
        except Exception as e:
            print(f"Warning: Could not load experiment summary: {e}")
    
    return test_data, experiment_data, summary_df


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


def generate_quantitative_tab(test_data: Dict) -> str:
    """Generate HTML for quantitative test results tab"""
    if not test_data:
        return "<p>No quantitative test results available</p>"
    
    html = []
    results = test_data["results"]
    
    for profile, strategies in results.items():
        profile_html = []
        for strategy, queries in strategies.items():
            # Calculate metrics
            all_mrr = []
            total_queries = 0
            
            for query_text, query_data in queries.items():
                if "metrics" in query_data:
                    all_mrr.append(query_data["metrics"].get("mean_reciprocal_rank", 0))
                    total_queries += 1
            
            avg_mrr = sum(all_mrr) / len(all_mrr) if all_mrr else 0
            
            # Strategy header
            strategy_html = f"""
            <div class="strategy-section">
                <h4>{strategy} 
                    <span class="metric-badge {'metric-good' if avg_mrr >= 0.7 else 'metric-medium' if avg_mrr >= 0.3 else 'metric-poor'}">
                        Avg MRR: {avg_mrr:.3f}
                    </span>
                    <span class="query-count">{total_queries} queries</span>
                </h4>
                <div class="query-results">
            """
            
            # Add each query
            for query_text, query_data in queries.items():
                expected = query_data.get("expected_videos", [])
                retrieved = query_data.get("retrieved_videos", [])
                mrr = query_data.get("metrics", {}).get("mean_reciprocal_rank", 0)
                
                query_html = f"""
                <div class="query-item">
                    <div class="query-header">
                        <span class="query-text">{query_text}</span>
                        {format_metric_badge(mrr)}
                    </div>
                    <div class="results-grid">
                """
                
                # Add retrieved videos
                for i, video_id in enumerate(retrieved[:5]):
                    query_html += f"""
                        <div class="result-item">
                            {format_position(i + 1)}
                            {format_video_tag(video_id, expected)}
                        </div>
                    """
                
                query_html += """
                    </div>
                </div>
                """
                strategy_html += query_html
            
            strategy_html += """
                </div>
            </div>
            """
            profile_html.append(strategy_html)
        
        html.append(f"""
        <div class="profile-section">
            <h3>{profile}</h3>
            {''.join(profile_html)}
        </div>
        """)
    
    return ''.join(html)


def generate_evaluation_tab(experiment_data: Dict, summary_df: pd.DataFrame) -> str:
    """Generate HTML for evaluation experiments tab"""
    if not experiment_data:
        return "<p>No evaluation experiment results available</p>"
    
    html = []
    
    # Summary statistics
    html.append(f"""
    <div class="evaluation-summary">
        <h3>Evaluation Summary</h3>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{experiment_data['summary']['total']}</div>
                <div class="stat-label">Total Experiments</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{experiment_data['summary']['successful']}</div>
                <div class="stat-label">Successful</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{experiment_data['summary']['failed']}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{experiment_data['summary']['successful']/experiment_data['summary']['total']*100:.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>
    </div>
    """)
    
    # Profile comparison
    if summary_df is not None:
        html.append("""
        <div class="profile-comparison">
            <h3>Profile Performance</h3>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Profile</th>
                        <th>Strategy</th>
                        <th>Status</th>
                        <th>Experiment</th>
                    </tr>
                </thead>
                <tbody>
        """)
        
        for _, row in summary_df.iterrows():
            status_icon = "✅" if "Success" in str(row.get('Status', '')) else "❌"
            html.append(f"""
                <tr>
                    <td>{row.get('Profile', 'N/A')}</td>
                    <td>{row.get('Strategy', 'N/A')}</td>
                    <td>{status_icon} {row.get('Status', 'N/A')}</td>
                    <td class="experiment-name">{row.get('Experiment Name', 'N/A')}</td>
                </tr>
            """)
        
        html.append("""
                </tbody>
            </table>
        </div>
        """)
    
    # Detailed experiment results
    html.append("""
    <div class="experiment-details">
        <h3>Experiment Details</h3>
    """)
    
    for exp in experiment_data.get('experiments', []):
        if exp['status'] == 'success':
            exp_html = f"""
            <div class="experiment-card">
                <h4>{exp.get('description', exp['experiment_name'])}</h4>
                <div class="experiment-meta">
                    <span class="profile-tag">{exp.get('profile', 'N/A')}</span>
                    <span class="strategy-tag">{exp.get('strategy', 'N/A')}</span>
                </div>
                <div class="experiment-url">
                    Phoenix URL: <a href="{experiment_data['dataset_url']}" target="_blank">View in Phoenix</a>
                </div>
            </div>
            """
            html.append(exp_html)
    
    html.append("</div>")
    
    return ''.join(html)


def generate_integrated_report(
    test_results_file: Optional[str] = None,
    experiment_results_dir: Optional[str] = None,
    output_file: Optional[str] = None
):
    """Generate integrated HTML report with tabs"""
    
    # Set default paths
    results_dir = Path("outputs/test_results")
    experiments_dir = Path(experiment_results_dir) if experiment_results_dir else Path("outputs/experiment_results")
    
    # Load data
    test_data, experiment_data, summary_df = load_latest_files(results_dir, experiments_dir)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Cogniverse Integrated Evaluation Report</title>
    <meta charset="utf-8">
    <style>
        {generate_css()}
    </style>
    <script>
        {generate_javascript()}
    </script>
</head>
<body>
    <div class="container">
        <header>
            <h1>Cogniverse Integrated Evaluation Report</h1>
            <div class="timestamp">Generated: {timestamp}</div>
        </header>
        
        <div class="tabs">
            <button class="tab-button active" onclick="showTab('quantitative')">Quantitative Results</button>
            <button class="tab-button" onclick="showTab('evaluation')">Evaluation Experiments</button>
            <button class="tab-button" onclick="showTab('comparison')">Comparison</button>
        </div>
        
        <div id="quantitative" class="tab-content active">
            <h2>Quantitative Test Results</h2>
            {generate_quantitative_tab(test_data)}
        </div>
        
        <div id="evaluation" class="tab-content">
            <h2>Evaluation Experiments</h2>
            {generate_evaluation_tab(experiment_data, summary_df)}
        </div>
        
        <div id="comparison" class="tab-content">
            <h2>Side-by-Side Comparison</h2>
            {generate_comparison_tab(test_data, experiment_data)}
        </div>
    </div>
</body>
</html>
    """
    
    # Save report
    if not output_file:
        output_dir = Path("outputs/reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"integrated_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    with open(output_file, 'w') as f:
        f.write(html)
    
    print(f"Report saved to: {output_file}")
    return output_file


def generate_comparison_tab(test_data: Dict, experiment_data: Dict) -> str:
    """Generate comparison between quantitative and evaluation results"""
    html = []
    
    if not test_data or not experiment_data:
        return "<p>Need both quantitative and evaluation results for comparison</p>"
    
    html.append("""
    <div class="comparison-grid">
        <div class="comparison-section">
            <h3>Performance Overview</h3>
            <p>This section compares the quantitative test results with the evaluation experiments.</p>
        </div>
    </div>
    """)
    
    # Add more detailed comparison logic here
    
    return ''.join(html)


def generate_css() -> str:
    """Generate CSS styles"""
    return """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        .timestamp {
            color: #7f8c8d;
            font-size: 14px;
        }
        
        /* Tabs */
        .tabs {
            background: white;
            padding: 0;
            border-radius: 10px 10px 0 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            overflow: hidden;
        }
        .tab-button {
            flex: 1;
            padding: 15px 20px;
            border: none;
            background: #ecf0f1;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s;
            border-right: 1px solid #bdc3c7;
        }
        .tab-button:last-child {
            border-right: none;
        }
        .tab-button:hover {
            background: #d5dbdb;
        }
        .tab-button.active {
            background: white;
            color: #2980b9;
            font-weight: 600;
        }
        .tab-content {
            display: none;
            background: white;
            padding: 30px;
            border-radius: 0 0 10px 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-height: 500px;
        }
        .tab-content.active {
            display: block;
        }
        
        /* Quantitative Results Styles */
        .profile-section {
            margin-bottom: 30px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            background: #fafafa;
        }
        .strategy-section {
            margin-bottom: 20px;
            background: white;
            padding: 15px;
            border-radius: 5px;
        }
        .query-item {
            margin-bottom: 15px;
            padding: 10px;
            border-left: 3px solid #3498db;
            background: #f8f9fa;
        }
        .query-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .query-text {
            font-weight: 500;
            color: #2c3e50;
        }
        .results-grid {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .result-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        /* Badges and Tags */
        .metric-badge {
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        .metric-good { background: #27ae60; color: white; }
        .metric-medium { background: #f39c12; color: white; }
        .metric-poor { background: #e74c3c; color: white; }
        
        .video-tag {
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 11px;
            font-family: monospace;
        }
        .correct-video { background: #2ecc71; color: white; }
        .incorrect-video { background: #95a5a6; color: white; }
        
        .position {
            display: inline-block;
            width: 20px;
            height: 20px;
            line-height: 20px;
            text-align: center;
            border-radius: 50%;
            background: #ecf0f1;
            font-size: 11px;
            font-weight: 600;
        }
        .position.first {
            background: #3498db;
            color: white;
        }
        
        /* Evaluation Styles */
        .evaluation-summary {
            margin-bottom: 30px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #e9ecef;
        }
        .stat-value {
            font-size: 36px;
            font-weight: 700;
            color: #2980b9;
        }
        .stat-label {
            font-size: 14px;
            color: #7f8c8d;
            margin-top: 5px;
        }
        
        /* Tables */
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        .comparison-table th,
        .comparison-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }
        .comparison-table th {
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }
        .comparison-table tr:hover {
            background: #f8f9fa;
        }
        
        /* Experiment Cards */
        .experiment-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            border: 1px solid #e9ecef;
        }
        .experiment-meta {
            margin: 10px 0;
        }
        .profile-tag, .strategy-tag {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            margin-right: 10px;
            background: #e9ecef;
            color: #495057;
        }
        .experiment-url {
            font-size: 14px;
            margin-top: 10px;
        }
        .experiment-url a {
            color: #3498db;
            text-decoration: none;
        }
        .experiment-url a:hover {
            text-decoration: underline;
        }
    """


def generate_javascript() -> str:
    """Generate JavaScript for tab functionality"""
    return """
        function showTab(tabName) {
            // Hide all tabs
            const tabs = document.querySelectorAll('.tab-content');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Remove active class from all buttons
            const buttons = document.querySelectorAll('.tab-button');
            buttons.forEach(button => button.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked button
            event.target.classList.add('active');
        }
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate integrated evaluation report")
    parser.add_argument("--test-results", help="Path to quantitative test results JSON")
    parser.add_argument("--experiments-dir", help="Path to experiments results directory")
    parser.add_argument("--output", help="Output HTML file path")
    
    args = parser.parse_args()
    
    generate_integrated_report(
        test_results_file=args.test_results,
        experiment_results_dir=args.experiments_dir,
        output_file=args.output
    )
