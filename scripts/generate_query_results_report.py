#!/usr/bin/env python3
"""
Generate comprehensive query results HTML report from test JSON data
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

def load_test_results(json_path: str) -> Dict:
    """Load test results from JSON file"""
    with open(json_path, 'r') as f:
        return json.load(f)

def format_video_tag(video_id: str, expected_videos: List[str], position: int, is_test: bool = False) -> str:
    """Format a video tag with appropriate styling"""
    if is_test or video_id == "test":
        return f'<span class="video-tag test-video">⚠ {video_id}</span>'
    elif video_id in expected_videos:
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

def generate_query_rows(queries: List[Dict]) -> str:
    """Generate table rows for queries"""
    rows = []
    
    for query_data in queries:
        query = query_data['query']
        expected = query_data['expected']
        results = query_data['results'][:5]  # Top 5 results
        mrr = query_data['metrics']['mrr']
        
        # Format expected videos
        expected_html = ' '.join([
            f'<span class="video-tag expected-video">{vid}</span>' 
            for vid in expected
        ])
        
        # Format results
        results_html = []
        for i, vid in enumerate(results):
            pos_html = format_position(i + 1)
            vid_html = format_video_tag(vid, expected, i + 1)
            results_html.append(pos_html + vid_html)
        
        results_html.append(format_metric_badge(mrr))
        
        row = f"""
        <tr>
            <td class="query-text">{query}</td>
            <td>{expected_html}</td>
            <td>{' '.join(results_html)}</td>
        </tr>
        """
        rows.append(row)
    
    return '\n'.join(rows)

def generate_strategy_section(strategy_name: str, strategy_data: Dict) -> str:
    """Generate HTML for a strategy section"""
    metrics = strategy_data['aggregate_metrics']
    queries = strategy_data['queries']
    
    # Get metrics with safe defaults
    mrr = metrics.get('mrr', {}).get('mean', 0)
    recall1 = metrics.get('recall@1', {}).get('mean', 0)
    recall5 = metrics.get('recall@5', {}).get('mean', 0)
    
    rows_html = generate_query_rows(queries)
    
    return f"""
    <div class="summary-stats">
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
            {rows_html}
        </tbody>
    </table>
    """

def generate_complete_report(data: Dict) -> str:
    """Generate the complete HTML report"""
    
    # Start with the template
    script_dir = Path(__file__).parent
    template_path = script_dir.parent / 'outputs' / 'test_results' / 'comprehensive_query_results_report.html'
    with open(template_path, 'r') as f:
        template = f.read()
    
    # For each profile in results
    for profile_data in data['results']:
        profile = profile_data['profile']
        strategies = profile_data.get('strategies', {})
        
        # Map profile names to HTML IDs
        profile_map = {
            'frame_based_colpali': 'colpali',
            'direct_video_global': 'videoprism-base',
            'direct_video_global_large': 'videoprism-large'
        }
        
        # Map strategy names to HTML IDs
        strategy_map = {
            'binary_binary': 'binary',
            'hybrid_binary_bm25_no_description': 'hybrid-no-desc',
            'hybrid_binary_bm25': 'hybrid-desc',
            'bm25_only': 'text-only',
            'float_binary': 'float-binary',
            'phased': 'phased',
            'float_float': 'float'
        }
        
        profile_id = profile_map.get(profile, profile)
        
        # Replace content for each strategy
        for strat_key, strat_data in strategies.items():
            strat_id = strategy_map.get(strat_key, strat_key)
            section_id = f"{profile_id}-{strat_id}"
            
            # Generate the content for this strategy
            content = generate_strategy_section(strat_data['name'], strat_data)
            
            # Find and replace the placeholder
            placeholder = f'<div id="{section_id}" class="strategy-content'
            if placeholder in template:
                # Find the table tbody content to replace
                start_marker = f'<div id="{section_id}"'
                end_marker = '</div>'
                
                # Find the section
                start_idx = template.find(start_marker)
                if start_idx != -1:
                    # Find the closing div for this section
                    div_count = 1
                    idx = template.find('>', start_idx) + 1
                    section_start = idx
                    
                    while div_count > 0 and idx < len(template):
                        if template[idx:idx+5] == '<div ':
                            div_count += 1
                        elif template[idx:idx+6] == '</div>':
                            div_count -= 1
                            if div_count == 0:
                                section_end = idx
                                break
                        idx += 1
                    
                    # Replace the content
                    if 'section_end' in locals():
                        template = (template[:section_start] + 
                                  content + 
                                  template[section_end:])
    
    return template

def main():
    # Get project root (parent of scripts directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / 'outputs' / 'test_results'
    
    if len(sys.argv) < 2:
        # Find the most recent JSON file
        json_files = list(results_dir.glob('comprehensive_v2_*.json'))
        if not json_files:
            print("No test results JSON files found")
            return
        json_path = max(json_files, key=lambda p: p.stat().st_mtime)
    else:
        json_path = Path(sys.argv[1])
        # If relative path provided, make it relative to current working directory
        if not json_path.is_absolute():
            json_path = Path.cwd() / json_path
    
    print(f"Loading results from: {json_path}")
    data = load_test_results(json_path)
    
    print("Generating HTML report...")
    html_content = generate_complete_report(data)
    
    output_path = results_dir / 'comprehensive_query_results_complete.html'
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Report generated: {output_path}")

if __name__ == "__main__":
    main()