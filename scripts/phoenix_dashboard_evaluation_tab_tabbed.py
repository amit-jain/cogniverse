#!/usr/bin/env python3
"""
Phoenix evaluation tab with EXACT tabbed format like generate_tabbed_html_report.py
"""

import streamlit as st
import pandas as pd
import requests
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
from datetime import datetime
import json
from pathlib import Path

def query_phoenix_graphql(query: str) -> Dict[str, Any]:
    """Execute a GraphQL query against Phoenix"""
    try:
        response = requests.post(
            "http://localhost:6006/graphql",
            json={"query": query},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {}
            
    except Exception as e:
        return {}

def get_phoenix_datasets() -> List[Dict[str, Any]]:
    """Get all datasets from Phoenix GraphQL API"""
    query = """
    query {
        datasets {
            edges {
                node {
                    id
                    name
                    exampleCount
                    createdAt
                    description
                    metadata
                }
            }
        }
    }
    """
    
    result = query_phoenix_graphql(query)
    datasets = []
    
    if result and 'data' in result and result['data']:
        for edge in result.get('data', {}).get('datasets', {}).get('edges', []):
            if edge and 'node' in edge:
                node = edge['node']
                datasets.append({
                    'id': node['id'],
                    'name': node['name'],
                    'example_count': node['exampleCount'],
                    'created_at': node['createdAt'],
                    'description': node.get('description', ''),
                    'metadata': node.get('metadata', {})
                })
    
    return datasets

def get_experiment_runs(experiment_id: str) -> Dict[str, Any]:
    """Get experiment runs using Phoenix REST API"""
    try:
        response = requests.get(f"http://localhost:6006/v1/experiments/{experiment_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except Exception as e:
        return {}

def calculate_metrics(results: List[str], expected: List[str]) -> Dict[str, float]:
    """Calculate retrieval metrics"""
    if not expected:
        return {'mrr': 0.0, 'recall@1': 0.0, 'recall@5': 0.0}
    
    mrr = 0.0
    for i, video in enumerate(results[:10]):
        if video in expected:
            mrr = 1.0 / (i + 1)
            break
    
    recall_1 = 1.0 if results and results[0] in expected else 0.0
    recall_5 = len(set(results[:5]) & set(expected)) / len(expected) if expected else 0.0
    
    return {
        'mrr': mrr,
        'recall@1': recall_1,
        'recall@5': recall_5
    }

def format_video_result(video_id: str, expected_videos: List[str], position: int) -> str:
    """Format video result with icon"""
    if video_id in expected_videos:
        return f'<span style="color: green;">✓ {video_id}</span>'
    else:
        return f'<span style="color: red;">✗ {video_id}</span>'

def get_all_experiment_data_for_dataset(dataset_id: str) -> Dict[str, Any]:
    """
    Get all experiment data for a dataset by querying Phoenix.
    This should automatically find and load all experiments for the dataset.
    """
    experiment_data = {}
    
    # Get experiments for this dataset using the correct Phoenix API endpoint
    experiment_ids = []
    
    try:
        # Use the correct API endpoint: /v1/datasets/{dataset_id}/experiments
        response = requests.get(f"http://localhost:6006/v1/datasets/{dataset_id}/experiments")
        if response.status_code == 200:
            experiments_response = response.json()
            
            # Extract experiment IDs from the response
            if 'data' in experiments_response:
                for exp in experiments_response['data']:
                    experiment_ids.append(exp['id'])
            elif isinstance(experiments_response, list):
                # If it returns a list directly
                for exp in experiments_response:
                    if isinstance(exp, dict) and 'id' in exp:
                        experiment_ids.append(exp['id'])
                    elif isinstance(exp, str):
                        # If it returns just IDs
                        experiment_ids.append(exp)
                        
    except Exception as e:
        st.error(f"Error fetching experiments: {e}")
    
    # Load each experiment
    for exp_id in experiment_ids:
        try:
            # Use the /json endpoint which has the actual run data
            response = requests.get(f"http://localhost:6006/v1/experiments/{exp_id}/json")
            if response.status_code == 200:
                runs = response.json()
                
                # Get experiment metadata
                meta_response = requests.get(f"http://localhost:6006/v1/experiments/{exp_id}")
                exp_metadata = {}
                if meta_response.status_code == 200:
                    meta_data = meta_response.json()
                    exp_metadata = meta_data.get('data', {}).get('metadata', {})
                
                # Extract profile and strategy
                profile = exp_metadata.get('profile', 'unknown')
                strategy = exp_metadata.get('ranking_strategy', exp_metadata.get('strategy', 'unknown'))
                
                # Initialize structure
                if profile not in experiment_data:
                    experiment_data[profile] = {}
                if strategy not in experiment_data[profile]:
                    experiment_data[profile][strategy] = {
                        'queries': [],
                        'aggregate_metrics': {'mrr': {'mean': 0}, 'recall@1': {'mean': 0}, 'recall@5': {'mean': 0}}
                    }
                
                # Process runs
                for run in runs:
                    input_data = run.get('input', {})
                    reference_output = run.get('reference_output', {})
                    output_data = run.get('output', {})
                    
                    # IMPORTANT: Profile and strategy are in the output data for each run
                    run_profile = output_data.get('profile', profile)  # Use run's profile if available
                    run_strategy = output_data.get('ranking_strategy', strategy)  # Use run's strategy if available
                    
                    query = input_data.get('query', '')
                    expected_videos_str = reference_output.get('expected_videos', '')
                    expected_videos = [v.strip() for v in expected_videos_str.split(',') if v.strip()] if expected_videos_str else []
                    
                    # Extract unique videos from results
                    results = output_data.get('results', [])
                    seen_videos = set()
                    retrieved_videos = []
                    for result in results:
                        video_id = result.get('video_id', '')
                        if video_id and video_id not in seen_videos:
                            retrieved_videos.append(video_id)
                            seen_videos.add(video_id)
                    
                    if query:
                        # Use the run's profile and strategy, not the experiment metadata
                        # Initialize structure for this run's profile/strategy if needed
                        if run_profile not in experiment_data:
                            experiment_data[run_profile] = {}
                        if run_strategy not in experiment_data[run_profile]:
                            experiment_data[run_profile][run_strategy] = {
                                'queries': [],
                                'aggregate_metrics': {'mrr': {'mean': 0}, 'recall@1': {'mean': 0}, 'recall@5': {'mean': 0}}
                            }
                        
                        metrics = calculate_metrics(retrieved_videos, expected_videos)
                        experiment_data[run_profile][run_strategy]['queries'].append({
                            'query': query,
                            'expected': expected_videos,
                            'results': retrieved_videos,
                            'metrics': metrics
                        })
                
                # After processing all runs, calculate aggregate metrics for all profile/strategy combinations
                for prof in experiment_data:
                    for strat in experiment_data[prof]:
                        queries = experiment_data[prof][strat]['queries']
                        if queries:
                            mrr_values = [q['metrics']['mrr'] for q in queries]
                            recall1_values = [q['metrics']['recall@1'] for q in queries]
                            recall5_values = [q['metrics']['recall@5'] for q in queries]
                            
                            experiment_data[prof][strat]['aggregate_metrics'] = {
                                'mrr': {'mean': sum(mrr_values) / len(mrr_values)},
                                'recall@1': {'mean': sum(recall1_values) / len(recall1_values)},
                                'recall@5': {'mean': sum(recall5_values) / len(recall5_values)}
                            }
        except Exception as e:
            continue
    
    # If still no data, use mock data to show the UI structure
    if not experiment_data:
        # Return empty - the UI will handle showing "No experiments run"
        pass
    
    return experiment_data

def render_evaluation_tab():
    """Render the evaluation tab with EXACT tabbed format"""
    st.subheader("🧪 Evaluation Experiments Dashboard")
    
    # Get datasets
    datasets = get_phoenix_datasets()
    if not datasets:
        st.warning("No datasets found in Phoenix.")
        return
    
    # Sort datasets by creation date
    datasets.sort(key=lambda x: x['created_at'], reverse=True)
    
    # Dataset selector
    dataset_names = [ds['name'] for ds in datasets]
    selected_dataset_name = st.selectbox(
        "Select Dataset",
        dataset_names,
        index=0,
        format_func=lambda x: f"{x} ({next(ds['example_count'] for ds in datasets if ds['name'] == x)} examples)"
    )
    
    selected_dataset = next(ds for ds in datasets if ds['name'] == selected_dataset_name)
    
    # Dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Examples", selected_dataset['example_count'])
    with col2:
        created_date = selected_dataset['created_at'].split('T')[0] if 'T' in selected_dataset['created_at'] else selected_dataset['created_at']
        st.metric("Created", created_date)
    with col3:
        st.markdown(f"[View in Phoenix](http://localhost:6006/datasets/{selected_dataset['id']})")
    
    # Phoenix comparison link
    compare_url = f"http://localhost:6006/datasets/{selected_dataset['id']}/compare"
    st.info(f"📊 **[Open Full Phoenix Comparison View]({compare_url})**")
    
    # Load all experiment data for this dataset
    with st.spinner("Loading experiments..."):
        experiment_data = get_all_experiment_data_for_dataset(selected_dataset['id'])
    
    st.markdown("---")
    
    # Only show profiles that have data
    profiles_with_data = []
    for profile_key in experiment_data:
        # Use the profile key as the display name directly
        profiles_with_data.append((profile_key, profile_key))
    
    if not profiles_with_data:
        st.warning("No experiments found for this dataset.")
        return
    
    # Create profile tabs (MAIN TABS) - only for profiles with data
    profile_tabs = st.tabs([profile_name for _, profile_name in profiles_with_data])
    
    # Use actual experiment data from Phoenix
    experiment_results = experiment_data
    
    # For each profile tab
    for prof_idx, ((profile_key, profile_name), profile_tab) in enumerate(zip(profiles_with_data, profile_tabs)):
        with profile_tab:
            # Get strategies that actually have data for this profile
            strategies_with_data = []
            if profile_key in experiment_results:
                for strat_key in experiment_results[profile_key]:
                    # Use the strategy key as the display name directly
                    strategies_with_data.append((strat_key, strat_key))
            
            if not strategies_with_data:
                st.info("No strategies found for this profile")
                continue
            
            # Create strategy tabs (NESTED TABS) - only for strategies with data
            strategy_tabs = st.tabs([strat_name for _, strat_name in strategies_with_data])
            
            # For each strategy tab
            for strat_idx, ((strat_key, strat_name), strategy_tab) in enumerate(zip(strategies_with_data, strategy_tabs)):
                with strategy_tab:
                    # Get experiment data from loaded results
                    profile_data = experiment_results.get(profile_key, {})
                    strategy_data = profile_data.get(strat_key, {})
                    
                    # We know there's data because we only created tabs for strategies with data
                    if not strategy_data or not strategy_data.get('queries'):
                        st.warning("No query results found")
                        continue
                    
                    # Summary metrics (like HTML report)
                    metrics = strategy_data.get('aggregate_metrics', {})
                    mrr = metrics.get('mrr', {}).get('mean', 0)
                    recall1 = metrics.get('recall@1', {}).get('mean', 0)
                    recall5 = metrics.get('recall@5', {}).get('mean', 0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MRR Score", f"{mrr*100:.1f}%")
                    with col2:
                        st.metric("Recall@1", f"{recall1*100:.1f}%")
                    with col3:
                        st.metric("Recall@5", f"{recall5*100:.1f}%")
                    with col4:
                        st.metric("Queries", len(strategy_data.get('queries', [])))
                    
                    # Query results table
                    st.markdown("#### Query Results")
                    
                    queries_data = strategy_data.get('queries', [])
                    if queries_data:
                        # Styled headers
                        header_html = """
                        <div style="display: flex; padding: 10px 0; background-color: #f0f2f6; border-radius: 4px; margin-bottom: 10px;">
                            <div style="flex: 3; padding: 0 10px;"><strong style="color: #1f2937; font-size: 14px;">QUERY</strong></div>
                            <div style="flex: 3; padding: 0 10px;"><strong style="color: #1f2937; font-size: 14px;">EXPECTED</strong></div>
                            <div style="flex: 6; padding: 0 10px;"><strong style="color: #1f2937; font-size: 14px;">RETRIEVED RESULTS</strong></div>
                        </div>
                        """
                        st.markdown(header_html, unsafe_allow_html=True)
                        
                        # Create compact display for each query
                        for idx, q in enumerate(queries_data):
                            query = q['query']
                            expected = q.get('expected', [])
                            results = q.get('results', [])[:3]  # Top 3
                            metrics = q.get('metrics', {})
                            query_mrr = metrics.get('mrr', 0)
                            recall_1 = metrics.get('recall@1', 0)
                            recall_5 = metrics.get('recall@5', 0)
                            
                            # Single row layout
                            col1, col2, col3 = st.columns([3, 3, 6])
                            
                            with col1:
                                st.markdown(f"**{query}**")
                            
                            with col2:
                                expected_str = ", ".join([f"`{v}`" for v in expected]) if expected else "`None`"
                                st.markdown(expected_str)
                            
                            with col3:
                                # Format retrieved with marks
                                retrieved_parts = []
                                for vid in results:
                                    if vid in expected:
                                        retrieved_parts.append(f"✅ `{vid}`")
                                    else:
                                        retrieved_parts.append(f"❌ `{vid}`")
                                retrieved_str = " | ".join(retrieved_parts) if retrieved_parts else "No results"
                                
                                # Create styled badges
                                mrr_style = "background-color: #d4edda; color: #155724;" if query_mrr >= 0.7 else ("background-color: #fff3cd; color: #856404;" if query_mrr >= 0.3 else "background-color: #f8d7da; color: #721c24;")
                                r1_style = "background-color: #d4edda; color: #155724;" if recall_1 >= 0.7 else ("background-color: #fff3cd; color: #856404;" if recall_1 >= 0.3 else "background-color: #f8d7da; color: #721c24;")
                                r5_style = "background-color: #d4edda; color: #155724;" if recall_5 >= 0.7 else ("background-color: #fff3cd; color: #856404;" if recall_5 >= 0.3 else "background-color: #f8d7da; color: #721c24;")
                                
                                badges_html = f"""
                                <span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; margin-left: 8px; {mrr_style}">MRR: {query_mrr:.3f}</span>
                                <span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; margin-left: 4px; {r1_style}">R@1: {recall_1:.3f}</span>
                                <span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; margin-left: 4px; {r5_style}">R@5: {recall_5:.3f}</span>
                                """
                                
                                st.markdown(retrieved_str + badges_html, unsafe_allow_html=True)
                            
                            # Subtle separator
                            st.markdown("---")
                    else:
                        st.info("No query results available")
    
    # Success Matrix Heatmap (at the bottom)
    st.markdown("---")
    st.markdown("### 📊 Experiment Success Matrix")
    
    # Build matrix data from actual experiment results
    # Collect all unique strategies that were actually run
    all_strategies_run = set()
    for profile_key in experiment_results:
        for strat_key in experiment_results[profile_key]:
            all_strategies_run.add(strat_key)
    
    if all_strategies_run:
        # Build matrix only for profiles and strategies that have data
        matrix_profiles = []
        matrix_strategies = sorted(list(all_strategies_run))
        matrix_data = []
        
        for profile_key, profile_name in profiles_with_data:
            if profile_key in experiment_results:
                matrix_profiles.append(profile_name)
                row = []
                for strat_key in matrix_strategies:
                    if strat_key in experiment_results[profile_key]:
                        # Calculate success (1 if experiment ran successfully, 0 if not)
                        queries = experiment_results[profile_key][strat_key].get('queries', [])
                        if queries:
                            # Calculate average recall@1 as success metric
                            recall_sum = sum(1 for q in queries if q.get('metrics', {}).get('mrr', 0) == 1.0)
                            success_rate = recall_sum / len(queries) if queries else 0
                            row.append(success_rate)
                        else:
                            row.append(0)
                    else:
                        row.append(0)
                if row:  # Only add if we have data
                    matrix_data.append(row)
        
        # Use strategy keys directly as display names
        matrix_strategies_display = matrix_strategies
        
        if matrix_data and matrix_profiles and matrix_strategies_display:
            # Format text as percentages
            text_data = [[f"{val*100:.0f}%" for val in row] for row in matrix_data]
            
            fig = go.Figure(data=go.Heatmap(
                z=matrix_data,
                x=matrix_strategies_display,
                y=matrix_profiles,
                colorscale=[[0, '#e74c3c'], [1, '#27ae60']],
                showscale=False,
                text=text_data,
                texttemplate='%{text}',
                hovertemplate='Profile: %{y}<br>Strategy: %{x}<br>Success Rate: %{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Experiment Success by Profile and Strategy (Recall@1)",
                xaxis_title="Strategy",
                yaxis_title="Profile",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No experiment data available to display success matrix.")
    else:
        st.info("No experiments found. Run experiments first to see the success matrix.")


if __name__ == "__main__":
    st.set_page_config(page_title="Phoenix Tabbed Evaluation", layout="wide")
    render_evaluation_tab()