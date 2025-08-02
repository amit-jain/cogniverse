#!/usr/bin/env python3
"""
Test script for comparing different video embedding approaches
"""

import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import requests
from datetime import datetime

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.search.search_service import SearchService
from src.tools.config import get_config
import torch


class VideoSearchComparison:
    """Compare different video search approaches"""
    
    def __init__(self, agent_configs: List[Dict[str, Any]]):
        """
        Initialize with multiple agent configurations
        
        Args:
            agent_configs: List of dicts with 'name', 'url', 'port', 'profile' keys
        """
        self.agents = agent_configs
        self.results = {}
        self.config = get_config()
        
        # Initialize search services for each profile
        self.search_services = {}
        for agent in agent_configs:
            profile = agent['profile']
            # Update config with agent-specific settings
            self.config['vespa_url'] = agent['url']
            self.config['vespa_port'] = agent['port']
            self.search_services[profile] = SearchService(self.config, profile)
    
    def search_profile(self, profile: str, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Search using a specific profile"""
        if profile not in self.search_services:
            return {"error": f"Profile {profile} not initialized"}
        
        try:
            # Use the search service for this profile
            search_service = self.search_services[profile]
            results = search_service.search(query, top_k=top_k)
            
            # Convert results to dict format
            return {
                "profile": profile,
                "query": query,
                "num_results": len(results),
                "results": [r.to_dict() for r in results]
            }
        except Exception as e:
            return {"error": str(e), "profile": profile}
    
    
    def search_agent(self, agent: Dict[str, Any], query: str, top_k: int = 10) -> Dict[str, Any]:
        """Search using a specific agent"""
        return self.search_profile(agent['profile'], query, top_k)
    
    def compare_searches(self, queries: List[str], top_k: int = 10):
        """Run searches across all agents and compare results"""
        
        for query in queries:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            print(f"{'='*80}")
            
            query_results = {}
            
            # Search with each agent
            for agent in self.agents:
                print(f"\n--- Searching with {agent['name']} ---")
                result = self.search_agent(agent, query, top_k)
                
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print(f"✅ Success: {result['num_results']} results for profile {result['profile']}")
                    
                    # Show top 3 results
                    for i, hit in enumerate(result['results'][:3]):
                        source_id = hit.get('source_id', hit['document_id'].split('_')[0])
                        if 'temporal_info' in hit and hit['temporal_info']:
                            start_time = hit['temporal_info']['start_time']
                            print(f"  {i+1}. {source_id} @ {start_time:.1f}s (score: {hit['score']:.3f})")
                        else:
                            print(f"  {i+1}. {source_id} (score: {hit['score']:.3f})")
                
                query_results[agent['name']] = result
            
            self.results[query] = query_results
            
            # Compare results
            self._compare_results(query, query_results)
    
    def _compare_results(self, query: str, results: Dict[str, Any]):
        """Compare search results across different approaches"""
        print(f"\n--- Comparison for: {query} ---")
        
        # Extract successful results
        successful = {name: res for name, res in results.items() if "error" not in res}
        
        if len(successful) < 2:
            print("Not enough successful results to compare")
            return
        
        # Compare top results overlap
        print("\nTop Results Overlap:")
        agent_names = list(successful.keys())
        for i in range(len(agent_names)):
            for j in range(i+1, len(agent_names)):
                agent1, agent2 = agent_names[i], agent_names[j]
                
                # Get video IDs from top 10 results
                videos1 = set(r.get('source_id', r['document_id'].split('_')[0]) for r in successful[agent1]['results'][:10])
                videos2 = set(r.get('source_id', r['document_id'].split('_')[0]) for r in successful[agent2]['results'][:10])
                
                overlap = len(videos1 & videos2)
                print(f"  {agent1} vs {agent2}: {overlap}/10 videos in common")
        
        # Compare score distributions
        print("\nScore Distributions:")
        for name, res in successful.items():
            scores = [r['score'] for r in res['results']]
            if scores:
                print(f"  {name}: min={min(scores):.3f}, max={max(scores):.3f}, avg={np.mean(scores):.3f}")
    
    def save_results(self, output_file: str):
        """Save comparison results to file"""
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "agents": self.agents,
            "results": self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare video search approaches")
    parser.add_argument("--queries", nargs="+", help="Search queries to test")
    parser.add_argument("--agents", nargs="+", help="Agent configurations (name:url:port:profile)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve")
    parser.add_argument("--output", default="video_search_comparison.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Default queries if none provided
    if not args.queries:
        args.queries = [
            "person walking in the snow",
            "doctor examining patient",
            "emergency room scene",
            "medical equipment close-up",
            "people talking in hospital"
        ]
    
    # Parse agent configurations
    agent_configs = []
    if args.agents:
        for agent_str in args.agents:
            parts = agent_str.split(":")
            if len(parts) == 4:
                agent_configs.append({
                    "name": parts[0],
                    "url": parts[1],
                    "port": int(parts[2]),
                    "profile": parts[3]
                })
    else:
        # Default configurations
        agent_configs = [
            {
                "name": "frame_based_colpali",
                "url": "localhost",
                "port": 8001,
                "profile": "frame_based_colpali"
            },
            {
                "name": "direct_video_colqwen",
                "url": "localhost",
                "port": 8002,
                "profile": "direct_video_colqwen"
            }
        ]
    
    print("Video Search Comparison Test")
    print(f"Agents: {[a['name'] for a in agent_configs]}")
    print(f"Queries: {len(args.queries)}")
    print(f"Top-K: {args.top_k}")
    
    # Run comparison
    comparison = VideoSearchComparison(agent_configs)
    comparison.compare_searches(args.queries, args.top_k)
    comparison.save_results(args.output)


if __name__ == "__main__":
    main()