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

from src.processing.vespa.vespa_search_client import VespaVideoSearchClient
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
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
        
        # Initialize query encoder (shared across all approaches)
        self._init_query_encoder()
    
    def _init_query_encoder(self):
        """Initialize ColPali model for query encoding"""
        print("Initializing query encoder...")
        
        # Device detection
        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            dtype = torch.float32
        else:
            self.device = "cpu"
            dtype = torch.float32
        
        # Load model
        model_name = "vidore/colsmol-500m"  # Use consistent model for queries
        self.col_model = ColIdefics3.from_pretrained(
            model_name, 
            torch_dtype=dtype, 
            device_map=self.device
        ).eval()
        self.col_processor = ColIdefics3Processor.from_pretrained(model_name)
        
        print(f"Query encoder loaded: {model_name} on {self.device}")
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode query text to embeddings"""
        batch_queries = self.col_processor.process_queries([query]).to(self.device)
        with torch.no_grad():
            query_embeddings = self.col_model(**batch_queries)
        return query_embeddings.cpu().numpy().squeeze(0)
    
    def search_agent(self, agent: Dict[str, Any], query: str, embeddings: np.ndarray, top_k: int = 10) -> Dict[str, Any]:
        """Search using a specific agent"""
        agent_url = f"http://{agent['url']}:{agent['port']}"
        
        # Check agent health
        try:
            health_response = requests.get(f"{agent_url}/agent.json", timeout=5)
            if health_response.status_code != 200:
                return {"error": f"Agent not healthy: {health_response.status_code}"}
        except Exception as e:
            return {"error": f"Agent not reachable: {str(e)}"}
        
        # Prepare search request
        task_data = {
            "id": f"test_{int(time.time())}",
            "messages": [{
                "role": "user",
                "parts": [{
                    "type": "data",
                    "data": {
                        "query": query,
                        "top_k": top_k
                    }
                }]
            }]
        }
        
        # Execute search
        start_time = time.time()
        try:
            response = requests.post(
                f"{agent_url}/tasks/send",
                json=task_data,
                timeout=30
            )
            
            if response.status_code == 202:
                result = response.json()
                search_time = time.time() - start_time
                
                return {
                    "success": True,
                    "results": result.get("results", []),
                    "search_time": search_time,
                    "agent": agent['name']
                }
            else:
                return {
                    "error": f"Search failed: {response.status_code}",
                    "details": response.text
                }
                
        except Exception as e:
            return {
                "error": f"Search exception: {str(e)}"
            }
    
    def compare_searches(self, queries: List[str], top_k: int = 10):
        """Run searches across all agents and compare results"""
        
        for query in queries:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            print(f"{'='*80}")
            
            # Encode query once
            print("Encoding query...")
            embeddings = self.encode_query(query)
            print(f"Query embeddings shape: {embeddings.shape}")
            
            query_results = {}
            
            # Search with each agent
            for agent in self.agents:
                print(f"\n--- Searching with {agent['name']} ---")
                result = self.search_agent(agent, query, embeddings, top_k)
                
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print(f"✅ Success: {len(result['results'])} results in {result['search_time']:.2f}s")
                    
                    # Show top 3 results
                    for i, hit in enumerate(result['results'][:3]):
                        print(f"  {i+1}. {hit['video_id']} - frame {hit['frame_id']} @ {hit['start_time']:.1f}s (score: {hit['relevance']:.3f})")
                
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
        
        # Compare timing
        print("\nSearch Times:")
        for name, res in successful.items():
            print(f"  {name}: {res['search_time']:.2f}s")
        
        # Compare top results overlap
        print("\nTop Results Overlap:")
        agent_names = list(successful.keys())
        for i in range(len(agent_names)):
            for j in range(i+1, len(agent_names)):
                agent1, agent2 = agent_names[i], agent_names[j]
                
                # Get video IDs from top 10 results
                videos1 = set(r['video_id'] for r in successful[agent1]['results'][:10])
                videos2 = set(r['video_id'] for r in successful[agent2]['results'][:10])
                
                overlap = len(videos1 & videos2)
                print(f"  {agent1} vs {agent2}: {overlap}/10 videos in common")
        
        # Compare score distributions
        print("\nScore Distributions:")
        for name, res in successful.items():
            scores = [r['relevance'] for r in res['results']]
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