#!/usr/bin/env python3
"""
Test manual routing in the composing agent
"""

import asyncio
import json
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.app.agents.composing_agents_main import route_and_execute_query


async def test_manual_routing():
    """Test manual routing functionality"""
    
    print("Testing Manual Routing in Composing Agent")
    print("=" * 60)
    
    # Test queries
    test_cases = [
        {
            "name": "Default routing (no preference)",
            "query": "show me videos about medical procedures",
            "preferred_agent": None
        },
        {
            "name": "Manual routing to port 8001",
            "query": "show me videos about medical procedures",
            "preferred_agent": "http://localhost:8001"
        },
        {
            "name": "Manual routing to port 8002",
            "query": "show me videos about medical procedures",
            "preferred_agent": "http://localhost:8002"
        }
    ]
    
    for test in test_cases:
        print(f"\n--- {test['name']} ---")
        print(f"Query: {test['query']}")
        print(f"Preferred Agent: {test['preferred_agent']}")
        
        try:
            # Execute with optional manual routing
            result = await route_and_execute_query(
                query=test['query'],
                top_k=5,
                preferred_agent=test['preferred_agent']
            )
            
            print(f"Execution Type: {result.get('execution_type')}")
            print(f"Agents Called: {result.get('agents_called')}")
            
            if "search_results" in result:
                search_results = result['search_results']
                if search_results.get('success'):
                    print(f"Results Found: {search_results.get('result_count', 0)}")
                else:
                    print(f"Search Error: {search_results.get('error')}")
            
            if "video_search_results" in result:
                video_results = result['video_search_results']
                if video_results.get('success'):
                    print(f"Video Results Found: {video_results.get('result_count', 0)}")
                else:
                    print(f"Video Search Error: {video_results.get('error')}")
            
        except Exception as e:
            print(f"Error: {str(e)}")


def main():
    """Run the manual routing test"""
    asyncio.run(test_manual_routing())


if __name__ == "__main__":
    main()