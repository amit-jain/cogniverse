"""
Example usage of the Letta Video Search Agent

This example demonstrates how to use the Letta-based video search agent
with memory capabilities and learning features.
"""

import asyncio
import json


# Example: Basic usage of the Letta Video Agent
async def basic_usage_example():
    """Basic example of using the Letta video agent"""

    # This would typically be imported from the main module
    from video_agent_letta import LettaVideoAgent

    # Initialize the agent
    agent = LettaVideoAgent(vespa_url="http://localhost", vespa_port=8080)

    # Perform a search
    results = agent.search(query="person walking in the park", top_k=5)

    print("Search Results:")
    if results["success"]:
        for i, result in enumerate(results["results"], 1):
            print(f"{i}. {result['video_title']} - {result['description']}")
            print(f"   Timestamp: {result['timestamp']}")
            print(f"   Relevance: {result['relevance_score']:.3f}")
            print()
    else:
        print(f"Search failed: {results.get('error')}")

    return results


# Example: Learning from user interactions
async def learning_example():
    """Example of how the agent learns from user interactions"""

    from video_agent_letta import LettaVideoAgent

    agent = LettaVideoAgent(vespa_url="http://localhost", vespa_port=8080)

    # Initial search
    query = "machine learning tutorial"
    results = agent.search(query=query, top_k=10)

    # Simulate user interactions
    user_interactions = [
        {
            "video_id": "video_123",
            "action": "click",
            "timestamp": "2024-01-15T10:30:00Z",
            "duration_watched": 120,  # seconds
        },
        {
            "video_id": "video_456",
            "action": "view",
            "timestamp": "2024-01-15T10:32:00Z",
            "duration_watched": 300,
        },
    ]

    # Record interactions for learning
    agent.record_user_interaction(
        query=query, results=results["results"], interactions=user_interactions
    )

    # Update user preferences
    agent.memory_manager.update_user_preferences(
        {
            "preferred_video_length": "medium",  # 5-15 minutes
            "preferred_topics": ["machine learning", "data science", "AI"],
            "language": "english",
            "difficulty_level": "intermediate",
        }
    )

    # Get personalized recommendations
    recommendations = agent.get_search_recommendations()
    print("Personalized Recommendations:")
    for rec in recommendations:
        print(f"- {rec}")


# Example: A2A Protocol Integration
async def a2a_integration_example():
    """Example of using the agent via A2A protocol"""

    import httpx

    # Example A2A task payload
    task_payload = {
        "id": "search_123",
        "messages": [
            {
                "role": "user",
                "parts": [
                    {
                        "type": "data",
                        "data": {
                            "query": "artificial intelligence presentation",
                            "top_k": 8,
                            "start_date": "2024-01-01",
                            "user_context": {
                                "user_id": "user_456",
                                "session_id": "session_789",
                            },
                        },
                    }
                ],
            }
        ],
    }

    # Send task to agent
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8001/tasks/send", json=task_payload
        )

        if response.status_code == 202:
            result = response.json()
            print("A2A Search Results:")
            print(f"Task ID: {result['task_id']}")
            print(f"Status: {result['status']}")
            print(f"Results: {len(result['results'])} videos found")
            print(f"Agent Response: {result['agent_response']}")
        else:
            print(f"A2A request failed: {response.status_code}")


# Example: Feedback Collection
async def feedback_example():
    """Example of collecting and processing user feedback"""

    import httpx

    # Example feedback payload
    feedback_payload = {
        "query": "data visualization examples",
        "results": [
            {"video_id": "viz_123", "relevance_score": 0.85},
            {"video_id": "viz_456", "relevance_score": 0.72},
        ],
        "interactions": [
            {
                "video_id": "viz_123",
                "action": "click",
                "timestamp": "2024-01-15T11:00:00Z",
                "duration_watched": 180,
            }
        ],
        "feedback": {
            "rating": 4,
            "comment": "Good results, but would like more advanced examples",
            "clicked_results": ["viz_123"],
            "relevant_results": ["viz_123"],
            "irrelevant_results": [],
        },
    }

    # Send feedback to agent
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8001/feedback", json=feedback_payload
        )

        if response.status_code == 200:
            print("Feedback recorded successfully")
        else:
            print(f"Feedback recording failed: {response.status_code}")


# Example: Configuration and Setup
def setup_example():
    """Example of setting up the Letta video agent"""

    # Required configuration in config.json or environment variables
    config_example = {
        "vespa_url": "http://localhost",
        "vespa_port": 8080,
        "colpali_model": "vidore/colsmol-500m",
        "search_backend": "vespa",
        "local_llm_model": "deepseek-r1:1.5b",
        "base_url": "http://localhost:11434",
    }

    print("Configuration Example:")
    print(json.dumps(config_example, indent=2))

    # Environment variables alternative
    env_vars = {
        "VESPA_URL": "http://localhost",
        "VESPA_PORT": "8080",
        "COLPALI_MODEL": "vidore/colsmol-500m",
        "LOCAL_LLM_MODEL": "deepseek-r1:1.5b",
    }

    print("\nEnvironment Variables:")
    for key, value in env_vars.items():
        print(f"export {key}={value}")


# Example: Running the server
def server_example():
    """Example of running the Letta video agent server"""

    setup_commands = [
        "# Install Letta",
        "pip install letta",
        "",
        "# Start Vespa (if not already running)",
        "vespa start",
        "",
        "# Start the Letta video agent server",
        "python -m uvicorn src.agents.video_agent_letta:app --reload --port 8001",
        "",
        "# Or use the script directly",
        "python src/agents/video_agent_letta.py",
    ]

    print("Server Setup Commands:")
    for cmd in setup_commands:
        print(cmd)


# Example: Comparison with original agent
def comparison_example():
    """Comparison between original and Letta-based agents"""

    comparison = {
        "Original Agent": {
            "Memory": "No persistent memory",
            "Learning": "No learning capabilities",
            "Personalization": "No personalization",
            "Context": "Single request context only",
            "Adaptation": "Static search strategies",
            "Feedback": "No feedback processing",
        },
        "Letta Agent": {
            "Memory": "Core + Archival memory",
            "Learning": "Learns from interactions",
            "Personalization": "User-specific recommendations",
            "Context": "Multi-turn conversations",
            "Adaptation": "Dynamic strategy selection",
            "Feedback": "Processes user feedback",
        },
    }

    print("Agent Comparison:")
    print(json.dumps(comparison, indent=2))


# Example: Testing the implementation
async def test_implementation():
    """Test the Letta agent implementation"""

    print("=== Testing Letta Video Agent ===")

    try:
        # Test basic functionality
        print("1. Testing basic search...")
        results = await basic_usage_example()
        print(f"✓ Basic search completed: {len(results.get('results', []))} results")

        # Test learning capabilities
        print("\n2. Testing learning capabilities...")
        await learning_example()
        print("✓ Learning example completed")

        # Test A2A integration
        print("\n3. Testing A2A protocol integration...")
        await a2a_integration_example()
        print("✓ A2A integration tested")

        # Test feedback collection
        print("\n4. Testing feedback collection...")
        await feedback_example()
        print("✓ Feedback collection tested")

        print("\n=== All tests completed successfully ===")

    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    print("Letta Video Agent Examples")
    print("=" * 50)

    print("\n1. Setup Example:")
    setup_example()

    print("\n2. Server Example:")
    server_example()

    print("\n3. Comparison Example:")
    comparison_example()

    print("\n4. Running Tests:")
    asyncio.run(test_implementation())
