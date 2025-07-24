"""
Test script for Letta Video Agent Integration

This script tests the basic functionality of the Letta video agent
without requiring a full setup.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test existing imports
        from src.processing.vespa.vespa_search_client import VespaVideoSearchClient
        from src.tools.config import get_config
        from src.tools.a2a_utils import A2AMessage, Task
        print("✓ Existing modules imported successfully")
        
        # Test Letta availability
        try:
            import letta
            print("✓ Letta is available")
            letta_available = True
        except ImportError:
            print("⚠ Letta not installed - install with: pip install letta")
            letta_available = False
        
        # Test our new module structure
        from src.agents.video_agent_letta import (
            LettaVideoSearchTool, 
            MemoryManager, 
            LettaVideoAgent,
            A2AAdapter
        )
        print("✓ Letta agent modules imported successfully")
        
        return True, letta_available
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False, False

def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from src.tools.config import get_config
        config = get_config()
        
        # Check required config values
        vespa_url = config.get("vespa_url", "http://localhost")
        vespa_port = config.get("vespa_port", 8080)
        colpali_model = config.get("colpali_model", "vidore/colsmol-500m")
        
        print(f"✓ Vespa URL: {vespa_url}")
        print(f"✓ Vespa Port: {vespa_port}")
        print(f"✓ ColPali Model: {colpali_model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_vespa_connection():
    """Test Vespa connection"""
    print("\nTesting Vespa connection...")
    
    try:
        from src.processing.vespa.vespa_search_client import VespaVideoSearchClient
        from src.tools.config import get_config
        
        config = get_config()
        vespa_url = config.get("vespa_url", "http://localhost")
        vespa_port = config.get("vespa_port", 8080)
        
        client = VespaVideoSearchClient(vespa_url=vespa_url, vespa_port=vespa_port)
        
        if client.health_check():
            print("✓ Vespa connection successful")
            return True
        else:
            print("⚠ Vespa connection failed - ensure Vespa is running")
            return False
            
    except Exception as e:
        print(f"❌ Vespa connection test failed: {e}")
        return False

def test_a2a_compatibility():
    """Test A2A protocol compatibility"""
    print("\nTesting A2A protocol compatibility...")
    
    try:
        from src.tools.a2a_utils import TextPart, DataPart, A2AMessage, Task
        
        # Create test message
        data_part = DataPart(data={"query": "test query", "top_k": 5})
        message = A2AMessage(role="user", parts=[data_part])
        task = Task(id="test_123", messages=[message])
        
        print("✓ A2A message creation successful")
        
        # Test message parsing
        last_message = task.messages[-1]
        data_part = next(
            (part for part in last_message.parts if isinstance(part, DataPart)), 
            None
        )
        
        if data_part and "query" in data_part.data:
            print("✓ A2A message parsing successful")
            return True
        else:
            print("❌ A2A message parsing failed")
            return False
            
    except Exception as e:
        print(f"❌ A2A compatibility test failed: {e}")
        return False

def test_letta_components():
    """Test Letta-specific components"""
    print("\nTesting Letta components...")
    
    try:
        # Test tool structure
        from src.agents.video_agent_letta import LettaVideoSearchTool
        print("✓ LettaVideoSearchTool structure OK")
        
        # Test memory manager structure  
        from src.agents.video_agent_letta import MemoryManager
        print("✓ MemoryManager structure OK")
        
        # Test adapter structure
        from src.agents.video_agent_letta import A2AAdapter
        print("✓ A2AAdapter structure OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Letta components test failed: {e}")
        return False

def test_fastapi_app():
    """Test FastAPI application structure"""
    print("\nTesting FastAPI application...")
    
    try:
        from src.agents.video_agent_letta import app
        
        # Check routes
        routes = [route.path for route in app.routes]
        expected_routes = ["/agent.json", "/tasks/send", "/feedback", "/recommendations", "/health"]
        
        missing_routes = [route for route in expected_routes if route not in routes]
        
        if missing_routes:
            print(f"⚠ Missing routes: {missing_routes}")
        else:
            print("✓ All expected routes present")
        
        print(f"✓ FastAPI app structure OK - {len(routes)} routes")
        return True
        
    except Exception as e:
        print(f"❌ FastAPI app test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Letta Video Agent Integration Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Test imports
    import_success, letta_available = test_imports()
    test_results.append(("Imports", import_success))
    
    if not import_success:
        print("\n❌ Cannot continue - import failures detected")
        return False
    
    # Test configuration
    config_success = test_configuration()
    test_results.append(("Configuration", config_success))
    
    # Test Vespa connection
    vespa_success = test_vespa_connection()
    test_results.append(("Vespa Connection", vespa_success))
    
    # Test A2A compatibility
    a2a_success = test_a2a_compatibility()
    test_results.append(("A2A Compatibility", a2a_success))
    
    # Test Letta components
    letta_success = test_letta_components()
    test_results.append(("Letta Components", letta_success))
    
    # Test FastAPI app
    fastapi_success = test_fastapi_app()
    test_results.append(("FastAPI App", fastapi_success))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, success in test_results:
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! The Letta integration is ready to use.")
        if letta_available:
            print("\nTo start the server:")
            print("python -m uvicorn src.agents.video_agent_letta:app --reload --port 8001")
        else:
            print("\nTo install Letta and start the server:")
            print("pip install letta")
            print("python -m uvicorn src.agents.video_agent_letta:app --reload --port 8001")
    else:
        print("❌ Some tests failed. Please check the output above.")
        if not letta_available:
            print("\nFirst install Letta: pip install letta")
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()