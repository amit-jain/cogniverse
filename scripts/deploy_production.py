#!/usr/bin/env python3
"""
Deploy Production API Script

Deploys just the production API to Modal.
Use this after optimization is complete.
"""

import subprocess
import sys
import time
from pathlib import Path

import requests


def deploy_production_api() -> str:
    """
    Deploy the production API to Modal.
    
    Returns:
        URL of the deployed API
    """
    print("🚀 Deploying Agentic Router Production API...")
    print("=" * 60)
    
    try:
        # Change to parent directory
        parent_dir = Path(__file__).parent.parent
        
        cmd = ["modal", "deploy", "src/inference/inference.py"]
        
        print(f"📁 Working directory: {parent_dir}")
        print(f"🔧 Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=parent_dir,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        print("📤 Deployment output:")
        print(result.stdout)
        
        if result.returncode != 0:
            print("❌ Deployment failed:")
            print(f"STDERR: {result.stderr}")
            raise Exception("Production API deployment failed")
        
        print("✅ Production API deployed successfully")
        
        # Try to extract URL from output (Modal typically shows the URL)
        api_url = "https://agentic-router-production-route.modal.run"
        
        # Test if URL is accessible
        try:
            print(f"🧪 Testing API at: {api_url}")
            
            # Test health endpoint first
            health_url = api_url.replace("-route", "-health-check")
            health_response = requests.get(health_url, timeout=30)
            
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"✅ Health check passed: {health_data.get('status', 'unknown')}")
                
                if health_data.get('artifacts_loaded', False):
                    print("📊 Optimization artifacts loaded successfully")
                else:
                    print("⚠️ Using default artifacts (optimization not found)")
                    
            else:
                print(f"⚠️ Health check returned: {health_response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Health check failed: {e}")
        
        print(f"🌐 API URL: {api_url}")
        return api_url
        
    except subprocess.TimeoutExpired:
        raise Exception("Deployment timed out after 10 minutes")
    except Exception as e:
        raise Exception(f"Failed to deploy production API: {e}")

def test_api(api_url: str):
    """Test the deployed API with sample queries."""
    
    print(f"\n🧪 Testing API at: {api_url}")
    
    test_cases = [
        {
            "query": "Show me how to cook pasta",
            "expected_modality": "video"
        },
        {
            "query": "Create a detailed report on climate change",
            "expected_modality": "text",
            "expected_type": "detailed_report"
        },
        {
            "query": "What's the summary of the research?",
            "expected_type": "summary"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            print(f"\n  Test {i}: '{test_case['query']}'")
            
            start_time = time.time()
            response = requests.post(
                api_url,
                json={
                    "user_query": test_case["query"],
                    "include_reasoning": False
                },
                timeout=30
            )
            latency = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                print(f"    ❌ HTTP {response.status_code}: {response.text}")
                continue
            
            result = response.json()
            
            # Check response format
            if "search_modality" not in result or "generation_type" not in result:
                print("    ❌ Invalid response format")
                continue
                
            modality = result["search_modality"]
            gen_type = result["generation_type"]
            api_latency = result.get("latency_ms", "unknown")
            
            print(f"    ✅ {modality}/{gen_type} (API: {api_latency}ms, E2E: {latency:.1f}ms)")
            
            # Check expectations
            if "expected_modality" in test_case:
                if modality != test_case["expected_modality"]:
                    print(f"    ⚠️ Expected modality: {test_case['expected_modality']}, got: {modality}")
            
            if "expected_type" in test_case:
                if gen_type != test_case["expected_type"]:
                    print(f"    ⚠️ Expected type: {test_case['expected_type']}, got: {gen_type}")
            
        except Exception as e:
            print(f"    ❌ Test failed: {e}")
    
    print("\n✅ API testing complete")

def main():
    """Main deployment function."""
    
    try:
        # Deploy the API
        api_url = deploy_production_api()
        
        # Test the API
        test_api(api_url)
        
        # Print usage instructions
        print("\n" + "=" * 60)
        print("🎉 PRODUCTION API DEPLOYED!")
        print("=" * 60)
        print(f"🌐 API URL: {api_url}")
        print(f"📋 Health: {api_url.replace('-route', '-health-check')}")
        print(f"📊 Info: {api_url.replace('-route', '-get-model-info')}")
        
        print("\n🔗 Usage Examples:")
        print(f"""
# Basic query
curl -X POST {api_url} \\
  -H "Content-Type: application/json" \\
  -d '{{"user_query": "Show me cooking videos"}}'

# With conversation history
curl -X POST {api_url} \\
  -H "Content-Type: application/json" \\
  -d '{{"user_query": "What about pasta?", "conversation_history": "User asked about cooking"}}'

# Health check
curl {api_url.replace('-route', '-health-check')}
        """)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
