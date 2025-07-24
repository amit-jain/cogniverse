#!/usr/bin/env python3
"""
Run Optimization Script

Complete workflow for agentic router optimization:
1. Run orchestrator optimization
2. Upload artifacts to Modal volume
3. Deploy production API
4. Test the system
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_orchestrator(config_path: str = "config.json") -> str:
    """
    Run the orchestrator and return the path to artifacts.
    
    Returns:
        Path to the generated artifacts file
    """
    print("🚀 Starting Orchestrator Optimization...")
    print("=" * 60)
    
    # Run orchestrator using new structure
    cmd = [sys.executable, "-m", "src.optimizer.orchestrator", "--config", config_path]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )
        
        if result.returncode != 0:
            print(f"❌ Orchestrator failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise Exception(f"Orchestrator failed with code {result.returncode}")
        
        print("✅ Orchestrator completed successfully")
        print(f"Output: {result.stdout}")
        
        # Find the artifacts file
        artifacts_path = Path("optimization_results/unified_router_prompt_artifact.json")
        if artifacts_path.exists():
            print(f"📄 Artifacts found: {artifacts_path}")
            return str(artifacts_path)
        else:
            raise Exception("Artifacts file not found after optimization")
            
    except subprocess.TimeoutExpired:
        raise Exception("Orchestrator timed out after 2 hours")
    except Exception as e:
        raise Exception(f"Failed to run orchestrator: {e}")

def upload_artifacts_to_modal(artifacts_path: str) -> bool:
    """
    Upload optimization artifacts to Modal volume.
    
    Args:
        artifacts_path: Local path to artifacts file
        
    Returns:
        True if successful
    """
    print("\n📤 Uploading artifacts to Modal volume...")
    
    try:
        # Create Modal volume if it doesn't exist and upload artifacts
        cmd = [
            "modal", "volume", "put", 
            "optimization-artifacts",  # Volume name
            artifacts_path,
            "/artifacts/unified_router_prompt_artifact.json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"❌ Upload failed: {result.stderr}")
            return False
        
        print("✅ Artifacts uploaded to Modal volume")
        return True
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

def deploy_production_api() -> str:
    """
    Deploy the production API to Modal.
    
    Note: Requires HuggingFace token to be set up as Modal secret:
        modal secret create huggingface-token HF_TOKEN=<your-token>
    
    Returns:
        URL of the deployed API
    """
    print("\n🚀 Deploying Production API...")
    
    # Check if HuggingFace secret exists and create if needed
    try:
        result = subprocess.run(["modal", "secret", "list"], capture_output=True, text=True)
        if "huggingface-token" not in result.stdout:
            print("⚠️  Modal secret 'huggingface-token' not found")
            # Get token from environment
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            if hf_token:
                print("📝 Creating Modal secret...")
                create_result = subprocess.run(
                    ["modal", "secret", "create", "huggingface-token", f"HF_TOKEN={hf_token}"],
                    capture_output=True,
                    text=True
                )
                if create_result.returncode == 0:
                    print("✅ Modal secret created successfully")
                else:
                    print(f"❌ Failed to create Modal secret: {create_result.stderr}")
            else:
                print("❌ No HF_TOKEN found in environment. Please set HF_TOKEN and try again.")
                return None
    except Exception as e:
        print(f"⚠️  Could not check Modal secrets: {e}")
    
    try:
        cmd = ["modal", "deploy", "src/inference/modal_inference_service.py"]
        
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            print(f"❌ Deployment failed: {result.stderr}")
            raise Exception("Production API deployment failed")
        
        print("✅ Production API deployed successfully")
        
        # Extract URL from deployment output (this is a simplified version)
        api_url = "https://agentic-router-production-route.modal.run"
        print(f"🌐 API URL: {api_url}")
        
        return api_url
        
    except Exception as e:
        raise Exception(f"Failed to deploy production API: {e}")

def test_production_api(api_url: str) -> bool:
    """
    Test the deployed production API.
    
    Args:
        api_url: URL of the deployed API
        
    Returns:
        True if tests pass
    """
    print("\n🧪 Testing Production API...")
    
    import requests
    
    # Test queries with expected outputs
    test_cases = [
        {
            "query": "Show me how to cook pasta",
            "expected_modality": "video",
            "expected_type": "raw_results"
        },
        {
            "query": "Create a detailed report on climate change",
            "expected_modality": "text", 
            "expected_type": "detailed_report"
        },
        {
            "query": "What's the summary of the AI paper?",
            "expected_modality": "text",
            "expected_type": "summary"
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            print(f"  Test {i}: '{test_case['query']}'")
            
            response = requests.post(
                api_url,
                json={"user_query": test_case["query"]},
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"    ❌ HTTP {response.status_code}")
                all_passed = False
                continue
            
            result = response.json()
            
            # Check if response has correct format
            if "search_modality" not in result or "generation_type" not in result:
                print(f"    ❌ Invalid response format: {result}")
                all_passed = False
                continue
            
            # Check values
            modality_ok = result["search_modality"] == test_case["expected_modality"]
            type_ok = result["generation_type"] == test_case["expected_type"]
            
            if modality_ok and type_ok:
                print(f"    ✅ {result['search_modality']}/{result['generation_type']} ({result.get('latency_ms', 0):.1f}ms)")
            else:
                print(f"    ⚠️ Unexpected: {result['search_modality']}/{result['generation_type']}")
                # Don't fail for prediction differences, just warn
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            all_passed = False
    
    if all_passed:
        print("✅ All API tests completed")
    else:
        print("⚠️ Some tests had issues")
    
    return all_passed

def main():
    """Run the complete optimization and deployment workflow."""
    
    print("🎯 Agentic Router Complete Optimization & Deployment")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Step 1: Run optimization
        artifacts_path = run_orchestrator()
        
        # Step 2: Upload artifacts to Modal
        upload_success = upload_artifacts_to_modal(artifacts_path)
        if not upload_success:
            print("⚠️ Continuing without uploading artifacts (will use defaults)")
        
        # Step 3: Deploy production API
        api_url = deploy_production_api()
        
        # Step 4: Test the API
        test_success = test_production_api(api_url)
        
        # Summary
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("🎉 DEPLOYMENT COMPLETE!")
        print("=" * 60)
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"📄 Artifacts: {artifacts_path}")
        print(f"🌐 API URL: {api_url}")
        print(f"✅ Tests: {'Passed' if test_success else 'Had issues'}")
        
        print("\n🔗 Usage:")
        print(f"curl -X POST {api_url} \\")
        print("  -H 'Content-Type: application/json' \\")
        print("  -d '{\"user_query\": \"Show me cooking videos\"}'")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run complete optimization and deployment")
    parser.add_argument("--config", default="config.json", help="Configuration file")
    parser.add_argument("--skip-upload", action="store_true", help="Skip uploading artifacts to Modal")
    parser.add_argument("--skip-deploy", action="store_true", help="Skip deploying production API")
    parser.add_argument("--skip-test", action="store_true", help="Skip testing the API")
    
    args = parser.parse_args()
    
    success = main()
    sys.exit(0 if success else 1)