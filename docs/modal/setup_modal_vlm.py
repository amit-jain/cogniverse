#!/usr/bin/env python3
"""
Helper script to set up Modal VLM service for video processing pipeline.
This script automates the deployment and configuration process.
"""

import json
import re
import subprocess
import sys
from pathlib import Path


def check_modal_installation():
    """Check if Modal is installed and authenticated."""
    try:
        result = subprocess.run(["modal", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Modal CLI found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Modal CLI not found")
            return False
    except FileNotFoundError:
        print("❌ Modal CLI not installed")
        return False

def install_modal():
    """Install Modal CLI."""
    print("📦 Installing Modal CLI...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "modal"], check=True)
        print("✅ Modal CLI installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Modal CLI")
        return False

def setup_modal_auth():
    """Set up Modal authentication."""
    print("🔐 Setting up Modal authentication...")
    try:
        subprocess.run(["modal", "setup"], check=True)
        print("✅ Modal authentication completed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Modal authentication failed")
        return False

def deploy_modal_service():
    """Deploy the Modal VLM service."""
    print("🚀 Deploying Modal VLM service...")
    try:
        result = subprocess.run(
            ["modal", "deploy", "modal_vlm_service.py"], 
            capture_output=True, 
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ Modal VLM service deployed successfully")
            
            # Extract endpoint URL from output
            output = result.stdout + result.stderr
            url_pattern = r'https://[^\s]+--generate-description\.modal\.run'
            matches = re.findall(url_pattern, output)
            
            if matches:
                endpoint_url = matches[0]
                print(f"🌐 Endpoint URL: {endpoint_url}")
                return endpoint_url
            else:
                print("⚠️ Could not extract endpoint URL from deployment output")
                print("Check Modal dashboard for the endpoint URL")
                return None
        else:
            print(f"❌ Deployment failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ Deployment timeout (5 minutes)")
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed: {e}")
        return None

def update_config(endpoint_url):
    """Update config.json with the VLM endpoint URL."""
    config_path = Path("config.json")
    
    if not config_path.exists():
        print("❌ config.json not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        config["vlm_endpoint_url"] = endpoint_url
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Updated config.json with endpoint URL")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update config.json: {e}")
        return False

def test_integration():
    """Test the VLM integration."""
    print("🧪 Testing VLM integration...")
    try:
        # Test just the model loading first
        result = subprocess.run([
            sys.executable, "-c", 
            "from src.processing.video_ingestion_pipeline import load_models; "
            "models = load_models('cpu'); "
            "print('✅ Model loading test passed!')"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Integration test passed!")
            return True
        else:
            print("⚠️ Integration test had issues:")
            print(result.stderr[-500:])  # Last 500 chars
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup workflow."""
    print("🎬 Setting up Modal VLM for Video Processing Pipeline")
    print("=" * 60)
    
    # Step 1: Check/install Modal
    if not check_modal_installation():
        if not install_modal():
            print("❌ Setup failed: Could not install Modal CLI")
            return
        if not check_modal_installation():
            print("❌ Setup failed: Modal CLI still not working after installation")
            return
    
    # Step 2: Authentication
    auth_result = input("🔐 Do you need to set up Modal authentication? (y/n): ").lower().strip()
    if auth_result in ['y', 'yes']:
        if not setup_modal_auth():
            print("❌ Setup failed: Authentication failed")
            return
    
    # Step 3: Deploy service
    endpoint_url = deploy_modal_service()
    if not endpoint_url:
        print("❌ Setup failed: Deployment failed")
        return
    
    # Step 4: Update configuration
    if not update_config(endpoint_url):
        print("❌ Setup failed: Could not update config.json")
        return
    
    # Step 5: Test integration
    test_integration()
    
    print("\n" + "=" * 60)
    print("🎉 Modal VLM setup completed!")
    print(f"📝 Endpoint URL: {endpoint_url}")
    print("🚀 You can now run: python scripts/test_video_processing.py")
    print("📚 For more details, see: deploy_modal_vlm.md")

if __name__ == "__main__":
    main() 
