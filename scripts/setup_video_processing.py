#!/usr/bin/env python3
"""
Setup script for video processing pipeline:
1. Deploy Modal VLM service
2. Update configuration
3. Process test videos
"""

import os
import json
import subprocess
import sys
import re
from pathlib import Path

def check_modal_setup():
    """Check if Modal is installed and working."""
    try:
        result = subprocess.run(["modal", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Modal CLI found: {result.stdout.strip()}")
            return True
        else:
            print("❌ Modal CLI not working properly")
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

def deploy_modal_vlm():
    """Deploy Modal VLM service and return endpoint URL."""
    print("🚀 Deploying Modal VLM service...")
    try:
        # First check if we're authenticated
        auth_check = subprocess.run(["modal", "config", "list"], capture_output=True, text=True)
        if auth_check.returncode != 0:
            print("🔐 Setting up Modal authentication...")
            subprocess.run(["modal", "setup"], check=True)
        
        # Deploy the service
        result = subprocess.run(
            ["modal", "deploy", "modal_vlm_service.py"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print("✅ Modal VLM service deployed successfully")
            
            # Extract endpoint URL from output
            output = result.stdout + result.stderr
            url_pattern = r'https://[^\\s]+--generate-description\\.modal\\.run'
            matches = re.findall(url_pattern, output)
            
            if matches:
                endpoint_url = matches[0]
                print(f"🌐 VLM Endpoint: {endpoint_url}")
                return endpoint_url
            else:
                print("⚠️ Could not extract endpoint URL from deployment output")
                print("💡 Check Modal dashboard at https://modal.com")
                return None
        else:
            print(f"❌ Deployment failed:")
            print(result.stderr)
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ Deployment timeout (10 minutes)")
        return None
    except Exception as e:
        print(f"❌ Deployment error: {e}")
        return None

def update_config_with_endpoint(endpoint_url):
    """Update config.json with VLM endpoint URL."""
    config_path = Path("config.json")
    
    if not config_path.exists():
        print("❌ config.json not found")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update VLM endpoint and enable descriptions
        config["vlm_endpoint_url"] = endpoint_url
        config["pipeline_config"]["generate_descriptions"] = True
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Updated config.json with VLM endpoint")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update config.json: {e}")
        return False

def test_vlm_endpoint(endpoint_url):
    """Test the VLM endpoint with a simple request."""
    print("🧪 Testing VLM endpoint...")
    try:
        import requests
        import base64
        from PIL import Image
        import io
        
        # Create a simple test image
        test_img = Image.new('RGB', (100, 100), color='red')
        img_buffer = io.BytesIO()
        test_img.save(img_buffer, format='JPEG')
        frame_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        payload = {
            "frame_base64": frame_base64,
            "prompt": "What color is this image?"
        }
        
        response = requests.post(
            endpoint_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            description = result.get("description", "")
            print(f"✅ VLM test successful: {description}")
            return True
        else:
            print(f"❌ VLM test failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ VLM test error: {e}")
        return False

def main():
    """Main setup function."""
    print("🎬 Video Processing Pipeline Setup")
    print("=" * 50)
    
    # Step 1: Check/install Modal
    print("\\n1️⃣ Checking Modal CLI...")
    if not check_modal_setup():
        print("Installing Modal CLI...")
        if not install_modal():
            print("❌ Setup failed: Could not install Modal CLI")
            return
        
        if not check_modal_setup():
            print("❌ Setup failed: Modal CLI still not working")
            return
    
    # Step 2: Deploy VLM service
    print("\\n2️⃣ Deploying Modal VLM service...")
    endpoint_url = deploy_modal_vlm()
    
    if not endpoint_url:
        print("❌ Setup failed: Could not deploy VLM service")
        return
    
    # Step 3: Update configuration
    print("\\n3️⃣ Updating configuration...")
    if not update_config_with_endpoint(endpoint_url):
        print("❌ Setup failed: Could not update configuration")
        return
    
    # Step 4: Test VLM endpoint
    print("\\n4️⃣ Testing VLM endpoint...")
    if not test_vlm_endpoint(endpoint_url):
        print("⚠️ VLM endpoint test failed, but continuing...")
    
    # Step 5: Process test videos
    print("\\n5️⃣ Processing test videos...")
    try:
        print("🎬 Starting video processing...")
        result = subprocess.run([
            sys.executable, "scripts/process_test_videos.py"
        ], check=True)
        print("✅ Video processing completed!")
    except subprocess.CalledProcessError:
        print("❌ Video processing failed")
        print("💡 You can run it manually: python scripts/process_test_videos.py")
    
    print("\\n" + "=" * 50)
    print("🎉 Video processing pipeline setup completed!")
    print(f"🌐 VLM Endpoint: {endpoint_url}")
    print("📁 Processed data saved to: data/videos/video_chatgpt_eval/processed/")
    print("🚀 Next step: Run video ingestion with processed data")

if __name__ == "__main__":
    main()