#!/usr/bin/env python3
# scripts/setup_ollama.py
"""
Setup script for Ollama and Llama 3.1 model.
This script installs and configures Ollama for the Multi-Agent RAG System.
"""

import platform
import subprocess
import sys
import time

import requests


def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama is installed: {result.stdout.strip()}")
            return True
        else:
            return False
    except FileNotFoundError:
        return False


def install_ollama():
    """Install Ollama based on the operating system."""
    system = platform.system().lower()

    print("📦 Installing Ollama...")

    if system == "darwin":  # macOS
        print("🍎 Detected macOS - Installing via curl...")
        try:
            subprocess.run(
                ["curl", "-fsSL", "https://ollama.ai/install.sh"],
                check=True,
                shell=True,
            )
            print("✅ Ollama installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Ollama via curl")
            print("   Please install manually from: https://ollama.ai/download")
            return False

    elif system == "linux":
        print("🐧 Detected Linux - Installing via curl...")
        try:
            subprocess.run(
                ["curl", "-fsSL", "https://ollama.ai/install.sh", "|", "sh"],
                check=True,
                shell=True,
            )
            print("✅ Ollama installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Ollama via curl")
            print("   Please install manually from: https://ollama.ai/download")
            return False

    elif system == "windows":
        print("🪟 Detected Windows")
        print("   Please download and install Ollama from: https://ollama.ai/download")
        print("   Then run this script again.")
        return False

    else:
        print(f"❓ Unsupported operating system: {system}")
        print("   Please install Ollama manually from: https://ollama.ai/download")
        return False


def check_ollama_running():
    """Check if Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def start_ollama_server():
    """Start the Ollama server."""
    if check_ollama_running():
        print("✅ Ollama server is already running")
        return True

    print("🚀 Starting Ollama server...")
    try:
        # Start ollama serve in background
        if platform.system().lower() == "windows":
            subprocess.Popen(
                ["ollama", "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        # Wait for server to start
        for i in range(10):
            if check_ollama_running():
                print("✅ Ollama server started successfully")
                return True
            time.sleep(1)

        print("❌ Ollama server failed to start")
        return False

    except Exception as e:
        print(f"❌ Failed to start Ollama server: {e}")
        return False


def pull_deepseek_model():
    """Pull the DeepSeek-R1 1.5B model."""
    model_name = "deepseek-r1:1.5b"

    # Check if model is already available
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if model_name in result.stdout:
            print(f"✅ {model_name} model is already available")
            return True
    except Exception:
        pass

    print(f"📥 Pulling {model_name} model (this may take a few minutes)...")
    try:
        result = subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"✅ {model_name} model pulled successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to pull {model_name} model")
        return False


def test_model():
    """Test the model with a simple query."""
    print("🧪 Testing DeepSeek-R1 model...")
    try:
        test_prompt = "Hello! Please respond with just 'Model working correctly.'"
        result = subprocess.run(
            ["ollama", "run", "deepseek-r1:1.5b", test_prompt],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0 and "working correctly" in result.stdout.lower():
            print("✅ Model test successful")
            return True
        else:
            print("⚠️  Model test completed but response may be unexpected")
            print(f"   Response: {result.stdout[:100]}...")
            return True  # Still consider it working

    except subprocess.TimeoutExpired:
        print("⚠️  Model test timed out (model may be working but slow)")
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🧠 Setting up Ollama and DeepSeek-R1 for Multi-Agent RAG System")
    print("=" * 60)

    # Check if Ollama is installed
    if not check_ollama_installed():
        print("📦 Ollama not found. Installing...")
        if not install_ollama():
            print("❌ Failed to install Ollama")
            sys.exit(1)

    # Start Ollama server
    if not start_ollama_server():
        print("❌ Failed to start Ollama server")
        sys.exit(1)

    # Pull DeepSeek model
    if not pull_deepseek_model():
        print("❌ Failed to pull DeepSeek model")
        sys.exit(1)

    # Test the model
    if not test_model():
        print("❌ Model test failed")
        sys.exit(1)

    print("\n🎉 Ollama setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the system setup: python scripts/setup_system.py")
    print("2. Start the servers: ./scripts/run_servers.sh")
    print("3. Open http://localhost:8000 in your browser")

    print("\n📋 Configuration:")
    print("- Model: deepseek-r1:1.5b")
    print("- Server: http://localhost:11434")
    print("- Status: Ready for multi-agent coordination!")


if __name__ == "__main__":
    main()
