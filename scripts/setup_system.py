#!/usr/bin/env python3
# scripts/setup_system.py
"""
Setup script for the Multi-Agent RAG System.
This script initializes the video database and prepares the system for use.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.config_utils import get_config, setup_environment


def create_directories():
    """Create necessary data directories."""
    from cogniverse_foundation.config.utils import create_default_config_manager, get_config
    config_manager = create_default_config_manager()
    config = get_config(tenant_id="default", config_manager=config_manager)
    
    directories = [
        config.get("video_data_dir", "data/videos"),
        config.get("text_data_dir", "data/text"),
        config.get("index_dir", "data/indexes"),
        ".byaldi"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def download_sample_videos():
    """Download sample videos for testing."""
    video_dir = Path("data/videos")
    
    # Create a simple test video using imageio (if available)
    try:
        import imageio
        import numpy as np
        
        # Create a simple test video
        test_video_path = video_dir / "sample_test_video.mp4"
        if not test_video_path.exists():
            print("📹 Creating sample test video...")
            
            # Create a simple animation
            writer = imageio.get_writer(str(test_video_path), fps=30)
            
            for i in range(90):  # 3 seconds at 30fps
                # Create a simple moving square
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                x_pos = int((i / 90) * 500 + 50)
                frame[200:280, x_pos:x_pos+80] = [255, 0, 0]  # Red square
                
                # Add some text-like patterns
                if i % 30 < 15:
                    frame[100:120, 50:590] = [255, 255, 255]  # White bar (simulating text)
                
                writer.append_data(frame)
            
            writer.close()
            print(f"✅ Created sample video: {test_video_path}")
        
    except ImportError:
        print("⚠️  imageio not available, skipping sample video creation")
        print("   You can add your own MP4 files to data/videos/ directory")

def create_sample_content():
    """Create sample content files."""
    video_dir = Path("data/videos")
    
    # Create a README in the videos directory
    readme_path = video_dir / "README.md"
    if not readme_path.exists():
        with open(readme_path, "w") as f:
            f.write("""# Video Content Directory

This directory contains video files for the Multi-Agent RAG System.

## Supported Formats
- MP4 (recommended)
- MOV
- AVI

## Usage
1. Place your video files in this directory
2. Run the ingestion pipeline: `python scripts/run_ingestion.py --video_dir data/videos --backend byaldi`
3. Start the servers: `./scripts/run_servers.sh`

## Sample Videos
- sample_test_video.mp4 - A simple test video with moving shapes

## Processing
The system will automatically:
- Extract keyframes from videos
- Generate visual descriptions using VLLM
- Transcribe audio using Whisper
    - Create multi-vector embeddings using ColPali
- Index everything for semantic search
""")
        print("✅ Created video directory README")

def setup_byaldi_index():
    """Set up the Byaldi index with sample content using existing ingestion pipeline."""
    video_dir = Path("data/videos")
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.mov")) + list(video_dir.glob("*.avi"))
    
    if not video_files:
        print("⚠️  No video files found in data/videos/")
        print("   Add some video files and run the ingestion pipeline manually:")
        print(f"   python scripts/run_ingestion.py --video_dir {video_dir} --backend byaldi")
        return False
    
    print(f"📹 Found {len(video_files)} video files:")
    for video_file in video_files:
        print(f"   - {video_file.name}")
    
    # Check if byaldi index already exists
    index_path = Path(".byaldi/my_video_index")
    if index_path.exists():
        print("✅ Byaldi index already exists")
        return True
    
    print("🔄 Processing videos and creating Byaldi index...")
    print("   This may take a few minutes depending on video content and models...")
    print("   The system will:")
    print("   - Extract keyframes from videos")
    print("   - Generate visual descriptions using Qwen2-VL")
    print("   - Transcribe audio using Faster-Whisper")
    print("   - Create multi-vector embeddings using ColPali")
    print("   - Index everything in Byaldi for fast retrieval")
    
    # Import and use the existing ingestion pipeline
    try:
        # Add the parent directory to sys.path to import the ingestion module
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        # Import the main function from the existing ingestion script
        import argparse

        from scripts.run_ingestion import main as run_ingestion_main
        
        # Simulate command line arguments for the ingestion script
        original_argv = sys.argv
        sys.argv = [
            "run_ingestion.py",
            "--video_dir", str(video_dir),
            "--backend", "byaldi",
            "--index_name", ".byaldi/my_video_index"
        ]
        
        try:
            # Run the ingestion pipeline
            run_ingestion_main()
            print("✅ Video processing completed successfully")
            return True
        except Exception as e:
            print(f"❌ Video processing failed: {e}")
            return False
        finally:
            # Restore original argv
            sys.argv = original_argv
            
    except ImportError as e:
        print(f"❌ Could not import ingestion pipeline: {e}")
        print("   Falling back to subprocess method...")
        
        # Fallback to subprocess if import fails
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, "scripts/run_ingestion.py",
                "--video_dir", str(video_dir),
                "--backend", "byaldi",
                "--index_name", ".byaldi/my_video_index"
            ], cwd=Path(__file__).parent.parent)
            
            if result.returncode == 0:
                print("✅ Video processing completed successfully")
                return True
            else:
                print(f"❌ Video processing failed with return code: {result.returncode}")
                return False
                
        except Exception as e:
            print(f"❌ Failed to run ingestion pipeline: {e}")
            return False

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_modules = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("byaldi", "Byaldi"),
        ("colpali_engine", "ColPali Engine"),
        ("faster_whisper", "Faster Whisper"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV")
    ]
    
    missing_modules = []
    
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"   ✅ {display_name}")
        except ImportError:
            missing_modules.append(display_name)
            print(f"   ❌ {display_name}")
    
    if missing_modules:
        print(f"\n❌ Missing dependencies: {', '.join(missing_modules)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies available")
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up Multi-Agent RAG System")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment and directories
    setup_environment()
    create_directories()
    create_sample_content()
    download_sample_videos()
    
    # Setup video index
    print("\n📊 Setting up video search index...")
    if setup_byaldi_index():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Start the servers: ./scripts/run_servers.sh")
        print("2. Open http://localhost:8000 in your browser")
        print("3. Select 'CoordinatorAgent' and start asking questions!")
        print("\nExample queries to try:")
        print("- 'Show me videos with moving objects'")
        print("- 'Find clips from the test video'")
        print("- 'Search for videos with red squares'")
    else:
        print("\n⚠️  Setup completed with warnings")
        print("Video index was not created - you may need to:")
        print("1. Add video files to data/videos/")
        print("2. Run: python scripts/run_ingestion.py --video_dir data/videos --backend byaldi")
        print("3. Then start the servers: ./scripts/run_servers.sh")

if __name__ == "__main__":
    main() 
