#!/usr/bin/env python3
"""
Test Video Player Tool

Test the VideoPlayerTool functionality.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from cogniverse_agents.tools.video_player_tool import VideoPlayerTool


async def test_video_player():
    """Test the video player tool with sample data"""
    print("🧪 Testing Video Player Tool")
    print("=" * 50)
    
    # Initialize the tool
    player_tool = VideoPlayerTool()
    
    # Sample search results that would come from video search
    sample_search_results = json.dumps({
        "results": [
            {
                "video_id": "sample_video",
                "frame_id": 1,
                "start_time": 15.5,
                "score": 0.95,
                "description": "Person talking about machine learning concepts"
            },
            {
                "video_id": "sample_video", 
                "frame_id": 2,
                "start_time": 42.3,
                "score": 0.87,
                "description": "Whiteboard diagram showing neural network architecture"
            },
            {
                "video_id": "sample_video",
                "frame_id": 3, 
                "start_time": 78.1,
                "score": 0.82,
                "description": "Code example demonstrating PyTorch implementation"
            }
        ]
    })
    
    # Find first available video for testing
    video_dir = Path("data/videos")
    if not video_dir.exists():
        print(f"❌ Video directory not found: {video_dir}")
        return False
    
    # Search recursively for video files
    video_files = list(video_dir.glob("**/*.mp4")) + list(video_dir.glob("**/*.mov"))
    if not video_files:
        print(f"❌ No video files found in {video_dir}")
        return False
    
    test_video = video_files[0]
    video_id = test_video.stem
    
    print(f"📹 Testing with video: {test_video.name}")
    print(f"🆔 Video ID: {video_id}")
    
    # Test 1: Basic video player without search results
    print("\n🧪 Test 1: Basic video player")
    result1 = await player_tool.execute(video_id=video_id)
    
    if result1["success"]:
        print("✅ Basic video player test passed")
        print(f"   Message: {result1['message']}")
        # Extract HTML from artifact for testing
        html_content = result1['video_player'].inline_data.data.decode('utf-8')
        print(f"   Generated HTML length: {len(html_content)} characters")
    else:
        print(f"❌ Basic video player test failed: {result1['error']}")
        return False
    
    # Test 2: Video player with search results and frame markers
    print("\n🧪 Test 2: Video player with frame markers")
    result2 = await player_tool.execute(
        video_id=video_id,
        search_results=sample_search_results,
        start_time=15.5
    )
    
    if result2["success"]:
        print("✅ Video player with markers test passed")
        print(f"   Frame markers found: {result2['frame_count']}")
        print(f"   Message: {result2['message']}")
        # Extract HTML from artifact for testing
        html_content = result2['video_player'].inline_data.data.decode('utf-8')
        print(f"   Generated HTML length: {len(html_content)} characters")
    else:
        print(f"❌ Video player with markers test failed: {result2['error']}")
        return False
    
    # Test 3: Non-existent video
    print("\n🧪 Test 3: Non-existent video handling")
    result3 = await player_tool.execute(video_id="nonexistent_video")
    
    if not result3["success"] and "not found" in result3["error"].lower():
        print("✅ Non-existent video handling test passed")
    else:
        print(f"❌ Non-existent video handling test failed: {result3}")
        return False
    
    # Save sample HTML output for inspection
    sample_output = Path("logs/sample_video_player.html")
    sample_output.parent.mkdir(exist_ok=True)
    
    # Extract HTML from artifact to save
    html_content = result2['video_player'].inline_data.data.decode('utf-8')
    with open(sample_output, 'w') as f:
        f.write(html_content)
    
    print(f"\n📄 Sample HTML output saved to: {sample_output}")
    print("🎬 Video player uses embedded data URLs - no external server needed")
    print(f"   You can test the video player by opening: {sample_output}")
    
    print("\n✅ All video player tests passed!")
    return True


async def main():
    """Run all tests"""
    try:
        success = await test_video_player()
        if success:
            print("\n🎉 Video player tool is working correctly!")
        else:
            print("\n❌ Some tests failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # No cleanup needed - no external servers
        pass


if __name__ == "__main__":
    asyncio.run(main())