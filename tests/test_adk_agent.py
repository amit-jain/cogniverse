#!/usr/bin/env python3
"""Test ADK Agent invocation"""

import asyncio
import sys
sys.path.append('video_rag_agent')
sys.path.append('.')

from video_rag_agent.agent import root_agent
from google.adk.invocation import ExecutorInvocation
from google.genai.types import Content, Part

async def test_agent():
    print("=== Testing ADK Video RAG Agent ===")
    
    # Create an invocation context
    invocation = ExecutorInvocation(
        new_message=Content(parts=[Part.from_text("show me videos about protecting your head while shoveling snow")])
    )
    
    try:
        print("\nSending query: 'show me videos about protecting your head while shoveling snow'")
        print("\nAgent response:")
        print("-" * 80)
        
        # Run the agent
        async for event in root_agent._run_async_impl(invocation):
            print(f"\nEvent from {event.author}:")
            
            # Display content parts
            for i, part in enumerate(event.content.parts):
                if hasattr(part, 'text') and part.text:
                    print(f"\nPart {i} (text):\n{part.text}")
                elif hasattr(part, 'inline_data') and part.inline_data:
                    mime_type = getattr(part.inline_data, 'mime_type', 'unknown')
                    data_size = len(getattr(part.inline_data, 'data', b''))
                    print(f"\nPart {i} (artifact): {mime_type} ({data_size} bytes)")
                else:
                    print(f"\nPart {i}: {type(part)}")
        
        print("-" * 80)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agent())