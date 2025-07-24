#!/usr/bin/env python3
"""Test BaseAgent implementation directly"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from video_rag_agent.agent import root_agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

async def test_agent():
    """Test the agent directly"""
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name="video_rag_agent",
        session_service=session_service
    )
    
    user_id = "test_user"
    session = await session_service.create_session(user_id=user_id, app_name="video_rag_agent")
    session_id = session.id
    
    print(f"Testing agent with session: {session_id}")
    
    try:
        events = []
        user_message = Content(parts=[Part(text="Show me videos about medical procedures")])
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=user_message
        ):
            print(f"Got event: {event}")
            events.append(event)
            
        print(f"\nTotal events: {len(events)}")
        if events:
            print(f"Final response: {events[-1].content.parts[0].text if events[-1].content else 'No content'}")
            
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agent())