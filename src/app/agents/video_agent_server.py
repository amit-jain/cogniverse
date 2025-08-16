# src/agents/video_agent_server.py
import os
import json
import torch
import numpy as np
import datetime
import uvicorn
import logging
import argparse
from typing import List, Dict, Any, Optional, Union, Literal, Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from vespa.application import Vespa
from src.backends.vespa.vespa_search_client import VespaVideoSearchClient
from src.common.config import get_config
from src.app.agents.query_encoders import QueryEncoderFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- A2A Protocol Data Models ---
class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str

class DataPart(BaseModel):
    type: Literal["data"] = "data"
    data: Dict[str, Any]

class A2AMessage(BaseModel):
    role: str
    parts: List[Annotated[Union[TextPart, DataPart], Field(discriminator="type")]]

class Task(BaseModel):
    id: str
    messages: List[A2AMessage]

# --- Video Search Agent Implementation ---
class VideoSearchAgent:
    """
    An agent that performs hybrid search over a video database using Vespa.
    """
    def __init__(self, **kwargs):
        print("Initializing VideoSearchAgent with Vespa backend...")
        
        vespa_url = kwargs.get("vespa_url")
        vespa_port = kwargs.get("vespa_port")
        config = get_config()
        
        # Get model from active profile or use default
        active_profile = config.get_active_profile()
        profiles = config.get("video_processing_profiles", {})
        if active_profile and active_profile in profiles:
            model_name = profiles[active_profile].get("embedding_model", config.get("colpali_model", "vidore/colsmol-500m"))
            self.embedding_type = profiles[active_profile].get("embedding_type", "frame_based")
        else:
            model_name = kwargs.get("model_name", config.get("colpali_model", "vidore/colsmol-500m"))
            self.embedding_type = "frame_based"
            active_profile = "frame_based_colpali"  # Default profile
        
        if not vespa_url or not vespa_port:
            raise ValueError("Vespa backend requires 'vespa_url' and 'vespa_port'.")
        
        # Initialize Vespa search client
        try:
            self.vespa_client = VespaVideoSearchClient(vespa_url=vespa_url, vespa_port=vespa_port)
            print(f"Successfully initialized Vespa search client at {vespa_url}:{vespa_port}")
            print(f"Active profile: {active_profile}, Embedding type: {self.embedding_type}")
        except Exception as e:
            print(f"Could not initialize Vespa search client. Please ensure Vespa is running. Error: {e}")
            self.vespa_client = None
        
        # Initialize query encoder based on profile
        try:
            self.query_encoder = QueryEncoderFactory.create_encoder(active_profile, model_name)
            print(f"Initialized query encoder for profile: {active_profile}")
            print(f"Embedding dimension: {self.query_encoder.get_embedding_dim()}")
        except Exception as e:
            print(f"Error initializing query encoder: {e}")
            # Fall back to ColPali encoder
            self.query_encoder = QueryEncoderFactory.create_encoder("frame_based_colpali")
            print("Falling back to ColPali query encoder")

    def search(self, query: str, top_k: int = 10, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._search_vespa(query, top_k, start_date, end_date)

    def _encode_query_for_vespa(self, query: str) -> np.ndarray:
        """Encode query text to embeddings for Vespa search"""
        import time
        if not self.query_encoder:
            raise RuntimeError("Query encoder is not initialized for Vespa.")
        
        t1 = time.time()
        result = self.query_encoder.encode(query)
        t2 = time.time()
        logger.info(f"   Query encoding took {t2-t1:.3f}s")
        logger.info(f"   Embedding shape: {result.shape}")
        
        return result

    def _search_vespa(self, query: str, top_k: int, start_date: Optional[str], end_date: Optional[str]) -> List[Dict[str, Any]]:
        if not self.vespa_client:
            raise ConnectionError("Vespa search client is not available.")
        
        logger.info(f"📺 VIDEO AGENT: Received search request")
        logger.info(f"   Query: '{query}'")
        logger.info(f"   Top-K: {top_k}")
        logger.info(f"   Time range: {start_date} to {end_date}")
        
        try:
            import time
            start_time = time.time()
            
            logger.info(f"🧠 Encoding query embeddings...")
            query_embeddings = self._encode_query_for_vespa(query)
            encode_time = time.time() - start_time
            logger.info(f"   Encoding completed in {encode_time:.3f}s")
            logger.info(f"   Embedding shape: {query_embeddings.shape if query_embeddings is not None else 'None'}")
            
            search_params = {
                "query": query,
                "ranking": "hybrid_binary_bm25_no_description",
                "top_k": top_k
            }
            if start_date:
                search_params["start_date"] = start_date
            if end_date:
                search_params["end_date"] = end_date
            
            logger.info(f"🔍 Forwarding to Vespa search client...")
            logger.info(f"   Search params: {search_params}")
            logger.info(f"   Embeddings provided: {'Yes' if query_embeddings is not None else 'No'}")
            logger.info(f"   Embeddings type: {type(query_embeddings)}")
            search_start = time.time()
            results = self.vespa_client.search(search_params, query_embeddings)
            search_time = time.time() - search_start
            total_time = time.time() - start_time
            
            logger.info(f"✅ Vespa search completed: {len(results)} hits in {search_time:.3f}s")
            logger.info(f"   Total request time: {total_time:.3f}s")
            
            if results:
                logger.info(f"   Score range: {results[0]['relevance']:.3f} to {results[-1]['relevance']:.3f}")
                videos = set(r['video_id'] for r in results)
                logger.info(f"   Videos found: {len(videos)} videos with frames")
                # Log first few results
                for i, result in enumerate(results[:3]):
                    logger.info(f"   Result {i+1}: {result['video_id']} frame {result['frame_id']} (score: {result['relevance']:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ VIDEO AGENT: Vespa search failed: {e}")
            import traceback
            logger.error(f"   Error details: {traceback.format_exc()}")
            return []

# --- A2A Compliant FastAPI Server ---
app = FastAPI(title="Video Search Agent")

agent_config = {
    "vespa_url": os.getenv("VESPA_URL", "http://localhost"),
    "vespa_port": int(os.getenv("VESPA_PORT", 8080))
}

video_agent = VideoSearchAgent(**agent_config)

@app.get("/agent.json", summary="Get Agent Card")
async def get_agent_card():
    return {
        "name": "VideoSearchAgent", "description": "Finds video clips based on text and time filters.",
        "url": "/tasks/send", "version": "1.0", "protocol": "a2a", "protocol_version": "0.2.1",
        "capabilities": ["tasks/send"],
        "skills": [
            {
                "name": "videoSearch",
                "description": "Performs semantic search over video content with optional temporal filtering",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10},
                        "start_date": {"type": "string", "format": "date"},
                        "end_date": {"type": "string", "format": "date"}
                    },
                    "required": ["query"]
                }
            }
        ]
    }

@app.post("/tasks/send", status_code=202, summary="Send a Search Task")
async def send_task(task: Task):
    if not task.messages:
        raise HTTPException(status_code=400, detail="Task contains no messages.")
    
    last_message = task.messages[-1]
    data_part = next((part for part in last_message.parts if isinstance(part, DataPart)), None)
    
    if not data_part:
        raise HTTPException(status_code=400, detail="No DataPart found in the message.")

    query_data = data_part.data
    query = query_data.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' in DataPart.")

    try:
        search_results = video_agent.search(
            query=query, top_k=query_data.get("top_k", 10),
            start_date=query_data.get("start_date"), end_date=query_data.get("end_date")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    return {"task_id": task.id, "status": "completed", "results": search_results}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Search Agent Server")
    parser.add_argument("--port", type=int, default=8001, help="Port to run the server on (default: 8001)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--profile", type=str, help="Video processing profile to use (sets VIDEO_PROFILE env var)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Set profile if specified
    if args.profile:
        os.environ["VIDEO_PROFILE"] = args.profile
        # Reload config with the new profile
        from src.common.config import get_config
        app_config = get_config()
        app_config.reload()
        print(f"--- Using video processing profile: {args.profile} ---")
    
    # Get active profile for display
    config = get_config()
    active_profile = config.get_active_profile()
    
    print(f"--- Starting Video Search Agent Server ---")
    print(f"--- Port: {args.port} ---")
    print(f"--- Host: {args.host} ---")
    print(f"--- Active Profile: {active_profile} ---")
    print(f"--- Vespa backend configured ---")
    print(f"--- To run manually: uvicorn video_agent_server:app --reload --port {args.port} ---")
    
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)