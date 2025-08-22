"""
Enhanced Video Search Agent with support for both text-to-video and video-to-video search.
Integrates with existing VideoSearchAgent and adds video upload/encoding capabilities.
"""

import os
import io
import logging
import tempfile
from typing import List, Dict, Any, Optional, Union, Literal, Annotated
from pathlib import Path

import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from src.tools.a2a_utils import A2AMessage, DataPart, TextPart, Task
from src.common.config import get_config
from src.app.agents.query_encoders import QueryEncoderFactory
from src.backends.vespa.vespa_search_client import VespaVideoSearchClient

logger = logging.getLogger(__name__)

# --- Enhanced Data Models ---
class VideoPart(BaseModel):
    """Video content part for A2A messages"""
    type: Literal["video"] = "video"
    video_data: bytes = Field(..., description="Raw video file bytes")
    filename: Optional[str] = Field(None, description="Original filename")
    content_type: Optional[str] = Field(None, description="MIME type")

class ImagePart(BaseModel):
    """Image content part for A2A messages"""
    type: Literal["image"] = "image"
    image_data: bytes = Field(..., description="Raw image file bytes")
    filename: Optional[str] = Field(None, description="Original filename")
    content_type: Optional[str] = Field(None, description="MIME type")

class EnhancedA2AMessage(BaseModel):
    """Enhanced A2A message supporting multimedia content"""
    role: str
    parts: List[Annotated[Union[TextPart, DataPart, VideoPart, ImagePart], Field(discriminator="type")]]

class EnhancedTask(BaseModel):
    """Enhanced task supporting multimedia content"""
    id: str
    messages: List[EnhancedA2AMessage]

# --- Video Processing Components ---
class VideoProcessor:
    """Handles video upload, processing, and encoding to embeddings"""
    
    def __init__(self, query_encoder):
        self.query_encoder = query_encoder
        self.temp_dir = Path(tempfile.gettempdir()) / "video_search_agent"
        self.temp_dir.mkdir(exist_ok=True)
        
    def process_video_file(self, video_data: bytes, filename: str) -> np.ndarray:
        """
        Process uploaded video file and extract embeddings.
        
        Args:
            video_data: Raw video file bytes
            filename: Original filename
            
        Returns:
            Video embeddings as numpy array
        """
        # Save video to temporary file
        temp_video_path = self.temp_dir / f"temp_{filename}"
        
        try:
            with open(temp_video_path, 'wb') as f:
                f.write(video_data)
            
            logger.info(f"Processing video file: {filename}")
            
            # Extract embeddings using query encoder's video processing capability
            embeddings = self._extract_video_embeddings(temp_video_path)
            
            logger.info(f"Extracted embeddings shape: {embeddings.shape}")
            return embeddings
            
        finally:
            # Clean up temporary file
            if temp_video_path.exists():
                temp_video_path.unlink()
    
    def process_image_file(self, image_data: bytes, filename: str) -> np.ndarray:
        """
        Process uploaded image file and extract embeddings.
        
        Args:
            image_data: Raw image file bytes
            filename: Original filename
            
        Returns:
            Image embeddings as numpy array
        """
        # Save image to temporary file
        temp_image_path = self.temp_dir / f"temp_{filename}"
        
        try:
            with open(temp_image_path, 'wb') as f:
                f.write(image_data)
            
            logger.info(f"Processing image file: {filename}")
            
            # Extract embeddings using query encoder's image processing capability
            embeddings = self._extract_image_embeddings(temp_image_path)
            
            logger.info(f"Extracted embeddings shape: {embeddings.shape}")
            return embeddings
            
        finally:
            # Clean up temporary file
            if temp_image_path.exists():
                temp_image_path.unlink()
    
    def _extract_video_embeddings(self, video_path: Path) -> np.ndarray:
        """Extract embeddings from video file using the query encoder"""
        if hasattr(self.query_encoder, 'encode_video'):
            return self.query_encoder.encode_video(str(video_path))
        elif hasattr(self.query_encoder, 'encode_frames'):
            # For frame-based encoders, extract frames and encode
            return self._extract_frames_and_encode(video_path)
        else:
            raise NotImplementedError("Query encoder does not support video encoding")
    
    def _extract_image_embeddings(self, image_path: Path) -> np.ndarray:
        """Extract embeddings from image file using the query encoder"""
        if hasattr(self.query_encoder, 'encode_image'):
            return self.query_encoder.encode_image(str(image_path))
        elif hasattr(self.query_encoder, 'encode'):
            # For text encoders, this won't work - need to implement image support
            raise NotImplementedError("Query encoder does not support image encoding")
        else:
            raise NotImplementedError("Query encoder does not support image encoding")
    
    def _extract_frames_and_encode(self, video_path: Path) -> np.ndarray:
        """Extract frames from video and encode them"""
        try:
            import cv2
            
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            
            # Extract a few key frames
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Extract frames at 1 FPS or every 30 frames, whichever is less frequent
            step = max(fps, 30)
            
            for i in range(0, frame_count, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                    if len(frames) >= 10:  # Limit to 10 frames
                        break
            
            cap.release()
            
            if not frames:
                raise ValueError("No frames extracted from video")
            
            # Encode frames using the query encoder
            if hasattr(self.query_encoder, 'encode_frames'):
                return self.query_encoder.encode_frames(frames)
            else:
                # Save frames as temporary images and encode
                frame_embeddings = []
                for i, frame in enumerate(frames):
                    temp_frame_path = self.temp_dir / f"temp_frame_{i}.jpg"
                    cv2.imwrite(str(temp_frame_path), frame)
                    
                    try:
                        if hasattr(self.query_encoder, 'encode_image'):
                            frame_emb = self.query_encoder.encode_image(str(temp_frame_path))
                            frame_embeddings.append(frame_emb)
                    finally:
                        temp_frame_path.unlink()
                
                if not frame_embeddings:
                    raise ValueError("No frame embeddings extracted")
                
                # Average the frame embeddings
                return np.mean(frame_embeddings, axis=0)
                
        except ImportError:
            raise ImportError("OpenCV is required for video frame extraction. Install with: pip install opencv-python")

# --- Enhanced Video Search Agent ---
class EnhancedVideoSearchAgent:
    """
    Enhanced video search agent supporting both text-to-video and video-to-video search.
    """
    
    def __init__(self, **kwargs):
        """Initialize enhanced video search agent"""
        logger.info("Initializing EnhancedVideoSearchAgent...")
        
        # Initialize base configuration
        self.config = get_config()
        vespa_url = kwargs.get("vespa_url", "http://localhost")
        vespa_port = kwargs.get("vespa_port", 8080)
        
        # Get model from active profile
        active_profile = self.config.get_active_profile()
        profiles = self.config.get("video_processing_profiles", {})
        
        if active_profile and active_profile in profiles:
            model_name = profiles[active_profile].get("embedding_model", "vidore/colsmol-500m")
            self.embedding_type = profiles[active_profile].get("embedding_type", "frame_based")
        else:
            model_name = kwargs.get("model_name", "vidore/colsmol-500m")
            self.embedding_type = "frame_based"
            active_profile = "frame_based_colpali"
        
        # Initialize Vespa search client
        try:
            self.vespa_client = VespaVideoSearchClient(vespa_url=vespa_url, vespa_port=vespa_port)
            logger.info(f"Vespa client initialized at {vespa_url}:{vespa_port}")
        except Exception as e:
            logger.error(f"Failed to initialize Vespa client: {e}")
            raise
        
        # Initialize query encoder
        try:
            self.query_encoder = QueryEncoderFactory.create_encoder(active_profile, model_name)
            logger.info(f"Query encoder initialized for profile: {active_profile}")
        except Exception as e:
            logger.error(f"Failed to initialize query encoder: {e}")
            raise
        
        # Initialize video processor
        self.video_processor = VideoProcessor(self.query_encoder)
        
        logger.info("EnhancedVideoSearchAgent initialization complete")
    
    def search_by_text(self, query: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Search videos using text query.
        
        Args:
            query: Text search query
            top_k: Number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        logger.info(f"Text-to-video search: '{query}' (top_k={top_k})")
        
        try:
            # Encode text query
            query_embeddings = self.query_encoder.encode(query)
            
            # Prepare search parameters
            search_params = {
                "query": query,
                "ranking": kwargs.get("ranking", "hybrid_binary_bm25_no_description"),
                "top_k": top_k
            }
            
            # Add optional filters
            if "start_date" in kwargs:
                search_params["start_date"] = kwargs["start_date"]
            if "end_date" in kwargs:
                search_params["end_date"] = kwargs["end_date"]
            
            # Execute search
            results = self.vespa_client.search(search_params, query_embeddings)
            
            logger.info(f"Text search completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            raise
    
    def search_by_video(self, video_data: bytes, filename: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Search videos using video query.
        
        Args:
            video_data: Raw video file bytes
            filename: Original filename
            top_k: Number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        logger.info(f"Video-to-video search with file: '{filename}' (top_k={top_k})")
        
        try:
            # Extract embeddings from uploaded video
            query_embeddings = self.video_processor.process_video_file(video_data, filename)
            
            # Prepare search parameters
            search_params = {
                "query": f"Video similarity search for {filename}",
                "ranking": kwargs.get("ranking", "hybrid_binary_bm25_no_description"),
                "top_k": top_k
            }
            
            # Add optional filters
            if "start_date" in kwargs:
                search_params["start_date"] = kwargs["start_date"]
            if "end_date" in kwargs:
                search_params["end_date"] = kwargs["end_date"]
            
            # Execute search
            results = self.vespa_client.search(search_params, query_embeddings)
            
            logger.info(f"Video search completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Video search failed: {e}")
            raise
    
    def search_by_image(self, image_data: bytes, filename: str, top_k: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Search videos using image query.
        
        Args:
            image_data: Raw image file bytes
            filename: Original filename
            top_k: Number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        logger.info(f"Image-to-video search with file: '{filename}' (top_k={top_k})")
        
        try:
            # Extract embeddings from uploaded image
            query_embeddings = self.video_processor.process_image_file(image_data, filename)
            
            # Prepare search parameters
            search_params = {
                "query": f"Image similarity search for {filename}",
                "ranking": kwargs.get("ranking", "hybrid_binary_bm25_no_description"),
                "top_k": top_k
            }
            
            # Add optional filters
            if "start_date" in kwargs:
                search_params["start_date"] = kwargs["start_date"]
            if "end_date" in kwargs:
                search_params["end_date"] = kwargs["end_date"]
            
            # Execute search
            results = self.vespa_client.search(search_params, query_embeddings)
            
            logger.info(f"Image search completed: {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Image search failed: {e}")
            raise
    
    def process_enhanced_task(self, task: EnhancedTask) -> Dict[str, Any]:
        """
        Process enhanced A2A task with multimedia support.
        
        Args:
            task: Enhanced A2A task
            
        Returns:
            Search results
        """
        if not task.messages:
            raise ValueError("Task contains no messages")
        
        last_message = task.messages[-1]
        results = []
        search_type = "unknown"
        
        for part in last_message.parts:
            if isinstance(part, DataPart):
                # Text search
                query_data = part.data
                query = query_data.get("query")
                
                if query:
                    search_type = "text"
                    text_results = self.search_by_text(
                        query=query,
                        top_k=query_data.get("top_k", 10),
                        start_date=query_data.get("start_date"),
                        end_date=query_data.get("end_date"),
                        ranking=query_data.get("ranking")
                    )
                    results.extend(text_results)
            
            elif isinstance(part, VideoPart):
                # Video search
                search_type = "video"
                video_results = self.search_by_video(
                    video_data=part.video_data,
                    filename=part.filename or "uploaded_video.mp4",
                    top_k=10  # Could be configurable
                )
                results.extend(video_results)
            
            elif isinstance(part, ImagePart):
                # Image search
                search_type = "image"
                image_results = self.search_by_image(
                    image_data=part.image_data,
                    filename=part.filename or "uploaded_image.jpg",
                    top_k=10  # Could be configurable
                )
                results.extend(image_results)
            
            elif isinstance(part, TextPart):
                # Simple text search
                search_type = "text"
                text_results = self.search_by_text(query=part.text, top_k=10)
                results.extend(text_results)
        
        if not results:
            logger.warning("No valid search parts found in task")
        
        return {
            "task_id": task.id,
            "status": "completed",
            "search_type": search_type,
            "results": results,
            "total_results": len(results)
        }

# --- FastAPI Server ---
app = FastAPI(
    title="Enhanced Video Search Agent",
    description="Video search agent with support for text, video, and image queries",
    version="3.0.0"
)

# Global agent instance - initialized on startup
enhanced_video_agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global enhanced_video_agent
    
    agent_config = {
        "vespa_url": os.getenv("VESPA_URL", "http://localhost"),
        "vespa_port": int(os.getenv("VESPA_PORT", 8080))
    }
    
    try:
        enhanced_video_agent = EnhancedVideoSearchAgent(**agent_config)
        logger.info("Enhanced video search agent initialized")
    except Exception as e:
        logger.error(f"Failed to initialize enhanced agent: {e}")
        # Don't raise during tests
        if not os.getenv("PYTEST_CURRENT_TEST"):
            raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not enhanced_video_agent:
        return {
            "status": "initializing",
            "agent": "enhanced_video_search"
        }
    
    return {
        "status": "healthy",
        "agent": "enhanced_video_search",
        "capabilities": ["text_search", "video_search", "image_search"],
        "embedding_type": enhanced_video_agent.embedding_type
    }

@app.get("/agent.json")
async def get_agent_card():
    """Agent card with enhanced capabilities"""
    return {
        "name": "EnhancedVideoSearchAgent",
        "description": "Advanced video search with text, video, and image query support",
        "url": "/process",
        "version": "3.0.0",
        "protocol": "a2a",
        "protocol_version": "0.2.1",
        "capabilities": ["text_search", "video_search", "image_search", "multimodal_search"],
        "skills": [
            {
                "name": "textVideoSearch",
                "description": "Search videos using text queries",
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
            },
            {
                "name": "videoVideoSearch",
                "description": "Search videos using video files as queries",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "video_data": {"type": "string", "format": "binary"},
                        "filename": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10}
                    },
                    "required": ["video_data"]
                }
            },
            {
                "name": "imageVideoSearch", 
                "description": "Search videos using image files as queries",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "image_data": {"type": "string", "format": "binary"},
                        "filename": {"type": "string"},
                        "top_k": {"type": "integer", "default": 10}
                    },
                    "required": ["image_data"]
                }
            }
        ]
    }

@app.post("/process")
async def process_task(task: EnhancedTask):
    """Process enhanced search task"""
    if not enhanced_video_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        result = enhanced_video_agent.process_enhanced_task(task)
        return result
    except Exception as e:
        logger.error(f"Task processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/video")
async def upload_video_search(
    file: UploadFile = File(...),
    top_k: int = 10
):
    """Upload video file and search for similar videos"""
    if not enhanced_video_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        video_data = await file.read()
        results = enhanced_video_agent.search_by_video(
            video_data=video_data,
            filename=file.filename or "uploaded_video.mp4",
            top_k=top_k
        )
        
        return {
            "status": "completed",
            "search_type": "video",
            "filename": file.filename,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Video upload search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/image") 
async def upload_image_search(
    file: UploadFile = File(...),
    top_k: int = 10
):
    """Upload image file and search for similar videos"""
    if not enhanced_video_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        image_data = await file.read()
        results = enhanced_video_agent.search_by_image(
            image_data=image_data,
            filename=file.filename or "uploaded_image.jpg",
            top_k=top_k
        )
        
        return {
            "status": "completed", 
            "search_type": "image",
            "filename": file.filename,
            "results": results,
            "total_results": len(results)
        }
        
    except Exception as e:
        logger.error(f"Image upload search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Video Search Agent Server")
    parser.add_argument("--port", type=int, default=8002, help="Port to run the server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--profile", type=str, help="Video processing profile to use")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    if args.profile:
        os.environ["VIDEO_PROFILE"] = args.profile
    
    logger.info(f"Starting Enhanced Video Search Agent on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)