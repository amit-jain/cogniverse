"""
Metadata fetcher for retrieving video information from Vespa or cache
"""

import logging
import json
from typing import Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoMetadataFetcher:
    """
    Fetches video metadata from Vespa or cache for evaluation
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize metadata fetcher
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._cache = {}
        self._load_cache_if_available()
    
    def _load_cache_if_available(self):
        """Load cached metadata if available"""
        cache_path = Path("outputs/cache/video_metadata.json")
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} cached video metadata entries")
            except Exception as e:
                logger.warning(f"Could not load metadata cache: {e}")
    
    async def fetch_metadata(self, video_id: str) -> Dict:
        """
        Fetch metadata for a video
        
        Args:
            video_id: Video ID to fetch
            
        Returns:
            Dictionary with video metadata
        """
        # Check cache first
        if video_id in self._cache:
            return self._cache[video_id]
        
        # Try to fetch from Vespa
        metadata = await self._fetch_from_vespa(video_id)
        
        if metadata:
            # Cache the result
            self._cache[video_id] = metadata
            self._save_cache()
        
        return metadata or self._get_default_metadata(video_id)
    
    async def _fetch_from_vespa(self, video_id: str) -> Optional[Dict]:
        """
        Fetch metadata from Vespa
        
        Args:
            video_id: Video ID to fetch
            
        Returns:
            Metadata dict or None
        """
        try:
            # Import here to avoid circular dependencies
            from src.search.vespa_search_backend import VespaSearchBackend
            
            # Create a temporary backend instance
            backend = VespaSearchBackend(self.config)
            
            # Query Vespa for the specific video
            query = f'select * from sources * where source_id contains "{video_id}"'
            response = backend.app.query(
                yql=query,
                hits=1
            )
            
            if response.hits:
                hit = response.hits[0]
                fields = hit.get("fields", {})
                
                return {
                    "video_id": video_id,
                    "title": fields.get("title", f"Video {video_id}"),
                    "description": fields.get("description", ""),
                    "transcript": fields.get("transcript", ""),
                    "frame_descriptions": fields.get("frame_descriptions", []),
                    "duration": fields.get("duration", 0),
                    "tags": fields.get("tags", []),
                    "source": "vespa"
                }
                
        except Exception as e:
            logger.debug(f"Could not fetch from Vespa: {e}")
        
        return None
    
    def _get_default_metadata(self, video_id: str) -> Dict:
        """
        Get default metadata when not found
        
        Args:
            video_id: Video ID
            
        Returns:
            Default metadata dict
        """
        # Try to extract some info from video_id
        # Common patterns: v_XXXX, video_name_001, etc.
        
        title = video_id
        if video_id.startswith("v_"):
            title = f"Video {video_id[2:]}"
        elif "_" in video_id:
            parts = video_id.split("_")
            title = " ".join(p.capitalize() for p in parts)
        
        return {
            "video_id": video_id,
            "title": title,
            "description": f"Video content for {video_id}",
            "transcript": "",
            "frame_descriptions": [],
            "duration": 0,
            "tags": [],
            "source": "default"
        }
    
    def _save_cache(self):
        """Save cache to disk"""
        cache_path = Path("outputs/cache/video_metadata.json")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save metadata cache: {e}")
    
    async def fetch_batch(self, video_ids: List[str]) -> Dict[str, Dict]:
        """
        Fetch metadata for multiple videos
        
        Args:
            video_ids: List of video IDs
            
        Returns:
            Dictionary mapping video_id to metadata
        """
        results = {}
        for video_id in video_ids:
            results[video_id] = await self.fetch_metadata(video_id)
        return results