"""
Metadata fetcher for retrieving video information from Vespa or cache
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoMetadataFetcher:
    """
    Fetches video metadata from Vespa or cache for evaluation
    """

    def __init__(self, config: dict = None):
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
                with open(cache_path) as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} cached video metadata entries")
            except Exception as e:
                logger.warning(f"Could not load metadata cache: {e}")

    async def fetch_metadata(self, video_id: str) -> dict:
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

    async def _fetch_from_vespa(self, video_id: str) -> dict | None:
        """
        Fetch metadata from Vespa

        Args:
            video_id: Video ID to fetch

        Returns:
            Metadata dict or None
        """
        try:
            # Use search service instead of direct backend access
            from src.app.search.service import SearchService

            # Use search service to find the video
            # This keeps evaluation independent of backend implementation
            search_service = SearchService(
                self.config, profile="video_colpali_smol500_mv_frame"
            )
            results = search_service.search(query=f"video_id:{video_id}", top_k=1)

            if results:
                # SearchService returns SearchResult objects
                result = results[0]
                fields = result.metadata if hasattr(result, "metadata") else {}

                return {
                    "video_id": video_id,
                    "title": fields.get(
                        "video_title", f"Video {video_id}"
                    ),  # Correct field name
                    "description": fields.get(
                        "frame_description", ""
                    ),  # Using frame_description
                    "transcript": fields.get(
                        "audio_transcript", ""
                    ),  # Correct field name
                    "frame_descriptions": [
                        fields.get("frame_description", "")
                    ],  # Single frame desc
                    "duration": fields.get("end_time", 0)
                    - fields.get("start_time", 0),  # Calculate from times
                    "tags": [],  # Not available in schema
                    "source": "vespa",
                }

        except Exception as e:
            logger.debug(f"Could not fetch from Vespa: {e}")

        return None

    def _get_default_metadata(self, video_id: str) -> dict:
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
            "source": "default",
        }

    def _save_cache(self):
        """Save cache to disk"""
        cache_path = Path("outputs/cache/video_metadata.json")
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(cache_path, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save metadata cache: {e}")

    async def fetch_batch(self, video_ids: list[str]) -> dict[str, dict]:
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
