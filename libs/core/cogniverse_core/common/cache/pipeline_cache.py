"""
Pipeline artifact caching system for video processing results
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .base import CacheManager

logger = logging.getLogger(__name__)


class PipelineArtifactCache:
    """
    Comprehensive caching for video processing pipeline artifacts.
    Supports caching of keyframes, transcripts, descriptions, and embeddings.

    Backed by ``CacheManager``'s tiered backends: a local-filesystem L1 plus an
    optional shared S3/MinIO L2 (survives pod restart, hits across pods). The
    live ingestion path uses this via ``ProcessingStrategySet``.
    """

    def __init__(
        self,
        cache_manager: CacheManager,
        ttl: int = 604800,
        profile: Optional[str] = None,
    ):  # 7 days default
        """
        Initialize pipeline artifact cache

        Args:
            cache_manager: The cache manager instance
            ttl: Time to live in seconds (default 7 days)
            profile: Optional profile name to namespace cache entries
        """
        self.cache = cache_manager
        self.ttl = ttl
        self.profile = profile

    def _generate_video_key(
        self, video_path: str, video_hash: Optional[str] = None
    ) -> str:
        """Generate cache key for a video.

        Keys on the SHA-256 of the canonical URI rather than the filename stem
        so that videos with the same basename in different roots (or on
        different mounts) do not collide. Accepts either a URI (``file://...``,
        ``s3://...``, etc.) or a bare path; bare paths are canonicalized to
        ``file://<absolute>`` before hashing.
        """
        canonical = (
            video_path
            if "://" in video_path
            else f"file://{Path(video_path).resolve()}"
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        base_key = f"video:{digest}"

        if self.profile:
            return f"{self.profile}:{base_key}"
        return base_key

    def _generate_artifact_key(
        self, video_key: str, artifact_type: str, **kwargs
    ) -> str:
        """Generate cache key for specific artifact"""
        base_key = f"{video_key}:{artifact_type}"

        # Add additional parameters to key
        if kwargs:
            params = ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
            base_key = f"{base_key}:{params}"

        return base_key

    async def get_keyframes(
        self,
        video_path: str,
        strategy: str = "similarity",
        threshold: Optional[float] = None,
        fps: Optional[float] = None,
        max_frames: int = 3000,
        load_images: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Get cached keyframes metadata and optionally images"""
        video_key = self._generate_video_key(video_path)

        # Build parameters based on strategy
        params = {"strategy": strategy, "max_frames": max_frames}
        if strategy == "similarity":
            params["threshold"] = threshold or 0.999
        elif strategy == "fps":
            params["fps"] = fps or 1.0

        artifact_key = self._generate_artifact_key(video_key, "keyframes", **params)

        metadata = await self.cache.get(artifact_key)
        if metadata:
            logger.info(f"Cache hit for keyframes: {Path(video_path).name}")

            # Load images if requested
            if load_images:
                keyframe_images = {}
                for frame_info in metadata.get("keyframes", []):
                    frame_id = frame_info["frame_id"]
                    image_key = f"{artifact_key}:frame_{frame_id}"
                    image_data = await self.cache.get(image_key)
                    if image_data:
                        # Decode image from bytes
                        nparr = np.frombuffer(image_data, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if image is not None:
                            keyframe_images[str(frame_id)] = image

                return metadata, keyframe_images

            return metadata

        logger.debug(f"Cache miss for keyframes: {Path(video_path).name}")
        return None

    async def set_keyframes(
        self,
        video_path: str,
        keyframes_metadata: Dict[str, Any],
        keyframe_images: Optional[Dict[str, np.ndarray]] = None,
        strategy: str = "similarity",
        threshold: Optional[float] = None,
        fps: Optional[float] = None,
        max_frames: int = 3000,
    ) -> bool:
        """Cache keyframes with metadata and images"""
        video_key = self._generate_video_key(video_path)

        # Build parameters based on strategy
        params = {"strategy": strategy, "max_frames": max_frames}
        if strategy == "similarity":
            params["threshold"] = threshold or 0.999
        elif strategy == "fps":
            params["fps"] = fps or 1.0

        # Store metadata
        metadata_key = self._generate_artifact_key(video_key, "keyframes", **params)

        # Store each keyframe image separately
        if keyframe_images:
            for frame_info in keyframes_metadata.get("keyframes", []):
                frame_id = frame_info["frame_id"]
                if str(frame_id) in keyframe_images:
                    image_key = f"{metadata_key}:frame_{frame_id}"
                    image_data = cv2.imencode(".jpg", keyframe_images[str(frame_id)])[
                        1
                    ].tobytes()
                    await self.cache.set(image_key, image_data, self.ttl)

        # Store metadata
        return await self.cache.set(metadata_key, keyframes_metadata, self.ttl)

    async def get_transcript(
        self, video_path: str, model_size: str = "base", language: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached audio transcript"""
        video_key = self._generate_video_key(video_path)
        artifact_key = self._generate_artifact_key(
            video_key, "transcript", model=model_size, lang=language or "auto"
        )

        transcript = await self.cache.get(artifact_key)
        if transcript:
            logger.info(f"Cache hit for transcript: {Path(video_path).name}")
            return transcript

        logger.debug(f"Cache miss for transcript: {Path(video_path).name}")
        return None

    async def set_transcript(
        self,
        video_path: str,
        transcript_data: Dict[str, Any],
        model_size: str = "base",
        language: Optional[str] = None,
    ) -> bool:
        """Cache audio transcript"""
        video_key = self._generate_video_key(video_path)
        artifact_key = self._generate_artifact_key(
            video_key, "transcript", model=model_size, lang=language or "auto"
        )

        return await self.cache.set(artifact_key, transcript_data, self.ttl)

    async def get_descriptions(
        self, video_path: str, model_name: str, batch_size: int = 500
    ) -> Optional[Dict[str, Any]]:
        """Get cached frame descriptions"""
        video_key = self._generate_video_key(video_path)
        artifact_key = self._generate_artifact_key(
            video_key, "descriptions", model=model_name, batch_size=batch_size
        )

        descriptions = await self.cache.get(artifact_key)
        if descriptions:
            logger.info(f"Cache hit for descriptions: {Path(video_path).name}")
            return descriptions

        logger.debug(f"Cache miss for descriptions: {Path(video_path).name}")
        return None

    async def set_descriptions(
        self,
        video_path: str,
        descriptions_data: Dict[str, Any],
        model_name: str,
        batch_size: int = 500,
    ) -> bool:
        """Cache frame descriptions"""
        video_key = self._generate_video_key(video_path)
        artifact_key = self._generate_artifact_key(
            video_key, "descriptions", model=model_name, batch_size=batch_size
        )

        return await self.cache.set(artifact_key, descriptions_data, self.ttl)

    def _segmentation_key(
        self,
        video_path: str,
        strategy: str,
        segment_duration: float,
        segment_overlap: float,
        transcript_fingerprint: str,
    ) -> str:
        video_key = self._generate_video_key(video_path)
        return self._generate_artifact_key(
            video_key,
            "segmentation",
            strategy=strategy,
            seg=segment_duration,
            overlap=segment_overlap,
            transcript=transcript_fingerprint,
        )

    async def get_segmentation(
        self,
        video_path: str,
        *,
        strategy: str,
        segment_duration: float,
        segment_overlap: float,
        transcript_fingerprint: str,
    ) -> Optional[Dict[str, Any]]:
        """Get a cached single-vector segmentation result (boundary math +
        transcript alignment; no frames)."""
        artifact_key = self._segmentation_key(
            video_path,
            strategy,
            segment_duration,
            segment_overlap,
            transcript_fingerprint,
        )

        segmentation = await self.cache.get(artifact_key)
        if segmentation:
            logger.info(f"Cache hit for segmentation: {Path(video_path).name}")
            return segmentation

        logger.debug(f"Cache miss for segmentation: {Path(video_path).name}")
        return None

    async def set_segmentation(
        self,
        video_path: str,
        result: Dict[str, Any],
        *,
        strategy: str,
        segment_duration: float,
        segment_overlap: float,
        transcript_fingerprint: str,
    ) -> bool:
        """Cache a single-vector segmentation result."""
        artifact_key = self._segmentation_key(
            video_path,
            strategy,
            segment_duration,
            segment_overlap,
            transcript_fingerprint,
        )

        return await self.cache.set(artifact_key, result, self.ttl)

    async def invalidate_video(self, video_path: str) -> int:
        """Invalidate all cached artifacts for a video"""
        video_key = self._generate_video_key(video_path)
        # Clear all keys starting with this video key
        return await self.cache.clear(f"{video_key}:*")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics by artifact type"""
        stats = await self.cache.get_stats()

        # Add artifact-specific stats if available
        # This would require the cache backend to support key pattern analysis
        return {
            "overall": stats,
            "artifacts": {
                "keyframes": "Not implemented",
                "transcripts": "Not implemented",
                "descriptions": "Not implemented",
            },
        }
