"""
Pipeline artifact caching system for video processing results
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiofiles
import cv2
import numpy as np

from .base import CacheManager

logger = logging.getLogger(__name__)


@dataclass
class VideoArtifacts:
    """Container for all video processing artifacts"""

    video_id: str
    keyframes: Optional[Dict[str, Any]] = None
    audio_transcript: Optional[Dict[str, Any]] = None
    frame_descriptions: Optional[Dict[str, Any]] = None
    embeddings: Optional[Dict[str, Any]] = None

    def is_complete(self, pipeline_config: Dict[str, Any]) -> bool:
        """Check if all required artifacts are present based on pipeline config"""
        if pipeline_config.get("extract_keyframes", False) and not self.keyframes:
            return False
        if pipeline_config.get("transcribe_audio", False) and not self.audio_transcript:
            return False
        if (
            pipeline_config.get("generate_descriptions", False)
            and not self.frame_descriptions
        ):
            return False
        if pipeline_config.get("generate_embeddings", False) and not self.embeddings:
            return False
        return True


class PipelineArtifactCache:
    """
    Comprehensive caching for video processing pipeline artifacts.
    Supports caching of keyframes, transcripts, descriptions, and embeddings.
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
        """Generate cache key for a video"""
        # Use the video filename as the key for human readability
        path = Path(video_path)
        video_name = path.stem  # Get filename without extension

        base_key = f"video:{video_name}"

        # Add profile namespace if specified
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

    async def get_all_artifacts(
        self, video_path: str, pipeline_config: Dict[str, Any]
    ) -> VideoArtifacts:
        """Get all cached artifacts for a video based on pipeline config"""
        artifacts = VideoArtifacts(video_id=Path(video_path).stem)

        # Get artifacts based on what's enabled in pipeline
        if pipeline_config.get("extract_keyframes", False):
            # Determine strategy from config
            strategy = pipeline_config.get("keyframe_strategy", "similarity")
            if strategy == "similarity":
                artifacts.keyframes = await self.get_keyframes(
                    video_path,
                    strategy=strategy,
                    threshold=pipeline_config.get("keyframe_threshold", 0.999),
                    max_frames=pipeline_config.get("max_frames_per_video", 3000),
                )
            elif strategy == "fps":
                artifacts.keyframes = await self.get_keyframes(
                    video_path,
                    strategy=strategy,
                    fps=pipeline_config.get("keyframe_fps", 1.0),
                    max_frames=pipeline_config.get("max_frames_per_video", 3000),
                )

        if pipeline_config.get("transcribe_audio", False):
            artifacts.audio_transcript = await self.get_transcript(
                video_path, model_size=pipeline_config.get("whisper_model", "base")
            )

        if pipeline_config.get("generate_descriptions", False):
            artifacts.frame_descriptions = await self.get_descriptions(
                video_path,
                model_name=pipeline_config.get(
                    "vlm_model", "Qwen/Qwen2-VL-2B-Instruct"
                ),
                batch_size=pipeline_config.get("vlm_batch_size", 500),
            )

        return artifacts

    async def get_segment_frames(
        self,
        video_path: str,
        segment_id: int,
        start_time: float,
        end_time: float,
        sampling_fps: float = 2.0,
        max_frames: int = 12,
        load_images: bool = True,
    ) -> Optional[Union[Dict[str, Any], Tuple[Dict[str, Any], List[np.ndarray]]]]:
        """Get cached segment frames"""
        video_key = self._generate_video_key(video_path)

        # Generate unique key for this segment configuration
        params = {
            "segment_id": segment_id,
            "start_time": start_time,
            "end_time": end_time,
            "sampling_fps": sampling_fps,
            "max_frames": max_frames,
        }

        artifact_key = self._generate_artifact_key(
            video_key, "segment_frames", **params
        )

        metadata = await self.cache.get(artifact_key)
        if metadata:
            logger.info(
                f"Cache hit for segment {segment_id} frames: {Path(video_path).name}"
            )

            # Load images if requested
            if load_images:
                frames = []
                for i, timestamp in enumerate(metadata.get("timestamps", [])):
                    image_key = f"{artifact_key}:frame_{i}"
                    image_data = await self.cache.get(image_key)
                    if image_data:
                        # Decode image from bytes
                        nparr = np.frombuffer(image_data, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if image is not None:
                            frames.append(image)

                return metadata, frames

            return metadata

        logger.debug(
            f"Cache miss for segment {segment_id} frames: {Path(video_path).name}"
        )
        return None

    async def set_segment_frames(
        self,
        video_path: str,
        segment_id: int,
        start_time: float,
        end_time: float,
        frames: List[np.ndarray],
        timestamps: List[float],
        sampling_fps: float = 2.0,
        max_frames: int = 12,
    ) -> bool:
        """Cache segment frames with metadata"""
        video_key = self._generate_video_key(video_path)

        # Generate unique key for this segment configuration
        params = {
            "segment_id": segment_id,
            "start_time": start_time,
            "end_time": end_time,
            "sampling_fps": sampling_fps,
            "max_frames": max_frames,
        }

        artifact_key = self._generate_artifact_key(
            video_key, "segment_frames", **params
        )

        # Store metadata
        metadata = {
            "segment_id": segment_id,
            "start_time": start_time,
            "end_time": end_time,
            "timestamps": timestamps,
            "num_frames": len(frames),
            "sampling_fps": sampling_fps,
            "max_frames": max_frames,
            "cached_at": time.time(),
        }

        # Store individual frames
        for i, frame in enumerate(frames):
            # Encode frame as JPEG
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_data = buffer.tobytes()

            image_key = f"{artifact_key}:frame_{i}"
            await self.cache.set(image_key, image_data, self.ttl)

        # Store metadata
        return await self.cache.set(artifact_key, metadata, self.ttl)

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


class FileSystemPipelineCache(PipelineArtifactCache):
    """
    File system optimized version that can directly use existing file structure
    """

    def __init__(self, base_path: Path, cache_manager: Optional[CacheManager] = None):
        """
        Initialize with direct file system access

        Args:
            base_path: Base directory for pipeline outputs
            cache_manager: Optional cache manager for metadata caching
        """
        super().__init__(cache_manager) if cache_manager else None
        self.base_path = Path(base_path)

    async def get_keyframes_from_disk(
        self, video_id: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, np.ndarray]]]:
        """Load keyframes directly from disk"""
        metadata_file = self.base_path / "metadata" / f"{video_id}_keyframes.json"
        keyframes_dir = self.base_path / "keyframes" / video_id

        if not metadata_file.exists():
            return None, None

        # Load metadata
        async with aiofiles.open(metadata_file, "r") as f:
            metadata = json.loads(await f.read())

        # Load keyframe images
        images = {}
        for frame_info in metadata.get("keyframes", []):
            frame_path = keyframes_dir / frame_info["filename"]
            if frame_path.exists():
                # Read image asynchronously
                image = cv2.imread(str(frame_path))
                if image is not None:
                    images[str(frame_info["frame_id"])] = image

        return metadata, images

    async def get_transcript_from_disk(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load transcript directly from disk"""
        transcript_file = self.base_path / "transcripts" / f"{video_id}.json"

        if not transcript_file.exists():
            return None

        async with aiofiles.open(transcript_file, "r") as f:
            return json.loads(await f.read())

    async def get_descriptions_from_disk(
        self, video_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load descriptions directly from disk"""
        descriptions_file = (
            self.base_path / "descriptions" / f"{video_id}_descriptions.json"
        )

        if not descriptions_file.exists():
            return None

        async with aiofiles.open(descriptions_file, "r") as f:
            return json.loads(await f.read())
