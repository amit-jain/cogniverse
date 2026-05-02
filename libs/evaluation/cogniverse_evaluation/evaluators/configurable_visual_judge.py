"""Visual relevance judge that calls any OpenAI-compatible vision model."""

import base64
import logging
from typing import Any

import requests

from cogniverse_core.common.media import MediaLocator
from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_foundation.config.utils import get_config

from ._media_helpers import extract_frames, resolve_video_from_result
from .base import Evaluator, create_evaluation_result

logger = logging.getLogger(__name__)


class ConfigurableVisualJudge(Evaluator):
    """Score query/result frame relevance via any OpenAI-compatible vision model."""

    def __init__(
        self,
        locator: MediaLocator,
        evaluator_name: str = "visual_judge",
    ):
        """
        Initialize visual judge from config.

        Args:
            locator: MediaLocator used to resolve ``source_url`` for each
                evaluated result. Required — callers must construct it from
                the tenant's media config.
            evaluator_name: Name of evaluator config to use.
        """
        from cogniverse_foundation.config.utils import create_default_config_manager

        config_manager = create_default_config_manager()
        config = get_config(tenant_id=SYSTEM_TENANT_ID, config_manager=config_manager)
        self.evaluator_name = evaluator_name

        evaluator_config = config.get("evaluators", {}).get(evaluator_name, {})
        if not evaluator_config:
            raise ValueError(
                f"Evaluator '{evaluator_name}' is not configured. Add it to "
                "the system-tenant 'evaluators' config section."
            )

        self.provider = evaluator_config.get("provider", "openai")
        self.model = evaluator_config["model"]
        self.base_url = evaluator_config["base_url"]
        self.api_key = evaluator_config.get("api_key")

        self.locator = locator

        logger.info(
            f"Initialized {self.provider} visual judge with model {self.model} at {self.base_url}"
        )

    def evaluate(self, *, input=None, output=None, **kwargs) -> Any:
        """Score the query against frames extracted from the top results."""
        # Extract query
        query = input.get("query", "") if isinstance(input, dict) else str(input)

        # Extract results
        if hasattr(output, "results"):
            results = output.results
        elif isinstance(output, dict) and "results" in output:
            results = output["results"]
        else:
            results = output if isinstance(output, list) else []

        if not results:
            return create_evaluation_result(
                score=0.0, label="no_results", explanation="No results to evaluate"
            )

        # Get frames from videos for top results
        frame_paths = []

        # Determine how many frames to extract based on config
        from cogniverse_foundation.config.utils import create_default_config_manager

        # Initialize ConfigManager for dependency injection
        frames_config_manager = create_default_config_manager()
        config = get_config(
            tenant_id=SYSTEM_TENANT_ID, config_manager=frames_config_manager
        )
        evaluator_config = config.get("evaluators", {}).get(self.evaluator_name, {})
        frames_per_video = evaluator_config.get("frames_per_video", 30)
        max_videos = evaluator_config.get("max_videos", 2)
        sample_all = evaluator_config.get("sample_all_frames", False)
        max_total_frames = evaluator_config.get("max_total_frames", 60)

        for _i, result in enumerate(results[:max_videos], 1):  # Top N videos
            # First try to get video path
            video_path = self._get_video_path(result)
            if video_path:
                # Calculate how many frames to extract from this video
                remaining_budget = max_total_frames - len(frame_paths)
                frames_to_extract = min(frames_per_video, remaining_budget)

                if frames_to_extract > 0:
                    # Use timestamp from result if available
                    timestamp = 0
                    if isinstance(result, dict):
                        timestamp = result.get("start_time", 0)

                    extracted_frames = self._extract_frames_from_video(
                        video_path,
                        num_frames=frames_to_extract,
                        timestamp=timestamp,
                        sample_all=sample_all,
                    )
                    if extracted_frames:
                        frame_paths.extend(extracted_frames)

        if not frame_paths:
            # Log what we tried to find
            logger.warning(
                f"No videos found for evaluation. Searched results: {[r.get('video_id', r.get('source_id')) if isinstance(r, dict) else str(r) for r in results[:3]]}"
            )
            return create_evaluation_result(
                score=0.0,
                label="no_frames",
                explanation="No video frames could be extracted for evaluation",
            )

        try:
            score, reasoning = self._score_frames(query, frame_paths)

            # Determine label
            if score >= 0.8:
                label = "excellent_match"
            elif score >= 0.6:
                label = "good_match"
            elif score >= 0.4:
                label = "partial_match"
            else:
                label = "poor_match"

            return create_evaluation_result(
                score=float(score),
                label=label,
                explanation=f"{self.provider}/{self.model}: {reasoning}",
                metadata={
                    "frames_evaluated": len(frame_paths),
                    "provider": self.provider,
                    "model": self.model,
                },
            )

        except Exception as e:
            logger.error(f"Visual evaluation failed: {e}")
            return create_evaluation_result(
                score=0.0,
                label="evaluation_failed",
                explanation=f"Visual evaluation failed: {str(e)}",
            )

    def _get_video_path(self, result: dict) -> str | None:
        """Resolve a result's video to a local path via the MediaLocator.

        Prefers ``source_url`` (the canonical URI written at ingest time);
        falls back to a legacy local-directory probe with a WARNING log so
        already-ingested corpora without ``source_url`` keep working.
        """
        path = resolve_video_from_result(result, self.locator)
        return str(path) if path is not None else None

    def _extract_frames_from_video(
        self,
        video_path: str,
        num_frames: int = 30,
        timestamp: float = 0,
        sample_all: bool = False,
    ) -> list[str]:
        """Extract frames from a local video; thin wrapper over the shared helper."""
        from pathlib import Path as _Path

        max_total_frames = 60
        if sample_all:
            from cogniverse_foundation.config.utils import (
                create_default_config_manager,
            )

            sample_config_manager = create_default_config_manager()
            config = get_config(
                tenant_id=SYSTEM_TENANT_ID, config_manager=sample_config_manager
            )
            evaluator_config = config.get("evaluators", {}).get(self.evaluator_name, {})
            max_total_frames = evaluator_config.get("max_total_frames", 60)

        paths = extract_frames(
            _Path(video_path),
            num_frames=num_frames,
            timestamp=timestamp,
            sample_all=sample_all,
            max_total_frames=max_total_frames,
        )
        logger.info("Extracted %d frames from video %s", len(paths), video_path)
        return [str(p) for p in paths]

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def _score_frames(self, query: str, frame_paths: list[str]) -> tuple[float, str]:
        """Send frames + query to the vision model and parse a 0-1 score."""
        encoded_images = [self._encode_image(path) for path in frame_paths]

        prompt = (
            f"Do these video frames match the search query '{query}'? "
            "Rate 0-10. Format: SCORE: X/10, REASONING: explanation"
        )

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
                + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                    }
                    for img in encoded_images
                ],
            }
        ]

        base = self.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{base}/chat/completions",
            json={"model": self.model, "messages": messages, "max_tokens": 300},
            headers=headers,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Vision API error: {response.status_code} - {response.text}"
            )

        result = response.json()
        response_text = (
            result.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        return self._parse_response(response_text)

    def _parse_response(self, response_text: str) -> tuple[float, str]:
        """Parse score and reasoning from response"""
        import re

        score = 0.5  # Default
        reasoning = response_text

        # Try to extract score
        score_match = re.search(
            r"SCORE:\s*(\d+(?:\.\d+)?)", response_text, re.IGNORECASE
        )
        if score_match:
            try:
                score = float(score_match.group(1))
                score = score / 10.0  # Normalize to 0-1
                score = min(max(score, 0.0), 1.0)
            except (ValueError, AttributeError) as e:
                logger.warning(f"Failed to parse score from match: {e}")
                score = 0.0

        # Try to extract reasoning
        reasoning_match = re.search(
            r"REASONING:\s*(.+)", response_text, re.IGNORECASE | re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return score, reasoning
