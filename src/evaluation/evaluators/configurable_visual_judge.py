"""
Configurable Visual Judge that uses config to determine provider (Ollama, Modal, etc.)
"""

import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import base64
import requests

from phoenix.experiments.evaluators.base import Evaluator
from phoenix.experiments.types import EvaluationResult
from src.common.config import get_config

logger = logging.getLogger(__name__)


class ConfigurableVisualJudge(Evaluator):
    """
    Visual judge that uses configured provider (Ollama, Modal, etc.)
    """

    def __init__(self, evaluator_name: str = "visual_judge"):
        """
        Initialize visual judge from config

        Args:
            evaluator_name: Name of evaluator config to use
        """
        config = get_config()
        self.evaluator_name = evaluator_name

        # Get evaluator config
        evaluator_config = config.get("evaluators", {}).get(evaluator_name, {})
        if not evaluator_config:
            # Fallback to default
            evaluator_config = {
                "provider": "ollama",
                "model": "llava:7b",
                "base_url": "http://localhost:11434",
                "api_key": None,
            }
            logger.warning(
                f"No config for evaluator '{evaluator_name}', using defaults"
            )

        self.provider = evaluator_config.get("provider", "ollama")
        self.model = evaluator_config.get("model", "llava:7b")
        self.base_url = evaluator_config.get("base_url", "http://localhost:11434")
        self.api_key = evaluator_config.get("api_key")

        logger.info(
            f"Initialized {self.provider} visual judge with model {self.model} at {self.base_url}"
        )

    def evaluate(self, *, input=None, output=None, **kwargs) -> EvaluationResult:
        """
        Evaluate visual relevance using configured provider
        """
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
            return EvaluationResult(
                score=0.0, label="no_results", explanation="No results to evaluate"
            )

        # Get frames from videos for top results
        frame_paths = []

        # Determine how many frames to extract based on config
        config = get_config()
        evaluator_config = config.get("evaluators", {}).get(self.evaluator_name, {})
        frames_per_video = evaluator_config.get("frames_per_video", 30)
        max_videos = evaluator_config.get("max_videos", 2)
        sample_all = evaluator_config.get("sample_all_frames", False)
        max_total_frames = evaluator_config.get("max_total_frames", 60)

        for i, result in enumerate(results[:max_videos], 1):  # Top N videos
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
            return EvaluationResult(
                score=0.0,
                label="no_frames",
                explanation="No video frames could be extracted for evaluation",
            )

        # Evaluate based on provider
        try:
            if self.provider == "ollama":
                score, reasoning = self._evaluate_with_ollama(query, frame_paths)
            elif self.provider == "modal":
                score, reasoning = self._evaluate_with_modal(query, frame_paths)
            elif self.provider == "openai":
                score, reasoning = self._evaluate_with_openai(query, frame_paths)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            # Determine label
            if score >= 0.8:
                label = "excellent_match"
            elif score >= 0.6:
                label = "good_match"
            elif score >= 0.4:
                label = "partial_match"
            else:
                label = "poor_match"

            return EvaluationResult(
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
            return EvaluationResult(
                score=0.0,
                label="evaluation_failed",
                explanation=f"Visual evaluation failed: {str(e)}",
            )

    def _get_video_path(self, result: Dict) -> Optional[str]:
        """Extract video path from result"""
        if isinstance(result, dict):
            video_id = result.get("video_id", result.get("source_id"))

            # Common video storage locations
            if video_id:
                possible_paths = [
                    f"data/testset/evaluation/sample_videos/{video_id}.mp4",
                    f"data/testset/evaluation/sample_videos/{video_id}.avi",
                    f"data/testset/evaluation/sample_videos/{video_id}.mov",
                    f"data/videos/{video_id}.mp4",
                    f"outputs/videos/{video_id}.mp4",
                ]

                for path in possible_paths:
                    if Path(path).exists():
                        return path

                # Try without extension (video_id might already include it)
                if Path(f"data/testset/evaluation/sample_videos/{video_id}").exists():
                    return f"data/testset/evaluation/sample_videos/{video_id}"

        return None

    def _extract_frames_from_video(
        self,
        video_path: str,
        num_frames: int = 30,
        timestamp: float = 0,
        sample_all: bool = False,
    ) -> List[str]:
        """Extract multiple frames from video

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract (evenly spaced)
            timestamp: Starting timestamp
            sample_all: If True, extract all frames (up to max_total_frames limit)

        Returns:
            List of paths to extracted frame images
        """
        import cv2
        import tempfile

        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if timestamp > 0:
                start_frame = int(timestamp * fps)
            else:
                start_frame = 0

            # If sample_all, extract every frame (with limit)
            if sample_all:
                # Get max_total_frames from config
                config = get_config()
                evaluator_config = config.get("evaluators", {}).get(
                    self.evaluator_name, {}
                )
                max_total = evaluator_config.get("max_total_frames", 60)

                # Extract frames at regular intervals to stay within limit
                num_frames = min(total_frames - start_frame, max_total)
                interval = max(1, (total_frames - start_frame) // num_frames)
            else:
                # Calculate frame interval for evenly spaced frames
                interval = max(1, (total_frames - start_frame) // num_frames)

            for i in range(num_frames):
                frame_number = start_frame + (i * interval)
                if frame_number >= total_frames:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()

                if ret:
                    # Save frame to temp file
                    with tempfile.NamedTemporaryFile(
                        suffix=".jpg", delete=False
                    ) as tmp:
                        cv2.imwrite(tmp.name, frame)
                        frames.append(tmp.name)

            cap.release()
            logger.info(f"Extracted {len(frames)} frames from video {video_path}")

        except Exception as e:
            logger.error(f"Could not extract frames from video: {e}")

        return frames

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def _evaluate_with_ollama(
        self, query: str, frame_paths: List[str]
    ) -> Tuple[float, str]:
        """
        Evaluate using Ollama (supports LLaVA, Qwen2-VL via Ollama)
        """
        # Encode images
        encoded_images = [self._encode_image(path) for path in frame_paths]

        prompt = f"""You are evaluating video search results.
The user searched for: "{query}"

I'm showing you {len(frame_paths)} frames from the top search results.

Please evaluate:
1. Do these frames show content that matches "{query}"?
2. How relevant is the visual content to the search query?
3. Rate the match quality from 0 to 10.

Provide your assessment in this format:
SCORE: [0-10]
REASONING: [Your explanation of what you see and how well it matches]"""

        # Prepare Ollama API request
        messages = [{"role": "user", "content": prompt, "images": encoded_images}]

        # Call Ollama API
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={"model": self.model, "messages": messages, "stream": False},
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama API error: {response.status_code} - {response.text}"
            )

        result = response.json()
        response_text = result.get("message", {}).get("content", "")

        # Parse response
        return self._parse_response(response_text)

    def _evaluate_with_modal(
        self, query: str, frame_paths: List[str]
    ) -> Tuple[float, str]:
        """
        Evaluate using Modal deployment
        """
        # Encode images
        encoded_images = [self._encode_image(path) for path in frame_paths]

        prompt = f"""Evaluate if these video frames match the search query: "{query}"
Rate from 0-10 and explain what you see.
Format: SCORE: X/10, REASONING: explanation"""

        # Call Modal endpoint
        response = requests.post(
            self.base_url,
            json={"prompt": prompt, "images": encoded_images, "model": self.model},
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Modal API error: {response.status_code} - {response.text}"
            )

        result = response.json()
        response_text = result.get("response", result.get("output", ""))

        # Parse response
        return self._parse_response(response_text)

    def _evaluate_with_openai(
        self, query: str, frame_paths: List[str]
    ) -> Tuple[float, str]:
        """
        Evaluate using OpenAI API (GPT-4V)
        """
        # Encode images
        encoded_images = [self._encode_image(path) for path in frame_paths]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Do these video frames match the search query '{query}'? Rate 0-10. Format: SCORE: X/10, REASONING: explanation",
                    }
                ]
                + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"},
                    }
                    for img in encoded_images
                ],
            }
        ]

        # Call OpenAI API
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={"model": self.model, "messages": messages, "max_tokens": 300},
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"OpenAI API error: {response.status_code} - {response.text}"
            )

        result = response.json()
        response_text = (
            result.get("choices", [{}])[0].get("message", {}).get("content", "")
        )

        # Parse response
        return self._parse_response(response_text)

    def _parse_response(self, response_text: str) -> Tuple[float, str]:
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


def create_configurable_visual_evaluators(
    evaluator_name: str = "visual_judge",
) -> List[Evaluator]:
    """
    Create visual evaluators using configured provider

    Args:
        evaluator_name: Name of evaluator config to use

    Returns:
        List of visual evaluator instances
    """
    return [ConfigurableVisualJudge(evaluator_name)]
