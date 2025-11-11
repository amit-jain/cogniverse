"""
Visual Judge using Qwen2-VL for actual visual understanding and evaluation

Uses Qwen2-VL-7B-Instruct to evaluate if retrieved frames actually match the query
by understanding the visual content, not just computing embeddings.
"""

import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from .base import Evaluator, create_evaluation_result

logger = logging.getLogger(__name__)


class Qwen2VLVisualJudge(Evaluator):
    """
    Visual judge using Qwen2-VL to understand and evaluate visual content

    This evaluator:
    1. Takes the query and retrieved video frames
    2. Uses Qwen2-VL to understand what's in the frames
    3. Judges if the visual content matches the query
    4. Returns evaluation with reasoning
    """

    def __init__(
        self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", device: str = None
    ):
        """
        Initialize Qwen2-VL visual judge

        Args:
            model_name: Qwen2-VL model to use
            device: Device to run on (auto-detect if None)
        """
        self.model_name = model_name

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Loading Qwen2-VL model: {model_name} on {self.device}")

        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Load with appropriate settings for device
        if self.device == "cuda":
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float32
            )
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info(f"Qwen2-VL loaded successfully on {self.device}")

    def evaluate(self, *, input=None, output=None, **kwargs) -> Any:
        """
        Evaluate visual relevance using Qwen2-VL

        Args:
            input: Query dict with 'query' field
            output: Search results with frame information

        Returns:
            EvaluationResult with visual evaluation
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
            return create_evaluation_result(
                score=0.0, label="no_results", explanation="No results to evaluate"
            )

        # Get frame paths for top results
        frame_paths = []
        for _i, result in enumerate(results[:3], 1):  # Top 3 for evaluation
            frame_path = self._get_frame_path(result)
            if frame_path and Path(frame_path).exists():
                frame_paths.append(frame_path)

        if not frame_paths:
            return create_evaluation_result(
                score=0.0,
                label="no_frames",
                explanation="No frame images found for evaluation",
            )

        # Use Qwen2-VL to evaluate
        try:
            score, reasoning = self._evaluate_with_qwen(query, frame_paths)

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
                explanation=f"Qwen2-VL: {reasoning}",
                metadata={
                    "frames_evaluated": len(frame_paths),
                    "model": self.model_name,
                },
            )

        except Exception as e:
            logger.error(f"Qwen2-VL evaluation failed: {e}")
            return create_evaluation_result(
                score=0.0,
                label="evaluation_failed",
                explanation=f"Visual evaluation failed: {str(e)}",
            )

    def _get_frame_path(self, result: dict) -> str | None:
        """Extract frame path from result"""
        if isinstance(result, dict):
            video_id = result.get("video_id", result.get("source_id"))
            frame_id = result.get("frame_id", 0)

            # Check if frame_path is provided
            frame_path = result.get("frame_path")

            # If not, try to construct it
            if not frame_path and video_id:
                possible_paths = [
                    f"data/frames/{video_id}/frame_{frame_id}.jpg",
                    f"data/frames/{video_id}/frame_{frame_id}.png",
                    f"outputs/frames/{video_id}/frame_{frame_id}.jpg",
                ]

                for path in possible_paths:
                    if Path(path).exists():
                        return path

            return frame_path

        return None

    def _evaluate_with_qwen(
        self, query: str, frame_paths: list[str]
    ) -> tuple[float, str]:
        """
        Use Qwen2-VL to evaluate if frames match the query

        Args:
            query: Search query
            frame_paths: Paths to frame images

        Returns:
            Tuple of (score, reasoning)
        """
        # Load images
        images = []
        for path in frame_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.warning(f"Could not load image {path}: {e}")

        if not images:
            return 0.0, "Could not load any images"

        # Construct prompt for Qwen2-VL
        prompt = f"""You are evaluating video search results.
The user searched for: "{query}"

I'm showing you {len(images)} frames from the top search results.

Please evaluate:
1. Do these frames show content that matches "{query}"?
2. How relevant is the visual content to the search query?
3. Rate the match quality from 0 to 10.

Provide your assessment in this format:
SCORE: [0-10]
REASONING: [Your explanation of what you see and how well it matches]
"""

        # Prepare messages for Qwen2-VL
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        # Add images to the message
        for i, img in enumerate(images):
            messages[0]["content"].insert(i, {"type": "image", "image": img})

        # Process with Qwen2-VL
        text_inputs = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_inputs], images=images, padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=256, temperature=0.1, do_sample=True
            )

        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Parse response
        score = 5.0  # Default
        reasoning = response

        # Try to extract score
        import re

        score_match = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)", response, re.IGNORECASE)
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
            r"REASONING:\s*(.+)", response, re.IGNORECASE | re.DOTALL
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return score, reasoning


def create_qwen_visual_evaluators() -> list[Evaluator]:
    """
    Create Qwen2-VL visual evaluators

    Returns:
        List of visual evaluator instances
    """
    return [Qwen2VLVisualJudge()]
