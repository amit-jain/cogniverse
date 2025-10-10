"""
Visual Judge Evaluator using ColPali/SmolLM for direct visual evaluation

Instead of sending images to an external LLM API, this uses the existing
ColPali infrastructure to directly evaluate visual relevance between
query and retrieved video frames.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from phoenix.experiments.evaluators.base import Evaluator
from phoenix.experiments.types import EvaluationResult
from PIL import Image

from src.app.agents.query_encoders import ColPaliQueryEncoder

# Use existing model infrastructure
from src.common.models import get_or_load_model

logger = logging.getLogger(__name__)


class VisualRelevanceEvaluator(Evaluator):
    """
    Evaluates visual relevance using ColPali similarity scoring

    This evaluator:
    1. Encodes the query using ColPali query encoder
    2. Encodes retrieved video frames using ColPali vision encoder
    3. Computes similarity scores between query and frames
    4. Returns evaluation based on visual similarity
    """

    def __init__(self, model_name: str = "vidore/colsmol-500m"):
        """
        Initialize visual evaluator with ColPali model

        Args:
            model_name: ColPali model to use for visual evaluation
        """
        self.model_name = model_name
        self.query_encoder = ColPaliQueryEncoder(model_name)

        # Get the same model and processor for image encoding
        config = {"colpali_model": model_name}
        self.model, self.processor = get_or_load_model(model_name, config, logger)
        self.device = next(self.model.parameters()).device

        logger.info(f"Initialized VisualRelevanceEvaluator with {model_name}")

    def evaluate(self, *, input=None, output=None, **kwargs) -> EvaluationResult:
        """
        Evaluate visual relevance of search results

        Args:
            input: Query dict with 'query' field
            output: Search results with frame information

        Returns:
            EvaluationResult with visual relevance score
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

        # Encode query once
        query_embedding = self.query_encoder.encode(query)

        # Evaluate top results visually
        visual_scores = []
        evaluated_frames = 0

        for result in results[:5]:  # Top 5 results
            frame_score = self._evaluate_frame(result, query_embedding)
            if frame_score is not None:
                visual_scores.append(frame_score)
                evaluated_frames += 1

        if not visual_scores:
            return EvaluationResult(
                score=0.0,
                label="no_visual_data",
                explanation="Could not evaluate visual content - no frames available",
            )

        # Calculate overall visual relevance score
        avg_score = np.mean(visual_scores)
        max_score = np.max(visual_scores)

        # Weight average and max scores
        final_score = 0.7 * avg_score + 0.3 * max_score

        # Determine label based on score
        if final_score >= 0.8:
            label = "highly_relevant"
            explanation = (
                f"Strong visual match (avg: {avg_score:.2f}, max: {max_score:.2f})"
            )
        elif final_score >= 0.6:
            label = "relevant"
            explanation = (
                f"Good visual match (avg: {avg_score:.2f}, max: {max_score:.2f})"
            )
        elif final_score >= 0.4:
            label = "partially_relevant"
            explanation = (
                f"Partial visual match (avg: {avg_score:.2f}, max: {max_score:.2f})"
            )
        else:
            label = "not_relevant"
            explanation = (
                f"Poor visual match (avg: {avg_score:.2f}, max: {max_score:.2f})"
            )

        return EvaluationResult(
            score=float(final_score),
            label=label,
            explanation=f"Visual evaluation: {explanation} ({evaluated_frames} frames evaluated)",
            metadata={
                "visual_scores": visual_scores,
                "frames_evaluated": evaluated_frames,
                "avg_score": float(avg_score),
                "max_score": float(max_score),
            },
        )

    def _evaluate_frame(
        self, result: dict, query_embedding: np.ndarray
    ) -> float | None:
        """
        Evaluate a single frame against the query

        Args:
            result: Search result with frame information
            query_embedding: Pre-computed query embedding

        Returns:
            Visual similarity score or None if frame not available
        """
        # Try to get frame path
        frame_path = None

        if isinstance(result, dict):
            video_id = result.get("video_id", result.get("source_id"))
            frame_id = result.get("frame_id", 0)

            # Check if frame_path is provided
            frame_path = result.get("frame_path")

            # If not, try to construct it
            if not frame_path and video_id:
                # Standard frame path pattern
                possible_paths = [
                    f"data/frames/{video_id}/frame_{frame_id}.jpg",
                    f"data/frames/{video_id}/frame_{frame_id}.png",
                    f"outputs/frames/{video_id}/frame_{frame_id}.jpg",
                ]

                for path in possible_paths:
                    if Path(path).exists():
                        frame_path = path
                        break

        if not frame_path or not Path(frame_path).exists():
            logger.debug(f"Frame not found: {frame_path}")
            return None

        try:
            # Load and encode the frame
            image = Image.open(frame_path).convert("RGB")

            # Process image with ColPali
            batch_images = self.processor.process_images([image]).to(self.device)

            with torch.no_grad():
                frame_embedding = self.model(**batch_images)

            # Convert to numpy
            frame_embedding = frame_embedding.cpu().numpy().squeeze(0)

            # Compute similarity (ColPali uses max-sim)
            similarity = self._compute_max_sim(query_embedding, frame_embedding)

            # Normalize to 0-1 range
            # ColPali similarities are typically in range [-1, 1] or similar
            normalized_score = (similarity + 1) / 2  # Simple normalization
            normalized_score = np.clip(normalized_score, 0, 1)

            return float(normalized_score)

        except Exception as e:
            logger.error(f"Error evaluating frame {frame_path}: {e}")
            return None

    def _compute_max_sim(self, query_emb: np.ndarray, frame_emb: np.ndarray) -> float:
        """
        Compute MaxSim between query and frame embeddings

        Args:
            query_emb: Query embedding [num_query_tokens, dim]
            frame_emb: Frame embedding [num_patches, dim]

        Returns:
            MaxSim score
        """
        # Ensure correct shapes
        if len(query_emb.shape) == 1:
            query_emb = query_emb.reshape(1, -1)
        if len(frame_emb.shape) == 1:
            frame_emb = frame_emb.reshape(1, -1)

        # Compute cosine similarity matrix
        # Normalize embeddings
        query_norm = query_emb / (
            np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-8
        )
        frame_norm = frame_emb / (
            np.linalg.norm(frame_emb, axis=1, keepdims=True) + 1e-8
        )

        # Compute similarity matrix
        sim_matrix = np.dot(query_norm, frame_norm.T)

        # MaxSim: max over patches for each query token, then sum
        max_sims = np.max(sim_matrix, axis=1)
        score = np.sum(max_sims)

        # Normalize by number of query tokens
        if len(max_sims) > 0:
            score = score / len(max_sims)

        return score


def create_visual_evaluators(
    model_name: str = "vidore/colsmol-500m",
) -> list[Evaluator]:
    """
    Create visual evaluators using ColPali models

    Args:
        model_name: ColPali model to use

    Returns:
        List of visual evaluator instances
    """
    return [VisualRelevanceEvaluator(model_name)]
