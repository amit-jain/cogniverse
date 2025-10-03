"""Shared VLM (Vision Language Model) Interface"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import dspy

from src.common.config_utils import get_config

logger = logging.getLogger(__name__)


class VisualAnalysisSignature(dspy.Signature):
    """Analyze visual content for search relevance."""

    images = dspy.InputField(desc="Description of images to analyze")
    query = dspy.InputField(desc="Original search query for context")

    descriptions = dspy.OutputField(desc="Visual descriptions (comma-separated)")
    themes = dspy.OutputField(desc="Visual themes (comma-separated)")
    key_objects = dspy.OutputField(desc="Key objects detected (comma-separated)")
    insights = dspy.OutputField(desc="Key insights (comma-separated)")
    relevance_score = dspy.OutputField(desc="Relevance to query (0.0-1.0)")


class DetailedVisualAnalysisSignature(dspy.Signature):
    """Analyze visual content for detailed reporting."""

    images = dspy.InputField(desc="Description of images to analyze")
    query = dspy.InputField(desc="Original search query for context")
    context = dspy.InputField(desc="Additional context for analysis")

    detailed_descriptions = dspy.OutputField(
        desc="Detailed visual descriptions (comma-separated)"
    )
    technical_analysis = dspy.OutputField(
        desc="Technical analysis findings (comma-separated)"
    )
    visual_patterns = dspy.OutputField(
        desc="Visual patterns identified (comma-separated)"
    )
    quality_score = dspy.OutputField(desc="Overall quality assessment (0.0-1.0)")
    annotations = dspy.OutputField(desc="Key annotations (comma-separated)")


class VLMInterface:
    """Interface for Vision Language Model operations using DSPy"""

    def __init__(self):
        self.config = get_config()
        self._initialize_vlm_client()

    def _initialize_vlm_client(self):
        """Initialize DSPy LM from configuration"""
        llm_config = self.config.get("llm", {})
        model_name = llm_config.get("model_name")
        base_url = llm_config.get("base_url")
        api_key = llm_config.get("api_key")

        if not all([model_name, base_url]):
            raise ValueError(
                "LLM configuration missing: model_name and base_url required"
            )

        if api_key:
            dspy.settings.configure(
                lm=dspy.LM(model=model_name, api_base=base_url, api_key=api_key)
            )
        else:
            dspy.settings.configure(lm=dspy.LM(model=model_name, api_base=base_url))

        logger.info(f"Configured DSPy LM: {model_name} at {base_url}")

    async def analyze_visual_content(
        self, image_paths: List[str], query: str
    ) -> Dict[str, Any]:
        """
        Analyze visual content using VLM

        Args:
            image_paths: List of paths to images/video frames
            query: Original search query for context

        Returns:
            Analysis results including descriptions, themes, and insights
        """
        visual_analysis = dspy.Predict(VisualAnalysisSignature)

        image_descriptions = []
        for image_path in image_paths:
            if Path(image_path).exists():
                image_descriptions.append(f"Image: {Path(image_path).name}")

        result = visual_analysis(images=", ".join(image_descriptions), query=query)

        return {
            "descriptions": result.descriptions.split(", "),
            "themes": result.themes.split(", "),
            "key_objects": result.key_objects.split(", "),
            "insights": result.insights.split(", "),
            "relevance_score": float(result.relevance_score),
        }

    async def analyze_visual_content_detailed(
        self, image_paths: List[str], query: str, context: str = ""
    ) -> Dict[str, Any]:
        """Perform detailed visual analysis using DSPy"""
        visual_analysis = dspy.Predict(DetailedVisualAnalysisSignature)

        image_descriptions = []
        for image_path in image_paths:
            image_descriptions.append(f"Image: {image_path}")

        logger.info(f"Analyzing {len(image_paths)} images with detailed VLM analysis")

        result = visual_analysis(
            images=", ".join(image_descriptions),
            query=query,
            context=context or "No additional context",
        )

        return {
            "detailed_descriptions": result.detailed_descriptions.split(", "),
            "technical_analysis": result.technical_analysis.split(", "),
            "visual_patterns": result.visual_patterns.split(", "),
            "quality_assessment": {
                "overall": float(result.quality_score),
                "clarity": float(result.quality_score),
                "relevance": float(result.quality_score),
            },
            "annotations": [
                {"element": ann, "confidence": float(result.quality_score)}
                for ann in result.annotations.split(", ")
            ],
        }
