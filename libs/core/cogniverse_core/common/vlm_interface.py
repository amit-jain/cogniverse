"""Shared VLM (Vision Language Model) Interface"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import dspy

from cogniverse_foundation.config.utils import get_config

if TYPE_CHECKING:
    from cogniverse_foundation.config.manager import ConfigManager

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


class VLMInterface:
    """Interface for Vision Language Model operations using DSPy"""

    def __init__(self, config_manager: "ConfigManager" = None, tenant_id: str = None):
        from cogniverse_core.common.tenant_utils import require_tenant_id

        if config_manager is None:
            raise ValueError(
                "config_manager is required for VLMInterface initialization"
            )
        tenant_id = require_tenant_id(tenant_id, source="VLMInterface")
        self.config = get_config(tenant_id=tenant_id, config_manager=config_manager)
        self._initialize_vlm_client()

    def _initialize_vlm_client(self):
        """Initialize DSPy LM from centralized llm_config."""
        from cogniverse_foundation.config.llm_factory import create_dspy_lm

        llm_config = self.config.get_llm_config()
        endpoint_config = llm_config.resolve("vlm_interface")

        self._dspy_lm = create_dspy_lm(endpoint_config)
        logger.info(
            f"Created DSPy LM: {endpoint_config.model} at {endpoint_config.api_base}"
        )

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
        with dspy.context(lm=self._dspy_lm):
            visual_analysis = dspy.Predict(VisualAnalysisSignature)

            image_descriptions = []
            for image_path in image_paths:
                if Path(image_path).exists():
                    image_descriptions.append(f"Image: {Path(image_path).name}")

            result = visual_analysis(images=", ".join(image_descriptions), query=query)

            def _split_or_empty(val: str | None) -> list[str]:
                return val.split(", ") if val else []

            return {
                "descriptions": _split_or_empty(result.descriptions),
                "themes": _split_or_empty(result.themes),
                "key_objects": _split_or_empty(result.key_objects),
                "insights": _split_or_empty(result.insights),
                "relevance_score": float(result.relevance_score)
                if result.relevance_score
                else 0.0,
            }
